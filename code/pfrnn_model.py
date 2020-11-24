import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from policy import Categorical, DiagGaussian
from torch.nn.init import xavier_normal_, orthogonal_
import encoder
import namedlist
from operator import mul
from functools import reduce
from pfrnns import PFGRUCell
import numpy as np
import ipdb as pdb
from particle_aggregators import *

# Container to return all required values from model
PolicyReturn = namedlist.namedlist('PolicyReturn', [
    ('latent_state', None),
    ('value_estimate', None),
    ('action', None),
    ('action_log_probs', None),
    ('dist_entropy', None),
])


class Policy(nn.Module):
    """parent class for the policy
    """
    def __init__(self, action_space, encoding_dimension):
        super().__init__()

        # Value function V(latent_state)
        self.critic_linear = nn.Linear(encoding_dimension, 1)
        self.h_dim = encoding_dimension

        # Policy \pi(a|latent_state)
        if action_space.__class__.__name__ == "Discrete":
            num_outputs = action_space.n
            self.dist = Categorical(encoding_dimension, num_outputs)
        elif action_space.__class__.__name__ == "Box":
            num_outputs = action_space.shape[0]
            self.dist = DiagGaussian(encoding_dimension, num_outputs)
        else:
            raise NotImplementedError

        self.encoding_bn = nn.BatchNorm1d(encoding_dimension)

    def encode(self, observation, actions, previous_latent_state):
        """To be provided by a child class. Recurrently encodes the observations and actions into a latent vector
        
        Arguments:
            observation {tensor} -- the current observation
            actions {tensor} -- the current action
            previous_latent_state {tensor} -- the previous hidden state
        
        Raises:
            NotImplementedError: this should be provided by the child class
        """

        raise NotImplementedError("Should be provided by child class, e.g. RNNPolicy or DVRLPolicy.")

    def new_latent_state(self):
        """
        To be provided by child class. Creates either n_s latent tensors for RNN or n_s particle
        ensembles for DVRL.
        """
        raise NotImplementedError("Should be provided by child class, e.g. RNNPolicy or DVRLPolicy.")

    def vec_conditional_new_latent_state(self, latent_state, mask):
        """
        To be provided by child class. Creates a new latent state for each environment in which the episode ended.
        """

    def forward(self, current_memory, deterministic=False):
        """Run the model and compute all the stuff we needed, see PolicyReturn namedTuple
        
        Arguments:
            current_memory {dict} -- all the info we need to forward the model
        
        Keyword Arguments:
            deterministic {bool} -- whether to use the deterministic policy (default: {False})
        
        Returns:
            PolicyReturn -- all the stuff we needed
        """

        policy_return = PolicyReturn()

        device = next(self.parameters()).device

        def cudify_state(state, device):
            if type(state) == tuple:
                return tuple([cudify_state(s, device) for s in state])
            else:
                return state.to(device)


        state_tuple, merged_state = self.encode(
                observation=current_memory['current_obs'].to(device),
                actions=current_memory['oneHotActions'].to(device).detach(),
                previous_latent_state=cudify_state(current_memory['states'],
                    device),
            )

        # Apply batch norm if so configured
        if self.policy_batch_norm:
            state = self.encoding_bn(merged_state)

        # Fill up policy_return with return values
        policy_return.latent_state = state_tuple

        policy_return.value_estimate = self.critic_linear(merged_state)
        action = self.dist.sample(merged_state, deterministic=deterministic)
        policy_return.action = action

        if self.dist.__class__.__name__ == 'Categorical':
            policy = self.dist(merged_state)
        else:
            policy, _ = self.dist(merged_state)

        action_log_probs, dist_entropy =\
            self.dist.logprobs_and_entropy(merged_state, action.detach())
        policy_return.action_log_probs = action_log_probs
        policy_return.dist_entropy = dist_entropy

        return policy_return


class PFRNN_Policy(Policy):
    def __init__(self,
                 action_space,
                 nr_inputs,
                 observation_type,
                 action_encoding,
                 cnn_channels,
                 h_dim,
                 encoder_batch_norm,
                 policy_batch_norm,
                 batch_size,
                 resample,
                 dropout=0.1,
                 num_particles=10,
                 num_features=256,
                 particle_aggregation='mgf',
                 ):
        """The PFRNN policy class
        
        Arguments:
            action_space {gym.space} -- action space
            nr_inputs {int} -- observation channels
            observation_type {string} -- observation type
            action_encoding {bool} -- if we want to encode the action
            cnn_channels {int} -- channels for the observation encoder
            h_dim {int} -- hidden dim
            encoder_batch_norm {bool} -- whether to use batch norm for encoder
            policy_batch_norm {bool} -- whether to use batch norm for policy
            batch_size {int} -- batch size
            resample {bool} -- whether to perform soft-resampling
        
        Keyword Arguments:
            dropout {float} -- drop out rate (default: {0.1})
            num_particles {int} -- number of particles (default: {15})
            num_features {int} -- number of mgf features (default: {256})
            particle_aggregation {str} -- method for aggregating the particles (default: {'mgf'})
        
        Raises:
            NotImplementedError
        """

        super().__init__(action_space, encoding_dimension=h_dim)

        self.h_dim = h_dim
        self.batch_size = batch_size
        self.encoder_batch_norm = encoder_batch_norm
        self.policy_batch_norm = policy_batch_norm
        self.observation_type = observation_type
        self.resample = resample
        self.dropout = dropout
        self.particle_aggregation = particle_aggregation
        self.num_features = num_features

        # All encoders and decoders are define centrally in one file
        self.encoder = encoder.get_encoder(
            observation_type,
            nr_inputs,
            cnn_channels,
            batch_norm=encoder_batch_norm
        )

        self.cnn_output_dimension = encoder.get_cnn_output_dimension(
            observation_type,
            cnn_channels
            )
        self.cnn_output_number = reduce(mul, self.cnn_output_dimension, 1)

        # Actions are encoded using one FC layer.
        if action_encoding > 0:
            if action_space.__class__.__name__ == "Discrete":
                action_shape = action_space.n
            else:
                action_shape = action_space.shape[0]
            if encoder_batch_norm:
                self.action_encoder = nn.Sequential(
                    nn.Linear(action_shape, action_encoding),
                    nn.BatchNorm1d(action_encoding),
                    nn.ReLU()
                )
            else:
                self.action_encoder = nn.Sequential(
                    nn.Linear(action_shape, action_encoding),
                    nn.ReLU())

        self.num_particles = num_particles
        self.rnn = PFGRUCell(self.num_particles, self.cnn_output_number +
            action_encoding, self.cnn_output_number, h_dim, 0.9, True,
            'relu')

        # initialize the particle aggregator for belief approximation
        if self.particle_aggregation == 'mgf':
            agg = MGF_Aggregator
        elif self.particle_aggregation == 'mean':
            agg = Mean_Aggregator
        elif self.particle_aggregation == 'gru':
            agg = GRU_Aggregator
        else:
            raise NotImplementedErro

        self.agg = agg(self.num_particles, self.num_features,
                self.h_dim, self.cnn_output_number)

        if observation_type == 'fc':
            self.obs_criterion = nn.MSELoss()
        else:
            self.obs_criterion = nn.BCEWithLogitsLoss()

        self.train()
        self.reset_parameters()

    def new_latent_state(self):
        """
        Return new latent state.
        self.batch_size is the number of parallel environments being used.
        """
        h0 = torch.zeros(self.batch_size * self.num_particles, self.h_dim)
        p0 = torch.zeros(self.batch_size * self.num_particles, 1)
        return (h0, p0)
        
    def logpdf(self, value, mean, var):
        return torch.sum((-0.5 * (value - mean)**2 / var - 0.5 * 
            torch.log(2 * var * np.pi)), dim=1)


    def vec_conditional_new_latent_state(self, latent_states, masks):
        """
        Set latent state to 0 when new episode beings.
        Masks and latent_states contain the values for each of the 16 environments.
        """
        h0 = latent_states[0]
        p0 = latent_states[1]
        return (h0 * masks, p0 * masks)

    def reset_parameters(self):
        kaimin_normal = torch.nn.init.kaiming_normal_
        xavier_normal = torch.nn.init.xavier_normal_
        gain = nn.init.calculate_gain('relu')
        orthogonal = torch.nn.init.orthogonal_

        def weights_init():
            def fn(m):
                classname = m.__class__.__name__
                if classname.find('Conv') != -1 or classname.find('Linear') != -1:
                    kaimin_normal(m.weight.data)
                    try:
                        m.bias.data.fill_(0)
                    except:
                        pass

            return fn

        self.apply(weights_init())
        if self.dist.__class__.__name__ == "DiagGaussian":
            self.dist.fc_mean.weight.data.mul_(0.01)

    def encode(self, observation, actions, previous_latent_state):
        x = self.encoder(observation)
        x = x.view(-1, self.cnn_output_number)

        encoded_actions = None
        if hasattr(self, 'action_encoder'):
            encoded_actions = self.action_encoder(actions)
            encoded_actions = F.relu(encoded_actions)
            x_act = torch.cat([x, encoded_actions], dim=1)

        # GRU
        if hasattr(self, 'rnn'):
            x_reshape = x_act.repeat(self.num_particles, 1)
            latent_state = self.rnn(x_reshape, previous_latent_state)

        state_tuple = latent_state

        merged_state, particles, weight = self.agg(state_tuple[0], state_tuple[-1])

        return state_tuple, merged_state

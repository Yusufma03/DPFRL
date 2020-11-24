import torch
from torch import nn
import numpy as np
import ipdb as pdb
import torch.nn.functional as F

class PFRNNBaseCell(nn.Module):
    """parent class for PFRNNs
    """
    def __init__(self, num_particles, input_size, hidden_size, resamp_alpha,
            use_resampling, activation):
        """init function
        
        Arguments:
            num_particles {int} -- number of particles
            input_size {int} -- input size
            hidden_size {int} -- particle vector length
            resamp_alpha {float} -- alpha value for soft-resampling
            use_resampling {bool} -- whether to use soft-resampling
            activation {str} -- activation function to use
        """
        super(PFRNNBaseCell, self).__init__()
        self.num_particles = num_particles
        self.input_size = input_size
        self.h_dim = hidden_size
        self.resamp_alpha = resamp_alpha
        self.use_resampling = use_resampling
        self.activation = activation

        self.batch_norm = nn.BatchNorm1d(self.num_particles)

    def resampling(self, particles, prob):
        """soft-resampling
        
        Arguments:
            particles {tensor} -- the latent particles
            prob {tensor} -- particle weights
        
        Returns:
            tuple -- particles
        """

        resamp_prob = self.resamp_alpha * torch.exp(prob) + (1 -
                self.resamp_alpha) * 1 / self.num_particles
        resamp_prob = resamp_prob.view(self.num_particles, -1)
        indices = torch.multinomial(resamp_prob.transpose(0, 1),
                num_samples=self.num_particles, replacement=True)
        batch_size = indices.size(0)
        indices = indices.transpose(1, 0).contiguous()
        offset = torch.arange(batch_size).type(torch.LongTensor).unsqueeze(0)
        offset = offset.to(indices.device)
        indices = offset + indices * batch_size
        flatten_indices = indices.view(-1, 1).squeeze()

        # PFLSTM
        if type(particles) == tuple:
            particles_new = (particles[0][flatten_indices],
                    particles[1][flatten_indices])
        # PFGRU
        else:
            particles_new = particles[flatten_indices]

        prob_new = torch.exp(prob.view(-1, 1)[flatten_indices])
        prob_new = prob_new / (self.resamp_alpha * prob_new + (1 -
            self.resamp_alpha) / self.num_particles)
        prob_new = torch.log(prob_new).view(self.num_particles, -1, 1)
        prob_new = prob_new - torch.logsumexp(prob_new, dim=0, keepdim=True)

        return particles_new, prob_new

    def reparameterize(self, mu, var):
        """Implements the reparameterization trick introduced in [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114)
        
        Arguments:
            mu {tensor} -- learned mean
            var {tensor} -- learned variance
        
        Returns:
            tensor -- sample
        """
        std = torch.nn.functional.softplus(var)
        if torch.cuda.is_available():
            eps = torch.cuda.FloatTensor(std.shape).normal_()
        else:
            eps = torch.FloatTensor(std.shape).normal_()

        return mu + eps * std


class PFGRUCell(PFRNNBaseCell):
    def __init__(self, num_particles, input_size, obs_size, hidden_size, resamp_alpha, use_resampling, activation):
        super().__init__(num_particles, input_size, hidden_size, resamp_alpha,
                use_resampling, activation)

        self.fc_z = nn.Linear(self.h_dim + self.input_size, self.h_dim)
        self.fc_r = nn.Linear(self.h_dim + self.input_size, self.h_dim)
        self.fc_n = nn.Linear(self.h_dim + self.input_size, self.h_dim * 2)

        self.fc_obs = nn.Linear(self.h_dim + self.input_size, 1)


    def forward(self, input_, hx):
        """One step forward for PFGRU
        
        Arguments:
            input_ {tensor} -- the input tensor
            hx {tuple} -- previous hidden state (particles, weights)
        
        Returns:
            tuple -- new tensor
        """

        h0, p0 = hx
        obs_in = input_

        z = torch.sigmoid(self.fc_z(torch.cat((h0, input_), dim=1)))
        r = torch.sigmoid(self.fc_r(torch.cat((h0, input_), dim=1)))
        n = self.fc_n(torch.cat((r * h0, input_), dim=1))

        mu_n, var_n = torch.split(n, split_size_or_sections=self.h_dim, dim=1)
        n = self.reparameterize(mu_n, var_n)

        if self.activation == 'relu':
            # if we use relu as the activation, batch norm is require
            n = n.view(self.num_particles, -1, self.h_dim).transpose(0,
                    1).contiguous()
            n = self.batch_norm(n)
            n = n.transpose(0, 1).contiguous().view(-1, self.h_dim)
            n = torch.relu(n)
        elif self.activation == 'tanh':
            n = torch.tanh(n)
        else:
            raise ModuleNotFoundError

        h1 = (1 - z) * n + z * h0

        p1 = self.observation_likelihood(h1, obs_in, p0)

        if self.use_resampling:
            h1, p1 = self.resampling(h1, p1)

        p1 = p1.view(-1, 1)

        return h1, p1

    def observation_likelihood(self, h1, obs_in, p0):
        """observation function based on compatibility function
        """
        logpdf_obs = self.fc_obs(torch.cat((h1, obs_in), dim=1))

        p1 = logpdf_obs + p0

        p1 = p1.view(self.num_particles, -1, 1)
        p1 = nn.functional.log_softmax(p1, dim=0)

        return p1
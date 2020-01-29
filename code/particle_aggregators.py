import torch
import torch.nn as nn

class Aggregator(nn.Module):
    def __init__(self, num_particles, num_features, h_dim, obs_encode_dim):
        """Base class for particle aggregators
        
        Arguments:
            nn {[type]} -- [description]
            num_particles {int} -- number of particles
            num_features {int} -- number of mgf features, used only for MGF aggregator
            h_dim {int} -- hidden dimension of particles
            obs_encode_dim {int} -- the size of the encoded observation
        """
        super().__init__()
        self.num_particles = num_particles
        self.num_features = num_features
        self.h_dim = h_dim
        self.obs_encode_dim = obs_encode_dim

    def forward(self, particles, weight):
        # merge particles and return the (merged_particle, particle, weight)
        raise NotImplementedError

class Mean_Aggregator(Aggregator):
    def __init__(self, num_particles, num_features, h_dim, obs_encode_dim):
        super().__init__(num_particles, num_features, h_dim, obs_encode_dim)

    def forward(self, particles, weight):
        particles = particles.view(self.num_particles, -1, self.h_dim)

        weight = weight.view(self.num_particles, -1, 1)

        merged = torch.sum(particles * torch.exp(weight), dim=0)

        return merged, particles.view(-1, self.h_dim), weight.view(-1, 1)


class MGF_Aggregator(Aggregator):
    def __init__(self, num_particles, num_features, h_dim, obs_encode_dim):
        super().__init__(num_particles, num_features, h_dim, obs_encode_dim)

        # One could consider to use relu as activation 
        # This generalize the MGF feature for a more stable gradient

        self.mgf_act = torch.relu

        self.mgf_fc = nn.Linear(self.h_dim, self.num_features, bias=False)
        self.merge_fc = nn.Linear(self.h_dim + self.num_features, self.h_dim)
        self.mgf_bn = nn.BatchNorm1d(self.num_particles)

    def forward(self, particles, weight):
        mgf_features = self.mgf_fc(particles).view(self.num_particles, -1,
                self.num_features).transpose(0, 1).contiguous()
        mgf_features = self.mgf_bn(mgf_features)
        mgf_features = self.mgf_act(mgf_features)
        mgf_features = mgf_features.transpose(0, 1).contiguous()

        weight = weight.view(self.num_particles, -1, 1)

        particles_reshape = particles.view(self.num_particles, -1, self.h_dim)

        # first-order moment
        mean = torch.sum(particles_reshape * torch.exp(weight), dim=0)

        # MGF features
        mgf = torch.sum(mgf_features * torch.exp(weight), dim=0)

        merged = torch.cat((mean, mgf), dim=1)

        # merge features
        merged = torch.relu(self.merge_fc(merged))

        return merged, particles.view(-1, self.h_dim), weight.view(-1, 1)


class GRU_Aggregator(Aggregator):
    def __init__(self, num_particles, num_features, h_dim, obs_encode_dim):
        super().__init__(num_particles, num_features, h_dim, obs_encode_dim)
        self.merge_gru = nn.GRU(self.h_dim + 1, h_dim, batch_first=False)

    def forward(self, particles, weight):
        weight = weight.view(self.num_particles, -1, 1)

        particles = particles.view(self.num_particles, -1, self.h_dim)

        feature = torch.cat((particles, torch.exp(weight)), dim=2)
        _, merged = self.merge_gru(feature)

        return merged[0], particles.view(-1, self.h_dim), weight.view(-1, 1)

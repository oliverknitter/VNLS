import torch
import torch.nn as nn

from .hamiltonian import Hamiltonian


class MaxCut(Hamiltonian):
    def __init__(self, adjacency):
        super(MaxCut, self).__init__()
        self.adjacency = adjacency

    def get_laplacian(self):
        # laplacian matrix = degree matrix - adjacency matrix
        adjacency = self.adjacency
        adjacency = torch.tensor(adjacency)
        laplacian = torch.diag(adjacency.sum(-1)) - adjacency
        return laplacian.float()

    def compute_local_energy(self, samples, model):
        batch_size = samples.shape[0]
        num_sites = samples.shape[1]
        laplacian = self.get_laplacian()
        # cut = sum_{all edges ij} (1-z_i*z_j)/2
        laplacian_batch = laplacian.unsqueeze(0).repeat(batch_size, 1, 1)
        cut = 0.25 * torch.bmm(torch.bmm(samples.unsqueeze(1), laplacian_batch), samples.unsqueeze(-1)).squeeze()
        # total local energy
        local_energy = -cut
        log_psi = model(samples)
        return local_energy, log_psi


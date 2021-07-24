import torch
import torch.nn as nn

class Hamiltonian(nn.Module):
    def __init__(self):
        super(Hamiltonian, self).__init__()

    def forward(self, config):
        pass


class Energy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, log_psi, local_energies):
        ctx.local_energies = local_energies
        ctx.mean_energy = local_energies.mean()
        return local_energies

    @staticmethod
    def backward(ctx, log_psi_grad):
        grad = ctx.local_energies - ctx.mean_energy
        grad_total = 2 * log_psi_grad * grad
        if grad_total.dtype == torch.complex64:
            grad_total = grad_total.real
        return grad_total, None
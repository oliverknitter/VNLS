import numpy as np
import scipy
import torch
from scipy.sparse.linalg import minres
from torch.optim.optimizer import Optimizer, required

from .model.util import apply_grad, mix_grad, vec_to_grad

class SR:
    def __init__(self):
        self.S = None
        self.cutoff = 1.0e-10

    def get_sr_mtx(self, flatten_log_grads_batch):
        # compute covariant matrix
        flatten_delta_mean = flatten_log_grads_batch.mean(0, keepdims=True) # [1, dim_w]
        flatten_delta_centred = flatten_log_grads_batch - flatten_delta_mean # [bs, dim_w]
        # S = (flatten_delta_centred.T @ flatten_delta_centred) / flatten_delta_centred.shape[0] # [dim_w, dim_w]
        S = torch.einsum("bj,jk->bk", (flatten_delta_centred.T, flatten_delta_centred)) / flatten_delta_centred.shape[0] # [dim_w, dim_w]
        S = (S + S.T) / 2
        return S

    def compute_grad(self, A, b):
        x = minres(A, b, x0=b)[0]
        # x = np.linalg.solve(A, b)
        # x = sp.linalg.cg(A, b, maxiter=5)[0]
        return x

    def apply_scale_invariant(self):
        self.diag_S = self.S.diagonal()
        # Diagonal of S should always be real since it is Hermitian.
        if self.diag_S.dtype == torch.complex64:
            self.diag_S = self.diag_S.real
        self.diag_S_sqrt = self.diag_S.sqrt()
        index = self.diag_S_sqrt <= self.cutoff
        self.diag_S_sqrt[index] = 1.0
        self.S[index, :] = 0
        self.S[:, index] = 0
        # diag = self.S.diagonal().unsqueeze(0)
        # scale = torch.einsum("bj,jk->bk", (diag.T, diag)).sqrt()
        # self.S = self.S / scale
        # self.S = self.S + (self.S.diagonal() * dump).diag()
        self.S_scaled = self.S / (self.diag_S_sqrt.unsqueeze(0) * self.diag_S_sqrt.unsqueeze(1))
        # self.S.fill_diagonal_(1)
        self.grad_scaled = self.grad / self.diag_S_sqrt

    # https://github.com/netket/netket/blob/master/Sources/Optimizer/py_stochastic_reconfiguration.cc
    # https://github.com/netket/netket/blob/2c43c5ada5b1137248d5949ff7cd2d4f5ac18d0c/netket/optimizer/numpy/stochastic_reconfiguration.py#L15
    @torch.no_grad()
    def compute_natural_grad(self, model, grad, flatten_log_grads_batch, dump, scale_invar):
        self.S = self.get_sr_mtx(flatten_log_grads_batch)
        self.grad = grad
        if scale_invar:
            self.apply_scale_invariant()
        else:
            self.S_scaled, self.grad_scaled = self.S, self.grad
        self.S_scaled = self.S_scaled + (torch.ones_like(self.S_scaled.diagonal()) * dump).diag()
        natural_grad_scaled = torch.tensor(self.compute_grad(self.S_scaled.numpy(), self.grad_scaled.numpy()))
        if scale_invar:
            natural_grad = natural_grad_scaled / self.diag_S_sqrt
        # truncate the gradient to avoid gradient blow
        natural_grad = natural_grad.clamp(-1, 1)
        return natural_grad

    @torch.no_grad()
    def apply_sr_grad(self, model, grad, flatten_log_grads_batch, dump, scale_invar):
        natural_grad = self.compute_natural_grad(model, grad, flatten_log_grads_batch, dump, scale_invar)
        apply_grad(model, vec_to_grad(natural_grad, model))
        # # record error of the solver
        # err1 = ((self.S_scaled @ natural_grad_scaled.unsqueeze(-1)).squeeze() - self.grad_scaled).norm()/self.grad_scaled.norm()
        # err2 = ((self.S @ natural_grad.unsqueeze(-1)).squeeze() - self.grad).norm()/self.grad.norm()
        # return grad.norm().item(), natural_grad.norm().item(), err1.item(), err2.item()


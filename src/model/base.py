import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

from backpack import backpack
from backpack.extensions import BatchGrad

class Base(nn.Module):
    def __init__(self, **kwargs):
        super(Base, self).__init__()

    def forward(self, configuration):
        pass

    def log_dev(self, log_psi, model):
        bs = log_psi.shape[0]
        with backpack(BatchGrad()):
            log_psi.sum().backward(retain_graph=True)
        ccat_log_grads_batch = torch.tensor([]) # [w_dim]
        for name, param in model.named_parameters():
            if param.grad is None:
                continue
            ccat_log_grads_batch = torch.cat((ccat_log_grads_batch, param.grad_batch.reshape(bs, -1)), dim=-1)
        model.zero_grad()
        return ccat_log_grads_batch
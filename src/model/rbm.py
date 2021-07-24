
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

from .base import Base

# https://github.com/bacnguyencong/rbm-pytorch/blob/master/rbm.py
class RBM(Base):
    def __init__(self, num_sites, **kwargs):
        super(RBM, self).__init__()
        n_vis = int(num_sites)
        n_hid = int(num_sites)
        self.c = nn.Linear(in_features=n_vis, out_features=n_hid, bias=True)
        self.a = nn.Linear(in_features=n_vis, out_features=1, bias=False)
        self.softplus = torch.nn.Softplus()
        # model initialization
        std = 0.01
        self.c.weight.data.normal_(std=std)
        self.c.bias.data.normal_(std=std)
        self.a.weight.data.normal_(std=std)

    def forward(self, configuration):
        c = self.c(configuration)
        # lncoshc = self.softplus(-2*c) + c
        lncoshc = self.softplus(-2*c) + c - np.log(2)
        sigma_a = self.a(configuration).squeeze(-1)
        log_psi = sigma_a + lncoshc.sum(-1)
        return log_psi

# https://github.com/pytorch/pytorch/blob/master/tools/autograd/gen_variable_type.py#L151-L164
class RBM_c(Base):
    def __init__(self, num_sites, **kwargs):
        super(RBM_c, self).__init__()
        n_vis = num_sites
        n_hid = int(num_sites)
        self.c_real = nn.Linear(in_features=n_vis, out_features=n_hid, bias=True)
        self.a_real = nn.Linear(in_features=n_vis, out_features=1, bias=False)
        self.c_imag = nn.Linear(in_features=n_vis, out_features=n_hid, bias=True)
        self.a_imag = nn.Linear(in_features=n_vis, out_features=1, bias=False)
        self.cosh = torch.cosh
        # model initialization
        std = 0.01
        self.c_real.weight.data.normal_(std=std)
        self.c_real.bias.data.normal_(std=std)
        self.a_real.weight.data.normal_(std=std)
        self.a_imag.weight.data.normal_(std=std)
        self.c_imag.bias.data.normal_(std=std)
        self.a_imag.weight.data.normal_(std=std)

    def forward(self, configuration):
        c_real = self.c_real(configuration)
        c_imag = self.c_imag(configuration)
        c = torch.view_as_complex(torch.stack((c_real, c_imag), dim=-1))
        coshc = self.cosh(c)
        lncoshc = coshc.log()
        sigma_a_real = self.a_real(configuration).squeeze(-1)
        sigma_a_imag = self.a_imag(configuration).squeeze(-1)
        sigma_a = torch.view_as_complex(torch.stack((sigma_a_real, sigma_a_imag), dim=-1))
        log_psi = sigma_a + lncoshc.sum(-1)
        return log_psi

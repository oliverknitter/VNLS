import re
import numpy as np
import logging
import torch
import torch.nn as nn
from pytorch_model_summary import summary


def get_model(model_name, device, print_model_info, **kwargs):
    if model_name in ['rbm']:
        from .rbm import RBM
        model = RBM(**kwargs)
        if print_model_info:
            print(summary(model, torch.zeros(10, list(model.parameters())[0].shape[1]), show_input=False))
    elif model_name in ['rbm_c']:
        from .rbm import RBM_c
        model = RBM_c(**kwargs)
        if print_model_info:
            print(summary(model, torch.zeros(10, list(model.parameters())[0].shape[1]), show_input=False))
    else:
        raise "Unknown model_name."
    model.eval()
    return model.to(device)

def load_model(model, model_load_path, strict=True):
    # logging.info("[*] Load model from {}...".format(model_load_path))
    bad_state_dict = torch.load(model_load_path, map_location='cpu')
    correct_state_dict = {re.sub(r'^module\.', '', k): v for k, v in \
                            bad_state_dict.items()}
    if not strict:
        logging.info("Loading {} params".format(len(correct_state_dict)))
        own_state = model.state_dict()
        final_state_dict = {}
        for name, param in correct_state_dict.items():
            if name not in own_state:
                    continue
            param = param.data
            own_param = own_state[name].data
            if own_param.shape == param.shape:
                final_state_dict[name] = param
        correct_state_dict =  final_state_dict
        logging.info("Loaded {} params".format(len(correct_state_dict)))
    model.load_state_dict(correct_state_dict, strict=strict)
    model.eval()
    model.zero_grad()
    return model

def save_model(model, model_save_path):
    # logging.info("[*] Save model to {}...".format(model_save_path))
    torch.save(model.state_dict(), model_save_path)
    return model_save_path

def compute_weight_diff(model_param1, model_param2):
    model_weight1 = torch.nn.utils.parameters_to_vector(model_param1)
    model_weight2 = torch.nn.utils.parameters_to_vector(model_param2)
    diff = (model_weight1 - model_weight2).norm()
    return diff

def apply_grad(model, grad):
    grad_norm = 0
    for p, g in zip(model.parameters(), grad):
        if p.grad is None:
            p.grad = g.to(p.device)
        else:
            p.grad += g.to(p.device)
        grad_norm += torch.sum(g**2)
    grad_norm = grad_norm ** (1/2)
    return grad_norm.item()

def mix_grad(grad_list):
    mixed_grad = []
    for g_list in zip(*grad_list):
        g_list = torch.stack([g_list[i] for i in range(len(g_list))])
        mixed_grad.append(torch.mean(g_list, dim=0))
    return mixed_grad

def vec_to_grad(vec, model):
    pointer = 0
    res = []
    for param in model.parameters():
        num_param = param.numel()
        res.append(vec[pointer:pointer+num_param].view_as(param).data)
        pointer += num_param
    return res

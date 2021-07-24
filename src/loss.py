
import torch

def get_loss(pb_type):
    if pb_type in ['maxcut', 'vqls']:
        from .objective.hamiltonian import Energy
        return Energy.apply

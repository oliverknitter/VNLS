
import os
import cv2
import random
import logging
import numpy as np
import matplotlib.pyplot as plt
import torch
from datetime import datetime
from scipy.signal import find_peaks


def set_seed(seed):
    # the following are for reproducibility on GPU, see https://pytorch.org/docs/master/notes/randomness.html
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def get_time():
    return datetime.now().strftime("%m%d_%H%M%S")

def folder_name_generator(cfg, opts):
    name_str = []
    name_str.append('{}'.format(cfg.MISC.MODE))
    for i,arg in enumerate(opts):
        if i % 2 == 1:
            name_str.append('{}'.format(arg))
    # name_str.append(get_time())
    return '-'.join(name_str)

def prepare_dirs(cfg, opts):
    if not os.path.exists('./datasets'):
        os.makedirs('./datasets')
    if not os.path.exists('./results'):
        os.makedirs('./results')
    if not os.path.exists('./logger'):
        os.makedirs('./logger')
    if cfg.MISC.DIR == '':
        cfg.MISC.DIR = folder_name_generator(cfg, opts)
        cfg.MISC.DIR = './logger/{}'.format(cfg.MISC.DIR)
    if not os.path.exists(cfg.MISC.DIR):
        os.makedirs(cfg.MISC.DIR)
    # set logger
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(cfg.MISC.DIR, 'debug.log')),
            logging.StreamHandler()
        ]
    )

def write_file(file_name, content):
    f=open(file_name, "a+")
    f.write(content)
    f.write("\n")
    f.close()

def plot_training_loss(prob_type, qubits, epochs, values, fidelities = None):
    epoch_range = np.arange(epochs+1)
    plt.plot(epoch_range, values, label='Loss')
    plt.xlabel('Training Epochs')
    plt.ylabel('Training Loss')
    plt.title('Training loss of '+str(qubits)+'-qubit VNLS problem.')
    plt.savefig('./results/plots/{}_{}_loss.png'.format(prob_type, qubits))
    np.save('./results/arrays/{}_{}_losses.npy'.format(prob_type, qubits), values)
    if fidelities is not None:
        #plt.clf()
        #plt.plot(epoch_range, 10*np.sqrt(values), label='Loss Bound')
        #trace_distances = np.sqrt(1-fidelities)
        #plt.plot(epoch_range, trace_distances, label='Trace Distance')
        #plt.legend()
        #plt.xlabel('Training Epochs')
        #plt.ylabel('Training Loss')
        #plt.title('Trace distance and VQLS bound of '+str(qubits)+'-qubit VQLS problem.')
        #plt.savefig('./results/plots/{}_{}_trace.png'.format(prob_type, qubits))

        plt.clf()
        plt.plot(epoch_range, fidelities)
        plt.xlabel('Training Epochs')
        plt.ylabel('Fidelity')
        plt.title('Fidelity of learned solution for '+str(qubits)+'-qubit VNLS problem.')
        plt.savefig('./results/plots/{}_{}_fidelity.png'.format(prob_type, qubits))
        np.save('./results/arrays/{}_{}_fidelities.npy'.format(prob_type, qubits), fidelities)
    
    plt.close()
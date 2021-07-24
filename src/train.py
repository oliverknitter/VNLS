import os
import numpy as np
import time
import logging
from tqdm import trange
import torch
from tensorboardX import SummaryWriter

from backpack import backpack, extend

from .sr import SR
from .objective.util import get_hamiltonian
from .sampler.metropolis_hasting import MetropolisHasting
from .evaluate import test
from .data_loader import load_data
from .loss import get_loss
from .scheduler import get_scheduler
from .optimizer import get_optimizer
from .model.util import get_model, load_model, save_model
from .util import plot_training_loss


def train_one_batch(model, sampler, criterion, hamiltonian, optimizer, scheduler, sr, dump_fac, iter_per_batch):
    device = list(model.parameters())[0].device
    if sr is not None:
        model = extend(model)
    # collect samples
    samples = torch.tensor(sampler()).float().to(device)
    if iter_per_batch > 1:
        # use ISGO trick: https://arxiv.org/pdf/1905.10730.pdf
        # denominator of renormalization factor
        log_psi_0 = model(samples).detach()
    for i in range(iter_per_batch):
        # train
        optimizer.zero_grad()
        local_energies, log_psi = hamiltonian.compute_local_energy(samples, model)
        loss = criterion(log_psi, local_energies)
        if i > 0:
            loss = loss * (2*(log_psi - log_psi_0)).exp().detach()
        loss = loss.mean()
        # compute losses
        if sr is not None:
            ccat_log_grads_batch = model.log_dev(log_psi, model)
            grad = torch.nn.utils.parameters_to_vector(torch.autograd.grad(loss, list(model.parameters())))
            sr.apply_sr_grad(model, grad, ccat_log_grads_batch, dump_fac, scale_invar=True)
        else:
            loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        scheduler.step()
    return loss

def train(cfg):
    # set hyper-parameters
    # settings
    exact_test_threshold = cfg.EVAL.EXACT_TEST_THRESHOLD # Default threshold for conducting exact testing for VQLS solution is at most 20 qubits.
    prob_type = cfg.DATA.PROBLEM_TYPE
    num_sites = cfg.DATA.NUM_SITES
    device = torch.device('cuda:0' if (cfg.SYSTEM.NUM_GPUS > 0) else 'cpu')
    logger_dir = cfg.MISC.DIR
    # train
    lr = cfg.TRAIN.LEARNING_RATE
    num_epochs = cfg.TRAIN.NUM_EPOCHS
    bs = cfg.TRAIN.BATCH_SIZE
    opt_name = cfg.TRAIN.OPTIMIZER_NAME
    sche_name = cfg.TRAIN.SCHEDULER_NAME
    dump_fac = cfg.TRAIN.DUMP_FACTOR
    apply_sr = cfg.TRAIN.APPLY_SR
    iter_per_batch = cfg.TRAIN.ITER_PER_BATCH
    if apply_sr:
        sr = SR()
    else:
        sr = None
    # sampling
    num_chains = cfg.DATA.NUM_CHAINS
    burn_in = cfg.DATA.BURN_IN
    init_prev_state = cfg.TRAIN.INIT_PREV_STATE
    # model
    model_name = cfg.MODEL.MODEL_NAME
    model_load_path = cfg.EVAL.MODEL_LOAD_PATH
    # load model
    model = get_model(model_name, device, print_model_info=True, num_sites=num_sites)
    if model_load_path:
        model = load_model(model, model_load_path)
    # load data
    info = load_data(cfg)
    # set up training
    hamiltonian = get_hamiltonian(prob_type, **info)
    criterion = get_loss(prob_type)
    optimizer = get_optimizer(opt_name, model, lr)
    scheduler = get_scheduler(sche_name, optimizer, lr, num_epochs)
    sampler = MetropolisHasting(model, bs, num_sites, burn_in, num_chains, init_prev_state)
    # tensorboard
    tensorboard = SummaryWriter(log_dir=logger_dir)
    tensorboard.add_text(tag='argument', text_string=str(cfg.__dict__))
    # train
    best_score = 0.0
    time_elapsed = 0.0
    progress_bar = trange(num_epochs, desc='Progress Bar', leave=True)

    loss_values = np.zeros(num_epochs + 1) # Set up losses for training loss plotter
    score, max_score, variance, avg_accept = test(model, sampler, hamiltonian)
    loss_values[0] = score
    if prob_type == 'vqls' and num_sites <= exact_test_threshold:
        fidelities = np.zeros(num_epochs + 1)
        learned, true, fidelity = hamiltonian.exact_test(num_sites, model)
        fidelities[0] = fidelity

    for epoch in range(1, num_epochs + 1):
        start_time = time.time()
        train_one_batch(model, sampler, criterion, hamiltonian, optimizer, scheduler, sr, dump_fac, iter_per_batch)
        end_time = time.time()
        time_elapsed += end_time - start_time
        # log
        score, max_score, variance, avg_accept = test(model, sampler, hamiltonian)
        tensorboard.add_scalar('test/score', score, epoch)
        if prob_type == 'vqls':
            message = '[Test Epoch {}] Score: {:.4f}, Variance: {:.8f}, Acceptance ratio: {:.4f}'.format(epoch, score, variance, avg_accept)
        else:
            message = '[Test Epoch {}] Score: {:.4f}, Top Score: {:.4f}, Var: {:.8f}, Acceptance ratio: {:.4f}'.format(epoch, score, max_score, variance, avg_accept)
        if score < best_score:
            best_score = score
            save_model(model, os.path.join(logger_dir, 'model_best.pth'))
        
        loss_values[epoch] = score
        if prob_type == 'vqls' and num_sites <= exact_test_threshold:
            learned, true, fidelity = hamiltonian.exact_test(num_sites, model)
            fidelities[epoch] = fidelity
            if epoch%100 == 0:
                print('Epoch: {}, Fidelity: {}'.format(epoch, fidelity))
        
        progress_bar.set_description(message)
        progress_bar.refresh() # to show immediately the update
        progress_bar.update(1)

    if prob_type == 'vqls' and num_sites <= exact_test_threshold: # Checks result against exact solution for small problem size.
        learned, true, fidelity = hamiltonian.exact_test(num_sites, model)
        if num_sites <= 10:
            print('Normalized learned state: '+str(['{:.4f}'.format(x) for x in learned.numpy()])+',\nNormalized true state: '+str(['{:.4f}'.format(x) for x in true.numpy()])+',\nFidelity: {:.8f}'.format(fidelity.numpy()))
        else:
            print('Fidelity: {:.8f}'.format(fidelity.numpy()))
        plot_training_loss(prob_type, num_sites, num_epochs, loss_values, fidelities)
    else:
        plot_training_loss(prob_type, num_sites, num_epochs, loss_values)

    return best_score, time_elapsed

def alt_train(cfg): # Alternate train function without loss-plotting components.
    # set hyper-parameters
    # settings
    exact_test_threshold = cfg.EVAL.EXACT_TEST_THRESHOLD # Default threshold for conduting exact testing for VQLS solution is less than ten qubits.
    prob_type = cfg.DATA.PROBLEM_TYPE
    num_sites = cfg.DATA.NUM_SITES
    device = torch.device('cuda:0' if (cfg.SYSTEM.NUM_GPUS > 0) else 'cpu')
    logger_dir = cfg.MISC.DIR
    # train
    lr = cfg.TRAIN.LEARNING_RATE
    num_epochs = cfg.TRAIN.NUM_EPOCHS
    bs = cfg.TRAIN.BATCH_SIZE
    opt_name = cfg.TRAIN.OPTIMIZER_NAME
    sche_name = cfg.TRAIN.SCHEDULER_NAME
    dump_fac = cfg.TRAIN.DUMP_FACTOR
    apply_sr = cfg.TRAIN.APPLY_SR
    iter_per_batch = cfg.TRAIN.ITER_PER_BATCH
    if apply_sr:
        sr = SR()
    else:
        sr = None
    # sampling
    num_chains = cfg.DATA.NUM_CHAINS
    burn_in = cfg.DATA.BURN_IN
    init_prev_state = cfg.TRAIN.INIT_PREV_STATE
    # model
    model_name = cfg.MODEL.MODEL_NAME
    model_load_path = cfg.EVAL.MODEL_LOAD_PATH
    # load model
    model = get_model(model_name, device, print_model_info=True, num_sites=num_sites)
    if model_load_path:
        model = load_model(model, model_load_path)
    # load data
    info = load_data(cfg)
    # set up training
    hamiltonian = get_hamiltonian(prob_type, **info)
    criterion = get_loss(prob_type)
    optimizer = get_optimizer(opt_name, model, lr)
    scheduler = get_scheduler(sche_name, optimizer, lr, num_epochs)
    sampler = MetropolisHasting(model, bs, num_sites, burn_in, num_chains, init_prev_state)
    # tensorboard
    tensorboard = SummaryWriter(log_dir=logger_dir)
    tensorboard.add_text(tag='argument', text_string=str(cfg.__dict__))
    # train
    best_score = 0.0
    time_elapsed = 0.0
    progress_bar = trange(num_epochs, desc='Progress Bar', leave=True)

    for epoch in range(1, num_epochs + 1):
        start_time = time.time()
        train_one_batch(model, sampler, criterion, hamiltonian, optimizer, scheduler, sr, dump_fac, iter_per_batch)
        end_time = time.time()
        time_elapsed += end_time - start_time
        # log
        score, max_score, variance, avg_accept = test(model, sampler, hamiltonian)
        tensorboard.add_scalar('test/score', score, epoch)
        if prob_type == 'vqls':
            message = '[Test Epoch {}] Score: {:.4f}, Variance: {:.8f}, Acceptance ratio: {:.4f}'.format(epoch, score, variance, avg_accept)
        else:
            message = '[Test Epoch {}] Score: {:.4f}, Top Score: {:.4f}, Var: {:.8f}, Acceptance ratio: {:.4f}'.format(epoch, score, max_score, variance, avg_accept)
        if score < best_score:
            best_score = score
            save_model(model, os.path.join(logger_dir, 'model_best.pth'))
        
        progress_bar.set_description(message)
        progress_bar.refresh() # to show immediately the update
        progress_bar.update(1)

    if prob_type == 'vqls' and num_sites <= exact_test_threshold: # Checks result against exact solution for small problem size.
        learned, true, fidelity = hamiltonian.exact_test(num_sites, model)
        print('Normalized learned state: '+str(['{:.4f}'.format(x) for x in learned.numpy()])+',\nNormalized true state: '+str(['{:.4f}'.format(x) for x in true.numpy()])+',\nFidelity: {:.8f}'.format(fidelity.numpy()))
        
    return best_score, time_elapsed
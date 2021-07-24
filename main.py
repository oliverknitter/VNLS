import os
import time
import argparse
import logging
import numpy as np
import torch

from config import get_cfg_defaults  # local variable usage pattern, or:
# from config import cfg  # global singleton usage pattern
from src.util import prepare_dirs, set_seed, write_file


def main(cfg):
    mode = cfg.MISC.MODE
    if mode in ['train']:
        from src.train import train
        best_score, time_elapsed = train(cfg)
    return best_score, time_elapsed


if __name__ == '__main__':
    # command line arguments
    parser = argparse.ArgumentParser(description="Command-Line Options")
    parser.add_argument(
        "--config_file",
        default="",
        metavar="FILE",
        help="Path to the yaml config file",
        type=str,
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    # configurations
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # set up directories (cfg.MISC.DIR)
    prepare_dirs(cfg, args.opts)
    # set the settings for MCMC samplers
    cfg.DATA.BURN_IN = int(max(cfg.TRAIN.BATCH_SIZE / cfg.DATA.NUM_CHAINS, 100 + cfg.DATA.NUM_SITES * 3))
    # freeze the configurations
    cfg.freeze()
    # set up configurations
    DIR = cfg.MISC.DIR
    PROBLEM_TYPE = cfg.DATA.PROBLEM_TYPE
    NUM_SITES = cfg.DATA.NUM_SITES
    NUM_TRIALS = cfg.MISC.NUM_TRIALS
    RANDOM_SEED = cfg.SYSTEM.RANDOM_SEED
    RESULT_LOGGER_NAME = cfg.EVAL.RESULT_LOGGER_NAME
    # run program over trials
    AVG_SCORE = 0.0
    AVG_TIME_ELAPSED = 0.0
    write_file(RESULT_LOGGER_NAME, "=============== {} ===============".format(DIR.split('/')[-1]))
    for trial in range(NUM_TRIALS):
        seed = RANDOM_SEED + trial
        # set random seeds
        set_seed(seed)
        best_score, time_elapsed = main(cfg)
        AVG_SCORE += best_score/NUM_TRIALS
        AVG_TIME_ELAPSED += time_elapsed/NUM_TRIALS
    RESULT_LOG = "[VMC--{}, {}] Best Score {:.4f}, Time elapsed {:.2f}".format(PROBLEM_TYPE, NUM_SITES, best_score, time_elapsed)
    write_file(RESULT_LOGGER_NAME, "Trial - {}".format(trial+1))
    write_file(RESULT_LOGGER_NAME, RESULT_LOG)
    # logging
    RESULT_LOG = "[{}][{}-{}] Score {:.4f}, Time elapsed {:.4f}, over {} trials".format(
                 DIR.split('/')[-1], PROBLEM_TYPE, NUM_SITES, AVG_SCORE, AVG_TIME_ELAPSED, NUM_TRIALS)
    write_file(RESULT_LOGGER_NAME, RESULT_LOG)
    logging.info('--------------- Finish ---------------')
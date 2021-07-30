
from yacs.config import CfgNode as CN
# yacs official github page https://github.com/rbgirshick/yacs

_C = CN()
''' System '''
_C.SYSTEM = CN()
# Number of GPUS to use in the experiment
_C.SYSTEM.NUM_GPUS = 0
# Number of workers for doing things
_C.SYSTEM.NUM_WORKERS = 0
# Random seed
_C.SYSTEM.RANDOM_SEED = 0

''' Miscellaneous '''
_C.MISC = CN()
# Functionality mode
# Available choices: ['train', 'eval', 'train_maml']
_C.MISC.MODE = ''
# Logging interval
_C.MISC.LOG_INTERVAL = 0
# Logger path
_C.MISC.DIR = ''
# Number of trials
_C.MISC.NUM_TRIALS = 0

''' Debugger '''
_C.DEBUG = CN()
# Debugging job name
_C.DEBUG.CASE = ''
# Ablation study index
_C.DEBUG.ABLATION_IDX = 0

''' Training hyper-parameters '''
_C.TRAIN = CN()
# Learning rate
_C.TRAIN.LEARNING_RATE = 0.0
# Training optimizer name
# Available choices: ['adam', 'sgd', 'adadelta', 'adamax', 'adagrad']
_C.TRAIN.OPTIMIZER_NAME = ''
# Learning rate scheduler name
# Available choices: ['decay', 'cyclic', 'trap', 'const']
_C.TRAIN.SCHEDULER_NAME = ''
# Training batch size
_C.TRAIN.BATCH_SIZE = 0
# Number of training epochs
_C.TRAIN.NUM_EPOCHS = 0
# The dumping factor for 2nd order optimizers
_C.TRAIN.DUMP_FACTOR = 0.0
# Whether using stochastic reconfiguration or not
_C.TRAIN.APPLY_SR = False
# [ISGO] parameter updating steps within each sampling iteration: https://arxiv.org/pdf/1905.10730.pdf
_C.TRAIN.ITER_PER_BATCH = 0
# Whether to use previously samples as initialization for the current MCMC sampling
_C.TRAIN.INIT_PREV_STATE = False

''' Model hyper-parameters '''
_C.MODEL = CN()
# The name of the model
# Available choices: ['rbm', 'rbm_c']
_C.MODEL.MODEL_NAME = ''

''' Data hyper-parameters '''
_C.DATA = CN()
# Problem type
# Available choices: ['maxcut', 'vqls']
_C.DATA.PROBLEM_TYPE = ''
# Choice of b-vector for vqls problem: ['one_sparse', 'alternation', 'cosine', 'rbm']
_C.DATA.VECTOR_CHOICE = 'one_sparse'
# Number of sites
_C.DATA.NUM_SITES = 0
# Number of burn in iterations for MCMC sampler
_C.DATA.BURN_IN = 0
# Number of sampling chains for MCMC sampler
_C.DATA.NUM_CHAINS = 0

''' Evaluation hyper-parameters '''
_C.EVAL = CN()
# Loading path of the saved model
_C.EVAL.MODEL_LOAD_PATH = ''
# Name of the results logger
_C.EVAL.RESULT_LOGGER_NAME = './results/results.txt'
# Name of the dictionary that stores the results
_C.EVAL.RESULT_DIC_NAME = ''
# Qubit threshold for comparison with exact solution for vqls problem type
_C.EVAL.EXACT_TEST_THRESHOLD = 16

def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for the project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()

# Alternatively, provide a way to import the defaults as
# a global singleton:
# cfg = _C  # users can `from config import cfg`

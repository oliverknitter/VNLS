#!/bin/bash
#SBATCH --job-name=vnls_24
#SBATCH --mail-user=knitter@umich.edu
#SBATCH --mail-type=BEGIN,END
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=32g 
#SBATCH --time=24:00:00
#SBATCH --account=shravan1
#SBATCH --partition=standard
#SBATCH --output=/home/knitter/logs/%x-%j.log

# The application(s) to execute along with its input arguments and options:

module purge
module load python3.8-anaconda/2020.07

eval "$(conda shell.bash hook)"
conda activate qmc
module load gcc/8.2.0

python -m main --config_file='./qaoa.yaml' TRAIN.OPTIMIZER_NAME sgd DATA.PROBLEM_TYPE vqls DATA.VECTOR_CHOICE constant TRAIN.APPLY_SR True DATA.NUM_SITES 24 TRAIN.BATCH_SIZE 128 TRAIN.NUM_EPOCHS 1500 TRAIN.LEARNING_RATE 0.005 DATA.NUM_CHAINS 8 MODEL.MODEL_NAME rbm_c

/bin/hostname
sleep 60

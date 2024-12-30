# VNLS

Implementation of the Variational Neural Linear Solver as presented in Toward Neural Network Simulation of Variational Quantum Algorithms by Oliver Knitter, James Stokes, and Shravan Veerapaneni. This implementation is primarily PyTorch-based and serves as the official implementation associated with the paper. This implementation is based on the restricted Boltzmann Machine (RBM) form of neural quantum states (NQS) by Carleo and Troyer (2017).

This repository is not being actively maintained.

## User Instructions
All necessary package dependencies are contained in the file 'qmc.yml'. Once the correct package versions are installed, the program may be run using the 'run.sh' script. The file 'vnls.yaml' is filled with configuration flags used by the program, though config flags manually entered into 'run.sh' override any listed in the .yaml file. Input linear systems may be entered as pickled dictionaries with keys 'A' and 'b' respectively encoding matrices and vectors. In the absence of a data.npy file for the indicated problem size, the program will generate a simple Ising-inspired linear system to run. The problem_type config flag determines if this program will solve linear systems 'vqls' or perform a baseline non-VNLS test of the NQS solver 'maxcut'. All results are stored in the '.results/' directory. 

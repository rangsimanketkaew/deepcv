# Call DeepCV's Module

All modules can be called via registered `deepcv` command or `main.py` API script. 
If you use conda, please ensure to activate to the environment, where you installed dependencies, before calling DeepCV.

```sh
$ deepcv

------------------------------------------------
DeepCV : Deep Learning for Collective Variables
-------------------------------------------------
version 2.0 : August 2025
University of Zurich, Switzerland
Manual: https://lubergroup.pages.uzh.ch/deepcv
Code: https://gitlab.uzh.ch/lubergroup/deepcv

Module         Description
-------------  ----------------------------------------------
calc_rep       Calculate molecular representations
gen_input      Input generator
daenn          Deep autoencoder neural network
deepcv2plumed  Create PLUMED input file
single_train   Single-data fully-connected neural network
multi_train    Multi-data fully-connected neural network
gan_train      Training generative adversarial network (GAN)
gan_predict    Generating samples using trained GAN
analyze_FES    FES validation
analyze_model  DAENN model analysis and parameters extraction
explore_abi    Calculate exploration ability
stack_array    Combine arrays of the same dimension
xyz2arr        Convert xyz to NumPy array

For more detail, please review 'README' in the repository.
```

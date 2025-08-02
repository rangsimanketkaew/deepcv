# Call DeepCV's Module

All modules can be called via `main.py` API script or via its entry point (i.e. `deepcv_daenn`). If you use conda, please ensure to activate to the environment, where you installed dependencies, before calling DeepCV.

```sh
$ python deepcv/src/main.py

------------------------------------------------
DeepCV : Deep Learning for Collective Variables
-------------------------------------------------
version 2.0 : August 2025
University of Zurich, Switzerland
https://gitlab.uzh.ch/lubergroup/deepcv

Module         Description
-------------  ----------------------------------------------
calc_rep       Calculate molecular representation
gen_input      Neural network input generator
single_train   Single-data fully-connected neural network
multi_train    Multi-data fully-connected neural network
daenn          Deep autoencoder neural network
gan_train      Training generative adversarial network (GAN)
gan_predict    Generating samples using trained GAN
deepcv2plumed  Create PLUMED input file
analyze_FES    FES validation
analyze_model  DAENN model analysis and parameters extraction
explor_abi     Calculate exploration ability

For more detail, please review 'README' in the repository.
```

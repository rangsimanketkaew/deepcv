# Call DeepCV's module

All modules can be called via `main.py` API script. Please make sure to activate to the (conda) environment, where you installed dependencies packages for DeepCV, before calling DeepCV.

```sh
$ cd deepcv/src/
$ python main.py

------------------------------------------------
DeepCV : Deep Learning for Collective Variables
-------------------------------------------------
version 1.0 : February 2022
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

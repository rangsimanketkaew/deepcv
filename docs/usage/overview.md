# Understanding how DeepCV works

To help the users easily understand the whole procedure of DAENN we provide an end-to-end workshop pipeline of DAENN below. The process starts from the step where we use unbiased molecular dynamics to generate a trajectory of reactant conformers. Then it proceeds to feature calculation, model training, and CV generation, respectively. The trained CVs are then used as input for metadynamics simulation (or any other enhanced sampling methods) to calculate free energy surfaces (FESs) of a studied chemical system.

<figure markdown>
  ![Image title](../img/daenn-workflow.png){ width="1200" }
  <figcaption>A complete workflow of DAENN and FES reconstruction.</figcaption>
</figure>

## Call DeepCV's module

All DeepCV's modules can be called via `main.py` API script (Don't forget to activate to the (conda) environment, where you installed dependencies packages for DeepCV).

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

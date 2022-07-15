# Training a DAENN model

Note: the current version of DeepCV accepts only datasets that are in NumPy's compressed file formats (.npz).

## Call DeepCV's module

All DeepCV's modules can be called via `main.py` API script.

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

## Train model

Execute the `main.py` with argument `daenn`, like below. Then it will start to train the model.
The training time depends on the size of the dataset and networks, the number of epochs, etc.

```sh
$ python main.py daenn -i input/input_ae_DA.json

============================== Program started ==============================
Project: Demo
=== Shape of dataset before splitting ===
>>> 1. Dataset: (99000, 15)
>>> 2. Dataset: (99000, 14)
>>> 3. Dataset: (99000, 13)
=== Shape of dataset after splitting ===
>>> 1. Train: (79200, 15) & Test: (19800, 15)
>>> 2. Train: (79200, 14) & Test: (19800, 14)
>>> 3. Train: (79200, 13) & Test: (19800, 13)
...
...
...
1/1 [==============================] - ETA: 0s - loss: 0.1283 - out_1_loss: 0.1268 - out_2_loss: 0.0154 - out_1_mse: 4.0202e-04 - out_2_mse: 61/1█ ETA: 00:00s - loss: 0.1283 - out_1_loss: 0.1268 - out_2_loss: 0.0154 - out_1_mse: 0.0004 - out_2_mse: 67.5435 - val_loss: 0.1176 - val_ou1/1 [==============================] - 0s 21ms/step - loss: 0.1283 - out_1_loss: 0.1268 - out_2_loss: 0.0154 - out_1_mse: 4.0202e-04 - out_2_mse: 67.5435 - val_loss: 0.1176 - val_out_1_loss: 0.1160 - val_out_2_loss: 0.0156 - val_out_1_mse: 0.0013 - val_out_2_mse: 66.8437
Training: 100%|█████████████████████████████████████████████████████████████████████████████████████████ 1000/1000 ETA: 00:00s,  42.66epochs/sDeepCV:INFO >>> Congrats! Training model is completed.
DeepCV:INFO >>> Model has been saved to /home/rketka/github/deepcv/output/model.h5
DeepCV:INFO >>> Weights of model have been saved to /home/rketka/github/deepcv/output/model_weights.h5
DeepCV:INFO >>> Weights of model have been saved to /home/rketka/github/deepcv/output/model_weights.npz
DeepCV:INFO >>> Biases of model have been saved to /home/rketka/github/deepcv/output/model_biases.npz
DeepCV:INFO >>> Directed graphs of all model have been saved to /mnt/c/Users/Nutt/Desktop/daenn-test
DeepCV:INFO >>> Loss history plot has been saved to /home/rketka/github/deepcv/output/loss.png
DeepCV:INFO >>> Metric accuracy history plot has been saved to /home/rketka/github/deepcv/output/metrics.png
============================== DONE ==============================
```

Once you see the line "=== DONE===" the training is completed.
You can then use the output saved in the folder you specified in the input file for further work, i.e., generating CVs.

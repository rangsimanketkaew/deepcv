# Autoencoder-based collective variable for Diels-Alder reaction

This tutorial shows how to prepare an input file for training an autoencoder neural network.
Input file defines general settings, necessary hyper-parameters, outputs, etc.

## Step 1: Generate input files (dataset)
DeepCV now accepts only NumPy's compress file formats (.npz) as dataset (train and test sets). 
```sh
$ python -m src.helpers.extract_zmat_features -i dataset/DA/R/DUMP*.npz

Input 1: dataset/DA/R/DUMP_DNN_DA_MetaD_XTB_NVT_300K-pos-1-partial-xyz-array-10.npz
Input 2: dataset/DA/R/DUMP_DNN_DA_MetaD_XTB_NVT_300K-pos-1-partial-xyz-array-11.npz
Input 3: dataset/DA/R/DUMP_DNN_DA_MetaD_XTB_NVT_300K-pos-1-partial-xyz-array-12.npz
Input 4: dataset/DA/R/DUMP_DNN_DA_MetaD_XTB_NVT_300K-pos-1-partial-xyz-array-13.npz
Input 5: dataset/DA/R/DUMP_DNN_DA_MetaD_XTB_NVT_300K-pos-1-partial-xyz-array-14.npz
Input 6: dataset/DA/R/DUMP_DNN_DA_MetaD_XTB_NVT_300K-pos-1-partial-xyz-array-15.npz
Input 7: dataset/DA/R/DUMP_DNN_DA_MetaD_XTB_NVT_300K-pos-1-partial-xyz-array-16.npz
Input 8: dataset/DA/R/DUMP_DNN_DA_MetaD_XTB_NVT_300K-pos-1-partial-xyz-array-17.npz
Input 9: dataset/DA/R/DUMP_DNN_DA_MetaD_XTB_NVT_300K-pos-1-partial-xyz-array-18.npz
Input 10: dataset/DA/R/DUMP_DNN_DA_MetaD_XTB_NVT_300K-pos-1-partial-xyz-array-19.npz
Input 11: dataset/DA/R/DUMP_DNN_DA_MetaD_XTB_NVT_300K-pos-1-partial-xyz-array-20.npz
Shape of NumPy array (after stacking): (99000, 16, 3)
Calculating distance, angle, torsion ...
---Done!---
```

## Step 2: Prepare input file
DeepCV's input file needed to be prepared in a JSON file format (dictionary-like). The following example show a complete input file for training the model using an autoencoder with 3 hidden layers. The first 3 hidden layers contain 2 encoded layers and 1 latent encoded layer (middle layer) and the last 2 hidden layers are decoded layers for reconstruction. On the other hand, the size of two hidden layers that opposite each other, e.g. input and output layers, 1st and 5th hidden layers, must be the same.
```JSON
{
  "_comment": "Configuration input file",
  "project": {
    "_comment": "Type of neural network to train",
    "name": "Demo",
    "neural_network": "ae"
  },
  "dataset": {
    "_comment": "Dataset manipulation: dataset splitting and normalization",
    "split": false,
    "split_ratio": 0.8,
    "shuffle": true,
    "normalize_scale": 0.0,
    "max_scale": 2
  },
  "model": {
    "_comment": "Define the optimizer, loss function, number of epochs, and batch size",
    "optimizer": "adam",
    "loss": "mean_absolute_error",
    "num_epoch": 2,
    "batch_size": 60
  },
  "network": {
    "_comment": "Number of neurons and activation function for each hidden layer",
    "hidden_layers": 5,
    "units_1": 32,
    "units_2": 8,
    "units_3": 2,
    "units_4": 8,
    "units_5": 32,
    "func_1": "tanh",
    "func_2": "tanh",
    "func_3": "tanh",
    "func_4": "tanh",
    "func_5": "tanh"
  },
  "performance": {
    "_comment": "Setting for training performance",
    "enable_gpu": true,
    "gpus": 1
  },
  "settings": {
    "_comment": "User-defined settings",
    "verbosity": 1,
    "show_summary": true,
    "save_tb": false,
    "save_model": true,
    "save_weights": true,
    "save_weights_npz": true,
    "save_biases_npz": true,
    "save_graph": true,
    "show_loss": true
  },
  "output": {
    "_comment": "Set name of output files",
    "out_dir": "output",
    "out_model": "model.h5",
    "out_weights": "model_weights.h5",
    "out_weights_npz": "model_weights.npz",
    "out_biases_npz": "model_biases.npz",
    "loss_plot": "loss.png"
  }
}
```

## Step 3: Train model
Execute the `ae_train.py` source using `-m`, like below. Then it will start to train the model. The training time depends the size of dataset and networks, the number of epochs, etc.
```sh
$ python -m src.modules.ae_train \
    -d dataset/DA/DNN_DA_R_distance.npz dataset/DA/DNN_DA_R_angle.npz dataset/DA/DNN_DA_R_torsion.npz \
    -k dist angle torsion \
    -i input/input_ae_DA.json

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
Epoch 1/200
1310/1320 [============================>.] - ETA: 0s - loss: 0.1711 - tf_op_layer_split_loss: 0.0480 - tf_op_layer_split_1_loss: 0.0373 - tf_op_layer_split_2_loss: 0.0857
1320/1320 [==============================] - 3s 2ms/step - loss: 0.1708 - tf_op_layer_split_loss: 0.0479 - tf_op_layer_split_1_loss: 0.0373 - tf_op_layer_split_2_loss: 0.0856 - val_loss: 0.1334 - val_tf_op_layer_split_loss: 0.0287 - val_tf_op_layer_split_1_loss: 0.0317 - val_tf_op_layer_split_2_loss: 0.0730
Epoch 2/200
1320/1320 [==============================] - 3s 2ms/step - loss: 0.1243 - tf_op_layer_split_loss: 0.0262 - tf_op_layer_split_1_loss: 0.0302 - tf_op_layer_split_2_loss: 0.0679 - val_loss: 0.1066 - val_tf_op_layer_split_loss: 0.0229 - val_tf_op_layer_split_1_loss: 0.0283 - val_tf_op_layer_split_2_loss: 0.0554
Epoch 3/200
1320/1320 [==============================] - 3s 2ms/step - loss: 0.1014 - tf_op_layer_split_loss: 0.0215 - tf_op_layer_split_1_loss: 0.0277 - tf_op_layer_split_2_loss: 0.0522 - val_loss: 0.0967 - val_tf_op_layer_split_loss: 0.0219 - val_tf_op_layer_split_1_loss: 0.0273 - val_tf_op_layer_split_2_loss: 0.0475
...
...
...
>>> Model has been saved to /home/rketka/github/deepcv/output/model.h5
>>> Weights of model have been saved to /home/rketka/github/deepcv/output/model_weights.h5
>>> Weights of model have been saved to /home/rketka/github/deepcv/output/model_weights.npz
>>> Biases of model have been saved to /home/rketka/github/deepcv/output/model_biases.npz
>>> Directed graphs of all model have been saved to /home/rketka/github/deepcv/output
>>> Loss history plot has been saved to Demo_loss_vs_epoch.png
============================== DONE ==============================
```

## Step 4: Create PLUMED input file
Once the training is completed, you can use `deecv2plumed` script to generate the PLUMED input file. It takes the same input as you used for `ae_train.py`. It will automatically extract the weight and bias from model and print out the file.
```sh
$ python -m src.scripts.deepcv2plumed -i input/input_ae_DA.json -n 16 -o plumed-NN.dat

>>> Plumed data have been successfully written to 'plumed-NN.dat'
>>> In order to run metadynamics using CP2K & PLUMED, specify the following input deck in CP2K input:

# Import PLUMED input file
&MOTION
    &FREE_ENERGY
        &METADYN
            USE_PLUMED .TRUE.
            PLUMED_INPUT_FILE plumed-NN.dat
        &END METADYN
    &END FREE_ENERGY
%END MOTION
```

## Step 5: Test plumed input file
This step is to check if a generated PLUMED input file works or not. You can use plumed driver to run a trial test on one-frame simple Diels-Alder trajectory.
```sh
$ plumed driver --ixyz reactant_DA_water_100atoms.xyz --plumed plumed-NN.dat
$ tree

.
├── bias.log
├── COLVAR.log
├── HILLS
├── input_plumed.log
├── layer1.log
├── layer2.log
├── layer3.log
├── plumed-NN.dat
└── reactant_DA_water_100atoms.xyz
```

## Step 6: Prepare input files for performing enhanced sampling simulation
Prepare all necessary files and run metadynamics simulation using CP2K.
```sh
$ tree

.
├── dftd3.dat
├── MetaD.inp
├── plumed-NN.dat
├── reactant_DA_water_100atoms.xyz
├── run_script.sh
└── xTB_parameters
```

## Step 7: Submit job
Submit the job on Pit Daint.
```sh
$ sbatch -J MetaD.inp run_script.sh
```

## Author
Rangsiman Ketkaew
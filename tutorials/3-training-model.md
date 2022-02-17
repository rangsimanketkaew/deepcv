# Training autoencoder neural network <!-- omit in toc -->

- [Step 1: Prepare input file for Diels-Alder reaction](#step-1-prepare-input-file-for-diels-alder-reaction)
- [Step 2: Train model](#step-2-train-model)

Note: DeepCV now accepts datasets only in NumPy's compressed file formats (.npz).

## Step 1: Prepare input file for Diels-Alder reaction

DeepCV's input file needed to be prepared in a JSON file format (dictionary-like).
The following example shows an input file for training a model using DAENN with five hidden layers.
The first three hidden layers contain two encoded layers and one latent encoded layer (middle layer).
The rest layers are two decoded layers for reconstruction.
On the other hand, the size of two hidden layers that are opposite of each other, e.g., input and output layers, 1st and 5th hidden layers, must be the same.

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
    "func_1": "relu",
    "func_2": "relu",
    "func_3": "relu",
    "func_4": "relu",
    "func_5": "relu"
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
    "save_loss": true,
    "show_loss": false
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

## Step 2: Train model

Execute the `ae_train.py` source using `-m`, like below. Then it will start to train the model. 
The training time depends on the size of the dataset and networks, the number of epochs, etc.

```sh
$ python ae_train \
    -d dataset/traj_zmat_distance.npz dataset/traj_zmat_angle.npz dataset/traj_zmat_torsion.npz \
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
>>> Loss history plot has been saved to /home/rketka/github/deepcv/output/loss.png
============================== DONE ==============================
```

Once you see the line "=== DONE===" the training is completed.
You can then use the output saved in the folder you specified in the input file for further work, i.e., generating CVs.

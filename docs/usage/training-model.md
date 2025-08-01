# Train DAENN model with DeepCV

Training a DAENN CVs model with DeepCV is very easy. 
All parameters needed for model training are defined in one input file that you just setup in the previous step.
On this page, you will learn how to use the `daenn` code script to train a CV of the Diels-Alder reaction of 
ethene and 1,3-butadiene. The reference paper is 
[https://pubs.acs.org/doi/full/10.1021/acs.jpclett.1c04004](https://pubs.acs.org/doi/full/10.1021/acs.jpclett.1c04004).

> Note that the current version of DeepCV accepts only datasets that are in NumPy's compressed file formats (.npz).

## Train model

Execute the `main.py` script with argument `daenn` and followed by `--input` or `-i` flag with the path to input file 
(e.g. [input_ae_DA.json](https://gitlab.uzh.ch/lubergroup/deepcv/-/blob/master/input/input_ae_DA.json)). 
The training time depends on the size of the dataset, the complexity of a neural net, the number of epochs, etc.
The following example is the training on dataset of 99,000 configurations of ethene and 1,3-butadiene.

```sh
$ python deepcv/src/main.py daenn --input input_ae_DA.json

DeepCV:INFO >>> ============================== Program started ==============================
DeepCV:INFO >>> Project: Demo
DeepCV:INFO >>> Date 30/06/2025 at 11:38:36
DeepCV:WARNING >>> No npz keys specified, the first key found in array.files is automatically used.
DeepCV:INFO >>> === Shape of dataset before splitting ===
DeepCV:INFO >>> 1. Dataset: (99000, 5)
DeepCV:INFO >>> 2. Dataset: (99000, 4)
DeepCV:INFO >>> 3. Dataset: (99000, 3)
DeepCV:INFO >>> 4. Dataset: (99000, 6)
DeepCV:INFO >>> 5. Dataset: (99000, 5)
DeepCV:INFO >>> === Shape of dataset after splitting ===
DeepCV:INFO >>> 1. Train: (79200, 5) & Test: (19800, 5)
DeepCV:INFO >>> 2. Train: (79200, 4) & Test: (19800, 4)
DeepCV:INFO >>> 3. Train: (79200, 3) & Test: (19800, 3)
DeepCV:INFO >>> 4. Train: (79200, 6) & Test: (19800, 6)
DeepCV:INFO >>> 5. Train: (79200, 5) & Test: (19800, 5)
Model: "daenn"
...
...
...
1/1 [==============================] - ETA: 0s - loss: 0.1283 - out_1_loss: 0.1268 - out_2_loss: 0.0154 - out_1_mse: 4.0202e-04 - out_2_mse: 61/1█ ETA: 00:00s - loss: 0.1283 - out_1_loss: 0.1268 - out_2_loss: 0.0154 - out_1_mse: 0.0004 - out_2_mse: 67.5435 - val_loss: 0.1176 - val_ou1/1 [==============================] - 0s 21ms/step - loss: 0.1283 - out_1_loss: 0.1268 - out_2_loss: 0.0154 - out_1_mse: 4.0202e-04 - out_2_mse: 67.5435 - val_loss: 0.1176 - val_out_1_loss: 0.1160 - val_out_2_loss: 0.0156 - val_out_1_mse: 0.0013 - val_out_2_mse: 66.8437
DeepCV:INFO >>> Congrats! Training model is completed.
DeepCV:INFO >>> Congrats! Training model is completed.
DeepCV:INFO >>> Model has been saved to /home/rketka/github/deepcv/output/model.h5
DeepCV:INFO >>> Weights of model have been saved to /home/rketka/github/deepcv/output/model_weights.h5
DeepCV:INFO >>> Weights of model have been saved to /home/rketka/github/deepcv/output/model_weights.npz
DeepCV:INFO >>> Biases of model have been saved to /home/rketka/github/deepcv/output/model_biases.npz
DeepCV:INFO >>> Directed graphs of all model have been saved to /mnt/c/Users/Nutt/Desktop/daenn-test
DeepCV:INFO >>> Loss history plot has been saved to /home/rketka/github/deepcv/output/loss.png
DeepCV:INFO >>> Metric accuracy history plot has been saved to /home/rketka/github/deepcv/output/metrics.png
============================== DONE ==============================
```

Once you see the line `=== DONE===`, the training is successfully completed.
You can then use the files saved in the output folder you specified in the input file for further post-training works, 
e.g., use the trained model to generate CVs.

## Outputs

DeepCV saves autoencoder, encoder, and decoder models separately in the respective output directory.

Example of a list of outputs generated and saved by DeepCV (sorted by name).

```sh
assets
latent_space_100_epochs.png
latent_space_120_epochs.png
latent_space_140_epochs.png
latent_space_160_epochs.png
latent_space_180_epochs.png
latent_space_200_epochs.png
latent_space_20_epochs.png
latent_space_40_epochs.png
latent_space_60_epochs.png
latent_space_80_epochs.png
loss.png
metrics.png
model_biases.npz
model.h5
model.png
model_summary.txt
model_weights.h5
model_weights.npz
saved_model.pb
variables
```

# Train DAENN model with DeepCV

Training a DAENN/DeepCV model is very easy. All parameters needed for model training are defined in one input file that you just setup in the previous step.
On this page, you will learn how to use the `daenn` module to train a CV of the Diels-Alder reaction of ethene and 1,3-butadiene. 
The reference paper is [https://pubs.acs.org/doi/full/10.1021/acs.jpclett.1c04004](https://pubs.acs.org/doi/full/10.1021/acs.jpclett.1c04004).

> Note that the current version of DeepCV accepts only dataset file that is in NumPy's compressed file formats (`.npz`).

## Train model

Call `deepcv daenn` or execute the `main.py` script with argument `daenn`, and followed by `--input` or `-i` flag with the path to the input file 
(e.g. [input_ae_DA.json](https://gitlab.uzh.ch/lubergroup/deepcv/-/blob/master/input/input_ae_DA.json)). 
The training time depends on the size of the dataset, the complexity of a neural net, the number of epochs, etc.
The following example is the training on dataset of 99,000 configurations of ethene and 1,3-butadiene.

```sh
$ deepcv daenn --input input_ae_DA.json

DeepCV:INFO >>> ============================== Program started ==============================
DeepCV:INFO >>> Project: Demo
DeepCV:INFO >>> Date 16/07/2025 at 10:05:24
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
Epoch 1/1000
243/291 ━━━━━━━━━━━━━━━━━━━━ 0s 835us/step - loss: 14.9380 - out_1_loss: 14.9332 - out_1_mse: 4.1317 - out_2_loss: 0.0482 - out_2_mse: 8.1241
DeepCV:INFO >>> Save model at epoch no. 0
500/500 ━━━━━━━━━━━━━━━━━━━━ 0s 277us/step
291/291 ━━━━━━━━━━━━━━━━━━━━ 2s 3ms/step - loss: 14.4393 - out_1_loss: 14.4345 - out_1_mse: 3.9086 - out_2_loss: 0.0470 - out_2_mse: 8.7680 - val_loss: 6.9172 - val_out_1_loss: 6.9104 - val_out_1_mse: 0.8723 - val_out_2_loss: 0.0275 - val_out_2_mse: 24.0786
...
...
DeepCV:INFO >>> Congrats! Training model is completed.
DeepCV:INFO >>> Model has been saved to /home/rketka/github/deepcv/output/model.h5
DeepCV:INFO >>> Weights of model have been saved to /home/rketka/github/deepcv/output/model_weights.h5
DeepCV:INFO >>> Weights of model have been saved to /home/rketka/github/deepcv/output/model_weights.npz
DeepCV:INFO >>> Biases of model have been saved to /home/rketka/github/deepcv/output/model_biases.npz
DeepCV:INFO >>> Directed graphs of all model have been saved to /mnt/c/Users/Nutt/Desktop/daenn-test
DeepCV:INFO >>> Loss history plot has been saved to /home/rketka/github/deepcv/output/loss.png
DeepCV:INFO >>> Metric accuracy history plot has been saved to /home/rketka/github/deepcv/output/metrics.png
DeepCV:INFO >>> ============================== DONE ==============================
```

Once you see the line `=== DONE===`, the training is successfully completed.
You can then use the files saved in the output folder (e.g. trained weights and biases) for generating CVs.

## Outputs

DeepCV saves autoencoder, encoder, and decoder models, and [TensorBoard](https://www.tensorflow.org/tensorboard) logs separately in the respective output directory.

Example of contents of output directory in a tree-like format generated and saved by DeepCV.

```sh
├── autoencoder
│   ├── daenn.png
│   ├── latent_space_0_epochs.png
│   ├── latent_space_10_epochs.png
│   ├── latent_space_20_epochs.png
│   ├── latent_space_30_epochs.png
│   ├── latent_space_40_epochs.png
│   ├── loss.png
│   ├── metrics.png
│   ├── model_biases.npz
│   ├── model_summary.txt
│   ├── model_weights.npz
│   └── model.weights.h5
├── decoder
│   ├── decoder.png
│   └── model_summary.txt
├── encoder
│   ├── encoder.png
│   └── model_summary.txt
└── tb
    ├── train
    │   ├── events.out.tfevents.1754068135.linux.local.51878.0.v2
    │   ├── events.out.tfevents.1754068547.linux.local.52214.0.v2
    │   ├── events.out.tfevents.1754132587.linux.local.57368.0.v2
    │   ├── events.out.tfevents.1754132721.linux.local.57483.0.v2
    │   └── events.out.tfevents.1754133579.linux.local.58283.0.v2
    └── validation
        ├── events.out.tfevents.1754068136.linux.local.51878.1.v2
        ├── events.out.tfevents.1754068548.linux.local.52214.1.v2
        ├── events.out.tfevents.1754132588.linux.local.57368.1.v2
        ├── events.out.tfevents.1754132722.linux.local.57483.1.v2
        └── events.out.tfevents.1754133580.linux.local.58283.1.v2
```

To visualize training log with TensorBoard, run this command

```sh
tensorboard --logdir output/tb
```

and open the local host URL, e.g. `http://localhost:6006/`, in your web browser.

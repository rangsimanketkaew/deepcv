#!/usr/bin/env python3

"""
Deep Learning for Collective Variables (DeepCV)
https://gitlab.uzh.ch/LuberGroup/deepcv

Info:
16/08/2022 : Rangsiman Ketkaew
"""

"Visualize latent space of DAENN"

import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import Model


def explained_variance(projections, k=2):
    """Calculate explained variance

    Args:
        projections (array): Projection data
        k (int): Number of neurons. Defaults to 2.

    Returns:
        float: Variance
    """
    n_sample = projections.shape[0]
    var = np.array([(projections[:, i] ** 2).sum() / (n_sample - 1) for i in range(k)]).round(2)
    return var


def encode_fig(i, model_inp, model_out, x_train, y_train, out_name="dense_2", folder="."):
    """Plot encoded data (latent space) from the encoder model

    Args:
        i (int): i-th epoch
        model_inp (tensor): TensorFlow's input layer
        model_out (tensor): TensorFlow's output layer
        x_train (array): Input
        y_train (array): True values
        out_name (str, optional): Name of the output layer of the encoder. Defaults to "dense_2".
        folder (str, optional): Nameo the output folder for saving images. Defaults to ".".
    """
    enc = Model(model_inp, model_out)
    Z_enc = enc.predict(x_train)
    ev = explained_variance(Z_enc)
    print("------------------------------------")
    print(f"At epoch {i} explained variance: {ev}")
    print("------------------------------------")
    plt.title(f"Encoded data visualization: EV = {ev}")
    plt.scatter(
        Z_enc[:, 0], Z_enc[:, 1], c=Z_enc[:, 0], s=8, cmap="tab10"
    )
    plt.savefig(folder + "/" + str(i) + ".png")
    plt.clf()


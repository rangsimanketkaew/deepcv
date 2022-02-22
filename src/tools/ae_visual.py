#!/usr/bin/env python3

"""
Deep Learning for Collective Variables (DeepCV)
https://gitlab.uzh.ch/LuberGroup/deepcv

Info:
16/08/2022 : Rangsiman Ketkaew
"""

"visualize DAENN feature representation"

import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import Model


def explained_variance(projections):
    """Calculate explained variance

    Args:
        projections (array): Projection data

    Returns:
        float: Variance
    """
    var = np.array([(projections[:, i] ** 2).sum() / (n_sample - 1) for i in range(k)]).round(2)
    return var


def encode_fig(i, model, x_train, y_train, out_name="dense_2", folder="."):
    """Plot encoded data (representation) from the encoder model

    Args:
        i (int): i-th epoch
        model (tensor): TensorFlow tensor
        x_train (array): Input
        y_train (array): True values
        out_name (str, optional): Name of the output layer of the encoder. Defaults to "dense_2".
        folder (str, optional): Nameo the output folder for saving images. Defaults to ".".
    """
    encoder = Model(model.input, ae.get_layer(out_name).output)
    Zenc = encoder.predict(x_train)
    print("At epoch ", i, ", explained variance: ", explained_variance(Zenc))
    plt.title("Encoded data visualization")
    plt.scatter(Zenc[:, 0], Zenc[:, 1], c=y_train[:], s=8, cmap="tab10")
    plt.savefig(folder + "/" + str(i) + ".png")
    plt.clf()


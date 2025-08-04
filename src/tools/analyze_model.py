#!/usr/bin/env python3

"""
Deep Learning for Collective Variables (DeepCV)
https://gitlab.uzh.ch/LuberGroup/deepcv

Info:
28/11/2020 : Rangsiman Ketkaew
"""

"Check, load, and show trained Model's outputs."

import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras import models
import matplotlib.pyplot as plt


# Loss for custom_object
@tf.function
def GRMSE(y_true, y_pred):
    """Calculate Geometric root mean-squared error (GRMSE)"""
    N = y_pred.shape[1]
    square_err = tf.math.square(tf.math.subtract(y_true, y_pred))
    mult_sum = tf.einsum("ij,ij->", square_err, square_err)
    return tf.math.pow(mult_sum, 1 / (2 * N))  # 2N^th root


def get_inter_model(f, layer_name="bottleneck", sum_ori=False, sum_new=False):
    """Load Keras model, show the summary of the model, get the intermediate layer (CV)
    and build a new model that will output this layer.

    Args:
        f (str): Relative or absolute path to Keras model (.h5)
        layer_name (str, optional): Name of layer you want to output it. Defaults to "bottleneck".
        sum_ori (bool, optional): Show summary of original model. Defaults to False.
        sum_new (bool, optional): Show summary of new model. Defaults to False.

    Returns:
        new_model (object): New model
    """
    try:
        model = models.load_model(f)
    except ValueError:
        try:
            model = models.load_model(f, custom_objects={"GRMSddE": GRMSE})
            print("\nA model is loaded with custom objects.\n")
        except ValueError:
            print("\nCannot load custom objects, so a model is loaded withtout it.\n")
            model = models.load_model(f, compile=False)

    # Show summary of the original model to check the size of input(s)
    if sum_ori:
        model.summary()

    new_model = models.Model(
        inputs=model.input, outputs=model.get_layer(layer_name).output
    )
    # alternative to the use of index: model.layers[index of layer].output

    # Show summary of the new model
    if sum_new:
        new_model.summary()

    return new_model


def main():
    parser = argparse.ArgumentParser(description="DeepCV: DAENN model analysis")
    parser.add_argument(
        "-m",
        "--model",
        dest="model",
        type=str,
        default=None,
        help="Path to model file (e.g., .h5)",
    )
    parser.add_argument(
        "-w",
        "--weight",
        dest="weight",
        type=str,
        default=None,
        help="Path to weight file (e.g., .npz)",
    )
    parser.add_argument(
        "-b",
        "--bias",
        dest="bias",
        type=str,
        default=None,
        help="Path to bias file (e.g., .npz)",
    )

    args = parser.parse_args()

    ## Model
    if args.model:
        new_model = get_inter_model(args.model, layer_name="dense_2")

    # # Test prediction
    # # Create/define inputs , e.g. random inputs
    # input1 = np.random.randint(2, size=(10000, 15))
    # input2 = np.random.randint(180, size=(10000, 14))
    # input3 = np.random.randint(360, size=(10000, 13))
    # output = new_model.predict([input1, input2, input3])
    # print("Shape of output: " + str(output[0].shape))

    # # Visualize output
    # plt.figure(figsize=(8, 4))
    # plt.title("Autoencoder")
    # plt.scatter(output[:10000, 0], output[:10000, 1], c=output[:5000], s=8, cmap="tab10")
    # plt.tight_layout()
    # plt.show()

    ## Weight
    if args.weight:
        weight = np.load(args.weight)
        print(weight.files)
        for i in weight.files:
            print("Layer: " + str(i) + " => " + str(weight[i].shape))

    ## Bias
    if args.bias:
        bias = np.load(args.bias)
        print(bias.files)


if __name__ == "__main__":
    main()

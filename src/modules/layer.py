"""
Deep Learning for Collective Variables (DeepCV)
https://gitlab.uzh.ch/LuberGroup/deepcv

Info:
06/10/2021 : Rangsiman Ketkaew
"""

"""Custom Keras layer

Documents:
1. https://www.tensorflow.org/guide/keras/custom_layers_and_models#the_add_loss_method
2. https://tensorflow.google.cn/guide/keras/train_and_evaluate#custom_losses
"""

import tensorflow as tf


class LayerWithRMS(tf.keras.layers.Layer):
    """Root mean squared
    """

    def __init__(self):
        super().__init__()

    def call(self, inputs):
        self.add_loss(tf.math.sqrt(tf.math.reduce_sum(tf.math.square(inputs)) / inputs.shape[1]))
        print(inputs.numpy())
        return inputs


class LayerWithNegative(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def call(self, inputs):
        self.add_loss(tf.math.multiply(tf.math.reduce_mean(inputs), -1.0))
        return inputs


class LayerWithRate(tf.keras.layers.Layer):
    def __init__(self, rate=1e-2):
        super().__init__()
        self.rate = rate

    def call(self, inputs):
        self.add_loss(self.rate * tf.math.reduce_sum(inputs))
        return inputs

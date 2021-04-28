"""
Deep Learning for Collective Variables (DeepCV)
https://gitlab.uzh.ch/LuberGroup/deepcv

Info:
28/11/2020 : Rangsiman Ketkaew
"""

"""Custom loss function
"""

import numpy as np
import tensorflow as tf
# from tensorflow.python.ops import math_ops
# from tensorflow.python.framework import ops


@tf.function
def GRMSE(y_true, y_pred):
    """Calculate Geometric root mean-squared error (GRMSE)
    """
    # convert numpy --> tensor
    # y_pred = ops.convert_to_tensor_v2(y_pred)
    # y_true = math_ops.cast(y_true, y_pred.dtype)
    N = y_pred.shape[1]
    square_err = tf.math.square(tf.math.subtract(y_true, y_pred))
    mult_sum = tf.einsum('ij,ij->', square_err, square_err)
    return tf.math.pow(mult_sum, 1/(2*N))  # 2N^th root

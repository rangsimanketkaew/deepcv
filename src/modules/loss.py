"""
Deep Learning for Collective Variables (DeepCV)
https://gitlab.uzh.ch/LuberGroup/deepcv

Info:
28/11/2020 : Rangsiman Ketkaew
"""

"""Custom loss function

Ref: https://doi.org/10.1016/j.trechm.2020.12.004
"""

import tensorflow as tf


class CustomLoss(tf.keras.losses.Loss):
    """Custom loss

    This class wraps __call__ which defines loss calculation. It can be used with model.compile, e.g.,

    custom_loss = CustomLoss(loss_1, loss_2, layer, 0.8)
    model.compile(loss=custom_loss)
    ...

    where loss_1 and loss_2 are loss functions that take y_true and y_pred and calculate its loss value

    Args:
        tf (class): Parent class
    """

    def __init__(self, main_loss, penalty_loss, layer, alpha, name="custom_loss"):
        super().__init__(name=name)
        self.main_loss = main_loss
        self.penalty_loss = penalty_loss
        self.layer = layer
        self.alpha = alpha

    def __call__(self, y_true, y_pred, sample_weight=None):
        return (
            (self.alpha * self.main_loss(y_true, y_pred))
            - ((1 - self.alpha) * self.penalty_loss(y_true, y_pred))
            # + tf.math.reduce_mean(self.layer)
        )


def n2t_std(array):
    """convert numpy --> tensor using standard method

    Args:
        array (array): NumPy array

    Returns:
        tensor: TensorFlow tensor
    """
    return tf.convert_to_tensor(array)


def n2t(y_true, y_pred):
    """convert numpy --> tensor

    Args:
        y_true (array): True values
        y_pred (array): Prediction

    Returns:
        y_true (tensor): True values
        y_pred (tensor): Prediction
    """
    from tensorflow.python.ops import math_ops
    from tensorflow.python.framework import ops

    y_pred = ops.convert_to_tensor_v2(y_pred)
    y_true = math_ops.cast(y_true, y_pred.dtype)
    return y_pred, y_true


@tf.function
def deviate(y_true, y_pred):
    """Deviation of y_pred from y_true

    (y_true - y_pred) / y_true

    Args:
        y_true (tensor, array): True values
        y_pred (tensor, array): Prediction

    Returns:
        tensor: Deviation
    """
    return tf.math.divide(tf.math.subtract(y_true, y_pred), y_true)


@tf.function
def MaxAE(y_true, y_pred):
    """Maximum absolute error

    max{y_true - y_pred}

    Args:
        y_true (tensor, array): True values
        y_pred (tensor, array): Prediction

    Returns:
        tensor: MaxAE
    """
    return tf.math.reduce_max(tf.math.abs(tf.math.subtract(y_true, y_pred)))


@tf.function
def MaxAPE(y_true, y_pred):
    """Maximum absolute percentage error

    max{|(y_true - y_pred)/y_true|*100}

    Args:
        y_true (tensor, array): True values
        y_pred (tensor, array): Prediction

    Returns:
        tensor: MaxAPE
    """
    return tf.math.reduce_max(tf.math.abs(deviate(y_true, y_pred) * 100.0))


@tf.function
def RMSE(y_true, y_pred):
    """Calculate root mean-squared error (RMSE)

                ----------------------------
               /  N                   2
    RMSE = \  /  sum (y_true - y_pred)
            \/   i=1

    Args:
        y_true (tensor, array): True values
        y_pred (tensor, array): Prediction

    Returns:
        tensor: RMSE
    """
    if len(y_pred.shape) == 1:
        N = y_pred.shape[0]
    elif len(y_pred.shape) == 2:
        N = y_pred.shape[1]
    square_err = tf.math.square(tf.math.subtract(y_true, y_pred))
    sum_avg = tf.math.reduce_sum(square_err) / N  # also convert int --> float
    return tf.math.sqrt(sum_avg)


@tf.function
def GMSE(y_true, y_pred):
    """Calculate Geometric mean-squared error (GRMSE)

                ----------------------------
           N   /  N                   2
    GMSE = \  /  mul (y_true - y_pred)
            \/   i=1

    Args:
        y_true (tensor, array): True values
        y_pred (tensor, array): Prediction

    Returns:
        tensor: GRMSE
    """
    # convert numpy --> tensor
    # y_pred = ops.convert_to_tensor_v2(y_pred)
    # y_true = math_ops.cast(y_true, y_pred.dtype)
    N = y_pred.shape[1]
    square_err = tf.math.square(tf.math.subtract(y_true, y_pred))
    mult_sum = tf.einsum("ij,ij->", square_err, square_err)
    return tf.math.pow(mult_sum, 1 / (N))  # N^th root


@tf.function
def GRMSE(y_true, y_pred):
    """Calculate Geometric root mean-squared error (GRMSE)

                 ----------------------------
           2N   /  1   N                   2
    GRMSE = \  /   -  mul (y_true - y_pred)
             \/    N  i=1

    Args:
        y_true (tensor, array): True values
        y_pred (tensor, array): Prediction

    Returns:
        tensor: GRMSE
    """
    # convert numpy --> tensor
    # y_pred = ops.convert_to_tensor_v2(y_pred)
    # y_true = math_ops.cast(y_true, y_pred.dtype)
    N = y_pred.shape[1]
    square_err = tf.math.square(tf.math.subtract(y_true, y_pred))
    mult_sum = tf.einsum("ij,ij->", square_err, square_err)
    return tf.math.pow(mult_sum, 1 / (2 * N))  # 2N^th root


@tf.function
def reduced_AE(y_true, y_pred):
    """Calculate reduced absolute error

                1
          -------------
    mu =   1     1
           -  +  - + ...
          m_1   m_2

    m_i = i-th element of absolute error

    Args:
        y_true (tensor, array): True values
        y_pred (tensor, array): Prediction

    Returns:
        tensor: reduced AE
    """
    ones = tf.ones(y_true.shape)
    reciprocal = tf.math.divide_no_nan(ones, tf.math.abs(tf.math.subtract(y_true, y_pred)))
    return 1 / tf.math.reduce_sum(reciprocal)


@tf.function
def reduced_APE(y_true, y_pred):
    """Calculate reduced absolute percentage error

                1
          -------------
    mu =   1     1
           -  +  - + ...
          m_1   m_2

    m_i = i-th element of absolute percentage error

    Args:
        y_true (tensor, array): True values
        y_pred (tensor, array): Prediction

    Returns:
        tensor: reduced AE
    """
    ones = tf.ones(y_true.shape)
    reciprocal = tf.math.divide_no_nan(ones, tf.math.abs(deviate(y_true, y_pred) * 100.0))
    return 1 / tf.math.reduce_sum(reciprocal)


def test():
    """
    Unit test
    """
    ts_1 = tf.random.uniform((30, 40))
    ts_2 = tf.random.uniform((30, 40))
    ones = tf.ones((30, 40))

    print(deviate(ts_1, ts_2))
    print(MaxAE(ts_1, ts_2))
    print(MaxAPE(ts_1, ts_2))
    print(RMSE(ts_1, ts_2))
    print(GRMSE(ts_1, ts_2))
    print(reduced_AE(ts_1, ts_2))
    print(reduced_APE(ts_1, ts_2))


if __name__ == "__main__":
    test()

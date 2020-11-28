"""
Deep learning-based collective variables (DeepCV)
https://gitlab.uzh.ch/LuberGroup/deepcv

Info:
28/11/2020 : Rangsiman Ketkaew
"""

# Execute this script to check if TensorFlow >= 2.x is using GPU accerelation.

import tensorflow as tf

assert tf.test.is_gpu_available()
assert tf.test.is_built_with_cuda()

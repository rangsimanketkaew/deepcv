#!/usr/bin/env python3

"""
Deep Learning for Collective Variables (DeepCV)
https://gitlab.uzh.ch/LuberGroup/deepcv

Info:
28/11/2020 : Rangsiman Ketkaew
"""

# Execute this script to check if TensorFlow >= 2.x is using GPU accerelation.

import tensorflow as tf

assert tf.test.is_gpu_available(), "TF can't see GPU on this machine."
assert tf.test.is_built_with_cuda(), "TF was not built with CUDA."

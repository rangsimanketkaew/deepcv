"""
Deep Learning for Collective Variables (DeepCV)
https://gitlab.uzh.ch/LuberGroup/deepcv

Info:
28/11/2020 : Rangsiman Ketkaew
"""

import os, sys

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import logging

# printing logging messages to stdout file still does not work!
logging.basicConfig(level=logging.INFO, format="%(name)s:%(levelname)s >>> %(message)s")

from utils import util

util.tf_logging(2, 3)  # warning level

# Bring all modules to the same level as main.py
from . import single_train
from . import multi_train
from . import daenn
from . import gan_train
from . import gan_predict

"""
Deep Learning for Collective Variables (DeepCV)
https://gitlab.uzh.ch/LuberGroup/deepcv

Info:
28/11/2020 : Rangsiman Ketkaew
"""

import sys
import logging

# printing logging messages to stdout file still does not work!
logging.basicConfig(stream=sys.stdout, level=logging.WARN, format="%(name)s - %(levelname)s - %(message)s")

from utils import util

util.tf_logging(2, 3)  # warning level

# Bring all modules to the same level as main.py
from . import single_train
from . import multi_train
from . import daenn
from . import gan_train
from . import gan_predict

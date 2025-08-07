"""
Deep Learning for Collective Variables (DeepCV)
https://gitlab.uzh.ch/LuberGroup/deepcv

Info:
28/11/2020 : Rangsiman Ketkaew
"""

import os
import sys
from pathlib import Path

# Add parent directory to path for relative imports
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

"""Set logging info before importing TensorFlow

 Level | Level for Humans | Level Description
-------|------------------|------------------------------------
    0     | DEBUG            | [Default] Print all messages
    1     | INFO             | Filter out INFO messages
    2     | WARNING          | Filter out INFO & WARNING messages
    3     | ERROR            | Filter out all messages
"""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_CPP_MIN_VLOG_LEVEL"] = "3"

from utils import util
import tensorflow as tf

try:
    assert tf.test.is_built_with_gpu_support()
    assert tf.config.list_physical_devices("GPU")
except AssertionError:
    pass
else:
    util.limit_gpu_growth(tf)

# Bring all modules to the same level as main.py
from . import daenn
from . import gan_train
from . import gan_predict

"""
Deep Learning for Collective Variables (DeepCV)
https://gitlab.uzh.ch/LuberGroup/deepcv

Info:
28/11/2020 : Rangsiman Ketkaew
"""

import json
import os


def load_json(file):
    """
    Extract data from JSON input file
    """
    with open(file, "r") as f:
        data = f.read()
        try:
            json_data = json.loads(data)
        except:
            print("Error: data in input file could not be converted to JSON")
            exit()

    return json_data


def tf_logging(log=2, vlog=2):
    """TensorFlow logging info

     Level | Level for Humans | Level Description
    -------|------------------|------------------------------------
     0     | DEBUG            | [Default] Print all messages
     1     | INFO             | Filter out INFO messages
     2     | WARNING          | Filter out INFO & WARNING messages
     3     | ERROR            | Filter out all messages

    Args:
        level (str, optional): Level of tensorflow information. Defaults to "INFO".
    """

    if log and vlog in [0, 1, 2, 3]:
        # It seems that this works only for Linux
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = f"{log}"
        # os.system(f"export TF_CPP_MIN_LOG_LEVEL={log}")
        os.environ["TF_CPP_MIN_VLOG_LEVEL"] = f"{vlog}"
        # os.system(f"export TF_CPP_MIN_VLOG_LEVEL={vlog}")
    else:
        print("Error: TF logging support only level: 0, 1, 2, 3")


def limit_gpu_growth():
    """Limiting GPU memory growth
    Solution is taken from https://www.tensorflow.org/guide/gpu#limiting_gpu_memory_growth
    """
    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")

    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices("GPU")
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)


def fix_autograph_warning():
    """Decorator to suppress autograph warning"""
    import tensorflow as tf

    tf.autograph.experimental.do_not_convert

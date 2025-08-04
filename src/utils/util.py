"""
Deep Learning for Collective Variables (DeepCV)
https://gitlab.uzh.ch/LuberGroup/deepcv

Info:
28/11/2020 : Rangsiman Ketkaew
"""

import json


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


def limit_gpu_growth(tensorflow):
    """Limiting GPU memory growth
    Solution is taken from https://www.tensorflow.org/guide/gpu#limiting_gpu_memory_growth

    Args:
        tensorflow (module): TensorFlow module
    """

    gpus = tensorflow.config.list_physical_devices("GPU")

    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tensorflow.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tensorflow.config.list_logical_devices("GPU")
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)


def fix_autograph_warning(tensorflow):
    """Decorator to suppress autograph warning

    Args:
        tensorflow (module): TensorFlow module
    """

    tensorflow.autograph.experimental.do_not_convert

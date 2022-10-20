#!/usr/bin/env python3

"""
Deep Learning for Collective Variables (DeepCV)
https://gitlab.uzh.ch/LuberGroup/deepcv

Info:
19/10/2022 : Rangsiman Ketkaew
"""

# Get input and output tensor names of SavedFormat model

import argparse
from tensorflow.python.tools import saved_model_utils

info = "Get input and output tensor names of SavedFormat model"
parser = argparse.ArgumentParser(description=info)
parser.add_argument(
    "--dir",
    dest="output_dir",
    metavar="DIR",
    type=str,
    required=True,
    help="Path to output folder that contains saved model in SavedModel format.",
)
parser.add_argument(
    "--tag_set",
    dest="tag_set",
    metavar="TAG_SET",
    type=str,
    default="serve",
    help="tag-set of graph in SavedModel to show. Defaults to 'serve'.",
)
parser.add_argument(
    "--signature_def",
    dest="signature_def",
    metavar="SIGNATURE_DEF",
    type=str,
    default="serving_default",
    help="key of SignatureDef to display input(s) and output(s). Defaults to 'serving_default'.",
)

arg = parser.parse_args()

meta_graph_def = saved_model_utils.get_meta_graph_def(arg.output_dir, arg.tag_set)

input_signatures = list(meta_graph_def.signature_def[arg.signature_def].inputs.values())
input_names = [signature.name for signature in input_signatures]
print("\nInput tensor names:")
print(input_names)

output_signatures = list(meta_graph_def.signature_def[arg.signature_def].outputs.values())
output_names = [signature.name for signature in output_signatures]
print("\nOutput tensor names:")
print(output_names)

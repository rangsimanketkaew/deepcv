"""
Deep Learning for Collective Variables (DeepCV)
https://gitlab.uzh.ch/LuberGroup/deepcv

Info:
11/06/2021 : Rangsiman Ketkaew
"""

import argparse
import os
import numpy as np

parser = argparse.ArgumentParser(
    description="Read PLUMED data file (.dat, .log, .out, .txt) and save data as NumPy's compressed file (.npz)"
)
parser.add_argument("--input", "-i", metavar="DATA", type=str, required=True, help="PLUMED data file")
parser.add_argument(
    "--key",
    "-k",
    metavar="KEYWORD",
    type=str,
    default="dat",
    help="Keyword name of the data array in .npz file",
)

args = parser.parse_args()

# Read file and remove first column
dat = np.loadtxt(args.input, skiprows=1)[:, 1:]

# save array as npz
out = os.path.splitext(args.input)[0] + "_" + args.key + ".npz"
d = {args.key: dat}
np.savez_compressed(out, **d)

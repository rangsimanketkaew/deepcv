"""
Deep Learning for Collective Variables (DeepCV)
https://gitlab.uzh.ch/LuberGroup/deepcv

Info:
28/11/2020 : Rangsiman Ketkaew
"""

import argparse
from collections import Counter

parser = argparse.ArgumentParser(description="Extract index of all atoms from .xyz file")
parser.add_argument(
    "--xyz", "-i", metavar="XYZ", type=str, required=True, help="Cartesian coordinate file (.xyz)"
)

args = parser.parse_args()

f = open(args.xyz, "r")
label = f.readlines()[2:]
index = [i.split()[0] for i in label]
index_ = list(set(index))
for k in index_:
    a = [str(i + 1) for i, j in enumerate(index) if k == j]
    print(k + "=" + ",".join(a))

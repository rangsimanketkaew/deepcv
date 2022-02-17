#!/usr/bin/env python3

"""
Deep Learning for Collective Variables (DeepCV)
https://gitlab.uzh.ch/LuberGroup/deepcv

Info:
28/11/2020 : Rangsiman Ketkaew
"""

import argparse
import os
import numpy as np
import ase.io


info = "Read cartesian coordinate file (.xyz) and save as NumPy's compress file"
parser = argparse.ArgumentParser(description=info)
parser.add_argument(
    "--xyz", "-i", metavar="XYZ", required=True, type=str, help="Cartesian coordinate in XYZ file format"
)
parser.add_argument(
    "--no-atoms", "-a", metavar="FIRST_N_ATOMS", type=int, help="First N atoms of molecule to be extracted"
)
parser.add_argument(
    "--key",
    "-k",
    metavar="KEYWORD",
    type=str,
    default="coord",
    help="Keyword name of the coordinate array to be saved in .npz file",
)

args = parser.parse_args()

if not args.no_atoms:
    f = open(args.xyz, "r")
    no_atoms = int(f.readline())
    f.close()
else:
    no_atoms = args.no_atoms

generator = ase.io.iread(args.xyz)
# num = sum(1 for atoms in generator)
# print(num)

# 3D empty array
# No. of structures x No. of atoms x 3 (xyz coord)
arr = np.empty((0, no_atoms, 3))
for atoms in generator:
    pos = atoms.positions
    pos = pos[:no_atoms]
    pos = pos.reshape((1, -1, 3))
    arr = np.append(arr, pos, axis=0)

# save array as npz
out = os.path.splitext(args.xyz)[0] + "_" + args.key + ".npz"
d = {args.key: arr}
np.savez_compressed(out, **d)

#!/usr/bin/env python3

#------------------
# Rangsiman Ketkaew
#------------------

import argparse
import os
import numpy as np
import ase.io


info="Read cartesian coordinate (.xyz) file and save as NumPy's compress file"
parser = argparse.ArgumentParser(description=info)
parser.add_argument("--xyz", "-i", metavar="XYZ", required=True, type=str, help="XYZ file")
parser.add_argument("--atoms", "-a", metavar="NUM_ATOMS", required=True, type=int, help="Number of atoms")
parser.add_argument("--key", "-k", metavar="KEYWORD", type=str, default="coord", help="Keyword name of the coordinate array in .npz file")

args = parser.parse_args()

f = args.xyz
no_atoms = args.atoms

generator = ase.io.iread(f)
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

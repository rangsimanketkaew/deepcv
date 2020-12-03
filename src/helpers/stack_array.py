#------------------
# Rangsiman Ketkaew
#------------------

import os
import glob
import argparse
import numpy as np

info="Extract internal coordinate (Z-matrix) and save as NumPy's compressed array format (.npz)."
parser = argparse.ArgumentParser(description=info)
parser.add_argument("--npz", "-i", dest="npz", metavar="FILE.npz", type=str, required=True, nargs="*",
    help="Cartesian coordinate in NumPy's compressed array format.")
parser.add_argument("--key", "-k", dest="key_npz", metavar="KEYWORD", default="coord", type=str,
    help="Keyword name that corresponds to array saved in the npz input file. \
        Note that all npz files must have the same keyword name. Defaults to 'coord'.")

arg = parser.parse_args()

for f in arg.input_npz:
    files += glob.glob(f)
npz = sorted(files)
for i, f in enumerate(npz):
    print(f"Input {i+1}: {f}")
xyz = np.load(npz[0])[arg.key_npz]
for i in range(1, len(npz)):
    xyz = np.vstack((xyz, np.load(npz[i])[arg.key_npz]))
print(f"Shape of NumPy array (after stacking): {xyz.shape}")
out = os.path.splitext(arg.xyz)[0]
np.savez_compressed(f"{out}" + "_stacked.npz", arr=xyz)
#------------------
# Rangsiman Ketkaew
#------------------

import os
import glob
import argparse
import numpy as np

info="Merge multiple NumPy's array compressed file (npz) of internal coordinates (z-matrix) into one npz file."
parser = argparse.ArgumentParser(description=info)
parser.add_argument("--npz", "--input", "-i", dest="npz", metavar="INPUT.npz", type=str, required=True, nargs="*",
    help="Internal coordinate in NumPy's compressed array format (.npz).")
parser.add_argument("--key", "-k", dest="key_npz", metavar="KEYWORD", default="coord", type=str,
    help="Keyword name that corresponds to array saved in the npz input file. \
        Note that all npz files must have the same keyword name. Defaults to 'coord'.")
parser.add_argument("--output", "-o", dest="output", metavar="OUTPUT.npz", type=str,
    help="File name of output of a merged NumPy's compressed array format (.npz).")

arg = parser.parse_args()

files = []
for f in arg.npz:
    files += glob.glob(f)
if len(files) == 0:
    print("Error: .npz input files not found. Please check if you specified the file name correctly.")
    exit()
npz = sorted(files)

# show info
for i, f in enumerate(npz):
    print(f"Input {i+1}: {f}")

# stack arrays
dat = np.load(npz[0])[arg.key_npz]
for i in range(1, len(npz)):
    dat = np.vstack((dat, np.load(npz[i])[arg.key_npz]))

print(f"Shape of output NumPy array (after stacking): {dat.shape}")
if arg.output:
    out = arg.output.split(".")[0] + ".npz"
else:
    out = "data_stacked_merge.npz"
np.savez_compressed(f"{out}", arr=dat)
print("-"*10 + " Done " + "-"*10)

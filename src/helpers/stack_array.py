#!/usr/bin/env python3

"""
Deep Learning for Collective Variables (DeepCV)
https://gitlab.uzh.ch/LuberGroup/deepcv

Info:
28/11/2020 : Rangsiman Ketkaew
"""

"""
Stacking multiple arrays into single array
"""

import sys
import logging
import glob
import argparse
import numpy as np
from tqdm import tqdm

logging = logging.getLogger("DeepCV")


def main():
    info = "Merge multiple NumPy's array compressed file (npz) of internal coordinates (z-matrix) into one npz file."
    parser = argparse.ArgumentParser(description=info)
    parser.add_argument(
        "--npz",
        "--input",
        "-i",
        dest="npz",
        metavar="INPUT.npz",
        type=str,
        required=True,
        nargs="*",
        help="Internal coordinate in NumPy's compressed array format (.npz).",
    )
    parser.add_argument(
        "--key",
        "-k",
        dest="key_npz",
        metavar="KEYWORD",
        type=str,
        help="Keyword name that corresponds to array saved in the npz input file. \
            Note that all npz files must have the same keyword name. Defaults to 'coord'.",
    )
    parser.add_argument(
        "--output",
        "-o",
        dest="output",
        metavar="OUTPUT.npz",
        type=str,
        help="Filename of output of a merged NumPy's compressed array format (.npz).",
    )

    args = parser.parse_args()

    files = []
    for f in args.npz:
        files += glob.glob(f)
    if len(files) == 0:
        logging.error(
            ".npz input files not found. Please check if you specified the filename correctly."
        )
        sys.exit(1)
    npz = sorted(files)

    # check npz key
    dat_load = np.load(npz[0])
    if not args.key_npz:
        if len(dat_load.files) == 1:
            logging.info("npz file contains one array.")
            args.key_npz = dat_load.files[0]
        elif len(dat_load.files) > 1:
            logging.error(
                "npz file contains more than one array, please specify the key name of the array you want to stack."
            )
            sys.exit(1)

    # stack arrays
    dat = dat_load[args.key_npz]
    for i in tqdm(range(1, len(npz))):
        dat = np.vstack((dat, np.load(npz[i])[args.key_npz]))

    logging.info(f"Shape of output NumPy array (after stacking): {dat.shape}")

    if args.output:
        out = args.output.split(".")[0] + ".npz"
    else:
        out = f"stacked_{len(npz)}arr_{args.key_npz}.npz"
    np.savez_compressed(f"{out}", arr=dat)


if __name__ == "__main__":
    main()

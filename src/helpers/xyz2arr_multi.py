#!/usr/bin/env python3

"""
Deep Learning for Collective Variables (DeepCV)
https://gitlab.uzh.ch/LuberGroup/deepcv

Info:
28/11/2020 : Rangsiman Ketkaew
"""

"""
Convert .xyz to .npz (parallel support)
"""

import os
import argparse
import logging
import numpy as np
import ase.io
import time
from multiprocessing import Pool

logging = logging.getLogger("DeepCV")


class AppArr:
    """Append array in parallel process."""

    def __init__(self, arr, generator):
        self.arr = arr
        self.generator = generator

    def gen(self):
        for i, atoms in enumerate(self.generator):
            yield atoms
            pos = atoms.positions
            pos = pos[:no_atoms]
            pos = pos.reshape((1, -1, 3))
            logging.info(f"Done {i+1}")
        # return pos

    def appendArr(self, new_arr):
        self.arr = np.append(self.arr, new_arr, axis=0)


if __name__ == "__main__":
    info = "Read cartesian coordinate file (.xyz) and save as NumPy's compress file"
    parser = argparse.ArgumentParser(description=info)
    parser.add_argument(
        "--xyz",
        "-i",
        metavar="XYZ",
        required=True,
        type=str,
        help="Cartesian coordinate in XYZ file format",
    )
    parser.add_argument(
        "--no-atoms",
        "-a",
        metavar="FIRST_N_ATOMS",
        type=int,
        help="First N atoms of molecule to be extracted",
    )
    parser.add_argument(
        "--no-processors",
        "-np",
        metavar="NUM_PROCESSORS",
        type=int,
        default=1,
        help="Number of parallel processors",
    )

    args = parser.parse_args()

    filename, file_extension = os.path.splitext(args.xyz)

    if not args.no_atoms:
        f = open(args.xyz, "r")
        no_atoms = int(f.readline())
        f.close()
    else:
        no_atoms = args.no_atoms

    generator = ase.io.iread(f)

    p = Pool(args.no_processors)

    # No. of structures x No. of atoms x 3 (xyz coord)
    arr = np.empty((0, no_atoms, 3))

    logging.info("Start to iterate")

    ProgApp = AppArr(arr, generator)
    y_gen = ProgApp.gen()

    start = time.time()
    p.map(ProgApp.appendArr, next(y_gen))
    end = time.time()

    logging.info(f"Time used for extracting file {start - end}")

    # save array as npz
    np.savez_compressed(filename, coords=arr)

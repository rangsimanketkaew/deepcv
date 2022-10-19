#!/usr/bin/env python3

"""
Deep Learning for Collective Variables (DeepCV)
https://gitlab.uzh.ch/LuberGroup/deepcv

Info:
28/11/2020 : Rangsiman Ketkaew
"""

import numpy as np
import ase.io
import time
from multiprocessing import Pool


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
            print(f"done {i+1}")
        # return pos

    def appendArr(self, new_arr):
        self.arr = np.append(self.arr, new_arr, axis=0)


if __name__ == "__main__":
    NUM_CORES = 8
    p = Pool(NUM_CORES)

    folder = "../DNN-DA-NVT-XTB-MetaD-SPRINT-all"
    name = "DUMP_DNN_DA_MetaD_XTB_NVT_300K-pos-1.xyz"
    f = folder + "/" + name
    no_atoms = 16

    generator = ase.io.iread(f)

    # No. of structures x No. of atoms x 3 (xyz coord)
    arr = np.empty((0, no_atoms, 3))

    print("Start to iterate")

    ProgApp = AppArr(arr, generator)
    y_gen = ProgApp.gen()

    start = time.time()
    p.map(ProgApp.appendArr, next(y_gen))
    end = time.time()

    print(f"Time used for extracting file {start - end}")

    # save array as npz
    out = "DUMP_DNN_DA_MetaD_XTB_NVT_300K-pos-1-sig0.1-height4"
    np.savez_compressed(out, coords=arr)

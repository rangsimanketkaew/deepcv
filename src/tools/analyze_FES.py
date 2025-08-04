#!/usr/bin/env python3

"""
Deep Learning for Collective Variables (DeepCV)
https://gitlab.uzh.ch/LuberGroup/deepcv

Info:
11/06/2021 : Rangsiman Ketkaew
"""

"Analyze free energy surface"

import argparse
import logging
import numpy as np

logging = logging.getLogger("DeepCV")


def readhills(files):
    count = 0
    with open(files, "r") as hill:
        for line in hill:
            if line[0] in ["#", "@"]:
                count += 1
            else:
                break
    dat = np.loadtxt(files, skiprows=count)
    return dat


def main():
    parser = argparse.ArgumentParser(description="DeepCV Analysis")
    parser.add_argument(
        "-f", dest="file", action="store", type=str, required=True, help="FES data file"
    )
    parser.add_argument(
        "-p",
        dest="plot_fes",
        action="store_true",
        default=False,
        help="Plot FES using Plumed",
    )
    parser.add_argument(
        "--save-npz",
        dest="save_npz",
        action="store_true",
        default=False,
        help="Save FES data as NumPy's compressed array",
    )
    args = parser.parse_args()

    dat = readhills(args.file)

    if dat.shape[1] == 5:
        logging.info("1D FES")
    elif dat.shape[1] == 7:
        logging.info("2D FES")
    elif dat.shape[1] == 9:
        logging.info("3D FES")

    if args.save_npz:
        np.savez_compressed("FES.dat", dat=dat)


if __name__ == "__main__":
    main()

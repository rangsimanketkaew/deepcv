"""
Deep Learning for Collective Variables (DeepCV)
https://gitlab.uzh.ch/LuberGroup/deepcv

Info:
11/06/2021 : Rangsiman Ketkaew
"""


import argparse
import numpy as np


def readhills(files):
    count = 0
    with open(files, 'r') as hill:
        for line in hill:
            if line[0] in ['#', '@']:
                count += 1
            else:
                break
    dat = np.loadtxt(files, skiprows=count)
    return dat


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DeepCV Analysis')
    parser.add_argument('-f',
                        dest='file',
                        action='store',
                        type=str,
                        required=True,
                        help='FES data file')
    parser.add_argument('-p',
                        dest='plot_fes',
                        action='store_true',
                        default=False,
                        help='Plot FES using Plumed')
    parser.add_argument('--save-npz',
                        dest='save_npz',
                        action='store_true',
                        default=False,
                        help='Save FES data as NumPy\'s compressed array')
    args = parser.parse_args()

    dat = readhills(args.file)

    if dat.shape[1] == 5:
        print("1D FES")
    elif dat.shape[1] == 7:
        print("2D FES")
    elif dat.shape[1] == 9:
        print("3D FES")

    if args.save_npz:
        np.savez_compressed("FES.dat", dat=dat)

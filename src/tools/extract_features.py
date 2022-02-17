#!/usr/bin/env python3

"""
Deep Learning for Collective Variables (DeepCV)
https://gitlab.uzh.ch/LuberGroup/deepcv

Info:
28/11/2020 : Rangsiman Ketkaew
"""

import os
import math
import numpy as np
from calc_rep import distance, angle, angle_sign, dihedral


def DA():
    """Diels-Alder (DA) reaction

    Product of the DA reaction

     C4 --- C3           ----> Dienophile
     /       \
    C5       C2         \
     \       /           |---> Diene
     C6 === C1          /
    """

    index_C = [0, 3, 9, 6, 7, 8]
    dir = os.path.dirname(__file__)
    # Load trajectories
    f = os.path.join(dir, r"DA.npz")
    traj = np.load(f)
    xyz = traj["xyz"]
    # traj = load.files
    print(xyz.shape)
    # print(dir(xyz))

    dist = np.zeros((xyz.shape[0], 2))
    for i in range(xyz.shape[0]):
        dist[i][0] = distance(xyz[i][index_C[0]], xyz[i][index_C[1]])
        dist[i][1] = distance(xyz[i][index_C[1]], xyz[i][index_C[2]])
        dist[i][2] = distance(xyz[i][index_C[2]], xyz[i][index_C[3]])
        dist[i][3] = distance(xyz[i][index_C[3]], xyz[i][index_C[4]])
        dist[i][4] = distance(xyz[i][index_C[4]], xyz[i][index_C[5]])
        dist[i][5] = distance(xyz[i][index_C[5]], xyz[i][index_C[0]])

    ang = np.zeros((xyz.shape[0], 6))
    for i in range(xyz.shape[0]):
        ang[i][0] = angle(xyz[i][index_C[0]], xyz[i][index_C[1]], xyz[i][index_C[2]])
        ang[i][1] = angle(xyz[i][index_C[1]], xyz[i][index_C[2]], xyz[i][index_C[3]])
        ang[i][2] = angle(xyz[i][index_C[2]], xyz[i][index_C[3]], xyz[i][index_C[4]])
        ang[i][3] = angle(xyz[i][index_C[3]], xyz[i][index_C[4]], xyz[i][index_C[5]])
        ang[i][4] = angle(xyz[i][index_C[4]], xyz[i][index_C[5]], xyz[i][index_C[0]])
        ang[i][5] = angle(xyz[i][index_C[5]], xyz[i][index_C[0]], xyz[i][index_C[1]])

    dih = np.zeros((xyz.shape[0], 6))
    for i in range(xyz.shape[0]):
        dih[i][0] = dihedral(xyz[i][index_C[0]], xyz[i][index_C[1]], xyz[i][index_C[2]], xyz[i][index_C[3]])
        dih[i][1] = dihedral(xyz[i][index_C[1]], xyz[i][index_C[2]], xyz[i][index_C[3]], xyz[i][index_C[4]])
        dih[i][2] = dihedral(xyz[i][index_C[2]], xyz[i][index_C[3]], xyz[i][index_C[4]], xyz[i][index_C[5]])
        dih[i][3] = dihedral(xyz[i][index_C[3]], xyz[i][index_C[4]], xyz[i][index_C[5]], xyz[i][index_C[0]])
        dih[i][4] = dihedral(xyz[i][index_C[4]], xyz[i][index_C[5]], xyz[i][index_C[0]], xyz[i][index_C[1]])
        dih[i][5] = dihedral(xyz[i][index_C[5]], xyz[i][index_C[0]], xyz[i][index_C[1]], xyz[i][index_C[2]])

    o = os.path.join(dir, "DA_dist.npz")
    np.savez_compressed(o, dist=dist, ang=ang, dih=dih)


if __name__ == "__main__":
    DA()


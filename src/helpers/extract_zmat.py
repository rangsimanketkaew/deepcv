#!/usr/bin/env python3

#------------------
# Rangsiman Ketkaew
#------------------

import os
import argparse
import ase.io
import numpy as np
from scipy.spatial.distance import cdist
try:
    from .extract_features_DA import angle_sign, dihedral
except ImportError:
    from extract_features_DA import angle_sign, dihedral


def distance(xyz):
    """Calculate distance between atom A and B.
    1
    2 - 1       bond 1
    3 - 1       bond 2
    4 - 1       bond 3
    5 - 2       bond 4
    6 - 3
    7 - 4
    8 - 5
    ...
    N - (N-3)   bond N-1
    """
    no_strct, no_atoms, _ = xyz.shape
    dist = np.zeros((no_strct, no_atoms - 1))
    for i in range(no_strct):
        alldist = cdist(xyz[i], xyz[i])
        dist[i][0] = alldist[0][1]
        dist[i][1] = alldist[0][2]
        for j in range(no_atoms - 3):
            dist[i][j + 2] = alldist[j][j + 3]
    return dist


def angle(xyz):
    """Calculate angle between atom A, B and C.
    1
    2 - 1
    3 - 1 - 2       angle 1
    4 - 1 - 2       angle 2
    5 - 2 - 3       angle 3
    6 - 3 - 4
    7 - 4 - 5
    8 - 5 - 6
    ...
    N - (N-3) - (N-2)   bond N-2
    """
    no_strct, no_atoms, _ = xyz.shape
    angle = np.zeros((no_strct, no_atoms - 2))
    for i in range(no_strct):
        angle[i][0] = angle_sign(xyz[i][2], xyz[i][0], xyz[i][1])
        for j in range(no_atoms - 3):
            angle[i][j + 1] = angle_sign(xyz[i][j + 3], xyz[i][j], xyz[i][j + 2])
    return angle


def torsion(xyz):
    """Calculate dihedral angle (torsion) between atom A, B, C and D.
    1
    2 - 1
    3 - 1 - 2
    4 - 1 - 2 - 3       dih 1
    5 - 2 - 3 - 4       dih 2
    6 - 3 - 4 - 5       dih 3
    7 - 4 - 5 - 6       dih 4
    8 - 5 - 6 - 7       dih 5
    ...
    N - (N-3) - (N-2) - (N-1)      dih N-3
    """
    no_strct, no_atoms, _ = xyz.shape
    dih = np.zeros((no_strct, no_atoms - 3))
    for i in range(no_strct):
        for j in range(no_atoms - 4):
            dih[i][j] = dihedral(xyz[i][j + 3], xyz[i][j], xyz[i][j + 1], xyz[i][j + 2])
    return dih


if __name__ == "__main__":
    info="Extract internal coordinate (Z-matrix) and save as NumPy's compressed array format (.npz)."
    parser = argparse.ArgumentParser(description=info)
    parser.add_argument("--xyz", "-i", dest="xyz", metavar="FILE.xyz", type=str, required=True,
        help="Cartesian coordinate in NumPy's compressed array format.")
    parser.add_argument("--atom-index", "-a", dest="index_list", metavar="ATOM_INDEX", type=int, nargs="+", default=None,
        help="List of atomic index that will be taken. (0-based array index)")

    arg = parser.parse_args()

    f = open(arg.xyz, "r")
    no_atoms = int(f.readline())
    f.close()
    generator = ase.io.iread(arg.xyz)
    # 3D empty array
    # No. of structures x No. of atoms x 3 (xyz coord)
    xyz = np.empty((0, no_atoms, 3))
    for atoms in generator:
        pos = atoms.positions
        pos = pos[:no_atoms]
        pos = pos.reshape((1, -1, 3))
        xyz = np.append(xyz, pos, axis=0)

    print(f"Shape of NumPy array: {xyz.shape}")
    index = arg.index_list
    if index:
        xyz = xyz[:,index]
        print(f"List of atom index: {index}")
        print(f"Shape of NumPy array with only specified atom index: {xyz.shape}")
    
    out = os.path.splitext(arg.xyz)[0]
    print("Calculating distance ...")
    dat = distance(xyz)
    np.savez_compressed(f"{out}" + "_zmat_distance.npz", dist=dat)
    print("Calculating angle ...")
    dat = angle(xyz)
    np.savez_compressed(f"{out}" + "_zmat_angle.npz", angle=dat)
    print("Calculating torsion ...")
    dat = torsion(xyz)
    np.savez_compressed(f"{out}" + "_zmat_torsion.npz", torsion=dat)
    print("All data have been saved as npz files!")
    print("-"*10 + " Done " + "-"*10)
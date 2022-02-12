"""
Deep Learning for Collective Variables (DeepCV)
https://gitlab.uzh.ch/LuberGroup/deepcv

Info:
28/11/2020 : Rangsiman Ketkaew
"""

"""
Functions for manipulating xyz
"""

import numpy as np
from . import calc_rmsd


def extract_xyz(xyz):
    """Extract Cartesian coordinate from .xyz file

    Args:
        xyz (str): XYZ file

    Returns:
        frame (array): NumPy Array containing trajectory coordinate of each structure/frame
    """
    data = open(xyz).read().splitlines()
    data = list(filter(None, data))
    natoms = int(data[0].split()[0])

    # remove atomic number from every frame
    data = [x for i, x in enumerate(data) if i % (natoms + 2) != 0]
    # remove comment from every frame
    data = [x for i, x in enumerate(data) if i % (natoms + 1) != 0]

    data_new = []
    for i in range(len(data)):
        data_new.append(data[i].split()[1:])

    frames = np.asarray(data_new, dtype=np.float32)
    nframes = int(len(data) / natoms)
    frames = np.reshape(frames, (nframes, -1, 3))

    return frames


def fitting(xyz, ref):
    """Fit structure to a given reference structure

    Args:
        xyz (str): Cartesian coordinate file of all frames
        ref (str): Cartesian coordinate file of reference molecule

    Returns:
        traj_fitted (array): Cartesian coordinate of all frames after superposing
    """
    ref = np.squeeze(ref, axis=0)
    traj_fitted = np.array([calc_rmsd.kabsch_fit(x, ref) for x in xyz])

    return traj_fitted

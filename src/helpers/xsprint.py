"""
Calculate Extended SPRINT coordinate
"""

import numpy as np
from scipy.spatial import distance
import ase.io
from ase.data import chemical_symbols

###############################
# Multi-structure xyz
# f = "../../../dataset/Co-Water/Co-Water-noH-Reactant.xyz"
f = r"C:\Users\Nutt\Desktop\gitlab\dataset\Co-Water\Co-Water-noH-Reactant.xyz"
no_struc = 1
no_atoms = 190
r_0 = {"CoCo": 2.22, "OO": 2.22, "HH": 1.50, "CoO": 2.22, "CoH": 2.22, "OH": 2.00}
n = 6
m = 12
M = 1
###############################


def calc_sprint(symbols, xyz, r_0, n, m, M):
    """Calculate SPRINT coordinate

    Args:
        symbols (list, array): Atomic symbols
        xyz (array): Atomic position with the shape of (number of atom, coordinate in xyz)

    Returns:
        a_i (array): Sorted index of atoms
        s (array): Sorted SPRINT coordinate
    """
    ####################################
    # Calculate a_ij
    # a_ij = (1 - frac^n) / (1 - frac^m)
    ####################################
    # matrix of each bond pair
    # 1. Two loops over atomic symbols to create bond pair
    # 2. Get the r_0 from value of the dict defined above
    tmp = [[r_0[first + second] if first + second in r_0 else r_0[second + first] for second in symbols] for first in symbols]
    r_0 = np.asarray(tmp)
    r_ij = distance.cdist(xyz, xyz)
    frac = np.divide(r_ij, r_0)  # r_ij / r_0
    a_ij = np.divide(1 - np.power(frac, n), 1 - np.power(frac, m))

    ######################################
    # Calculate eigenvalue and eigenvector
    ######################################
    w, v = np.linalg.eig(a_ij)

    ########################
    # Calculate v_i max
    # v_i max = 1 / \lambda_max^M * \sum_(j) a_ij^M * v_j^max
    ########################
    w_max = np.abs(np.max(w))  # lambda max
    a_ij_M = np.power(a_ij, M)
    # where M is the number of walks

    v_i = np.zeros(a_ij.shape[0])
    # loop over atom i
    for i in range(a_ij_M.shape[0]):
        # loop over column j
        _sum = 0.0
        for j in range(a_ij_M.shape[1]):
            _sum += a_ij_M[i][j] * np.max(v.T[j])
        v_i[i] = _sum / (w_max ** M)

    # sort v_i that contains all nonzero with equal sign
    v_i_sorted = np.sort(np.abs(v_i))

    # Get the index of atom according to sorted v_i
    a_i = np.argsort(v_i)

    ############################
    # Calculate sprint coord s_i
    ############################
    s = np.array([np.sqrt(a_ij.shape[0]) * w_max * i for i in v_i_sorted])

    return a_i, s


if __name__ == "__main__":
    # Single structure xyz
    # xyz = ase.io.read("/home/rketka/gitlab/deepcv/dataset/DA/DA_single_ex.xyz")
    # symbols = [chemical_symbols[x] for x in xyz.numbers]
    # xyz = xyz.get_positions()

    ########################
    # Extract atomic symbols
    ########################
    generator = ase.io.iread(f)
    # print(*generator, sep='\n')

    numbers = np.array([])
    arr = np.empty((0, no_atoms, 3))

    for atoms in generator:
        numbers = atoms.numbers[:no_atoms]
        pos = atoms.positions
        pos = pos[:no_atoms]
        pos = pos.reshape((1, -1, 3))
        arr = np.append(arr, pos, axis=0)

    symbols = [chemical_symbols[x] for x in numbers]

    print("\nSorted atom index and corresponding sprint coordinate\n")

    from pprint import pprint

    for i in range(no_struc):
        sorted_index, sorted_SPRINT = calc_sprint(symbols, arr[i], r_0, n, m, M)
        print(f"Structure:\t{i+1}")
        pprint(sorted_index)
        pprint(sorted_SPRINT)
        print("-"*10 + " DONE " + "-"*10)

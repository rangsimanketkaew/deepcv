#!/usr/bin/env python3

"""
Deep Learning for Collective Variables (DeepCV)
https://gitlab.uzh.ch/LuberGroup/deepcv

Info:
28/11/2020 : Rangsiman Ketkaew
"""


import os
import argparse
import math
import numpy as np
from scipy import spatial
import ase.io
from ase.data import chemical_symbols

# ++++++++++++++++++++++++++++++++++++++++++
# Default parameters for adjacency matrix
r_0 = {
    "CoCo": 2.22,
    "CC": 2.65,
    "OO": 2.22,
    "HH": 1.50,
    "CoO": 2.22,
    "CoH": 2.22,
    "OH": 2.00,
    "CH": 2.20,
    "CO": 2.20,
}
n = 6
m = 12
M = 1
# ++++++++++++++++++++++++++++++++++++++++++


def _distance(p1, p2):
    """Calculate bond length between atoms

    Args:
        p1 (array): Cartesian coordinate of the first atom
        p2 (array): Cartesian coordinate of the second atom

    Returns:
        float: Bond length
    """
    return math.sqrt(((p1[0] - p2[0]) ** 2) + ((p1[1] - p2[1]) ** 2) + ((p1[2] - p2[2]) ** 2))


def _angle(a, b, c):
    """Calculate bond angle between atoms

    Args:
        a (array): Cartesian coordinate of the first atom
        b (array): Cartesian coordinate of the second atom (a common atom)
        c (array): Cartesian coordinate of the third atom

    Returns:
        float: Bond angle
    """
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    ang = np.arccos(cosine_angle)
    return np.degrees(ang)


def _angle_sign(a, b, c, degree=False):
    """Calculate angle between three atoms and return value with sign in radian

    Args:
        a (array): Cartesian coordinate of the first atom
        b (array): Cartesian coordinate of the second atom (a common atom)
        c (array): Cartesian coordinate of the third atom
        degree (bool, optional): Return bond angle in degree. Defaults to False.

    Returns:
        float: Bond angle
    """
    ba = a - b
    bc = c - b
    cos_theta = np.dot(ba, bc)
    sin_theta = np.linalg.norm(np.cross(ba, bc))
    theta = np.arctan2(sin_theta, cos_theta)
    if degree:
        return theta * 180.0 / np.pi
    else:
        return theta


def _torsion(p0, p1, p2, p3, degree=False):
    """Calculate torsion angle

    Praxeolitic formula
    1 sqrt, 1 cross product
    https://stackoverflow.com/a/34245697

    Args:
        p0 (array): Cartesian coordinate of the first atom
        p1 (array): Cartesian coordinate of the second atom (a common atom)
        p2 (array): Cartesian coordinate of the third atom (a common atom)
        p3 (array): Cartesian coordinate of the third atom
        degree (bool, optional): _description_. Defaults to False.

    Returns:
        float: Torsion or dihedral angle
    """
    b0 = -1.0 * (p1 - p0)
    b1 = p2 - p1
    b2 = p3 - p2
    # normalize b1 so that it does not influence magnitude of vector
    # rejections that come next
    b1 /= np.linalg.norm(b1)
    # vector rejections
    # v = projection of b0 onto plane perpendicular to b1
    #   = b0 minus component that aligns with b1
    # w = projection of b2 onto plane perpendicular to b1
    #   = b2 minus component that aligns with b1
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    # angle between v and w in a plane is the torsion angle
    # v and w may not be normalized but that's fine since tan is y/x
    x = np.dot(v, w)
    y = np.dot(np.cross(b1, v), w)
    if degree:
        return np.degrees(np.arctan2(y, x))
    else:
        return np.arctan2(y, x)


def calc_int_coord(xyz, filename="structures"):
    """Compute internal coordinates of a given molecule

    Args:
        xyz (array): (dims M x 3) Cartesian coordinates of M atoms in a molecule.
        filename (str, optional): Output filename. Defaults to "structures".

    Returns:
        dist (array): (dims M x N) N distances for each M structure
        angle (array): (dims M x N) N angles for each M structure
        torsion (array): (dims M x N) N torsion angles for each M structure
    """
    no_strct, no_atoms, _ = xyz.shape
    out = filename
    # ---------------------------------------------
    print("Calculating distance ...")
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
    dist = np.zeros((no_strct, no_atoms - 1))
    for i in range(no_strct):
        alldist = spatial.distance.cdist(xyz[i], xyz[i])
        dist[i][0] = alldist[0][1]
        dist[i][1] = alldist[0][2]
        for j in range(no_atoms - 3):
            dist[i][j + 2] = alldist[j][j + 3]

    # ---------------------------------------------
    print("Calculating angle ...")
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
    angle = np.zeros((no_strct, no_atoms - 2))
    for i in range(no_strct):
        angle[i][0] = _angle_sign(xyz[i][2], xyz[i][0], xyz[i][1])
        for j in range(no_atoms - 3):
            angle[i][j + 1] = _angle_sign(xyz[i][j + 3], xyz[i][j], xyz[i][j + 2])

    # ---------------------------------------------
    print("Calculating torsion ...")
    """Calculate torsion (dihedral) angle between atom A, B, C and D.
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
    torsion = np.zeros((no_strct, no_atoms - 3))
    for i in range(no_strct):
        for j in range(no_atoms - 4):
            torsion[i][j] = _torsion(xyz[i][j + 3], xyz[i][j], xyz[i][j + 1], xyz[i][j + 2])

    return dist, angle, torsion


def calc_adj_mat(symbols, xyz, r_0, n, m):
    """Compute adjacency matrix

    Args:
        xyz (array): (dims M x 3) Cartesian coordinates of M atoms in a molecule.
        r_0 (float): cutoff in Angstroms
        n (int): Switching function parameter
        m (int): Switching function parameter

    Returns:
        array: Adjacency matrix
    """
    """Calculate adjacency matrix

    a_ij = (1 - frac^n) / (1 - frac^m)

    Matrix of each bond pair
     1. Two loops over atomic symbols to create bond pair
     2. Get the r_0 from value of the dict defined above
    """
    tmp = [
        [r_0[first + second] if first + second in r_0 else r_0[second + first] for second in symbols]
        for first in symbols
    ]
    r_0 = np.asarray(tmp)
    r_ij = spatial.distance.cdist(xyz, xyz)
    frac = np.divide(r_ij, r_0)  # r_ij / r_0
    a_ij = np.divide(1 - np.power(frac, n), 1 - np.power(frac, m))

    return a_ij


def calc_sprint(symbols, xyz, r_0, n, m, M=1):
    """Compute SPRINT coordinates

    s^x_i = \sqrt{n} \lambda v_i

    References:
    1. https://journals.aps.org/prl/pdf/10.1103/PhysRevLett.107.085504
    2. https://pubs.acs.org/doi/10.1021/acs.jctc.7b01289

    Args:
        symbols (list, array): Atomic symbol
        xyz (array): (dims M x 3) Cartesian coordinates of M atoms in a molecule.
        r_0 (float): cutoff in Angstroms
        n (int): Switching function parameter
        m (int): Switching function parameter
        M (int, optional): Number of walks between atoms. Defaults to 1.

    Returns:
        a_i (array): Sorted index of atoms
        s (array): Sorted SPRINT coordinate
    """
    a_ij = calc_adj_mat(symbols, xyz, r_0, n, m)
    # Calculate eigenvalue and eigenvector
    w, v = np.linalg.eig(a_ij)

    # Calculate v_i max
    # v_i max = 1 / \lambda_max^M * \sum_(j) a_ij^M * v_j^max
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

    # Calculate sprint coord s_i
    s = np.array([np.sqrt(a_ij.shape[0]) * w_max * i for i in v_i_sorted])

    return a_i, s


def calc_xsprint(symbols, xyz, r_0, n, m, M=1, r_0x=1.5, sh=1):
    """Calculate eXtended SPRINT (xSPRINT) coordinates

         /-> s_i                       ; r_ij < r_0
    S^x_i
         \-> 1/w * \sum^N_i r_n * s_i  ; otherwise

    where w is (\sum r_n)**2.

    Args:
        symbols (list, array): Atomic symbol
        xyz (array): (dims M x 3) Cartesian coordinates of M atoms in a molecule.
        r_0 (float): cutoff in Angstroms
        n (int): Switching function parameter
        m (int): Switching function parameter
        M (int, optional): Number of walks between atoms. Defaults to 1.
        r_0x (float): Cutoff of the shell (layer) for xSPRINT in Angstroms. Default to 1.5.
        sh (int): The number of shells. Default to 1.

    Returns:
        a_i (array): Sorted index of atoms
        s_x (array): Sorted SPRINT coordinate
    """
    # Calculate normal SPRINT
    a_i, s = calc_sprint(symbols, xyz, r_0, n, m, M)

    # calculate w
    _w = 0
    for i in range(sh):
        _w += (i + 1) * r_0x
    w = _w ** 2

    # calculate individual extended SPRINT
    for i in range(s.shape[0]):
        s[i] = (i + 1) * r_0x * s[i]

    # Calculate working extended SPRINT
    s_x = s / w

    return a_i, s_x


def main():
    des = "Calculate molecular representation for DeepCV"
    parser = argparse.ArgumentParser(description=des)
    parser.add_argument(
        "--xyz",
        "--input",
        "-i",
        dest="input",
        metavar="FILE",
        type=str,
        required=True,
        help="Cartesian coordinate in XYZ file format (.xyz) or in NumPy's compressed array format (npz).",
    )
    parser.add_argument(
        "--atom-index",
        "-a",
        dest="index_list",
        metavar="ATOM_INDEX",
        type=int,
        nargs="+",
        default=None,
        help="List of atomic index that will be taken. (0-based array index)",
    )
    parser.add_argument(
        "--rep",
        "-r",
        dest="rep",
        choices=["int-coord", "adj-mat", "sprint", "xsprint"],
        help="Representation (descriptor) to calculate",
    )
    parser.add_argument(
        "--save",
        "-s",
        dest="save",
        default=False,
        action="store_true",
        help="Whether save output as NumPy's compressed array format (npz).",
    )
    parser.add_argument(
        "--output",
        "--out",
        "-o",
        dest="output",
        help="Output file. If not specified the default output filename is used",
    )

    args = parser.parse_args()

    ###########################
    # Check input (.xyz) file #
    ###########################

    filename, filext = os.path.splitext(args.input)

    if filext == ".xyz":
        print("Converting text data to NumPy array...")
        f = open(args.input, "r")
        no_atoms = int(f.readline())
        f.close()
        generator = ase.io.iread(args.input)
        # 3D empty array
        # No. of structures x No. of atoms x 3 (xyz coord)
        numbers = np.array([])
        xyz = np.empty((0, no_atoms, 3))
        for atoms in generator:
            numbers = atoms.numbers[:no_atoms]
            pos = atoms.positions
            pos = pos[:no_atoms]
            pos = pos.reshape((1, -1, 3))
            xyz = np.append(xyz, pos, axis=0)
    elif filext == ".npz":
        dat = np.load(args.input)
        xyz = dat[dat.files[0]]
    else:
        exit(f"Error: File type {filext} is not supported.")

    print(f"Shape of NumPy array: {xyz.shape}")

    index = args.index_list
    if index:
        xyz = xyz[:, index]
        print(f"List of atom index: {index}")
        print(f"Shape of NumPy array with only specified atom index: {xyz.shape}")

    ############################
    # Calculate representation #
    ############################

    # Internal coordinates
    if args.rep == "int-coord":
        print("Calculate internal coordinates of all structures")
        dist, angle, torsion = calc_int_coord(xyz, filename)
        print("-"*26)
        print("Shape of NumPy array:")
        print(f"Distance : {dist.shape}")
        print(f"Angle    : {angle.shape}")
        print(f"Torsion  : {torsion.shape}")
        if args.save:
            np.savez_compressed(f"{out}" + "_zmat_distance.npz", dist=dist)
            np.savez_compressed(f"{out}" + "_zmat_angle.npz", angle=angle)
            np.savez_compressed(f"{out}" + "_zmat_torsion.npz", torsion=torsion)

    # find atomic symbols for adj matrix descriptors
    if args.rep in ["adj-mat", "sprint", "xsprint"]:
        symbols = [chemical_symbols[x] for x in numbers]
        no_struc = xyz.shape[0]

    # Adjacency matrix
    if args.rep == "adj-mat":
        print("Calculate adjacency matrix")
        for i in range(no_struc):
            print(f"Structure:\t{i+1}")
            calc_adj_mat(symbols, xyz[i], r_0, n, m)

    # SPRINT coordinates
    if args.rep == "sprint":
        print("Calculate SPRINT coordinates and sorted atom index")
        for i in range(no_struc):
            print(f"Structure:\t{i+1}")
            sorted_index, sorted_SPRINT = calc_sprint(symbols, xyz[i], r_0, n, m, M)
            if args.save:
                np.saved_compressed(
                    f"{filename}_SPRINT_strc_{i}.npz", index=sorted_index, sprint=sorted_SPRINT
                )

    # xSPRINT coordinates
    if args.rep == "xsprint":
        print("Calculate xSPRINT coordinates and sorted atom index")
        for i in range(no_struc):
            print(f"Structure:\t{i+1}")
            sorted_index, sorted_xSPRINT = calc_xsprint(symbols, xyz[i], r_0, n, m, M)
            if args.save:
                np.saved_compressed(f"{filename}_xSPRINT_strc_{i}.npz")

    print("-" * 10 + " Done " + "-" * 10)


if __name__ == "__main__":
    main()

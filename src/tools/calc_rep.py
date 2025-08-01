#!/usr/bin/env python3

"""
Deep Learning for Collective Variables (DeepCV)
https://gitlab.uzh.ch/LuberGroup/deepcv

Info:
28/11/2020 : Rangsiman Ketkaew
"""


import os
import argparse
import multiprocessing as mp
from functools import partial
import numpy as np
from scipy import spatial
import ase.io
from ase.data import chemical_symbols
from tools import adjmat_param as param


def find_atomic_symbol(numbers, index):
    # find atomic symbols for adj matrix descriptors
    symbols = [chemical_symbols[x] for x in numbers]
    # filter the symbols
    if index:
        symbols = [symbols[i] for i in index]

    return symbols


def _distance(p1, p2):
    """Calculate bond length between atoms

    Args:
        p1 (array): Cartesian coordinate of the first atom
        p2 (array): Cartesian coordinate of the second atom

    Returns:
        float: Bond length
    """
    return np.sqrt(
        ((p1[0] - p2[0]) ** 2) + ((p1[1] - p2[1]) ** 2) + ((p1[2] - p2[2]) ** 2)
    )


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


def calc_dist(xyz):
    """Compute distances between atoms A and B of a given molecule

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

    Args:
        xyz (array): (dims M x 3) Cartesian coordinates of M atoms in a molecule.

    Returns:
        dist (array): (dims M-1) Distances for M atoms
    """
    no_atoms, _ = xyz.shape
    dist = np.zeros(no_atoms - 1)
    # for i in range(no_strct):
    alldist = spatial.distance.cdist(xyz, xyz)
    dist[0] = alldist[0][1]
    dist[1] = alldist[0][2]
    for i in range(no_atoms - 3):
        dist[i + 2] = alldist[i][i + 3]

    return dist


def calc_angle(xyz):
    """Compute angles between atoms A, B and C of a given molecule

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

    Args:
        xyz (array): (dims M x 3) Cartesian coordinates of M atoms in a molecule.

    Returns:
        angle (array): (dims M-2) Bond angles for each M structure
    """
    no_atoms, _ = xyz.shape
    angle = np.zeros(no_atoms - 2)
    angle[0] = _angle_sign(xyz[2], xyz[0], xyz[1])
    for i in range(no_atoms - 3):
        angle[i + 1] = _angle_sign(xyz[i + 3], xyz[i], xyz[i + 2])

    return angle


def calc_torsion(xyz):
    """Compute torsion (dihedral) angles between atoms A, B, C and D of a given molecule

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

    Args:
        xyz (array): (dims M x 3) Cartesian coordinates of M atoms in a molecule.

    Returns:
        torsion (array): (dims M-3) Torsion angles for each M structure
    """
    no_atoms, _ = xyz.shape
    torsion = np.zeros(no_atoms - 3)
    for i in range(no_atoms - 4):
        torsion[i] = _torsion(xyz[i + 3], xyz[i], xyz[i + 1], xyz[i + 2])

    return torsion


def calc_adjmat(xyz, symbols, r_0, n, m):
    """Compute adjacency matrix

    a_ij = (1 - frac^n) / (1 - frac^m)

    Matrix of each bond pair
     1. Two loops over atomic symbols to create bond pair
     2. Get the r_0 from value of the dict defined above

    Args:
        xyz (array): (dims M x 3) Cartesian coordinates of M atoms in a molecule.
        symbols (list, array): Atomic symbol
        r_0 (float): cutoff in Angstroms
        n (int): Switching function parameter
        m (int): Switching function parameter

    Returns:
        array: Adjacency matrix
    """
    try:
        tmp = [
            [
                r_0[first + second] if first + second in r_0 else r_0[second + first]
                for second in symbols
            ]
            for first in symbols
        ]
    except KeyError as err:
        exit(
            f'Error: Chemical symbol pair {err} is not defined in "src/tools/adjmat_param.py". Please check!'
        )

    r_0 = np.asarray(tmp)
    r_ij = spatial.distance.cdist(xyz, xyz)
    frac = np.divide(r_ij, r_0)  # r_ij / r_0
    a_ij = np.divide(1 - np.power(frac, n), 1 - np.power(frac, m))

    return a_ij


def calc_sprint(xyz, symbols, r_0, n, m, M=1):
    r"""Compute SPRINT coordinates

    s^x_i = \sqrt{n} \lambda v_i

    References:
    1. https://journals.aps.org/prl/pdf/10.1103/PhysRevLett.107.085504
    2. https://pubs.acs.org/doi/10.1021/acs.jctc.7b01289

    Args:
        xyz (array): (dims M x 3) Cartesian coordinates of M atoms in a molecule.
        symbols (list, array): Atomic symbol
        r_0 (float): cutoff in Angstroms
        n (int): Switching function parameter
        m (int): Switching function parameter
        M (int, optional): Number of walks between atoms. Defaults to 1.

    Returns:
        a_i (array): Sorted index of atoms
        s (array): Sorted SPRINT coordinates
    """
    a_ij = calc_adjmat(xyz, symbols, r_0, n, m)
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
        v_i[i] = _sum / (w_max**M)

    # sort v_i that contains all nonzero with equal sign
    v_i_sorted = np.sort(np.abs(v_i))

    # Get the index of atom according to sorted v_i
    a_i = np.argsort(v_i)

    # Calculate sprint coord s_i
    s = np.array([np.sqrt(a_ij.shape[0]) * w_max * i for i in v_i_sorted])

    return a_i, s


def calc_xsprint(xyz, symbols, r_0, n, m, M=1, r_0x=1.5, sh=1):
    r"""Calculate eXtended SPRINT (xSPRINT) coordinates

         /-> s_i                       ; r_ij < r_0
    S^x_i
         \-> 1/w * \sum^N_i r_n * s_i  ; otherwise

    where w is (\sum r_n)**2.

    Args:
        xyz (array): (dims M x 3) Cartesian coordinates of M atoms in a molecule.
        symbols (list, array): Atomic symbol
        r_0 (float): cutoff in Angstroms
        n (int): Switching function parameter
        m (int): Switching function parameter
        M (int, optional): Number of walks between atoms. Defaults to 1.
        r_0x (float): Cutoff of the shell (layer) for xSPRINT in Angstroms. Default to 1.5.
        sh (int): The number of shells. Default to 1.

    Returns:
        a_i (array): Sorted index of atoms
        s_x (array): Sorted xSPRINT coordinates
    """
    # Calculate normal SPRINT
    a_i, s = calc_sprint(xyz, symbols, r_0, n, m, M)

    # calculate w
    _w = 0
    for i in range(sh):
        _w += (i + 1) * r_0x
    w = _w**2

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
        required=True,
        help="List of atomic index that will be taken. (1-based array index)",
    )
    parser.add_argument(
        "--rep",
        "-r",
        dest="rep",
        choices=["zmat", "adjmat", "sprint", "xsprint"],
        help="Representation (descriptor) to calculate",
    )
    parser.add_argument(
        "--num-procs",
        "-np",
        dest="num_procs",
        metavar="NUM_PROCS",
        type=int,
        default=0,
        required=False,
        help="Number of processors for parallel calculation",
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
    # Check index
    if min(index) <= 0:
        exit(
            f"Error: there is at least one index in {index} that is equal or less than 0."
        )
    # Change index of atoms from 1-based to 0-based
    index = [i - 1 for i in index]
    if index:
        xyz = xyz[:, index]
        print(f"List of atom index: {index}")
        print(f"Shape of NumPy array with only specified atom index: {xyz.shape}")

    ############################
    # Calculate representation #
    ############################

    no_struc, _, _ = xyz.shape
    if args.rep in ["adjmat", "sprint", "xsprint"]:
        symbols = find_atomic_symbol(numbers, index)

    # Setting up multiprocessing
    if args.num_procs == 0:
        num_workers = 1
    elif args.num_procs > 0:
        num_workers = int(args.num_procs)

    if num_workers > os.cpu_count():
        print(
            f"Requested number of processors ({num_workers}) is greater than the number of physical processors ({os.cpu_count()})!!"
        )
    chunksize = max(1, int(no_struc / num_workers))

    # Internal coordinates
    if args.rep == "zmat":
        print("Calculate distance coordinates of all structures")
        with mp.Pool(num_workers) as p:
            matrix = np.array(p.map(calc_dist, xyz, chunksize))
        if args.save:
            np.savez_compressed(f"{filename}_dist.npz", dist=matrix)

        print("Calculate bond angle coordinates of all structures")
        with mp.Pool(num_workers) as p:
            matrix = np.array(p.map(calc_angle, xyz, chunksize))
        if args.save:
            np.savez_compressed(f"{filename}_angle.npz", angle=matrix)

        print("Calculate torsion angle coordinates of all structures")
        with mp.Pool(num_workers) as p:
            matrix = np.array(p.map(calc_angle, xyz, chunksize))
        if args.save:
            np.savez_compressed(f"{filename}_torsion.npz", torsion=matrix)

    # Adjacency matrix
    elif args.rep == "adjmat":
        print("Calculate adjacency matrix")
        with mp.Pool(num_workers) as p:
            worker = partial(
                calc_adjmat, symbols=symbols, r_0=param.r_0, n=param.n, m=param.m
            )
            matrix = np.array(p.map(worker, xyz, chunksize))
        if args.save:
            np.savez_compressed(f"{filename}_{args.rep}.npz", adjmat=matrix)

    # SPRINT coordinates
    elif args.rep == "sprint":
        print("Calculate SPRINT coordinates and sorted atom index")
        with mp.Pool(num_workers) as p:
            worker = partial(
                calc_sprint, symbols=symbols, r_0=param.r_0, n=param.n, m=param.m
            )
            sorted_index, sorted_SPRINT = zip(*p.map(worker, xyz, chunksize))
        if args.save:
            np.savez_compressed(
                f"{filename}_{args.rep}.npz", index=sorted_index, sprint=sorted_SPRINT
            )

    # xSPRINT coordinates
    elif args.rep == "xsprint":
        print("Calculate xSPRINT coordinates and sorted atom index")
        with mp.Pool(num_workers) as p:
            worker = partial(
                calc_xsprint, symbols=symbols, r_0=param.r_0, n=param.n, m=param.m
            )
            sorted_index, sorted_xSPRINT = zip(*p.map(worker, xyz, chunksize))
        if args.save:
            np.savez_compressed(
                f"{filename}_{args.rep}.npz", index=sorted_index, sprint=sorted_xSPRINT
            )


if __name__ == "__main__":
    main()

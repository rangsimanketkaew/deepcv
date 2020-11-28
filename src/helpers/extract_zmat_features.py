import os
import glob
import argparse
import numpy as np
from scipy.spatial.distance import cdist
from .extract_features_DA import angle_sign, dihedral


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
    info="Extract internal coordinate and save as NumPy's compressed array format."
    parser = argparse.ArgumentParser(description=info)
    parser.add_argument("--input", "-i", dest="input_npz", metavar="files.npz", type=str, required=True, nargs="*",
        help="Cartesian coordinate in NumPy's compressed array format with 'coords' as a key.")
    parser.add_argument("--key", "-k", dest="key_npz", metavar="keyword", default="coords", type=str,
        help="Keyword name that corresponds to array saved in the npz file. \
            Note that all npz files must have the same keyword name. Defaults to 'coords'.")
    parser.add_argument("--output-dir", "-d", metavar="directory", default=os.getcwd(), type=str,
        help="Output directory to store npz files of distance, angle, and torsion. \
            Defaults to current directory where this code is executed.")
    arg = parser.parse_args()
    files = []
    for f in arg.input_npz:
        files += glob.glob(f)
    npz = sorted(files)
    for i, f in enumerate(npz):
        print(f"Input {i+1}: {f}")
    xyz = np.load(npz[0])[arg.key_npz]
    for i in range(1, len(npz)):
        xyz = np.vstack((xyz, np.load(npz[i])[arg.key_npz]))
    print(f"Shape of NumPy array (after stacking): {xyz.shape}")
    print("Calculating distance, angle, torsion ...")
    int_dist = distance(xyz)
    int_angle = angle(xyz)
    int_torsion = torsion(xyz)
    np.savez_compressed(f"{arg.output_dir}" + "/distance.npz", dist=int_dist)
    np.savez_compressed(f"{arg.output_dir}" + "/angle.npz", angle=int_angle)
    np.savez_compressed(f"{arg.output_dir}" + "/torsion.npz", torsion=int_torsion)
    print("---Done!---")

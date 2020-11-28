import os
import math
import numpy as np


def distance(p1, p2):
    """Calculate bond distance between Carbon
    """
    return math.sqrt(((p1[0] - p2[0]) ** 2) + ((p1[1] - p2[1]) ** 2) + ((p1[2] - p2[2]) ** 2))


def angle(a, b, c):
    """Calculate angle between carbon
    """
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    ang = np.arccos(cosine_angle)
    return np.degrees(ang)


def angle_sign(a, b, c, degree=False):
    """Calculate angle between three atoms and return value with sign in radian.
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

def dihedral(p0, p1, p2, p3, degree=False):
    """Praxeolitic formula
    1 sqrt, 1 cross product
    https://stackoverflow.com/a/34245697
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


if __name__ == "__main__":

    """
    Product of Diels-Alder reaction

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

    o = os.path.join(dir, r"c:\Users\Nutt\Desktop\DA_dist.npz")
    np.savez_compressed(o, dist=dist, ang=ang, dih=dih)

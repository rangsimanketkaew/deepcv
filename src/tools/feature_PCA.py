"""
Deep Learning for Collective Variables (DeepCV)
https://gitlab.uzh.ch/LuberGroup/deepcv

Info:
28/11/2020 : Rangsiman Ketkaew
"""

import argparse
import os
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="Feature analysis using PCA")
parser.add_argument(
    "--npz",
    "-i",
    dest="npz",
    metavar="INPUT.npz",
    type=str,
    required=True,
    help="Input file (.npz) containing multiple arrays of descriptors",
)
parser.add_argument(
    "--plot",
    "-p",
    dest="plot",
    type=str,
    default="2d",
    choices=["2d", "3d"],
    help="Show PCA plot. Supports both 2D and 3D plots.",
)

args = parser.parse_args()

try:
    dat = np.load(args.npz)
    print("Data: ", dat.files)
except:
    print("Input file does not exist or cannot load .npz file")

# Use specific keys for the array that saved by DeepCV's function
# of descriptor calculation: dist, ang, and dih
# However, the users can also change key
mat = np.concatenate((dat["dist"], dat["ang"], dat["dih"]), axis=1)
print("Shape of combinded array: ", mat.shape)

# standardize the data
mat_std = StandardScaler().fit_transform(mat)

#### Run PCA ####
# Set the number of components for PCA
pca = PCA(n_components=mat_std.shape[1])
pca = pca.fit(mat_std)

print(pca.explained_variance_)
print(pca.explained_variance_ratio_)
print(pca.explained_variance_ratio_.cumsum())

N = 18

#### Plot components ####
feature_weights = pca.components_
fig, (ax1, ax2) = plt.subplots(1, 2)
bars1 = ax1.bar(range(N), feature_weights[0])
bars2 = ax2.bar(range(N), feature_weights[1])
for i, b in enumerate(bars1):
    b.set_color(plt.cm.jet(1.0 * i / (N - 1)))
for i, b in enumerate(bars2):
    b.set_color(plt.cm.jet(1.0 * i / (N - 1)))
plt.show()

data_pca = pca.transform(mat_std)

#### Plot the data in this projection ####

# 2 components plot
if args.plot == "2d":
    plt.figure()
    plt.scatter(data_pca[:, 0], data_pca[:, 1], marker="x", s=0.1)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Dihedral PCA: alanine tripeptide")
    # cbar = plt.colorbar()
    # cbar.set_label('Energy (kcal/mol)')
    plt.show()

# # 3 components plot
if args.plot == "3d":
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax = Axes3D(fig)
    p = ax.scatter(*zip(*reduced_mat), marker="o", s=0.1)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    # cbar = fig.colorbar(p)
    # cbar.set_label('Energy (kcal/mol)')
    plt.show()

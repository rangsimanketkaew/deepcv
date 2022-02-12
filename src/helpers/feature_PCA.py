"""
Deep Learning for Collective Variables (DeepCV)
https://gitlab.uzh.ch/LuberGroup/deepcv

Info:
28/11/2020 : Rangsiman Ketkaew
"""

import numpy as np
import os

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

dir = os.path.dirname(__file__)
f = os.path.join(dir, "../dataset/DA/features_DA_Prod_100000.npz")

dat = np.load(f)
print(dat.files)

mat = np.concatenate((dat["dist"], dat["ang"], dat["dih"]), axis=1)
print(mat.shape)

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
plt.figure()
plt.scatter(data_pca[:, 0], data_pca[:, 1], marker="x", s=0.1)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Dihedral PCA: alanine tripeptide")
# cbar = plt.colorbar()
# cbar.set_label('Energy (kcal/mol)')
plt.show()

# # 3 components plot
# fig = plt.figure()
# ax = Axes3D(fig)
# p = ax.scatter(*zip(*reduced_mat), marker='o', s=0.1)
# ax.set_xlabel('PC1')
# ax.set_ylabel('PC2')
# ax.set_zlabel('PC3')
# # cbar = fig.colorbar(p)
# # cbar.set_label('Energy (kcal/mol)')
# plt.show()

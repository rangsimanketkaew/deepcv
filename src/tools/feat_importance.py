#!/usr/bin/env python3

"""
Deep Learning for Collective Variables (DeepCV)
https://gitlab.uzh.ch/LuberGroup/deepcv

Info:
05/02/2022 : Rangsiman Ketkaew
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load dataset
f_dir = "/home/rketka/server/projects/04_deepcv/daenn-test/"
arr_1 = f_dir + "stacked_50arr_dist.npz"
arr_2 = f_dir + "stacked_50arr_angle.npz"
arr_3 = f_dir + "stacked_50arr_torsion.npz"

arr_1 = np.load(arr_1)["arr"]
arr_2 = np.load(arr_2)["arr"]
arr_3 = np.load(arr_3)["arr"]

npz = np.concatenate((arr_1, arr_2, arr_3), axis=1)
dat = pd.DataFrame(data=npz)

# scale data
scaler = StandardScaler()
scaler.fit(dat)
X = scaler.transform(dat)

# PCA
pca = PCA()
x_new = pca.fit_transform(X)

# print(pca.explained_variance_ratio_)
# print(pca.singular_values_)

# Plot
coeff = np.transpose(pca.components_[0:2, :])
# print(coeff)
xs = x_new[:, 0]
ys = x_new[:, 1]
scalex = 1.0 / (xs.max() - xs.min())
scaley = 1.0 / (ys.max() - ys.min())

plt1 = plt.figure("PCA")
plt.scatter(xs * scalex, ys * scaley)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt2 = plt.figure("Coefficients")
plt.scatter(coeff[:, 0], coeff[:, 1])
plt.grid()
plt.show()

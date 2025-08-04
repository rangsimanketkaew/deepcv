#!/usr/bin/env python3

"""
Deep Learning for Collective Variables (DeepCV)
https://gitlab.uzh.ch/LuberGroup/deepcv

Info:
05/02/2022 : Rangsiman Ketkaew
"""

"""
Analyze feature importance
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def main():
    info = "Feature importance analysis."
    parser = argparse.ArgumentParser(description=info)
    parser.add_argument(
        "--input",
        "-i",
        dest="input",
        metavar="FILE",
        type=str,
        required=True,
        help="Feature in NumPy's compressed array format (npz).",
    )

    arg = parser.parse_args()
    npz = np.load(arg.input)
    npz = npz[npz.files[0]]
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


if __name__ == "__main__":
    main()

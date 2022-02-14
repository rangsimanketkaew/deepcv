"""
Deep Learning for Collective Variables (DeepCV)
https://gitlab.uzh.ch/LuberGroup/deepcv

Info:
05/01/2022 : Rangsiman Ketkaew
"""

"""
Exploration ability of the state on configurational space
"""

import numpy as np
import sympy as sp

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter


def get_FES(f):
    """Read free energy surface from PLUMED hills file

    Args:
        f (str): PLUMED hills file
    """
    dat = open(f).read().splitlines()
    cv1_bin = int(dat[3].split()[3])
    cv2_bin = int(dat[7].split()[3])
    cv1 = np.zeros(cv1_bin)
    cv2 = np.zeros(cv2_bin)
    ener = np.zeros((cv1_bin, cv2_bin))
    dat = dat[9:]
    dat = [x for x in dat if x]
    full = np.zeros((cv1_bin * cv2_bin, 3), dtype=np.float64)

    for j in range(cv1_bin * cv2_bin):
        d = dat[j].split()
        full[j] = d[0], d[1], d[2]

    full = full.reshape(cv2_bin, -1, 3)

    cv1 = full[0][:, 0]
    cv2 = full[:, 0, 1:2].reshape(-1)
    ener = full[:, :, 2].T

    #################
    cv1, cv2 = np.meshgrid(cv2, cv1)
    ener = ener / 4.184

    return cv1, cv2, ener


def z_func(x, y):
    return x ** 2 + y ** 2


f = r"C:\Users\Nutt\Desktop\fes.dat"
cv1, cv2, ener = get_FES(f)

print(cv1.shape)
print(cv2.shape)
print(ener.shape)

fig = plt.figure("DeepCV analyzer")
ax = fig.gca(projection="3d")
plt.figure(1, figsize=[25, 15])
plt.title("FES")
# plt.xlabel("d($C_{1}⋅⋅⋅C_{3}$) ($\AA$)")
# plt.ylabel("d($C_{2}⋅⋅⋅C_{4}$) ($\AA$)")
plt.xlabel("CV1")
plt.ylabel("CV2")

# Plot energy surface
surf = ax.plot_surface(cv1, cv2, ener, cmap=cm.coolwarm, linewidth=0, antialiased=False)
# invert
plt.gca().invert_xaxis()
plt.gca().invert_yaxis()

ax.set_zlabel("Free Energy $(kcal/mol)$")
ax.zaxis.set_major_locator(LinearLocator(5))
ax.zaxis.set_major_formatter(FormatStrFormatter("%.02f"))

print(cv1[0, :])
print(cv1[0, :].shape)
# print(cv1[:,50])

grad_cv1 = np.gradient(ener, cv1[0, :], axis=1)
grad_cv2 = np.gradient(ener, cv2[:, 0], axis=0)

grad_cv1 = np.array(grad_cv1)
grad_cv2 = np.array(grad_cv2)

# Plot gradients
surf = ax.plot_surface(cv1, cv2, grad_cv1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
surf = ax.plot_surface(cv1, cv2, grad_cv2, cmap=cm.coolwarm, linewidth=0, antialiased=False)

plt.show()

# x = sp.Symbol("x")
# y = sp.Symbol("y")
# f = z_func(x, y)
# gradi = sp.diff(f, x)
# gradj = sp.diff(f, y)

# print(gradi)

# f_z = sp.lambdify([x, y], f, "numpy")

# np_gradi = sp.lambdify(x, gradi, "numpy")
# np_gradj = sp.lambdify(y, gradj, "numpy")

# Z = f_z(X, Y)
# grad_x = np_gradi(X)
# grad_y = np_gradj(Y)

# print(np.gradient(Z)[0].shape)

# # print(grad_y)

# fig = plt.figure()
# ax = fig.gca(projection="3d")  # Create the axes

# # Plot the 3d surface
# surface = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, rstride=2, cstride=2)
# # surface = ax.plot_surface(X, Y, grad_x, cmap=cm.coolwarm, rstride=2, cstride=2)
# surface = ax.plot_surface(X, Y, np.gradient(Z)[0], cmap=cm.coolwarm, rstride=2, cstride=2)

# # Set some labels
# ax.set_xlabel("x-axis")
# ax.set_ylabel("y-axis")
# ax.set_zlabel("z-axis")

# plt.show()

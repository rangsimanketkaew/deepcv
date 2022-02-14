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


def z_func(x, y):
    return x ** 2 + y ** 2


x = sp.Symbol("x")
y = sp.Symbol("y")
f = z_func(x, y)
gradi = sp.diff(f, x)
gradj = sp.diff(f, y)

print(gradi)

f_z = sp.lambdify([x, y], f, "numpy")

np_gradi = sp.lambdify(x, gradi, "numpy")
np_gradj = sp.lambdify(y, gradj, "numpy")

X = np.linspace(-8, 8, 100)
Y = np.linspace(-4, 4, 100)
X, Y = np.meshgrid(X, Y)

# Xx = np.random.uniform(-2,5,(100,100))

Z = f_z(X, Y)
grad_x = np_gradi(X)
grad_y = np_gradj(Y)

print(np.gradient(Z)[0].shape)

# print(grad_y)

fig = plt.figure()
ax = fig.gca(projection="3d")  # Create the axes

# Plot the 3d surface
surface = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, rstride=2, cstride=2)
# surface = ax.plot_surface(X, Y, grad_x, cmap=cm.coolwarm, rstride=2, cstride=2)
surface = ax.plot_surface(X, Y, np.gradient(Z)[0], cmap=cm.coolwarm, rstride=2, cstride=2)

# Set some labels
ax.set_xlabel("x-axis")
ax.set_ylabel("y-axis")
ax.set_zlabel("z-axis")

plt.show()

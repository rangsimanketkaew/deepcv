import numpy as np
import ase.io

f = "/home/rketka/works/DUMP_DNN_DA_MetaD_XTB_NVT_300K-pos-1.xyz"
no_atoms = 16

generator = ase.io.iread(f)
# num = sum(1 for atoms in generator)
# print(num)

# 3D empty array
# No. of structures x No. of atoms x 3 (xyz coord)
arr = np.empty((0, no_atoms, 3))

for atoms in generator:
    pos = atoms.positions
    pos = pos[:no_atoms]
    pos = pos.reshape((1, -1, 3))
    arr = np.append(arr, pos, axis=0)

# save array as npz
out = "coord"
np.savez_compressed(out, coords=arr)

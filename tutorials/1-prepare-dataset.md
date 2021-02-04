# Prepare everything you need before training a model

## Step 1: Run normal molecular dynamics (MD) simulation to generate a trajectory of, i.e., reactant state of the system.

This step can be done by any MD packages. I personally recommend either CP2K or GROMACS because they have an interface with
PLUMED which is a plugin for running metadynamics simulation.

```sh
$ ls

traj.xyz
```

## Step 2: Split a trajectory file into smaller files

It is often that a trajectory file (.xyz) is so large. So we can split it into multiple smaller files using `split` command in linux. For example, my trajectory contains 4000 structures with 50 atoms each. In .xyz file, each structure has 1 line denoting total number of atoms in a molecule, 1 comment line, and 50 lines of coordinates, and resulting in total of 52 lines. If we want to split every 20-th structures, we have to define the `--lines` with 1040 (52x20). Other options can also be used.

```sh
$ split --lines=1040 --numeric-suffixes=001 --suffix-length=3 traj.xyz traj-partial- --additional-suffix=.xyz
$ ls
traj-partial-001.xyz
traj-partial-001.xyz
traj-partial-001.xyz
traj-partial-001.xyz
...
traj-partial-100.xyz
```

## Step 3: Extract z-matrix (internal coordinate) and generate input files (dataset) for neural network 

```sh
$ python deepcv/src/helpers/extract_zmat.py --input traj-partial-001.xyz
Shape of NumPy array: (2000, 190, 3)
Calculating distance ...
Calculating angle ...
Calculating torsion ...
All data have been saved as npz files!
---------- Done ----------

$ ls *zmat*
traj-partial-001_zmat_distance.npz
traj-partial-001_zmat_angle.npz
traj-partial-001_zmat_torsion.npz
```

And you can use for loop to speed up this step

```sh
$ for i in traj-partial-*.xyz; do echo $i; python deepcv/src/helpers/extract_zmat.py --input $i; done
```

## Step 4: Merge multiple npz files into one npz file

This step we are merging all individual npz files for the same kind of distance, angle, and torsion (separately).

```sh
$ python deepcv/src/helpers/stack_array.py -i traj-partial-*_zmat_distance.npz -k dist
Input 1: traj-partial-001_zmat_distance.npz
Input 2: traj-partial-002_zmat_distance.npz
Input 3: traj-partial-003_zmat_distance.npz
...
Input 100: traj-partial-100_zmat_distance.npz
Shape of output NumPy array (after stacking): (4000, 188)
---------- Done ----------
```

---

## Optional: Convert .xyz to .npz

You can use a script called `xyz2npz.py` to convert .xyz file to NumPy's compress file formats (.npz). It will also save new file with a prefix of the key in npz (default key is `coord`).

```sh
$ python deepcv/src/helpers/xyz2arr.py -i traj-partial-001.xyz
$ ls *.npz
traj-partial-001.npz

## and use for loop for automated task

$ for i in traj-partial-*.xyz; do echo $i; python deepcv/src/helpers/xyz2arr.py -i $i; done
$ ls *.npz
traj-partial-001_coord.npz
traj-partial-002_coord.npz
traj-partial-003_coord.npz
traj-partial-004_coord.npz
...
traj-partial-100_coord.npz
```

then run Python's command prompt to check if everything about saved npz goes well

```sh
$ python
>>> import numpy as np
>>> a = np.load("traj-partial-001_coord.npz")
>>> a.files
['coord']
>>> a['coord'].shape
(20, 50, 3)
```
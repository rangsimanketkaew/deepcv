# Prepare everything you need before training a model <!-- omit in toc -->

- [Step 1: Run a normal molecular dynamics (MD) simulation to generate a trajectory of, i.e., reactant state of the system.](#step-1-run-a-normal-molecular-dynamics-md-simulation-to-generate-a-trajectory-of-ie-reactant-state-of-the-system)
- [Step 2: Split a trajectory file into smaller files](#step-2-split-a-trajectory-file-into-smaller-files)
- [Step 3: Calculate molecular representations and generate input files (dataset) for neural network](#step-3-calculate-molecular-representations-and-generate-input-files-dataset-for-neural-network)
  - [1. Z-matrix (internal coordinate)](#1-z-matrix-internal-coordinate)
  - [2. SPRINT and xSPRINT](#2-sprint-and-xsprint)
- [Step 4: Merge multiple npz files into one npz file](#step-4-merge-multiple-npz-files-into-one-npz-file)
- [Optional: Convert .xyz to .npz](#optional-convert-xyz-to-npz)

## Step 1: Run a normal molecular dynamics (MD) simulation to generate a trajectory of, i.e., reactant state of the system.

This step can be done by any MD packages. I recommend CP2K or GROMACS because they have an interface with
PLUMED which is a plugin for running metadynamics simulation.

```sh
$ ls
traj.xyz
```

## Step 2: Split a trajectory file into smaller files

It is often that a trajectory file (.xyz) is so large. So we can split it into multiple smaller files using `split` command in Linux.
For example, my trajectory contains 4000 structures with 50 atoms each. In .xyz file, each structure has 1 line denoting total number of atoms in a molecule, 1 comment line, and 50 lines of coordinates, resulting in total of 52 lines. If we want to split every 20th structure, we have to define the `--lines` with 1040 (52x20). Other options can also be used.

```sh
$ split --lines=1040 --numeric-suffixes=001 --suffix-length=3 traj.xyz traj-partial- --additional-suffix=.xyz
$ ls
traj-partial-001.xyz
traj-partial-002.xyz
traj-partial-003.xyz
...
traj-partial-100.xyz
```

## Step 3: Calculate molecular representations and generate input files (dataset) for neural network

### 1. Z-matrix (internal coordinate)

```sh
$ deepcv/src/tools/calc_rep.py --input traj-partial-001.xyz --rep zmat
Converting text data to NumPy array...
Shape of NumPy array: (100, 16, 3)
Calculate internal coordinates of all structures
Calculating distance ...
Calculating angle ...
Calculating torsion ...
--------------------------
Shape of NumPy array:
Distance : (100, 15)
Angle    : (100, 14)
Torsion  : (100, 13)
---------- Done ----------
```

Check files

```sh
$ ls *zmat*
traj-partial-001_zmat_distance.npz
traj-partial-001_zmat_angle.npz
traj-partial-001_zmat_torsion.npz
```

### 2. SPRINT and xSPRINT

```sh
$ deepcv/src/tools/calc_rep.py --input traj-partial-001.xyz --rep sprint
Converting text data to NumPy array...
Shape of NumPy array: (3, 16, 3)
Calculate SPRINT coordinates and sorted atom index
Structure:      1
Structure:      2
Structure:      ...
Structure:      100
---------- Done ----------
```

And you can loop over all files, e.g.,

```sh
$ for i in traj-partial-*.xyz ; do echo $i ; deepcv/src/tools/calc_rep.py --input $i --rep zmat ; done
```

## Step 4: Merge multiple npz files into one npz file

In this step, we will merge all individual npz files for the same kind of distance, angle, and torsion (separately).

```sh
$ deepcv/src/helpers/stack_array.py -i traj-partial-*_zmat_distance.npz -k dist
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

You can use a script called `xyz2npz.py` to convert .xyz file to NumPy's compress file formats (.npz). It will also save a new file with a prefix of the key in npz (default key is `coord`).

```sh
$ deepcv/src/helpers/xyz2arr.py -i traj-partial-001.xyz
$ ls *.npz
traj-partial-001.npz

## and use for loop for automated task

$ for i in traj-partial-*.xyz ; do echo $i ; deepcv/src/helpers/xyz2arr.py -i $i ; done
...
output snipped out
...

$ ls *.npz
traj-partial-001_coord.npz
traj-partial-002_coord.npz
traj-partial-003_coord.npz
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

# Preparing dataset

First step prior to calculation of representation is to generate a trajectory of reactant conformers. 
This can be done by using (unbiased) molecular dynamics (MD) simulation to generate a trajectory of the system of interest.
To do so, we recommend general MD packages, e.g., CP2K or GROMACS, because they have an interface with
PLUMED, which is a plugin for running metadynamics simulation.

<figure markdown>
  ![Image title](../img/da-reactant-in-water.png){ width="1200" }
  <figcaption>Ethene and 1,3-butadiene in water (reactant of Diels-Alder reaction). The box size is 8.964321 x 8.964321 x 8.964321 Angstroms^{3}.</figcaption>
</figure>

```sh
$ ls
traj.xyz
```

Coordinates of the Diels-Alder reaction's reactants (water molecules are removed).
```sh
$ less traj.xyz
 16
 8.964321 8.964321 8.964321
 C         2.430478        7.701602        5.168173
 C         1.681471        7.470311        6.245923
 C         3.060065        7.178442        8.580067
 C         3.473648        8.476497        8.540420
 C         2.603304       10.157928        5.590882
 C         2.924473        9.043101        4.926539
 H         2.737233        6.932679        4.545779
 H         1.141304        8.068218        6.905137
 H         2.058858        6.902197        9.062827
 H         2.914139        9.373862        8.811097
 H         1.896734       10.304312        6.395830
 H         3.657252        9.144499        4.106499
 H         3.010528       11.097670        5.178681
 H         4.396297        8.612995        8.046517
 H         3.734648        6.372926        8.296480
 H         1.332488        6.508595        6.561897
...
...
...
```

## [Optional] Split a trajectory file into smaller files

It is often that a trajectory file (.xyz) is so large. So we can split it into multiple smaller files using `split` command in Linux.
For example, my trajectory contains 4000 structures with 50 atoms each. In .xyz file, each structure has 1 line 
denoting total number of atoms in a molecule, 1 comment line, and 50 lines of coordinates, resulting in total of 52 lines. 
If we want to split every 20th structure, we have to define the `--lines` with 1040 (52x20). Other options are also available.

```sh
$ split --lines=1040 --numeric-suffixes=001 --suffix-length=3 traj.xyz traj-partial- --additional-suffix=.xyz
$ ls
traj-partial-001.xyz
traj-partial-002.xyz
traj-partial-003.xyz
...
traj-partial-100.xyz
```

## Calculate molecular representations and generate input files (dataset) for neural network

The script `calc_rep.py` will calculate all representations (molecular feature vectors) needed to trained 
DAENN model, and save the data as .npz.

### 1. Z-matrix (internal coordinate)

Calculate bond distance, bond angle, and dihedral angle (torsion), and save .npz separately.

The following example calculates the features using only all carbon atoms (index 1, 2, 3, 4, 5, 6).

```sh
$ deepcv/src/main.py calc_rep --input traj.xyz --atom-index {1..6} --rep zmat --save
Converting text data to NumPy array...
Shape of NumPy array: (50, 16, 3)
List of atom index: [0, 1, 2, 3, 4, 5]
Shape of NumPy array with only specified atom index: (50, 6, 3)
Calculate internal coordinates of all structures
100%|█████████████████████████████████████████████████████████████| 50/50 [00:00<00:00, 141.18it/s]
```

Check files
```sh
$ ls *zmat*
traj-partial-001_zmat_dist_strc_1.npz
traj-partial-001_zmat_dist_strc_2.npz
traj-partial-001_zmat_dist_strc_3.npz
...
...
traj-partial-002_zmat_dist_strc_1.npz
traj-partial-002_zmat_dist_strc_2.npz
traj-partial-002_zmat_dist_strc_3.npz
...
...
traj-partial-003_zmat_dist_strc_1.npz
traj-partial-003_zmat_dist_strc_2.npz
traj-partial-003_zmat_dist_strc_3.npz
```

### 2. SPRINT and xSPRINT coordinates

- SPRINT coordinates

```sh
$ deepcv/src/main.py calc_rep --input traj-partial-001.xyz --atom-index {1..6} --rep sprint --save
Converting text data to NumPy array...
Shape of NumPy array: (50, 16, 3)
List of atom index: [0, 1, 2, 3, 4, 5]
Shape of NumPy array with only specified atom index: (50, 6, 3)
Calculate SPRINT coordinates and sorted atom index
100%|█████████████████████████████████████████████████████████████| 50/50 [00:03<00:00, 14.00it/s]
```

- xSPRINT coordinates

```sh
$ deepcv/src/main.py calc_rep --input traj-partial-001.xyz --atom-index {1..6} --rep xsprint --save
Converting text data to NumPy array...
Shape of NumPy array: (50, 16, 3)
List of atom index: [0, 1, 2, 3, 4, 5]
Shape of NumPy array with only specified atom index: (50, 6, 3)
Calculate xSPRINT coordinates and sorted atom index
100%|█████████████████████████████████████████████████████████████| 50/50 [00:03<00:00, 14.00it/s]
```

You can also loop over all files, e.g.,

```sh
$ for i in traj-partial-*.xyz ; do echo $i ; deepcv/src/main.py calc_rep --input $i --atom-index {1..6} --rep zmat --save ; done
```

## Merge multiple npz files into one npz file

In this step, we will merge all individual npz files for the same kind of distance, angle, torsion, and sprint.

```sh
$ deepcv/src/main.py stack_array --input traj-partial-*_zmat_dist_strc_* -k dist
100%|█████████████████████████████████████████████████████████████| 49/49 [00:00<00:00, 376.17it/s]
Shape of output NumPy array (after stacking): (50, 5)

$ deepcv/src/main.py stack_array --input traj-partial-*_zmat_angle_strc_* -k angle
100%|█████████████████████████████████████████████████████████████| 49/49 [00:00<00:00, 285.08it/s]
Shape of output NumPy array (after stacking): (50, 4)

$ deepcv/src/main.py stack_array --input traj-partial-*_zmat_torsion_strc_* -k torsion
100%|█████████████████████████████████████████████████████████████| 49/49 [00:00<00:00, 394.47it/s]
Shape of output NumPy array (after stacking): (50, 3)

deepcv/src/main.py stack_array --input traj-partial-*_sprint_strc_* -k sprint
100%|█████████████████████████████████████████████████████████████| 49/49 [00:00<00:00, 5018.58it/s]
Shape of output NumPy array (after stacking): (50, 6)
```

---

## [Optional] Convert .xyz to .npz

You can use a script called `xyz2npz.py` to convert .xyz file to NumPy's compressed file formats (.npz). 
It will also save a new file with a prefix of the key in npz (default key is `coord`).

```sh
$ deepcv/src/main.py xyz2arr.py --input traj-partial-001.xyz
$ ls *.npz
traj-partial-001.npz

## and use for loop for automated task

$ for i in traj-partial-*.xyz ; do echo $i ; deepcv/src/main.py xyz2arr.py --input $i ; done
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

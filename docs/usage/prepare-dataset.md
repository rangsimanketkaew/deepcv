# Preparing dataset

First step prior to calculation of representation is to generate a trajectory of reactant conformers. 
This can be done by using (unbiased) molecular dynamics (MD) simulation to generate a trajectory of the system of interest.
To do so, we recommend general MD packages, e.g., CP2K or GROMACS, because they have an interface with
PLUMED, which is a plugin for running metadynamics simulation.

```sh
$ ls
traj.xyz
```

Have a quick look
```sh
$ less traj.xyz
      16
 i =    30001, time =    15000.500, E =       -40.1294679285
  C        12.3532404797       12.7883020167       12.9546191569
  C        10.9349645617       12.9268428583       12.4266306770
  C        10.4726684318       11.5913894793       11.7534018686
  C        11.4526722998       11.0145362339       10.7692286079
  C        12.8903740451       10.8477161880       11.3763566620
  C        13.1991245312       11.8274104171       12.4666365340
  H        12.6714727462       13.4173826747       13.8229481000
  H        10.1880003865       13.0361781393       13.2477703446
  H        10.3282610842       10.8760558526       12.6037198425
  H        11.0090996016       10.0824366484       10.3794834737
  H        13.0325922755        9.8023404465       11.7113249162
  H        14.2022701994       11.7133680816       12.9393779136
  H        13.6820029718       10.8746085031       10.5459170088
  H        11.5166788971       11.7331571449        9.8991081713
  H         9.4917024284       11.7401620001       11.2334315040
  H        10.7502439325       13.8984885781       11.8872384235
      16
 i =    30002, time =    15001.000, E =       -40.1296861256
  C        12.3555708061       12.7924289541       12.9517171551
  C        10.9344464890       12.9271978793       12.4232294009
  C        10.4706704791       11.5931253522       11.7508223931
  C        11.4520709609       11.0131667104       10.7746594368
  C        12.8874968790       10.8435588465       11.3758828167
  C        13.1979334810       11.8271208015       12.4645029324
  H        12.6736986177       13.4128109393       13.8232565719
  H        10.2015021106       13.0259726707       13.2521027734
  H        10.3236937869       10.8789058260       12.6244014449
  H        11.0085685828       10.0906491331       10.3870245887
  H        13.0245828375        9.8043862545       11.7167813813
  H        14.1997505121       11.7038804305       12.9297350963
  H        13.6695238792       10.8629610830       10.5434629294
  H        11.5229096263       11.7310542409        9.9113463812
  H         9.4787600124       11.7383873760       11.2423234767
  H        10.7481751502       13.8865557981       11.8843902978
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
$ deepcv/src/main.py calc_rep --input traj-partial-001.xyz --rep sprint --save
Converting text data to NumPy array...
Shape of NumPy array: (50, 16, 3)
List of atom index: [0, 1, 2, 3, 4, 5]
Shape of NumPy array with only specified atom index: (50, 6, 3)
Calculate SPRINT coordinates and sorted atom index
100%|█████████████████████████████████████████████████████████████| 50/50 [00:03<00:00, 14.00it/s]
```

- xSPRINT coordinates

```sh
$ deepcv/src/main.py calc_rep --input traj-partial-001.xyz --rep xsprint --save
Converting text data to NumPy array...
Shape of NumPy array: (50, 16, 3)
List of atom index: [0, 1, 2, 3, 4, 5]
Shape of NumPy array with only specified atom index: (50, 6, 3)
Calculate xSPRINT coordinates and sorted atom index
100%|█████████████████████████████████████████████████████████████| 50/50 [00:03<00:00, 14.00it/s]
```

You can also loop over all files, e.g.,

```sh
$ for i in traj-partial-*.xyz ; do echo $i ; deepcv/src/main.py calc_rep --input $i --rep zmat --save ; done
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

## Optional: Convert .xyz to .npz

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

# Running metadynamics simulation

## Converting DAENN/DeepCV model output to PLUMED input file

Now we are ready to convert DAENN CV from DeepCV output to PLUMED file format. PLUMED is a plugin for enhanced sampling simulations
which supports several standard molecular simulation packages, e.g., CP2K, LAMMPS, and GROMACS.

You can use `deecv2plumed` module to generate a PLUMED input file. It takes the same input as you used for `daenn` module.
Basically, it will read and extract the neural network outputs (the optimized weights and biases) and print out a CV function 
in PLUMED format using `MATHEVAL` or `CUSTOM` function ([see details here](https://www.plumed.org/doc-v2.10/user-doc/html/_m_a_t_h_e_v_a_l.html)).

```sh
$ python deepcv/src/main.py deepcv2plumed --input input_ae_DA.json --atom-index {1..6} --sprint-index C=1,2,3,4,5,6 

>>> Reading DeepCV input file ...
>>> Checking weight output file: model_weights.npz
>>> Checking bias output file  : model_biases.npz
>>>   |- Number of Z-matrix inputs: 12
>>>   |- Number of SPRINT inputs: 6
>>>   |- Number of inputs for secondary loss function: 5
>>>   |- Number of total inputs: 23
>>> Encoder info
>>>   |- Activation functions: relu, relu, relu
>>>   |- Size of layers: 32, 8, 2
>>> Plumed data have been written successfully to 'plumed_NN.dat'
```

The example of a generated `plumed_NN.dat` for the Diels-Alder reaction is [plumed_NN.dat](https://gitlab.uzh.ch/lubergroup/deepcv/-/blob/master/output/plumed_NN.dat).

## Test plumed input file

This step is to check if a generated PLUMED input file works or not.
You can use PLUMED driver to run a trial test on a 1-frame simple Diels-Alder trajectory.

```sh
$ plumed driver --ixyz reactant_DA_water_100atoms.xyz --plumed plumed-NN.dat --kt 1 --box 10,10,10
$ tree

.
├── bias.log
├── COLVAR.log
├── HILLS
├── input_plumed.log
├── layer1.log
├── layer2.log
├── layer3.log
├── plumed-NN.dat
└── reactant_DA_water_100atoms.xyz
```

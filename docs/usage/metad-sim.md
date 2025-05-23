# Running metadynamics simulation

## Create a PLUMED input file

DeepCV can be interfaced with PLUMED, a plugin for enhanced sampling simulations
which supports several standard molecular simulation packages, e.g., CP2K, LAMMPS, and GROMACS.
Once the training is complete, you can use `deecv2plumed` script to generate the PLUMED input file. It takes the same input as you used for `daenn.py`.
It will automatically extract the weight and bias from the model and print out the file.

```sh
$ python deepcv/src/main.py deepcv2plumed -i input/input_ae_DA.json -n 16 -o plumed-NN.dat

>>> Plumed data have been written successfully to 'plumed-NN.dat'
>>> In order to run metadynamics using CP2K & PLUMED, specify the following input deck in CP2K input:

# Import PLUMED input file
&MOTION
    &FREE_ENERGY
        &METADYN
            USE_PLUMED .TRUE.
            PLUMED_INPUT_FILE plumed-NN.dat
        &END METADYN
    &END FREE_ENERGY
%END MOTION
```

## Test plumed input file

This step is to check if a generated PLUMED input file works or not.
You can use PLUMED driver to run a trial test on a 1-frame simple Diels-Alder trajectory.

```sh
$ plumed driver --ixyz reactant_DA_water_100atoms.xyz --plumed plumed-NN.dat
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

## Prepare CP2K input files for performing metadynamics simulation

Prepare all necessary files and run metadynamics simulation using CP2K.

```sh
$ tree

.
├── dftd3.dat       # DFT-D3 parameter
├── MetaD.inp       # CP2K input file
├── plumed-NN.dat   # PLUMED input file
├── reactant_DA_water_100atoms.xyz      # Coordinate file
├── run_script.sh   # SLURM script
└── xTB_parameters  # xTB parameter needed only you want to use extended Tight binding
```

## Run a metadynamics simulation using CP2K

```sh
cp2k-8.2-Linux-x86_64.ssmp -i MetaD.inp -o MetaD.out &
```

CP2K is an open-source quantum chemistry and solid state physics software for performing atomistic simulations.
CP2K source code can be obtained free of charge: https://www.cp2k.org.

## Optional: Run CP2K on Piz Daint supercomputer

Prepare a SLURM input file, for example:

```sh
#!/bin/bash -l
#

# ----- SLURM JOB SUBMIT SCRIPT -----
#SBATCH --export=ALL
#SBATCH --error=slurm.%J.err
#SBATCH --output=slurm.%J.out
#SBATCH --exclusive

################ CHANGE this section  (begin) ##########################
# -- job info --
#SBATCH --account=s1036
#SBATCH --partition=normal
#SBATCH --time=24:00:00

# -- number of nodes and CPU usage --
#SBATCH --nodes=48               # of nodes            (default =  1)
#SBATCH --ntasks-per-node=12    # of MPI tasks/node   (default = 36)
#SBATCH --cpus-per-task=1       # of OMP threads/task (default =  1)
#SBATCH --ntasks-per-core=1     # HT (default = 1, HyperThreads = 2)
#SBATCH --constraint=gpu         # CPU partition

###########CP2K#########################################################
# -- the program and input file (basename) --
EXE="/full/path/to/executable/cp2k.psmp"
SLURM_NTASKS_PER_NODE=12

# you can also define/declare/load modules and packages needed for CP2K and PLUMED here

################ NOTHING to be changed here ############################
PROJECT=${SLURM_JOB_NAME}
PROJECT=$(basename $PROJECT .inp)

INP="${PROJECT}.inp"
OUT="${PROJECT}.out"

INPDIR="$PWD"
PROJECTDIR="${INPDIR//scratch\/snx3000/project/s745}"

SLURM_NTASKS_PER_NODE=12

echo ' --------------------------------------------------------------'
echo ' |        --- RUNNING JOB ---                                 |'
echo ' --------------------------------------------------------------'

echo "${SLURM_NTASK_PER_CORE}"

  # stop if maximum number of processes per node is exceeded
  if [ ${SLURM_NTASKS_PER_CORE} -eq 1 ]; then
    if [ $((${SLURM_NTASKS_PER_NODE} * ${SLURM_CPUS_PER_TASK})) -gt 36 ]; then
       echo 'Number of processes per node is too large! (STOPPING)'
       exit 1
    fi
  else
    if [ $((${SLURM_NTASKS_PER_NODE} * ${SLURM_CPUS_PER_TASK})) -gt 72 ]; then
       echo 'Number of processes per node is too large! (STOPPING)'
       exit 1
    fi
  fi

  # build SRUN command
  srun_options="\
    --exclusive \
    --bcast=/tmp/${USER} \
    --nodes=${SLURM_NNODES} \
    --ntasks=${SLURM_NTASKS} \
    --ntasks-per-node=${SLURM_NTASKS_PER_NODE} \
    --cpus-per-task=${SLURM_CPUS_PER_TASK} \
    --ntasks-per-core=${SLURM_NTASKS_PER_CORE}"

  if [ ${SLURM_CPUS_PER_TASK} -gt 1 ]; then
    export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
  else
    export OMP_NUM_THREADS=1
  fi
  srun_command="/usr/bin/time -p srun ${srun_options}"

  # print some informations
  nnodes=${SLURM_NNODES}
  nranks_per_node=${SLURM_NTASKS_PER_NODE}
  nranks=$((${nranks_per_node} * ${nnodes}))
  nthreads_per_rank=${SLURM_CPUS_PER_TASK}
  nthreads=$((${nthreads_per_rank} * ${nranks}))
  echo "SRUN-command: ${srun_command}"
  echo "JOB-config:   nodes=${nnodes} ranks/nodes=${nranks_per_node} threads/rank=${nthreads_per_rank}"
  echo "JOB-total:    nodes=${nnodes} ranks=${nranks} threads=${nthreads}"

  # run the program
  ${srun_command} ${EXE} -i ${INP} -o ${OUT}

echo ' --------------------------------------------------------------'
echo ' |        --- DONE ---                                        |'
echo ' --------------------------------------------------------------'

exit 0
```

and then submit the job

```sh
$ sbatch -J MetaD.inp run_script.sh
```

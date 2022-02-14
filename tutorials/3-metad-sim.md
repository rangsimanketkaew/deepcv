# Running Metadynamics simulation

## Step 1: Create PLUMED input file

Once the training is completed, you can use `deecv2plumed` script to generate the PLUMED input file. It takes the same input as you used for `ae_train.py`. It will automatically extract the weight and bias from model and print out the file.

```sh
$ python deepcv2plumed.py -i input/input_ae_DA.json -n 16 -o plumed-NN.dat

>>> Plumed data have been successfully written to 'plumed-NN.dat'
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

## Step 2: Test plumed input file

This step is to check if a generated PLUMED input file works or not. You can use plumed driver to run a trial test on one-frame simple Diels-Alder trajectory.

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

## Step 3: Prepare CP2K input files for performing metadynamics simulation

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

## Step 4: Submit job on Pit Daint

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

# -- the program and input file (basename) --
EXE="/project/s1001/cp2k_with_plumed/CRAY_XC50-gfortran_mc_Plumed_2.6.2/cp2k.psmp"
SLURM_NTASKS_PER_NODE=12

###########CP2K#########################################################
module load daint-mc
#module load PrgEnv-gnu/6.0.8
module load gcc/8.3.0
module load cray-fftw/3.3.8.7
module load GSL/2.5-CrayGNU-20.08
export PLUMED_ROOT=/project/s1001/cp2k_with_plumed/plumed-2.6.2-MPI
export PATH=$PLUMED_ROOT/bin/:$PATH
export LD_LIBRARY_PATH=$PLUMED_ROOT/lib/:$LD_LIBRARY_PATH

PROJECT=${SLURM_JOB_NAME}
PROJECT=$(basename $PROJECT .inp)

################ NOTHING to be changed here ############################
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

  #save in project folder
  echo "slurm*" > exclude_list
  echo "cp2k.psmp" >> exclude_list
  echo "BASIS_*" >> exclude_list
  echo "*.wfn.bak-*" >> exclude_list
  echo "*.dat" >> exclude_list
  echo "*_POTENTIAL*" >> exclude_list

  #ssh daint101 mkdir -p $PROJECTDIR
  #ssh daint101 rsync -arv --exclude-from=$INPDIR/exclude_list $INPDIR/ $PROJECTDIR

echo ' --------------------------------------------------------------'
echo ' |        --- DONE ---                                        |'
echo ' --------------------------------------------------------------'

exit 0
```

and then submit the job

```sh
$ sbatch -J MetaD.inp run_script.sh
```

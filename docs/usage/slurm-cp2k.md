# Run CP2K on SLURM cluster

Prepare a SLURM input file, for example:

`run_script.sh` is a script for submitting a job on a cluster that uses SLUMR as queue controller.


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
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=normal
#SBATCH --time=24:00:00

# -- number of nodes and CPU usage --
#SBATCH --nodes=48              # of nodes            (default =  1)
#SBATCH --ntasks-per-node=12    # of MPI tasks/node   (default = 36)
#SBATCH --cpus-per-task=1       # of OMP threads/task (default =  1)
#SBATCH --ntasks-per-core=1     # HT (default = 1, HyperThreads = 2)
#SBATCH --constraint=gpu        # CPU partition

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

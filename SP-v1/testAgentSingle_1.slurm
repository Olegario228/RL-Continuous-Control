#!/bin/bash
#SBATCH --job-name=SingleLinkTest1

# In this example we allocate 8GB for the main thread of the parameter variation which holds the full lists of parameters and 500MB for all other tasks. Be aware that the parameters must be specified in the SBATCH AND in the resources below! 
#SBATCH --ntasks=1 --cpus-per-task=1 --mem-per-cpu=8000 --time=56:00:00 --mail-user=peter.manzl@uibk.ac.at 
#SBATCH hetjob ## heterogeneous job ...
#SBATCH --ntasks=120 --cpus-per-task=1 --mem-per-cpu=500 --time=56:00:00 --mail-user=peter.manzl@uibk.ac.at # activate hyperthreading

#SBATCH --mail-user=peter.manzl@uibk.ac.at # Send status information to this email address.
#SBATCH --mail-type=END # Send an e-mail when the job is done.


module purge # unload current modules
resources='--ntasks=1 --cpus-per-task=1 --mem-per-cpu=8000 --time=56:00:00 : --ntasks=120 --cpus-per-task=1 --mem-per-cpu=500 --time=56:00:00' # string given to the call of srun must be equal to what is specified above; could be read from the job information using a script but currently not done. 
# resources='--ntasks=1 --cpus-per-task=1 --mem-per-cpu=800 : --ntasks=400 --cpus-per-task=1 --mem-per-cpu=200'

# load module and surpress output
module load Anaconda3/2023.03/miniconda-base-2023.03 > /dev/null  
eval "$($UIBK_CONDA_DIR/bin/conda shell.bash hook)"

conda activate venvP39 # activate anaconda environment for python

# fom ZID: 
set -x
# scontrol show job $SLURM_JOBID
# cat $0

echo python script starts now

settingsTest='nArms=1 iVar=[1,3] nTests1=1000 nTests2=1000 xState=[-1.75,1.75] yState=[-7.5,7.5] inputPath=SingleLinkAllMethods/SinglePendulumF_Model0_A2C nTestsRand=100'
#settingsTest='nArms=1 iVar=[1,3] nTests1=20 nTests2=20 xState=[-1.75,1.75] yState=[-7.5,7.5] inputPath=SingleLinkAllMethods/SinglePendulumF_Model0_A2C nTestsRand=20'
srun --export=ALL --mpi=pmi2 $resources python3 -m mpi4py.futures ./postProcessingNArmTestsV7.py $settingsTest  # heterogeneous job call
# you can run the test in python without using slurm by calling 
# python postProcessingNArmTestsV7.py nArms=1 iVar=[1,3] nTests1=50 nTests2=50 xState=[-1,1] yState=[-1,1] inputPath=SingleLinkAllMethods/SinglePendulumF_Model0_A2C nTestsRand=10
#srun --mpi=pmi2 python3 -m mpi4py.futures ./postProcessingNArmTestsV6.py # homogeneous call of job



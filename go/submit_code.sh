#!/bin/bash

#SBATCH --mail-type=ALL
##SBATCH --mail-user=username@url
#SBATCH --job-name=go
#SBATCH --output=%j_%x.out
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=50000M

source "/etc/slurm/local_job_dir.sh"
mkdir -p "${LOCAL_JOB_DIR}/job_results"
cmd="python $1 $2"
echo "Command: $cmd"
apptainer exec --nv image.sif $cmd

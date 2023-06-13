#!/bin/bash

#SBATCH --mail-type=ALL
##SBATCH --mail-user=username@url
#SBATCH --job-name=ec
#SBATCH --output=models/%j_%x.out
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=128G

source "/etc/slurm/local_job_dir.sh"
mkdir -p "${LOCAL_JOB_DIR}/job_results"

export APPTAINER_BINDPATH="./:/opt/code,${LOCAL_JOB_DIR}/job_results:/opt/output,${LOCAL_JOB_DIR}/data:/data/"

# Choose --model from bert_bfd or t5_xl_uniref50 and --ec_level from 0 or 1 or 2:
cmd="python /opt/code/prot_ec.py --max_epochs 20 --model bert_bfd --ec_level 1"

datasetfile="ec50_level1.zip" # Select datasetfile from ec50/ec40 and level 0-2

cp $SLURM_SUBMIT_DIR/data/$datasetfile ${LOCAL_JOB_DIR}
mkdir -p ${LOCAL_JOB_DIR}/data
unzip -j -d ${LOCAL_JOB_DIR}/data ${LOCAL_JOB_DIR}/$datasetfile
echo "datasetfile: $datasetfile"

echo "Command: $cmd"
apptainer exec --nv image.sif $cmd

cd "$LOCAL_JOB_DIR"
tar cf zz_${SLURM_JOB_ID}.tar job_results
cp zz_${SLURM_JOB_ID}.tar $SLURM_SUBMIT_DIR/models
rm -rf ${LOCAL_JOB_DIR}/job_results

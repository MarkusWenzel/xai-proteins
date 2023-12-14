#!/bin/bash

#SBATCH --mail-type=ALL
##SBATCH --mail-user=username@url
#SBATCH --job-name=go
#SBATCH --output=models/%j_%x.out
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=128G

source "/etc/slurm/local_job_dir.sh"
mkdir -p "${LOCAL_JOB_DIR}/job_results"

# Select either (1) temporal split ("2016") or (2) CAFA3
datasetfile="clas_go_deepgoplus_temporalsplit.tar.gz"
#datasetfile="clas_go_deepgoplus_cafa.tar.gz"

cp $SLURM_SUBMIT_DIR/data/$datasetfile ${LOCAL_JOB_DIR}
mkdir -p ${LOCAL_JOB_DIR}/data
tar xvzf ${LOCAL_JOB_DIR}/$datasetfile -C ${LOCAL_JOB_DIR}/data
echo "datasetfile: $datasetfile"

export APPTAINER_BINDPATH="./:/opt/code,${LOCAL_JOB_DIR}/job_results:/opt/output,${LOCAL_JOB_DIR}/data:/data/"

cmd="python /opt/code/prot_go.py --nr_frozen_epochs 4 --min_epochs 4 --max_epochs 4 --datasetfile $datasetfile --model bert_bfd"

echo "Command: $cmd"
apptainer exec --nv image.sif $cmd

cd "$LOCAL_JOB_DIR"
tar cf zz_${SLURM_JOB_ID}.tar job_results
cp zz_${SLURM_JOB_ID}.tar $SLURM_SUBMIT_DIR/models
rm -rf ${LOCAL_JOB_DIR}/job_results

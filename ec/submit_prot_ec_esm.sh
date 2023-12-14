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

# Choose --model (bert_bfd, t5_xl_uniref50, esm2_t6_8M_UR50D, esm2_t33_650M_UR50D) and --ec_level (0 or 1 or 2):
cmd="python /opt/code/prot_ec_esm.py --max_epochs 20 --model esm2_t6_8M_UR50D --ec_level 1"
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

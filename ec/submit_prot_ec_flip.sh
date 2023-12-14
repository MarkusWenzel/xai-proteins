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
cmd="python /opt/code/prot_ec.py --model bert_bfd --ec_level 1"
datasetfile="ec50_level1.zip" # Select datasetfile from ec50/ec40 and level 0-2

cp $SLURM_SUBMIT_DIR/data/$datasetfile ${LOCAL_JOB_DIR}
mkdir -p ${LOCAL_JOB_DIR}/data
unzip -j -d ${LOCAL_JOB_DIR}/data ${LOCAL_JOB_DIR}/$datasetfile
echo "datasetfile: $datasetfile"

sleep 1m  

# Flip (substitute with alanine) n most relevant residues
attributionfile=${SLURM_SUBMIT_DIR}/data/ec50_level1/ig_outputs/embedding_attribution.csv
# create /data/test_flip_rel_{n}.json's where most relevant residues are flipped (and /data/test_flip_rnd_{n}.json's where random residues are flipped; with 10 repetitions)
apptainer exec image.sif python flip.py $attributionfile

sleep 2m

# Use checkpoint from previous job
previousresult="zz_<job_id>.tar"
# Adjust --max_epochs according to checkpoint, because we want to test only and thus avoid retraining
cmd="$cmd --max_epochs 3"
echo "previousresult: $previousresult"
cp $SLURM_SUBMIT_DIR/models/$previousresult ${LOCAL_JOB_DIR}
mkdir -p ${LOCAL_JOB_DIR}/job_results/lightning_logs/version_0/checkpoints
tar xvf ${LOCAL_JOB_DIR}/$previousresult -C ${LOCAL_JOB_DIR}/job_results/lightning_logs/version_0/checkpoints --strip-components=4 
checkpointfile=`awk '{cmd=sprintf("basename %s",FILENAME);cmd | getline out; print out; exit}' ${LOCAL_JOB_DIR}/job_results/lightning_logs/version_0/checkpoints/epoch=*.ckpt`
cmd="$cmd --resume_from_checkpoint /opt/output/lightning_logs/version_0/checkpoints/$checkpointfile"


# Flip more and more relevant vs. random residues
for n in {1,2,4,8,16,32,64}; do
    echo ""
    echo ">>>>>>>>>>>>>>>>> Flipping $n RELEVANT residues <<<<<<<<<<<<<<<<<<<<<<"
    cmd_n="$cmd --test_json /data/test_flip_rel_$n.json"
    echo "Command: $cmd_n"
    apptainer exec --nv image.sif $cmd_n 
    sleep 1s

    # Now repeat but flip random residues not the most relevant residues for comparison
    echo ""
    echo ">>>>>>>>>>>>>>>>> Flipping $n RANDOM residues <<<<<<<<<<<<<<<<<<<<<<"
    cmd_n="$cmd --test_json /data/test_flip_rnd_$n.json"
    echo "Command: $cmd_n"
    apptainer exec --nv image.sif $cmd_n    
    sleep 1s
done

cd "$LOCAL_JOB_DIR"
tar cf zz_${SLURM_JOB_ID}.tar job_results
cp zz_${SLURM_JOB_ID}.tar $SLURM_SUBMIT_DIR/models
rm -rf ${LOCAL_JOB_DIR}/job_results

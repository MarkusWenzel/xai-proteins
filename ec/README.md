# Finetuning transformers to Enzyme Commission (EC) number prediction and gaining insights into their inner workings with an adaptation of integrated gradients

## Enter GPU cluster, clone repository and build Apptainer image file

1. We run our analysis on a GPU cluster using [Apptainer](https://apptainer.org/) (version 1.1.2) and [Slurm](https://slurm.schedmd.com/quickstart.html).
2. SSH into the head node of the GPU cluster with `ssh <username>@<hostname>`
3. Clone code repository with `git clone https://github.com/markuswenzel/xai-proteins`
4. Change directory with `cd xai-proteins/ec`
5. Build the [Apptainer](https://apptainer.org/) image file with `apptainer build --force --fakeroot image.sif image.def`
6. Make directory for data with `mkdir data`, and for models with `mkdir models`
7. Leave GPU cluster again with `exit`


## Prepare data for finetuning

8. Preprocess EC data as detailed on <https://github.com/nstrodt/UDSMProt> with <https://github.com/nstrodt/UDSMProt/blob/master/code/create_datasets.sh>, resulting in separate files for ec40 and ec50 on levels L0, L1, and L2. See publication for details: Nils Strodthoff, Patrick Wagner, Markus Wenzel, and Wojciech Samek (2020). [UDSMProt: universal deep sequence models for protein classification](https://doi.org/10.1093/bioinformatics/btaa003). _Bioinformatics_, 36(8), 2401–2409. Alternatively, you can download the already preprocessed six EC-datasets with your web browser from [here](https://datacloud.hhi.fraunhofer.de/s/odHaAoLyTyq4GjL). In particular, we work with ec50_level1, which is EC-classification on level 1 (differentiation between six major enzyme classes), using a cluster/similarity treshold of 50 used to split train and test data.
9. Upload the EC-datasets to the GPU cluster with the terminal:
```
scp ~/Downloads/ec50_level0.zip <username>@<hostname>:~/xai-proteins/ec/data/
scp ~/Downloads/ec50_level1.zip <username>@<hostname>:~/xai-proteins/ec/data/
scp ~/Downloads/ec50_level2.zip <username>@<hostname>:~/xai-proteins/ec/data/
scp ~/Downloads/ec40_level0.zip <username>@<hostname>:~/xai-proteins/ec/data/
scp ~/Downloads/ec40_level1.zip <username>@<hostname>:~/xai-proteins/ec/data/
scp ~/Downloads/ec40_level2.zip <username>@<hostname>:~/xai-proteins/ec/data/
```
You can also use Ubuntu's file manager Nautilus (a.k.a. Files) to copy and paste between your local computer and the file system of the GPU cluster. In Nautilus click on "+ Other Locations" (on the lower left) and then add the server address (next to "Connect to Server"): `sftp://<hostname>/`


## Finetune

10. Run finetuning code on GPU cluster with Slurm/Apptainer: 
```
ssh <username>@<hostname>
cd ~/xai-proteins/ec
sbatch submit_prot_ec.sh
```
* These commands are helpful to track the progress of your cluster jobs:
```
squeue --me
head -n 30 models/<jobid>_ec.out
tail -n 30 models/<jobid>_ec.out
```
* If you would like to switch between the six EC-datasets, you can edit the file "submit_prot_ec.sh" (with an editor like _vi_ ) and change the variable "datasetfile" in this script (from ec50_level1.zip to ec50_level0.zip etc.). Then, launch again with: `sbatch submit_prot_ec.sh`
* Edit "submit_prot_ec.sh" also if you would like to switch between BERT and T5 Encoder (or if you want to re-submit a Slurm job based on a previous checkpoint).


## Explainability analysis of finetuned model with adaptation of integrated gradients

10. Enter the GPU cluster again, if you have logged out in the meantime:
```
ssh <username>@<hostname>
cd ~/xai-proteins/ec
```
11. Extract the *.zip file (explainability analysis is for ec50_level1): 
```
unzip -j -d ./data/ec50_level1 data/ec50_level1.zip
```
12. Continue on the cluster and download the Uniprot-Swissprot 2017_03 dataset: 
`bash ./download_swissprot.sh` 
If needed, give permission to the file with:
`chmod +x ./download_swissprot.sh`
13. Execute `sbatch submit_code.sh processing.py` to create the files `data/ec*_level*/test.json` (and `motif_test.json`/`active_test.json`/`binding_test.json`, which is a subset of test.json only with samples that are annotated.)
14. Make sure that any job result potentially extracted earlier has been removed: `rm -rf models/job_results`
Then, untar the selected job result with `tar -xf models/zz_<?>.tar -C models/` ("<?>" denotes the respective job identifier)
15. (Remove any potential old checkpoints and) copy the new checkpoints/ folder to `./models/ec50_level1`.
```
rm -rf ./models/ec50_level1/checkpoints/
mkdir -p ./models/ec50_level1/
cp -r ./models/job_results/lightning_logs/version_0/checkpoints/ ./models/ec50_level1/
```
16. The resulting directory structure should now look like this (containing further files):
```bash
├── Files .md .py .sh .def .sif
├── data/ # Data files
│   ├── uniprot_sprot_2017_03.xml # UniProt Swissprot data from March 2017
│   ├── *_site.pkl # Enzyme site annotations for annotation_eval.py
│   └── ec50_level1/ # Data needed to run ig.py for ec50_level1
│   	├── test.json # [n,3] sequence, name, label
├── models/ # Models
│   └── ec50_level1/checkpoints/*.ckpt # finetuned model for ec50_level1 prediction 
```
17. Run integrated gradients on embedding level: `sbatch submit_code.sh ig_embedding.py`. Output: ./data/ec50_level1/ig_outputs/embedding_attribution.csv
18. Run ig.py for all layers: `bash call_all.sh`. (Run potentially several times if cluster wall time is hit. You can also select one layer only: `sbatch submit_code.sh ig.py 0` for layer 0, for example. If you want to start all over again from scratch, make sure that you have deleted beforehand any potential previous output with `rm -rf ./data/ec50_level1/ig_outputs` and `rm -rf ./data/ec50_level1/ig_outputs_combined` before you run `bash call_all.sh` again.) 
19. Combine results of all layers and proteins: `sbatch submit_code.sh combining_ig_output_files.py`. Output: ./data/ec50_level1/ig_outputs_combined/*.csv and ./data/ec50_level1/test_rel.pkl
Note that explainability code integrated_gradient_helper.py, ig_embedding.py, ig.py, combining_ig_output_files.py is tailored to ec50_level1.
20. Dimensionality reduction with PCA and t-SNE on GPU cluster: `sbatch submit_code.sh ig_cluster.py`
21. (Leave GPU cluster with `exit`and) continue on local computer (workstation, notebook). Download the output files `./data/ec50_level1/test_rel.pkl`, `./data/ec50_level1/ig_outputs/embedding_attribution.csv`, and `./data/ec50_level1/ig_outputs_combined/*.csv` from cluster to local computer (clone same repository; use same sub-folder structure on local computer like on cluster):
```
git clone https://github.com/markuswenzel/xai-proteins
cd xai-proteins/ec
mkdir -p ./data/ec50_level1/ig_outputs/
mkdir -p ./data/ec50_level1/ig_outputs_combined/
scp <username>@<hostname>:~/xai-proteins/ec/data/ec50_level1/ig_outputs/embedding_attribution.csv ./data/ec50_level1/ig_outputs/
scp <username>@<hostname>:~/xai-proteins/ec/data/ec50_level1/test_rel.pkl ./data/ec50_level1/
scp <username>@<hostname>:~/xai-proteins/ec/data/ec50_level1/ig_outputs_combined/*.csv ./data/ec50_level1/ig_outputs_combined/
```
Also download and uncompress the EC-datasets `/data/*.zip`
```
scp <username>@<hostname>:~/xai-proteins/ec/data/ec50_level1.zip ./data/
unzip -j -d ./data/ec50_level1 data/ec50_level1.zip
```
22. Create new conda environment `stats` on local computer:
```
conda create -n stats ipython
conda activate stats
conda install -c conda-forge "scipy>=1.6.0" pandas seaborn pytorch scikit-learn tqdm lxml statsmodels
```
23. Download Swiss-Prot to local computer with `bash ./download_swissprot.sh`
24. Run (pre-)processing again locally (in conda env `stats`) with `python processing.py`
25. Run: `python annotation_eval.py` which computes correlation between relevances on the sequence-level (computed with an adaptation of integrated gradients) with the active/binding/motif/transmembrane sites found in the protein database UniProt.
26. Run: `python stat_eval.py` which identifies heads with a positive relevance summed along the sequence.

### Note 

The file prot_ec.py used for finetuning was copied to prot_ec_mod.py, which was modified for the explainability analysis as follows:

* Commented out: `def  __init__(self) -> None:`, `self.preprocess_dataset()`, `trainer.fit(model)`, `trainer.test`
* Added `attention_mask=torch.ones_like(input_ids)` to forward function
* Changed `return TensorBoardLogger(save_dir="/opt/output"` to `return TensorBoardLogger(save_dir=""`
* Changed --train_json (below "Data arguments") from `/data/train.json` to `data/ec50_level1/train.json` (same for valid.json & test.json)


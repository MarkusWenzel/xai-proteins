# Finetuning transformers to Gene Ontology (GO) term prediction and gaining insights into their inner workings with an adaptation of integrated gradients

## Enter GPU cluster, clone repository and build Apptainer image file

1. We run our analysis on a GPU cluster using [Apptainer](https://apptainer.org/) (version 1.1.2) and [Slurm](https://slurm.schedmd.com/quickstart.html).
2. SSH into the head node of the GPU cluster with `ssh <username>@<hostname>`
3. Clone code repository with `git clone https://github.com/markuswenzel/xai-proteins`
4. Change directory with `cd xai-proteins/go`
5. Build the [Apptainer](https://apptainer.org/) image file with `apptainer build --force --fakeroot image.sif image.def`
6. Make directory for data with `mkdir data`, and for models with `mkdir models`
7. Leave GPU cluster again with `exit`


## Prepare data for finetuning

8. Preprocess the GO data by Maxat Kulmanov and Robert Hoehndorf, 2020 ([DeepGOPlus: improved protein function prediction from sequence](https://doi.org/10.1093/bioinformatics/btz595). _Bioinformatics_, 36(2), 422–429; data accessible on <https://deepgo.cbrc.kaust.edu.sa/data/>; [data-2016.tar.gz](https://deepgo.cbrc.kaust.edu.sa/data/data-2016.tar.gz), [data-cafa.tar.gz](https://deepgo.cbrc.kaust.edu.sa/data/data-cafa.tar.gz) ), as detailed on <https://github.com/nstrodt/UDSMProt> with <https://github.com/nstrodt/UDSMProt/blob/master/code/create_datasets.sh>, resulting in separate files for the "2016" (a.k.a. "temporalsplit") dataset and the "CAFA3" dataset. For details, see: Nils Strodthoff, Patrick Wagner, Markus Wenzel, and Wojciech Samek (2020). [UDSMProt: universal deep sequence models for protein classification](https://doi.org/10.1093/bioinformatics/btaa003). _Bioinformatics_, 36(8), 2401–2409. Alternatively, you can download the already preprocessed two GO-datasets with your web browser from [here](https://datacloud.hhi.fraunhofer.de/s/px6CMpMp3HzAtjE).
    * "clas_go_deepgoplus_temporalsplit.tar.gz" serves for comparison with UDSMProt (Table 3) and DeepGOPlus ("part 1") and GOLabeler and DeepText2GO.
    * "clas_go_deepgoplus_cafa.tar.gz" can be used for comparison with all participants of [CAFA3](https://doi.org/10.1186/s13059-019-1835-8), with DeepGOPlus ("part 2"), and with [Littmann et al., 2021](https://doi.org/10.1038/s41598-020-80786-0) etc.
9. Upload both GO-datasets to the GPU cluster:
```
scp ~/Downloads/clas_go_deepgoplus_temporalsplit.tar.gz <username>@<hostname>:~/xai-proteins/go/data/
scp ~/Downloads/clas_go_deepgoplus_cafa.tar.gz <username>@<hostname>:~/xai-proteins/go/data/
```
You can also use Ubuntu's file manager Nautilus (a.k.a. Files) to copy and paste between your local computer and the file system of the GPU cluster. In Nautilus click on "+ Other Locations" (on the lower left) and then add the server address (next to "Connect to Server"): `sftp://<hostname>/`


## Finetune

10. Run finetuning code on GPU cluster with Slurm/Apptainer: 
```
ssh <username>@<hostname>
cd ~/xai-proteins/go
sbatch submit_prot_go.sh
```
* These commands are helpful to track the progress of your cluster jobs:
```
squeue --me
head -n 30 models/<jobid>_go.out
tail -n 30 models/<jobid>_go.out
```
* If you would like to switch between the two GO-datasets (2016/temporalsplit and CAFA3), open "submit_prot_go.sh" with an editor, e.g. _vi_, and change the variable "datasetfile" in this script (from clas_go_deepgoplus_temporalsplit.tar.gz to clas_go_deepgoplus_cafa.tar.gz). Then, launch again: `sbatch submit_prot_go.sh`
* Edit the file "submit_prot_go.sh" also to switch between BERT and T5 Encoder (or between different pooling strategies or if you want to re-submit Slurm job based on a previous checkpoint).
* Important: If you train with more than one GPU, you need to submit a final job on 1 GPU at the end, such that the model is tested on 1 GPU, which keeps the test sample order. This is required for the final evaluation with the final, intricate GO evaluation metrics. For this purpose, you must, as last step during finetuning, edit "submit_prot_go.sh" and re-submit a final Slurm job based on the previous checkpoint on one GPU only. Otherwise, the sample order can be mixed.


## Evaluate performance on local computer

11. When the finetuning is completed, `exit` the cluster. Final evaluation of the GO predictions is conducted on your local computer (workstation, notebook). Clone the repository and download the result (`./models/zz_<?>.tar`; "<?>" denotes the respective job identifier) of script `submit_prot_go.sh` (which submits `prot_go.py`) (the last Slurm job must have been run on 1 GPU only, to keep order of test samples!) from GPU cluster to your local computer. Place `zz_<?>.tar` again in the same sub-folder `./models/`:
```
git clone https://github.com/markuswenzel/xai-proteins
cd xai-proteins/go
mkdir models
scp <username>@<hostname>:~/xai-proteins/go/models/zz_<?>.tar ./models/
```
Then, create a conda environment and run the evaluation script with:
```
conda create -n go ipython pandas numpy pyyaml
conda activate go
conda install pytorch cpuonly -c pytorch
python evaluate_go.py
```
* The evaluation script is based on <https://github.com/nstrodt/UDSMProt/blob/master/code/utils/evaluate_deepgoplus.py> 
which was adapted from <https://github.com/bio-ontology-research-group/deepgoplus/blob/master/evaluate_deepgoplus.py>
* See [UDSMProt](https://doi.org/10.1093/bioinformatics/btaa003) section "3.3 GO prediction" for details on the used metrics.
* Note: This final evaluation is performed with (a: 2016/temporal split) [data-2016.tar.gz](https://deepgo.cbrc.kaust.edu.sa/data/data-2016.tar.gz) and (b: CAFA3) [data-cafa.tar.gz](https://deepgo.cbrc.kaust.edu.sa/data/data-cafa.tar.gz) from <https://deepgo.cbrc.kaust.edu.sa/data/> (<https://github.com/bio-ontology-research-group/deepgoplus> states that "data-cafa.tar.gz - CAFA3 challenge dataset" and "data-2016.tar.gz - Dataset which is used to compare DeepGOPlus with GOLabeler and DeepText2GO".)


## Explainability analysis of finetuned model with adaptation of integrated gradients

12. Enter the GPU cluster again, if you have logged out in the meantime:
```
ssh <username>@<hostname>
cd ~/xai-proteins/go
```
13. Make sure that any job result potentially extracted earlier has been removed: `rm -rf models/job_results`
Then, untar the selected job result with `tar -xf models/zz_<?>.tar -C models/`
14. (Remove potential old checkpoints and) copy the new checkpoints/ folder to `./models/clas_go_deepgoplus_temporalsplit` (or `./models/clas_go_deepgoplus_cafa`). For the 2016/temporalsplit dataset:
```
rm -rf ./models/clas_go_deepgoplus_temporalsplit/checkpoints/
mkdir -p ./models/clas_go_deepgoplus_temporalsplit/
cp -r ./models/job_results/lightning_logs/version_0/checkpoints/ ./models/clas_go_deepgoplus_temporalsplit/
```
We continue the explainability analysis with the temporal split (we could also do the same for CAFA3; note that the protein names in `clas_go_deepgoplus_temporalsplit` match UniProt nomenclature in contrast to `clas_go_deepgoplus_cafa` where proteins have different names like "T12345"; in this case, we would need to match via the sequence with the annotations).
15. Extract the compressed files: 
```
tar xvzf data/clas_go_deepgoplus_temporalsplit.tar.gz -C data/
```
16. Download the Uniprot-Swissprot 2017_03 dataset with 
`bash ./download_swissprot.sh`
(If needed: give permission to the file with `chmod +x ./download_swissprot.sh`)
17. Download and extract Prosite:
```
wget https://ftp.expasy.org/databases/prosite/prosite_alignments.tar.gz
tar xvzf prosite_alignments.tar.gz
```
18. Execute `bash download_go.sh` in order to download the two Gene Ontology obo files for the temporal and the CAFA3 split.
19. Execute `sbatch submit_code.sh processing.py` to create the files `data/clas_go_deepgoplus_*/test.json` (and `active_test.json`,`binding_test.json`,`transmembrane_test.json`,`prosite_test.json`, `motif_test.json`, which are subsets of test.json only with samples that are annotated.)
20. The resulting directory structure should now look like this (containing further files):
```bash
├── Files .md .py .sh .def .sif
├── data/ # Data files
│   ├── uniprot_sprot_2017_03.xml # UniProt Swissprot data from March 2017
│   ├── *_site.pkl # Annotations for annotation_eval.py
│   └── clas_go_deepgoplus_temporalsplit/ # Data needed to run ig.py for temporal split
│   |	├── test.json # [n,3] sequence, name, label
│   └── clas_go_deepgoplus_cafa/ # Data needed to run ig.py for CAFA3 split
│   	├── test.json # [n,3] sequence, name, label
├── models/ # Models
│   ├── clas_go_deepgoplus_temporalsplit/checkpoints/*.ckpt # finetuned model for GO 2016/temporalsplit prediction
│   └── clas_go_deepgoplus_cafa/checkpoints/*.ckpt # finetuned model for GO CAFA3 prediction
```
21. (Delete output from potential previous runs: `rm -rf ./data/clas_go_deepgoplus_*/ig_outputs` and `rm -rf ./data/clas_go_deepgoplus_*/ig_outputs_combined`.) 
22. Run integrated gradients on embedding level with `sbatch submit_code.sh ig_embedding.py`, resulting in the output of ./data/clas_go_deepgoplus_temporalsplit/ig_outputs/embedding_attribution.csv
23. Run ig.py on GPU cluster for all layers: `bash call_all.sh`. Run potentially several times if cluster wall time is hit.
24. Combine results of all layers and proteins: `sbatch submit_code.sh combining_ig_output_files.py` (which is processing only temporalsplit, if not changed in ig.py). This results in the output of: ./data/clas_go_deepgoplus_temporalsplit/ig_outputs_combined/*.csv and ./data/clas_go_deepgoplus_temporalsplit/test_rel.json. 
The script combines the 30 csv files (for all 30 layers) with 16xN to a single 480xN csv file, which represents all head vectors for a single protein. These single files are then moved to `data/clas_go_*/ig_outputs_combined`. It also creates a unified DataFrame of all proteins at `data/clas_go_*/test_rel.json`, with a 480 array of relevances to be used in stat_eval.py.
25. Leave the cluster with `exit`and continue on local computer (where you had cloned the same repository, see above). 
26. (Delete output from potential previous runs: `rm -rf ./data/clas_go_deepgoplus_*/ig_outputs` and `rm -rf ./data/clas_go_deepgoplus_*/ig_outputs_combined`.) 
27. Download the output files `./data/clas_go_deepgoplus_temporalsplit/test_rel.json`, `./data/clas_go_deepgoplus_temporalsplit/ig_outputs/embedding_attribution.csv`, and `./data/clas_go_deepgoplus_temporalsplit/ig_outputs_combined/*.csv` from cluster to local computer:
```
#cd xai-proteins/go # if needed
mkdir -p ./data/clas_go_deepgoplus_temporalsplit/ig_outputs/
mkdir -p ./data/clas_go_deepgoplus_temporalsplit/ig_outputs_combined/
scp <username>@<hostname>:~/xai-proteins/go/data/clas_go_deepgoplus_temporalsplit/ig_outputs/embedding_attribution.csv ./data/clas_go_deepgoplus_temporalsplit/ig_outputs/
scp <username>@<hostname>:~/xai-proteins/go/data/clas_go_deepgoplus_temporalsplit/test_rel.json ./data/clas_go_deepgoplus_temporalsplit/
scp <username>@<hostname>:~/xai-proteins/go/data/clas_go_deepgoplus_temporalsplit/ig_outputs_combined/*.csv ./data/clas_go_deepgoplus_temporalsplit/ig_outputs_combined/
```
28. Download GO-dataset `/data/clas_go_deepgoplus_temporalsplit.tar.gz` to the data folder on the local computer and uncompress it:
```
scp <username>@<hostname>:~/xai-proteins/go/data/clas_go_deepgoplus_temporalsplit.tar.gz ./data/
tar xvzf data/clas_go_deepgoplus_temporalsplit.tar.gz -C data/
```
29. Download Swiss-Prot to local computer with `bash ./download_swissprot.sh` 
30. Download the two Gene Ontology obo files to the local computer with: `bash download_go.sh`
31. Create new conda environment `stats` on local computer:
```
conda create -n stats ipython
conda activate stats
conda install -c conda-forge ipython scipy==1.8.1 pandas seaborn pytorch scikit-learn tqdm lxml statsmodels
# respectively with
# conda create -n stats
# conda activate stats
# conda install -c conda-forge "scipy>=1.6.0" pandas==1.3.5 seaborn pytorch scikit-learn tqdm lxml statsmodels ipython
```
32. Run (pre-)processing again locally (in conda env `stats`) with `python processing.py` (re-run if you've switched to a different GO-term)
33. Run: `python annotation_eval.py` which computes correlation between relevances on sequence-level (for each head in each layer: for the positive class i.e. the selected GO-term) with the annotations found in the protein database UniProt, respectively Prosite (active/binding/transmembrane sites, motifs, prosite patterns). Creates figures in results folder.
34. Run: `python stat_eval.py` which identifies heads/layers whose relevances (computed with integrated gradients) are (statistically significantly) positive for the selected classes (either GO-term catalytic activity or binding or membrane). Creates figure in results folder.
35. If you want to repeat the entire explainability analysis for a different GO-term, edit `parameters.json` switching between one of these options for `SELECTED_GO_TERM`: GO:0003824 (catalytic activity), GO:0016020 (membrane), GO:0005488 (binding) (loaded by `processing.py`, `integrated_gradient_helper.py`, `stat_eval.py`, `annotation_eval.py`). (Note: For CAFA3 instead of 2016/temporalsplit, you would need to adapt: `processing.py`, `ig.py`, `ig_embedding.py`, `combining_ig_output_files.py`, `annotation_eval.py`, `stat_eval.py`)


## Additional experiments

36. Additional experiments and analyses can be run with these commands:
```
submit_prot_go_esm.sh # finetune ESM-2 model
submit_prot_go_pretrained_shuffled.sh # frozen pretrained or shuffled encoder
python jaccard.py
python annotation_eval_transmembrane.py
```


### Note
 
The file prot_go.py used for finetuning was copied to prot_go_mod.py, which was modified for the explainability analysis as follows:

* Commented out: `self.dataset.preprocess_temporalsplit()`, `self.dataset.preprocess_cafa3()`, `shutil.copy` in `__init__`, `gradient_checkpointing=self.hparams.gradient_checkpointing)`, `trainer.fit(model)`, `trainer.test`
* Added `attention_mask=torch.ones_like(input_ids)` to forward function
* Changed `return TensorBoardLogger(save_dir="/opt/output"` to `return TensorBoardLogger(save_dir=""`
* Changed "Data arguments" from `/data/train.json` to `data/clas_go_deepgoplus_temporalsplit/train.json` (same for valid.json & test.json)
* Changed `/opt/output/y_all.pt` to `y_all.pt`, and changed `/opt/output/y_hat_all.pt` to `y_hat_all.pt`


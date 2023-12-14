# Notes on the Apptainer container respectively conda environments, for reproducibility

We use an [Apptainer](https://apptainer.org/) (version 1.1.2) container (image.def/image.sif) and two [conda](https://docs.conda.io/) environments (go, stats), that you can create as explained below (and also in the go/README.md and the ec/README.md). For reproducibility, we collect information about the installed package versions in this folder.


## Apptainer

You can create the Apptainer image file (in the go/ or ec/ folder) with:
```
apptainer build --force --fakeroot image.sif image.def
```

We then retrieved information about the installed package versions with:
```
apptainer shell image.sif
conda env export > image_environment.yml
conda list --explicit > image_spec_file.txt
pip freeze > image_requirements.txt
```

If you don't use Apptainer, these conda and pip commands might help to install the same packages (not tested):
```
conda create --name image --file image_spec_file.txt
conda env update --name image --file image_environment.yml
pip install -r image_requirements.txt
```

## Conda environments

### go 

You can create the `go' conda environment as follows:
```
conda create -n go ipython pandas numpy pyyaml
conda activate go
conda install pytorch cpuonly -c pytorch
```

(We had then saved this `go' conda environment with:
```
conda env export > go_environment.yml
conda list --explicit > go_spec_file.txt
pip freeze > go_requirements.txt
```
)

Alternatively, you can reproduce the `go' conda environment from the files in this folder:
```
conda create --name go --file go_spec_file.txt
conda env update --name go --file go_environment.yml
conda activate go
pip install -r go_requirements.txt
```


### stats 

You can create the `stats' conda environment as follows:
```
conda create -n stats ipython
conda activate stats
conda install -c conda-forge ipython scipy==1.8.1 pandas seaborn pytorch scikit-learn tqdm lxml statsmodels
# respectively with
# conda create -n stats
# conda activate stats
# conda install -c conda-forge "scipy>=1.6.0" pandas==1.3.5 seaborn pytorch scikit-learn tqdm lxml statsmodels ipython
```

(We had then saved this `stats' conda environment with:
```
conda activate stats
conda env export > stats_environment.yml
conda list --explicit > stats_spec_file.txt
pip freeze > stats_requirements.txt
```
)

Alternatively, you can reproduce the `stats' conda environment from the files in this folder:
```
conda create --name stats_new --file stats_spec_file.txt
conda env update --name stats_new --file stats_environment.yml
pip install -r stats_requirements.txt
```

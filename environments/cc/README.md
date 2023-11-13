# Running on Narval

Running on Narval requires one to recreate an appropriate environment. 

## Installing via conda and pip

A standard route is to install `conda` and `pip`. This requires internet access and has to be done on a login node. You can use `build_env_cc.sh` for that. Typically, `requirements_cc.txt` will be out-of-date, so you will need to recreate from its current state using `requirements.txt` as a template. A typical set of steps might include:

* Removing torch and pydantic as dependencies (handled by conda)
* Removing the pandas version qualifier (conflicts with compute-canada-built packages)

Then, you have to *both* load the right Python module and activate the right conda environment, e.g.:

```
source ~/.bashrc  # To make conda available
load module python/3.9.6
conda activate poyo
```

A symptom of not doing the latter is e.g. getting glibc errors when loading some libraries, e.g. h5py. To run the allen sdk, you will in addition need to `module load postgresql`.

After that, you can pip install the module via regular means, i.e. `pip install -e .`.

## Pre-packaged environment

[Conda pack](https://conda.github.io/conda-pack/) is an interesting alternative that could be used, eventually, to build an environment on the mila cluster and ship it to Narval. We have not confirmed that it works, however.

## Working around multi-node training
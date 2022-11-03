# Themis
LLNL's Themis is a Python-based scientific workflow ensemble manager for running concurrent UQ simulations on high-performance computers. Using a simple, non-intrusive interface to simulation models, it provides the following capabilities:

- generating ensemble of simulations leveraging LC's HPC resources
- analyzing ensemble of simulations output

Themis has been used for simulations in the domains of Inertial Confinement Fusion, National Ignition Facility experiments, climate, as well as other programs and projects.

The `themis` package manages the execution of simulations. Given a set of inputs (sample points) to run a simulation on, this package will execute them in parallel, monitor their progress, and collect the results. The `themis` package work with Python 2 and 3.


## Basic Installation

### via pip:

```bash
export THEMIS_PATH = themis                              # `themis` can be any name/directory you want
pip install virtualenv                                   # just in case
python3 -m virtualenv $THEMIS_PATH   
source ${THEMIS_PATH}/bin/activate
pip install numpy scikit-learn scipy matplotlib networkx
git clone https://github.com/LLNL/themis
cd themis
pip install .
```

### via conda:

```bash
conda create -n themis -c conda-forge "python>=3.6" numpy scikit-learn scipy matplotlib networkx
conda activate themis
git clone https://github.com/LLNL/themis
cd themis
pip install .
```
## Build Docs

### via pip:

```bash
pip install sphinx sphinx_rtd_theme
```
### via conda:

```bash
conda install -n themis -c conda-forge sphinx sphinx_rtd_theme sphinx-autoapi nbsphinx
```

## Beefy Installation

### via pip:

```bash
export THEMIS_PATH = themis                           # `themis` can be any name/directory you want
pip install virtualenv                                # just in case
python3 -m virtualenv $THEMIS_PATH   
source ${THEMIS_PATH}/bin/activate
pip install numpy scikit-learn scipy matplotlib networkx six pip sphinx sphinx_rtd_theme ipython jupyterlab pytest
git clone https://github.com/LLNL/themis
cd themis
pip install .
```
### via conda:

```bash
conda create -n themis -c conda-forge "python>=3.6" numpy scikit-learn scipy matplotlib six pip networkx sphinx sphinx_rtd_theme sphinx-autoapi nbsphinx jupyterlab ipython ipywidgets nb_conda nb_conda_kernels pytest
conda activate themis
git clone https://github.com/LLNL/themis
cd themis
pip install .
```

### Register your Python env via Jupyter:

```bash
python -m ipykernel install --user --name themis --display-name "Themis Environment"
```


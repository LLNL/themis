# Themis
LLNL's Themis is a Python-based scientific workflow ensemble manager for running concurrent UQ simulations on high-performance computers. Using a simple, non-intrusive interface to simulation models, it provides the following capabilities:

- generating ensemble of simulations leveraging LC's HPC resources
- analyzing ensemble of simulations output

Themis has been used for simulations in the domains of Inertial Confinement Fusion, National Ignition Facility experiments, climate, as well as other programs and projects.

The `themis` package manages the execution of simulations. Given a set of inputs (sample points) to run a simulation on, this package will execute them in parallel, monitor their progress, and collect the results. The `themis` package work with Python 2 and 3.


## Installation

```
# Clone the repo
$ cd <repo_dir>
$ python3 -m venv --system-site-packages themis_venv
$ source themis_venv/bin/activate
$ pip install --trusted-host www-lc.llnl.gov --upgrade pip setuptools
$ python3 setup.py install
$ pip list
```

    
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

## Notes for developers

To make changes and run tests, follow the following steps.
### Clone the repo
```
$ git clone -b develop ssh://git@czgitlab.llnl.gov:7999/weave/themis.git
$ cd themis
$ git checkout -b <your branch name>
```
After you create your own branch, you can make any changes to the code and/or tests.
    
### Create a test virtual environment
```
$ make create_env
```
A virtual environment will be created under /usr/workspace/$USER/gitlab/weave/themis/    

### Install themis into the test virtual environment
```
$ make install
```

### Run unit tests
```
$ make run_unit_tests
```

### Specify which unit tests to run
```
$ ls tests/unit
$ make run_unit_tests UNIT_TESTS=test_manager
$ make run_unit_tests UNIT_TESTS=test_runtime
$ make run_unit_tests UNIT_TESTS="test_runtime test_manager"
```
        
### Run integration tests
```
$ make run_integration_tests
```

### Specify which integration tests to run
```
$ make run_integration_tests INTEGRATION_TESTS=test_laptop
$ make run_integration_tests INTEGRATION_TESTS="test_laptop test_hpc"
    
```

### Commit and push your branch. Let CI tests your changes
```
$ git commit -a -m"message about your commit"
$ git push origin <your branch name>
```
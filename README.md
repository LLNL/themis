# Themis
LLNL's Themis is a Python-based scientific workflow ensemble manager for running concurrent UQ simulations on high-performance computers. Using a simple, non-intrusive interface to simulation models, it provides the following capabilities:

- generating ensemble of simulations leveraging LC's HPC resources
- analyzing ensemble of simulations output

Themis has been used for simulations in the domains of Inertial Confinement Fusion, National Ignition Facility experiments, climate, as well as other programs and projects.

The `themis` package manages the execution of simulations. Given a set of inputs (sample points) to run a simulation on, this package will execute them in parallel, monitor their progress, and collect the results. The `themis` package work with Python 2 and 3.


## Installation

To get the latest public version:

```
pip install llnl-themis

```

To get the latest stable from a cloned repo, simply run:

```
pip install .

```
Alternatively, add the path to this repo to your PYTHONPATH environment variable or in your code with:

```
import sys
sys.path.append(path_to_themis_repo)

```

## Documentation

The documentation can be built from the `docs` directory using:

```bash
make html
```

Read the Docs coming soon.

## Contact Info

Themis maintainer can be reached at: domyancic1@llnl.gov

## Contributing

Contributing to Themis is relatively easy. Just send us a pull request. When you send your request, make develop the destination branch on the Themis repository.

Your PR must pass Themis' unit tests and documentation tests, and must be PEP 8 compliant. We enforce these guidelines with our CI process. To run these tests locally, and for helpful tips on git, see our [Contribution Guide](.github/workflows/CONTRIBUTING.md).

Themis' `develop` branch has the latest contributions. Pull requests should target `develop`, and users who want the latest package versions, features, etc. can use `develop`.


Contributions should be submitted as a pull request pointing to the `develop` branch, and must pass Themis' CI process; to run the same checks locally, use:

```bash
pytest tests/
```

## Releases
See our [change log](CHANGELOG.md) for more details.

## Code of Conduct
Please note that Themis has a [Code of Conduct](.github/workflows/CODE_OF_CONDUCT.md). By participating in the Themis community, you agree to abide by its rules.

## License

Themis is distributed under the terms of the MIT license. All new contributions must be made under the MIT license. See LICENSE and NOTICE for details.

LLNL-CODE-838977


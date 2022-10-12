"""This module provides a python API and command-line
interface to submit a batch script repeatedly.

The API consists of the ``BatchSubmitter`` class.

Run the module as a script to produce the command-line interface.
"""

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import os
import argparse

import themis
from themis import utils
from themis import resource
from themis.backend.worker.prepper import populate_run_dir

class BatchSubmitter(object):  # pylint: disable=too-many-instance-attributes
    """Instances of this class are used to create and launch ensembles.

    The ensemble proceeds by submitting a series of batch scripts to a
    resource manager. Each run of the ensemble is a new allocation (a new "batch job")
    for the resource manager, so each run will have to wait in the batch queue.

    Before the batch scripts are launched, parse them for variables declared with
    "%%".

    :param batch_script: the path to the batch script to execute. The
        batch script will be hard-copied into each run directory, parsed,
        and then executed.
    :type batch_script: str
    :param resource_mgr: the resource manager to submit the script to.
        Accepted values are ``"slurm"``, ``"moab"``, ``"lsf"``, and ``"flux"``.
    :type resource_mgr: str
    :param samples: an iterable of mappings defining the sample for each run.
        The number of samples is equal to the number of runs; furthermore, each
        sample is assigned a unique "run ID" corresponding to its position
        in the ``samples`` iterable.
    :param run_parse: A file path or iterable of file paths. The files specified
        will be hard-copied into the run directories and parsed;
        see :ref:`here <text_replacement>` for more.
    :type run_parse: str or iterable of str, optional
    :param run_copy: A file path or iterable of file paths. Unix-style path
        patterns are supported as well. The files/directories
        specified will be hard-copied into the run directories. ``None``, the
        default, specifies that no files will be copied.
    :type run_copy: str or iterable of str, optional
    :param run_symlink: A file path or iterable of file paths. Unix-style path
        patterns are supported as well. The files/directories
        specified will be symlinked into the run directories. ``None``, the
        default, specifies that no files will be symlinked.
    :type run_symlink: str or iterable of str, optional
    :param run_dir_names: The file system paths of the run directories.
        This argument should be a `python format string
        <https://docs.python.org/3/library/string.html#formatstrings>`_, where
        the field names correspond to the names of the variables in the samples
        argument. For instance, if the variables in the samples argument are
        "hydrostatics" and "viscocity", you might pass in the string
        ``"hydro={hydrostatics}/visc={viscocity}"``. This string will be
        formatted each run to yield the run directory; so one directory
        might be ``hydro=17.6/visc=35``. Note that
        the posix directory separator character "/"
        in the example string means that the resulting run directory
        will be in fact a sequence of two directories.
        The default value of ``None`` lets the naming scheme be determined internally.
    :type run_dir_names: str, optional
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        batch_script,
        samples,
        resource_mgr,
        run_parse=None,
        run_copy=None,
        run_symlink=None,
        run_dir_names=None,
    ):
        """Initialize a new instance."""
        self._batch_script = utils.validate_application(batch_script)
        self._resource_mgr = resource.identify_resource_manager(resource_mgr)
        self._allocator = self._resource_mgr.allocator()
        run_parse, run_copy, run_symlink = utils.validate_run_files(
            run_parse, run_copy, run_symlink
        )
        run_parse.append(batch_script)
        self._run_parse = run_parse
        self._run_copy = run_copy
        self._run_symlink = run_symlink
        if run_dir_names is None:
            self._run_dir_names = os.path.abspath(utils.DEFAULT_RUN_DIR_NAMES)
        else:
            self._run_dir_names = os.path.abspath(run_dir_names)
        self._samples = utils.validate_samples(
            samples, run_dir_names, check_nonempty=True
        )

    def __repr__(self):
        return "{}({!r}, {}, {})".format(
            type(self).__name__, self._batch_script, self._resource_mgr, self._samples
        )

    def execute(self, verbosity=0):
        """Submit the batch scripts to the resource manager.

        :returns: the job IDs of the submitted batch scripts.
        :param verbosity: If > 0, print messages about the progress of the submissions.
        """
        job_ids = []
        for i in range(len(self._samples)):
            run_dir = self._dry_run_one(i, verbosity)
            parsed_batch_script = os.path.join(
                run_dir, os.path.basename(self._batch_script)
            )
            jobid = self._allocator.launch_batch_script(parsed_batch_script, run_dir)
            if verbosity > 0:
                print(
                    "Submitted batch script, job ID is {}".format(
                        self._allocator.job_id
                    )
                )
            job_ids.append(jobid)
        return job_ids

    def dry_run(self, *run_ids, **kwargs):
        """Populate run directories with the `run_*` files.

        That is, symlink the `run_symlink` files, and so on.

        :param run_ids: the run IDs of the runs to dry-run.
        :param verbosity: If > 0, print messages about the progress of the dry runs.
        """
        verbosity = kwargs.get("verbosity", 0)
        if not run_ids:
            run_ids = range(len(self._samples))
        for run_id in run_ids:
            self._dry_run_one(run_id, verbosity)

    def _dry_run_one(self, run_id, verbosity):
        """Dry run a single run.

        :returns: the path to the run directory.
        """
        sample = self._samples[run_id]
        run_dir = utils.get_run_dir(run_id, self._run_dir_names, sample)
        if not os.path.exists(run_dir):
            os.makedirs(run_dir)
        if verbosity > 0:
            print(
                "Populating {!r} for run #{}...".format(
                    os.path.relpath(run_dir), run_id
                )
            )
        populate_run_dir(
            self._run_symlink, self._run_copy, self._run_parse, sample, run_dir
        )
        return run_dir


def setup_parser():
    """Setup the command-line parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Submit a batch script to a resource manager repeatedly, modifying it "
            "each time. This tool is a simpler offshoot of Themis with very different "
            "behavior. Themis runs inside of a single allocation and manages each run "
            "individually. This tool, `themis-laf`, just submits a collection of "
            "batch scripts, and then quits---nothing more."
        )
    )
    parser.add_argument(
        "batchscript", type=str, help="Path to the batch script to modify and submit"
    )
    utils.add_parameterfile_arg(parser)
    parser.add_argument(
        "resourcemgr",
        type=lambda x: str(x).lower(),
        choices=["slurm", "moab", "lsf", "sbatch", "none"],
        help="Resource manager to submit the script to",
    )
    utils.add_common_creation_args(parser)
    parser.add_argument(
        "-d",
        "--dry-run",
        type=int,
        nargs="*",
        help=(
            "Perform dry runs: don't submit the batch scripts, but set everything "
            "up for submission. Pass one or more integers to perform specific dry "
            "runs. (Each run is identified by its 0-based row in the CSV.) "
            "Default behavior is to dry-run every run."
        ),
        default=None,
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase verbosity of output",
    )
    parser.add_argument(
        "--version", action="version", version="%(prog)s " + str(themis.__version__)
    )
    return parser.parse_args()


def main():
    """Run the command-line interface to the ``BatchSubmitter``."""
    args = setup_parser()

    print(args)
    samples = utils.read_csv(args.parameterfile)
    mgr = BatchSubmitter(
        args.batchscript,
        samples,
        args.resourcemgr,
        args.run_parse,
        args.run_copy,
        args.run_symlink,
        args.run_dir_names,
    )
    if args.dry_run is not None:
        mgr.dry_run(*args.dry_run)
    else:
        job_id_strs = [str(job_id) for job_id in mgr.execute()]
        print("Batch job IDs are: " + ", ".join(job_id_strs))


if __name__ == "__main__":
    main()

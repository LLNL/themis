# -*- coding: utf-8 -*-

"""
This module defines utility functions and classes.

The classes and functions may be used by multiple other modules within this package.
This module is not intended for use by client code.
"""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import shutil
import signal
import glob
import warnings
import csv
import functools
import shlex

from themis.versions import Sequence, csvargs


DEFAULT_RUN_DIR_NAMES = "runs{sep}{{run_id}}".format(sep=os.sep)
DEFAULT_SETUP_DIR = os.path.join(os.curdir, ".themis_setup")
URI_ENV_VAR = "THEMIS_URI"
RUNID_ENV_VAR = "THEMIS_RUNID"
SETUPDIR_ENV_VAR = "THEMIS_SETUPDIR"
EXECDIR_ENV_VAR = "THEMIS_EXECDIR"
URI_SPLITCHAR = ":"


class Step(object):
    """Instances of this class represent the execution of one application.

    Instances do not do anything themselves; they are meant to be passed into
    other methods and functions.

    Many attributes map directly to ``lrun``, ``srun``, and ``flux mini run``
    arguments.

    :param args: the application and its arguments, either as a list of strings
        or as a single string, e.g. ``"/bin/bash -c 'exit 1;'"``
        or ``["/bin/bash", "-c", "exit 1;"]``. If a single string is passed,
        it will be converted to a list via ``shlex.split(args)``. The first
        element of the Sequence should be the *absolute* path to the
        application to execute.
    :type args: str or Sequence of str
    :param tasks: total MPI tasks. Corresponds to srun, lrun, and flux "-n" options.
    :type tasks: int, optional
    :param cores_per_task: cores per task. Corresponds to srun,
        lrun, and flux "-c" options.
    :type cores_per_task: int, optional
    :param gpus_per_task: gpus per task. Corresponds to lrun and flux "-g" option.
    :type gpus_per_task: int, optional
    :param timeout: the max running time, in minutes, for this run; if the limit is
        exceeded, the run will be killed.
    :type timeout: int, optional
    :param batch_script: a boolean indicating whether the application given by
        ``args[0]`` is a batch script. If it is, it will be parsed for tokens and
        copied into ``cwd`` before execution.
    :type batch_script: bool
    :param cwd: the working directory for the application given by ``args[0]``.
        If the path is relative, it is treated as relative the run directory for
        the owning run. By default, the CWD is the run directory.
    :type cwd: str
    """

    __slots__ = (
        "args",
        "tasks",
        "cores_per_task",
        "gpus_per_task",
        "timeout",
        "batch_script",
        "cwd",
    )

    def __init__(  # pylint: disable=too-many-arguments
        self,
        args,
        tasks=1,
        cores_per_task=1,
        gpus_per_task=0,
        timeout=0,
        batch_script=True,
        cwd=os.curdir,
    ):
        if isinstance(args, str):
            self.args = shlex.split(args)
        else:
            self.args = list(args)
        self.args[0] = os.path.abspath(self.args[0])
        self.tasks = int(tasks)
        self.cores_per_task = int(cores_per_task)
        self.gpus_per_task = int(gpus_per_task)
        self.timeout = int(timeout)
        self.batch_script = bool(batch_script)
        self.cwd = cwd if isinstance(cwd, str) else str(cwd)

    def __repr__(self):
        """Return string representation of constructor call for this object"""
        return (
            "{cls}(args={args}, tasks={tasks}, cores_per_task={cores}, "
            "gpus_per_task={gpus}, timeout={timeout}, batch_script={bscript}, "
            "cwd={cwd})"
        ).format(
            cls=type(self).__name__,
            args=self.args,
            tasks=self.tasks,
            cores=self.cores_per_task,
            gpus=self.gpus_per_task,
            timeout=self.timeout,
            bscript=self.batch_script,
            cwd=self.cwd,
        )

    @staticmethod
    def encode(obj):
        """JSON Encoder for Step objects."""
        if isinstance(obj, Step):
            return {key: getattr(obj, key) for key in obj.__slots__}
        raise TypeError()


class CompositeRun(object):  # pylint: disable=too-few-public-methods
    """Instances of this class represent one run of a Themis ensemble.

    ``CompositeRun`` instances are the most general kind of run in an ensemble, and
    consist of two parts: a mapping defining arbitrary key-value pairs, and a sequence
    of ``Step`` objects.

    Instances do not do anything themselves; they are meant to be passed into
    other methods and functions.

    Many attributes map directly to ``lrun``, ``srun``, and ``flux mini run``
    arguments.

    :param sample: a mapping from sample labels to values, such as one produced
        by iterating through a ``Samples`` object.
        Generally, sample labels (the keys of the mapping) should be constant
        across all runs. The sample will be used for parsing
        :ref:`text files <text_replacement>`.
    :type sample: Mapping
    :param steps: nonempty sequence of ``themis.Step`` objects defining the
        execution instructions for the run. Each ``Step``
        object owned by the run will be executed in order: first ``steps[0]``,
        then ``steps[1]``, then ``steps[2]`` and so on.
    :type steps: Sequence
    """

    __slots__ = ("sample", "steps")

    def __init__(self, sample, steps):
        self.sample = sample if sample is not None else {}
        self.steps = steps
        if not isinstance(steps, Sequence):
            raise TypeError(
                "Expected sequence type for steps, got {!r}".format(
                    type(steps).__name__
                )
            )
        if not steps:
            raise ValueError("len(steps) must be > 0")


class Run(CompositeRun):  # pylint: disable=too-many-instance-attributes
    """Represents one simple run of a Themis ensemble.

    This class should be named ``SimpleRun`` but is not for backwards compatibility.

    The ``Run`` class is a restrictive, simplifying derivation of the
    ``CompositeRun`` class. ``Run`` instances support only a single ``Step``,
    i.e. only a single application makes up the run. Other limitations include:

    * All ``Run`` instances in an ensemble are assumed to share the same application.
      As a consequence, whether or not the application is a batch script is set on an
      ensemble-wide basis.
    * The working directory for each ``Run`` in an ensemble is assumed to be specified
      by the ``run_dir_names`` ensemble-wide constant.

    Instances do not do anything themselves; they are meant to be passed into
    other methods and functions.

    Many attributes map directly to ``lrun``, ``srun``, and ``flux mini run``
    arguments.

    :param sample: a mapping from sample labels to values, such as one produced
        by iterating through a ``Samples`` object.
        Generally, sample labels (the keys of the mapping) should be constant
        across all runs. The sample will be used for parsing
        :ref:`text files <text_replacement>`.
    :param args: the command-line arguments to pass to the application.
    :type args: str, optional
    :param tasks: total MPI tasks. Corresponds to srun, lrun, and flux "-n" options.
    :type tasks: int, optional
    :param cores_per_task: cores per task. Corresponds to srun,
        lrun, and flux "-c" options.
    :type cores_per_task: int, optional
    :param gpus_per_task: gpus per task. Corresponds to lrun and flux "-g" option.
    :type gpus_per_task: int, optional
    :param timeout: the max running time, in minutes, for this run; if the limit is
        exceeded, the run will be killed.
    :type timeout: int, optional
    """

    __slots__ = (
        "sample",
        "args",
        "tasks",
        "cores_per_task",
        "gpus_per_task",
        "timeout",
        "application",
        "app_is_batch_script",
        "_steps",
    )

    def __init__(  # pylint: disable=too-many-arguments,super-init-not-called
        self,
        sample=None,
        args=None,
        tasks=1,
        cores_per_task=1,
        gpus_per_task=0,
        timeout=0,
    ):
        self.sample = sample
        self.args = args
        self.tasks = tasks
        self.cores_per_task = cores_per_task
        self.gpus_per_task = gpus_per_task
        self.timeout = timeout
        self.application = None  # set by Themis instance
        self.app_is_batch_script = True
        self._steps = None

    def __repr__(self):
        """Return string representation of constructor call for this object"""
        return (
            "{}(sample={}, args={!r}, tasks={}, cores_per_task={}, "
            "gpus_per_task={}, timeout={})"
        ).format(
            type(self).__name__,
            self.sample,
            self.args,
            self.tasks,
            self.cores_per_task,
            self.gpus_per_task,
            self.timeout,
        )

    @property
    def steps(self):
        """Dynamically construct the steps making up this run.

        A horrible hack, this requires that the owning themis.Themis
        set ``self.application`` so that ``args[0]`` can be the path
        to the application when the ``Step`` is constructed.
        """
        if self._steps is not None:
            return self._steps
        arguments = [self.application]
        if self.args is not None:
            arguments.extend(shlex.split(self.args))
        return (
            Step(
                arguments,
                self.tasks,
                self.cores_per_task,
                self.gpus_per_task,
                self.timeout,
                self.app_is_batch_script,
            ),
        )

    @steps.setter
    def steps(self, new_steps):
        self._steps = new_steps


class AugmentedRun(CompositeRun):  # pylint: disable=too-few-public-methods
    """A Run object used by Themis's backend with additional fields."""

    __slots__ = CompositeRun.__slots__ + ("status", "result")

    def __init__(self, *args, **kwargs):
        self.status = kwargs.pop("status", None)
        self.result = kwargs.pop("result", None)
        super(AugmentedRun, self).__init__(*args, **kwargs)


def convert_none(value, none_replacement):
    """If value is None, return the replacement value."""
    if value is None:
        return none_replacement
    return value


def validate_application(application):
    """Validate the application; raise a ValueError if it isn't found.

    If application isn't an absolute or relative path, search for it along
    the PATH environment variable.
    """
    if not os.path.exists(application):
        found_application = which(application)
        if found_application is None:
            raise ValueError(
                "{} is not a valid application: it either does not exist"
                " or is not executable.".format(application)
            )
        return found_application
    if not os.path.isfile(application) or not os.access(application, os.X_OK):
        raise ValueError(
            "{} is not a valid application: it either does not exist"
            " or is not executable.".format(application)
        )
    return os.path.abspath(application)


def which(program):
    """Search for `program` along the PATH environment variable."""
    try:
        # works in Python 3; not in 2
        return shutil.which(program)
    except AttributeError:
        pass
    path_var = os.getenv("PATH")
    for path_entry in path_var.split(os.path.pathsep):
        potential_match = os.path.join(path_entry, program)
        if os.path.isfile(potential_match) and os.access(potential_match, os.X_OK):
            return potential_match
    return None


def existing_file(file_path):
    """Check that `file_path` exists; raise a ValueError if it doesn't."""
    if not os.path.isfile(file_path):
        if os.path.isdir(file_path):
            raise ValueError(
                "{!r} is a directory, but a file is required".format(file_path)
            )
        raise ValueError("{!r} is not an existing file".format(file_path))
    return os.path.abspath(file_path)


def match_path(path_pattern, no_dirs):
    """Return files matching `path_pattern`; raise a ValueError if no matches."""
    if no_dirs:
        path_func = existing_file
    else:
        path_func = os.path.abspath
    pattern_matched = glob.glob(path_pattern)
    if not pattern_matched:
        raise ValueError("{!r} does not match anything.".format(path_pattern))
    return [path_func(match) for match in pattern_matched]


def sequence_of_path_patterns(path_patterns, no_dirs=False):
    """Expand an iterable of unix-style path patterns into a sequence of actual paths.

    :param no_dirs: if `True`, raise an error if any of the matches is a directory

    :raises ValueError: if any of the path patterns match no files.
    """
    if path_patterns is None:
        path_patterns = ()
    elif isinstance(path_patterns, str):
        path_patterns = (path_patterns,)
    expanded_paths = set()
    for path in path_patterns:
        for match in match_path(path, no_dirs):
            expanded_paths.add(match)
    return tuple(expanded_paths)


def validate_run_files(input_decks, run_copy, run_symlink):
    """Validate the 3 sets of files/paths."""
    input_decks = set(sequence_of_path_patterns(input_decks, True))
    run_copy = set(sequence_of_path_patterns(run_copy))
    run_symlink = set(sequence_of_path_patterns(run_symlink))
    remove_overlap(
        input_decks, "run_parse", (run_copy, "run_copy"), (run_symlink, "run_symlink")
    )
    remove_overlap(run_copy, "run_copy", (run_symlink, "run_symlink"))
    return (list(input_decks), list(run_copy), list(run_symlink))


def remove_overlap(high_priority_set, high_priority_name, *lower_priority_tuples):
    """Remove the overlap (intersection) between two or more sets."""
    for low_priority_set, low_priority_name in lower_priority_tuples:
        if high_priority_set & low_priority_set:
            warnings.warn(
                "\n{high_priority} and {low_priority} should be distinct, but:\n"
                "{overlap}\nare common to both. "
                "Giving priority to {high_priority}...".format(
                    high_priority=high_priority_name,
                    low_priority=low_priority_name,
                    overlap="\n".join(high_priority_set & low_priority_set),
                ),
                stacklevel=3,
            )
        low_priority_set -= high_priority_set


def type_check(value, *types_to_check):
    """Raise a ValueError if `value` isn't one of `types_to_check`"""
    if not isinstance(value, types_to_check):
        raise TypeError(
            "Expected a type in {}, got {}".format(types_to_check, type(value))
        )
    return value


def range_check(value, min_val=None, max_val=None, name="value"):
    """Check that a variable falls within a range, then return it.

    :param min_val: the minimum acceptable value, or None for no minimum.
    :param min_val: the maximum acceptable value, or None for no maximum.
    :param name: the name of the variable to check. Used for error messages.
    """
    if min_val is not None and value < min_val:
        raise ValueError("{} should be at least {}".format(name, min_val))
    if max_val is not None and value > max_val:
        raise ValueError("{} should be at most {}".format(name, max_val))
    return value


class DirectoryManager(object):
    """Context manager for changing the working directory."""

    def __init__(self, new_directory):
        """Constructor. Saves the name of the directory to change to."""
        self.new_directory = new_directory
        self.old_directory = os.getcwd()

    def __enter__(self):
        """Change to the new working directory."""
        os.chdir(self.new_directory)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Return to the original working directory.

        Return False, so the interpreter will re-raise any exceptions
        in the managed block.
        """
        os.chdir(self.old_directory)
        return False


def print_progress_bar(
    iteration, total, suffix="", decimals=1, length=100,
):
    """
    Create a terminal progress bar

    :param iteration: current iteration (int)
    :param total: total iterations (int)
    :param suffix: suffix for the progress bar
    :param decimals: positive number of decimals in percent complete (int)
    :param length: character length of bar (int)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    for fill_char in ("█", "#"):  # use an ascii character for ascii-only consoles
        progress_bar = fill_char * filled_length + "-" * (length - filled_length)
        try:
            print(
                "|{}| {}% {} ({}/{})".format(
                    progress_bar, percent, suffix, iteration, total
                )
            )
        except UnicodeEncodeError:
            pass
        else:
            break


def import_app_interface(app_interface_path):
    """Import the user's application interface module and return it.

    Don't suppress any exceptions raised by the import.
    """
    if app_interface_path is not None:
        module_name = (
            "app_interface_" + os.path.splitext(os.path.basename(app_interface_path))[0]
        )
        try:
            import importlib.util
        except ImportError:
            import imp

            app_interface = imp.load_source(module_name, app_interface_path)
        else:
            module_spec = importlib.util.spec_from_file_location(
                module_name, app_interface_path
            )
            app_interface = importlib.util.module_from_spec(module_spec)
            module_spec.loader.exec_module(app_interface)
    else:
        app_interface = None
    return app_interface


def get_run_dir(run_id, run_dir_names, sample):
    """Return the run directory for the run given by run_id.

    The default run directory is generated by grouping runs into slots of 10,000
        and then naming the run directory by its run_id.

    :param run_dir_names: A format string that, when formatted by sample, should
        yield the run directory, or None to use the default run directory.
    :param setup_dir: a directory path, which the run directory is relative to
    :param sample: the sample used to format run_dir_names, if run_dir_names is not None
    """
    run_min = (run_id // 10000) * 10000
    run_max = ((run_id // 10000) + 1) * 10000 - 1
    return run_dir_names.format(
        run_id=run_id, run_id_min=run_min, run_id_max=run_max, **sample
    )


def _validate_run_dirs(samples, run_dir_names):
    """Check that the samples will produce unique run directories."""
    set_of_run_dir_names = set()
    for sham_run_id, sample in enumerate(samples):
        try:
            set_of_run_dir_names.add(get_run_dir(sham_run_id, run_dir_names, sample))
        except KeyError:
            raise ValueError(
                "The samples do not have keys matching the run_dir_names format string"
            )
    if len(samples) > len(set_of_run_dir_names):
        raise ValueError(
            "That combination of samples and run_dir_names will "
            "not produce unique run directories."
        )


def validate_runs(runs, run_dir_names=None, application=None, batch_script=True):
    """Validate the type and value of runs and return standardized form."""
    run_dir_names = DEFAULT_RUN_DIR_NAMES if run_dir_names is None else run_dir_names
    runs = list(runs) if not isinstance(runs, Sequence) else runs
    for run in runs:
        # add missing data to the outdated ``Run`` class
        if isinstance(run, Run):
            run.application = application
            run.app_is_batch_script = batch_script
    # validate a reasonable number of run dirs
    _validate_run_dirs([run.sample for run in runs[:5000]], run_dir_names)
    return runs


def validate_samples(samples, run_dir_names=None, check_nonempty=False):
    """Validate the type and value of samples and return standardized form."""
    run_dir_names = DEFAULT_RUN_DIR_NAMES if run_dir_names is None else run_dir_names
    samples = list(samples) if not isinstance(samples, Sequence) else samples
    if not samples and check_nonempty:
        raise ValueError("There are no samples")
    _validate_run_dirs(samples[:5000], run_dir_names)
    return samples


def get_application(application, run_dir, app_is_batch_script):
    """Get the path to the application to the application to execute.

    Usually nothing needs to be done. However, if the application is
    a batch script, it was parsed and copied into the run directory.
    """
    if app_is_batch_script:
        return os.path.join(run_dir, os.path.basename(application))
    return application


class CleanDirectory(object):
    """Context manager for changing the working directory to a new, clean one."""

    def __init__(self, new_directory):
        """Constructor. Saves the name of the directory to change to."""
        self.new_directory = new_directory
        self.old_directory = os.getcwd()

    def go_to_new(self):
        """Create or clear the new directory, and change into it."""
        if os.path.exists(self.new_directory):
            shutil.rmtree(self.new_directory, ignore_errors=True)
        if not os.path.exists(self.new_directory):
            os.mkdir(self.new_directory)
        os.chdir(self.new_directory)
        return self.old_directory

    def go_to_old(self):
        """Return to the old working directory."""
        os.chdir(self.old_directory)

    def __enter__(self):
        """Change to the new working directory and empty it.

        Return the path of the original working directory.
        """
        return self.go_to_new()

    def __exit__(self, exc_type, exc_value, traceback):
        """Return to the original working directory.

        Return False, so the interpreter will re-raise any exceptions
        in the managed block.
        """
        self.go_to_old()
        return False


def clean_directory_decorator(new_directory=None):
    """Wrap a function in a CleanDirectory context.

    If the name of a new directory is not given,
    use the name of the passed function.
    """

    def decorator(func):
        dir_context = new_directory or func.__name__

        @functools.wraps(func)
        def nested(*args, **kwargs):
            with CleanDirectory(dir_context):
                return func(*args, **kwargs)

        return nested

    return decorator


def _alarm_exception(signum, frame):
    """Signal handler function."""
    raise OSError("Alarm handler called")


def timeout_decorator(timeout):
    """Raise a TimeoutError if a function takes more than `timeout` seconds."""

    def decorator(func):
        """Wrap a function in calls to SIGALRM."""

        @functools.wraps(func)
        def nested(*args, **kwargs):
            """Request an alarm, call a function, disable the alarm."""
            signal.signal(signal.SIGALRM, _alarm_exception)
            signal.alarm(timeout)
            try:
                return_val = func(*args, **kwargs)
            finally:
                signal.alarm(0)  # disable the alarm, the function finished
            return return_val

        return nested

    return decorator


def add_parameterfile_arg(parser, nargs=None):
    """Add a parameterfile argument to an `argparse.ArgumentParser`."""
    parser.add_argument(
        "parameterfile",
        type=str,
        help=(
            "Path to the file defining the parameters for each run."
            " The file is assumed to be a CSV with a header row."
        ),
        nargs=nargs,
    )


def add_common_creation_args(parser):
    """Add argparse.ArgumentParser arguments common to __main__.py and laf.py"""
    parser.add_argument(
        "-r",
        "--run-dir-names",
        type=str,
        help=(
            "The Python format string yielding the run directories. "
            "Default behavior is to put each run into `runs/####`."
        ),
        metavar="FORMATSTRING",
    )
    parser.add_argument(
        "--parse",
        nargs="+",
        type=str,
        help=(
            "Paths to text files to parse and copy into each run directory. "
            "The files are parsed for variables declared with '%%%%'. "
            "For example, '%%%%my_sample_1%%%%' will be replaced with the value "
            "of 'my_sample_1' for each run."
        ),
        metavar="FILE",
        dest="run_parse",
    )
    parser.add_argument(
        "--symlink",
        nargs="+",
        type=str,
        help=(
            "Paths to files/directories to symlink into each run directory. "
            "E.g. `--symlink *` will symlink everything in the current working "
            "directory. Default behavior is not to symlink anything. "
        ),
        metavar="PATH",
        dest="run_symlink",
    )
    parser.add_argument(
        "--copy",
        nargs="+",
        type=str,
        help=(
            "Paths to files/directories to hard-copy into each run directory. "
            "E.g. `--copy *` will copy everything in the current working directory. "
            "Default behavior is not to copy anything. "
        ),
        metavar="PATH",
        dest="run_copy",
    )


def read_csv(path):
    """Read a CSV and return an iterable of dictionaries representing the rows."""
    with open(path, **csvargs) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            yield row

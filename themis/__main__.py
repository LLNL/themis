"""This module provides a command-line interface to the ensemble themis."""
# pylint: disable=too-many-lines

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import argparse
import sys
import os
import json
import traceback


if __name__ == "__main__":
    sys.path.append(os.path.abspath(__file__).rsplit(os.sep, 2)[0])


# pylint: disable=wrong-import-position
import themis
from themis import utils
from themis.versions import reprlib
from themis import database
from themis import runtime


def get_themis(args):
    """Get the Themis object from the given setup directory or exit."""
    try:
        return themis.Themis(args.setup_dir)
    except ValueError as val_err:
        sys.exit("Invalid value given for the `--setup-dir` argument: " + str(val_err))


def add_subparser_aliases(subparsers, aliases, *args, **kwargs):
    """Add aliases to subcommands."""
    if sys.version_info.major > 2:
        return subparsers.add_parser(*args, aliases=aliases, **kwargs)
    return subparsers.add_parser(*args, **kwargs)


def add_verbose(parser):
    """Add a verbose argument to a parser."""
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Write more information to stdout",
    )


def add_parallel(parser):
    """Add a parallel argument to a parser."""
    parser.add_argument(
        "--parallel",
        type=int,
        default=None,
        help="reserve N cores for Themis to run in parallel on",
        metavar="N",
    )


def add_multiple_option(parser):
    """Add a multiple argument to a parser."""
    parser.add_argument(
        "--allow-multiple",
        action="store_true",
        help=(
            "Allow multiple concurrent executions. "
            "This flag must be set on all concurrent executions."
        ),
    )


def composite_create_common_args(parser):
    """Arguments common to 'create' and 'create-composite'."""
    parser.add_argument(
        "--flux",
        nargs="?",
        const=True,
        default=False,
        help=(
            "Use the Flux resource manager, optionally passing "
            "a path to the installation to use"
        ),
        metavar="FLUX_PATH",
    )
    parser.add_argument(
        "--max-restarts",
        type=int,
        default=0,
        metavar="N",
        help=(
            "The number of times to restart a run upon failure "
            "(i.e. a nonzero returncode)."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=("Create a new Themis ensemble, force-removing any existing one"),
    )
    parser.add_argument("--interface", help=argparse.SUPPRESS)
    utils.add_common_creation_args(parser)
    parser.add_argument(
        "--abort-on",
        nargs="+",
        type=int,
        help=(
            "One or more OS process return codes (positive integers). "
            "If `application` exits with one of these return codes, "
            "the run will be marked as 'aborted' and will not be restarted."
        ),
    )


def handle_status_subcommand(args):
    """Display the run IDs of runs with certain statuses."""
    themis_obj = get_themis(args)
    status_enums = {themis_obj.translate_enum(name) for name in args.status_names}
    if args.count:
        print(themis_obj.count_by_status(*status_enums))
    else:
        # print the run IDs by adding a space between every element
        print(" ".join((str(i) for i in themis_obj.filter_by_status(*status_enums))))


def setup_status_parser(subparsers):
    """Setup the parser for the status subcommand."""
    status_choices = [
        themis.Themis.translate_enum(enum) for enum in themis.Themis.STATUS_ENUMS
    ]
    status_choices.append([])
    status_parser = subparsers.add_parser(
        "status",
        help="Filter runs by status",
        description=(
            "Filter runs by status; return the IDs of runs with the "
            "specified statuses. If no statuses are specified, return the IDs "
            "of all runs."
        ),
    )
    status_parser.add_argument(
        "status_names",
        choices=status_choices,
        nargs="*",
        type=lambda x: str(x).lower(),
        help="The status filters to apply",
    )
    status_parser.add_argument(
        "-c",
        "--count",
        action="store_true",
        help="count the number of runs instead of returning their ids",
    )
    status_parser.set_defaults(handler=handle_status_subcommand)


def handle_kill_subcommand(args):
    """Kill runs."""
    themis_obj = get_themis(args)
    themis_obj.kill_runs(args.run_ids)


def setup_kill_parser(subparsers):
    """Setup the parser for the kill subcommand."""
    kill_parser = add_subparser_aliases(
        subparsers,
        ["dequeue"],
        "kill",
        help="Remove runs from the queue",
        description=(
            "Remove runs from the queue, so they will not be executed. "
            "Has no effect on completed runs."
        ),
    )
    kill_parser.add_argument(
        "run_ids", type=int, nargs="+", help="ID of a run to kill", metavar="ID"
    )
    kill_parser.set_defaults(handler=handle_kill_subcommand)


def _repr_unknown(repr_instance, obj, maxsize):
    """Compactly repr an unknown, possibly large and nested, object."""
    old_maxstring = repr_instance.maxstring
    repr_instance.maxstring = maxsize
    return_val = repr_instance.repr(repr_instance.repr(obj)).strip("\"'")
    repr_instance.maxstring = old_maxstring
    return return_val


def _repr_str(repr_instance, str_to_repr, maxsize):
    old_maxstring = repr_instance.maxstring
    repr_instance.maxstring = maxsize
    return_val = repr_instance.repr(str_to_repr).strip("\"'")
    repr_instance.maxstring = old_maxstring
    return return_val


def handle_display_subcommand(args):  # pylint: disable=too-many-locals
    """Print a table of data about the ensemble.

    Certain columns are only displayed optionally.
    """
    themis_obj = get_themis(args)
    id_args_repr = reprlib.Repr()
    id_args_repr.maxstring = 15
    id_args_repr.maxlong = 9
    resource_repr = reprlib.Repr()
    resource_repr.maxlong = 5
    custom_repr = reprlib.Repr()
    run_dicts = themis_obj.runs(range(args.lowerbound, args.upperbound))
    sample_header = None
    sample_cols = ()
    run_dir_names = database.get_app_spec(themis_obj.setup_dir)["run_dir_names"]
    if not run_dicts:
        print("No runs found in that range")
        return
    if args.all:
        args.directory = max(args.directory, args.all)
        args.arguments = max(args.arguments, args.all)
        args.sample = max(args.sample, args.all)
        args.result = max(args.result, args.all)
    format_string = "|{run_id:^7}|{step:^6}|{status:^12}|"
    if args.directory:
        # ensure column has at least minimum width
        args.directory = max(args.directory, len("Directory"))
        format_string += "{cwd:^" + str(args.directory) + "}|"
    if args.arguments:
        # ensure column has at least minimum width
        args.arguments = max(args.arguments, len("Args"))
        format_string += "{args:^" + str(args.arguments) + "}|"
    if args.resources or args.all:
        format_string += "{tasks:^5}|{cores:^5}|{gpus:^4}|"
    if args.sample:
        # ensure column has at least minimum width
        args.sample = max(args.sample, 5)
        # get the first sample and use its keys as the columns
        sample_cols = sorted(next(iter(run_dicts.values())).sample.keys(), key=str)[
            : args.colcount
        ]
        format_string += "{sample}"
        sample_header = "".join(
            [
                ("{:^" + str(args.sample) + "}|").format(
                    _repr_unknown(custom_repr, col, args.sample)
                )
                for col in sample_cols
            ]
        )
    if args.result:
        # ensure column has at least minimum width
        args.result = max(args.result, len("Result"))
        format_string += "{result:^" + str(args.result) + "}|"
    header_row = format_string.format(
        run_id="Run",
        status="Status",
        step="Step",
        cwd="Directory",
        args="Args",
        tasks="Tasks",
        cores="Cores",
        gpus="GPUs",
        sample=sample_header,
        result="Result",
    )
    line_row = len(header_row) * "-"
    print("\n" + line_row + "\n" + header_row + "\n" + line_row)
    for run_id, run in run_dicts.items():
        for i, step in enumerate(run.steps):
            if i == 0:
                status = themis_obj.translate_enum(run.status)
                result = _repr_unknown(id_args_repr, run.result, args.result)
                run_id_display = id_args_repr.repr(run_id)
                get_sample_val = lambda sample, key: _repr_unknown(
                    custom_repr, sample.get(key, None), args.sample
                )
            else:
                run_id_display = status = result = ""
                get_sample_val = lambda _1, _2: ""  # return empty string
            print(
                format_string.format(
                    step=i,
                    run_id=run_id_display,
                    cwd=_repr_str(
                        custom_repr,
                        os.path.relpath(
                            utils.get_run_dir(run_id, run_dir_names, run.sample)
                        ),
                        args.directory,
                    ),
                    args=_repr_unknown(custom_repr, step.args, args.arguments),
                    tasks=resource_repr.repr(step.tasks),
                    cores=resource_repr.repr(step.cores_per_task * step.tasks),
                    gpus=resource_repr.repr(step.gpus_per_task * step.tasks),
                    sample="".join(
                        [
                            ("{:^" + str(args.sample) + "}|").format(
                                get_sample_val(run.sample, key)
                            )
                            for key in sample_cols
                        ]
                    ),
                    status=status,
                    result=result,
                )
            )
    print(line_row + "\n")


def setup_display_parser(subparsers):
    """Setup the parser for the display subcommand."""
    display_parser = subparsers.add_parser(
        "display",
        help="Print information about runs",
        description=(
            "Print a table of information about runs. By default, only prints out "
            "the status of each run. Some entries in the table will be cut down if "
            "they become too long; use the `write` subcommand "
            "for a complete representation."
        ),
    )
    display_parser.add_argument(
        "lowerbound",
        type=int,
        help="The lower bound of the runs to display, inclusive",
    )
    display_parser.add_argument(
        "upperbound",
        type=int,
        help="The upper bound of the runs to display, exclusive",
    )
    display_parser.add_argument(
        "-a",
        "--arguments",
        nargs="?",
        type=int,
        default=0,
        const=15,
        help=(
            "Display (with a given column width) the command-line "
            "arguments for each run"
        ),
    )
    display_parser.add_argument(
        "-r",
        "--result",
        nargs="?",
        type=int,
        default=0,
        const=15,
        help="Display (with a given column width) the result for each run",
    )
    display_parser.add_argument(
        "-t",
        "--resources",
        action="store_true",
        help=(
            "Display the resources for each run, in terms of total tasks, cores, "
            "and gpus"
        ),
    )
    display_parser.add_argument(
        "-s",
        "--sample",
        nargs="?",
        type=int,
        default=0,
        const=12,
        help=(
            "Display (with a given column width) the sample for each run. "
            "Use the `-c` option to change the number of columns displayed."
        ),
    )
    display_parser.add_argument(
        "-c",
        "--colcount",
        type=int,
        default=4,
        help="Set the max number of sample columns to display. Defaults to 4.",
        metavar="N",
    )
    display_parser.add_argument(
        "-d",
        "--directory",
        help="Display (with a given column width) the working directory for each run",
        nargs="?",
        type=int,
        default=0,
        const=15,
    )
    display_parser.add_argument(
        "--all",
        nargs="?",
        type=int,
        default=0,
        const=15,
        help="Display all the available information about each run",
    )
    display_parser.set_defaults(handler=handle_display_subcommand)


def handle_post_subcommand(args):
    """Call the `themis.Themis.call_post_run` method."""
    themis_obj = get_themis(args)
    restart = themis.Themis(themis_obj.setup_dir)
    if not args.run_ids:
        args.run_ids = themis_obj.filter_by_status()
    for run_id in args.run_ids:
        post_run_result = restart.call_post_run(run_id)
        if args.verbose > 0:
            print("Result of post_run {}: {}".format(run_id, post_run_result))
        if args.save:
            themis_obj.set_result(run_id, post_run_result)


def setup_post_parser(subparsers):
    """Setup the parser for the post subcommand."""
    post_parser = subparsers.add_parser(
        "post",
        help="Call the `post_run` application interface function",
        description=(
            "Call the `post_run` application interface function for a number of runs. "
            "Has no effect if the `post_run` function is not defined. "
            "Unless specific run IDs are passed, call `post_run` for the whole "
            "ensemble."
        ),
    )
    post_parser.add_argument(
        "run_ids",
        nargs="*",
        type=int,
        help="the run IDs of the runs to call `post_run` for",
        metavar="ID",
    )
    post_parser.add_argument(
        "-s",
        "--save",
        action="store_true",
        help=(
            "Save the result internally, overwriting the result currently stored "
            "for the run. The new result will appear when fetching data about the "
            "ensemble."
        ),
    )
    add_verbose(post_parser)
    post_parser.set_defaults(handler=handle_post_subcommand)


def handle_dryrun_subcommand(args):
    """Call the `themis.Themis.dry_run` method."""
    themis_obj = get_themis(args)
    themis.Themis(themis_obj.setup_dir).dry_run(*args.run_ids, verbosity=args.verbose)


def setup_dryrun_parser(subparsers):
    """Setup the parser for the dryrun subcommand."""
    dryrun_parser = subparsers.add_parser(
        "dryrun",
        help="Set up runs, but do not execute",
        description=(
            "Dry-run an ensemble or specific runs within an ensemble. "
            "Create the runs' directories and populate them. "
            "Call the `prep_run` application interface function, if it exists. "
            "Unless specific run IDs are passed, dry run the whole ensemble. "
            "The dry-runs are done in serial, so it is safe to call this from "
            "the login node of a HPC cluster."
        ),
    )
    dryrun_parser.add_argument(
        "run_ids", nargs="*", type=int, help="run ID to dry-run", metavar="ID"
    )
    add_verbose(dryrun_parser)
    dryrun_parser.set_defaults(handler=handle_dryrun_subcommand)


def get_runs_from_parameterfile(args):
    """Return a list of Run objects as read from a CSV"""
    if args.parameterfile is None:
        return None
    samples = utils.read_csv(args.parameterfile)
    if args.vary_all:
        return [
            themis.Run(
                sample=sample,
                args=sample.get("args", args.args),
                tasks=sample.get("tasks", args.ntasks),
                cores_per_task=sample.get("cores_per_task", args.cores_per_task),
                gpus_per_task=sample.get("gpus_per_task", args.gpus_per_task),
            )
            for sample in samples
        ]
    return [
        themis.Run(
            sample=sample,
            args=args.args,
            tasks=args.ntasks,
            cores_per_task=args.cores_per_task,
            gpus_per_task=args.gpus_per_task,
        )
        for sample in samples
    ]


def add_related_parameterfile_args(parser):
    """Add arguments relating to

    Doesn't add the parameterfile argument itself.
    """
    parser.add_argument(
        "-a",
        "--args",
        type=str,
        default=None,
        help="Command-line arguments to pass to the application",
    )
    parser.add_argument(
        "-n", "--ntasks", type=int, default=1, help="MPI tasks per run", metavar="N"
    )
    parser.add_argument(
        "-c",
        "--cores-per-task",
        type=int,
        default=1,
        help="Cores per MPI task",
        metavar="N",
    )
    parser.add_argument(
        "-g",
        "--gpus-per-task",
        type=int,
        default=0,
        help="GPUs per MPI task",
        metavar="N",
    )
    parser.add_argument(
        "--vary-all",
        action="store_true",
        help=(
            "Look for `args`, `tasks`, `cores_per_task`, and `gpus_per_task` entries "
            "in `parameterfile` and use those values instead of `--tasks`, `--args`, "
            "etc."
        ),
    )


def create_ensemble(args, runs):
    """Create an ensemble given runs and args."""
    if themis.Themis.exists(args.setup_dir) and not args.overwrite:
        print(
            "Themis setup exists in {!r} and `--overwrite` flag not given, "
            "doing nothing...".format(args.setup_dir)
        )
    else:
        themis.Themis.create_overwrite(
            application=args.application,
            runs=runs,
            setup_dir=args.setup_dir,
            run_parse=args.run_parse,
            run_symlink=args.run_symlink,
            run_copy=args.run_copy,
            run_dir_names=args.run_dir_names,
            app_is_batch_script=not args.no_batch_script,
            use_flux=args.flux,
            max_restarts=args.max_restarts,
            app_interface=args.interface,
            abort_on=args.abort_on,
        )


def handle_create_subcommand(args):
    """Create a new ensemble from an application and a samples file."""
    runs = get_runs_from_parameterfile(args)
    create_ensemble(args, runs)


def setup_create_parser(subparsers):
    """Setup the parser for the create subcommand."""
    create_parser = subparsers.add_parser(
        "create",
        help="Create a new Themis ensemble",
        description=(
            "Create a new Themis ensemble from an application and a file defining the "
            "parameters for each run. Resources and command-line arguments are "
            "assumed to be constant across runs unless `--vary-all` is set. "
            "Setup files are put into the `--setup-dir` directory. "
            "If there are already setup files in `--setup-dir`, nothing is done "
            "unless the `--overwrite` flag is given."
        ),
    )
    create_parser.add_argument(
        "application", type=str, help="Path to the application to execute in parallel"
    )
    create_parser.add_argument(
        "--no-batch-script",
        action="store_true",
        help=(
            "Add this flag if `application` is NOT a batch "
            "script or any other sort of script that launches other "
            "applications."
        ),
    )
    composite_create_common_args(create_parser)
    add_related_parameterfile_args(create_parser)
    utils.add_parameterfile_arg(create_parser, "?")
    create_parser.set_defaults(handler=handle_create_subcommand)


def build_composite_runs(parameterfile, stepfile):
    """Build composite runs from two CSVs."""
    samples = utils.read_csv(parameterfile)
    steps = [themis.Step(**step) for step in utils.read_csv(stepfile)]
    return [themis.CompositeRun(sample, steps) for sample in samples]


def add_stepfile_arg(parser):
    """Add a stepfile argument to a parser."""
    parser.add_argument(
        "stepfile",
        help=(
            "Path to the CSV file defining the steps for each run. "
            "Each row must define a single step. "
            "The accepted columns (you may specify only a subset) are "
            "args, tasks, cores_per_task, gpus_per_task, timeout, "
            "batch_script, and cwd."
        ),
    )


def handle_create_comp_subcommand(args):
    """Create a new ensemble of composite runs."""
    runs = build_composite_runs(args.parameterfile, args.stepfile)
    args.no_batch_script = False
    args.application = None
    create_ensemble(args, runs)


def setup_create_composite_parser(subparsers):
    """Setup the parser for the create-composite subcommand."""
    create_parser = subparsers.add_parser(
        "create-composite",
        help="Create a new composite Themis ensemble",
        description=(
            "Create a new Themis ensemble from a file defining the samples for each "
            "run and a file defining the steps for each run. "
            "Steps are assumed to be constant across all runs. "
            "Setup files are put into the `--setup-dir` directory. "
            "If there are already setup files in `--setup-dir`, nothing is done "
            "unless the `--overwrite` flag is given."
        ),
    )
    composite_create_common_args(create_parser)
    utils.add_parameterfile_arg(create_parser, None)
    add_stepfile_arg(create_parser)
    create_parser.set_defaults(handler=handle_create_comp_subcommand)


def handle_add_subcommand(args):
    """Handle the `add_runs` subcommand."""
    themis_obj = get_themis(args)
    runs = get_runs_from_parameterfile(args)
    themis_obj.add_runs(runs)


def setup_add_parser(subparsers):
    """Add parser for add runs subcommand"""
    parser = subparsers.add_parser(
        "add",
        help="Add new runs",
        description=(
            "Add runs to an ensemble. Resources and command-line arguments are "
            "assumed to be constant across runs unless `--vary-all` is set. "
        ),
    )
    utils.add_parameterfile_arg(parser)
    add_related_parameterfile_args(parser)
    parser.set_defaults(handler=handle_add_subcommand)


def handle_add_composite_command(args):
    """Build composite runs and add them to the ensemble."""
    themis_obj = get_themis(args)
    runs = build_composite_runs(args.parameterfile, args.stepfile)
    themis_obj.add_runs(runs)


def setup_add_composite_parser(subparsers):
    """Add a parser for adding composite runs."""
    parser = subparsers.add_parser(
        "add-composite",
        help="Add new composite (multi-step) runs",
        description=(
            "Add composite (multi-step) runs to an ensemble. Steps are assumed "
            "to be constant across all runs."
        ),
    )
    utils.add_parameterfile_arg(parser)
    add_stepfile_arg(parser)
    parser.set_defaults(handler=handle_add_composite_command)


def handle_progress_subcommand(args):
    """Print a summary progress line."""
    themis_obj = get_themis(args)
    completed, total = themis_obj.progress()
    if args.verbose > 0:
        for enum in themis_obj.STATUS_ENUMS:
            translation = themis_obj.translate_enum(enum)
            print(
                translation.capitalize()
                + " runs: "
                + str(themis_obj.count_by_status(enum))
            )
    print()
    utils.print_progress_bar(completed, total, suffix="Complete", length=35)
    # print newline to clear, plus another newline to add an actual blank line
    # one newline is implicit in the call to print
    print("\n")


def setup_progress_parser(subparsers):
    """Setup the parser for the progress subcommand."""
    progress_parser = subparsers.add_parser(
        "progress",
        help="Print out a short progress summary",
        description=(
            "Print out a short summary of an ensemble's progress. "
            "Shows the percentage and number of completed runs."
        ),
    )
    add_verbose(progress_parser)
    progress_parser.set_defaults(handler=handle_progress_subcommand)


def handle_requeue_subcommand(args):
    """Restart runs."""
    get_themis(args).requeue_runs(args.run_ids, args.hard)


def setup_requeue_parser(subparsers):
    """Setup the parser for the restart subcommand."""
    requeue_parser = add_subparser_aliases(
        subparsers,
        ["requeue"],
        "restart",
        help="Mark runs as eligible for execution",
        description=(
            "Mark runs as eligible for restart. The specified runs will be given "
            " the 'queued' status and will be executed "
            "by Themis the next time it starts, "
            "or immediately if Themis is currently active."
        ),
    )
    requeue_parser.add_argument(
        "run_ids", type=int, nargs="+", help="ID of a run to restart", metavar="ID"
    )
    requeue_parser.add_argument(
        "--hard",
        action="store_true",
        help=(
            "If set, reset the runs back to step 0. "
            "Otherwise, leave the runs' progress unchanged."
        ),
    )
    requeue_parser.set_defaults(handler=handle_requeue_subcommand)


def handle_execute_alloc_subcommand(args):
    """Execute the ensemble and print the job ID."""
    themis_obj = get_themis(args)
    print(
        "Batch job ID is "
        + str(
            themis_obj.execute_alloc(
                nodes=args.nodes,
                timeout=args.timeout,
                partition=args.partition,
                bank=args.bank,
                name=args.name,
                repeats=args.repeats,
                parallelism=args.parallel,
                early_stop=args.early_stop,
                allow_multiple=args.allow_multiple,
            )
        )
    )


def setup_execute_alloc_parser(subparsers):
    """Setup the parser for the execute subcommand."""
    execute_parser = subparsers.add_parser(
        "execute-alloc",
        help="Execute an ensemble in a new batch allocation",
        description=(
            "Request a new batch allocation and launch an ensemble within it. "
            "Finish any incomplete runs. Already-completed runs will not be restarted."
        ),
    )
    execute_parser.add_argument(
        "-N", "--nodes", help="the number of nodes to request", type=int, default=1
    )
    execute_parser.add_argument(
        "-p",
        "--partition",
        help="the partition to request the allocation in",
        type=str,
        metavar="NAME",
        default=None,
    )
    execute_parser.add_argument(
        "-b",
        "--bank",
        help="the bank to charge the allocation to",
        type=str,
        metavar="NAME",
        default=None,
    )
    execute_parser.add_argument(
        "-m",
        "--name",
        help="the name to give the allocation",
        type=str,
        default="themis",
    )
    execute_parser.add_argument(
        "-t",
        "--timeout",
        help="the timeout on the allocation in minutes",
        type=int,
        default=60,
        metavar="N",
    )
    execute_parser.add_argument(
        "-r",
        "--repeats",
        help=(
            "The number of times to replicate the allocation if time expires but "
            "the ensemble is not yet complete. -1 allows infinite repeats."
        ),
        type=int,
        default=0,
        metavar="N",
    )
    execute_parser.add_argument(
        "--early-stop",
        help=(
            "The number of minutes before the allocation expires that Themis should "
            "stop submitting new jobs. Makes Themis's overall runtime equivalent to "
            "`--timeout` minus `--early-stop`. Useful if jobs pay attention to the "
            "remaining allocation time."
        ),
        type=int,
        default=0,
        metavar="N",
    )
    add_multiple_option(execute_parser)
    add_parallel(execute_parser)
    execute_parser.set_defaults(handler=handle_execute_alloc_subcommand)


def handle_execute_local_subcommand(args):
    """Handle execute-local subcommand."""
    get_themis(args).execute_local(
        args.block, parallelism=args.parallel, allow_multiple=args.allow_multiple
    )


def setup_execute_local_parser(subparsers):
    """Setup parser for the execute-local subcommand."""
    execute_parser = subparsers.add_parser(
        "execute-local",
        help="Execute an ensemble locally",
        description=(
            "Execute an ensemble locally (i.e. do not request an allocation); "
            "finish any incomplete runs. Already-completed runs will not be restarted."
        ),
    )
    execute_parser.add_argument(
        "-b",
        "--block",
        action="store_true",
        help=("Block and do not exit until the ensemble is complete."),
    )
    add_multiple_option(execute_parser)
    add_parallel(execute_parser)
    execute_parser.set_defaults(handler=handle_execute_local_subcommand)


def handle_write_subcommand(args):
    """Write data about the ensemble to a file."""
    themis_obj = get_themis(args)
    if args.format == "json":
        themis_obj.write_json(sys.stdout)
    elif args.format == "yaml":
        themis_obj.write_yaml(sys.stdout)
    elif args.format == "csv":
        themis_obj.write_csv(sys.stdout)


def setup_write_parser(subparsers):
    """Setup the parser for the write subcommand."""
    write_parser = subparsers.add_parser(
        "write",
        help="Write a complete set of data in a variety of formats",
        description=(
            "Write information about an ensemble, such as parameters, resources,"
            " and outputs, to a file in a variety of formats."
        ),
    )
    write_parser.add_argument(
        "format",
        help="the file format in which to write the data",
        choices=["yaml", "json", "csv"],
        type=lambda x: str(x).lower(),
    )
    write_parser.set_defaults(handler=handle_write_subcommand)


def handle_rundir_subcommand(args):
    """Handle rundir subcommand."""
    themis_obj = get_themis(args)
    run_dirs = themis_obj.run_dirs(args.nonexistent)
    if args.json:
        print(json.dumps(list(run_dir for _, run_dir in run_dirs)))
    else:
        for _, run_dir in run_dirs:
            print(run_dir)


def setup_rundir_parser(subparsers):
    """Setup parser for the rundirs subcommand."""
    rundir_parser = subparsers.add_parser(
        "rundirs",
        help="Print the working directory for each run",
        description="Print the working directory for each run",
    )
    rundir_parser.add_argument(
        "--json", action="store_true", help="Print the run directories as JSON"
    )
    rundir_parser.add_argument(
        "-n",
        "--nonexistent",
        action="store_true",
        help="Include directories which haven't yet been created",
    )
    rundir_parser.set_defaults(handler=handle_rundir_subcommand)


def handle_completion_subcommand(args):
    """Handle the completion subcommand."""
    get_themis(args).on_completion(
        args.args, stdout=args.stdout, stderr=args.stderr, cwd=args.cwd
    )


def setup_completion_parser(subparsers):
    """Setup parser for the completion subcommand."""
    completion_parser = subparsers.add_parser(
        "completion",
        help="Provide an executable for Themis to launch upon completion",
        description=(
            "Provide an executable and arguments for Themis to launch "
            "once the ensemble has completed. Only the most recent invocation "
            "of this subcommand is remembered; multiple executables are not "
            "supported."
        ),
    )
    completion_parser.add_argument(
        "args",
        help="The executable and any associated arguments.",
        nargs=argparse.REMAINDER,
    )
    completion_parser.add_argument(
        "--stdout",
        help=(
            "Path to redirect stdout. Default redirects "
            "stdout to one of Themis's log files."
        ),
    )
    completion_parser.add_argument(
        "--stderr",
        help=(
            "Path to redirect stderr. Default redirects "
            "stderr to one of Themis's log files."
        ),
    )
    completion_parser.add_argument(
        "--cwd",
        help=(
            "Working directory for the executable. "
            "Default is the current working directory."
        ),
        default=os.curdir,
    )
    completion_parser.set_defaults(handler=handle_completion_subcommand)


def handle_runtime_parse(args):
    """Handle `themis runtime parse` command."""
    runtime.parse_file(args.source, args.dest)


def handle_runtime_collect(args):
    """Handle `themis runtime collect` subcommand."""
    with open(args.source, "rb") as file_handle:
        runtime.set_result(file_handle.read())


def setup_runtime_parser(subparsers):
    """Setup the runtime subparsers."""
    runtime_parser = subparsers.add_parser(
        "runtime",
        help="Themis runtime operations",
        description=(
            "A set of utilities available to applications launched by Themis."
        ),
    )
    runtime_subparsers = runtime_parser.add_subparsers(
        title="subcommands", description="", dest="runtime_subcommand"
    )
    parse_subparser = runtime_subparsers.add_parser(
        "parse",
        help="Parse and token-replace a text file",
        description=(
            "Search a text file for Themis tokens and replace them "
            "with values associated with the current run."
        ),
    )
    parse_subparser.add_argument("source", help="The file to parse")
    parse_subparser.add_argument("dest", help="The destination of the parsed file")
    parse_subparser.set_defaults(handler=handle_runtime_parse)
    collect_subparser = runtime_subparsers.add_parser(
        "collect",
        help="Collect and store the contents of a file",
        description=(
            "Collect the contents of a file and store it as the "
            "result of the current run."
        ),
    )
    collect_subparser.add_argument("source", help="The file to collect and store")
    collect_subparser.set_defaults(handler=handle_runtime_collect)


def setup_parsers():
    """Return an argparse.ArgumentParser with all the options and subcommands."""
    parser = argparse.ArgumentParser(
        description=(
            "Interact with or create a Themis ensemble. Usage of this "
            "program is broken into subcommands; see the help message on "
            "each subcommand for specifics. All subcommands except `create` "
            "will fail if an existing Themis ensemble "
            "is not found. To create and run a new ensemble, use `create`, and then "
            "one of the `execute-` commands. The `dryrun` command is useful for "
            "checking that an ensemble is configured properly."
        )
    )
    parser.add_argument(
        "--setup-dir",
        default=utils.DEFAULT_SETUP_DIR,
        help=(
            "The path to the directory of the ensemble to interact with or create. "
            "Defaults to {0!r}".format(utils.DEFAULT_SETUP_DIR)
        ),
        metavar="DIR",
    )
    add_verbose(parser)
    parser.add_argument(
        "--version", action="version", version="%(prog)s " + str(themis.__version__)
    )
    subparsers = parser.add_subparsers(
        title="subcommands", description="", dest="subcommand"
    )
    setup_create_parser(subparsers)
    setup_create_composite_parser(subparsers)
    setup_dryrun_parser(subparsers)
    setup_execute_alloc_parser(subparsers)
    setup_execute_local_parser(subparsers)
    setup_add_parser(subparsers)
    setup_add_composite_parser(subparsers)
    setup_progress_parser(subparsers)
    setup_display_parser(subparsers)
    setup_status_parser(subparsers)
    setup_write_parser(subparsers)
    setup_kill_parser(subparsers)
    setup_requeue_parser(subparsers)
    setup_rundir_parser(subparsers)
    setup_post_parser(subparsers)
    setup_completion_parser(subparsers)
    setup_runtime_parser(subparsers)
    return parser


def main():
    """Run the command-line program."""
    parser = setup_parsers()
    args = parser.parse_args()
    if not hasattr(args, "handler"):  # no subcommand given
        parser.print_usage()
        return
    try:
        args.handler(args)
    except Exception as exc:  # pylint: disable=broad-except
        if args.verbose:
            traceback.print_exc()
            sys.exit(1)
        else:
            sys.exit(type(exc).__name__ + ": " + str(exc))


if __name__ == "__main__":
    main()

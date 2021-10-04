"""
This script prepares an individual application run for execution.

That consists of copying, symlinking, and parsing files, then
calling the user's prep_run function.
"""
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import os
import shutil
import re
import sys
import logging

try:
    import jinja2
except ImportError:
    JINJA2_AVAIL = False
else:
    JINJA2_AVAIL = True

if __name__ == "__main__":
    # when running as a script, sys.path needs to be set for imports
    sys.path.insert(0, os.path.abspath(__file__).rsplit(os.sep, 4)[0])

# pylint: disable=wrong-import-position
from themis.utils import DirectoryManager
from themis.backend.worker import utils
from themis import resource


def user_prep_run(run_dir, prep_run):
    """Call the user's prep_run function, if it exists, and return its return value."""
    if callable(prep_run):
        with DirectoryManager(run_dir):
            prep_run()


def remove(path):
    """Remove a file or directory at `path`"""
    if os.path.isfile(path):
        os.remove(path)
    elif os.path.isdir(path):
        shutil.rmtree(path)


def populate_run_dir(symlinks, copies, input_decks, sample, run_dir):
    """Create the run directory and populate it.

    The directory is populated with symlinks of every file/dir in `symlinks`,
    hard copies of every file/dir in `copies`,
    and _parsed_ hard copies of every file in input_decks.
    """
    # symlink required files into the new directory
    for source in symlinks:
        sym_target_path = os.path.join(run_dir, os.path.basename(source))
        # if a file exists with the same name as the new symlink, delete it
        if os.path.lexists(sym_target_path):
            os.remove(sym_target_path)
        os.symlink(source, sym_target_path)
    for source in copies:
        copy_dest = os.path.join(run_dir, os.path.basename(source))
        if os.path.isdir(source):
            if os.path.exists(copy_dest):
                remove(copy_dest)
            shutil.copytree(source, copy_dest, symlinks=True)
        else:
            shutil.copy(source, copy_dest)
    # hard copy input deck files into the new directory
    for source in input_decks:
        dest = os.path.join(run_dir, os.path.basename(source))
        # if copy_dest (the target of the new input deck) exists, overwrite it
        parse_input_deck(source, dest, sample)


def parse_input_deck(template, output_path, sample):
    """Check for jinja availability and forward on to the right function."""
    if JINJA2_AVAIL:
        parse_input_deck_jinja(template, output_path, sample)
    else:
        parse_input_deck_base(template, output_path, sample)
    shutil.copystat(template, output_path)


def _alternate_input_deck_syntax(line, sample):
    """Limited upport for old-style "UQP_VARIABLE" parsing.

    Takes a line like "var = ... # UQP_VARIABLE = foo", converts
    it to "var = %%foo%%", and return the modified line.
    """
    if "UQP_VARIABLE" not in line:
        return line
    pattern_replacements = [
        (r"=.*#\s*UQP_VARIABLE\s*=\s*{}".format(key), "= %%{}%%".format(key))
        for key in sample.keys()
    ]
    for pattern, replacement in pattern_replacements:
        line = re.sub(pattern, replacement, line)
    return line


def parse_input_deck_base(template, output_path, sample):
    """Parse a text file for variables declared with '%%' characters.

    Also supports old-style "UQP_VARIABLE" parsing.

    :param template: path to the template input deck to parse
    :param output_path: path to write the parsed template
    :param sample: mapping from variable names to values
    """
    if not sample:
        shutil.copyfile(template, output_path)
        return
    patterns_to_values = {"%%{}%%".format(key): str(val) for key, val in sample.items()}
    sample_regex = re.compile(
        "|".join(["({})".format(key) for key in patterns_to_values.keys()])
    )
    lines = []
    # collect and format lines in the template
    with open(template) as template_fd:
        for line in template_fd:
            line = _alternate_input_deck_syntax(line, sample)
            lines.append(
                sample_regex.sub(lambda match: patterns_to_values[match.group(0)], line)
            )
    # write formatted lines out to the target
    with open(output_path, "w") as output_fd:
        output_fd.writelines(lines)


def parse_input_deck_jinja(template_path, output_path, sample):
    """Parse a file using jinja templating."""
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(os.path.dirname(template_path)),
        autoescape=True,
        keep_trailing_newline=True,
        newline_sequence=os.linesep,
        variable_start_string="%%",
        variable_end_string="%%",
    )
    template = env.get_template(os.path.basename(template_path))
    with open(output_path, "w") as file_handle:
        template.stream(sample).dump(file_handle)


def preparation(prep_run, run_dir, app_spec, sample, steps):
    """Complete the preparation for a single run.

    Populate the run directory, and call user's prep_run function.
    """
    resource_mgr = resource.identify_resource_manager(app_spec["resource_mgr"])
    populate_run_dir(
        app_spec["run_symlink"],
        app_spec["run_copy"],
        app_spec["run_parse"],
        sample,
        run_dir,
    )
    for step in steps:
        if step.batch_script:
            step.batch_script = False  # set so build_cmd works
            sample["themis_launch"] = " ".join(resource_mgr.build_cmd(step))
            step.batch_script = True
            parse_input_deck(
                step.args[0],
                os.path.join(run_dir, os.path.basename(step.args[0])),
                sample,
            )

    user_prep_run(run_dir, prep_run)


def main(run_id, server_args, setup_dir):
    """Collect setup information and pass it to the `preparation` function."""
    _, app_interface, run_dir, app_spec, run_info = utils.setup_script(
        run_id, server_args, setup_dir
    )
    prep_run = getattr(app_interface, "prep_run", None)
    logger = logging.getLogger(__name__)
    if not JINJA2_AVAIL:
        logger.warning(
            "jinja2 not available, using built-in support for template files"
        )
    else:
        logger.debug("Using jinja2 for file templates...")
    logger.info("Populating run directory with files and calling user's prep_run")
    preparation(prep_run, run_dir, app_spec, run_info.sample, run_info.steps)
    logger.debug("Run setup complete")


if __name__ == "__main__":
    utils.profile_worker(main, "prepper")

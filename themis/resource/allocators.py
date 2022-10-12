"""
This module defines classes and functions related to creating HPC allocations.

Instances of the Allocator class interface with resource managers
(e.g. LSF on LC's CORAL systems) to create allocations.

Allocations are sometimes also known as "jobs"; the applications launched inside
them (via, e.g., `srun`) are called "job steps", although the terminology is used
inconsistently.

Instances of the Allocation class are used to describe an allocation to
an Allocator (which then creates that allocation).

The ShellScript class is used to write batch scripts. All the supported
resource managers only accept jobs (allocations) in the form of scripts.
"""
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import subprocess as sp
import time
import os
import stat

from themis import utils
from themis.versions import basestring


class ShellScript(object):
    """Create an object representation of a shell script."""

    def __repr__(self):
        return "{}({}, {})".format(type(self).__name__, self.path, self.shell)

    def __init__(self, path, shell="bash"):
        """Create an object representing a new shell script.

        :param path: the path to the new shell script.
        :param shell: the shell to use. Only affects the shebang.
        """
        self.path = path
        self.shell = shell
        self.headers = []
        self.commands = []
        self.working_directory = None

    def write(self):
        """Write the shell script to the file system.

        Ensures the file has execute permissions.
        Returns the path of the written file.
        """
        with open(self.path, "w") as file_handle:
            file_handle.write("#!" + utils.which(self.shell) + "\n")
            for header in self.headers:
                file_handle.write(header + "\n")
            file_handle.write("\n")
            if self.working_directory is not None:
                file_handle.write("cd " + self.working_directory + "\n")
            for command in self.commands:
                file_handle.write(command + "\n")
        # ensure usr has r/w/x permisions
        os.chmod(self.path, stat.S_IXUSR | stat.S_IRUSR | stat.S_IWUSR)
        return os.path.abspath(self.path)


def _get_launch_script_name(script_dir):
    """Return a path to the ensemble's allocation script."""
    if script_dir is None:
        script_dir = os.getcwd()
    return os.path.join(script_dir, "ensemble_launch_script.sh")


class Allocation(object):  # pylint: disable=too-few-public-methods
    """Represents an allocation on a HPC system.

    :param nodes: sets the allocation size for the ensemble in terms of total nodes.
    :type nodes: int
    :param partition: the compute partition to use for the ensemble.
    :type partition: str
    :param bank: the bank to use for the ensemble, e.g. "wbronze". The default value
        of None allows the resource manager to choose the bank.
    :type bank: str
    :param name: the name to give the allocation.
    :type name: str
    :param timeout: the time limit to request for the allocation, in minutes.
    :type timeout: int
    :param repeats: the number of times to replicate the allocation if time expires
        but the ensemble is not yet complete
    """

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        nodes=1,
        partition=None,
        bank=None,
        name="themis",
        timeout=None,
        repeats=0,
    ):
        self.partition = utils.type_check(partition, basestring, type(None))
        self.bank = utils.type_check(bank, basestring, type(None))
        self.name = utils.type_check(name, basestring)
        self.repeats = repeats
        if timeout is not None:
            utils.range_check(timeout, min_val=0, name="timeout")
        self.timeout = timeout
        self.nodes = utils.range_check(nodes, min_val=1, name="nodes")

    def __repr__(self):
        """Return str representation of constructor call."""
        return "{}(nodes={}, partition={!r}, bank={!r}, name={!r}, timeout={})".format(
            type(self).__name__,
            self.nodes,
            self.partition,
            self.bank,
            self.name,
            self.timeout,
        )


class Allocator(object):
    """Abstract base class; defines interface for interactions with resource managers.

    Allocator classes are designed to request an allocation of resources from the
    system, and then run a designated script within that allocation.

    Allocators for HPC resources generally build a batch submission script, then pass it
    to the resource manager for that system. Sometimes they may launch a command
    that requests an interactive allocation.

    Allocators for generic computers are usually trivial, because in that case the way
    to request an 'allocation of resources' for a set of programs is just to launch
    those programs.
    """

    def __repr__(self):
        """Return string representation of constructor call"""
        return "{}()".format(type(self).__name__)

    def start(self, allocation, applications, script_dir=None, working_dir=None):
        """Request an allocation of resources and launch an application within it.

        :param allocation: an Allocation instance representing the
            allocation to request.
        :param applications: the applications to launch within the allocation. This
            argument should be an iterable of str,
            e.g. ["cd /my/dir", "srun -n16 my/app", "cat app_results | grep hydro"]
        :param script_dir: the directory in which to write the allocation script
            to write (if any). Should be ignored if not applicable
        """

    def wait(self, sleep_interval=10):
        """Wait for the ensemble to complete. For use by testing infrastructure."""
        raise NotImplementedError()


class BatchScriptAllocator(Allocator):
    """Abstract base class for all Allocators that write batch scripts."""

    def __init__(self, job_id=None):
        """Constructor"""
        self.job_id = job_id

    def start(self, allocation, applications, script_dir=None, working_dir=None):
        if working_dir is None:
            working_dir = os.getcwd()
        script_path = self._write_batch_script(
            allocation, applications, script_dir, working_dir
        )
        self.job_id = self.launch_batch_script(script_path, working_dir)
        return self.job_id

    def _write_batch_script(self, allocation, applications, script_dir, working_dir):
        """Write a batch script."""
        raise NotImplementedError()

    def launch_batch_script(self, script_path, working_dir):
        """Submit a batch script and return the job ID."""
        raise NotImplementedError()


class MoabAllocator(Allocator):
    """A partially-implemented allocator.

    Moab is not a supported resource manager. However,
    the launch_batch_script method is needed.
    """

    @staticmethod
    def launch_batch_script(script_path, working_dir):
        """Submit a batch script to Moab; return the job ID."""
        subproc = sp.Popen(
            ["msub", script_path],
            stdout=sp.PIPE,
            stderr=sp.PIPE,
            universal_newlines=True,
            cwd=working_dir,
        )
        stdout, stderr = subproc.communicate()
        if subproc.returncode != 0:
            raise Exception(
                "Couldn't launch Moab batch script, "
                "msub exited with returncode {}:\n{}".format(subproc.returncode, stderr)
            )
        return int(stdout)

    def start(self, allocation, applications, script_dir=None, working_dir=None):
        raise NotImplementedError()


class SbatchAllocator(BatchScriptAllocator):
    """A Allocator for the Slurm resource manager.

    Launches by creating a sbatch script, the submitting it to Slurm by running
    sbatch as a subprocess. Sbatch adds the job to the queue, returns a job ID,
    and exits immediately.
    """

    @classmethod
    def _write_batch_script(cls, allocation, applications, script_dir, working_dir):
        """Build a Slurm batch script."""
        launch_script = ShellScript(_get_launch_script_name(script_dir))
        launch_script.working_directory = working_dir
        declaration = "#SBATCH"
        standard_slurm_headers = [
            "{} -J {}".format(declaration, allocation.name),
            "{} -N {}".format(declaration, allocation.nodes),
            "{} --exclusive".format(declaration),
        ]
        launch_script.headers.extend(standard_slurm_headers)
        if allocation.bank is not None:
            launch_script.headers.append(
                "{} -A {}".format(declaration, allocation.bank)
            )
        if allocation.partition is not None:
            launch_script.headers.append(
                "{} -p {}".format(declaration, allocation.partition)
            )
        if allocation.timeout is not None:
            launch_script.headers.append(
                "{} -t {}".format(declaration, int(allocation.timeout))
            )
        launch_script.commands.extend(applications)
        launch_script.commands.append("wait")
        return launch_script.write()

    @staticmethod
    def launch_batch_script(script_path, working_dir):
        """Submit a Sbatch batch script; return the jobid."""
        subproc = sp.Popen(
            ["sbatch", script_path],
            stdout=sp.PIPE,
            stderr=sp.STDOUT,
            universal_newlines=True,
            cwd=working_dir,
        )
        stdout, _ = subproc.communicate()
        if subproc.returncode != 0:
            raise Exception(
                "Couldn't launch Slurm batch script, Sbatch exited with "
                "returncode {}:\n{}".format(subproc.returncode, stdout)
            )
        return int(stdout.split()[-1])

    def wait(self, sleep_interval=5):
        while not self._done():
            time.sleep(sleep_interval)

    def _done(self):
        """Return True if ensemble job is done, False otherwise.

        Runs the sacct command to see the if the ensemble is done.
        """
        subproc = sp.Popen(
            ["sacct", "-X", "-j", str(self.job_id)],
            stdout=sp.PIPE,
            universal_newlines=True,
        )
        stdout, _ = subproc.communicate()
        job_completed = ("COMPLETED", "CANCELLED", "TIMEOUT", "FAILED")
        for status in job_completed:
            if status in stdout:
                return True
        return False


class BsubAllocator(BatchScriptAllocator):
    """Allocator for the LSF resource manager.

    Launches by creating a bsub script, the submitting it to LSF by running
    bsub as a subprocess. Bsub adds the job to the queue, returns a job ID,
    and exits immediately.
    """

    @classmethod
    def _write_batch_script(cls, allocation, applications, script_dir, working_dir):
        """Write a Bsub batch script."""
        launch_script = ShellScript(_get_launch_script_name(script_dir))
        launch_script.working_directory = working_dir
        declaration = "#BSUB"
        standard_lsf_headers = [
            "{} -J {}".format(declaration, allocation.name),
            "{} -nnodes {}".format(declaration, allocation.nodes),
        ]
        launch_script.headers.extend(standard_lsf_headers)
        if allocation.bank is not None:
            launch_script.headers.append(
                "{} -G {}".format(declaration, allocation.bank)
            )
        if allocation.partition is not None:
            launch_script.headers.append(
                "{} -q {}".format(declaration, allocation.partition)
            )
        if allocation.timeout is not None:
            launch_script.headers.append(
                "{} -W {}".format(declaration, int(allocation.timeout))
            )
        launch_script.commands.extend(applications)
        launch_script.commands.append("wait")
        return launch_script.write()

    @staticmethod
    def launch_batch_script(script_path, working_dir):
        """Submit a Bsub batch script to LSF; return the jobid."""
        with open(script_path, "r") as file_handle:
            subproc = sp.Popen(
                ["bsub"],
                stdin=file_handle,
                stdout=sp.PIPE,
                stderr=sp.PIPE,
                universal_newlines=True,
                cwd=working_dir,
            )
        stdout, stderr = subproc.communicate()
        if subproc.returncode != 0:
            raise Exception(
                "Couldn't launch LSF batch script; "
                "Bsub exited with returncode {}:\n{}".format(subproc.returncode, stderr)
            )
        return int(stdout.split()[1].strip("><"))

    def wait(self, sleep_interval=5):
        while not self._done():
            time.sleep(sleep_interval)

    def _done(self):
        """Return True if the ensemble is done, False otherwise.

        Runs bjobs as a subprocess and checks if the job ID is in the 'done'
        category of information printed to stdout.
        """
        subproc = sp.Popen(
            ["bjobs", "-d"], stdout=sp.PIPE, stderr=sp.PIPE, universal_newlines=True
        )
        stdout, _ = subproc.communicate()
        return str(self.job_id) in stdout


class InteractiveAllocator(BatchScriptAllocator):
    """An allocator for a generic computer with no special allocation system.

    This allocator works on any system, including laptops, but
    it is a bad choice for production runs on HPC systems.
    No allocation is requested--the scheduler is simply submitted as a subprocess.
    However, for running interactively on HPC systems (either within an allocation
    or on a login node), this Allocator should work well.
    """

    def __init__(self, job_id=None):
        """Constructor"""
        super(InteractiveAllocator, self).__init__(job_id)
        self._proc_handle = None

    @classmethod
    def _write_batch_script(cls, _, applications, script_dir, working_dir):
        """Write a script to execute applications. The allocation is ignored."""
        script = ShellScript(_get_launch_script_name(script_dir))
        script.commands.extend(applications)
        script.commands.append("wait")
        working_dir = working_dir if working_dir is not None else os.getcwd()
        script.working_directory = working_dir
        return script.write()

    def launch_batch_script(self, script_path, working_dir):
        """Launch the script as a subprocess; return the PID."""
        stdio_path = os.path.join(working_dir, "themis_interactive_script.log")
        with open(stdio_path, "a") as stdio:
            self._proc_handle = sp.Popen(
                [script_path], cwd=working_dir, stdout=stdio, stderr=sp.STDOUT
            )
        return self._proc_handle.pid

    def wait(self, sleep_interval=10, debug=False):  # pylint: disable=arguments-differ
        """Wait for the subprocess to complete, then return"""
        if self._proc_handle is not None:
            if self._proc_handle.wait() != 0 and debug:
                raise RuntimeError("Wait returned nonzero")
        else:
            os.waitpid(self.job_id, os.WNOHANG)

    def done(self):
        """Return True if the subprocess is done, False otherwise"""
        if self._proc_handle is not None:
            return self._proc_handle.poll() is not None
        return not self._check_pid(self.job_id)

    @classmethod
    def _check_pid(cls, pid):
        """Return true if a process given by pid has completed."""
        try:
            os.kill(pid, 0)
        except OSError:
            return True
        else:
            return False


class FluxAllocator(Allocator):
    def __init__(self, job_id=None):
        self.job_id = None

    def start(self, allocation, applications, script_dir=None, working_dir=None):
        args = [
            "flux",
            "mini",
            "batch",
            "--wrap",
            "--exclusive",
            "-N",
            str(allocation.nodes),
        ]
        if allocation.name is not None:
            args.append("--job-name=" + str(allocation.name))
        if allocation.timeout is not None:
            args += ["-t", str(allocation.timeout) + "m"]
        separated_applications
        for app in applications:
            separated_applications.append(app + ";")
        proc = sp.Popen(
            args + applications + ["wait"],
            stdout=sp.PIPE,
            stderr=sp.STDOUT,
            universal_newlines=True,
            cwd=working_dir,
        )
        stdout, _ = proc.communicate()
        if proc.returncode != 0:
            raise Exception(
                "Couldn't launch Flux job, flux mini batch exited with "
                "returncode {}:\n{}".format(subproc.returncode, stdout)
            )
        return stdout

    def wait(self, sleep_interval=10):
        """Wait for the ensemble to complete. For use by testing infrastructure."""
        proc = sp.Popen(
            ["flux", "job", "attach", str(self.job_id)],
            stdout=sp.DEVNULL,
            stderr=sp.PIPE,
            universal_newlines=True,
        )
        _, stderr = proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(stderr)

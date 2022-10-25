"""
This module defines Executor classes.

Executors are used to submit and monitor jobs.
"""


from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import glob
import os
import subprocess

from themis.utils import (
    URI_ENV_VAR,
    RUNID_ENV_VAR,
    URI_SPLITCHAR,
    SETUPDIR_ENV_VAR,
    EXECDIR_ENV_VAR,
)

try:
    import concurrent.futures as cf
    import flux
    import flux.job
except (ImportError, SyntaxError):
    FLUX_BINDINGS_AVAIL = False
else:
    FLUX_BINDINGS_AVAIL = True


RUN_LOG_FILE = 'run' + os.extsep + 'log'
LOG_EXT = os.extsep + 'log'
RUNLOG_GLBR = 'run' + '*' + LOG_EXT

def _exitstatus_to_returncode(status):
    """Convert an `os.wait` exit status to a returncode."""
    if os.WIFEXITED(status):
        return os.WEXITSTATUS(status)
    if os.WIFSIGNALED(status):
        return -os.WTERMSIG(status)
    return -256


class Executor(object):
    """Abstract class that provides methods to submit and wait on jobs
    asynchronously."""

    def __init__(self, setup_dir, themis_uri):
        self.themis_uri = URI_SPLITCHAR.join(themis_uri)
        self.setup_dir = setup_dir
        self.exec_dir = os.getcwd()

    # pylint: disable=too-many-arguments
    def submit(self, run_id, step):
        """Submit a job and return a job ID.

        Stdout and stderr for the job will be redirected to `cwd/run.log`.

        :param step: The themis.utils.Step to launch.

        :returns: an integer identifying the job.
        """
        raise NotImplementedError()

    def wait(self, timeout=None):
        """Wait for any job to complete and return the pair (jobid, exit_status).

        The exit_status entry will be either a process exit code or an exception.

        If there are no jobs, return `None`.

        :param timeout: the number of seconds to wait for a job to complete.
            If None, wait indefinitely.
        """
        raise NotImplementedError()

    def kill(self, jobid):
        """Cancel or kill a job."""


class ProcessExecutor(Executor):
    """Submit jobs as individual processes."""

    def __init__(self, setup_dir, themis_uri, command_builder):
        super(ProcessExecutor, self).__init__(setup_dir, themis_uri)
        self._command_builder = command_builder
        self._processes = {}
        self._environ = dict(os.environ)
        self._environ[URI_ENV_VAR] = self.themis_uri
        self._environ[SETUPDIR_ENV_VAR] = self.setup_dir


    # pylint: disable=too-many-arguments
    def submit(self, run_id, step):
        """Submit an application as a process. Return the process ID (pid)."""
        command = self._command_builder(step)  # get the prefix
        command.extend(step.args)
        env = dict(self._environ)
        env[RUNID_ENV_VAR] = str(run_id)

        print("source/executors.py submit")
        print(step)
        print(step.cwd)
        cwd = step.cwd[:-1]

        if os.path.isfile(os.path.join(cwd, RUN_LOG_FILE)):
            count = len(glob.glob(os.path.join(cwd, RUNLOG_GLBR)))
            os.rename(os.path.join(cwd,  RUN_LOG_FILE),
                     "%s_%04d%s" % (os.path.join(cwd,  'run'), count, LOG_EXT))

        with open(os.path.join(step.cwd, RUN_LOG_FILE), "a") as run_log:
            process = subprocess.Popen(
                command,
                cwd=step.cwd,
                stdout=run_log,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                env=env,
            )
        self._processes[process.pid] = process
        return process.pid

    def wait(self, timeout=None):
        """Wait for a job to a complete and return its returncode.

        Any timeout value other than `None` has no effect.
        """
        if not self._processes:
            return None
        if timeout is None and hasattr(os, "wait"):
            pid, exit_status = os.wait()
            completed_proc = self._processes.pop(pid, None)
            if completed_proc is not None:
                completed_proc.poll()
                return (pid, _exitstatus_to_returncode(exit_status))
        else:
            for proc in self._processes.values():
                if proc.poll() is not None:
                    completed_proc = proc
                    self._processes.pop(completed_proc.pid)
                    return (completed_proc.pid, completed_proc.returncode)
        return None

    def kill(self, jobid):
        """Kill or cancel a job."""
        self._processes[jobid].terminate()


class FluxBindingsExecutor(Executor):
    """Submit applications via `flux.job.FluxExecutor`."""

    def __init__(self, setup_dir, themis_uri):
        super(FluxBindingsExecutor, self).__init__(setup_dir, themis_uri)
        self._executor = flux.job.FluxExecutor()
        self._futures = {}
        self._next_jobid = 0
        self._flux_handle = flux.Flux()  # for cancelling futures
        self._environ = dict(os.environ)
        self._environ[URI_ENV_VAR] = self.themis_uri
        self._environ[SETUPDIR_ENV_VAR] = self.setup_dir
        self._environ[EXECDIR_ENV_VAR] = self.exec_dir

    # pylint: disable=too-many-arguments
    def submit(self, run_id, step):
        """Submit an application via Flux. Return an integer."""
        if step.batch_script:
            jobspec = flux.job.JobspecV1.from_nest_command(
                step.args,
                num_slots=step.tasks,
                cores_per_slot=step.cores_per_task,
                gpus_per_slot=step.gpus_per_task,
            )
        else:
            jobspec = flux.job.JobspecV1.from_command(
                step.args,
                num_tasks=step.tasks,
                cores_per_task=step.cores_per_task,
                gpus_per_task=step.gpus_per_task,
            )
        env = dict(self._environ)
        env[RUNID_ENV_VAR] = str(run_id)
        jobspec.environment = env
        if step.cwd is not None:
            jobspec.cwd = step.cwd
        if step.timeout is not None and step.timeout > 0:
            jobspec.duration = float(step.timeout * 60)

        cwd = step.cwd
        if os.path.isfile(os.path.join(cwd, RUN_LOG_FILE)):
            count = len(glob.glob(os.path.join(cwd, RUNLOG_GLBR)))
            os.rename(os.path.join(cwd,  RUN_LOG_FILE),
                     "%s_%04d%s" % (os.path.join(cwd,  'run'), count, LOG_EXT))

        jobspec.stdout = os.path.join(step.cwd, RUN_LOG_FILE)
        self._futures[self._executor.submit(jobspec)] = self._next_jobid
        self._next_jobid += 1
        return self._next_jobid - 1

    def wait(self, timeout=None):
        """Wait for a job to complete."""
        if not self._futures:
            return None
        done, _ = cf.wait(self._futures, timeout, return_when=cf.FIRST_COMPLETED)
        if done:
            completed_fut = done.pop()
            completed_jobid = self._futures.pop(completed_fut)
            try:
                result = completed_fut.result()
            except Exception as exc:  # pylint: disable=broad-except
                result = exc
            return (completed_jobid, result)
        return None

    def kill(self, jobid):
        """Kill/cancel a job."""
        for fut, iter_jobid in self._futures.items():
            if jobid == iter_jobid:
                if not fut.cancel():
                    flux.job.cancel(self._flux_handle, fut.jobid())
                return

"""This module provides the primary interface to resource-manager-specific code.

Aside from the ResourceManager class itself, which defines the primary interface,
there are also a handful of utility functions in this module.
"""
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import sys
import os
import math
import time
import socket
import subprocess
import platform

from themis.resource import executors
from themis.resource import allocators
from themis import utils


_UNIDENTIFIED_RMGR = None

_RESOURCE_MANAGERS = {}

_BACKEND_PATH = os.path.join(os.path.abspath(__file__).rsplit(os.sep, 2)[0], "backend")

_WORKER_PATH = os.path.join(
    os.path.abspath(__file__).rsplit(os.sep, 2)[0], "backend", "worker"
)


def list_resource_mgr_identifiers():
    """List the identifiers of supported resource managers."""
    return list(key for key in _RESOURCE_MANAGERS if key != Moab.identifier)


def valid_resource_mgr(identifier):
    """Return True if `identifier` identifies a supported resource manager."""
    if isinstance(identifier, str):
        identifier = identifier.lower()
    return identifier in _RESOURCE_MANAGERS


def _resource_mgr_from_env():
    """Attempt to identify the resource manager from the environment.

    Return the empty string if no match is found
    """
    sys_type = os.getenv("SYS_TYPE", "")
    if sys_type == "" and platform.system().lower() in ["darwin", "windows"]:
        return NoResourceManager.identifier
    if sys_type.startswith("toss") or (utils.which("srun") and utils.which("sbatch")):
        if os.path.isdir("/opt/cray"):
            return LANLSlurm.identifier
        return Slurm.identifier
    if sys_type.startswith("blueos") or (utils.which("bsub") and utils.which("jsrun")):
        return Lsf.identifier
    # check Flux last because it can run inside the others
    if (
        os.getenv("FLUX_URI") is not None
        or subprocess.Popen(["flux", "resource", "list"]).wait() == 0
    ):
        return Flux.identifier
    return _UNIDENTIFIED_RMGR


def default_resource_manager_id():
    """Return the current type of system. Return values are one of None,
    'toss', 'blueos', 'flux' and 'suse-linux'.

    Note that None is actually an identifier for a resource manager,
    the NoResourceManager.

    Raises NotImplementedError if system is not recognized.
    """

    rsrc_mgr = _resource_mgr_from_env()
    if rsrc_mgr == _UNIDENTIFIED_RMGR:
        not_supported_msg = (
            "Your machine {!r} is not recognized, and may not be supported."
        )
        raise NotImplementedError(not_supported_msg.format(socket.gethostname()))
    return rsrc_mgr


def identify_resource_manager(identifier=_UNIDENTIFIED_RMGR, path=None):
    """Return the resource manager denoted by the given identifier.

    If identifier is None, get the default resource manager for this machine.
    """
    if identifier == _UNIDENTIFIED_RMGR:
        identifier = default_resource_manager_id()
    elif isinstance(identifier, str):
        identifier = identifier.lower()
    try:
        return _RESOURCE_MANAGERS[identifier](path)
    except KeyError:
        raise ValueError("No resource manager recognized by {!r}".format(identifier))


def _register_rmgr(cls):
    """Register a ResourceManager subclass with the identify_resource_manager func."""
    _RESOURCE_MANAGERS[cls.identifier] = cls
    return cls


def validate_flux_path(use_flux):
    """Search a couple of predefined paths for a Flux installation."""
    if isinstance(use_flux, str):
        path = utils.which(use_flux)
        if path is None:
            raise ValueError(
                "Can't find Flux installation at user-specified path {!r}".format(
                    use_flux
                )
            )
        return path
    # path = "/g/g12/corbett8/local/bin/flux"
    path = "/usr/global/tools/flux/{}/flux-c0.28.0.pre-s0.17.0.pre/bin/flux".format(
        os.getenv("SYS_TYPE")
    )
    if not os.path.isfile(path):
        path = utils.which("flux")
    if path is None:
        raise RuntimeError("Cannot find Flux installation")
    return path


def backend_dir(setup_dir, alloc_id):
    """Return the execution directory for a given alloc id."""
    return os.path.join(setup_dir, "execution_" + str(alloc_id))


class ProcessGroup(object):
    """Represents a group of `subprocess.Popen` objects."""

    def __init__(self, process_iterable):
        self._processes = process_iterable

    def poll(self):
        """Return None if one or more processes are still running."""
        if any(process.poll() is None for process in self._processes):
            return None
        return True

    def terminate(self, wait=0):
        """Kill all the processes."""
        for process in self._processes:
            try:
                process.terminate()
            except OSError:  # process has already exited
                pass
        sleep_sec = 0.1
        while wait > 0:
            if self.poll() is not None:
                break
            time.sleep(sleep_sec)
            wait -= sleep_sec


class ResourceManager(object):
    """Encapsulates high-level, general information about a resource manager.

    Subclass should expose the following attributes:
    :param identifier: string identifying the resource manager
    :param allocator: the allocators.Allocator class to use for getting an allocation
    :param batch_script_occupies_core: whether, when running a batch script, an extra
        core should be allocated for the batch script itself.
    """

    def __init__(self, _):
        pass

    def __repr__(self):
        """Return str representation of constructor call"""
        return "{}()".format(type(self).__name__)

    @classmethod
    def commands_to_launch_backend(  # pylint: disable=too-many-arguments
        cls,
        timeout,
        parallelism,
        alloc_id,
        early_stop,
        multiple,
        setup_dir,
        max_concurrency,
    ):
        """Return the shell commands to start the ensemble's backend."""

        return (
            "{py} {backend_path} -t{runtime} -a{aid} -p{parallel} "
            "-e{early_stop} --setup-dir {setup_dir} -c {max_concur} {multiple_flag}"
            ">> {backend_dir}{sep}themis_backend.log 2>&1"
        ).format(
            py=sys.executable,
            backend_path=_BACKEND_PATH,
            parallel=parallelism,
            aid=alloc_id,
            runtime=timeout,
            early_stop=early_stop,
            setup_dir=setup_dir,
            backend_dir=backend_dir(setup_dir, alloc_id),
            sep=os.sep,
            max_concur=max_concurrency,
            multiple_flag="--multiple " if multiple else "",
        )

    @classmethod
    def launch_workers(cls, parallelism, server_args, setup_dir, max_concurrency):
        """Launch workers that will submit and monitor runs.

        :param parallelism: the number of workers to launch
        :param server_args: the network connection information for the workers
        :returns: an object with `poll()` and `kill(wait)` attributes representing
            the workers.
        """
        if parallelism <= 0:
            return ProcessGroup(
                [
                    cls._launch_single_worker(
                        NoResourceManager.build_cmd(utils.Step(sys.executable)),
                        0,
                        server_args,
                        setup_dir,
                        max_concurrency,
                    )
                ]
            )
        return ProcessGroup(
            [
                cls._launch_single_worker(
                    cls.build_cmd(utils.Step(sys.executable)),
                    i,
                    server_args,
                    setup_dir,
                    math.ceil(max_concurrency / parallelism),
                )
                for i in range(parallelism)
            ]
        )

    # pylint: disable=too-many-arguments
    @classmethod
    def _launch_single_worker(
        cls, launch_command, identifier, server_args, setup_dir, max_concurrency,
    ):
        launch_command.extend(
            (
                "{py} {worker_path} --id {identifier} --server-args "
                "{server_args} --setup-dir {setup_dir} -c {max_concur}"
            )
            .format(
                py=sys.executable,
                identifier=identifier,
                worker_path=_WORKER_PATH,
                server_args=" ".join(server_args),
                setup_dir=setup_dir,
                max_concur=int(max_concurrency),
            )
            .split()
        )
        with open("themis_worker_{}.log".format(identifier), "a") as file_handle:
            return subprocess.Popen(
                launch_command, stdout=file_handle, stderr=subprocess.STDOUT,
            )

    @classmethod
    def executor(cls, setup_dir, server_args):
        """Return an Executor instance for use with this resource manager."""
        return executors.ProcessExecutor(setup_dir, server_args, cls.build_cmd)

    @classmethod
    def build_cmd(cls, step):
        """Return a command to launch an application with resource requirements."""
        raise NotImplementedError("Using abstract class")


@_register_rmgr
class Moab(ResourceManager):
    """The Moab resource manager.

    Only partially implemented because only used by the ``laf.BatchSubmitter`` class.
    """

    identifier = "moab"
    allocator = allocators.MoabAllocator

    @classmethod
    def build_cmd(cls, step):
        raise NotImplementedError("MOAB only partially implemented")


@_register_rmgr
class Sbatch(ResourceManager):
    """The Sbatch resource manager.

    Only partially implemented because only used by the ``laf.BatchSubmitter`` class.
    """

    identifier = "sbatch"
    allocator = allocators.SbatchAllocator

    @classmethod
    def build_cmd(cls, step):
        raise NotImplementedError("SBATCH only partially implemented")


@_register_rmgr
class Slurm(ResourceManager):
    """The Slurm resource manager, as it works on LC."""

    identifier = "slurm"
    allocator = allocators.SbatchAllocator
    batch_script_occupies_core = True
    _BASE_CMD = ("srun", "--exclusive", "--mpibind=off")

    @classmethod
    def build_cmd(cls, step):
        """Return the run command ('srun') for TOSS systems"""
        base_cmd = list(cls._BASE_CMD)
        if step.batch_script:
            base_cmd.extend(("-n1", "-c1", "-N1"))
        else:
            if step.tasks > 0:
                base_cmd.extend(("-n", str(step.tasks)))
            if step.cores_per_task > 0:
                base_cmd.extend(("-c", str(step.cores_per_task)))
        if step.timeout > 0:
            base_cmd.extend(("-t", str(step.timeout)))
        return base_cmd


@_register_rmgr
class LANLSlurm(Slurm):
    """LANL's Slurm is different than LC's---primarily in ``srun`` options."""

    identifier = "lanl-slurm"
    _BASE_CMD = ("srun", "--exclusive", "--gres=craynetwork:0")

    @classmethod
    def build_cmd(cls, step):
        """Return the run command ('srun') for TOSS systems"""
        if step.batch_script:
            return []
        return super(LANLSlurm, cls).build_cmd(step)


@_register_rmgr
class Flux(ResourceManager):
    """The Flux resource manager.

    Flux is an unusual resource manager because it can run on top of another
    resource manager---e.g. Slurm.
    """

    identifier = "flux"
    batch_script_occupies_core = False

    def __init__(self, path):
        super(Flux, self).__init__(None)
        # check whether this is a Flux-native machine or if another RM is running
        if default_resource_manager_id() == self.identifier:
            self.underlying_resource_mgr = None  # flux-native
        else:
            #  another RM is running
            self.underlying_resource_mgr = identify_resource_manager()
        self.path = path

    def __repr__(self):
        """Return str representation of constructor call"""
        return "{}(path={}, underlying_resource_mgr={})".format(
            type(self).__name__, self.path, self.underlying_resource_mgr
        )

    @property
    def allocator(self):
        """Return the allocator of the underlying rmgr."""
        if self.underlying_resource_mgr is not None:
            return self.underlying_resource_mgr.allocator
        return allocators.FluxAllocator

    def commands_to_launch_backend(  # pylint: disable=too-many-arguments
        self,
        timeout,
        parallelism,
        alloc_id,
        early_stop,
        multiple,
        setup_dir,
        max_concurrency,
    ):
        """Return the shell commands to start the ensemble's backend.

        Use the `flux tree` utility if needed.
        """
        base_cmd = ""
        if self.underlying_resource_mgr is not None:
            if isinstance(self.underlying_resource_mgr, Slurm):
                base_cmd = "srun --ntasks-per-node=1 --mpibind=off "
            elif isinstance(self.underlying_resource_mgr, Lsf):
                base_cmd = "lrun -T1 --mpibind=off "
            base_cmd += self.path + " start "
        return base_cmd + super(Flux, self).commands_to_launch_backend(
            timeout,
            parallelism,
            alloc_id,
            early_stop,
            multiple,
            setup_dir,
            max_concurrency,
        )

    def launch_workers(self, parallelism, server_args, setup_dir, max_concurrency):
        if parallelism == 0:
            return super(Flux, self).launch_workers(
                parallelism, server_args, setup_dir, max_concurrency
            )
        return ProcessGroup(
            [
                subprocess.Popen(
                    (
                        "flux tree -T{parallelism} -c{cores_per_node} -J{parallelism} "
                        "-- {py} {worker_path} --server-args {server_args} "
                        "-c {max_concur} --setup_dir {setup_dir}"
                    )
                    .format(
                        parallelism=parallelism,
                        cores_per_node=os.getenv("SLURM_CPUS_ON_NODE"),
                        py=sys.executable,
                        worker_path=_WORKER_PATH,
                        server_args=" ".join(server_args),
                        setup_dir=setup_dir,
                        max_concur=math.ceil(max_concurrency / parallelism),
                    )
                    .split()
                )
            ]
        )

    @classmethod
    def _stringify_topology(cls, topology_as_iterable):
        """Convert a topology to a string recognized by Flux-tree."""
        topology = [str(entry) for entry in topology_as_iterable]
        return "x".join(topology)

    @classmethod
    def _calculate_topology(cls, target_leaf_nodes, max_split=50, default_split=5):
        """Recursively calculate a topology given a target number of leaf nodes."""
        if target_leaf_nodes > max_split:
            result = [default_split]
            result.extend(
                cls._calculate_topology(math.ceil(target_leaf_nodes / default_split))
            )
            return result
        return [target_leaf_nodes]

    @classmethod
    def available(cls):
        """Return True if Flux is installed and available."""
        try:
            validate_flux_path(None)
        except ValueError:
            return False
        except RuntimeError:
            return False
        else:
            return True

    @classmethod
    def build_cmd(cls, step):
        """Return the run command for flux."""
        if step.batch_script:
            base_cmd = ["flux", "mini", "alloc"]
        else:
            base_cmd = ["flux", "mini", "run"]
        if step.tasks > 0:
            base_cmd.extend(("-n", str(step.tasks)))
        if step.cores_per_task > 0:
            base_cmd.extend(("-c", str(step.cores_per_task)))
        if step.gpus_per_task > 0:
            base_cmd.extend(("-g", str(step.gpus_per_task)))
        if step.timeout > 0:
            base_cmd.extend(("-t", str(step.timeout) + "m"))
        return base_cmd

    @classmethod
    def executor(cls, setup_dir, server_args):
        if executors.FLUX_BINDINGS_AVAIL:
            return executors.FluxBindingsExecutor(setup_dir, server_args)
        return super(Flux, cls).executor(setup_dir, server_args)


@_register_rmgr
class Lsf(ResourceManager):
    """The LSF resource manager, for Sierra and Lassen."""

    identifier = "lsf"
    allocator = allocators.BsubAllocator
    batch_script_occupies_core = True

    @classmethod
    def build_cmd(cls, step):
        """Return the run command ('lrun') for LSF systems"""
        base_cmd = ["lrun", "--pack"]
        if step.tasks > 0:
            base_cmd.extend(("-n", str(step.tasks)))
        if step.cores_per_task > 0:
            base_cmd.extend(("-c", str(step.cores_per_task)))
        if step.gpus_per_task > 0:
            base_cmd.extend(("-g", str(step.gpus_per_task)))
        return base_cmd


@_register_rmgr
class NoResourceManager(ResourceManager):
    """For systems without a resource manager running on top of the OS."""

    identifier = "none"
    allocator = allocators.InteractiveAllocator
    batch_script_occupies_core = True

    @classmethod
    def build_cmd(cls, step):
        """Return a prefix command to launch an application with a given resource set.

        Should be overridden in subclasses. Base implementation launches
        applications directly, with no regard to their resource requirements.
        """
        return []

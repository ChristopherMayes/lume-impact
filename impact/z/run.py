from __future__ import annotations

import logging
import os
import pathlib
import platform
import shlex
import shutil
import traceback
from time import monotonic
from typing import Any, ClassVar, Dict, Optional, Sequence, Union

import h5py
import psutil
from lume import tools as lume_tools
from lume.base import CommandWrapper
from pmd_beamphysics import ParticleGroup
from pmd_beamphysics.units import pmd_unit
from typing_extensions import override

from . import tools
from . import units as units_mod
from .errors import ImpactRunFailure
from .input import ImpactZInput
from .output import ImpactZOutput, RunInfo
from .tools import read_if_path
from .types import AnyPath

logger = logging.getLogger(__name__)


def find_mpirun():
    """
    Simple helper to find the mpi run command for macports and homebrew,
    as well as custom commands for Perlmutter at NERSC.
    """

    for p in [
        # Highest priority is what our PATH says:
        shutil.which("mpirun"),
        # Second, macports:
        "/opt/local/bin/mpirun",
        # Third, homebrew:
        "/opt/homebrew/bin/mpirun",
    ]:
        if p and os.path.exists(p):
            return f'"{p}"' + " -n {nproc} {command_mpi}"

    if os.environ.get("NERSC_HOST") == "perlmutter":
        srun = "srun -n {nproc} --ntasks-per-node {nproc} -c 1 {command_mpi}"
        hostname = platform.node()
        assert hostname  # This must exist
        if hostname.startswith("nid"):
            # Compute node
            return srun
        else:
            # This will work on a login node
            return "salloc -N {nnode} -C cpu -q interactive -t 04:00:00 " + srun

    # Default
    return "mpirun -n {nproc} {command_mpi}"


def find_workdir():
    if os.environ.get("NERSC_HOST") == "perlmutter":
        return os.environ.get("SCRATCH")
    else:
        return None


class ImpactZ(CommandWrapper):
    """
    IMPACT-Z command wrapper for Python-defined configurations and lattices.

    Files will be written into a temporary directory within workdir.
    If workdir=None, a location will be determined by the system.

    Parameters
    ---------
    input : ImpactZInput, str, or pathlib.Path
        Input settings for the IMPACT-Z run.  This may be a full configuration
        (`ImpactZInput`), or a path to an existing file with main input
        settings (e.g., ``ImpactZ.in``).
    command : str, default="ImpactZexe"
        The command to run to execute IMPACT-Z.
    command_mpi : str, default="ImpactZexe-mpi"
        The IMPACT-Z executable to run under MPI.
    use_mpi : bool, default=False
        Enable parallel processing with MPI.
    mpi_run : str, default=""
        The template for invoking ``mpirun``. If not specified, the class
        attribute ``MPI_RUN`` is used. This is expected to be a formated string
        taking as parameters the number of processors (``nproc``) and the
        command to be executed (``command_mpi``).
    use_temp_dir : bool, default=True
        Whether or not to use a temporary directory to run the process.
    workdir : path-like, default=None
        The work directory to be used.
    verbose : bool, optional
        Whether or not to produce verbose output.
        Defaults to `global_display_options.verbose`, which is tied to the
        `LUME_VERBOSE` environment variable.
        When the variable is unset, `verbose` is `False` by default.
    timeout : float, default=None
        The timeout in seconds to be used when running IMPACT-Z.
    initial_particles : ParticleGroup, optional
        Initial particles to use in the simulation, using the
        OpenPMD-beamphysics standard.
    """

    COMMAND: ClassVar[str] = "ImpactZexe"
    COMMAND_MPI: ClassVar[str] = "ImpactZexe-mpi"
    MPI_RUN: ClassVar[str] = find_mpirun()
    WORKDIR: ClassVar[Optional[str]] = find_workdir()

    # Environmental variables to search for executables
    command_env: str = "IMPACTZ_BIN"
    command_mpi_env: str = "IMPACTZ_MPI_BIN"
    original_path: AnyPath

    _input: ImpactZInput
    output: Optional[Any]

    def __init__(
        self,
        input: Optional[Union[ImpactZInput, str, pathlib.Path]] = None,
        *,
        workdir: Optional[Union[str, pathlib.Path]] = None,
        output: Optional[Any] = None,  # TODO
        alias: Optional[Dict[str, str]] = None,
        units: Optional[Dict[str, pmd_unit]] = None,
        command: Optional[str] = None,
        command_mpi: Optional[str] = None,
        use_mpi: bool = False,
        mpi_run: str = "",
        use_temp_dir: bool = True,
        verbose: bool = tools.global_display_options.verbose >= 1,
        timeout: Optional[float] = None,
        initial_particles: Optional[ParticleGroup] = None,
        **kwargs: Any,
    ):
        super().__init__(
            command=command,
            command_mpi=command_mpi,
            use_mpi=use_mpi,
            mpi_run=mpi_run,
            use_temp_dir=use_temp_dir,
            workdir=workdir,
            verbose=verbose,
            timeout=timeout,
            **kwargs,
        )

        if input is None:
            input = ImpactZInput()
        elif isinstance(input, AnyPath):
            path, contents = read_if_path(input)
            if "\n" not in contents:
                raise FileNotFoundError(
                    f"{input!r} looks like a path but it does not exist on disk"
                )
            input = ImpactZInput.from_contents(contents, filename=path)
        elif isinstance(input, ImpactZInput):
            pass
        else:
            raise ValueError(f"Unsupported 'input' type: {type(input).__name__}")

        if (
            input.initial_particles is not initial_particles
            and initial_particles is not None
        ):
            input.initial_particles = initial_particles

        if workdir is None:
            workdir = pathlib.Path(".")

        self.original_path = workdir

        if pathlib.Path(workdir).exists() and not pathlib.Path(workdir).is_dir():
            raise ValueError(f"`workdir` exists and is not a directory: {workdir}")

        self._input = input
        self.output = output

        # Internal
        self._units = dict(units or units_mod.known_unit)
        self._alias = dict(alias or {})

        # MPI
        self.nproc = 1
        self.nnode = 1

    @property
    def input(self) -> ImpactZInput:
        """The input, including settings and lattice information."""
        return self._input

    @input.setter
    def input(self, inp: Any) -> None:
        if not isinstance(inp, ImpactZInput):
            raise ValueError(
                f"The provided input is of type {type(inp).__name__} and not `ImpactZInput`. "
                f"Please consider creating a new ImpactZ object instead with the "
                f"new parameters!"
            )
        self._input = inp

    @property
    def nproc(self):
        """
        Number of MPI processes to use.
        """
        return self._nproc

    @nproc.setter
    def nproc(self, nproc: Optional[int]):
        if nproc is None or nproc == 0:
            nproc = psutil.cpu_count(logical=False)
        elif nproc < 0:
            nproc += psutil.cpu_count(logical=False)

        if nproc <= 0:
            raise ValueError(f"Calculated nproc is invalid: {nproc}")

        self._nproc = nproc

    @override
    def configure(self):
        """
        Configure and set up for run.
        """
        self.setup_workdir(self._workdir)
        self.vprint("Configured to run in:", self.path)
        self.configured = True
        self.finished = False

    @override
    def run(
        self,
        load_fields: bool = False,
        load_particles: bool = False,
        smear: bool = True,
        raise_on_error: bool = True,
        verbose: Optional[bool] = None,
    ) -> ImpactZOutput:
        """
        Execute IMPACT-Z with the configured input settings.

        Parameters
        ----------
        load_fields : bool, default=False
            After execution, load all field files.
        load_particles : bool, default=False
            After execution, load all particle files.
        smear : bool, default=True
            If set, for particles, this will smear the phase over the sample
            (skipped) slices, preserving the modulus.
        raise_on_error : bool, default=True
            If it fails to run, raise an error. Depending on the error,
            output information may still be accessible in the ``.output``
            attribute.

        Returns
        -------
        ImpactZOutput
            The output data.  This is also accessible as ``.output``.
        """
        if verbose is not None:
            self.verbose = verbose

        if not self.configured:
            self.configure()

        if self.path is None:
            raise ValueError("Path (base_path) not yet set")

        self.finished = False

        runscript = self.get_run_script()

        start_time = monotonic()
        self.vprint(f"Running ImpactZ in {self.path}")
        self.vprint(runscript)

        self.write_input()

        if self.timeout:
            self.vprint(
                f"Timeout of {self.timeout} is being used; output will be "
                f"displayed after IMPACT-Z exits."
            )
            execute_result = tools.execute2(
                shlex.split(runscript),
                timeout=self.timeout,
                cwd=self.path,
            )
            self.vprint(execute_result["log"])
        else:
            log = []
            try:
                for line in tools.execute(shlex.split(runscript), cwd=self.path):
                    self.vprint(line, end="")
                    log.append(line)
            except Exception as ex:
                log.append(f"IMPACT-Z exited with an error: {ex}")
                self.vprint(log[-1])
                execute_result = {
                    "log": "".join(log),
                    "error": True,
                    "why_error": "error",
                }
            else:
                execute_result = {
                    "log": "".join(log),
                    "error": False,
                    "why_error": "",
                }

        end_time = monotonic()

        self.finished = True
        run_info = RunInfo(
            run_script=runscript,
            error=execute_result["error"],
            error_reason=execute_result["why_error"],
            start_time=start_time,
            end_time=end_time,
            run_time=end_time - start_time,
            output_log=execute_result["log"].replace("\x00", ""),
        )

        success_or_failure = "Success" if not execute_result["error"] else "Failure"
        self.vprint(f"{success_or_failure} - execution took {run_info.run_time:0.2f}s.")

        try:
            self.output = self.load_output(
                load_fields=load_fields,
                load_particles=load_particles,
                smear=smear,
            )
        except Exception as ex:
            stack = traceback.format_exc()
            run_info.error = True
            run_info.error_reason = (
                f"Failed to load output file. {ex.__class__.__name__}: {ex}\n{stack}"
            )
            self.output = ImpactZOutput(run=run_info)
            if hasattr(ex, "add_note"):
                # Python 3.11+
                ex.add_note(
                    f"\nIMPACT-Z output was:\n\n{execute_result['log']}\n(End of IMPACT-Z output)"
                )
            if raise_on_error:
                raise

        self.output.run = run_info
        if run_info.error and raise_on_error:
            raise ImpactRunFailure(f"IMPACT-Z failed to run: {run_info.error_reason}")

        return self.output

    def get_executable(self):
        """
        Gets the full path of the executable from .command, .command_mpi
        Will search environmental variables:
                ImpactZ.command_env='IMPACTZ_BIN'
                ImpactZ.command_mpi_env='IMPACTZ_MPI_BIN'
        """
        if self.use_mpi:
            return lume_tools.find_executable(
                exename=self.command_mpi, envname=self.command_mpi_env
            )
        return lume_tools.find_executable(
            exename=self.command, envname=self.command_env
        )

    def get_run_prefix(self) -> str:
        """Get the command prefix to run IMPACT-Z (e.g., 'mpirun' or 'ImpactZexe-mpi')."""
        exe = self.get_executable()

        if self.nproc != 1 and not self.use_mpi:
            self.vprint(f"Setting use_mpi = True because nproc = {self.nproc}")
            self.use_mpi = True

        if self.use_mpi:
            return self.mpi_run.format(
                nnode=self.nnode, nproc=self.nproc, command_mpi=exe
            )
        return exe

    @override
    def get_run_script(self, write_to_path: bool = True) -> str:
        """
        Assembles the run script using self.mpi_run string of the form:
            'mpirun -n {n} {command_mpi}'
        Optionally writes a file 'run' with this line to path.

        mpi_exe could be a complicated string like:
            'srun -N 1 --cpu_bind=cores {n} {command_mpi}'
            or
            'mpirun -n {n} {command_mpi}'
        """
        if self.path is None:
            raise ValueError("path (base_path) not yet set")

        runscript = shlex.join(
            [
                *shlex.split(self.get_run_prefix()),
                # *self.input.arguments,
            ]
        )
        if write_to_path:
            self.write_run_script(path=pathlib.Path(self.path) / "run")
        return runscript

    def write_run_script(self, path: Optional[AnyPath] = None) -> pathlib.Path:
        """
        Write the 'run' script which can be used in a terminal to launch IMPACT-Z.

        This is also performed automatically in `write_input` and
        `get_run_script`.

        Parameters
        -------
        path : pathlib.Path or str
            Where to write the run script.  Defaults to `{self.path}/run`.

        Returns
        -------
        pathlib.Path
            The run script location.
        """
        path = path or self.path
        if path is None:
            raise ValueError("path (base_path) not yet set and no path specified")

        path = pathlib.Path(path)
        if path.is_dir():
            path = path / "run"

        self.input.write_run_script(
            path,
            command_prefix=self.get_run_prefix(),
        )
        logger.debug("Wrote run script to %s", path)
        return path

    @override
    def write_input(
        self,
        path: Optional[AnyPath] = None,
        write_run_script: bool = True,
    ):
        """
        Write the input parameters into the file.

        Parameters
        ----------
        path : str, optional
            The directory to write the input parameters
        """
        if not self.configured:
            self.configure()

        if path is None:
            path = self.path

        if path is None:
            raise ValueError("Path has not yet been set; cannot write input.")

        path = pathlib.Path(path)
        self.input.write(workdir=path)

        if write_run_script:
            self.input.write_run_script(
                path / "run",
                command_prefix=self.get_run_prefix(),
            )

    @property
    @override
    def initial_particles(self) -> Optional[ParticleGroup]:
        """Initial particles, if defined.  Property is alias for `.input.main.initial_particles`."""
        return self.input.initial_particles

    @initial_particles.setter
    def initial_particles(
        self,
        value: Optional[ParticleGroup],
    ) -> None:
        self.input.initial_particles = value

    # @property
    # @override
    # def initial_field(self) -> Optional[FieldFile]:
    #     """Initial field, if defined.  Property is alias for `.input.main.initial_field`."""
    #     return self.input.initial_field
    #
    # @initial_field.setter
    # def initial_field(self, value: Optional[FieldFile]) -> None:
    #     self.input.initial_field = value
    #
    # def _archive(self, h5: h5py.Group):
    #     self.input.archive(h5.create_group("input"))
    #     if self.output is not None:
    #         self.output.archive(h5.create_group("output"))
    #
    @override
    def archive(self, dest: Union[AnyPath, h5py.Group]) -> None:
        """
        Archive the latest run, input and output, to a single HDF5 file.

        Parameters
        ----------
        dest : filename or h5py.Group
        """
        raise NotImplementedError()
        # if isinstance(dest, (str, pathlib.Path)):
        #     with h5py.File(dest, "w") as fp:
        #         self._archive(fp)
        # elif isinstance(dest, (h5py.File, h5py.Group)):
        #     self._archive(dest)

    to_hdf5 = archive

    #
    # def _load_archive(self, h5: h5py.Group):
    #     self.input = ImpactZInput.from_archive(h5["input"])
    #     if "output" in h5:
    #         self.output = ImpactZOutput.from_archive(h5["output"])
    #     else:
    #         self.output = None
    #
    @override
    def load_archive(self, arch: Union[AnyPath, h5py.Group]) -> None:
        """
        Load an archive from a single HDF5 file into this ImpactZ object.

        Parameters
        ----------
        arch : filename or h5py.Group
        """
        raise NotImplementedError()
        if isinstance(arch, (str, pathlib.Path)):
            with h5py.File(arch, "r") as fp:
                self._load_archive(fp)
        elif isinstance(arch, (h5py.File, h5py.Group)):
            self._load_archive(arch)

    @override
    @classmethod
    def from_archive(cls, arch: Union[AnyPath, h5py.Group]) -> ImpactZ:
        """
        Create a new ImpactZ object from an archive file.

        Parameters
        ----------
        arch : filename or h5py.Group
        """
        inst = cls()
        inst.load_archive(arch)
        return inst

    # @classmethod
    # def from_tao(
    #     cls, tao, ele_start="beginning", ele_end="end", universe=1, branch=0
    # ) -> ImpactZ:
    #     """
    #     Create a new ImpactZ object from a pytao.Tao instance
    #
    #     Parameters
    #     ----------
    #     tao : Tao
    #         The Tao instance to extract elements from.
    #     ele_start : str, optional
    #         Element to start. Defaults to "beginning".
    #     ele_end : str, optional
    #         Element to end. Defaults to "end".
    #     branch : int, optional
    #         The branch index within the specified Tao universe. Defaults to 0.
    #     universe : int, optional
    #         The universe index within the Tao object. Defaults to 1.
    #
    #     Returns
    #     -------
    #     ImpactZ
    #
    #     """
    #     input = ImpactZInput.from_tao(
    #         tao, ele_start=ele_start, universe=universe, branch=branch
    #     )
    #     lattice = Lattice.from_tao(
    #         tao, ele_start=ele_start, ele_end=ele_end, universe=universe, branch=branch
    #     )
    #     return cls(input=input, lattice=lattice)

    @override
    def load_output(
        self,
        load_fields: bool = False,
        load_particles: bool = False,
        smear: bool = True,
    ) -> ImpactZOutput:
        """
        Load the IMPACT-Z output files from disk.

        Parameters
        ----------
        load_fields : bool, default=True
            Load all field files.
        load_particles : bool, default=True
            Load all particle files.
        smear : bool, default=True
            If set, this will smear the particle phase over the sample
            (skipped) slices, preserving the modulus.

        Returns
        -------
        ImpactZOutput
        """
        return ImpactZOutput.from_input_settings(
            input=self.input,
            workdir=pathlib.Path(self.path),
            #     load_fields=load_fields,
            #     load_particles=load_particles,
            #     smear=smear,
        )

    @override
    def plot(
        self,
        y: str | Sequence[str] = ("sigma_x", "sigma_y"),
        x: str = "mean_z",
        xlim=None,
        ylim=None,
        ylim2=None,
        y2=[],
        nice=True,
        include_layout=True,
        include_labels=False,
        include_markers=True,
        include_particles=True,
        include_field=True,
        field_t=0,
        include_legend=True,
        return_figure=False,
        tex=True,
        **kwargs,
    ):
        """
        Plots output multiple keys.

        Parameters
        ----------
        y : list
            List of keys to be displayed on the Y axis
        x : str
            Key to be displayed as X axis
        xlim : list
            Limits for the X axis
        ylim : list
            Limits for the Y axis
        ylim2 : list
            Limits for the secondary Y axis
        yscale: str
            one of "linear", "log", "symlog", "logit", ... for the Y axis
        yscale2: str
            one of "linear", "log", "symlog", "logit", ... for the secondary Y axis
        y2 : list
            List of keys to be displayed on the secondary Y axis
        nice : bool
            Whether or not a nice SI prefix and scaling will be used to
            make the numbers reasonably sized. Default: True
        include_layout : bool
            Whether or not to include a layout plot at the bottom. Default: True
            Whether or not the plot should include the legend. Default: True
        return_figure : bool
            Whether or not to return the figure object for further manipulation.
            Default: True
        kwargs : dict
            Extra arguments can be passed to the specific plotting function.

        Returns
        -------
        fig : matplotlib.pyplot.figure.Figure
            The plot figure for further customizations or `None` if `return_figure` is set to False.
        """

        if self.output is None:
            raise RuntimeError(
                "IMPACT-Z has not yet been run; there is no output to plot."
            )

        if not tools.is_jupyter():
            # If not in jupyter mode, return a figure by default.
            return_figure = True

        return self.output.plot(
            y=y,
            x=x,
            xlim=xlim,
            ylim=ylim,
            ylim2=ylim2,
            y2=y2,
            nice=nice,
            include_layout=include_layout,
            include_labels=include_labels,
            include_markers=include_markers,
            include_particles=include_particles,
            include_field=include_field,
            field_t=field_t,
            include_legend=include_legend,
            return_figure=return_figure,
            tex=tex,
            **kwargs,
        )

    def stat(self, key: str):
        """
        Calculate a statistic of the beam or field along z.
        """
        if self.output is None:
            raise RuntimeError(
                "IMPACT-Z has not yet been run; there is no output to get statistics from."
            )
        return self.output.stat(key=key)

    @override
    @staticmethod
    def input_parser(path: AnyPath) -> ImpactZInput:
        """
        Invoke the specialized main input parser and returns the `ImpactZInput`
        instance.

        Parameters
        ----------
        path : str or pathlib.Path
            Path to the main input file.

        Returns
        -------
        ImpactZInput
        """
        return ImpactZInput.from_file(path)

    @override
    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, ImpactZ):
            return False
        return self.input == other.input and self.output == other.output

    @override
    def __ne__(self, other: Any) -> bool:
        if not isinstance(other, ImpactZ):
            return False
        return self.input != other.input or self.output != other.output
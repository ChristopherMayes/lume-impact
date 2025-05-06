from __future__ import annotations

import logging
import os
import pathlib
import platform
import re
import shlex
import shutil
import traceback
import typing
from collections.abc import Sequence
from contextlib import contextmanager
from time import monotonic
from typing import Any, ClassVar, NamedTuple

import h5py
import numpy as np
from lume import tools as lume_tools
from lume.base import CommandWrapper
from pmd_beamphysics import ParticleGroup
from pmd_beamphysics.units import pmd_unit
from typing_extensions import override

from . import tools
from . import units as units_mod
from .constants import IntegratorType
from .errors import ImpactRunFailure
from .input import ImpactZInput
from .output import ImpactZOutput, RunInfo
from .particles import ImpactZParticles
from .tools import is_jupyter, read_if_path
from .types import AnyPath

if typing.TYPE_CHECKING:
    from .interfaces.bmad import Tao
    from .interfaces.bmad import Which as TaoWhich

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


_run_update_regex = re.compile(
    r"^enter elment \(type code\):\s*(?P<element_index>-?\d+)\s+(?P<type_code>-?\d+)$"
)


class _RunUpdate(NamedTuple):
    # enter elment (type code):            1          -2
    element_index: int
    type_code: int

    @classmethod
    def from_line(cls, line: str) -> _RunUpdate | None:
        match = _run_update_regex.match(line.strip())
        if not match:
            return None

        group = match.groupdict()
        return _RunUpdate(
            element_index=int(group["element_index"]),
            type_code=int(group["type_code"]),
        )


@contextmanager
def _maybe_progress_bar(enable: bool, **kwargs):
    if is_jupyter():
        from tqdm.notebook import tqdm
    else:
        from tqdm import tqdm

    if enable:
        with tqdm(**kwargs) as pbar:
            yield pbar
    else:
        yield None


@contextmanager
def _verbosity_context(I: ImpactZ, verbose: bool | None):
    if verbose is None:
        yield
    else:
        orig_verbose = I.verbose
        I.verbose = verbose
        yield
        I.verbose = orig_verbose


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
    WORKDIR: ClassVar[str | None] = find_workdir()

    # Environmental variables to search for executables
    command_env: str = "IMPACTZ_BIN"
    command_mpi_env: str = "IMPACTZ_MPI_BIN"
    original_path: AnyPath

    _input: ImpactZInput
    output: ImpactZOutput | None

    def __init__(
        self,
        input: ImpactZInput | str | pathlib.Path | None = None,
        *,
        workdir: str | pathlib.Path | None = None,
        output: Any | None = None,  # TODO
        alias: dict[str, str] | None = None,
        units: dict[str, pmd_unit] | None = None,
        command: str | None = None,
        command_mpi: str | None = None,
        use_mpi: bool = False,
        mpi_run: str = "",
        use_temp_dir: bool = True,
        verbose: bool = tools.global_display_options.verbose >= 1,
        timeout: float | None = None,
        initial_particles: ParticleGroup | ImpactZParticles | None = None,
        file_data: dict[str, np.ndarray] | None = None,
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
        elif isinstance(input, (pathlib.Path, str)):
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

        input.file_data.update(file_data or {})

        self._input = input
        self.output = output

        if (
            input.initial_particles is not initial_particles
            and initial_particles is not None
        ):
            self.initial_particles = initial_particles

        if workdir is None:
            workdir = pathlib.Path(".")

        self.original_path = workdir

        if pathlib.Path(workdir).exists() and not pathlib.Path(workdir).is_dir():
            raise ValueError(f"`workdir` exists and is not a directory: {workdir}")

        # Internal
        self._units = dict(units or units_mod.known_unit)
        self._alias = dict(alias or {})

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
        return self.input.nproc

    @nproc.setter
    def nproc(self, nproc: int | None):
        self.input.nproc = nproc

    @property
    def use_mpi(self):
        """
        Whether or not MPI should be used if supported.
        """
        return self.nproc != 1

    @use_mpi.setter
    def use_mpi(self, use_mpi: bool) -> None:
        if not use_mpi:
            self.nproc = 1

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
        raise_on_error: bool = True,
        verbose: bool | None = None,
        progress_bar: bool = True,
    ) -> ImpactZOutput:
        """
        Execute IMPACT-Z with the configured input settings.

        Parameters
        ----------
        raise_on_error : bool, default=True
            If it fails to run, raise an error. Depending on the error,
            output information may still be accessible in the ``.output``
            attribute.

        Returns
        -------
        ImpactZOutput
            The output data.  This is also accessible as ``.output``.
        """

        by_z = self.input.by_z

        def update_progress_bar(line: str):
            assert pbar is not None

            update = _RunUpdate.from_line(line)
            if update is None:
                return
            try:
                ele = by_z[update.element_index - 1]
            except IndexError:
                pass
            else:
                pbar.set_postfix(
                    {
                        "Name": ele.ele.name,
                        "Z": ele.z_start,
                    },
                    refresh=False,
                )
            pbar.n = update.element_index
            pbar.refresh()

        def run():
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
                return execute_result

            log = []
            try:
                for line in tools.execute(shlex.split(runscript), cwd=self.path):
                    if pbar is not None:
                        update_progress_bar(line)

                    self.vprint(line, end="")
                    log.append(line)
            except Exception as ex:
                log.append(f"IMPACT-Z exited with an error: {ex}")
                self.vprint(log[-1])
                if pbar is not None:
                    try:
                        pbar.leave = True
                    except Exception:
                        logger.debug("Failed to set pbar.leave", exc_info=True)
                return {
                    "log": "".join(log),
                    "error": True,
                    "why_error": "error",
                }
            return {
                "log": "".join(log),
                "error": False,
                "why_error": "",
            }

        with _verbosity_context(self, verbose):
            if not self.configured:
                self.configure()

            if self.path is None:
                raise ValueError("Path (base_path) not yet set")

            self.finished = False

            runscript = self.get_run_script()

            start_time = monotonic()
            mpi = "with MPI " if self.use_mpi else ""
            self.vprint(f"Running Impact-Z {mpi}in {self.path}")
            self.vprint(runscript)

            self.write_input()

            with _maybe_progress_bar(
                progress_bar,
                total=len(self.input.lattice),
                leave=False,
            ) as pbar:
                execute_result = run()

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
            self.vprint(
                f"{success_or_failure} - execution took {run_info.run_time:0.2f}s."
            )

            try:
                self.output = self.load_output()
            except Exception as ex:
                stack = traceback.format_exc()
                run_info.error = True
                run_info.error_reason = f"Failed to load output file. {ex.__class__.__name__}: {ex}\n{stack}"
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
                raise ImpactRunFailure(
                    f"IMPACT-Z failed to run: {run_info.error_reason}"
                )

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

    def write_run_script(self, path: AnyPath | None = None) -> pathlib.Path:
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
        path: AnyPath | None = None,
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
        self.input.write(workdir=path, check=True)

        if write_run_script:
            self.input.write_run_script(
                path / "run",
                command_prefix=self.get_run_prefix(),
            )

    @property
    @override
    def initial_particles(self) -> ParticleGroup | None:
        """Initial particles, if defined.  Property is alias for `.input.main.initial_particles`."""
        return self.input.initial_particles

    @initial_particles.setter
    def initial_particles(
        self,
        value: ParticleGroup | ImpactZParticles | None,
    ) -> None:
        if isinstance(value, ImpactZParticles):
            value = value.to_particle_group(
                reference_frequency=self.input.reference_frequency,
                reference_kinetic_energy=self.input.reference_kinetic_energy,
                phase_reference=self.input.initial_phase_ref,
            )
        self.input.initial_particles = value

    def _archive(self, h5: h5py.Group):
        self.input.archive(h5.create_group("input"))
        if self.output is not None:
            self.output.archive(h5.create_group("output"))

    @override
    def archive(self, dest: AnyPath | h5py.Group) -> None:
        """
        Archive the latest run, input and output, to a single HDF5 file.

        Parameters
        ----------
        dest : filename or h5py.Group
        """
        if isinstance(dest, (str, pathlib.Path)):
            with h5py.File(dest, "w") as fp:
                self._archive(fp)
        elif isinstance(dest, (h5py.File, h5py.Group)):
            self._archive(dest)
        else:
            raise NotImplementedError(type(dest))

    to_hdf5 = archive

    def _load_archive(self, h5: h5py.Group):
        self.input = ImpactZInput.from_archive(h5["input"])
        if "output" in h5:
            self.output = ImpactZOutput.from_archive(h5["output"])
        else:
            self.output = None

    @override
    def load_archive(self, arch: AnyPath | h5py.Group) -> None:
        """
        Load an archive from a single HDF5 file into this ImpactZ object.

        Parameters
        ----------
        arch : filename or h5py.Group
        """
        if isinstance(arch, (str, pathlib.Path)):
            with h5py.File(arch, "r") as fp:
                self._load_archive(fp)
        elif isinstance(arch, (h5py.File, h5py.Group)):
            self._load_archive(arch)

    @override
    @classmethod
    def from_archive(cls, arch: AnyPath | h5py.Group) -> ImpactZ:
        """
        Create a new ImpactZ object from an archive file.

        Parameters
        ----------
        arch : filename or h5py.Group
        """
        inst = cls()
        inst.load_archive(arch)
        return inst

    @override
    def load_output(self) -> ImpactZOutput:
        """
        Load the IMPACT-Z output files from disk.

        Returns
        -------
        ImpactZOutput
        """
        return ImpactZOutput.from_input_settings(
            input=self.input,
            workdir=pathlib.Path(self.path),
        )

    @override
    def plot(
        self,
        y: str | Sequence[str] = ("sigma_x", "sigma_y"),
        x: str = "z",
        xlim=None,
        ylim=None,
        ylim2=None,
        y2=[],
        nice=True,
        include_layout=True,
        include_labels=False,
        include_markers=True,
        include_particles=True,
        # include_field=True,
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
        include_legend : bool
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
            input=self.input,
            nice=nice,
            include_layout=include_layout,
            include_labels=include_labels,
            include_markers=include_markers,
            include_particles=include_particles,
            # include_field=include_field,
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

    @classmethod
    def from_tao(
        cls,
        tao: Tao,
        track_start: str | None = None,
        track_end: str | None = None,
        *,
        radius_x: float = 0.0,
        radius_y: float = 0.0,
        ncpu_y: int = 1,
        ncpu_z: int = 1,
        nx: int = 64,
        ny: int = 64,
        nz: int = 64,
        which: TaoWhich = "model",
        ix_uni: int = 1,
        ix_branch: int = 0,
        reference_frequency: float = 1300000000.0,
        verbose: bool = False,
        initial_particles_file_id: int = 100,
        final_particles_file_id: int = 101,
        initial_rfdata_file_id: int = 500,
        initial_write_full_id: int = 200,
        write_beam_eles: str | Sequence[str] = ("monitor::*", "marker::*"),
        include_collimation: bool = False,
        integrator_type: IntegratorType = IntegratorType.linear_map,
        workdir: str | pathlib.Path | None = None,
        output: Any | None = None,  # TODO
        alias: dict[str, str] | None = None,
        units: dict[str, pmd_unit] | None = None,
        command: str | None = None,
        command_mpi: str | None = None,
        use_mpi: bool = False,
        mpi_run: str = "",
        use_temp_dir: bool = True,
        timeout: float | None = None,
        initial_particles: ParticleGroup | ImpactZParticles | None = None,
    ):
        """
        Create an ImpactZ object from a Tao instance's lattice.

        This function converts a Tao model into an ImpactZInput by extracting the
        relevant lattice and particle information, and packages it into a structure
        suitable for running IMPACT-Z simulations.

        Standard `ImpactZ` arguments (aside from the auto-generated `input`)
        will be passed to the new instance.

        Parameters
        ----------
        tao : Tao
            The Tao instance.
        track_start : str or None, optional
            Name of the element in the Tao model where tracking begins.
            If None, defaults to the first element.
        track_end : str or None, optional
            Name of the element in the Tao model where tracking ends.
            If None, defaults to the last element.
        radius_x : float, optional
            The transverse aperture radius in the x-dimension.
        radius_y : float, optional
            The transverse aperture radius in the y-dimension.
        ncpu_y : int, optional
            Number of processor divisions along the y-axis.
        ncpu_z : int, optional
            Number of processor divisions along the z-axis.
        nx : int, optional
            Space charge grid mesh points along the x-axis.
        ny : int, optional
            Space charge grid mesh points along the y-axis.
        nz : int, optional
            Space charge grid mesh points along the z-axis.
        which : "model", "base", or "design", optional
            Specifies the source of lattice data used from Tao.
        ix_uni : int, optional
            The universe index.
        ix_branch : int, optional
            The branch index.
        reference_frequency : float, optional
            The reference frequency for IMPACT-Z.
        verbose : bool, optional
            If True, prints additional diagnostic information.
        initial_particles_file_id : int, optional
            File ID for the initial particle distribution.
        final_particles_file_id : int, optional
            File ID for the final particle distribution.
        initial_rfdata_file_id : int, optional
            File ID for the first RF data file.
        initial_write_full_id : int, optional
            File ID for the first WriteFull instance.
        write_beam_eles : str or Sequence[str], optional
            Element(s) by name or Tao-supported match to use at which to write
            particle data via `WriteFull`.
        include_collimation : bool, optional
            If True, includes collimation elements in the lattice conversion.
            Defaults to False as this doesn't work quite yet.
        integrator_type : IntegratorType, optional
            The integrator scheme to be used in the lattice conversion.
            Defaults to 'linear_map', but this may be switched automatically to
            Runge-Kutta depending on IMPACT-Z run requirements.

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
        timeout : float, default=None
            The timeout in seconds to be used when running IMPACT-Z.
        initial_particles : ParticleGroup, optional
            Initial particles to use in the simulation, using the
            OpenPMD-beamphysics standard.

        Returns
        -------
        ImpactZInput
        """
        input = ImpactZInput.from_tao(
            tao=tao,
            track_start=track_start,
            track_end=track_end,
            radius_x=radius_x,
            radius_y=radius_y,
            ncpu_y=ncpu_y,
            ncpu_z=ncpu_z,
            nx=nx,
            ny=ny,
            nz=nz,
            which=which,
            ix_uni=ix_uni,
            ix_branch=ix_branch,
            reference_frequency=reference_frequency,
            verbose=verbose,
            initial_particles_file_id=initial_particles_file_id,
            final_particles_file_id=final_particles_file_id,
            initial_rfdata_file_id=initial_rfdata_file_id,
            initial_write_full_id=initial_write_full_id,
            write_beam_eles=write_beam_eles,
            include_collimation=include_collimation,
            integrator_type=integrator_type,
        )
        return cls(
            input=input,
            workdir=workdir,
            output=output,
            alias=alias,
            units=units,
            command=command,
            command_mpi=command_mpi,
            use_mpi=use_mpi,
            mpi_run=mpi_run,
            use_temp_dir=use_temp_dir,
            verbose=verbose,
            timeout=timeout,
            initial_particles=initial_particles,
        )

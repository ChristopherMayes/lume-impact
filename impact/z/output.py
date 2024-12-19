from __future__ import annotations

import logging
import pathlib
import typing
from typing import Any, Dict, Generator, Optional, TypeVar

import numpy as np
import pydantic
import pydantic.alias_generators
from typing_extensions import override

from . import parsers
from .input import ImpactZInput
from .types import AnyPath, BaseModel, OutputDataType, SequenceBaseModel

try:
    from collections.abc import Mapping
except ImportError:
    pass

if typing.TYPE_CHECKING:
    pass


logger = logging.getLogger(__name__)


class RunInfo(BaseModel):
    """
    Impact-Z run information.

    Attributes
    ----------
    error : bool
        True if an error occurred during the Impact-Z run.
    error_reason : str or None
        Error explanation, if `error` is set.
    run_script : str
        The command-line arguments used to run Impact-Z
    output_log : str
        Impact-Z output log
    start_time : float
        Start time of the process
    end_time : float
        End time of the process
    run_time : float
        Wall clock run time of the process
    """

    error: bool = pydantic.Field(
        default=False, description="`True` if an error occurred during the Impact-Z run"
    )
    error_reason: Optional[str] = pydantic.Field(
        default=None, description="Error explanation, if `error` is set."
    )
    run_script: str = pydantic.Field(
        default="", description="The command-line arguments used to run Impact-Z"
    )
    output_log: str = pydantic.Field(
        default="", repr=False, description="Impact-Z output log"
    )
    start_time: float = pydantic.Field(
        default=0.0, repr=False, description="Start time of the process"
    )
    end_time: float = pydantic.Field(
        default=0.0, repr=False, description="End time of the process"
    )
    run_time: float = pydantic.Field(
        default=0.0, description="Wall clock run time of the process"
    )

    @property
    def success(self) -> bool:
        """`True` if the run was successful."""
        return not self.error


def _empty_ndarray():
    return np.zeros(0)


class _OutputBase(BaseModel):
    """Output model base class."""

    # units: Dict[str, PydanticPmdUnit] = pydantic.Field(default_factory=dict, repr=False)
    extra: Dict[str, OutputDataType] = pydantic.Field(
        default_factory=dict, description="Additional output data."
    )


class ImpactZOutput(Mapping, BaseModel, arbitrary_types_allowed=True):
    """
    IMPACT-Z command output.

    Attributes
    ----------
    run : RunInfo
        Execution information - e.g., how long did it take and what was
        the output from Impact-Z.
    alias : dict[str, str]
        Dictionary of aliased data keys.
    """

    run: RunInfo = pydantic.Field(
        default_factory=RunInfo,
        description="Run-related information - output text and timing.",
    )
    files: dict[int, np.ndarray] = {}

    @override
    def __eq__(self, other: Any) -> bool:
        return BaseModel.__eq__(self, other)

    @override
    def __ne__(self, other: Any) -> bool:
        return BaseModel.__ne__(self, other)

    @override
    def __getitem__(self, key: str) -> Any:
        """Support for Mapping -> easy access to data."""
        # _check_for_unsupported_key(key)

        # parent, array_attr = self._split_parent_and_attr(key)
        # return getattr(parent, array_attr)
        # return None

    @override
    def __iter__(self) -> Generator[str, None, None]:
        """Support for Mapping -> easy access to data."""
        yield from self.alias

    @override
    def __len__(self) -> int:
        """Support for Mapping -> easy access to data."""
        return len(self.alias)

    @classmethod
    def from_input_settings(
        cls,
        input: ImpactZInput,
        workdir: pathlib.Path,
        load_fields: bool = False,
        load_particles: bool = False,
        smear: bool = True,
    ) -> ImpactZOutput:
        """
        Load ImpactZ output based on the configured input settings.

        Parameters
        ----------
        load_fields : bool, default=False
            After execution, load all field files.
        load_particles : bool, default=False
            After execution, load all particle files.
        smear : bool, default=True
            If set, for particles, this will smear the phase over the sample
            (skipped) slices, preserving the modulus.

        Returns
        -------
        ImpactZOutput
            The output data.
        """
        files = {}
        for fnum, cls in file_number_to_cls.items():
            fn = workdir / f"fort.{fnum}"
            if fn.exists():
                files[fnum] = cls.from_file(fn)
        return ImpactZOutput(files=files)
        # output_filename = cls.get_output_filename(input, workdir)
        # return cls.from_files(
        #     output_filename,
        #     load_fields=load_fields,
        #     load_particles=load_particles,
        #     smear=smear,
        # )
        #
        #


file_number_to_cls = {}
T = TypeVar("T", bound="FortranOutputFileData")


class FortranOutputFileData(SequenceBaseModel):
    def __init_subclass__(cls, file_id: int, **kwargs):
        super().__init_subclass__(**kwargs)

        assert isinstance(file_id, int)
        assert file_id not in file_number_to_cls, f"Duplicate element ID {file_id}"
        file_number_to_cls[file_id] = cls

    @classmethod
    def from_file(cls: type[T], filename: AnyPath) -> np.ndarray:
        items = []
        with open(filename, "rt") as fp:
            for line in fp.read().splitlines():
                data = parsers.parse_input_line(line)
                items.append(data)

        dtype = cls.model_dtype()
        return np.asarray(items, dtype=dtype)

    @classmethod
    def model_dtype(cls):
        return np.dtype([(fld, np.float64) for fld in cls.model_fields])

    def to_numpy(self) -> np.ndarray:
        values = [getattr(self, attr) for attr in self.model_fields]
        return np.ndarray(values, dtype=self.model_dtype())


class File18(FortranOutputFileData, file_id=18):
    """
    write(18,99)z,this%refptcl(5),gam,energy,bet,sqrt(glrmax)*xl
    """

    z: float
    ref_particle_attribute: float
    gamma: float
    energy: float
    beta: float
    sqrt_glrmax_times_xl: float


class File24(FortranOutputFileData, file_id=24):
    """
    write(24,100)z,x0*xl,xrms*xl,px0/gambet,pxrms/gambet,-xpx/epx,epx*xl
    """

    z: float
    x0: float
    xrms: float
    px0_over_gambet: float
    pxrms_over_gambet: float
    neg_xpx_over_epx: float
    epx_times_xl: float


class File25(FortranOutputFileData, file_id=25):
    """
    write(25,100)z,y0*xl,yrms*xl,py0/gambet,pyrms/gambet,-ypy/epy,epy*xl
    """

    z: float
    y0: float
    yrms: float
    py0_over_gambet: float
    pyrms_over_gambet: float
    neg_ypy_over_epy: float
    epy_times_xl: float


class File26(FortranOutputFileData, file_id=26):
    """
    write(26,100)z,z0*xt,zrms*xt,pz0*qmc,pzrms*qmc,-zpz/epz,epz*qmc*xt
    """

    z: float
    z0: float
    zrms: float
    pz0_times_qmc: float
    pzrms_times_qmc: float
    neg_zpz_over_epz: float
    epz_times_qmc_xt: float


class File27(FortranOutputFileData, file_id=27):
    """
    write(27,100)z,glmax(1)*xl,glmax(2)/gambet,glmax(3)*xl,&
                 glmax(4)/gambet,glmax(5)*xt,glmax(6)*qmc
    """

    z: float
    glmax1_times_xl: float
    glmax2_over_gambet: float
    glmax3_times_xl: float
    glmax4_over_gambet: float
    glmax5_times_xt: float
    glmax6_times_qmc: float


class File28(FortranOutputFileData, file_id=28):
    """
    write(28,101)z,npctmin,npctmax,nptot
    """

    z: float
    npctmin: float
    npctmax: float
    nptot: float


class File29(FortranOutputFileData, file_id=29):
    """
    write(29,100)z,x03*xl,px03/gambet,y03*xl,py03/gambet,z03*xt,&
                 pz03*qmc
    """

    z: float
    x03: float
    px03_over_gambet: float
    y03: float
    py03_over_gambet: float
    z03: float
    pz03_times_qmc: float


class File30(FortranOutputFileData, file_id=30):
    """
    write(30,100)z,x04*xl,px04/gambet,y04*xl,py04/gambet,z04*xt,&
                 pz04*qmc
    """

    z: float
    x04: float
    px04_over_gambet: float
    y04: float
    py04_over_gambet: float
    z04: float
    pz04_times_qmc: float


class File32(FortranOutputFileData, file_id=32):
    """
    write(32,*)z,nptlist(1:nchrg)
    """

    z: float
    nptlist: list[float]

    @classmethod
    def from_file(cls, filename: AnyPath) -> np.ndarray:
        items = []
        with open(filename, "rt") as fp:
            for line in fp.read().splitlines():
                z, *nptlist = parsers.parse_input_line(line)
                items.append(np.array([z, *nptlist]))

        if not items:
            return np.zeros((0,))

        npts = len(items[-1]) - 1
        dtype = [("z", np.float64), *[(f"pt{n}", np.float64) for n in range(npts)]]
        return np.asarray(items, dtype=dtype)

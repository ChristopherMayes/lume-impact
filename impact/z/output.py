from __future__ import annotations

import logging
import pathlib
import typing
from typing import Any, Dict, Generator, Optional

import numpy as np
import pydantic
import pydantic.alias_generators
from typing_extensions import override

from ..parsers import load_many_fort

from .input import ImpactZInput
from .types import BaseModel, OutputDataType

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
        print(load_many_fort(workdir, verbose=True))
        return ImpactZOutput()
        # output_filename = cls.get_output_filename(input, workdir)
        # return cls.from_files(
        #     output_filename,
        #     load_fields=load_fields,
        #     load_particles=load_particles,
        #     smear=smear,
        # )

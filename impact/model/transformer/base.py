from abc import ABC, abstractmethod
from typing import Any


class Transformer(ABC):
    """Abstract base class for property transformers."""

    @abstractmethod
    def get_property(self, tool: Any, name: str) -> Any:
        """Return the current value of the named property from *tool*."""

    @abstractmethod
    def set_property(self, tool: Any, name: str, value: Any) -> None:
        """Write *value* to the named property on *tool*."""

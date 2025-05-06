class ImpactZError(Exception):
    pass


class ImpactRunFailure(ImpactZError):
    """Impact-{T,Z} failed to run."""


class ElementAccessError(ImpactZError):
    pass


class NoSuchElementError(ElementAccessError):
    """No element of the given type defined in the lattice."""


class MultipleElementError(ElementAccessError):
    """
    More than one element of the given type is defined.

    Access is ambiguous.
    """

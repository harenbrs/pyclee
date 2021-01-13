from enum import Enum, auto
from typing import NewType, Iterable
import warnings

try:
    from ordered_set_37 import OrderedSet as Set
except ImportError:
    warnings.warn(
        "ordered_set_37 not available - falling back to built-in unordered set, expect"
        " non-repeatability"
    )
    Set = set


Element = NewType('Element', Iterable[float])
Timestamp = NewType('Timestamp', float)
Interval = NewType('Interval', float)


class UnsupportedConfiguration(UserWarning):
    pass


class SpatialIndexMethod(Enum):
    KDTREE = auto()
    RTREE = auto()

from enum import Enum, auto
from typing import NewType, Iterable


Element = NewType('Element', Iterable)
Timestamp = NewType('Timestamp', float)
Interval = NewType('Interval', float)


class UnsupportedConfiguration(UserWarning):
    pass


class SpatialIndexMethod(Enum):
    KDTREE = auto()
    RTREE = auto()

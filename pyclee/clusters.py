from __future__ import annotations
from typing import TYPE_CHECKING, Optional, Union, Iterable

import numpy as np

try:
    from sklearn.neighbors import KDTree
except ImportError:
    KDTree = None

from .types import Set, SpatialIndexMethod

if TYPE_CHECKING:
    from .dyclee import DyCleeContext, RTreeIndex
    from .types import Element, Timestamp


class MicroCluster:
    def __init__(
        self,
        element: Element,
        time: Timestamp,
        context: DyCleeContext,
        index: int,
        label: Optional[int] = None
    ):
        self.n_elements: Union[int, float] = 1
        self.linear_sum: np.ndarray = np.copy(element)
        self.sum_squares: np.ndarray = np.power(element, 2)
        self.first_time = time
        self.last_time = time
        self.context = context
        self.index = index
        self.label = label
        self.elements: Set[Element] = (
            Set([tuple(element)]) if self.context.store_elements else Set()
        )
        self.once_dense = False
    
    @property
    def centroid(self) -> np.ndarray:
        # Time dependence not required, as both numerator and denominator are forgotten
        # according the same multiplicative factor
        return self.linear_sum/self.n_elements
    
    def forgetting_factor(self, time: Timestamp) -> float:
        return self.context.forgetting_method(time - self.last_time)
    
    def density(self, time: Timestamp) -> float:
        return self.n_elements*self.forgetting_factor(time)/self.context.hyperbox_volume
    
    @property
    def bounding_box(self) -> np.ndarray:
        """Returns (x1min, x2min, ..., xnmin, x1max, x2max, ..., xnmax)"""
        return (
            self.centroid[None, :]
            + [-self.context.hyperbox_lengths/2, self.context.hyperbox_lengths/2]
        ).ravel()
    
    def add(self, element: Element, time: Timestamp):
        factor = self.forgetting_factor(time)
        
        self.n_elements *= factor
        self.n_elements += 1
        
        self.linear_sum *= factor
        self.linear_sum += element
        
        self.sum_squares *= factor
        self.sum_squares += np.power(element, 2)
        
        self.last_time = time
        
        if self.context.store_elements:
            self.elements.add(tuple(element))
    
    def distance(self, element: Element) -> float:
        return np.linalg.norm(np.asarray(element) - self.centroid, 1)
    
    def is_reachable(self, element: Element) -> bool:
        return (
            np.linalg.norm(np.asarray(element) - self.centroid, np.inf)
            < self.context.reachable_radius
        )
    
    def is_directly_connected(self, other: MicroCluster) -> bool:
        return (
            np.linalg.norm(other.centroid - self.centroid, np.inf)
            < self.context.connected_radius
        )
    
    def get_neighbours(
        self,
        µclusters: Iterable[MicroCluster],
        index: Optional[RTreeIndex],
        µcluster_map: Optional[dict[int, MicroCluster]]
    ) -> Set[MicroCluster]:
        µclusters = list(µclusters)
        
        # We're looking for neighbours, so exclude the current microcluster
        if self in µclusters:
            µclusters.remove(self)
        
        if not µclusters:
            return Set()
        
        if (
            self.context.density_index == SpatialIndexMethod.RTREE
            and index is not None
            and µcluster_map is not None
        ):
            return Set(
                [
                    µcluster_map[hash_]
                    for hash_ in index.intersection(self.bounding_box)
                    if µcluster_map[hash_] in µclusters
                ]
            )
        elif self.context.density_index == SpatialIndexMethod.KDTREE:
            tree = KDTree(
                np.row_stack([µcluster.centroid for µcluster in µclusters]), p=np.inf
            )
            
            # Neighbours are defined by the infinity norm
            idcs, = tree.query_radius(
                self.centroid.reshape(1, -1), self.context.connected_radius
            )
            
            return Set([µclusters[i] for i in idcs])
        else:
            # Brute force
            return Set(
                [
                    µcluster
                    for µcluster in µclusters
                    if np.linalg.norm(µcluster.centroid - self.centroid, np.inf)
                    < self.context.connected_radius
                ]
            )


class Cluster:
    def __init__(self, µcluster: MicroCluster):
        self.µclusters = Set([µcluster])
        self.label = µcluster.label
    
    def centroid(self, time: Timestamp) -> np.ndarray:
        return sum(
            µ.centroid*µ.n_elements*µ.forgetting_factor(time) for µ in self.µclusters
        )/self.n_elements(time)
    
    def n_elements(self, time: Timestamp) -> np.ndarray:
        return sum(µ.n_elements*µ.forgetting_factor(time) for µ in self.µclusters)
    
    def add(self, µcluster: MicroCluster):
        self.µclusters.add(µcluster)

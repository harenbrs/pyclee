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
        self.squared_sum: np.ndarray = np.power(element, 2)
        self.first_time = time
        self.last_time = time
        self.context = context
        self.index = index
        self.label = label
        self.times: list[Timestamp] = [time] if self.context.store_times else []
        self.elements: list[Element] = (
            [tuple(element)] if self.context.store_elements else []
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
        
        self.squared_sum *= factor
        self.squared_sum += np.power(element, 2)
        
        self.last_time = time
        
        if self.context.store_times:
            self.times.append(time)
        
        if self.context.store_elements:
            self.elements.append(tuple(element))
    
    def distance(self, element: Element) -> float:
        return np.linalg.norm(np.asarray(element) - self.centroid, 1)
    
    def is_reachable(self, element: Element) -> bool:
        return np.all(
            abs(np.asarray(element) - self.centroid) < self.context.hyperbox_lengths/2
        )
    
    def is_directly_connected(self, other: MicroCluster) -> bool:
        return (
            sum(abs(other.centroid - self.centroid) < self.context.hyperbox_lengths)
            >= self.context.n_features - self.context.uncommon_dimensions
        )
    
    def get_neighbours(
        self,
        µclusters: Iterable[MicroCluster],
        rtree_index: Optional[RTreeIndex],
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
            and rtree_index is not None
            and µcluster_map is not None
        ):
            return Set(
                [
                    µcluster_map[hash_]
                    for hash_ in rtree_index.intersection(self.bounding_box)
                    if µcluster_map[hash_] in µclusters
                    and self.is_directly_connected(µcluster_map[hash_])
                ]
            )
        elif self.context.density_index == SpatialIndexMethod.KDTREE:
            tree = KDTree(
                np.row_stack([µcluster.centroid for µcluster in µclusters]), p=np.inf
            )
            
            # Potential neighbours are defined by the infinity norm
            idcs, = tree.query_radius(
                self.centroid.reshape(1, -1), self.context.potential_neighbour_radius
            )
            
            return Set(
                [µclusters[i] for i in idcs if self.is_directly_connected(µclusters[i])]
            )
        else:
            # Brute force
            return Set(
                [
                    µcluster
                    for µcluster in µclusters
                    if self.is_directly_connected(µcluster)
                ]
            )


class Cluster:
    def __init__(self, µcluster: MicroCluster, time: Timestamp):
        self.µclusters = Set([µcluster])
        self.label = µcluster.label
        self.linear_sum = (
            µcluster.centroid*µcluster.n_elements*µcluster.forgetting_factor(time)
        )
        self.n_elements = µcluster.n_elements*µcluster.forgetting_factor(time)
    
    @property
    def centroid(self) -> np.ndarray:
        return self.linear_sum/self.n_elements
    
    def add(self, µcluster: MicroCluster, time: Timestamp):
        self.µclusters.add(µcluster)
        self.linear_sum += (
            µcluster.centroid*µcluster.n_elements*µcluster.forgetting_factor(time)
        )
        self.n_elements += µcluster.n_elements*µcluster.forgetting_factor(time)
    
    @property
    def times(self):
        return sum([µ.times for µ in self.µclusters], [])
    
    @property
    def elements(self):
        return sum([µ.elements for µ in self.µclusters], [])

from __future__ import annotations
from typing import Iterable, Union, Optional
import warnings

import numpy as np

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

try:
    from rtree.index import Index as RTreeIndex, Property as RTreeProperty
except ImportError:
    RTreeIndex = None

try:
    from sklearn.neighbors import KDTree
except ImportError:
    KDTree = None

from .clusters import MicroCluster, Cluster
from .forgetting import ForgettingMethod, NoForgettingMethod
from .types import (
    Element,
    Timestamp,
    Interval,
    Set,
    UnsupportedConfiguration,
    SpatialIndexMethod
)


class DyCleeContext:
    def __init__(
        self,
        n_features: int,
        hyperbox_fractions: Union[float, Iterable[float]],
        feature_ranges: Iterable[[float, float]],
        *,
        update_ranges: bool = True,
        uncommon_dimensions: int = 0,
        forgetting_method: Optional[ForgettingMethod] = None,
        long_term_memory: bool = False,
        outlier_rejection: bool = True,
        sparse_rejection: bool = False,
        multi_density: bool = False,
        density_interval: Interval = 1.0,
        distance_index: Optional[SpatialIndexMethod] = SpatialIndexMethod.RTREE,
        density_index: Optional[SpatialIndexMethod] = SpatialIndexMethod.RTREE,
        store_elements: bool = False
    ):
        """
        Creates a DyCleeContext object for use with the DyClee class.
        
        ### Parameters
         - `n_features: int`
            Number of features/dimensions of the input data.
         - `hyperbox_fractions: float | Iterable[float]`
            Relative size of each dimension of the microclusters' hyperboxes, as a
            fraction of the total range of each dimension of the input data.
            If a scalar is given, the same fraction is used for all dimensions.
         - `feature_ranges: Iterable[[float, float]]`
            Range of each dimension of the input data in the form:
            `[(xmin, xmax), (ymin, ymax), ...]`
         - `update_ranges: bool`
            This flag controls whether the feature ranges and derived parameters are to
            be automatically updated at each timestep. Defaults to `True`.
         - `uncommon_dimensions: int`
            Number of dimensions to ignore for microcluster connectedness calculations.
            Ignored if `density_method == SpatialIndexMethod.RTREE`. Defaults to `0`.
         - `forgetting_method: Optional[ForgettingMethod]`
            Function that will be applied to microclusters' element accumulators to
            "forget" older samples (as a function of time intervals). `None` implies
            unlimited temporal memory. Defaults to `None`.
         - `long_term_memory: bool`
            TODO. Defaults to `False`.
         - `outlier_rejection: bool`
            TODO. Defaults to `True`.
         - `sparse_rejection: bool`
            TODO. Defaults to `False`.
         - `multi_density: bool`
            TODO. Defaults to `False`.
         - `density_interval: Interval`
            Controls how many timesteps pass between applications of the density-based
            clustering stage. Increasing this may help with performance. Set to `0` to
            enforce density-based clustering after every timestep, no matter how small.
            Defaults to `1.0`.
         - `distance_index: Optional[SpatialIndexMethod]`
            Which spatial indexing to use for the distance-based stage. The options are:
              - `SpatialIndexMethod.RTREE` (default): Maintains a single R*-tree which
                is updated with changes to the microcluster structure as they occur.
                Best scaling performance. Requires `rtree`.
              - `SpatialIndexMethod.KDTREE`: Builds a single-use KD-Tree every time a
                microcluster search is required. Requires `scikit-learn`.
              - `None`: Performs a brute-force search (useful for debugging).
         - `density_index: Optional[SpatialIndexMethod]`
            See options for `distance_index`. If `SpatialIndexMethod.KDTREE` is
            selected for both stages, a single R*-tree is shared between the stages for
            efficiency. Note that `uncommon_dimensions` will be treated as zero if
            `SpatialIndexMethod.KDTREE` is selected, due to incompatibility.
            Defaults to `SpatialIndexMethod.KDTREE`.
         - `store_elements: bool`
            Whether to store each input element in its corresponding microcluster.
            If `False`, microclusters only maintain accumulators of the required
            statistics of the input elements, which is more memory efficient.
            Defaults to `False`.
        
        ### Raises
         - `ImportError`
            If `SpatialIndexMethod.RTREE` or `.KDTREE` are selected for either of
            `distance_index` or `density_index` but the corresponding packages could
            not be imported
        """
        self.n_features = n_features
        
        if not isinstance(hyperbox_fractions, Iterable):
            hyperbox_fractions = n_features*[hyperbox_fractions]
        
        self.hyperbox_fractions: np.ndarray = np.asarray(hyperbox_fractions)
        
        self.feature_ranges = np.asarray(feature_ranges)
        
        self.update_ranges = update_ranges
        self.uncommon_dimensions = uncommon_dimensions
        
        self.update_geometry()
        
        if forgetting_method is None:
            self.forgetting_method = NoForgettingMethod()
        else:
            self.forgetting_method = forgetting_method
        
        self.long_term_memory = long_term_memory
        self.outlier_rejection = outlier_rejection
        self.sparse_rejection = sparse_rejection
        self.multi_density = multi_density
        self.density_interval = density_interval
        
        self.distance_index = distance_index
        self.density_index = density_index
        
        if SpatialIndexMethod.RTREE in (distance_index, density_index):
            if RTreeIndex is None:
                raise ImportError("could not import Index from package rtree")
            
            if density_index == SpatialIndexMethod.RTREE and self.uncommon_dimensions:
                warnings.warn(
                    "ignoring uncommon_dimensions != 0 due to RTree setting",
                    UnsupportedConfiguration
                )
        
        if (
            SpatialIndexMethod.KDTREE in (distance_index, density_index)
            and KDTree is None
        ):
            raise ImportError("could not import KDTree from package scikit-learn")
        
        self.maintain_rtree = SpatialIndexMethod.RTREE in (
            self.distance_index,
            self.density_index
        )
        self.store_elements = store_elements
    
    def update_feature_ranges(self, element: Element):
        self.feature_ranges[:, 0] = np.minimum(self.feature_ranges[:, 0], element)
        self.feature_ranges[:, 1] = np.maximum(self.feature_ranges[:, 1], element)
        
        self.update_geometry()
    
    def update_geometry(self):
        self.hyperbox_lengths: np.ndarray = self.hyperbox_fractions*abs(
            np.diff(self.feature_ranges, axis=1).squeeze()
        )
        self.hyperbox_volume: float = np.product(self.hyperbox_lengths)
        
        self.reachable_radius: float = np.min(self.hyperbox_lengths)/2
        
        # L-inf distances will be compared with the `phi`th smallest length
        # NOTE: Paper Eq. 5 says /2, but that would be inconsistent with the text
        self.connected_radius: float = np.sort(self.hyperbox_lengths)[
            self.uncommon_dimensions
        ]


class DyClee:
    """
    Implementation roughly as per https://doi.org/10.1016/j.patcog.2019.05.024.
    A few apparent errors have been corrected, needing clarification from the authors.
    """
    
    def __init__(self, context: DyCleeContext):
        self.context = context
        
        self.dense_µclusters: Set[MicroCluster] = Set()
        self.semidense_µclusters: Set[MicroCluster] = Set()
        self.outlier_µclusters: Set[MicroCluster] = Set()
        
        self.next_class_label = 0
        self.last_density_time: Timestamp = None
        
        if self.context.maintain_rtree:
            p = RTreeProperty(dimension=self.context.n_features)
            self.rtree = RTreeIndex(properties=p)
            # This mapping is used to retrieve microcluster objects from their hashes
            # stored with their locations in the R*-tree
            self.µcluster_map: Optional[dict[int, MicroCluster]] = {}
        else:
            self.rtree = None
            self.µcluster_map = None
    
    @property
    def active_µclusters(self) -> Set[MicroCluster]:
        return self.dense_µclusters | self.semidense_µclusters
    
    @property
    def all_µclusters(self) -> Set[MicroCluster]:
        return self.active_µclusters | self.outlier_µclusters
    
    def get_next_class_label(self) -> int:
        label = self.next_class_label
        self.next_class_label += 1
        return label
    
    def update_density_partitions(self):
        densities = np.array([µcluster.density for µcluster in self.all_µclusters])
        mean_density = np.mean(densities)
        median_density = np.median(densities)
        
        n_µclusters = len(self.all_µclusters)
        
        # NOTE: boundary between semidense and outliers differs slightly from paper
        self.dense_µclusters, self.semidense_µclusters, self.outlier_µclusters = (
            Set(
                [
                    µcluster
                    for µcluster in self.all_µclusters
                    if mean_density <= µcluster.density >= median_density
                ]
            ),
            Set(
                [
                    µcluster
                    for µcluster in self.all_µclusters
                    if (µcluster.density >= mean_density)
                    != (µcluster.density > median_density)
                ]
            ),
            Set(
                [
                    µcluster
                    for µcluster in self.all_µclusters
                    if mean_density > µcluster.density <= median_density
                ]
            )
        )
        
        assert n_µclusters == len(self.all_µclusters)
    
    def distance_step(self, element: Element, time: Timestamp) -> MicroCluster:
        if self.context.update_ranges:
            self.context.update_feature_ranges(element)
        
        if not self.all_µclusters:
            # Create new microcluster
            µcluster = MicroCluster(element, time, context=self.context)
            self.outlier_µclusters.add(µcluster)
            
            if self.context.maintain_rtree:
                # Add microcluster to R*-tree
                self.µcluster_map[hash(µcluster)] = µcluster
                self.rtree.insert(hash(µcluster), µcluster.bounding_box)
            
            return µcluster
        else:
            closest: MicroCluster = None
            
            for candidate_µclusters in self.active_µclusters, self.outlier_µclusters:
                # First search actives, then others for reachable microclusters
                
                if not candidate_µclusters:
                    continue
                
                if self.context.distance_index == SpatialIndexMethod.RTREE:
                    matches: list[MicroCluster] = [
                        self.µcluster_map[_hash]
                        for _hash in self.rtree.nearest((*element, *element), 1)
                        if self.µcluster_map[_hash].is_reachable(element)
                    ]
                    
                    if matches:
                        closest = matches[
                            np.argmax([µcluster.density for µcluster in matches])
                        ]
                elif self.context.distance_index == SpatialIndexMethod.KDTREE:
                    # Ensure predictable order for indexability
                    candidate_µclusters = list(candidate_µclusters)
                    
                    candidate_centroids: np.ndarray = np.row_stack(
                        [µcluster.centroid for µcluster in candidate_µclusters]
                    ).reshape(len(candidate_µclusters), -1)
                    
                    # Find reachable microclusters (using L-inf norm)
                    idcs, = KDTree(candidate_centroids, p=np.inf).query_radius(
                        np.reshape(element, (1, -1)), self.context.reachable_radius
                    )
                    
                    if not len(idcs):
                        continue
                    
                    min_dist = None
                    
                    # Find closest (L-1 norm) microcluster among the reachable ones
                    for i in idcs:
                        µcluster = candidate_µclusters[i]
                        dist = µcluster.distance(element)
                        
                        # Higher density is tie-breaker in case of equal distances
                        if (
                            closest is None
                            or dist < min_dist
                            or (dist == min_dist and µcluster.density > closest.density)
                        ):
                            closest = µcluster
                            min_dist = dist
                else:
                    # Brute force
                    min_dist = None
                    
                    for µcluster in candidate_µclusters:
                        if not µcluster.is_reachable(element):
                            continue
                        
                        dist = µcluster.distance(element)
                        
                        if (
                            closest is None
                            or dist < min_dist
                            or (dist == min_dist and µcluster.density > closest.density)
                        ):
                            closest = µcluster
                            min_dist = dist
                
                if closest is not None:
                    # Match found, no need to check next set
                    break
            
            if closest is not None:
                if self.context.maintain_rtree:
                    # Remove microcluster from R*-tree
                    self.rtree.delete(hash(closest), closest.bounding_box)
                
                # Add element to closest microcluster
                closest.add(element, time)
                
                if self.context.maintain_rtree:
                    # Add modified microcluster to R*-tree
                    self.rtree.insert(hash(closest), closest.bounding_box)
                
                return closest
            else:
                # Create new microcluster
                µcluster = MicroCluster(element, time, context=self.context)
                self.outlier_µclusters.add(µcluster)
                
                if self.context.maintain_rtree:
                    # Add microcluster to R*-tree
                    self.µcluster_map[hash(µcluster)] = µcluster
                    self.rtree.insert(hash(µcluster), µcluster.bounding_box)
                
                return µcluster
    
    def global_density_step(self) -> tuple[list[Cluster], Set[MicroCluster]]:
        # NOTE: Deviates from the paper's apparently inconsistent Algorithm 2.
        
        self.update_density_partitions()
        
        clusters: list[Cluster] = []
        seen: Set[MicroCluster] = Set()
        
        for µcluster in self.dense_µclusters:
            if µcluster in seen:
                continue
            
            seen.add(µcluster)
            
            if µcluster.label is None:
                µcluster.label = self.get_next_class_label()
            
            cluster = Cluster(µcluster)
            clusters.append(cluster)
            
            # Get dense and semi-dense directly connected neighbours
            connected = µcluster.get_neighbours(
                (self.dense_µclusters | self.semidense_µclusters) - seen,
                index=self.rtree,
                µcluster_map=self.µcluster_map
            )
            
            while connected:
                neighbour = connected.pop()
                
                if neighbour in seen:
                    continue
                
                seen.add(neighbour)
                
                # Outlier microclusters are ignored
                if neighbour in self.outlier_µclusters:
                    continue
                
                # Dense and semi-dense microclusters become part of the cluster
                neighbour.label = µcluster.label
                cluster.add(neighbour)
                
                # Semi-dense neighbours may only form the boundary
                if neighbour not in self.dense_µclusters:
                    continue
                
                # Get neighbour's dense and semi-dense directly connected neighbours
                # and add to set of microclusters connected to the parent
                connected |= neighbour.get_neighbours(
                    (self.dense_µclusters | self.semidense_µclusters) - seen,
                    index=self.rtree,
                    µcluster_map=self.µcluster_map
                )
        
        # Find all microclusters that were not grouped into a cluster
        unclustered = self.all_µclusters
        for cluster in clusters:
            unclustered -= cluster.µclusters
        
        # Remove their labels
        for µcluster in unclustered:
            µcluster.label = None
        
        return clusters, unclustered
    
    def local_density_step(self) -> tuple[list[Cluster], Set[MicroCluster]]:
        raise NotImplementedError("TODO")
    
    def step(
        self, element: Element, time: Timestamp
    ) -> tuple[MicroCluster, Optional[list[Cluster]], Optional[Set[MicroCluster]]]:
        µcluster = self.distance_step(element, time)
        
        if (
            self.last_density_time is None
            or time >= self.last_density_time + self.context.density_interval
        ):
            if self.context.multi_density:
                clusters, unclustered = self.local_density_step()
            else:
                clusters, unclustered = self.global_density_step()
            
            self.last_density_time = time
        else:
            clusters = None
            unclustered = None
        
        return µcluster, clusters, unclustered
    
    def run(
        self,
        elements: Iterable[Element],
        times: Optional[Iterable[Timestamp]] = None,
        progress: bool = True
    ) -> Optional[list[Cluster]]:
        if progress and tqdm is not None:
            elements = tqdm(elements)
        
        if times is None:
            times = range(len(elements))
        
        clusters = None
        
        for element, time in zip(elements, times):
            clusters = self.step(element, time)[1] or clusters
        
        return clusters

from __future__ import annotations
from typing import Sequence, Iterable, Union, Optional
from itertools import count

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
from .types import Element, Timestamp, Set, SpatialIndexMethod


class DyCleeContext:
    def __init__(
        self,
        n_features: int,
        hyperbox_fractions: Union[float, Sequence[float]],
        feature_ranges: Sequence[Sequence[float, float]],
        *,
        update_ranges: bool = False,
        uncommon_dimensions: int = 0,
        forgetting_method: Optional[ForgettingMethod] = None,
        long_term_memory: bool = False,
        outlier_rejection: bool = True,
        sparse_rejection: bool = False,
        multi_density: bool = False,
        partitioning_interval: int = 1,
        density_interval: int = 1,
        distance_index: Optional[SpatialIndexMethod] = SpatialIndexMethod.RTREE,
        density_index: Optional[SpatialIndexMethod] = SpatialIndexMethod.RTREE,
        store_times: bool = False,
        store_elements: bool = False
    ):
        """
        Creates a DyCleeContext object for use with the DyClee class.
        
        ### Parameters
         - `n_features: int`
            Number of features/dimensions of the input data.
         - `hyperbox_fractions: float | Sequence[float]`
            Relative size of each dimension of the microclusters' hyperboxes, as a
            fraction of the total range of each dimension of the input data.
            If a scalar is given, the same fraction is used for all dimensions.
         - `feature_ranges: Sequence[[float, float]]`
            Range of each dimension of the input data in the form:
            `[(xmin, xmax), (ymin, ymax), ...]`
         - `update_ranges: bool`
            This flag controls whether the feature ranges and derived parameters are to
            be automatically updated at each timestep. Defaults to `False`.
         - `uncommon_dimensions: int`
            Number of dimensions to ignore for microcluster connectedness calculations.
            Defaults to `0`.
         - `forgetting_method: Optional[ForgettingMethod]`
            Function that will be applied to microclusters' element accumulators to
            "forget" older samples (as a function of time intervals). `None` implies
            unlimited temporal memory. Defaults to `None`.
         - `long_term_memory: bool`
            Whether to save formerly dense microclusters into a long-term storage to
            speed up recognition when relevant samples reappear. Defaults to `False`.
         - `outlier_rejection: bool`
            TODO. Defaults to `True`.
         - `sparse_rejection: bool`
            TODO. Defaults to `False`.
         - `multi_density: bool`
            TODO. Defaults to `False`.
         - `partitioning_interval: int`
            Controls how many steps pass between the partitioning step which groups
            microclusters into dense, semi-dense and outlier groups and manages the
            long-term memory (if configured) and elimination. Set to `1` to partition
            after every step. Defaults to `1`.
         - `density_interval: int`
            Controls how many steps pass between applications of the density-based
            clustering stage. Increasing this may help with performance. Set to `1` to
            enforce density-based clustering after every step. Defaults to `1`.
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
            efficiency. Defaults to `SpatialIndexMethod.KDTREE`.
         - `store_times: bool`
            Whether to store each input element's timestamp in its corresponding
            microcluster. Defaults to `False`.
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
        
        if not isinstance(hyperbox_fractions, Sequence):
            hyperbox_fractions = n_features*[hyperbox_fractions]
        
        self.hyperbox_fractions: np.ndarray = np.asarray(hyperbox_fractions)
        
        self.feature_ranges: np.ndarray = np.asarray(feature_ranges)
        
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
        self.partitioning_interval = partitioning_interval
        self.density_interval = density_interval
        
        self.distance_index = distance_index
        self.density_index = density_index
        
        if (
            SpatialIndexMethod.RTREE in (distance_index, density_index)
            and RTreeIndex is None
        ):
            raise ImportError("could not import Index from package rtree")
        
        if (
            SpatialIndexMethod.KDTREE in (distance_index, density_index)
            and KDTree is None
        ):
            raise ImportError("could not import KDTree from package scikit-learn")
        
        self.maintain_rtree = SpatialIndexMethod.RTREE in (
            self.distance_index,
            self.density_index
        )
        self.store_times = store_times
        self.store_elements = store_elements
    
    @property
    def elimination_threshold(self):
        return 0.5/self.hyperbox_volume
    
    def update_feature_ranges(self, element: Element):
        self.feature_ranges[:, 0] = np.minimum(self.feature_ranges[:, 0], element)
        self.feature_ranges[:, 1] = np.maximum(self.feature_ranges[:, 1], element)
        
        self.update_geometry()
    
    def update_geometry(self):
        self.hyperbox_lengths: np.ndarray = self.hyperbox_fractions*abs(
            np.diff(self.feature_ranges, axis=1).squeeze()
        )
        self.hyperbox_volume: float = np.product(self.hyperbox_lengths)
        self.potentially_reachable_radius: float = np.max(self.hyperbox_lengths)/2
        self.potential_neighbour_radius: float = np.max(self.hyperbox_lengths)


class DyClee:
    """
    Implementation roughly as per https://doi.org/10.1016/j.patcog.2019.05.024.
    """
    
    def __init__(self, context: DyCleeContext):
        self.context = context
        
        self.dense_µclusters: Set[MicroCluster] = Set()
        self.semidense_µclusters: Set[MicroCluster] = Set()
        self.outlier_µclusters: Set[MicroCluster] = Set()
        self.long_term_memory: Set[MicroCluster] = Set()
        self.eliminated: Set[MicroCluster] = Set()
        
        self.next_µcluster_index: int = 0
        self.next_class_label: int = 0
        self.n_steps: int = 0
        self.last_partitioning_step: int = 0
        self.last_density_step: int = 0
        
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
        return self.active_µclusters | self.outlier_µclusters | self.long_term_memory
    
    def get_next_µcluster_index(self) -> int:
        index = self.next_µcluster_index
        self.next_µcluster_index += 1
        return index
    
    def get_next_class_label(self) -> int:
        label = self.next_class_label
        self.next_class_label += 1
        return label
    
    def update_density_partitions(self, time: Timestamp) -> Set[MicroCluster]:
        densities = np.array(
            [µcluster.density(time) for µcluster in self.all_µclusters]
        )
        mean_density = np.mean(densities)
        median_density = np.median(densities)
        
        dense: Set[MicroCluster] = Set()
        semidense: Set[MicroCluster] = Set()
        outliers: Set[MicroCluster] = Set()
        memory: Set[MicroCluster] = Set()
        eliminated: Set[MicroCluster] = Set()
        
        for µcluster in self.all_µclusters:
            density = µcluster.density(time)
            
            if mean_density <= density >= median_density:
                # Any may become dense
                dense.add(µcluster)
                µcluster.once_dense = True
            elif (
                µcluster in self.dense_µclusters
                or µcluster in self.semidense_µclusters
                or µcluster in self.outlier_µclusters
            ) and (density >= mean_density) != (density >= median_density):
                # Dense and outliers may become dense
                # Semi-dense may stay semi-dense
                semidense.add(µcluster)
            elif (
                (
                    µcluster in self.dense_µclusters
                    or µcluster in self.semidense_µclusters
                )
                and mean_density > density < median_density
            ) or (
                µcluster in self.outlier_µclusters
                and density >= self.context.elimination_threshold
            ):
                # Dense and semi-dense may become outliers
                # Outliers may stay outliers
                outliers.add(µcluster)
            elif (
                self.context.long_term_memory
                and µcluster in self.outlier_µclusters
                and µcluster.once_dense
            ):
                # Outliers may be put into long-term memory
                memory.add(µcluster)
            else:
                # If none of the conditions are met, the microcluster is eliminated
                eliminated.add(µcluster)
                
                if self.context.maintain_rtree:
                    # Remove microcluster from R*-tree
                    self.rtree.delete(hash(µcluster), µcluster.bounding_box)
        
        # Store the final sets, sorting by index for predictable ordering
        self.dense_µclusters = Set(sorted(dense, key=lambda µ: µ.index))
        self.semidense_µclusters = Set(sorted(semidense, key=lambda µ: µ.index))
        self.outlier_µclusters = Set(sorted(outliers, key=lambda µ: µ.index))
        self.long_term_memory = Set(sorted(memory, key=lambda µ: µ.index))
        
        if self.context.store_elements:
            # Keep track of eliminated microclusters (to not lose elements)
            self.eliminated |= eliminated
        
        return eliminated
    
    def distance_step(self, element: Element, time: Timestamp) -> MicroCluster:
        if self.context.update_ranges:
            self.context.update_feature_ranges(element)
        
        if not self.all_µclusters:
            # Create new microcluster
            µcluster = MicroCluster(
                element,
                time,
                context=self.context,
                index=self.get_next_µcluster_index()
            )
            self.outlier_µclusters.add(µcluster)
            
            if self.context.maintain_rtree:
                # Add microcluster to R*-tree
                self.µcluster_map[hash(µcluster)] = µcluster
                self.rtree.insert(hash(µcluster), µcluster.bounding_box)
            
            return µcluster
        else:
            closest: Optional[MicroCluster] = None
            
            if self.context.distance_index == SpatialIndexMethod.RTREE:
                # The R*-tree searches all microclusters regardless of precedence, so we
                # need to filter by priority after the index search
                
                # Find all reachable microclusters
                matches: Set[MicroCluster] = Set(
                    [
                        self.µcluster_map[hash_]
                        for hash_ in self.rtree.intersection((*element, *element))
                    ]
                )
                
                min_dist = None
                
                for candidate_µclusters in (
                    self.active_µclusters,
                    self.outlier_µclusters,
                    self.long_term_memory
                ):
                    # First match active microclusters, then others
                    
                    for µcluster in matches & candidate_µclusters:
                        dist = µcluster.distance(element)
                        
                        if (
                            closest is None
                            or dist < min_dist
                            or (
                                dist == min_dist
                                and µcluster.density(time) > closest.density(time)
                            )
                        ):
                            closest = µcluster
                            min_dist = dist
            else:
                for candidate_µclusters in (
                    self.active_µclusters,
                    self.outlier_µclusters,
                    self.long_term_memory
                ):
                    # First search actives, then others for reachable microclusters
                    
                    if not candidate_µclusters:
                        continue
                    
                    if self.context.distance_index == SpatialIndexMethod.KDTREE:
                        # Ensure predictable order for indexability
                        candidate_µclusters = list(candidate_µclusters)
                        
                        candidate_centroids: np.ndarray = np.row_stack(
                            [µcluster.centroid for µcluster in candidate_µclusters]
                        )
                        
                        # Find potentially reachable microclusters (using L-inf norm)
                        idcs, = KDTree(candidate_centroids, p=np.inf).query_radius(
                            np.reshape(element, (1, -1)),
                            self.context.potentially_reachable_radius
                        )
                        
                        if not len(idcs):
                            continue
                        
                        min_dist = None
                        
                        # Find closest (L-1 norm) microcluster among the reachable ones
                        for i in idcs:
                            µcluster = candidate_µclusters[i]
                            
                            if not µcluster.is_reachable(element):
                                continue
                            
                            dist = µcluster.distance(element)
                            
                            # Higher density is tie-breaker in case of equal distances
                            if (
                                closest is None
                                or dist < min_dist
                                or (
                                    dist == min_dist
                                    and µcluster.density(time) > closest.density(time)
                                )
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
                                or (
                                    dist == min_dist
                                    and µcluster.density(time) > closest.density(time)
                                )
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
                µcluster = MicroCluster(
                    element,
                    time,
                    context=self.context,
                    index=self.get_next_µcluster_index()
                )
                self.outlier_µclusters.add(µcluster)
                
                if self.context.maintain_rtree:
                    # Add microcluster to R*-tree
                    self.µcluster_map[hash(µcluster)] = µcluster
                    self.rtree.insert(hash(µcluster), µcluster.bounding_box)
                
                return µcluster
    
    def global_density_step(
        self, time: Timestamp
    ) -> tuple[list[Cluster], Set[MicroCluster]]:
        clusters: list[Cluster] = []
        seen: Set[MicroCluster] = Set()
        
        for µcluster in self.dense_µclusters:
            if µcluster in seen:
                continue
            
            seen.add(µcluster)
            
            if µcluster.label is None:
                µcluster.label = self.get_next_class_label()
            
            cluster = Cluster(µcluster, time)
            clusters.append(cluster)
            
            # Get dense and semi-dense directly connected neighbours
            connected = µcluster.get_neighbours(
                (self.dense_µclusters | self.semidense_µclusters) - seen,
                rtree_index=self.rtree,
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
                cluster.add(neighbour, time)
                
                # Semi-dense neighbours may only form the boundary
                if neighbour not in self.dense_µclusters:
                    continue
                
                # Get neighbour's dense and semi-dense directly connected neighbours
                # and add to set of microclusters connected to the parent
                connected |= neighbour.get_neighbours(
                    (self.dense_µclusters | self.semidense_µclusters) - seen,
                    rtree_index=self.rtree,
                    µcluster_map=self.µcluster_map
                )
        
        # Find all microclusters that were not grouped into a cluster
        unclustered = self.all_µclusters
        for cluster in clusters:
            unclustered -= cluster.µclusters
        
        return clusters, unclustered
    
    def local_density_step(
        self, time: Timestamp
    ) -> tuple[list[Cluster], Set[MicroCluster]]:
        raise NotImplementedError("TODO")
    
    def density_step(self, time: Timestamp) -> tuple[list[Cluster], Set[MicroCluster]]:
        if self.context.multi_density:
            return self.local_density_step(time)
        else:
            return self.global_density_step(time)
    
    def step(
        self, element: Element, time: Timestamp, skip_density_step: bool = False
    ) -> tuple[
        MicroCluster,
        Optional[list[Cluster]],
        Optional[Set[MicroCluster]],
        Optional[Set[MicroCluster]]
    ]:
        self.n_steps += 1
        
        µcluster = self.distance_step(element, time)
        
        if (
            self.n_steps
            >= self.last_partitioning_step + self.context.partitioning_interval
        ):
            eliminated = self.update_density_partitions(time)
            
            self.last_partitioning_step = self.n_steps
        else:
            eliminated = None
        
        if (
            not skip_density_step
            and self.n_steps >= self.last_density_step + self.context.density_interval
        ):
            clusters, unclustered = self.density_step(time)
            
            self.last_density_step = self.n_steps
        else:
            clusters = None
            unclustered = None
        
        return µcluster, clusters, unclustered, eliminated
    
    def run(
        self,
        elements: Iterable[Element],
        times: Optional[Iterable[Timestamp]] = None,
        progress: bool = True
    ) -> list[Cluster]:
        if progress and tqdm is not None:
            elements = tqdm(elements)
        
        if times is None:
            times = count()
        
        for element, time in zip(elements, times):
            self.step(element, time, skip_density_step=True)
        
        clusters, _ = self.density_step(time)
        
        return clusters

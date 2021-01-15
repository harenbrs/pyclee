from __future__ import annotations
from abc import ABC, abstractmethod
from collections import defaultdict
from itertools import cycle
from typing import TYPE_CHECKING, Optional, Iterable, Union

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.animation import FuncAnimation
from matplotlib.collections import PathCollection, PatchCollection
from matplotlib.patches import Rectangle
import seaborn as sns
from tqdm.auto import tqdm

from .types import Set

if TYPE_CHECKING:
    from .dyclee import DyClee
    from .clusters import MicroCluster, Cluster
    from .types import Element, Timestamp


__all__ = ['ElementPlotter', 'CentroidPlotter', 'BoundaryPlotter', 'MultiPlotter']


sns.set()


class BasePlotter(ABC):
    cmap = cm.tab10
    outlier_c = (0.7, 0.7, 0.7, 0.8)
    title = None
    
    def __init__(self, dyclee: DyClee, ax: Optional[plt.Axes] = None):
        assert dyclee.context.n_features == 2, "plotting only supported for 2D data"
        
        self.dyclee = dyclee
        
        if ax is None:
            fig, ax = plt.subplots()
        
        self.fig = ax.figure
        self.ax = ax
        
        self.ax.set_xlim(*dyclee.context.feature_ranges[0])
        self.ax.set_ylim(*dyclee.context.feature_ranges[1])
        
        if self.title is not None:
            title = self.ax.get_title()
            if title:
                title += f" + {self.title}"
            else:
                title = self.title
            
            self.ax.set_title(title)
    
    @abstractmethod
    def update(
        self,
        element: Element,
        µcluster: MicroCluster,
        clusters: Optional[list[Cluster]],
        unclustered: Optional[Set[MicroCluster]]
    ):
        ...
    
    def animate(self, elements: Iterable[Element], times: Iterable[Timestamp] = None):
        if times is None:
            times = range(len(elements))
        
        def animate(frame):
            element, time = frame
            return self.update(element, *self.dyclee.step(element, time))
        
        anim = FuncAnimation(
            self.fig,
            animate,
            zip(tqdm(elements), times),
            save_count=len(elements),
            interval=40
        )
        
        return anim
    
    @abstractmethod
    def plot_snapshot(self, clusters: list[Cluster]):
        ...


class ElementPlotter(BasePlotter):
    title = "Samples"
    
    def __init__(self, dyclee: DyClee, ax: Optional[plt.Axes] = None):
        super().__init__(dyclee, ax)
        
        self.path_map: defaultdict[MicroCluster, list[PathCollection]] = defaultdict(
            list
        )
    
    def update(
        self,
        element: Element,
        µcluster: MicroCluster,
        clusters: Optional[list[Cluster]],
        unclustered: Optional[Set[MicroCluster]]
    ):
        path_collection = self.ax.scatter(*element, marker='.')
        
        self.path_map[µcluster].append(path_collection)
        
        if clusters is not None:
            for cluster, c in zip(clusters, cycle(self.cmap.colors)):
                for µcluster in cluster.µclusters:
                    for path_collection in self.path_map[µcluster]:
                        path_collection.set_color(c)
        
        if unclustered is not None:
            for µcluster in unclustered:
                for path_collection in self.path_map[µcluster]:
                    path_collection.set_color(self.outlier_c)
    
    def plot_snapshot(self, clusters: list[Cluster]):
        if not self.dyclee.context.store_elements:
            raise ValueError(
                "element snapshot plotting requires DyCleeContext.store_elements"
            )
        
        unclustered = self.dyclee.all_µclusters
        
        for cluster, c in zip(clusters, cycle(self.cmap.colors)):
            cluster_elements: Set[Element] = Set()
            
            for µcluster in cluster.µclusters:
                cluster_elements |= µcluster.elements
            
            unclustered -= cluster.µclusters
            
            self.ax.scatter(*zip(*cluster_elements), color=c, marker='.')
        
        unclustered_elements: Set[Element] = Set()
        
        for µcluster in unclustered:
            unclustered_elements |= µcluster.elements
        
        self.ax.scatter(*zip(*unclustered_elements), color=self.outlier_c, marker='.')


class CentroidPlotter(BasePlotter):
    title = "µcluster centroids"
    
    def __init__(self, dyclee: DyClee, ax: Optional[plt.Axes] = None):
        super().__init__(dyclee, ax)
        
        self.path_map: dict[MicroCluster, PathCollection] = {}
    
    def update(
        self,
        element: Element,
        µcluster: MicroCluster,
        clusters: Optional[list[Cluster]],
        unclustered: Optional[Set[MicroCluster]]
    ):
        if µcluster in self.path_map:
            self.path_map[µcluster].remove()
        
        path_collection = self.ax.scatter(*µcluster.centroid, marker='o')
        
        self.path_map[µcluster] = path_collection
        
        if clusters is not None:
            for cluster, c in zip(clusters, cycle(self.cmap.colors)):
                for µcluster in cluster.µclusters:
                    self.path_map[µcluster].set_color(c)
        
        if unclustered is not None:
            for µcluster in unclustered:
                self.path_map[µcluster].set_color(self.outlier_c)
    
    def plot_snapshot(self, clusters: list[Cluster]):
        unclustered = self.dyclee.all_µclusters
        
        for cluster, c in zip(clusters, cycle(self.cmap.colors)):
            unclustered -= cluster.µclusters
            
            self.ax.scatter(
                *zip(*[µcluster.centroid for µcluster in cluster.µclusters]),
                color=c,
                marker='o'
            )
        
        self.ax.scatter(
            *zip(*[µcluster.centroid for µcluster in unclustered]),
            color=self.outlier_c,
            marker='o'
        )


class BoundaryPlotter(BasePlotter):
    title = "µcluster boundaries"
    
    def __init__(self, dyclee: DyClee, ax: Optional[plt.Axes] = None):
        super().__init__(dyclee, ax)
        
        self.patch_map: dict[MicroCluster, PatchCollection] = {}
    
    def update(
        self,
        element: Element,
        µcluster: MicroCluster,
        clusters: Optional[list[Cluster]],
        unclustered: Optional[Set[MicroCluster]]
    ):
        max_density = max([µcluster.density for µcluster in self.dyclee.all_µclusters])
        
        if µcluster in self.patch_map:
            self.patch_map[µcluster].remove()
        
        patch_collection = PatchCollection(
            [
                Rectangle(
                    µcluster.centroid - µcluster.context.hyperbox_lengths/2,
                    *µcluster.context.hyperbox_lengths
                )
            ]
        )
        
        patch_collection.set_facecolor((0, 0, 0, 0.5*µcluster.density/max_density))
        self.patch_map[µcluster] = patch_collection
        
        self.ax.add_collection(patch_collection)
        
        if clusters is not None:
            for cluster, c in zip(clusters, cycle(self.cmap.colors)):
                for µcluster in cluster.µclusters:
                    self.patch_map[µcluster].set_edgecolor(c)
                    self.patch_map[µcluster].set_facecolor(
                        (0, 0, 0, 0.5*µcluster.density/max_density)
                    )
        
        if unclustered is not None:
            for µcluster in unclustered:
                self.patch_map[µcluster].set_edgecolor(self.outlier_c)
                self.patch_map[µcluster].set_facecolor(
                    (0, 0, 0, 0.5*µcluster.density/max_density)
                )
    
    def plot_snapshot(self, clusters: list[Cluster]):
        max_density = max([µcluster.density for µcluster in self.dyclee.all_µclusters])
        
        unclustered = self.dyclee.all_µclusters
        
        for cluster, c in zip(clusters, cycle(self.cmap.colors)):
            unclustered -= cluster.µclusters
            
            for µcluster in cluster.µclusters:
                patch_collection = PatchCollection(
                    [
                        Rectangle(
                            µcluster.centroid - µcluster.context.hyperbox_lengths/2,
                            *µcluster.context.hyperbox_lengths
                        )
                    ]
                )
                patch_collection.set_edgecolor(c)
                patch_collection.set_facecolor(
                    (0, 0, 0, 0.5*µcluster.density/max_density)
                )
                self.ax.add_collection(patch_collection)
        
        for µcluster in unclustered:
            patch_collection = PatchCollection(
                [
                    Rectangle(
                        µcluster.centroid - µcluster.context.hyperbox_lengths/2,
                        *µcluster.context.hyperbox_lengths
                    )
                ]
            )
            patch_collection.set_edgecolor(self.outlier_c)
            patch_collection.set_facecolor((0, 0, 0, 0.5*µcluster.density/max_density))
            self.ax.add_collection(patch_collection)


class MultiPlotter(BasePlotter):
    def __init__(
        self,
        dyclee: DyClee,
        axes: Optional[Union[Iterable[plt.Axes], plt.Axes]] = None,
        elements: bool = True,
        centroids: bool = True,
        boundaries: bool = True
    ):
        self.dyclee = dyclee
        
        n_plots = elements + centroids + boundaries
        
        if axes is None:
            fig, axes = plt.subplots(
                1, n_plots, sharex=True, sharey=True, figsize=(4*n_plots + 1, 4)
            )
        elif isinstance(axes, plt.Axes):
            axes = n_plots*[axes]
        
        self.fig = axes[0].figure
        self.axes = axes
        
        self.plotters = []
        
        if elements:
            self.plotters.append(ElementPlotter(dyclee, self.axes[len(self.plotters)]))
        
        if centroids:
            self.plotters.append(CentroidPlotter(dyclee, self.axes[len(self.plotters)]))
        
        if boundaries:
            self.plotters.append(BoundaryPlotter(dyclee, self.axes[len(self.plotters)]))
    
    def update(
        self,
        element: Element,
        µcluster: MicroCluster,
        clusters: Optional[list[Cluster]],
        unclustered: Optional[Set[MicroCluster]]
    ):
        for plotter in self.plotters:
            plotter.update(element, µcluster, clusters, unclustered)
    
    def plot_snapshot(self, clusters: list[Cluster]):
        for plotter in self.plotters:
            plotter.plot_snapshot(clusters)

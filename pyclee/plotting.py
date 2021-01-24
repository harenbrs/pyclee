from __future__ import annotations
from abc import ABC, abstractmethod
from collections import defaultdict
from itertools import cycle, count
from typing import TYPE_CHECKING, Optional, Iterable, Sequence, Union, Any

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


class ColourManager:
    def __init__(
        self, cmap=cm.tab10, outlier_colour: tuple[float, ...] = (0.7, 0.7, 0.7, 0.8)
    ):
        self.colours = cycle(cmap.colors)
        self.colour_map: dict[Any, tuple[float, ...]] = {None: outlier_colour}
    
    def get_colour(self, label: Any, opacity: Optional[float] = None):
        if label not in self.colour_map:
            self.colour_map[label] = next(self.colours)
        
        if opacity is not None:
            return (*self.colour_map[label][:3], opacity)
        
        return self.colour_map[label]


class BasePlotter(ABC):
    title: Optional[str] = None
    unclustered_opacity: float = 0.2
    
    def __init__(
        self,
        dyclee: DyClee,
        ax: Optional[plt.Axes] = None,
        colour_manager: Optional[ColourManager] = None,
        legend_loc='best'
    ):
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
                title += f" + {self.title.lower()}"
            else:
                title = self.title
            
            self.ax.set_title(title)
        
        if colour_manager is None:
            colour_manager = ColourManager()
        
        self.colour_manager = colour_manager
        
        self.legend_loc = legend_loc
    
    @abstractmethod
    def update(
        self,
        element: Element,
        time: Timestamp,
        µcluster: MicroCluster,
        clusters: Optional[list[Cluster]],
        unclustered: Optional[Set[MicroCluster]],
        eliminated: Optional[Set[MicroCluster]]
    ):
        """
        Makes changes to the current plot based on the inputs (`element`, `time`) and
        outputs (`µcluster`, `clusters`, `unclustered`, `eliminated`) of a given
        `DyClee.step(...)` execution.
        """
        ...
    
    def animate(
        self, elements: Iterable[Element], times: Optional[Iterable[Timestamp]] = None
    ):
        if times is None:
            times = count()
        
        def animate(frame):
            element, time = frame
            return self.update(element, time, *self.dyclee.step(element, time))
        
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
    
    def __init__(
        self,
        dyclee: DyClee,
        ax: Optional[plt.Axes] = None,
        colour_manager: Optional[ColourManager] = None,
        legend_loc='best',
        keep_eliminated: bool = True
    ):
        super().__init__(dyclee, ax, colour_manager, legend_loc)
        
        self.path_map: defaultdict[MicroCluster, list[PathCollection]] = defaultdict(
            list
        )
        
        self.keep_eliminated = keep_eliminated
    
    def update(
        self,
        element: Element,
        time: Timestamp,
        µcluster: MicroCluster,
        clusters: Optional[list[Cluster]],
        unclustered: Optional[Set[MicroCluster]],
        eliminated: Optional[Set[MicroCluster]]
    ):
        path_collection = self.ax.scatter(*element, marker='.', color='w')
        
        self.path_map[µcluster].append(path_collection)
        
        if clusters is not None:
            for cluster in clusters:
                for µcluster in cluster.µclusters:
                    for path_collection in self.path_map[µcluster]:
                        path_collection.set_color(
                            self.colour_manager.get_colour(cluster.label)
                        )
            
            order = sorted(range(len(clusters)), key=lambda i: clusters[i].label)
            self.ax.legend(
                [self.path_map[clusters[i].µclusters[0]][0] for i in order],
                [clusters[i].label for i in order],
                loc=self.legend_loc
            )
        
        if unclustered is not None:
            for µcluster in unclustered:
                for path_collection in self.path_map[µcluster]:
                    path_collection.set_color(
                        self.colour_manager.get_colour(
                            µcluster.label, self.unclustered_opacity
                        )
                    )
        
        if eliminated is not None and not self.keep_eliminated:
            for µcluster in eliminated:
                for path_collection in self.path_map[µcluster]:
                    path_collection.remove()
                del self.path_map[µcluster]
    
    def plot_snapshot(self, clusters: list[Cluster]):
        if not self.dyclee.context.store_elements:
            raise ValueError(
                "element snapshot plotting requires DyCleeContext.store_elements"
            )
        
        unclustered = self.dyclee.all_µclusters | self.dyclee.eliminated
        
        paths: dict[int, PathCollection] = {}
        
        for cluster in clusters:
            cluster_elements: Set[Element] = Set()
            
            for µcluster in cluster.µclusters:
                cluster_elements |= µcluster.elements
            
            unclustered -= cluster.µclusters
            
            paths[cluster.label] = self.ax.scatter(
                *zip(*cluster_elements),
                color=self.colour_manager.get_colour(cluster.label),
                marker='.'
            )
        
        for µcluster in unclustered:
            self.ax.scatter(
                *zip(*µcluster.elements),
                color=self.colour_manager.get_colour(
                    µcluster.label, self.unclustered_opacity
                ),
                marker='.'
            )
        
        labels = sorted(paths)
        self.ax.legend([paths[label] for label in labels], labels, loc=self.legend_loc)


class BoundaryPlotter(BasePlotter):
    title = "µcluster boundaries"
    
    def __init__(
        self,
        dyclee: DyClee,
        ax: Optional[plt.Axes] = None,
        colour_manager: Optional[ColourManager] = None,
        legend_loc='best'
    ):
        super().__init__(dyclee, ax, colour_manager, legend_loc)
        
        self.patch_map: dict[MicroCluster, PatchCollection] = {}
    
    def update(
        self,
        element: Element,
        time: Timestamp,
        µcluster: MicroCluster,
        clusters: Optional[list[Cluster]],
        unclustered: Optional[Set[MicroCluster]],
        eliminated: Optional[Set[MicroCluster]]
    ):
        max_density = max(
            [µcluster.density(time) for µcluster in self.dyclee.all_µclusters]
        )
        
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
        
        patch_collection.set_edgecolor('w')
        patch_collection.set_facecolor(
            (0, 0, 0, 0.5*µcluster.density(time)/max_density)
        )
        self.patch_map[µcluster] = patch_collection
        
        self.ax.add_collection(patch_collection)
        
        if clusters is not None:
            for cluster in clusters:
                for µcluster in cluster.µclusters:
                    self.patch_map[µcluster].set_edgecolor(
                        self.colour_manager.get_colour(cluster.label)
                    )
                    self.patch_map[µcluster].set_facecolor(
                        (0, 0, 0, 0.5*µcluster.density(time)/max_density)
                    )
            
            order = sorted(range(len(clusters)), key=lambda i: clusters[i].label)
            self.ax.legend(
                [
                    Rectangle(
                        (0, 0),
                        1,
                        1,
                        facecolor='none',
                        edgecolor=self.colour_manager.get_colour(clusters[i].label)
                    )
                    for i in order
                ],
                [clusters[i].label for i in order],
                loc=self.legend_loc
            )
        
        if unclustered is not None:
            for µcluster in unclustered:
                self.patch_map[µcluster].set_edgecolor(
                    self.colour_manager.get_colour(
                        µcluster.label, self.unclustered_opacity
                    )
                )
                self.patch_map[µcluster].set_facecolor(
                    (0, 0, 0, 0.5*µcluster.density(time)/max_density)
                )
        
        if eliminated is not None:
            for µcluster in eliminated:
                self.patch_map[µcluster].remove()
                del self.patch_map[µcluster]
    
    def plot_snapshot(self, clusters: list[Cluster]):
        time = max([µcluster.last_time for µcluster in self.dyclee.all_µclusters])
        
        max_density = max(
            [µcluster.density(time) for µcluster in self.dyclee.all_µclusters]
        )
        
        unclustered = self.dyclee.all_µclusters
        
        for cluster in clusters:
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
                patch_collection.set_edgecolor(
                    self.colour_manager.get_colour(cluster.label)
                )
                patch_collection.set_facecolor(
                    (0, 0, 0, 0.5*µcluster.density(time)/max_density)
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
            patch_collection.set_edgecolor(
                self.colour_manager.get_colour(µcluster.label, self.unclustered_opacity)
            )
            patch_collection.set_facecolor(
                (0, 0, 0, 0.5*µcluster.density(time)/max_density)
            )
            self.ax.add_collection(patch_collection)
        
        order = sorted(range(len(clusters)), key=lambda i: clusters[i].label)
        self.ax.legend(
            [
                Rectangle(
                    (0, 0),
                    1,
                    1,
                    facecolor='none',
                    edgecolor=self.colour_manager.get_colour(clusters[i].label)
                )
                for i in order
            ],
            [clusters[i].label for i in order],
            loc=self.legend_loc
        )


class CentroidPlotter(BasePlotter):
    title = "Cluster centroids"
    
    def __init__(
        self,
        dyclee: DyClee,
        ax: Optional[plt.Axes] = None,
        colour_manager: Optional[ColourManager] = None,
        legend_loc='best'
    ):
        super().__init__(dyclee, ax, colour_manager, legend_loc)
        
        self.path_map: dict[Cluster, PathCollection] = {}
    
    def update(
        self,
        element: Element,
        time: Timestamp,
        µcluster: MicroCluster,
        clusters: Optional[list[Cluster]],
        unclustered: Optional[Set[MicroCluster]],
        eliminated: Optional[Set[MicroCluster]]
    ):
        
        if clusters is not None:
            for path in self.path_map.values():
                path.remove()
            
            self.path_map = {}
            
            for cluster in clusters:
                path_collection = self.ax.scatter(
                    *cluster.centroid(time),
                    marker='o',
                    s=80,
                    color=self.colour_manager.get_colour(cluster.label),
                    edgecolors='w'
                )
                
                self.path_map[cluster] = path_collection
            
            order = sorted(range(len(clusters)), key=lambda i: clusters[i].label)
            self.ax.legend(
                [self.path_map[clusters[i]] for i in order],
                [clusters[i].label for i in order],
                loc=self.legend_loc
            )
    
    def plot_snapshot(self, clusters: list[Cluster]):
        time = max(max(µ.last_time for µ in cluster.µclusters) for cluster in clusters)
        
        paths: dict[int, PathCollection] = {}
        
        for cluster in clusters:
            paths[cluster.label] = self.ax.scatter(
                *cluster.centroid(time),
                marker='o',
                s=80,
                color=self.colour_manager.get_colour(cluster.label),
                edgecolors='w'
            )
        
        labels = sorted(paths)
        self.ax.legend([paths[label] for label in labels], labels, loc=self.legend_loc)


class MultiPlotter(BasePlotter):
    def __init__(
        self,
        dyclee: DyClee,
        axes: Optional[Union[Sequence[plt.Axes], plt.Axes]] = None,
        colour_manager: Optional[ColourManager] = None,
        legend_loc='best',
        elements: bool = True,
        boundaries: bool = True,
        centroids: bool = True
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
        
        if colour_manager is None:
            colour_manager = ColourManager()
        
        self.colour_manager = colour_manager
        
        self.plotters = []
        
        if elements:
            self.plotters.append(
                ElementPlotter(
                    dyclee, self.axes[len(self.plotters)], colour_manager, legend_loc
                )
            )
        
        if boundaries:
            self.plotters.append(
                BoundaryPlotter(
                    dyclee, self.axes[len(self.plotters)], colour_manager, legend_loc
                )
            )
        
        if centroids:
            self.plotters.append(
                CentroidPlotter(
                    dyclee, self.axes[len(self.plotters)], colour_manager, legend_loc
                )
            )
    
    def update(
        self,
        element: Element,
        time: Timestamp,
        µcluster: MicroCluster,
        clusters: Optional[list[Cluster]],
        unclustered: Optional[Set[MicroCluster]],
        eliminated: Optional[Set[MicroCluster]]
    ):
        for plotter in self.plotters:
            plotter.update(element, time, µcluster, clusters, unclustered, eliminated)
    
    def plot_snapshot(self, clusters: list[Cluster]):
        for plotter in self.plotters:
            plotter.plot_snapshot(clusters)

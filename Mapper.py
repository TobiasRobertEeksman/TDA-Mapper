from dataclasses import dataclass, field
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from sklearn.cluster import DBSCAN
from gudhi.cover_complex import MapperComplex

from ShapeClass import ShapeSample

@dataclass
class MapperParams:
    # 1D cover on the filter values f \in R
    resolutions: int = 10           # number of cover intervals
    gains: float = 0.5            # fractional overlap in (0,1)

    # clustering in each pullback set
    clusterer: str = "dbscan"
    # dbscan params
    eps: float = 0.5
    min_samples: int = 5

@dataclass
class MapperSample:
    item: ShapeSample
    params: MapperParams = field(default_factory=MapperParams)
    visualize: bool = True

    # outputs
    mapper_graph: Optional[nx.Graph] = field(default=None, repr=False)
    node_data: dict[int, dict] = field(default_factory=dict, repr=False)


    def run(self) -> nx.Graph:
        """
        Build a Mapper graph from ShapeSample.sampled_points & sampled_f.
        """
        X = self.item.sampled_points
        f = self.item.sampled_f

        if X is None or f is None:
            raise RuntimeError("ShapeSample has no samples. Call item.sample(...) first.")
        
        return self._run_gudhi_mapper(X, f)

    # ------------------- backend -------------------

    def _run_gudhi_mapper(self, X: np.ndarray, f: np.ndarray) -> nx.Graph:

        f = np.asarray(f, float).ravel()
        if f.shape[0] != X.shape[0]:
            raise ValueError("filter length mismatch with X")

        # clusterer
        clusterer = DBSCAN(eps=self.params.eps, min_samples=self.params.min_samples)

        mapper = MapperComplex(
            input_type="point cloud",
            resolutions=[int(self.params.resolutions)],
            gains=[float(self.params.gains)],
            clustering=clusterer,                       # <<< add the filter as a color
        )
        mapper.fit(X, f)                         # list with one filter

        # ask GUDHI to copy color_* onto nodes
        G = mapper.get_networkx(set_attributes_from_colors=True)
        if self.visualize:
            colors = [G.nodes[n]['attr_name'] for n in G.nodes]
            nx.draw(G, node_color=colors)
            plt.show()
        self.mapper_graph = G
        return G
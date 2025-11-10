from dataclasses import dataclass, field
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from gudhi.cover_complex import MapperComplex
from contextlib import nullcontext
import gudhi
from .ShapeClass import ShapeSample

from pathlib import Path
import re


@dataclass
class MapperParams:
    # 1D cover on the filter values f \in R
    resolutions: int = 10           # number of cover intervals
    gains: float = 0.5              # fractional overlap in (0,1)

    # clustering in each pullback set
    clusterer: str = "dbscan" # or "hierarchical"
    # dbscan params
    eps: float = 0.5
    min_samples: int = 5

    # hierarchical params
    n_clusters: float = 4
    metric: str = "euclidean" # or l1, l2, manhatten, cosine not implemented?"

@dataclass
class MapperSample:
    item: ShapeSample
    params: MapperParams
    visualize: bool = True
    save:  bool = False

    # outputs
    simplex_tree: Optional[gudhi.SimplexTree] = field(default=None, init=False)
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

        #helper for safe filenames
        def _slug(s: str) -> str:
            # safe-ish folder/file name
            s = str(s)
            s = re.sub(r"\s+", "_", s.strip())
            return re.sub(r"[^-_.A-Za-z0-9]", "-", s)

        def _fmt_float(x: float) -> str:
            # compact + filesystem friendly (replace '.' with 'p')
            return f"{x:.4g}".replace(".", "p")

        f = np.asarray(f, float).ravel()
        if f.shape[0] != X.shape[0]:
            raise ValueError("filter length mismatch with X")

        # clusterer
        if self.params.clusterer == "dbscan": 
            clusterer = DBSCAN(eps=self.params.eps, min_samples=self.params.min_samples)
        elif self.params.clusterer == "hierarchical":
            clusterer = AgglomerativeClustering(n_clusters=self.params.n_clusters, metric = self.params.metric)
        else:
            raise ValueError("clusterer needs to be dbscan or hierarchical")

        mapper = MapperComplex(
            input_type="point cloud",
            resolutions=[int(self.params.resolutions)], 
            gains=[float(self.params.gains)],
            clustering=clusterer,
        )

        f2d = f[:, None]
        mapper.fit(X, filters=f2d, colors=f2d)

        self.simplex_tree = mapper.simplex_tree_
        G = mapper.get_networkx(set_attributes_from_colors=True)
        self.mapper_graph = G
        self.node_data = {
            int(n): {"mean_f": float(d["attr_name"][0])} for n, d in G.nodes(data=True)
            }

        # decide if we need a figure at all
        need_fig = bool(getattr(self, "save", False) or self.visualize)
        fig = ax = None

        if need_fig:
            # turn off interactive drawing if we're just saving
            ctx = plt.ioff() if not self.visualize else nullcontext()
            try:
                with ctx:
                    colors = [G.nodes[n]["attr_name"] for n in G.nodes]
                    fig, ax = plt.subplots()
                    nx.draw(G, node_color=colors, ax=ax)

                    fig.suptitle("Mapper experiment", fontsize=12, fontweight="bold")
                    if clusterer == "dbscan":
                        ax.set_title(
                            f"Res: {int(self.params.resolutions)} | "
                            f"Gains: {float(self.params.gains)} | "
                            f"DBSCAN: eps={self.params.eps}, min_samples={self.params.min_samples} | "
                            f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}",
                            fontsize=9, loc="left",
                        )
                    elif clusterer == "hierarchical":
                        ax.set_title(
                            f"n: {int(self.params.n_clusters)} | "
                            f"metric: {str(self.params.metric)}",
                            fontsize=9,  loc="left",
                        )
                    plt.tight_layout()

                    # --- saving ---
                    if getattr(self, "save", False):
                        base = Path("mapper_results") / _slug(self.item.name)
                        base.mkdir(parents=True, exist_ok=True)
                        stem = (
                            f"res{int(self.params.resolutions)}"
                            f"_gain{_fmt_float(float(self.params.gains))}"
                            f"_eps{_fmt_float(float(self.params.eps))}"
                            f"_min{int(self.params.min_samples)}"
                        )
                        (base / f"{stem}.png").parent.mkdir(parents=True, exist_ok=True)
                        fig.savefig(base / f"{stem}.png", dpi=200, bbox_inches="tight")
                        # optional: also persist the graph for later analysis
                        # nx.write_gml(G, base / f"{stem}.gml")

                    if self.visualize:
                        plt.show()
            finally:
                # ALWAYS close to avoid accumulating figures
                if fig is not None:
                    plt.close(fig)

        return G

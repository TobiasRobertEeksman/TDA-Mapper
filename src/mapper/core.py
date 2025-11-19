from dataclasses import dataclass, field
from typing import Optional, Callable, Any, Dict

# #bug fix for automato + matplotlib interaction
import matplotlib
matplotlib.use("Agg")  # non-interactive backend, no Tk
import matplotlib.pyplot as plt

import numpy as np
import networkx as nx
from gudhi.cover_complex import MapperComplex
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from contextlib import nullcontext
import gudhi
from src.shapes.base import ShapeSample

from pathlib import Path
from src.helper import _slug, _fmt_float


@dataclass
class MapperParams:
    # 1D cover on the filter values f \in R
    resolutions: int = 10           # number of cover intervals
    gains: float = 0.5              # fractional overlap in (0,1)

    # clustering in each pullback set
    clusterer_name: str = None
    clusterer_function: Callable[..., Any] = DBSCAN  # e.g. DBSCAN, AgglomerativeClustering
    clusterer_params: Dict[str, Any] = field(
        default_factory=lambda: {"eps": 0.4, "min_samples": 5}
    )

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

        f = np.asarray(f, float).ravel()
        if f.shape[0] != X.shape[0]:
            raise ValueError("filter length mismatch with X")

        # clusterer
        if self.params.clusterer_function is None:
            raise ValueError("clusterer_function must be given")
        clusterer =  self.params.clusterer_function(**self.params.clusterer_params)

        #set name for saving purposes
        if self.params.clusterer_name is None:
            self.params.clusterer_name = getattr(self.params.clusterer_function, "__name__", str(self.params.clusterer_function))

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
                    param_str = ", ".join(f"{k}={v}" for k, v in self.params.clusterer_params.items())
                    ax.set_title(
                        f"Res: {int(self.params.resolutions)} | "
                        f"Gains: {float(self.params.gains)} | "
                        f"{self.params.clusterer_name}: {param_str} | "
                        f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}",
                        fontsize=9, loc="left",
                    )
                    plt.tight_layout()

                    # --- saving ---
                    if getattr(self, "save", False):
                        base = Path("mapper_results") /_slug(self.item.name) / _slug(self.params.clusterer_name) 
                        base.mkdir(parents=True, exist_ok=True)
                        param_slug = "_".join(
                            f"{k}{_fmt_float(v) if isinstance(v, float) else v}"
                            for k, v in self.params.clusterer_params.items()
                        )
                        stem = (
                            f"res{int(self.params.resolutions)}"
                            f"_gain{_fmt_float(float(self.params.gains))}"
                            f"_{_slug(self.params.clusterer_name)}"
                        )
                        if param_slug:
                            stem += f"_{param_slug}"
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

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Union, Dict, Any

import json
import numpy as np
import trimesh
import networkx as nx
from matplotlib import pyplot as plt
from cereeberus import ReebGraph

from src.Sampler import Sampler 
from src.Shapes import Generate
from src.rg_shapes import Shapes_rg

Geometry = Union[trimesh.Trimesh, trimesh.path.Path2D, trimesh.path.Path3D]


def _default_height(points_xyz: np.ndarray) -> np.ndarray:
    """Default filter: x-height."""
    return points_xyz[:, 0].astype(float)


@dataclass
class ShapeSample:
    name: str
    shape: Geometry
    rg: ReebGraph
    height_function: Optional[Callable[[np.ndarray], np.ndarray]] = None  # (n,3)->(n,)

    samples: int = 500
    seed: Optional[int] = 2

    _sampler: Sampler = field(init=False, repr=False)

    # filled after sample()
    sampled_points: Optional[np.ndarray] = field(default=None, repr=False)
    sampled_f: Optional[np.ndarray] = field(default=None, repr=False)
    sampled_mode: Optional[str] = None

    def __post_init__(self):
        self._sampler = Sampler(seed=self.seed)
        if self.height_function is None:
            self.height_function = _default_height

    def sample(self, *, mode: Optional[str] = None, curve_detail: int = 64) -> None:
        """
        Sample points and evaluate the filter on them.
        - Trimesh: mode in {'surface','even'} (default 'surface')
        - Path2D/3D: mode must be 'length'
        """
        res = self._sampler.sample(self.shape, self.samples, mode=mode, curve_detail=curve_detail)
        self.sampled_points = res.points
        self.sampled_mode = res.mode

        fvals = np.asarray(self.height_function(self.sampled_points), dtype=float)
        if fvals.shape != (self.sampled_points.shape[0],):
            raise ValueError("height_function must return shape (n,) for n sampled points")
        self.sampled_f = fvals

    def visualize_points(self) -> None:
        """
        Use the Sampler's viewer to display the shape + sampled points.
        """
        if self.sampled_points is None:
            raise RuntimeError("Call .sample(...) first.")
        self._sampler.visualize(self.shape, self.sampled_points)

    def visualize_rg(self) -> None:
        """
        Visualize the Reeb graph with f-values.
        """
        if self.rg is None:
            raise RuntimeError("No Reeb graph to visualize.")
        self.rg.draw(cpx = 2.0)
        plt.show()


    def save(self, out_dir: Path) -> Dict[str, Path]:
        """
        Write a Mapper-friendly bundle:
          - points.csv (x,y,z,f)
          - meta.json
          - reeb.json (optional, lightweight)
        """
        if self.sampled_points is None or self.sampled_f is None:
            raise RuntimeError("Call .sample(...) before saving")

        folder = Path(out_dir) / self.name
        folder.mkdir(parents=True, exist_ok=True)

        # points
        ptsf = np.c_[self.sampled_points, self.sampled_f]
        csv_path = folder / "points.csv"
        np.savetxt(csv_path, ptsf, delimiter=",", header="x,y,z,f", comments="")

        # meta
        meta = {
            "id": self.id,
            "name": self.name,
            "samples": self.samples,
            "sampled_mode": self.sampled_mode,
            "geometry_type": self.shape.__class__.__name__,
            "height_fn": getattr(self.height_function, "__name__", "custom"),
            "has_reeb": True,
        }
        meta_path = folder / "meta.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        # reeb.json (don’t treat 0.0 as missing)
        rg_path = None
        try:
            G = self.rg if isinstance(self.rg, nx.MultiDiGraph) else getattr(self.rg, "G", self.rg)
            nodes = []
            for u, data in G.nodes(data=True):
                if "f_vertex" in data:
                    f_u = float(data["f_vertex"])
                elif "f" in data:
                    f_u = float(data["f"])
                else:
                    f_u = None
                try:
                    uid = int(u)
                except Exception:
                    uid = u
                nodes.append({"id": uid, "f": f_u})
            edges = [{"u": u, "v": v, "key": k} for u, v, k in G.edges(keys=True)]
            rg_path = folder / "reeb.json"
            with open(rg_path, "w", encoding="utf-8") as f:
                json.dump({"nodes": nodes, "edges": edges}, f, indent=2, ensure_ascii=False)
        except Exception:
            pass

        return {"points_csv": csv_path, "meta_json": meta_path, "reeb_json": rg_path}


''' Test case for ShapeSample with Path2D and x-height
def circle_item(radius=1.0, samples=500, seed=2) -> ShapeSample:
    circle = Generate.make_circle(radius=radius, sections=64)      # Path2D
    rg = Shapes_rg.circle_rg(radius=radius)

    def f_y(pts: np.ndarray) -> np.ndarray:
        return pts[:, 0]   # y-height

    item = ShapeSample(
        id=0,
        name=f"1D_circle_{samples}_y",
        shape=circle,
        rg=rg,
        height_function=f_y,
        samples=samples,
        seed=seed,
    )

    # No external sampler needed:
    item.sample(mode="length")   # Path -> 'length'
    item.visualize_points()             # uses internal sampler's viewer
    item.visualize_rg()          # built-in visualizer
    # item.save(Path("./dataset"))
    return item

if __name__ == "__main__":
    circle_item()
'''
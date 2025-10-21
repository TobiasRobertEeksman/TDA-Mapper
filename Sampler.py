import numpy as np
from dataclasses import dataclass
from pathlib import Path
import trimesh
from Shapes import Generate

@dataclass
class SampleResult:
    points: np.ndarray   # (n,3), z=0 for 2D
    mode: str            # 'surface' |  'length'


class Sampler:
    """
    Simplified sampler:
      - Trimesh:   'surface' (default)
      - Path2D/3D: 'length' (length-weighted along polyline discretization)
    """
    def __init__(self, seed: int | None = None):
        self.rng = np.random.default_rng(seed)

    def sample(
        self,
        obj: trimesh.Trimesh | trimesh.path.Path2D | trimesh.path.Path3D,
        n: int,
        *,
        mode: str | None = None,
        curve_detail: int = 48,  # discretization density for curved entities
    ) -> SampleResult:
        if isinstance(obj, trimesh.Trimesh):
            mode = mode or "surface"
            if mode == "surface":
                pts, _ = trimesh.sample.sample_surface(obj, n)
            else:
                raise ValueError("For Trimesh, mode must be 'surface''.")
            return SampleResult(points=pts, mode=mode)

        elif isinstance(obj, (trimesh.path.Path2D, trimesh.path.Path3D)):
            # Path sampling ignores mode except expecting 'length'
            mode = mode or "length"
            if mode != "length":
                raise ValueError("For Path2D/Path3D, mode must be 'length'.")
            pts = self._sample_along_path(obj, n, curve_detail=curve_detail)
            return SampleResult(points=pts, mode="length")

        else:
            raise TypeError(f"Unsupported type: {type(obj)}")

    # ---------- internals ----------

    def _sample_along_path(
        self,
        path: trimesh.path.Path2D | trimesh.path.Path3D,
        n: int,
        *,
        curve_detail: int,
    ) -> np.ndarray:
        """
        Length-weighted sampling along all entities.
        Works for graphs (Lines) and curved entities (Arcs, Splines) by discretizing.
        Returns (n,3); for 2D paths, z=0.
        """
        # Try robust discretization:
        polylines = []
        try:
            # Many trimesh versions provide a cached discretization as a property
            polylines = list(path.discrete)  # list of (k,2) or (k,3)
        except Exception:
            pass

        if not polylines:
            # Fallback: discretize each entity directly (more reliable on custom graphs)
            polylines = []
            V = np.asarray(path.vertices)
            for ent in path.entities:
                try:
                    # Some entities support a 'vertices' kw only; others also accept 'scale'
                    P = ent.discrete(V)
                except TypeError:
                    # last resort: try with a scale that increases samples on long curves
                    P = ent.discrete(V, scale=max(1.0, curve_detail / 16.0))
                if P is None or len(P) < 2:
                    continue
                polylines.append(np.asarray(P))

        if not polylines:
            raise ValueError("Path has no discretizable entities (no polylines).")

        # Build segment table across all polylines
        seg_starts, seg_ends, seg_lens = [], [], []
        for P in polylines:
            P = np.asarray(P)
            if P.ndim != 2 or P.shape[0] < 2:
                continue
            # Upgrade 2D -> 3D (z=0)
            if P.shape[1] == 2:
                P = np.c_[P, np.zeros((P.shape[0],), float)]
            # If entity is curved, optionally densify further for smoother length sampling
            if curve_detail and P.shape[0] < curve_detail:
                # linearly upsample to at least 'curve_detail' points
                P = _resample_polyline(P, max(curve_detail, P.shape[0]))

            d = np.linalg.norm(P[1:] - P[:-1], axis=1)
            keep = d > 0
            if np.any(keep):
                seg_starts.append(P[:-1][keep])
                seg_ends.append(P[1:][keep])
                seg_lens.append(d[keep])

        if not seg_lens:
            raise ValueError("No non-degenerate segments found in path.")

        starts = np.vstack(seg_starts)
        ends = np.vstack(seg_ends)
        lens = np.concatenate(seg_lens)

        # Length-weighted segment choice
        probs = lens / np.sum(lens)
        seg_idx = self.rng.choice(len(lens), size=n, p=probs)
        t = self.rng.random(n)  # uniform along chosen segment
        pts = starts[seg_idx] * (1.0 - t)[:, None] + ends[seg_idx] * t[:, None]
        return pts

    # ---------- quick viewer ----------

    def visualize(
        self,
        obj: trimesh.Trimesh | trimesh.path.Path2D | trimesh.path.Path3D,
        pts: np.ndarray,
        *,
        mesh_alpha: float = 0.3,
        point_size: float = 1.0,
        point_color=(250, 50, 50, 255),
    ):
        scene = trimesh.Scene()
        if isinstance(obj, trimesh.Trimesh):
            m = obj.copy()
            m.visual.face_colors = [200, 200, 220, int(255 * mesh_alpha)]
            scene.add_geometry(m)
        elif isinstance(obj, trimesh.path.Path2D):
            scene.add_geometry(obj.to_3D())
        elif isinstance(obj, trimesh.path.Path3D):
            scene.add_geometry(obj)
        cloud = trimesh.points.PointCloud(pts, colors=point_color)
        cloud.metadata["point_size"] = point_size
        scene.add_geometry(cloud)
        scene.show()


# ---- helper: simple polyline resampling ----
def _resample_polyline(P: np.ndarray, m: int) -> np.ndarray:
    """
    Linear resample a polyline P (n,3) to exactly m points along arc length.
    """
    seg = np.linalg.norm(P[1:] - P[:-1], axis=1)
    arclen = np.concatenate(([0.0], np.cumsum(seg)))
    total = arclen[-1]
    if total <= 0:
        return P.copy()
    u = np.linspace(0.0, total, m)
    # find segment for each u
    idx = np.searchsorted(arclen, u, side="right") - 1
    idx = np.clip(idx, 0, len(seg) - 1)
    t = (u - arclen[idx]) / np.maximum(seg[idx], 1e-12)
    return P[idx] * (1 - t)[:, None] + P[idx + 1] * t[:, None]



if __name__ == "__main__":
    S = Sampler(seed=42)

    # 3D mesh (surface)
    m = Generate.make_torus()
    res = S.sample(m, 2000, mode="surface")   # or mode="even"
    S.visualize(m, res.points)

    # 2D path (length along curves)
    p2 = Generate.make_annulus()              # Path2D of two circles
    res = S.sample(p2, 1500, mode="surface")
    S.visualize(p2, res.points)

    p3 = Generate.make_double_circle()
    res = S.sample(p3, 500, mode="length")
    S.visualize(p3, res.points)

import numpy as np
import trimesh
import gudhi
from cereeberus import ReebGraph, LowerStar, computeReeb
from shapely.geometry import Polygon, MultiPolygon

from src.ShapeClass import ShapeSample

from src.rg_shapes import Shapes_rg
from src.DataGenerator import subdivide

def triangulate_multigon(g):
    parts = []
    def add(x):
        if isinstance(x, Polygon):
            p = x.buffer(0)            # clean topology
            if not p.is_empty: parts.append(p)
        elif hasattr(x, "geoms"):       # MultiPolygon / GeometryCollection
            for y in x.geoms: add(y)
    add(g)
    V, F, off = [], [], 0
    for p in parts:
        v2, f = trimesh.creation.triangulate_polygon(p, engine="earcut")
        V.append(v2); F.append(f + off); off += len(v2)
    return np.vstack(V), np.vstack(F)   # verts2d, faces


def f_x(pts: np.ndarray) -> np.ndarray:
    return pts[:, 0].astype(float)

def create_lower_star(V,F):
    st = LowerStar()
    # vertices
    for i,v in enumerate(V):
        st.insert([i])

    for face in F:
        st.insert(face)

    for i, v in enumerate(V):
        st.assign_filtration([i], f_x(V)[i])

    return st

def sample_points(V2, F, n, seed=None):
    if seed is not None: np.random.seed(seed)
    m = trimesh.Trimesh(vertices=np.c_[V2, np.zeros(len(V2))], faces=F, process=False)
    P3, _ = trimesh.sample.sample_surface(m, int(n))
    return P3[:, :2]  # (n,2)

def convert(name: str, m: Polygon  | MultiPolygon, samples = 1000, seed = None, visualize=False):
    #shape as trimesh
    V, F = triangulate_multigon(m)
    shape = trimesh.Trimesh(vertices = V.tolist(), faces = F.tolist())

    #compute ReebGraph
    st1 = create_lower_star(V,F)
    # rg = computeReeb(st1)

    #placeholder rg until computeReeb fixed
    rg = subdivide(Shapes_rg.torus_rg(R=2.0, r=1.0))

    #sampling points
    sampled_points = sample_points(V,F,samples,seed)

    #sampled_f
    fvals = np.asarray(f_x(sampled_points), dtype=float)

    item = ShapeSample(
        name = name,
        shape=shape,
        rg=rg,
        height_function=f_x,
        samples=samples,
        seed = seed,

        sampled_points = sampled_points,
        sampled_f = fvals,
        sampled_mode = "surface"
        )
    
    if visualize:
        item.visualize_points()

    return item
import numpy as np
from .ShapeClass import ShapeSample
from .Shapes import Generate
from .rg_shapes import Shapes_rg
from pathlib import Path
from cereeberus import ReebGraph
import trimesh
from typing import Callable, Optional, Union

Geometry = Union[trimesh.Trimesh, trimesh.path.Path2D, trimesh.path.Path3D]

Current_shapes = ['annulus.json', 
                    'circle.json', 
                    'double_annulus.json', 
                    'double_circle.json', 
                    'double_torus.json', 
                    'double_triangle.json', 
                    'graph_grid.json', 
                    'graph_Y.json', 
                    'sphere.json', 
                    'tetraeder_hole.json', 
                    'torus.json']


def _fmt_float(x: float) -> str:
    # compact + filesystem friendly (replace '.' with 'p')
    return f"{x:.4g}".replace(".", "p")

class DataGenerator:

    def circle_item(radius=1.0, samples=500, seed=2, visualize = True) -> ShapeSample:
        circle = Generate.make_circle(radius=radius, sections=64)      # Path2D
        rg = Shapes_rg.circle_rg(radius=radius)

        def f_x(pts: np.ndarray) -> np.ndarray:
            return pts[:, 0]   # x-height

        item = ShapeSample(
            id=0,
            name=f"1D_circle_r{_fmt_float(float(radius))}_S{samples}_x",
            shape=circle,
            rg=rg,
            height_function=f_x,
            samples=samples,
            seed=seed,
        )

        # No external sampler needed:
        item.sample(mode="length")   # Path -> 'length'
        if visualize:
            item.visualize_points()             # uses internal sampler's viewer
            item.visualize_rg()          # built-in visualizer
        # item.save(Path("./dataset"))
        return item
    
    def double_circle_item(r1=1.0, r2=3.0, samples=500, seed=2, visualize=True) -> ShapeSample:
        double_circle = Generate.make_double_circle(r1=r1, r2=r2, sections=64)  # Path2D
        rg = Shapes_rg.double_circle_rg(r1=r1, r2=r2)

        def f_x(pts: np.ndarray) -> np.ndarray:
            return pts[:, 0]   # x-height

        item = ShapeSample(
            id=1,
            name=f"1D_double_circle_r1{_fmt_float(float(r1))}_r2{_fmt_float(float(r2))}_S{samples}_x",
            shape=double_circle,
            rg=rg,
            height_function=f_x,
            samples=samples,
            seed=seed,
        )

        item.sample(mode="length")   # Path -> 'length'
        if visualize:
            item.visualize_points()             # uses internal sampler's viewer
            item.visualize_rg()          # built-in visualizer
        # item.save(Path("./dataset"))
        return item
    
    def annulus_item(R=2.0, r = 1.0, samples=500, seed=2, visualize=True) -> ShapeSample:
        annulus = Generate.make_annulus(R=R, r=r, sections=64)  # Path2D
        rg = Shapes_rg.torus_rg(R=R, r=r)

        def f_x(pts: np.ndarray) -> np.ndarray:
            return pts[:, 0]   # x-height

        item = ShapeSample(
            id=2,
            name=f"2D_annulus_R{_fmt_float(float(R))}_r{_fmt_float(float(r))}_S{samples}_x",
            shape=annulus,
            rg=rg,
            height_function=f_x,
            samples=samples,
            seed=seed,
        )

        item.sample(mode="surface")   # Path -> 'length'
        if visualize:
            item.visualize_points()             # uses internal sampler's viewer
            item.visualize_rg()          # built-in visualizer
        # item.save(Path("./dataset"))
        return item
    
    def double_annulus_item(R1=1.0, r1=0.5, R2=0.8, r2=0.3, samples=500, seed=2, visualize = True) -> ShapeSample:
        double_annulus = Generate.make_double_annulus(R1=R1, r1=r1, R2=R2, r2=r2)  # Path2D
        rg = Shapes_rg.double_torus_rg(R1=R1, r1=r1, R2=R2, r2=r2, shift = R1+0.5*R2)

        def f_x(pts: np.ndarray) -> np.ndarray:
            return pts[:, 0]   # x-height

        item = ShapeSample(
            id=3,
            name=f"2D_double_annulus_R1{_fmt_float(float(R1))}_r1{_fmt_float(float(r1))}_R2{_fmt_float(float(R2))}_r2{_fmt_float(float(r2))}_S{samples}_x",
            shape=double_annulus,
            rg=rg,
            height_function=f_x,
            samples=samples,
            seed=seed,
        )

        item.sample(mode="surface")   # Path -> 'length'
        if visualize:
            item.visualize_points()             # uses internal sampler's viewer
            item.visualize_rg()          # built-in visualizer
        # item.save(Path("./dataset"))
        return item
    
    def sphere_item(radius=1.0, samples=2000, seed=2, visualize=True) -> ShapeSample:
        sphere = Generate.make_sphere(radius=radius, subdivisions=3)  # Mesh
        rg = Shapes_rg.sphere_rg(radius=radius)

        def f_x(pts: np.ndarray) -> np.ndarray:
            return pts[:, 0]   # x-height

        item = ShapeSample(
            id=5,
            name=f"3D_sphere_radius{_fmt_float(float(radius))}_S{samples}_x",
            shape=sphere,
            rg=rg,
            height_function=f_x,
            samples=samples,
            seed=seed,
        )

        item.sample(mode="surface")   # Mesh -> 'surface'
        if visualize:
            item.visualize_points()             # uses internal sampler's viewer
            item.visualize_rg()          # built-in visualizer
        # item.save(Path("./dataset"))
        return item
    
    def torus_item(R=2.0, r=1.0, samples=2000, seed=2, visualize=True) -> ShapeSample:
        torus = Generate.make_torus(R=R, r=r)  # Mesh
        rg = Shapes_rg.torus_rg(R=R, r=r)

        def f_x(pts: np.ndarray) -> np.ndarray:
            return pts[:, 0]   # x-height

        item = ShapeSample(
            id=4,
            name=f"3D_torus_R{_fmt_float(float(R))}_r{_fmt_float(float(r))}_S{samples}_x",
            shape=torus,
            rg=rg,
            height_function=f_x,
            samples=samples,
            seed=seed,
        )

        item.sample(mode="surface")   # Mesh -> 'surface'
        if visualize:
            item.visualize_points()             # uses internal sampler's viewer
            item.visualize_rg()          # built-in visualizer
        # item.save(out_dir=OUT)
        return item
    
    def double_torus_item(R1=2.0, r1=0.6, R2=1.6, r2=0.6, samples=2000, seed=2, visualize=True) -> ShapeSample:
        double_torus = Generate.make_double_torus(R1=R1, r1=r1, R2=R2, r2=r2)  # Mesh
        rg = Shapes_rg.double_torus_rg(R1=R1, r1=r1, R2=R2, r2=r2, shift=R1 + R2)

        def f_x(pts: np.ndarray) -> np.ndarray:
            return pts[:, 0]   # x-height

        item = ShapeSample(
            id=6,
            name=f"3D_double_torus_R1{_fmt_float(float(R1))}_r1{_fmt_float(float(r1))}_R2{_fmt_float(float(R2))}_r2{_fmt_float(float(r2))}_S{samples}_x",
            shape=double_torus,
            rg=rg,
            height_function=f_x,
            samples=samples,
            seed=seed,
        )

        item.sample(mode="surface")   # Mesh -> 'surface'
        if visualize:
            item.visualize_points()             # uses internal sampler's viewer
            item.visualize_rg()          # built-in visualizer
        # item.save(out_dir=OUT)
        return item
    

    def add_shape(
        *,
        shape: Geometry,
        rg: ReebGraph,
        f: Callable[[np.ndarray], np.ndarray] | None,
        mode: Optional[str],
        samples: int,
        seed: Optional[int] = 42,
        name: str,
        id: int,
        visualize: bool = True,
    ) -> ShapeSample:
        """
        Create a ShapeSample from external components, with validation, sampling,
        and optional visualization. Returns the populated ShapeSample.
        """
        # ---- validations (with clear messages) ----
        ok = True

        if not isinstance(shape, (trimesh.Trimesh, trimesh.path.Path2D, trimesh.path.Path3D)):
            print("[add_shape] ERROR: 'shape' must be a trimesh.Trimesh or Path2D/Path3D.")
            ok = False

        # Accept ceREEBerus ReebGraph or raw networkx MultiDiGraph
        if not (isinstance(rg, ReebGraph) or isinstance(rg, nx.MultiDiGraph)):
            print("[add_shape] ERROR: 'rg' must be a ceREEBerus ReebGraph")
            ok = False

        if not isinstance(samples, int) or samples <= 0:
            print(f"[add_shape] ERROR: 'samples' must be a positive int, got {samples!r}.")
            ok = False

        if f is not None and not callable(f):
            print("[add_shape] ERROR: 'f' (height_function) must be callable:" \
            "example: f = lambda pts: pts[:,2] for z-height.")
            ok = False

        # Mode defaults & compatibility
        if mode is None:
            mode = "surface" if isinstance(shape, trimesh.Trimesh) else "length"

        if mode not in ("surface", "length"):
            print("[add_shape] ERROR: 'mode' must be one of {'surface','length'}.")
            ok = False

        if isinstance(shape, trimesh.Trimesh) and mode not in ("surface"):
            print("[add_shape] ERROR: Mesh shape requires mode 'surface'.")
            ok = False

        if isinstance(shape, (trimesh.path.Path2D, trimesh.path.Path3D)) and mode != "length":
            print("[add_shape] ERROR: Graph shape requires mode 'length'.")
            ok = False

        if not ok:
            raise ValueError("[add_shape] Aborting due to invalid inputs.")

        # ---- build sample item ----
        item = ShapeSample(
            id=id,
            name=name,
            shape=shape,
            rg=rg,
            height_function=f,   # defaults to z-height if None
            samples=samples,
            seed=seed,
        )

        # ---- sample + evaluate f (with helpful error reporting) ----
        try:
            item.sample(mode=mode)
        except Exception as e:
            print(f"[add_shape] ERROR during sampling/evaluating f: {e}")
            raise

        # ---- visualize (optional) ----
        if visualize:
            try:
                item.visualize_points()
                item.visualize_rg()
            except Exception as e:
                print(f"[add_shape] WARNING: visualization failed: {e}")

        return item

    
    def save_all():
        ROOT = Path(".")
        OUT = ROOT / "data" / "processed_shapes"
        OUT.mkdir(parents=True, exist_ok=True)

        for i, shape_name in enumerate(Current_shapes):
            print(f"Generating shape {i+1}/{len(Current_shapes)}: {shape_name}")
            if shape_name == 'circle.json':
                item = DataGenerator.circle_item()
            elif shape_name == 'double_circle.json':
                item = DataGenerator.double_circle_item()
            elif shape_name == 'annulus.json':
                item = DataGenerator.annulus_item()
            elif shape_name == 'double_annulus.json':
                item = DataGenerator.double_annulus_item()
            elif shape_name == 'sphere.json':
                item = DataGenerator.sphere_item()
            elif shape_name == 'torus.json':
                item = DataGenerator.torus_item()
            elif shape_name == 'double_torus.json':
                item = DataGenerator.double_torus_item()
            else:
                print(f"Shape {shape_name} not implemented.")
                continue
            item.save(out_dir=OUT)


if __name__ == "__main__":
    DataGenerator.save_all()
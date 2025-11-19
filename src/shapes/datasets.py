import numpy as np
from pathlib import Path
from cereeberus import ReebGraph
import trimesh
from typing import Callable, Optional, Union
import networkx as nx

from src.shapes.base import ShapeSample
from src.shapes.meshes import Generate
from src.shapes.reeb_graphs import Shapes_rg, subdivide

from src.helper import _fmt_float

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


class DataGenerator:

    @staticmethod
    def circle_item(radius=1.0, samples=500, seed=2, visualize = True) -> ShapeSample:
        circle = Generate.make_circle(radius=radius, sections=64)      # Path2D
        rg = subdivide(Shapes_rg.circle_rg(radius=radius))

        def f_x(pts: np.ndarray) -> np.ndarray:
            return pts[:, 0]   # x-height

        item = ShapeSample(
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
    
    @staticmethod
    def double_circle_item(r1=1.0, r2=3.0, samples=500, seed=2, visualize=True) -> ShapeSample:
        double_circle = Generate.make_double_circle(r1=r1, r2=r2, sections=64)  # Path2D
        rg = subdivide(Shapes_rg.double_circle_rg(r1=r1, r2=r2))

        def f_x(pts: np.ndarray) -> np.ndarray:
            return pts[:, 0]   # x-height

        item = ShapeSample(
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
    
    @staticmethod
    def annulus_item(R=2.0, r = 1.0, samples=500, seed=2, visualize=True) -> ShapeSample:
        annulus = Generate.make_annulus(R=R, r=r, sections=64)  # Path2D
        rg = subdivide(Shapes_rg.torus_rg(R=R, r=r))

        def f_x(pts: np.ndarray) -> np.ndarray:
            return pts[:, 0]   # x-height

        item = ShapeSample(
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
    
    @staticmethod
    def double_annulus_item(R1=1.0, r1=0.5, R2=0.8, r2=0.3, samples=500, seed=2, visualize = True) -> ShapeSample:
        double_annulus = Generate.make_double_annulus(R1=R1, r1=r1, R2=R2, r2=r2)  # Path2D
        rg = subdivide(Shapes_rg.double_torus_rg(R1=R1, r1=r1, R2=R2, r2=r2, shift = R1+0.5*R2))

        def f_x(pts: np.ndarray) -> np.ndarray:
            return pts[:, 0]   # x-height

        item = ShapeSample(
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
    
    @staticmethod
    def sphere_item(radius=1.0, samples=2000, seed=2, visualize=True) -> ShapeSample:
        sphere = Generate.make_sphere(radius=radius, subdivisions=3)  # Mesh
        rg = subdivide(Shapes_rg.sphere_rg(radius=radius))

        def f_x(pts: np.ndarray) -> np.ndarray:
            return pts[:, 0]   # x-height

        item = ShapeSample(
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
    
    @staticmethod
    def torus_item(R=2.0, r=1.0, samples=2000, seed=2, visualize=True) -> ShapeSample:
        torus = Generate.make_torus(R=R, r=r)  # Mesh
        rg = subdivide(Shapes_rg.torus_rg(R=R, r=r))

        def f_x(pts: np.ndarray) -> np.ndarray:
            return pts[:, 0]   # x-height

        item = ShapeSample(
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
    
    @staticmethod
    def double_torus_item(R1=2.0, r1=0.6, R2=1.6, r2=0.6, samples=2000, seed=2, visualize=True) -> ShapeSample:
        double_torus = Generate.make_double_torus(R1=R1, r1=r1, R2=R2, r2=r2)  # Mesh
        rg = subdivide(Shapes_rg.double_torus_rg(R1=R1, r1=r1, R2=R2, r2=r2, shift=R1 + R2))

        def f_x(pts: np.ndarray) -> np.ndarray:
            return pts[:, 0]   # x-height

        item = ShapeSample(
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
    
    @staticmethod
    def add_shape(
        *,
        shape: Geometry,
        rg: ReebGraph,
        f: Callable[[np.ndarray], np.ndarray] | None,
        mode: Optional[str],
        samples: int,
        seed: Optional[int] = 42,
        name: str,
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
            "example: f = lambda pts: pts[:,0] for x-height.")
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
            name=name,
            shape=shape,
            rg=subdivide(rg),
            height_function=f,   # defaults to x-height if None
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

    @staticmethod
    def double_torus_overlap(R1 = 2.0, r1 = 0.2, R2 = 1.0, r2 = 0.2, samples = 1000, visualize = True):

        #trimesh shape
        torus1 = trimesh.creation.torus(major_radius=R1, minor_radius=r1)
        torus2 = trimesh.creation.torus(major_radius=R2, minor_radius=r2)
        shift = np.array([0.0, 2.0, 0.0])
        torus2.apply_translation(shift)
        shape = trimesh.util.concatenate([torus1, torus2])
        
        #Reeb Graph
        rg = ReebGraph()
        rg.add_node(0, f_vertex=-R1)
        rg.add_node(1, f_vertex=-R2)
        rg.add_node(2, f_vertex=-R2)
        rg.add_node(3, f_vertex=R2)
        rg.add_node(4, f_vertex=R2)
        rg.add_node(5, f_vertex=R1)

        rg.add_edge(0, 1)
        rg.add_edge(0, 2)
        rg.add_edge(1, 3)
        rg.add_edge(1, 3)
        rg.add_edge(2, 4)
        rg.add_edge(2, 4)
        rg.add_edge(3, 5)
        rg.add_edge(4, 5)


        #height function
        f_x = lambda pts: pts[:, 0]  # x-height

        return DataGenerator.add_shape(
            name = f"3D_double_torus_yshift2_R1{_fmt_float(float(R1))}_r1{_fmt_float(float(r1))}_R2{_fmt_float(float(R2))}_r2{_fmt_float(float(r2))}_S{samples}_x",
            shape=shape,
            rg=rg,
            mode="surface",
            f=f_x,
            samples=samples,
            visualize=visualize,
        )

    @staticmethod
    def box_item(l = 2.0, samples = 1000, visualize = True):
        #trimesh shape
        box = trimesh.creation.box(extents=(l, l, l))
        
        #Reeb Graph
        rg = ReebGraph()
        rg.add_node(0, f_vertex=-l/2)
        rg.add_node(1, f_vertex=l/2)
        rg.add_edge(0, 1)

        #height function
        f_x = lambda pts: pts[:, 0]  # x-height

        return DataGenerator.add_shape(
            name = f"3D_box_l{_fmt_float(float(l))}_S{samples}_x",
            shape=box,
            rg=rg,
            mode="surface",
            f=f_x,
            samples=samples,
            visualize=visualize,
        )

    @staticmethod
    def briefcase_item(x = 2.0, y = 4.0, z = 1.0, R = 0.5, r = 0.1, samples = 1000, visualize = True):
        #trimesh shape
        box = trimesh.creation.box(extents=(x, y, z))
        torus = trimesh.creation.torus(major_radius=R, minor_radius=r)
        shift = np.array([x/2, 0.0, 0.0])
        torus.apply_translation(shift)
        briefcase = trimesh.util.concatenate([box, torus])

        #Reeb Graph
        rg = ReebGraph()
        rg.add_node(0, f_vertex=-x/2)
        rg.add_node(1, f_vertex=x/2)
        rg.add_node(2, f_vertex = x/2 + R-r)
        rg.add_node(3, f_vertex = x/2 + R+r)
        rg.add_edge(0, 1)
        rg.add_edge(1, 2)
        rg.add_edge(1, 2)
        rg.add_edge(2, 3)

        #height function
        f_y = lambda pts: pts[:, 1]  # y-height

        return DataGenerator.add_shape(
            name = f"3D_briecase_x{_fmt_float(float(x))}_y{_fmt_float(float(y))}_z{_fmt_float(float(z))}_S{samples}_y",
            shape=briefcase,
            rg=rg,
            mode="surface",
            f=f_y,
            samples=samples,
            visualize=visualize,
        )


    @staticmethod
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


# if __name__ == "__main__":
#     DataGenerator.save_all()
from pathlib import Path
import numpy as np
import trimesh
import json
import os




class Generate():

    @staticmethod
    def make_sphere(radius=1.0, subdivisions=4) -> trimesh.Trimesh:
        return trimesh.creation.icosphere(subdivisions=subdivisions, radius=radius)

    @staticmethod
    def make_circle(radius=1.0, sections=32):
        return trimesh.path.creation.circle(
            radius=radius,
            sections=sections,
        )
    
    @staticmethod
    def make_graph_path_Y():
        # Simple 3D 'Y' graph with straight edges
        v = np.array([
            [0.0, 0.0, 0.0],   # 0 center
            [0.0, 1.2, 0.0],   # 1 up
            [-1.0, -0.8, 0.0], # 2 left-down
            [ 1.0, -0.8, 0.0], # 3 right-down
        ])
        entities = [
            trimesh.path.entities.Line([0, 1]),
            trimesh.path.entities.Line([0, 2]),
            trimesh.path.entities.Line([0, 3]),
        ]
        return trimesh.path.Path3D(entities=entities, vertices=v)

    @staticmethod
    def make_graph_path_grid(n=5, spacing=0.4):
        verts, lines = [], []
        idx = lambda i, j: i*n + j
        for i in range(n):
            for j in range(n):
                verts.append([i*spacing, j*spacing, 0.0])
        for i in range(n):
            for j in range(n):
                if i+1 < n: lines.append(trimesh.path.entities.Line([idx(i,j), idx(i+1,j)]))
                if j+1 < n: lines.append(trimesh.path.entities.Line([idx(i,j), idx(i,j+1)]))
        return trimesh.path.Path3D(entities=lines, vertices=np.array(verts, dtype=float))

    @staticmethod
    def make_double_circle(r1=1.0, r2=0.6, sections=256):
        """
        Two disjoint circles in the plane that touch (are externally tangent) at exactly one point.
        Centers lie on the x-axis, distance = r1 + r2.

        Returns a Path2D with two closed curves (no winding inversion; just two circles).
        """
        assert r1 > 0 and r2 > 0, "Radii must be positive."

        # place centers so they touch at x = r1 and x = -(r2) + (r1+r2) = r1  -> same point
        c1 = np.array([0.0, 0.0])
        c2 = np.array([r1 + r2, 0.0])  # external tangency

        circ1 = trimesh.path.creation.circle(radius=r1, sections=sections)
        circ2 = trimesh.path.creation.circle(radius=r2, sections=sections)

        circ1.apply_translation(c1)
        circ2.apply_translation(c2)

        # For exactly one touching point, ensure they are not welded/merged.
        return trimesh.util.concatenate([circ1, circ2])

    # ---------------------------
    # 3D: two tori tangent at 1 point
    # ---------------------------
    @staticmethod
    def make_torus(R=2.0, r=0.5) -> trimesh.Trimesh:
        """
        Minimal, dependency-light torus generator (axis = z).
        R = major radius (distance from center to tube center),
        r = minor radius (tube radius).
        """
        assert R > r > 0, "Require R > r > 0 for a standard (ring) torus."
        return trimesh.creation.torus(major_radius=R, minor_radius=r)

    @staticmethod
    def make_double_torus(
        R1=2.0, r1=0.6,
        R2=1.6, r2=0.6,
    ) -> trimesh.Trimesh:
        """
        Two standard tori (axis=z) that touch at exactly one point (inner sides),
        with no overlap. We place their centers on the x-axis.

        If the tori share the z-axis orientation and are centered at x = -d/2 and +d/2,
        the facing (inner) sides lie at x = center ± (R - r).
        To be tangent there: d = (R1 - r1) + (R2 - r2).

        Returns a single Trimesh with two disconnected components touching at one point.
        """
        assert R1 > r1 > 0 and R2 > r2 > 0, "Each torus must satisfy R > r > 0."
        # centers along x-axis
        c1 = np.array([0.0, 0.0, 0.0])
        c2 = np.array([R1+R2, 0.0, 0.0])

        t1 = Generate.make_torus(R=R1, r=r1)
        t2 = Generate.make_torus(R=R2, r=r2)

        t1.apply_translation(c1)
        t2.apply_translation(c2)

        # Important: don't merge vertices; keep them as two components that just kiss.
        return trimesh.util.concatenate([t1, t2])
    
    @staticmethod
    def make_annulus(R=1.0, r=0.3, sections=64):
        """
        Flat ring (annulus) in the XY plane, centered at the origin.
        Inner radius < outer radius.
        """
        assert 0 < r < R, "Require 0 < r_inner < r_outer."
        return trimesh.creation.annulus(r_min=r, r_max=R, height=0, sections=sections)

    @staticmethod
    def make_double_annulus(
        r1=0.5, R1=1.0,
        r2=0.3, R2=0.8,
        sections=64,
    ):
        """
        Two annuli (flat rings) in the XY plane that touch.
        The annuli are centered on the x-axis, distance = r1_outer + 1/2* r2_outer.

        Returns a Path2D with two closed curves (no winding inversion; just two annuli).
        """
        assert 0 < r1 < R1, "Require 0 < r1_inner < r1_outer."
        assert 0 < r2 < R2, "Require 0 < r2_inner < r2_outer."

        # place centers so they touch at x = r1_outer and x = -(r2_inner) + (r1_outer+r2_inner) = r1_outer  -> same point
        c1 = np.array([0.0, 0.0])
        c2 = np.array([R1+0.5*R2, 0.0])  # external tangency

        ann1 = trimesh.creation.annulus(r_min=r1, r_max=R1, height = 0, sections=sections)
        ann2 = trimesh.creation.annulus(r_min=r2, r_max=R2, height = 0, sections=sections)

        ann1.apply_translation(np.r_[c1, 0.0])  # Path2D accepts (x, y, z)
        ann2.apply_translation(np.r_[c2, 0.0])

        # For exactly one touching point, ensure they are not welded/merged.
        return trimesh.util.concatenate([ann1, ann2]) 
    
    @staticmethod
    def make_double_triangle():
        """
        Two triangles in the plane that touch (are externally tangent) at exactly one point.
        One triangle is upright, the other upside-down.
        Centers lie on the x-axis, distance = 1.5.

        Returns a Path2D with two closed curves (no winding inversion; just two triangles).
        """
        v = np.array([
            [0.5, 0.0],
            [1.0, 1.0],
            [1.0, -1.0],
            [-0.5, 0.0],
            [-1.0, 1.0],
            [-1.0, -1.0],
        ])

        entities = [
            trimesh.path.entities.Line([0, 1]),
            trimesh.path.entities.Line([1, 2]),
            trimesh.path.entities.Line([2, 0]),
            trimesh.path.entities.Line([0, 3]),
            trimesh.path.entities.Line([3, 4]),
            trimesh.path.entities.Line([4, 5]),
            trimesh.path.entities.Line([5, 3]),
        ]

        return trimesh.path.Path3D(entities=entities, vertices=v)
    
    @staticmethod
    def make_tetraeder_hole() -> trimesh.Trimesh:
        V = np.array([
            [ 1.0,  1.0,  1.0],   # 0
            [-1.0, -1.0,  1.0],   # 1
            [-1.0,  1.0, -1.0],   # 2
            [ 1.0, -1.0, -1.0],   # 3
        ], dtype=float)

        # Faces with consistent winding (outward normals).
        # These work with the vertex ordering above.
        F = np.array([
            [0, 1, 2],
            [0, 2, 3],
            [1, 2, 3]
        ], dtype=int)

        mesh = trimesh.Trimesh(vertices=V, faces=F)

        return mesh

''' Save geometry and metadata, but now in DataGenerator.py 

ROOT = Path(".")
OUT = ROOT / "data" / "raw_shapes"
OUT.mkdir(parents=True, exist_ok=True)

def save_geometry(
    obj: trimesh.Trimesh | trimesh.path.Path2D | trimesh.path.Path3D,
    name: str,
    out_dir: Path = OUT,
) -> tuple[Path, dict]:
    """
    Save a geometry to a viewer-friendly file.
      - Trimesh -> .ply
      - Path2D/Path3D -> .dxf (explicit file_type to avoid inference issues)

    Returns (file_path, metadata) where metadata is:
      {"name": <name>, "type": "1D"|"2D"|"3D", "path": <absolute path>}
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(obj, trimesh.Trimesh):
        ext = "ply"
        dim_type = "3D"
        file_path = out_dir / f"{name}.{ext}"
        # For meshes, file type inference is usually fine, but we can be explicit anyway
        obj.export(str(file_path), file_type=ext)

    elif isinstance(obj, (trimesh.path.Path2D, trimesh.path.Path3D)):
        ext = "dxf"
        dim_type = "1D"
        file_path = out_dir / f"{name}.{ext}"
        try:
            # Be explicit for paths; DXF needs ezdxf installed.
            obj.export(str(file_path), file_type=ext)
        except (KeyError, ImportError) as e:
            # Optional fallback for 2D paths if DXF export isn't available
            if isinstance(obj, trimesh.path.Path2D):
                ext = "svg"
                file_path = out_dir / f"{name}.{ext}"
                obj.export(str(file_path), file_type=ext)
            else:
                raise RuntimeError(
                    "DXF export for Path3D failed. Install the optional dependency:\n"
                    "    pip install ezdxf\n"
                    f"Original error: {e}"
                )

    else:
        raise TypeError(f"Unsupported type: {type(obj)}. Expected Trimesh or Path2D/Path3D.")

    meta = {
        "name": name,
        "type": dim_type,
        "path": str(file_path.resolve()),
    }
    return file_path, meta


def save_metadata_for_each(
    metas: list[dict],
    meta_dir: Path | None = None,
    relative_paths: bool = False,
    base_for_relative: Path | None = None,
) -> list[Path]:
    """
    Write one JSON per object with {name, type, path}.

    - meta_dir: where to put the JSONs (default: OUT/metadata)
    - relative_paths: if True, store 'path' relative to `base_for_relative` (or meta_dir)
    """
    if meta_dir is None:
        meta_dir = ROOT / "data" / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)

    out_paths = []
    for m in metas:
        m = dict(m)  # copy
        if relative_paths:
            base = base_for_relative or meta_dir
            m["path"] = os.path.relpath(m["path"], start=base)

        json_path = meta_dir / f"{m['name']}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(m, f, ensure_ascii=False, indent=2)
        out_paths.append(json_path)

    return out_paths
'''

def show(path_obj):
    if isinstance(path_obj, trimesh.Scene):
        scene = path_obj
    elif isinstance(path_obj, trimesh.Trimesh):
        scene = trimesh.Scene(path_obj)
    elif isinstance(path_obj, (trimesh.path.Path2D, trimesh.path.Path3D)):
        scene = path_obj.scene()
    else:
        raise TypeError(f"Don't know how to show {type(path_obj)}")
    scene.show()  # viewer window (pyglet backend)

def show_all() -> None:
    """
    Preview all demo geometries.
    """
    to_show = [
        Generate.make_sphere(),
        Generate.make_torus(),
        Generate.make_double_torus(),
        Generate.make_circle(),
        Generate.make_double_circle(),
        Generate.make_graph_path_Y(),
        Generate.make_graph_path_grid(),
        Generate.make_annulus(),
        Generate.make_double_annulus(),
        Generate.make_double_triangle(),
        Generate.make_tetraeder_hole(),
    ]
    for obj in to_show:
        show(obj)

    
if __name__ == "__main__":
    show_all()




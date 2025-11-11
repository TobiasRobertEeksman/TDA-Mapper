# TDAMapper (Mapper experiments on synthetic shapes)

Tools to generate synthetic geometric datasets (meshes & paths), define filters, and run the Mapper algorithm.  
Results (graphs + PNGs) are saved per shape and can be summarized in a grid image.

---

```bash
git clone https://github.com/TobiasRobertEeksman/TDA-Mapper.git
cd TDA-Mapper
python -m venv .venv
. .venv/Scripts/activate   # or source .venv/bin/activate (mac/linux)
```

## 🧩 Requirements

- Python 3.10+
- `numpy`, `networkx`, `scikit-learn`, `trimesh`, `Pillow`, `matplotlib`
- `cereeberus` for ReebGraph utilities

Install dependencies inside your virtual environment:
```bash
python -m pip install -r requirements.txt
```

---

## 📁 Project structure

```
TDAMapper/
  main.py                      # personal / custom experiments (multiple clusterers, etc.)
  mapper_generator.py          # reusable grid runner + simple example
  distance.py
  src/
    DataGenerator.py
    distance_grid.py
    Mapper.py
    rg_shapes.py
    Sampler.py
    ShapeClass.py
    Shapes.py
    visualize_rg.py
  mapper_results/              # outputs per shape (created at runtime)
    <shape-name>/
      <clusterer-name>/
        res*_gain*_*.png
    distance_grid/             # heatmap(s)
```

Run scripts as modules from the repo root:
```bash
# Example grid run on a predefined shape
python mapper_generator.py

# Your own multi-clusterer experiment (see main.py)
python main.py
```

---
## `run_grid_experiment`: core helper

Most Mapper experiments in this repo go through a single helper function defined in `mapper_generator.py`:

```python

from src.distance_grid import DistanceGrid
from src.Mapper import MapperParams, MapperSample
from distance import sublevel_distance_combined

def run_grid_experiment(
    *,
    item,
    resolutions,
    gains,
    clusterer_name,
    clusterer_function,
    clusterer_params,
    save_mapper: bool = True,
) -> tuple[str, str]:
    """
    Run the (res, gain) grid search for a fixed item + clusterer.
    Returns paths to the CSV and PNG.
    """
    grid = DistanceGrid()

    for res in resolutions:
        for g in gains:
            mapper_params = MapperParams(
                resolutions=res,
                gains=g,
                clusterer_name=clusterer_name,
                clusterer_function=clusterer_function,
                clusterer_params=clusterer_params,
            )

            mapper_sample = MapperSample(
                item=item,
                params=mapper_params,
                visualize=False,        # no GUI in batch mode
                save=save_mapper,       # save each Mapper graph as PNG
            )
            mapper_sample.run()

            d = sublevel_distance_combined(m=mapper_sample, rg=item.rg)
            grid.add(resolution=res, gain=g, distance=d)

    csv_path, png_path = grid.save(
        item_name=item.name,
        title=(
            f"Combined Sublevel distance to ReebGraph with clusterer: "
            f"{clusterer_name} and {clusterer_params}"
        ),
        base_dir="mapper_results",
        filename_stub="sublevel_distance",
        clusterer_name=clusterer_name,
    )
    return csv_path, png_path

```
You can either:

Run `mapper_generator.py` directly to see a complete working example, or

Import `run_grid_experiment` in your own scripts (e.g. `main.py`) and plug in different shapes / clusterers

---

## 🚀 Example usage with a predefined shape (using `mapper_generator.py`)

`mapper_generator.py` contains a minimal example that:

- Builds a synthetic shape (`double_torus_item`),
- Runs Mapper over a grid of `(resolutions, gains)` for a chosen clusterer,
- Saves the resulting Mapper graphs and a heatmap of distances to the Reeb graph.

A simplified version looks like this:

```python
# mapper_generator.py (simplified example)

from src.DataGenerator import DataGenerator
from sklearn.cluster import DBSCAN
from mapper_generator import run_grid_experiment  # if imported elsewhere

if __name__ == "__main__":
    # 1) build a shape / sample points (no GUI in batch)
    item = DataGenerator.double_torus_item(
        R1=1.9, r1=0.6,
        R2=0.8, r2=0.2,
        samples=1000,
        visualize=False,
    )

    # 2) grid of Mapper parameters
    resolutions = list(range(6, 16))
    gains = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]

    # 3) choose a clusterer
    clusterer_name = "dbscan"
    clusterer_function = DBSCAN
    clusterer_params = {"eps": 0.4, "min_samples": 5}

    # 4) run the grid
    csv_path, png_path = run_grid_experiment(
        item=item,
        resolutions=resolutions,
        gains=gains,
        clusterer_name=clusterer_name,
        clusterer_function=clusterer_function,
        clusterer_params=clusterer_params,
        save_mapper=True,
    )

    print(f"Saved grid CSV -> {csv_path}")
    print(f"Saved heatmap  -> {png_path}")

```
Run it with:
```bash
python mapper_generator.py
```

This will generate Mapper graphs for multiple parameter combinations and save the resulting
visualizations in `mapper_results/<shape-name>/<clusterer_name>`.
and a 2D heatmap of distances in `mapper_results/distance_grid`.

---

## 🧪 Custom experiments (e.g. multiple clusterers) with `main.py`


`main.py` is intended for personal experiments where you might:

- reuse the same sampled item,
- compare multiple clusterers on the exact same data,
- or change resolution/gain ranges.

Example:

```python
# main.py (example use)

from src.DataGenerator import DataGenerator
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from automato import Automato
from mapper_generator import run_grid_experiment


def main():
    # 1) sample ONCE and reuse
    item = DataGenerator.double_torus_item(
        R1=1.9, r1=0.6,
        R2=0.8, r2=0.2,
        samples=1000,
        visualize=False,
    )

    resolutions = list(range(6, 16))
    gains = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]

    clusterers = [
        ("dbscan", DBSCAN, {"eps": 0.4, "min_samples": 5}),
        ("hierarchical", AgglomerativeClustering, {"n_clusters": 2}),
        ("automato", Automato, {"random_state": 42}),
    ]

    for clusterer_name, clusterer_function, clusterer_params in clusterers:
        csv_path, png_path = run_grid_experiment(
            item=item,
            resolutions=resolutions,
            gains=gains,
            clusterer_name=clusterer_name,
            clusterer_function=clusterer_function,
            clusterer_params=clusterer_params,
            save_mapper=True,
        )
        print(f"[{clusterer_name}] CSV -> {csv_path}")
        print(f"[{clusterer_name}] PNG -> {png_path}")


if __name__ == "__main__":
    main()

```

---

## 🚀 Example usage with own shape

You can define your own mesh + Reeb graph and plug it into the same machinery.
```python
from src.DataGenerator import DataGenerator, _fmt_float
from src.Mapper import MapperParams, MapperSample
from sklearn.cluster import DBSCAN
from distance import sublevel_distance_combined
from src.distance_grid import DistanceGrid

import trimesh
from cereeberus import ReebGraph
import numpy as np


def double_torus_overlap(
    R1=2.0, r1=0.2,
    R2=1.0, r2=0.2,
    samples=1000,
    visualize=True,
):
    # --- geometry ---
    torus1 = trimesh.creation.torus(major_radius=R1, minor_radius=r1)
    torus2 = trimesh.creation.torus(major_radius=R2, minor_radius=r2)
    shift = np.array([0.0, 2.0, 0.0])
    torus2.apply_translation(shift)
    shape = trimesh.util.concatenate([torus1, torus2])

    # --- Reeb graph (height in x) ---
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

    f_x = lambda pts: pts[:, 0]  # x-height

    return DataGenerator.add_shape(
        id=7,
        name=(
            "3D_double_torus_yshift2_"
            f"R1{_fmt_float(float(R1))}_r1{_fmt_float(float(r1))}_"
            f"R2{_fmt_float(float(R2))}_r2{_fmt_float(float(r2))}_"
            f"S{samples}_x"
        ),
        shape=shape,
        rg=rg,
        mode="surface",
        f=f_x,
        samples=samples,
        visualize=visualize,
    )


if __name__ == "__main__":
    item = double_torus_overlap(
        R1=2.0, r1=0.2,
        R2=1.0, r2=0.2,
        samples=1000,
        visualize=False,   # no viewer in batch
    )

    resolutions = list(range(6, 16))
    gains = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]

    clusterer_name = "dbscan"
    clusterer_function = DBSCAN
    clusterer_params = {"eps": 0.4, "min_samples": 5}

    from mapper_generator import run_grid_experiment

    csv_path, png_path = run_grid_experiment(
        item=item,
        resolutions=resolutions,
        gains=gains,
        clusterer_name=clusterer_name,
        clusterer_function=clusterer_function,
        clusterer_params=clusterer_params,
        save_mapper=True,
    )

    print(f"Saved grid CSV -> {csv_path}")
    print(f"Saved heatmap  -> {png_path}")


```


## 🧱 Data structures

### `ShapeSample`
Container for one geometric dataset (mesh or path) and its filter function.

```python
class ShapeSample:
    id: int
    name: str
    shape: Geometry                    # trimesh.Trimesh or trimesh.path.Path2D/Path3D
    rg: ReebGraph
    height_function: Optional[Callable[[np.ndarray], np.ndarray]] = None
    samples: int = 500
    seed: Optional[int] = 2

    # Filled after sampling
    sampled_points: Optional[np.ndarray] = None
    sampled_f: Optional[np.ndarray] = None
    sampled_mode: Optional[str] = None
```

- Created via `DataGenerator.torus_item(...)`, `DataGenerator.annulus_item(...)`, or  
  `DataGenerator.add_shape(...)`.
- `mode` specifies how sampling is done:
  - **"surface"** → uniform over mesh faces (`Trimesh`)
  - **"length"** → length-weighted along polylines (`Path2D` / `Path3D`)

---

### `MapperParams`
Holds the parameters for the Mapper construction.

```python
class MapperParams:
    resolutions: int = 10        # number of cover intervals
    gains: float = 0.5           # fractional overlap (0,1)
    clusterer_name: str = None
    clusterer_function: Callable[..., Any] = DBSCAN  # e.g. DBSCAN, AgglomerativeClustering
    clusterer_params: Dict[str, Any] = field(
        default_factory=lambda: {"eps": 0.4, "min_samples": 5}
    )

```

---

### `MapperSample`
Encapsulates one Mapper run, given a `ShapeSample` and parameter set.

```python
class MapperSample:
    item: ShapeSample
    params: MapperParams
    visualize: bool = True
    save: bool = False

    # Outputs
    simplex_tree: Optional[gudhi.SimplexTree] = field(default=None, init=False)
    mapper_graph: Optional[nx.Graph] = field(default=None, repr=False)
    node_data: dict[int, dict] = field(default_factory=dict, repr=False)
```

When you call:
```python
mapper_sample.run()
```
it computes the Mapper graph (via GUDHI’s `MapperComplex`), assigns colors by filter values,
and optionally saves a `.png` inside:

```
mapper_results/<item.name>/<clusterer_name>
  res<r>_gain<g>_params<param>.png
```

---

## 📜 License

MIT License (feel free to reuse / modify).

# TDAMapper (Mapper experiments on synthetic shapes)

This toolkit generates synthetic shapes (or imports user-defined Shapely shapes), converts them into a unified `ShapeSample` representation, and evaluates the Mapper algorithm.

We systematically vary Mapper parameters (cover resolution, gains, clustering) and compute corresponding Mapper graphs. 
Each result is evaluated against the shape’s true ReebGraph using a dedicated distance function that quantifies topological distortion.

Distance values across the parameter space are gathered into a structured grid visualization (heatmaps), 
enabling rapid assessment of Mapper stability and parameter sensitivity.
All Mapper outputs and intermediate data products are stored as well.

---

```bash
git clone https://github.com/TobiasRobertEeksman/TDA-Mapper.git
cd TDA-Mapper
python -m venv .venv
. .venv/Scripts/activate   # or source .venv/bin/activate (mac/linux)
```

## 🧩 Requirements

- Python 3.10+
- `numpy`, `networkx`, `scikit-learn`, `trimesh`, `Pillow`, `matplotlib`, `shapely`, `gudhi`
- `cereeberus` for ReebGraph utilities (newest version (currently not via pip) )

Install dependencies inside your virtual environment:
```bash
python -m pip install -r requirements.txt
```

---

## 📁 Project structure

```
TDAMapper/
  main.py                      # personal / custom experiments (multiple clusterers, etc.)
  playground.py                # Create your own shapes with shapely or take predefined from datasets.py
  distance.py                  # distance functions between Mapper and ReebGraph
  src/
      __init__.py
      run_experiment.py         # main loop to generate the different heatmaps
      helper.py                 # helper functions for folder naming
      
      shapes/
        __init__.py
        base.py                  # ShapeSample class, which is used for the whole pipeline
        meshes.py                #  basic trimesh meshes (2d/3d)
        reeb_graphs.py           # hand defined Reeb graphs for the basic meshes
        sampling.py              # sampling utilities
        datasets.py              # DataGenerator: build ShapeSample datasets
        shapely_shapes.py        # Shapely-based 2D basic shapes for the playground
        converters.py            # convert Shapely → ShapeSample
  
      mapper/
        __init__.py
        core.py                  # Mapper classes & algorithm
        distance_grid.py         # distance grids over parameter space

  README.md
  requirements.txt

  mapper_results/              # outputs per shape (created at runtime)
    <shape-name>/
      <clusterer-name>/
        res*_gain*_*.png
    distance_grid/             # heatmap(s)
```

Run scripts as modules from the repo root:
```bash
# Your own multi-clusterer experiment (see main.py)
python main.py
```

## Example usage with own shape

You can define your own shapes in playground.py. Here are some example shapes. It is very important that you create an item through convert, and save this item in the item_list.
You can also use some predefined shapes with varying parameters. Those can be found in datasets.py DataGenerator.
```python
import numpy as np

from shapely.geometry import Point
from shapely.ops import unary_union
from shapely.affinity import translate, rotate
from shapely.geometry import JOIN_STYLE
from shapely.geometry import Polygon, MultiPolygon

from src.shapes.shapely_shapes import annulus, triangle, square
from src.shapes.converter import convert
from src.shapes.datasets import DataGenerator

# 1) “Snowman” — two touching annuli
m1 = unary_union([
    annulus((0.0, +0.9), 0.9, 0.25, res),
    annulus((0.0, -0.2), 0.7, 0.22, res)
])
m1_item = convert("snowman", m1, samples=1000, visualize=False)


# 2) “Tri-ring” — triangle ring + centered annulus
m2 = unary_union([
    triangle(center=(0,0), width=0.18, side=2.6, rotation_deg=0),
    annulus((0,0), 0.55, 0.18, res)
])
m2_item = convert("tri-ring", m2, samples=1000, visualize=False)

### 3D Objects from src.shapes.datasets DataGenerator ###
T1_item = DataGenerator.torus_item(R=2.0, r=1.0, samples=1000, seed=None, visualize=False)

item_list = [m1_item,
             m2_item,
             T1_item]

```
---

## Custom experiments (e.g. multiple clusterers) with `main.py`


`main.py` is intended for personal experiments where you might:

- reuse the same sampled item,
- compare multiple clusterers on the exact same data,
- or change resolution/gain ranges.

Example:

```python
# main.py (example use)

from sklearn.cluster import DBSCAN, AgglomerativeClustering
from automato import Automato

from src.run_experiment import run_experiment
from distance import betti_number_distance, sublevel_distance_dim, sublevel_distance_combined

from playground import item_list


def main():
    # 1) shapes defined in playground.py
    items = item_list[0:2]

    # 2) grid
    resolutions = list(range(6, 16))
    gains = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]

    # 3) Different clusterer configs
    clusterers = [
        ("dbscan", DBSCAN, {"eps": 0.4, "min_samples": 5}),
        ("hierarchical", AgglomerativeClustering, {"n_clusters": 2}),
        ("automato", Automato, {"random_state": 42}),
    ]

    for item in items:
        for clusterer_name, clusterer_function, clusterer_params in clusterers:
            csv_path, png_path = run_experiment(
                item=item,
                resolutions=resolutions,
                gains=gains,
                clusterer_name=clusterer_name,
                clusterer_function=clusterer_function,
                clusterer_params=clusterer_params,
                distance_function=sublevel_distance_combined,
                save_mapper=True,
            )
            print(f"[{clusterer_name}] CSV -> {csv_path}")
            print(f"[{clusterer_name}] PNG -> {png_path}")


if __name__ == "__main__":
    main()

```

---
## `run__experiment`: core helper

Most Mapper experiments in this repo go through a single helper function defined in `run_experiment.py`:

```python

from src.mapper.core import MapperParams, MapperSample
from distance import betti_number_distance, sublevel_distance_mappers, sublevel_distance_dim, sublevel_distance_combined
from src.mapper.distance_grid import DistanceGrid


def run_experiment(
    *,
    item,
    resolutions,
    gains,
    clusterer_name,
    clusterer_function,
    clusterer_params,
    distance_function = sublevel_distance_combined,
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
                visualize=False,
                save=save_mapper,
            )
            mapper_sample.run()

            d = distance_function(m=mapper_sample, rg=item.rg)
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

Import `run_experiment` in your own scripts (e.g. `main.py`), plug in different ShapeSamples (e.g from `playground.py`) and define parameters.

---


## 🧱 Data structures

### `ShapeSample`
Container for one geometric dataset (mesh or path) and its filter function.

```python
class ShapeSample:
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

# TDAMapper (Mapper experiments on synthetic shapes)

Tools to generate synthetic geometric datasets (meshes & paths), define filters, and run the Mapper algorithm.  
Results (graphs + PNGs) are saved per shape and can be collated into contact-sheet grids.

---

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
  mapper_generator.py          # scripts / examples
  src/
    DataGenerator.py
    Mapper.py
    Sampler.py
    ShapeClass.py
    Shapes.py
    rg_shapes.py
    visualize_rg.py
  mapper_results/              # outputs per shape
    <shape-name>/
      res*_gain*_eps*_min*.png
    img_grids/                 # contact sheets
```

Run scripts as modules from the repo root:
```bash
python -m src.mapper_generator
```

---

## 🚀 Example usage

```python
from src.DataGenerator import DataGenerator
from src.Mapper import MapperParams, MapperSample

if __name__ == "__main__":
    # Generate a torus dataset
    item = DataGenerator.torus_item(R=2.0, r=0.6, samples=1000, visualize=True)

    resolutions = list(range(7, 12))
    gains = {0.2, 0.3, 0.4, 0.5, 0.6}

    # Sweep Mapper parameters
    for res in resolutions:
        for g in gains:
            mapper_params = MapperParams(resolutions=res, gains=g)
            mapper_sample = MapperSample(item=item, params=mapper_params, visualize=False, save=True)
            G = mapper_sample.run()
            print(f"Mapper graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
```

This will generate Mapper graphs for multiple parameter combinations and save the resulting
visualizations in `mapper_results/<shape-name>/`.

---

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
    clusterer: str = "dbscan"    # clustering method
    eps: float = 0.5             # DBSCAN epsilon
    min_samples: int = 5         # DBSCAN min_samples
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
    mapper_graph: Optional[nx.Graph] = None
    node_data: dict[int, dict] = field(default_factory=dict)
```

When you call:
```python
G = mapper_sample.run()
```
it computes the Mapper graph (via GUDHI’s `MapperComplex`), assigns colors by filter values,
and optionally saves a `.png` and `.gml` inside:

```
mapper_results/<item.name>/
  res<r>_gain<g>_eps<e>_min<m>.png
```

---

## 📜 License

MIT License (feel free to reuse / modify).

from src.DataGenerator import DataGenerator
from src.Mapper import MapperParams, MapperSample
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from distance import betti_number_distance, sublevel_distance_mappers, sublevel_distance_dim, sublevel_distance_combined
from src.distance_grid import DistanceGrid

''' ## Example of creating a new shape
import trimesh
from cereeberus import ReebGraph
import numpy as np
from src.DataGenerator import _fmt_float, subdivide

def double_torus_overlap(R1 = 2.0, r1 = 0.2, R2 = 1.0, r2 = 0.2, samples = 1000, visualize = True):

    #trimesh shape
    torus1 = trimesh.creation.torus(major_radius=R1, minor_radius=r1)
    torus2 = trimesh.creation.torus(major_radius=R2, minor_radius=r2)
    shift = np.array([0.0, 2.0, 0.0])
    torus2.apply_translation(shift)
    shape = trimesh.util.concatenate([torus1, torus2])
    
    #Reeb Graph
    rg = ReebGraph()
    rg.add_node(0, f_vertex=-2.0)
    rg.add_node(1, f_vertex=-1.0)
    rg.add_node(2, f_vertex=-1.0)
    rg.add_node(3, f_vertex=1.0)
    rg.add_node(4, f_vertex=1.0)
    rg.add_node(5, f_vertex=2.0)

    rg.add_edge(0, 1)
    rg.add_edge(0, 2)
    rg.add_edge(1, 3)
    rg.add_edge(1, 3)
    rg.add_edge(2, 4)
    rg.add_edge(2, 4)
    rg.add_edge(3, 5)
    rg.add_edge(4, 5)

    rg = subdivide(rg)
    #height function
    f_x = lambda pts: pts[:, 0]  # x-height

    return DataGenerator.add_shape(
        id = 6,
        name = f"3D_double_torus_yshift2_R1{_fmt_float(float(R1))}_r1{_fmt_float(float(r1))}_R2{_fmt_float(float(R2))}_r2{_fmt_float(float(r2))}_S{samples}_x",
        shape=shape,
        rg=rg,
        mode="surface",
        f=f_x,
        samples=samples,
        visualize=visualize,
    )
'''

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
                visualize=False,
                save=save_mapper,
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


if __name__ == "__main__":
    """
    Simple example script so others see how to use this:

    - build one item
    - choose clusterer
    - run the grid
    """
    item = DataGenerator.double_torus_item(
        R1=1.9, r1=0.6, R2=0.8, r2=0.2,
        samples=1000,
        visualize=False,
    )

    resolutions = list(range(6, 16))
    gains = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]

    clusterer_name = "dbscan"
    clusterer_function = DBSCAN
    clusterer_params = {"eps": 0.4, "min_samples": 5}

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

    

from src.DataGenerator import DataGenerator
from src.Mapper import MapperParams, MapperSample
from distance import betti_number_distance, sublevel_distance_mappers, sublevel_distance_to_rg
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

if __name__ == "__main__":

    item = DataGenerator.torus_item(R=1.0, r=0.8, samples=1000, visualize=True)    
    resolutions = list(range(10,11)) 
    gains = [0.2, 0.3, 0.4, 0.5]

    grid = DistanceGrid()

    for res in resolutions:
        for g in gains:
            mapper_params = MapperParams(resolutions=res, gains=g)
            mapper_sample = MapperSample(item=item, params=mapper_params, visualize=False, save = True)
            G = mapper_sample.run()
            d = sublevel_distance_to_rg(m=mapper_sample, rg=item.rg, dim = 1)
            grid.add(resolution=res, gain=g, distance=d)
    
    csv_path, png_path = grid.save(item_name=item.name,
                                title="Sublevel distance to ReebGraph (H1)",
                                base_dir="mapper_results",
                                filename_stub="sublevel_distance")
    
    print(f"Saved grid CSV -> {csv_path}")
    print(f"Saved heatmap  -> {png_path}")

    

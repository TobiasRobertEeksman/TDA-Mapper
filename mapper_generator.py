from src.DataGenerator import DataGenerator
from src.Mapper import MapperParams, MapperSample

import trimesh
from cereeberus import ReebGraph
import numpy as np
from src.DataGenerator import _fmt_float
    
#example of generating a new shape
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
        id = 7,
        name = f"3D_box_l{_fmt_float(float(l))}_S{samples}_x",
        shape=box,
        rg=rg,
        mode="surface",
        f=f_x,
        samples=samples,
        visualize=visualize,
    )

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
        id = 8,
        name = f"3D_briecase_x{_fmt_float(float(x))}_y{_fmt_float(float(y))}_z{_fmt_float(float(z))}_S{samples}_y",
        shape=briefcase,
        rg=rg,
        mode="surface",
        f=f_y,
        samples=samples,
        visualize=visualize,
    )


if __name__ == "__main__":
    
    item = briefcase_item(samples = 1000)
    '''
    resolutions = list(range(9, 14)) 
    gains = {0.2, 0.3, 0.4, 0.5, 0.6}

    for res in resolutions:
        for g in gains:
            mapper_params = MapperParams(resolutions=res, gains=g)
            mapper_sample = MapperSample(item=item, params=mapper_params, visualize=False, save = False)
            G = mapper_sample.run()
            print(f"Mapper graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
            '''
    
    

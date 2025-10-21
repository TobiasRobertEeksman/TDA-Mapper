from DataGenerator import DataGenerator
from Mapper import MapperParams, MapperSample


if __name__ == "__main__":
    
    
    item = DataGenerator.annulus_item(R=2.0, r=0.5, samples=1000)

    mapper_params = MapperParams(resolutions=12, gains=0.4, eps=0.5, min_samples=4)
    mapper_sample = MapperSample(item=item, params=mapper_params)
    G = mapper_sample.run()

    print(f"Mapper graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

    '''
    #example of generating a new shape
    import trimesh
    from cereeberus import ReebGraph
    import numpy as np

    #trimesh shape
    circ1 = trimesh.path.creation.circle(radius=2.0)
    circ2 = trimesh.path.creation.circle(radius=1.0)
    shift = np.array([0.0, 2.0])
    circ2.apply_translation(shift)
    shape = trimesh.util.concatenate([circ1, circ2])
    
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

    new_item = DataGenerator.add_shape(
        id = 6,
        name = "double_circle_yshift_500_x",
        shape=shape,
        rg=rg,
        mode="length",
        f=f_x,
        samples=500,
        visualize=True,
    )

    #run mapper on new shape
    mapper_params = MapperParams(resolutions=10, gains=0.3, eps=0.5, min_samples=4)
    mapper_sample = MapperSample(item=new_item, params=mapper_params)
    G = mapper_sample.run()
    '''

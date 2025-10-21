import numpy as np
import networkx as nx
from cereeberus import ReebGraph
from DataGenerator import DataGenerator
from Mapper import MapperParams, MapperSample
from typing import Any, Dict
import matplotlib.pyplot as plt
from vedo import Mesh, show
from visualize_rg import draw_reeb_graph


if __name__ == "__main__": 
    item = DataGenerator.torus_item(R=2.0, r=0.5, samples=500)

    mesh = item.shape
    vm = Mesh(mesh)
    vm.pointdata["height"] = vm.coordinates[:, 1]  # any scalar field works
    reeb_graph = vm.to_reeb_graph(field_id=2)  
    show([[vm.cmap('viridis').add_scalarbar('height')], [reeb_graph]], N=2, sharecam=False)

    rg = item.rg
    draw_reeb_graph(rg)
    
    '''
    mapper_params = MapperParams(resolutions=12, gains=0.4, eps=0.4, min_samples=4)
    mapper_sample = MapperSample(item=item, params=mapper_params, visualize=True)
    G_mapper = mapper_sample.run()
    '''






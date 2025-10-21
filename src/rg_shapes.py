import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

# The main class for the Reeb Graph
from cereeberus import ReebGraph

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


class Shapes_rg():

    @staticmethod
    def circle_rg(radius=1.0):
        rg = ReebGraph()
        rg.add_node(0, f_vertex=-radius)
        rg.add_node(1, f_vertex=radius)
        rg.add_edge(0, 1)
        rg.add_edge(0, 1)
        return rg

    @staticmethod
    def double_circle_rg(r1=1.0, r2=3.0):
        rg = ReebGraph()
        rg.add_node(0, f_vertex=-2*r1)
        rg.add_node(1, f_vertex=0.0)
        rg.add_node(2, f_vertex=2*r2)
        rg.add_edge(0, 1)
        rg.add_edge(0, 1)
        rg.add_edge(1, 2)
        rg.add_edge(1, 2)
        return rg

    @staticmethod
    def sphere_rg(radius=1.0):
        rg = ReebGraph()
        rg.add_node(0, f_vertex=-radius)
        rg.add_node(1, f_vertex=radius)
        rg.add_edge(0, 1)
        return rg

    @staticmethod
    def torus_rg(R=2.0, r=1.0):
        rg = ReebGraph()
        rg.add_node(0, f_vertex=-R)
        rg.add_node(1, f_vertex=-r)
        rg.add_node(2, f_vertex=r)
        rg.add_node(3, f_vertex=R)

        rg.add_edge(0, 1)
        rg.add_edge(1, 2)
        rg.add_edge(1, 2)
        rg.add_edge(2, 3)
        return rg

    @staticmethod
    def double_torus_rg(R1=2.0, r1=1.0, R2=1.6, r2=1.0, shift = 3.6):
        rg = ReebGraph()
        rg.add_node(0, f_vertex=-R1)
        rg.add_node(1, f_vertex=-r1)
        rg.add_node(2, f_vertex=r1)
        rg.add_node(3, f_vertex=shift - R2 + r2)
        rg.add_node(4, f_vertex=shift + r1)
        rg.add_node(5, f_vertex=shift + R2)

        rg.add_edge(0, 1)
        rg.add_edge(1, 2)
        rg.add_edge(1, 2)
        rg.add_edge(2, 3)
        rg.add_edge(3, 4)
        rg.add_edge(3, 4)
        rg.add_edge(4, 5)
        return rg


''' Built-in visualizer for ReebGraph, new one in visualize_rg.py
if __name__ == "__main__":
    R = double_torus_rg()
    R.set_pos_from_f(seed = 1)
    R.draw(cpx = 3, with_labels=True)
    plt.tight_layout()
    plt.show()              # <-- required in a script
    # Or save instead of showing:
    # plt.savefig("sphere_reeb.png", dpi=200)
'''
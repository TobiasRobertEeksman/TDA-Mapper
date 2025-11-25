import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

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


def subdivide(rg):
    G: nx.MultiDiGraph = rg         # ReebGraph is a MultiDiGraph
    f = dict(rg.f)                  # node -> value
    orig = list(G.edges(keys=True)) # snapshot original multiedges

    # fresh integer labels if your nodes are ints
    next_id = (max(G.nodes) + 1) if G.number_of_nodes() else 0
    def fresh():
        nonlocal next_id
        w = next_id; next_id += 1; return w

    for u, v, _k in orig:
        fu, fv = float(f[u]), float(f[v])
        if fu == fv:
            # ceREEBerus collapses equal-height edges; skip or handle separately
            continue
        if fu > fv:
            u, v = v, u
            fu, fv = fv, fu

        if not rg.has_edge(u, v):
            continue

        w = fresh()
        f_w = 0.5 * (fu + fv)
        rg.subdivide_edge(u, v, w, f_w)
        f[w] = f_w
        
    return rg

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
        rg.add_node(0, f_vertex=-(R+r))
        rg.add_node(1, f_vertex=-(R-r))
        rg.add_node(2, f_vertex=R-r)
        rg.add_node(3, f_vertex=R+r)

        rg.add_edge(0, 1)
        rg.add_edge(1, 2)
        rg.add_edge(1, 2)
        rg.add_edge(2, 3)
        return rg

    @staticmethod
    def double_torus_rg(R1=2.0, r1=1.0, R2=1.6, r2=1.0, shift=3.6):
        outer1 = R1 + r1
        inner1 = R1 - r1
        outer2 = R2 + r2
        inner2 = R2 - r2
        s = shift

        rg = ReebGraph()
        rg.add_node(0, f_vertex=-outer1)
        rg.add_node(1, f_vertex=-inner1)
        rg.add_node(2, f_vertex= inner1)


        left_inner2 = s - inner2

        # Case 1: full hole visible – no overlap with outer1
        if left_inner2 >= outer1:
            f3 = left_inner2
        else:
            # Case 2: cut case – inner2 intersects outer1
            o = outer1
            j = inner2
            x_cut = (o**2 + s**2 - j**2) / (2.0 * s)
            f3 = x_cut


        rg.add_node(3, f_vertex=f3)
        rg.add_node(4, f_vertex=s + inner2)
        rg.add_node(5, f_vertex=s + outer2)

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
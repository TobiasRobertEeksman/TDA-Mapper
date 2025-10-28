from src.Mapper import MapperSample
import gudhi
import numpy as np
import matplotlib.pyplot as plt
from cereeberus import ReebGraph
import math
from gudhi import bottleneck_distance


def betti_number_distance(m1: MapperSample, m2: MapperSample, save = False, visualize=False):

    if m1.simplex_tree is None or m2.simplex_tree is None:
        raise ValueError("Both MapperSample.simplex_tree must be set (run your pipeline first).")
    
    st1 = m1.simplex_tree
    st2 = m2.simplex_tree

    p1 = st1.compute_persistence(persistence_dim_max = True)
    p2 = st2.compute_persistence(persistence_dim_max = True)

    if save:
        st1.write_persistence_diagram("s1_diagram")
        st2.write_persistence_diagram("s2_diagram")
    
    if visualize:
    # Option A: plot full diagram (including infinities) and save
        ax = gudhi.plot_persistence_diagram(persistence=p1)
        plt.savefig("s1_diagram.png", dpi=150, bbox_inches="tight")
        plt.clf()

        ax = gudhi.plot_persistence_diagram(persistence=p2)
        plt.savefig("s2_diagram.png", dpi=150, bbox_inches="tight")
        plt.clf()
        
    b1 = st1.betti_numbers()
    b2 = st2.betti_numbers()
    print("Betti_1:", b1)
    print("Betti_2:", b2)

    # pad to same length
    L = max(len(b1), len(b2))
    b1 = np.pad(b1, (0, L - len(b1)))
    b2 = np.pad(b2, (0, L - len(b2)))

    diff = b1 - b2
    mse = float(np.mean(diff.astype(float)**2))

    return mse


def st_from_graph_lower_star(G, f):
    st = gudhi.SimplexTree()
    # vertices
    for v, val in f.items():
        st.insert([v], filtration=float(val))
    # undirected edges (collapse direction & multiplicity)
    seen = set()
    for u, v in G.edges():
        if u == v:
            continue
        e = (u, v) if u < v else (v, u)
        if e in seen:
            continue
        seen.add(e)
        st.insert([u, v], filtration=float(max(f[u], f[v])))
    st.make_filtration_non_decreasing()
    return st

def node_avgs_from_graph(G, color_attr="attr_name", color_index=0):
    return {int(n): float(d[color_attr][color_index]) for n, d in G.nodes(data=True)}

def sublevel_distance_mappers(m1: MapperSample, m2: MapperSample, dim: int = 1) -> float:
    if m1.simplex_tree is None or m2.simplex_tree is None:
        raise ValueError("Run fit() first.")

    # 1) read mean f per node
    av1 = node_avgs_from_graph(m1.mapper_graph)  # uses "attr_name" by default
    av2 = node_avgs_from_graph(m2.mapper_graph)

    # 2) build graph-only STs
    st1 = st_from_graph_lower_star(m1.mapper_graph, av1)
    st2 = st_from_graph_lower_star(m2.mapper_graph, av2)

    # 3) compute persistence
    st1.compute_persistence(persistence_dim_max = True)
    print("betti_shape1:", st1.betti_numbers())   # should be [1, 1]

    st2.compute_persistence(persistence_dim_max = True)
    print("betti_shape2:", st1.betti_numbers())

    # 4) H1 diagrams and bottleneck
    I1 = st1.persistence_intervals_in_dimension(dim)
    I2 = st2.persistence_intervals_in_dimension(dim)
    print("I1:", I1)
    print("I2:", I2)

    dist = bottleneck_distance(I1, I2)
    print("bottleneck H1:", dist)
    return dist

def node_values_from_rg(rg):
    return {n: float(val) for n, val in rg.f.items()}


def sublevel_distance_to_rg(m: MapperSample,
                            rg: ReebGraph,
                            agg: str = "max",
                            w0: float = 1.0,
                            w1: float = 1.0) -> float:

    if m.simplex_tree is None or rg is None:
        raise ValueError("Run fit() first.")

    # 1) read mean f per node
    av = node_avgs_from_graph(m.mapper_graph)  # uses "attr_name" by default
    val_rg = node_values_from_rg(rg)

    # 2) define SimplexTrees with edge values
    st = st_from_graph_lower_star(m.mapper_graph, av)
    st_rg = st_from_graph_lower_star(rg, val_rg)

    for simplex, filt in st_rg.get_filtration():
        print(simplex, float(filt))
    
    #3) compute persistence
    st.compute_persistence(persistence_dim_max = True)
    st_rg.compute_persistence(persistence_dim_max = True)

    print("betti_shape_mapper:", st.betti_numbers())
    print("betti_shape_rg:", st_rg.betti_numbers())

    D0_A = st.persistence_intervals_in_dimension(0)
    D0_B = st_rg.persistence_intervals_in_dimension(0)
    D1_A = st.persistence_intervals_in_dimension(1)
    D1_B = st_rg.persistence_intervals_in_dimension(1)

    d0 = bottleneck_distance(D0_A, D0_B) if (D0_A or D0_B) else 0.0
    d1 = bottleneck_distance(D1_A, D1_B) if (D1_A or D1_B) else 0.0

    print("Bottleneck H0:", d0) 
    print("Bottleneck H1:", d1)   

    if agg == "l2":
        return math.sqrt(w0 * d0 * d0 + w1 * d1 * d1)

    return max(d0,d1)


    


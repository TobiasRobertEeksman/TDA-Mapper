import cereeberus
from DataGenerator import DataGenerator
import networkx as nx
from Mapper import MapperParams, MapperSample
from visualize_rg import draw_reeb_graph, _extract_nx_and_f
import numpy as np
import matplotlib.pyplot as plt
from cereeberus import ReebGraph

def gudhi_mapper_to_rg_input(G: nx.Graph) -> tuple[nx.MultiDiGraph, dict]:
    """
    Turn GUDHI's to_networkx() graph into (MultiDiGraph, f_map) for draw_reeb_graph.
    Assumes you called get_networkx(set_attributes_from_colors=True)
    so each node has 'attr_name' = a float or a length-1 list/array.
    """
    attrs = nx.get_node_attributes(G, "attr_name")
    if not attrs:
        raise ValueError("Expected node attribute 'attr_name'. Call get_networkx(set_attributes_from_colors=True).")

    def as_float(x):
        if isinstance(x, (list, tuple, np.ndarray)):
            if len(x) == 0:
                raise ValueError("Empty 'attr_name' on a node.")
            return float(x[0])
        return float(x)

    f_map = {n: as_float(v) for n, v in attrs.items()}
    Gm = nx.MultiDiGraph(G)  # upcast; preserves nodes/edges/attributes
    return Gm, f_map

if __name__ == "__main__":
    item = DataGenerator.annulus_item(R=2.0, r=1.2, samples=500, visualize=True)

    # item.rg.draw(cpx = 2.0)
    # item.visualize_rg()

    Grg, fvals = _extract_nx_and_f(item.rg)
    pos = nx.kamada_kawai_layout(Grg)  # deterministic
    nx.draw(Grg, pos, with_labels=True)  # to visualize

    mapper_params = MapperParams(resolutions=12, gains=0.5, eps=0.6, min_samples=4)
    mapper_sample = MapperSample(item=item, params=mapper_params, visualize=False)
    G = mapper_sample.run()
    rg_input = gudhi_mapper_to_rg_input(G)
    draw_reeb_graph(rg_input, title="GUDHI Mapper graph")
    plt.show()

    posmapper = nx.kamada_kawai_layout(G)  # deterministic
    nx.draw(G, posmapper, with_labels=True)  # to visualize





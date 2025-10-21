from collections import defaultdict
from typing import Optional, Dict, Any, Union

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import numpy as np
import networkx as nx
from cereeberus import ReebGraph

# -------- helpers -------------------------------------------------------------

def _extract_nx_and_f(rg) -> tuple[nx.MultiDiGraph, Dict[Any, float]]:
    """
    Make this robust to ceREEBerus' internal storage:
    - If rg is already a MultiDiGraph -> use it directly
    - Else look for an underlying graph in attributes like .G, .graph, ._graph, ...
    - Pull f-values primarily from node attribute 'f_vertex';
      otherwise try rg.f / rg.f_vertex dicts.
    """
    # locate the underlying nx graph
    if isinstance(rg, nx.MultiDiGraph):
        G = rg
    else:
        G = None
        for attr in ("G", "graph", "_G", "_graph", "nx_graph"):
            if hasattr(rg, attr):
                candidate = getattr(rg, attr)
                if isinstance(candidate, nx.MultiDiGraph):
                    G = candidate
                    break
        if G is None:
            raise TypeError(
                "Could not find a networkx.MultiDiGraph inside the ReebGraph. "
                "Expected rg to be a MultiDiGraph or to expose one via .G / .graph / ._graph."
            )

    # pull f-values
    fvals: Dict[Any, float] = {}
    for n, data in G.nodes(data=True):
        if "f_vertex" in data:
            fvals[n] = float(data["f_vertex"])
        elif "f" in data:
            fvals[n] = float(data["f"])

    if len(fvals) != G.number_of_nodes():
        # try dict attributes on the wrapper
        for attr in ("f", "f_vertex", "function_values"):
            if hasattr(rg, attr):
                d = getattr(rg, attr)
                if isinstance(d, dict):
                    for n in G.nodes:
                        if n in d and n not in fvals:
                            fvals[n] = float(d[n])

    if len(fvals) != G.number_of_nodes():
        missing = [n for n in G.nodes if n not in fvals]
        raise ValueError(
            "Missing f-values for some nodes. "
            f"Provide them as a node attribute 'f_vertex' or in a dict like rg.f. "
            f"Missing: {missing}"
        )

    return G, fvals


def _parallel_radii(m: int, base: float = 0.15) -> list[float]:
    """
    Radii for m parallel arcs: [0, +b, -b, +2b, -2b, ...]
    Ensures a straight edge if m is odd.
    """
    rads = []
    if m % 2 == 1:
        rads.append(0.0)
    k = 1
    while len(rads) < m:
        rads.extend([+k * base, -k * base])
        k += 1
    return rads[:m]

# -------- main visualizer -----------------------------------------------------

def draw_reeb_graph(
    rg: Union[ReebGraph, tuple[nx.Graph, Dict[Any, float]]],                         # from cereeberus import ReebGraph  (or your own wrapper)
    ax: Optional[plt.Axes] = None,
    *,
    node_size: float = 220,
    node_fc: str = "white",
    node_ec: str = "black",
    edge_color: str = "black",
    edge_alpha: float = 0.95,
    arrow: bool = False,
    annotate_f: bool = True,
    label_nodes: bool = False,
    y_separation: float = 0.2,     # vertical spread for nodes sharing the same f
    arc_base: float = 0.3,         # base curvature for parallel edges
    title: Optional[str] = "Reeb Graph",
):
    """
    Draw a ceREEBerus ReebGraph on a horizontal f-axis.
    - nodes at x=f
    - multi-edges as curved arcs (alternating up/down)
    - tiny vertical jitter if multiple nodes share the same f

    Returns the Matplotlib Axes (no plt.show() so you can compose/save).
    """
    if isinstance(rg, tuple) and len(rg) == 2:
        G_in, fvals = rg
        G = G_in if isinstance(G_in, nx.MultiDiGraph) else nx.MultiDiGraph(G_in)
    else:
        G, fvals = _extract_nx_and_f(rg)  # your existing path for ReebGraph

    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 3))

    # group nodes by identical f to avoid overlap; assign small vertical offsets
    bins = defaultdict(list)
    for n, f in fvals.items():
        bins[float(f)].append(n)

    pos: Dict[Any, tuple[float, float]] = {}
    for f, nodes_at_f in bins.items():
        k = len(nodes_at_f)
        if k == 1:
            pos[nodes_at_f[0]] = (f, 0.0)
        else:
            ys = np.linspace(-y_separation, y_separation, k)
            for n, y in zip(nodes_at_f, ys):
                pos[n] = (f, float(y))

    # axis limits and baseline
    xs = list(fvals.values())
    xmin, xmax = min(xs), max(xs)
    dx = (xmax - xmin) or 1.0
    pad = 0.15 * dx
    ax.axhline(0, linewidth=1.0, color="0.85", zorder=0)
    ax.set_xlim(xmin - pad, xmax + pad)

    # ticks at the unique f-values (optional)
    if annotate_f:
        uniq = sorted(bins.keys())
        ax.set_xticks(uniq)
        ax.set_xticklabels([f"{u:g}" for u in uniq], rotation=0)
    else:
        ax.set_xticks([])

    # draw nodes
    for n, (x, y) in pos.items():
        ax.scatter([x], [y], s=node_size, facecolor=node_fc, edgecolor=node_ec, zorder=3)

    if label_nodes:
        for n, (x, y) in pos.items():
            ax.text(x, y + 0.03, str(n), ha="center", va="bottom", fontsize=9, zorder=4)

    # draw edges with multiplicity-aware arcs
    pair_to_keys = defaultdict(list)
    for u, v, key in G.edges(keys=True):
        pair_to_keys[(u, v)].append(key)

    for (u, v), keys in pair_to_keys.items():
        start = pos[u]
        end = pos[v]
        rads = _parallel_radii(len(keys), base=arc_base)
        for rad, key in zip(rads, sorted(keys)):
            style = f"arc3,rad={rad}"
            patch = FancyArrowPatch(
                start, end,
                connectionstyle=style,
                arrowstyle="-|>" if arrow else "-",
                mutation_scale=12.0,
                lw=1.6,
                color=edge_color,
                alpha=edge_alpha,
                zorder=2,
                shrinkA=5.0, shrinkB=5.0,
            )
            ax.add_patch(patch)

    # cosmetics
    ax.set_ylim(-0.5, 0.5)
    ax.set_yticks([])
    ax.set_xlabel("f")
    if title:
        ax.set_title(title)
    plt.tight_layout()
    return ax


if __name__ == "__main__":
    from rg_shapes import circle_rg

    R = circle_rg()
    draw_reeb_graph(R)
    plt.show()
from src.rg_shapes import Shapes_rg
import networkx as nx
import matplotlib.pylab as plt


def subdivide(rg):
    """
    Subdivide each *original* multiedge (u,v) exactly once into u–w–v
    with f(w) = (f(u)+f(v))/2. Skips equal-value edges (fu==fv) to avoid collapse.
    """
    G: nx.MultiDiGraph = rg                  # ReebGraph subclasses MultiDiGraph
    f = dict(rg.f)                           # node -> function value
    orig = list(G.edges(keys=True))          # snapshot so we don't revisit new edges

    # fresh labels: keep integer scheme if your nodes are ints
    if all(isinstance(n, int) for n in G.nodes):
        next_id = (max(G.nodes) + 1) if G.number_of_nodes() else 0
        def fresh():
            nonlocal next_id
            w = next_id; next_id += 1; return w
    else:
        c = 0
        def fresh():
            nonlocal c
            w = f"_aux_{c}"; c += 1; return w

    for u, v, _k in orig:
        fu, fv = float(f[u]), float(f[v])
        # orient from low to high for clarity
        if fu > fv:
            u, v = v, u
            fu, fv = fv, fu

        if fu == fv:
            # equal-height edges collapse in ceREEBerus; skip (or handle separately)
            continue

        if not rg.has_edge(u, v):
            continue  # something else mutated; be safe

        f_w = 0.5 * (fu + fv)
        w = fresh()
        rg.subdivide_edge(u, v, w, f_w)
        f[w] = f_w

    return rg



rg = Shapes_rg.sphere_rg()


print("old")
print(rg.nodes())
print(rg.edges())


print("new")
rg_ext = subdivide(rg)
print(rg_ext.nodes())
print(rg_ext.edges())

rg_ext.draw()
plt.show()



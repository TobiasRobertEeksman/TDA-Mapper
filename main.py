from sklearn.cluster import DBSCAN, AgglomerativeClustering
from automato import Automato

from src.run_experiment import run_experiment
from distance import betti_number_distance, sublevel_distance_dim, sublevel_distance_combined

from playground import item_list


def main():
    # 1) shapes defined in playground.py
    items = item_list[0:2]

    # 2) grid
    resolutions = list(range(6, 16))
    gains = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]

    # 3) Different clusterer configs
    clusterers = [
        ("dbscan", DBSCAN, {"eps": 0.4, "min_samples": 5}),
        ("hierarchical", AgglomerativeClustering, {"n_clusters": 2}),
        ("automato", Automato, {"random_state": 42}),
    ]

    for item in items:
        for clusterer_name, clusterer_function, clusterer_params in clusterers:
            csv_path, png_path = run_experiment(
                item=item,
                resolutions=resolutions,
                gains=gains,
                clusterer_name=clusterer_name,
                clusterer_function=clusterer_function,
                clusterer_params=clusterer_params,
                distance_function=sublevel_distance_combined,
                save_mapper=True,
            )
            print(f"[{clusterer_name}] CSV -> {csv_path}")
            print(f"[{clusterer_name}] PNG -> {png_path}")


if __name__ == "__main__":
    main()

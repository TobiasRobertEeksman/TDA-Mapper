from sklearn.cluster import DBSCAN, AgglomerativeClustering
from automato import Automato

from src.run_experiment import run_experiment
from distance import betti_number_distance, sublevel_distance_dim, sublevel_distance_combined

from playground import item_list


def main():
    # 1) shapes defined in playground.py
    items = item_list

    # 2) grid
    resolutions = list(range(5, 15))
    gains = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]

    # 3) Different clusterer configs
    clusterers = [
        ("dbscan-tight", DBSCAN, {"eps": 0.40, "min_samples": 4}),
        ("dbscan-medium", DBSCAN, {"eps": 0.55, "min_samples": 4}),
        ("dbscan-loose", DBSCAN, {"eps": 0.70, "min_samples": 5}),
        ("dbscan-dense", DBSCAN, {"eps": 0.70, "min_samples": 8}),
        ("hierarchical", AgglomerativeClustering, {"n_clusters": 2}),
        ("hierarchical-thresh", AgglomerativeClustering,
            {"n_clusters": None, "distance_threshold": 0.5, "linkage": "single"}),
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

from src.mapper.core import MapperParams, MapperSample
from distance import betti_number_distance, sublevel_distance_mappers, sublevel_distance_dim, sublevel_distance_combined
from src.mapper.distance_grid import DistanceGrid


def run_experiment(
    *,
    item,
    resolutions,
    gains,
    clusterer_name,
    clusterer_function,
    clusterer_params,
    distance_function = sublevel_distance_combined,
    save_mapper: bool = True,
) -> tuple[str, str]:
    """
    Run the (res, gain) grid search for a fixed item + clusterer.
    Returns paths to the CSV and PNG.
    """
    grid = DistanceGrid()

    for res in resolutions:
        for g in gains:
            mapper_params = MapperParams(
                resolutions=res,
                gains=g,
                clusterer_name=clusterer_name,
                clusterer_function=clusterer_function,
                clusterer_params=clusterer_params,
            )

            mapper_sample = MapperSample(
                item=item,
                params=mapper_params,
                visualize=False,
                save=save_mapper,
            )
            mapper_sample.run()

            d = distance_function(m=mapper_sample, rg=item.rg)
            grid.add(resolution=res, gain=g, distance=d)

    csv_path, png_path = grid.save(
        item_name=item.name,
        title=(
            f"Combined Sublevel distance to ReebGraph with clusterer: "
            f"{clusterer_name} and {clusterer_params}"
        ),
        base_dir="mapper_results",
        filename_stub="sublevel_distance",
        clusterer_name=clusterer_name,
    )
    return csv_path, png_path
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from src.helper import _slug

class DistanceGrid:
    def __init__(self):
        self._vals = {}          # (res, gain) -> float
        self._res_order = []     # preserve insertion order
        self._gain_order = []

    def add(self, resolution: int, gain: float, distance: float):
        if resolution not in self._res_order:
            self._res_order.append(resolution)
        if gain not in self._gain_order:
            self._gain_order.append(gain)
        self._vals[(resolution, gain)] = float(distance)

    def to_dataframe(self) -> pd.DataFrame:
        # rows: resolutions in insertion order; columns: gains in insertion order
        rows = self._res_order
        cols = self._gain_order
        data = [[self._vals.get((r, g), float("nan")) for g in cols] for r in rows]
        df = pd.DataFrame(data, index=rows, columns=cols)
        df.index.name = "resolution"
        df.columns.name = "gain"
        return df

    def save(self, item_name: str, clusterer_name: str, clusterer_params: str, title: str = "Sublevel distance to ReebGraph Combined",
             base_dir: str = "mapper_results", filename_stub: str = "sublevel_distance") -> tuple[str, str]:
        df = self.to_dataframe()

        # Make folder: mapper_results/<slug(item_name)>/
        base = Path(base_dir) / (item_name)
        base.mkdir(parents=True, exist_ok=True)

        # Save CSV
        csv_path = base / f"{filename_stub}_{_slug(clusterer_name)}_{_slug(clusterer_params)}_grid.csv"
        df.to_csv(csv_path, float_format="%.6g")

        # Plot heatmap (finite values gradient, inf shown as gray with ∞)
        A = df.to_numpy(dtype=float)
        mask = ~np.isfinite(A)
        A_masked = np.ma.array(A, mask=mask)

        cmap = plt.cm.magma.copy()
        cmap.set_bad(color="#CCCCCC")  # for inf/NaN

        fig, ax = plt.subplots(figsize=(0.8*len(df.columns)+2.5, 0.8*len(df.index)+2.5))
        im = ax.imshow(A_masked, cmap=cmap, aspect="auto", origin="lower", interpolation="nearest")

        ax.set_xticks(np.arange(len(df.columns)), labels=[f"{g:.2f}" for g in df.columns])
        ax.set_yticks(np.arange(len(df.index)),   labels=[str(r) for r in df.index])

        ax.set_xlabel("gain")
        ax.set_ylabel("resolution")
        ax.set_title(title)

        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Bottleneck distance")

        # annotate numeric values on top of the heatmap
        fmt = "{:.3f}"
        norm = im.norm  # same normalization used by imshow

        def text_color_for(val):
            # perceived luminance of the cmap color to decide white/black text
            r, g, b, _ = cmap(norm(val))
            lum = 0.2126*r + 0.7152*g + 0.0722*b
            return "black" if lum > 0.6 else "white"

        for (i, j), val in np.ndenumerate(A):
            if np.isfinite(val):
                ax.text(
                    j, i, fmt.format(val),
                    ha="center", va="center",
                    color=text_color_for(val),
                    fontsize=10, fontweight="bold"
                )

        # overlay ∞ where masked (inf) – keep your original loop
        for (i, j), m in np.ndenumerate(mask):
            if m and np.isinf(A[i, j]):
                ax.text(j, i, "∞", ha="center", va="center",
                        color="black", fontsize=12, fontweight="bold")
            elif m and np.isnan(A[i, j]):
                ax.text(j, i, "–", ha="center", va="center",
                        color="black", fontsize=12)
        
        # highlight the global minimum (handles ties)
        finite = np.isfinite(A)
        if np.any(finite):
            vmin = A[finite].min()
            mins = np.argwhere((A == vmin) & finite)

            for i, j in mins:
                # draw a two-tone outline for contrast on any background
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1,
                                       fill=False, linewidth=3, edgecolor="white", zorder=3))
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1,
                                       fill=False, linewidth=1.2, edgecolor="black", zorder=4))

        fig.tight_layout()

        png_path = base / f"{filename_stub}_{_slug(clusterer_name)}_{_slug(clusterer_params)}_heatmap.png"
        fig.savefig(png_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

        return str(csv_path), str(png_path)

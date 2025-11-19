from pathlib import Path
from PIL import Image, ImageOps
import math

def make_mapper_panel(
    folder,
    top_three,                 # list of 3 filenames inside `folder`
    out_path=None,             # auto -> mapper_results/img_grids/<name>.png
    grid_cols=5,
    cell_top=(320, 240),
    cell_grid=(480, 360),
    pad=12,
    margin=24,
    bg=(255, 255, 255),
    also_pdf=False,
    scale=1.0,
):
    folder = Path(folder)

    # ==== build output path ====
    name = folder.name                   # e.g. "3D_torus_R2_r0p5_S1000_x"
    out_dir = folder.parent / "img_grids"
    out_dir.mkdir(parents=True, exist_ok=True)
    if out_path is None:
        out_path = out_dir / f"{name}.png"
    else:
        out_path = Path(out_path)
        if not out_path.is_absolute():
            out_path = out_dir / out_path  # drop relative names into img_grids/

    # ==== collect images ====
    top_three = [folder / Path(t) for t in top_three]
    for t in top_three:
        if not t.exists():
            raise FileNotFoundError(f"Top image not found: {t}")

    pool = sorted([p for p in folder.iterdir()
                   if p.suffix.lower() in {".png", ".jpg", ".jpeg"}])
    pool = [p for p in pool if p not in top_three]
    grid_n = min(25, len(pool))
    grid_paths = pool[:grid_n]

    # ==== scaling ====
    pad = int(round(pad * scale))
    margin = int(round(margin * scale))
    cell_top = (int(round(cell_top[0] * scale)), int(round(cell_top[1] * scale)))
    cell_grid = (int(round(cell_grid[0] * scale)), int(round(cell_grid[1] * scale)))

    # ==== layout ====
    top_cols = 3
    top_cell_w, top_cell_h = cell_top
    grid_cell_w, grid_cell_h = cell_grid
    W_top  = top_cols * top_cell_w + pad * (top_cols - 1)
    W_grid = grid_cols * grid_cell_w + pad * (grid_cols - 1)
    W = margin * 2 + max(W_top, W_grid)
    grid_rows = math.ceil(grid_n / grid_cols) if grid_n else 0
    H_top  = top_cell_h
    H_grid = grid_rows * grid_cell_h + pad * (grid_rows - 1) if grid_rows else 0
    H = margin * 2 + H_top + (pad if grid_rows else 0) + H_grid

    sheet = Image.new("RGB", (W, H), bg)

    def paste_center(pil_img, box_xy, cell_size):
        cw, ch = cell_size
        thumb = ImageOps.contain(pil_img, (cw, ch))
        cx, cy = box_xy
        off = (cx + (cw - thumb.width)//2, cy + (ch - thumb.height)//2)
        sheet.paste(thumb, off)

    # top row
    x0 = margin + (max(W_top, W_grid) - W_top)//2
    y0 = margin
    for i, p in enumerate(top_three):
        im = Image.open(p).convert("RGB")
        x = x0 + i * (top_cell_w + pad)
        paste_center(im, (x, y0), (top_cell_w, top_cell_h))

    # grid
    if grid_n:
        y_grid = margin + H_top + pad
        x_start = margin + (max(W_top, W_grid) - W_grid)//2
        for idx, p in enumerate(grid_paths):
            r, c = divmod(idx, grid_cols)
            x = x_start + c * (grid_cell_w + pad)
            y = y_grid + r * (grid_cell_h + pad)
            im = Image.open(p).convert("RGB")
            paste_center(im, (x, y), (grid_cell_w, grid_cell_h))

    # ==== save ====
    sheet.save(out_path)
    if also_pdf:
        sheet.save(out_path.with_suffix(".pdf"))
    return str(out_path)


if __name__ == "__main__":
    # ===== Example usage =====

    name = "3D_briecase_x2_y4_z1_S1000_y"
    folder = Path(r"C:\Users\Tobias\OneDrive - Schulen kvBL\Dokumente\ETH\TDAMapper\mapper_results") / name
    top_three = ["shape1.png", "shape2.png", "ReebGraph.png"]

    # out goes to: <mapper_results>\img_grids\<name>.png (and .pdf)
    make_mapper_panel(folder, top_three, also_pdf=False, scale=1.2)
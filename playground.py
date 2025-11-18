import numpy as np

from shapely.geometry import Point
from shapely.ops import unary_union
from shapely.affinity import translate, rotate
from shapely.geometry import JOIN_STYLE
from shapely.geometry import Polygon, MultiPolygon

from src.shapely_shapes import annulus, triangle, square
from src.converter import convert
from src.DataGenerator import DataGenerator


R = 1
width = 0.25
gap = 0.20
sep = 2*R + gap          # safe separation between centers

# 1) A single annulus
single = annulus((0.0, 0.0), R, width)

# 2) Two annuluses next to each other (left–right), no overlap
left  = annulus((-sep/2, 0.0), R, width)
right = annulus((+sep/2, 0.0), R, width)
side_by_side = MultiPolygon([left, right])      # or: unary_union([left, right])

# 3) Two annuluses above one another (vertical), no overlap
bottom = annulus((0.0, -sep/2), R, width)
top    = annulus((0.0, +sep/2), R, width)
stacked = MultiPolygon([bottom, top])           # or: unary_union([bottom, top])


res = 256  # smoothness for annulus

# 1) “Snowman” — two touching annuli
m1 = unary_union([
    annulus((0.0, +0.9), 0.9, 0.25, res),
    annulus((0.0, -0.2), 0.7, 0.22, res)
])
m1_item = convert("snowman", m1, samples=1000, visualize=False)


# 2) “Tri-ring” — triangle ring + centered annulus
m2 = unary_union([
    triangle(center=(0,0), width=0.18, side=2.6, rotation_deg=0),
    annulus((0,0), 0.55, 0.18, res)
])
m2_item = convert("tri-ring", m2, samples=1000, visualize=False)


# 3) “Windmill” — 3 small annuli around a square ring
m3 = unary_union([
    square(center=(0,0), width=0.20, side=2.2, rotation_deg=30),
    annulus((+1.2,0.0), 0.35, 0.12, res),
    annulus((-1.2,0.0), 0.35, 0.12, res),
    annulus((0.0,+1.2), 0.35, 0.12, res)
])
m3_item = convert("windmill", m3, samples=1000, visualize=False)


# 4) “Kite” — rotated square ring + offset triangle ring
m4 = unary_union([
    square(center=(0,0), width=0.22, side=2.8, rotation_deg=45),
    triangle(center=(1.1,0.0), width=0.16, side=1.8, rotation_deg=20)
])
m4_item = convert("kite", m4, samples=1000, visualize=False)


# 5) “Flower” — central annulus + 4 surrounding annuli
m5 = unary_union([
    annulus((0,0), 0.55, 0.18, res),
    annulus((+0.95,0.0), 0.35, 0.12, res),
    annulus((-0.95,0.0), 0.35, 0.12, res),
    annulus((0.0,+0.95), 0.35, 0.12, res),
    annulus((0.0,-0.95), 0.35, 0.12, res)
])
m5_item = convert("flower", m5, samples=1000, seed=None, visualize=False)


### 3D Objects from DataGenerator ###
T1_item = DataGenerator.torus_item(R=2.0, r=1.0, samples=1000, seed=None, visualize=False)
T2_item = DataGenerator.torus_item(R=2.0, r=0.2, samples=1000, seed=None, visualize=False)
T3_item = DataGenerator.torus_item(R=2.0, r=1.8, samples=1000, seed=None, visualize=False)
DT1_item = DataGenerator.double_torus_item(
    R1=2.0, r1=0.5,
    R2=1.5, r2=0.3,
    samples=1000,
    seed=None,
    visualize=False,
)
DT2_item = DataGenerator.double_torus_item(
    R1=1.7, r1=0.9,
    R2=0.8, r2=0.2,
    samples=1000,
    seed=None,
    visualize=False,
)

item_list = [m1_item,
             m2_item,
             m3_item,
             m4_item,
             m5_item,
             T1_item,
             T2_item,
             T3_item,
             DT1_item,
             DT2_item]


if __name__ == "__main__":
    print(type(m1_item.shape))

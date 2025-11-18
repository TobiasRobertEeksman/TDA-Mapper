import numpy as np 
import matplotlib.pyplot as plt

from shapely.geometry import Point
from shapely.ops import unary_union
from shapely.affinity import translate, rotate
from shapely.geometry import JOIN_STYLE
from shapely.geometry import Polygon, MultiPolygon


R = 1
width = 0.25
res=256



def annulus(center_xy, R, width, res=256):
    outer = Point(center_xy).buffer(R, resolution=res)
    inner = Point(center_xy).buffer(R - width, resolution=res)
    return outer.difference(inner)


# Triangle ----------------------------------------------------------------------------


def triangle(center, width, *, side=2.0, rotation_deg=0.0):
    """
    Triangle ring (triangle with thickness `width`).

    Parameters
    ----------
    center : (x, y)
        Center of the triangle.
    width : float
        Band thickness (distance inset from the outer triangle).
        Must be < inradius = side/(2*sqrt(3)).
    side : float, optional (default 2.0)
        Outer triangle side length.
    rotation_deg : float, optional (default 0.0)
        Rotation of the triangle (degrees), 0° = one vertex up.

    Returns
    -------
    shapely.geometry.Polygon
        Polygon with one interior (the hole).
    """
    if width <= 0:
        raise ValueError("width must be > 0")

    # Build an equilateral triangle centered at the origin
    R = side / np.sqrt(3)  # circumradius
    angles = np.deg2rad([90, 210, 330])     # upright
    pts = [(R*np.cos(a), R*np.sin(a)) for a in angles]
    outer = Polygon(pts)

    # Rotate and move to the requested center
    if rotation_deg:
        outer = rotate(outer, rotation_deg, origin=(0, 0), use_radians=False)
    outer = translate(outer, xoff=center[0], yoff=center[1])

    # Ensure the width is feasible (inner triangle must exist)
    inradius = side / (2*np.sqrt(3))
    if width >= inradius:
        raise ValueError(f"width must be < inradius = {inradius:.4f} for side={side}")

    # Inset the outer triangle and subtract -> ring
    inner = outer.buffer(-width, join_style=JOIN_STYLE.mitre)
    return outer.difference(inner)



#Square ----------------------------------------------------------------------------------


def square(center, width, *, side=2.0, rotation_deg=0.0):
    """
    Square ring (square with thickness `width`).

    Parameters
    ----------
    center : (x, y)
        Center of the square.
    width : float
        Band thickness (distance inset from the outer square).
        Must be < side/2 so the inner square exists.
    side : float, optional (default 2.0)
        Outer square side length.
    rotation_deg : float, optional (default 0.0)
        Rotation of the square in degrees.

    Returns
    -------
    shapely.geometry.Polygon
        Polygon with one interior (the hole).
    """
    if width <= 0:
        raise ValueError("width must be > 0")
    if width >= side/2:
        raise ValueError(f"width must be < side/2 = {side/2:.4f} for side={side}")

    # Outer square centered at origin
    h = side / 2.0
    outer = Polygon([(-h, -h), (h, -h), (h, h), (-h, h)])

    # Rotate and translate to requested center
    if rotation_deg:
        outer = rotate(outer, rotation_deg, origin=(0, 0), use_radians=False)
    outer = translate(outer, xoff=center[0], yoff=center[1])

    # Inset to create the hole, keep sharp corners
    inner = outer.buffer(-width, join_style=JOIN_STYLE.mitre)
    return outer.difference(inner)


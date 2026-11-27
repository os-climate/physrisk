import shapely
from shapely.geometry.base import BaseGeometry

# Degrees (EPSG:4326), ~11 cm at the equator. Single source of truth for the coordinate
# grid used both when constructing derived geometries (e.g. buffers) and when hashing
# geometries for cache keys. These MUST use the same grid: snapping to a grid is
# idempotent, so a geometry already snapped at construction is unchanged when snapped
# again for its key. Snapping to two different grids in sequence is not equivalent to a
# single snap and would double the rounding error.
#
# Coordinates already rounded to 6 decimal places (a common convention) lie on this grid,
# so are unchanged by snapping. Computed coordinates (e.g. buffer vertices) are in effect
# uniformly distributed relative to any grid, so the choice of grid size does not affect
# how likely they are to fall near a rounding boundary.
GEOMETRY_GRID_SIZE = 1e-6


def snap_to_grid(
    geometry: BaseGeometry, grid_size: float = GEOMETRY_GRID_SIZE
) -> BaseGeometry:
    """Snap geometry coordinates to a grid of `grid_size` (no-op if grid_size is 0)."""
    if not grid_size:
        return geometry
    return shapely.set_precision(geometry, grid_size=grid_size)

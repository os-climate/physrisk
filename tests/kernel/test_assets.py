import numpy as np
from shapely import Point
from shapely.ops import transform

from physrisk.kernel.assets import Asset, project_4326_to_3857


def test_buffered_geometry_contains_origin():
    """Buffered point contains the original point."""
    pt = Point(10.0, 45.0)
    buffered = Asset.buffered_geometry(pt, buffer=1000.0)
    assert buffered.contains(pt)


def test_buffered_geometry_approximate_size():
    """Bounding box of the buffer in EPSG:3857 is approximately 2*buffer metres wide."""
    buffer_m = 1000.0
    pt = Point(0.0, 0.0)
    buffered = Asset.buffered_geometry(pt, buffer_m)

    buffered_3857 = transform(project_4326_to_3857, buffered)
    minx, miny, maxx, maxy = buffered_3857.bounds
    np.testing.assert_allclose(maxx - minx, 2 * buffer_m, rtol=0.01)
    np.testing.assert_allclose(maxy - miny, 2 * buffer_m, rtol=0.01)


def test_buffered_geometry_excludes_far_point():
    """Point more than buffer_m away from origin is not contained."""
    buffer_m = 1000.0
    pt = Point(0.0, 0.0)
    buffered = Asset.buffered_geometry(pt, buffer_m)
    far = Point(0.0, 1.1 * buffer_m / 111319)  # ~1.1x buffer distance north
    assert not buffered.contains(far)


def test_asset_lat_lon_with_buffer():
    """Asset with lat/lon and buffer has a polygon geometry; lat/lon are preserved."""
    asset = Asset(id="a", latitude=51.5, longitude=-0.1, buffer=500.0)
    assert asset.geometry is not None
    assert asset.geometry.area > 0
    assert asset.latitude == 51.5
    assert asset.longitude == -0.1


def test_asset_wkt_with_buffer():
    """Asset with wkt_geometry and buffer has a buffered polygon; centroid matches the input point."""
    asset = Asset(id="a", wkt_geometry="POINT (10.0 45.0)", buffer=200.0)
    assert asset.geometry is not None
    assert asset.geometry.area > 0
    assert abs(asset.latitude - 45.0) < 0.01
    assert abs(asset.longitude - 10.0) < 0.01


def test_asset_no_buffer_has_no_geometry():
    """Asset constructed from lat/lon without buffer has no geometry."""
    asset = Asset(id="a", latitude=51.5, longitude=-0.1)
    assert asset.geometry is None

"""Task 1 verification of the generic periodic voxel tools against analytic shapes."""

from __future__ import annotations

import math

import numpy as np
import pytest

from voxlat.geometry.voxel_tools import (
    interface_distance,
    local_thickness,
    periodic_components,
    periodic_edt,
    periodic_surface_area,
    periodic_surface_points,
    throat_diameter,
    voxel_face_area,
)

N = 40
_X = np.arange(N, dtype=float)  # voxel centre k sits at coordinate k
X, Y, Z = np.meshgrid(_X, _X, _X, indexing="ij")


def pdist(a, c, n=N):
    d = np.abs(a - c)
    return np.minimum(d, n - d)


# -----------------------------------------------------------------------------
# Surface area
# -----------------------------------------------------------------------------
def test_surface_area_of_flat_slab_is_exact():
    f = np.abs(X - 20.3) - 6.1  # two planes x = 14.2, 26.4
    assert periodic_surface_area(f, 0.0) == pytest.approx(2 * N * N, rel=1e-12)
    # staircase area of a slab aligned with the grid is exact too
    assert voxel_face_area(f <= 0) == 2 * N * N


def test_surface_area_of_sphere_converges_and_staircase_does_not():
    R = 12.0
    f = np.sqrt((X - 20.2) ** 2 + (Y - 19.7) ** 2 + (Z - 20.1) ** 2) - R
    exact = 4 * math.pi * R**2
    assert periodic_surface_area(f, 0.0) == pytest.approx(exact, rel=0.01)
    # voxel faces over-estimate by ~3/2 for a curved surface (cube-projection argument)
    assert voxel_face_area(f <= 0) / exact == pytest.approx(1.5, rel=0.05)


def test_surface_area_counts_wrapped_cubes():
    """A plane that crosses the periodic boundary must keep its full area."""
    f = np.abs(pdist(X, 0.3)) - 4.0  # slab centred on the x = 0 face
    assert periodic_surface_area(f, 0.0) == pytest.approx(2 * N * N, rel=1e-12)


# -----------------------------------------------------------------------------
# Distance transform
# -----------------------------------------------------------------------------
def test_periodic_edt_matches_brute_force():
    rng = np.random.default_rng(3)
    n = 10
    p = rng.random((n, n, n)) < 0.8
    d = periodic_edt(p)
    pts = np.argwhere(p)
    bg = np.argwhere(~p)
    for q in pts[rng.choice(len(pts), 40, replace=False)]:
        diff = np.abs(bg - q)
        diff = np.minimum(diff, n - diff)
        assert d[tuple(q)] == pytest.approx(np.sqrt((diff**2).sum(1)).min())


def test_interface_distance_subvoxel_for_plane():
    """Slab 8.5 +- 8.87 (wraps through x = 0): exact sub-voxel distance to its faces."""
    f = pdist(X, 8.5) - 8.87
    ph = f <= 0
    D = interface_distance(ph, periodic_surface_points(f, [0.0]))
    exact = 8.87 - pdist(X, 8.5)
    assert np.max(np.abs(D[ph] - exact[ph])) < 1e-5  # marching cubes works in float32
    # voxel-only fallback (EDT - 1/2) is within half a voxel
    D0 = interface_distance(ph)
    assert np.max(np.abs(D0[ph] - exact[ph])) <= 0.5 + 1e-9


# -----------------------------------------------------------------------------
# Local thickness and throat (calibration against analytic shapes)
# -----------------------------------------------------------------------------
def _lt_bias(seed, shape_kind):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(6):
        if shape_kind == "cylinder":
            R = rng.uniform(3.0, 9.0)
            cx, cy = 20 + rng.uniform(0, 1, 2)
            f = np.hypot(X - cx, Y - cy) - R
            true = 2 * R
        else:
            T = rng.uniform(5.0, 16.0)
            f = np.abs(X - (10 + rng.uniform(0, 1) + T / 2)) - T / 2
            true = T
        ph = f <= 0
        D = interface_distance(ph, periodic_surface_points(f, [0.0]))
        lt = local_thickness(ph, distance=D)[ph]
        out.append([lt.min() - true, np.percentile(lt, 5) - true, np.median(lt) - true])
    return np.array(out)


@pytest.mark.parametrize("shape_kind", ["cylinder", "slab"])
def test_local_thickness_calibration(shape_kind):
    """Sub-voxel local thickness: unbiased on average, within ~1 voxel worst case."""
    b = _lt_bias(11, shape_kind)
    assert abs(b[:, 2].mean()) < 0.4  # median: mean bias
    assert np.all(np.abs(b[:, 1]) <= 1.1)  # p5
    assert np.all(np.abs(b[:, 0]) <= 1.5)  # min


def test_local_thickness_without_level_set_is_reasonable():
    ph = np.abs(X - 20.0) <= 4.0  # 9 voxel centres -> thickness 9 voxels
    lt = local_thickness(ph)[ph]
    assert np.median(lt) == pytest.approx(9.0, abs=1.0)


def _rod_network(R, c):
    dx, dy, dz = pdist(X, c[0]), pdist(Y, c[1]), pdist(Z, c[2])
    return np.minimum.reduce([np.hypot(dx, dy), np.hypot(dy, dz), np.hypot(dx, dz)]) - R


def test_throat_of_rod_network():
    rng = np.random.default_rng(5)
    for _ in range(4):
        R = rng.uniform(3.5, 7.5)
        f = _rod_network(R, 20 + rng.uniform(0, 1, 3))
        ph = f <= 0
        D = interface_distance(ph, periodic_surface_points(f, [0.0]))
        assert throat_diameter(ph, distance=D) == pytest.approx(2 * R, abs=1.0)


def test_throat_zero_for_non_percolating_phase():
    ph = np.hypot(X - 20, Y - 20) <= 6  # rods along z only: no path in x or y
    assert throat_diameter(ph) == 0.0


# -----------------------------------------------------------------------------
# Periodic connectivity
# -----------------------------------------------------------------------------
def test_components_slab_ball_rod():
    slab = np.abs(X - 20) <= 3
    c = periodic_components(slab)
    assert (c.n_components, c.percolates[0], c.wrap_rank[0]) == (1, (False, True, True), 2)

    ball = (X - 20) ** 2 + (Y - 20) ** 2 + (Z - 20) ** 2 <= 36
    c = periodic_components(ball)
    assert (c.n_components, c.percolates[0], c.wrap_rank[0]) == (1, (False, False, False), 0)

    rod = (pdist(Y, 0.0) <= 3) & (pdist(Z, 0.0) <= 3)  # along x, split by the y/z faces
    c = periodic_components(rod)
    assert (c.n_components, c.percolates[0], c.wrap_rank[0]) == (1, (True, False, False), 1)


def test_components_counts_and_fractions():
    two = (np.abs(X - 10) <= 2) | (np.abs(X - 30) <= 4)
    c = periodic_components(two)
    assert c.n_components == 2
    assert c.volume_fractions[0] == pytest.approx(9 / 14)
    assert c.labels.max() == 2


def test_rod_network_is_3d_connected_and_complement_too():
    ph = _rod_network(5.0, (20.3, 20.1, 19.8)) <= 0
    for phase in (ph, ~ph):
        c = periodic_components(phase)
        assert c.single_percolating and c.wrap_rank[0] == 3


def test_diagonal_contact_6_vs_26():
    """Two voxels touching only along an edge across the periodic boundary."""
    n = 6
    p = np.zeros((n, n, n), bool)
    p[n - 1, n - 1, 2] = True
    p[0, 0, 2] = True  # edge-neighbour of the first through the x and y faces
    assert periodic_components(p, connectivity=6).n_components == 2
    c26 = periodic_components(p, connectivity=26)
    assert c26.n_components == 1 and c26.wrap_rank[0] == 0


def test_staircase_diagonal_rod_wraps_along_110():
    n = 12
    p = np.zeros((n, n, n), bool)
    for k in range(n):  # 6-connected staircase climbing x and y together, 2 layers thick in z
        p[k, k, :2] = True
        p[(k + 1) % n, k, :2] = True
    c = periodic_components(p)
    assert c.n_components == 1
    assert c.percolates[0] == (True, True, False)
    assert c.wrap_rank[0] == 1


@pytest.mark.parametrize("connectivity", [6, 26])
def test_random_media_component_count_matches_tiled_labelling(connectivity):
    """Periodic count == number of distinct labels seen by a big tiled non-periodic labelling."""
    from scipy import ndimage

    rng = np.random.default_rng(7)
    n = 8
    p = rng.random((n, n, n)) < 0.25
    c = periodic_components(p, connectivity=connectivity)
    # every voxel of the central cell of a 3x3x3 tiling inherits its periodic component
    structure = ndimage.generate_binary_structure(3, 1 if connectivity == 6 else 3)
    big, _ = ndimage.label(np.tile(p, (3, 3, 3)), structure=structure)
    centre = big[n:2 * n, n:2 * n, n:2 * n]
    pairs = set(zip(c.labels[p].tolist(), centre[p].tolist()))
    # one tiled label may only map to one periodic label
    by_tiled = {}
    for a, b in pairs:
        by_tiled.setdefault(b, set()).add(a)
    assert all(len(v) == 1 for v in by_tiled.values())
    assert len(set(c.labels[p].tolist())) == c.n_components

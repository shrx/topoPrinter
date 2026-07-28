"""Tests for the vector cleanup that turns a raw satellite mask into an insert.

``apcsf_clean`` is an area-preserving curve-shortening flow, and its whole contract
is a pair of claims that pull against each other: the shape must get *rounder and
thicker* (hair-thin NDSI finger tips retract until the piece can actually be printed
and handled) while the *enclosed area stays fixed* (it is a measurement of snow
cover, so the flow may redistribute area but not invent or destroy it). Neither
claim is visible in a render -- a drifting area looks perfectly plausible -- so both
are asserted numerically here.

The flow's geometry is pinned via a circle, where the analytic answer is known: one
step moves a resampled ring inward by ``dt * spacing * kappa``, so halving the radius
must exactly double the step. That is the "high curvature retracts fastest" property
the finger retraction depends on, measured directly instead of eyeballed.
"""

import json

import numpy as np
import pytest
from shapely.geometry import MultiPolygon, Point, Polygon, box, mapping
from shapely.ops import unary_union

import masks.sentinel2 as so
from masks.sentinel2 import (_csf_ring, _src_epsg, apcsf_clean, apcsf_step, despeckle,
                          despeckle_area_m2, load_geojson_layer, npoly,
                          resample_ring, snow_to_ref_geoms, thin_fraction)


SMALL_FEATURE_M2 = 100.0        # test threshold, well below the fixture sizes
SPACING = 15.0
DT = 4.0


def _disc(r, at=(0.0, 0.0), quad_segs=64):
    return Point(*at).buffer(r, quad_segs=quad_segs)


def _fingered_blob():
    """A round mass with one hair-thin 16 m strand -- the shape APCSF exists for."""
    return unary_union([_disc(150, quad_segs=48), box(140, -8, 420, 8)])


# --------------------------------------------------------------------------
# despeckle and the feature-size constants
# --------------------------------------------------------------------------

class TestDespeckle:
    def test_drops_polygons_below_the_area(self):
        big, small = box(0, 0, 100, 100), box(500, 500, 502, 502)
        kept = despeckle(MultiPolygon([big, small]), min_area=1000.0)
        assert kept.geom_type == "Polygon"
        assert kept.area == pytest.approx(big.area)

    def test_keeps_polygons_at_or_above_the_area(self):
        """The test is ``< min_area``, so exactly min_area survives."""
        square = box(0, 0, 10, 10)          # area exactly 100
        assert not despeckle(square, min_area=100.0).is_empty

    def test_does_not_move_the_boundary_of_what_it_keeps(self):
        """Exact-area cleanup: no buffering, so kept coordinates are untouched."""
        big = box(0, 0, 100, 100)
        kept = despeckle(MultiPolygon([big, box(500, 500, 501, 501)]), 1000.0)
        assert kept.equals(big)

    def test_fills_holes_below_the_area(self):
        ring = Polygon(box(0, 0, 100, 100).exterior.coords,
                       [box(40, 40, 42, 42).exterior.coords])
        kept = despeckle(ring, min_area=1000.0)
        assert not kept.interiors, "a 4 m^2 pinhole is not printable detail"
        assert kept.area == pytest.approx(10000.0)

    def test_keeps_holes_at_or_above_the_area(self):
        hole = box(20, 20, 80, 80)
        ring = Polygon(box(0, 0, 100, 100).exterior.coords, [hole.exterior.coords])
        kept = despeckle(ring, min_area=1000.0)
        assert len(kept.interiors) == 1
        assert kept.area == pytest.approx(10000.0 - hole.area)

    def test_returns_the_input_when_everything_is_too_small(self):
        """The ``if kept else geom`` guard: a layer is never annihilated wholesale.

        Inside the APCSF loop this is deliberate -- a batch that momentarily shrinks
        every component below the threshold must not wipe the geometry out. Note that
        it also makes despeckle non-monotonic in a way callers can be surprised by:
        one sub-threshold polygon alone is KEPT, while the same polygon beside a
        large one is dropped. ``terrain_compose.drop_unprintable`` inherits that.
        """
        speck = box(0, 0, 1, 1)
        assert despeckle(speck, min_area=1000.0).equals(speck)
        pair = MultiPolygon([speck, box(500, 500, 600, 600)])
        assert despeckle(pair, min_area=1000.0).geom_type == "Polygon"

    def test_area_threshold_scales_with_the_square_of_the_print_scale(self):
        assert despeckle_area_m2(1.0) == pytest.approx(so.MIN_FEATURE_MM ** 2)
        assert despeckle_area_m2(100.0) == pytest.approx(
            despeckle_area_m2(50.0) * 4.0), "area goes as scale^2"

    def test_npoly_counts_components(self):
        assert npoly(box(0, 0, 1, 1)) == 1
        assert npoly(MultiPolygon([box(0, 0, 1, 1), box(5, 5, 6, 6)])) == 2


# --------------------------------------------------------------------------
# resample_ring
# --------------------------------------------------------------------------

class TestResampleRing:
    def test_returns_an_open_ring(self):
        """The flow rolls neighbours cyclically, so a duplicated end point would
        make one vertex its own neighbour and freeze it."""
        pts = resample_ring(_disc(100).exterior.coords, SPACING)
        assert not np.allclose(pts[0], pts[-1])

    def test_spacing_is_uniform_including_the_closing_segment(self):
        pts = resample_ring(_disc(100).exterior.coords, SPACING)
        loop = np.vstack([pts, pts[0]])
        seg = np.hypot(*np.diff(loop, axis=0).T)
        assert seg.std() / seg.mean() < 0.05, "arc-length spacing should be even"
        assert seg.mean() == pytest.approx(SPACING, rel=0.1)

    def test_point_count_follows_the_perimeter(self):
        pts = resample_ring(_disc(100).exterior.coords, SPACING)
        assert len(pts) == pytest.approx(2 * np.pi * 100 / SPACING, rel=0.1)

    def test_finer_spacing_gives_more_points(self):
        coords = _disc(100).exterior.coords
        assert len(resample_ring(coords, 5.0)) > len(resample_ring(coords, 20.0))

    def test_shape_is_preserved(self):
        pts = resample_ring(_disc(100).exterior.coords, SPACING)
        assert Polygon(pts).area == pytest.approx(np.pi * 100 ** 2, rel=0.02)

    def test_already_open_input_is_accepted(self):
        square = np.array([[0.0, 0], [10, 0], [10, 10], [0, 10]])
        assert Polygon(resample_ring(square, 2.0)).area == pytest.approx(100.0, rel=0.05)

    def test_never_returns_fewer_than_three_points(self):
        """max(3, ...) keeps a tiny ring constructible."""
        assert len(resample_ring(_disc(0.001).exterior.coords, SPACING)) >= 3

    def test_degenerate_input_is_passed_through(self):
        assert len(resample_ring(np.array([[0.0, 0.0], [1.0, 1.0]]), SPACING)) < 3


# --------------------------------------------------------------------------
# the curve-shortening step itself
# --------------------------------------------------------------------------

class TestCurveShortening:
    """One step moves a ring inward by dt * spacing * curvature."""

    def _mean_radius(self, pts):
        return float(np.hypot(*np.asarray(pts).T).mean())

    def _step(self, radius, dt=DT):
        """Inward movement of one step, measured against the RESAMPLED ring.

        Not against the nominal radius: resample_ring interpolates along the disc's
        chords, so its vertices start a sagitta inside the true circle (0.02 m at
        R=400). That offset is a fixed discretisation artefact, and comparing to
        ``radius`` would fold it into the measurement -- enough to break the dt
        ratio below, whose step at dt=1 is only 0.07 m.
        """
        coords = _disc(radius).exterior.coords
        before = self._mean_radius(resample_ring(coords, SPACING))
        return before - self._mean_radius(_csf_ring(coords, dt, SPACING))

    @pytest.mark.parametrize("radius", [100.0, 200.0, 400.0])
    def test_inward_step_matches_dt_times_spacing_times_curvature(self, radius):
        assert self._step(radius) == pytest.approx(DT * SPACING / radius, rel=0.02)

    def test_halving_the_radius_doubles_the_retraction(self):
        """The mechanism behind finger retraction, stated as a ratio.

        A hair-thin tip is a high-curvature region, so it must pull back faster than
        the bulk it is attached to -- that is what makes the flow produce handleable
        blobs instead of merely smoothing everything equally.
        """
        assert self._step(200.0) / self._step(400.0) == pytest.approx(2.0, rel=0.02)

    def test_a_convex_ring_moves_strictly_inward(self):
        before = _disc(150)
        moved = Polygon(_csf_ring(before.exterior.coords, DT, SPACING))
        assert moved.area < before.area
        assert before.buffer(1e-9).contains(moved)

    def test_the_step_shortens_the_ring(self):
        """It is a gradient step on LENGTH, which is what 'curve shortening' means."""
        before = _disc(150).exterior
        moved = Polygon(_csf_ring(before.coords, DT, SPACING)).exterior
        assert moved.length < before.length

    def test_a_bigger_step_moves_further(self):
        small, large = self._step(200.0, dt=1.0), self._step(200.0, dt=8.0)
        assert large > small
        assert large / small == pytest.approx(8.0, rel=0.02), "linear in dt"


# --------------------------------------------------------------------------
# apcsf_step / apcsf_clean -- the headline invariant
# --------------------------------------------------------------------------

class TestAreaPreservation:
    def test_apcsf_step_restores_the_target_area(self):
        disc = _disc(200)
        a0 = disc.area
        out = apcsf_step(disc, a0, DT, min_area=SMALL_FEATURE_M2, spacing=SPACING)
        assert out.area == pytest.approx(a0, rel=1e-3)

    def test_the_area_restore_is_first_order_but_converges(self):
        """a0 is a fixed TARGET, not "whatever came in" -- that is how the area of a
        vanishing component migrates into the survivors.

        The restore offsets the boundary by ``(a0 - area) / length``, which is the
        linear term only: outsetting by d actually adds ``length*d + pi*d^2``. Asked
        to close a 20% gap in one go it therefore overshoots by ~0.9%. That is not a
        defect at the size of gap the flow really produces (one curve-shortening step
        loses a fraction of a percent, where the quadratic term is negligible), and
        iterating converges -- but it does mean a single step is not a way to resize
        a polygon, so the behaviour is pinned rather than assumed exact.
        """
        disc = _disc(200)
        target = disc.area * 1.2
        one = apcsf_step(disc, target, DT, min_area=SMALL_FEATURE_M2, spacing=SPACING)
        assert one.area > disc.area, "it must move toward the target"
        assert one.area == pytest.approx(target, rel=0.02)
        assert one.area > target, "first-order offset overshoots a large gap"

        again = apcsf_step(one, target, DT, min_area=SMALL_FEATURE_M2, spacing=SPACING)
        assert again.area == pytest.approx(target, rel=1e-3), "and it converges"

    @pytest.mark.parametrize("iterations", [1, 10, 30, 60])
    def test_area_is_preserved_across_the_whole_flow(self, iterations):
        disc = _disc(200)
        out = apcsf_clean(disc, iterations, dt=DT, min_feature_m2=SMALL_FEATURE_M2,
                          resample_m=SPACING)
        assert out.area == pytest.approx(disc.area, rel=1e-3)

    def test_area_is_preserved_for_a_fingered_shape(self):
        blob = _fingered_blob()
        out = apcsf_clean(blob, 50, dt=DT, min_feature_m2=SMALL_FEATURE_M2,
                          resample_m=SPACING)
        assert out.area == pytest.approx(blob.area, rel=1e-3)

    def test_area_is_measured_after_the_initial_despeckle(self):
        """Specks are discarded first, so their area is NOT redistributed."""
        blob = unary_union([_disc(200), _disc(2, at=(600, 600))])
        out = apcsf_clean(blob, 20, dt=DT, min_feature_m2=SMALL_FEATURE_M2,
                          resample_m=SPACING)
        despeckled = despeckle(blob, SMALL_FEATURE_M2)
        assert despeckled.area < blob.area, "fixture must lose a speck"
        assert out.area == pytest.approx(despeckled.area, rel=1e-3)

    def test_zero_iterations_is_just_a_despeckle(self):
        blob = unary_union([_disc(200), _disc(2, at=(600, 600))])
        out = apcsf_clean(blob, 0, dt=DT, min_feature_m2=SMALL_FEATURE_M2,
                          resample_m=SPACING)
        assert out.equals(despeckle(blob, SMALL_FEATURE_M2))

    def test_batching_does_not_change_the_result(self):
        """geos_every is a performance knob; the geometry must not depend on it.

        The batch defers the GEOS validity/union/area-restore to the batch boundary,
        which is only sound while a few small steps barely self-intersect.
        """
        disc = _disc(200)
        a = apcsf_clean(disc, 20, dt=DT, min_feature_m2=SMALL_FEATURE_M2,
                        resample_m=SPACING, geos_every=1)
        b = apcsf_clean(disc, 20, dt=DT, min_feature_m2=SMALL_FEATURE_M2,
                        resample_m=SPACING, geos_every=10)
        assert a.area == pytest.approx(b.area, rel=1e-3)
        assert a.length == pytest.approx(b.length, rel=0.02)


class TestPrintability:
    """The flow's actual purpose: make the mask printable and handleable."""

    def test_thin_strands_retract(self):
        blob = _fingered_blob()
        r = 40.0
        before = thin_fraction(blob, r)
        after = thin_fraction(apcsf_clean(blob, 50, dt=DT,
                                         min_feature_m2=SMALL_FEATURE_M2,
                                         resample_m=SPACING), r)
        assert before > 0.01, "fixture must actually have a thin strand"
        assert after < before / 4.0, "the strand should be largely gone"

    def test_the_boundary_gets_shorter_as_the_flow_runs(self):
        """Same area, less perimeter -- the shape is becoming rounder, not smaller."""
        blob = _fingered_blob()
        lengths = [apcsf_clean(blob, n, dt=DT, min_feature_m2=SMALL_FEATURE_M2,
                               resample_m=SPACING).length for n in (0, 20, 50)]
        assert lengths[0] > lengths[1] > lengths[2]

    def test_a_disc_is_a_fixed_point(self):
        """A circle already minimises perimeter for its area, so the flow must leave
        it alone -- a drifting disc would mean the area restore is mis-scaled."""
        disc = _disc(200)
        out = apcsf_clean(disc, 60, dt=DT, min_feature_m2=SMALL_FEATURE_M2,
                          resample_m=SPACING)
        assert out.area == pytest.approx(disc.area, rel=1e-3)
        assert out.length == pytest.approx(disc.length, rel=0.01)

    def test_specks_are_removed_during_the_flow(self):
        blob = unary_union([_disc(200), _disc(3, at=(700, 700))])
        out = apcsf_clean(blob, 20, dt=DT, min_feature_m2=SMALL_FEATURE_M2,
                          resample_m=SPACING)
        assert npoly(out) == 1

    def test_thin_fraction_is_zero_for_a_chunky_shape(self):
        assert thin_fraction(_disc(300), 40.0) == pytest.approx(0.0, abs=0.01)

    def test_thin_fraction_is_never_negative(self):
        """buffer(-r).buffer(r) can exceed the input on a convex shape."""
        assert thin_fraction(box(0, 0, 500, 500), 20.0) >= 0.0

    def test_thin_fraction_is_high_for_a_strand(self):
        strand = box(0, 0, 600, 20)          # 20 m wide, thinner than 2r
        assert thin_fraction(strand, 40.0) == pytest.approx(1.0, abs=0.05)


# --------------------------------------------------------------------------
# GeoJSON I/O
# --------------------------------------------------------------------------

class TestGeojsonIo:
    def _write(self, path, geoms, crs_name="EPSG:32638"):
        fc = {"type": "FeatureCollection",
              "features": [{"type": "Feature", "properties": {},
                            "geometry": mapping(g)} for g in geoms]}
        if crs_name is not None:
            fc["crs"] = {"type": "name", "properties": {"name": crs_name}}
        path.write_text(json.dumps(fc))
        return str(path)

    def test_features_are_unioned(self):
        import tempfile, pathlib
        with tempfile.TemporaryDirectory() as d:
            p = self._write(pathlib.Path(d) / "s.geojson",
                            [box(0, 0, 10, 10), box(5, 0, 15, 10)])
            geom, epsg = load_geojson_layer(p)
        assert geom.geom_type == "Polygon", "the two overlapping boxes should merge"
        assert geom.area == pytest.approx(150.0)
        assert epsg == 32638

    def test_disjoint_features_stay_separate(self, tmp_path):
        p = self._write(tmp_path / "s.geojson",
                        [box(0, 0, 10, 10), box(50, 50, 60, 60)])
        geom, _ = load_geojson_layer(p)
        assert npoly(geom) == 2

    def test_crs_defaults_to_wgs84_when_absent(self, tmp_path):
        p = self._write(tmp_path / "s.geojson", [box(0, 0, 1, 1)], crs_name=None)
        assert load_geojson_layer(p)[1] == 4326

    def test_src_epsg_parses_a_short_name(self):
        assert _src_epsg({"crs": {"properties": {"name": "EPSG:32633"}}}) == 32633

    def test_src_epsg_parses_a_urn_name(self):
        """Splitting on ':' and taking the last field also handles the OGC URN form."""
        assert _src_epsg({"crs": {"properties": {
            "name": "urn:ogc:def:crs:EPSG::32638"}}}) == 32638

    def test_src_epsg_defaults_without_a_crs_key(self):
        assert _src_epsg({}) == 4326

    def test_reprojection_returns_one_dict_per_component(self):
        multi = MultiPolygon([box(500000, 4000000, 500100, 4000100),
                              box(600000, 4100000, 600100, 4100100)])
        out = snow_to_ref_geoms(multi, 32638, "EPSG:4326")
        assert len(out) == 2
        assert all(g["type"] == "Polygon" for g in out)

    def test_reprojection_reaches_plausible_lonlat(self):
        poly = box(500000, 4000000, 500100, 4000100)      # UTM 38N
        out = snow_to_ref_geoms(poly, 32638, "EPSG:4326")
        lons, lats = zip(*out[0]["coordinates"][0])
        assert 44.0 < min(lons) < 46.0, "UTM zone 38N spans ~42-48E"
        assert 35.0 < min(lats) < 37.0

    def test_identity_reprojection_keeps_coordinates(self):
        poly = box(500000, 4000000, 500100, 4000100)
        out = snow_to_ref_geoms(poly, 32638, "EPSG:32638")
        assert Polygon(out[0]["coordinates"][0]).equals_exact(poly, 1e-6)

    def test_a_single_polygon_gives_a_one_element_list(self):
        out = snow_to_ref_geoms(box(500000, 4000000, 500100, 4000100), 32638,
                                "EPSG:32638")
        assert len(out) == 1

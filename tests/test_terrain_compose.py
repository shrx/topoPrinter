"""Tests for the satellite layer resolver: base plate + mutually exclusive inserts.

``resolve_layers`` is where the print's manufacturability is decided, and its
invariant is exact: ``base`` plus every insert must TILE the cutout -- disjoint,
with no gap and no overlap. A gap here is not a cosmetic defect; a sliver of base
left standing between two recessed pockets prints as a full-height razor fin, and
two near-miss copies of a shared seam collapse into non-manifold pinches when the
mesh is exported as float32. Both were real bugs, so the partition is asserted
directly rather than inferred from the pieces looking plausible.

All geometry is in the cutout's CRS (metres). The tests pass
``scale_m_per_mm=1.0`` so that one metre is one printed millimetre and the
feature-size rules can be read straight off the fixtures.
"""

import itertools

import pytest
from shapely.geometry import MultiPolygon, Polygon, box
from shapely.ops import unary_union

from masks import (TERRAIN_FOLIAGE, TERRAIN_GLACIER,
                   TERRAIN_PRECEDENCE, TERRAIN_ROCK, TERRAIN_WATER)
from terrain_compose import (drop_unprintable, fretted_bit_moves, open_min_width,
                            resolve_foliage_base, resolve_layers)


SCALE = 1.0          # 1 m == 1 printed mm, so mm thresholds read as metres
THICK = 1.0          # min_thickness_mm
BLOB = 2.0           # min_blob_mm


def _assert_tiles_exactly(cutout, base, inserts, tol=1e-6):
    """base + inserts must tile the cutout: no gap, no overlap, nothing lost."""
    parts = [g for g in [base, *inserts.values()] if not g.is_empty]
    assert parts, "the resolver must produce something"

    total = sum(g.area for g in parts)
    assert total == pytest.approx(cutout.area, rel=1e-9), "area is not conserved"

    covered = unary_union(parts)
    assert covered.symmetric_difference(cutout).area == pytest.approx(0.0, abs=tol), \
        "the pieces do not cover the cutout exactly"

    for a, b in itertools.combinations(parts, 2):
        assert a.intersection(b).area == pytest.approx(0.0, abs=tol), \
            "two pieces overlap; the partition is not disjoint"


# --------------------------------------------------------------------------
# open_min_width / drop_unprintable -- the two feature-size rules
# --------------------------------------------------------------------------

class TestOpenMinWidth:
    def test_strip_wider_than_the_rule_survives(self):
        kept = open_min_width(box(0, 0, 50, 2.0), 1.0)
        assert not kept.is_empty
        assert kept.area == pytest.approx(box(0, 0, 50, 2.0).area, rel=0.05)

    def test_strip_thinner_than_the_rule_vanishes(self):
        assert open_min_width(box(0, 0, 50, 0.4), 1.0).is_empty

    def test_strip_at_exactly_the_rule_width_vanishes(self):
        """The opening erodes by half the width from each side, so it closes up."""
        assert open_min_width(box(0, 0, 50, 1.0), 1.0).is_empty

    def test_a_thin_neck_is_severed_but_the_blobs_remain(self):
        """This is why satellite inserts are never re-opened -- it cuts their necks."""
        dumbbell = unary_union([box(0, 0, 10, 10), box(10, 4.7, 14, 5.3),
                                box(14, 0, 24, 10)])
        opened = open_min_width(dumbbell, 1.0)
        assert opened.geom_type == "MultiPolygon", "the 0.6 m neck should be cut"
        assert len(opened.geoms) == 2


class TestDropUnprintable:
    """Two INDEPENDENT rules: elongated features by width, compact ones by area."""

    def test_chunky_region_is_kept(self):
        kept = drop_unprintable(box(0, 0, 10, 10), THICK, BLOB, SCALE)
        assert kept.area == pytest.approx(100.0, rel=0.05)

    def test_long_thin_ridge_is_dropped_despite_its_large_area(self):
        """The decisive case for having a width rule at all.

        This ridge is 0.6 m wide but 60 m^2 -- fifteen times the compact-blob
        threshold -- so an area-only filter would keep it and print a razor fin.
        """
        ridge = box(0, 0, 100, 0.6)
        assert ridge.area > (BLOB * SCALE) ** 2 * 10, "fixture must be large in area"
        assert drop_unprintable(ridge, THICK, BLOB, SCALE).is_empty

    def test_small_blob_is_dropped_even_though_it_survives_the_opening(self):
        """The decisive case for having an area rule as well as a width rule."""
        printable = box(0, 0, 10, 10)
        speck = box(50, 50, 51.5, 51.5)          # 1.5 m across: wider than THICK...
        assert not open_min_width(speck, THICK * SCALE).is_empty, \
            "fixture must survive the width rule, so only the area rule can drop it"
        kept = drop_unprintable(MultiPolygon([printable, speck]), THICK, BLOB, SCALE)
        assert kept.geom_type == "Polygon", "the speck should be gone"
        assert kept.area == pytest.approx(printable.area, rel=0.05)

    def test_a_lone_sub_minimum_blob_is_kept(self):
        """Documents a real asymmetry in ``masks.sentinel2.despeckle``.

        despeckle ends ``return unary_union(kept) if kept else geom`` -- when EVERY
        polygon is below the threshold it returns its input untouched instead of
        empty. In its own APCSF loop that is a deliberate guard against annihilating
        a layer; reached through drop_unprintable it means a region consisting of one
        sub-2 mm island is emitted as an insert that cannot be printed or handled,
        whereas the identical island alongside a printable neighbour is dropped (see
        the test above). Asserted so the inconsistency is visible and a deliberate
        fix breaks this test rather than passing silently.
        """
        speck = box(0, 0, 1.5, 1.5)
        assert not drop_unprintable(speck, THICK, BLOB, SCALE).is_empty

    def test_scale_converts_mm_rules_into_metres(self):
        """The same shape is printable at one scale and not at another."""
        strip = box(0, 0, 200, 5.0)              # 5 m wide
        assert not drop_unprintable(strip, THICK, BLOB, scale_m_per_mm=1.0).is_empty
        # At 1:10, 5 m of terrain is 0.5 printed mm -- under the 1 mm rule.
        assert drop_unprintable(strip, THICK, BLOB, scale_m_per_mm=10.0).is_empty


# --------------------------------------------------------------------------
# fretted_bit_moves
# --------------------------------------------------------------------------

class TestFrettedBitMoves:
    """Small pieces a locked seam frets off are handed to their main neighbour."""

    def _layers(self):
        # ROCK owns a printable blob plus a 1 m^2 bit sitting away to the right.
        # FOLIAGE hugs that bit's long right edge; WATER only grazes its left edge,
        # so the two candidates are ranked by contact, not by size or order.
        bit = box(12.0, 0.0, 13.0, 1.0)
        return {
            TERRAIN_ROCK: MultiPolygon([box(0, 0, 10, 10), bit]),
            TERRAIN_FOLIAGE: box(13.0, -2.0, 20.0, 3.0),
            TERRAIN_WATER: box(8.0, 0.0, 11.6, 1.0),
        }, bit

    def _boundary(self, bit):
        return bit.exterior          # a locked seam running along the bit

    def test_bit_goes_to_its_largest_contact_neighbour(self):
        layers, bit = self._layers()
        moves = fretted_bit_moves(layers, self._boundary(bit), BLOB, SCALE)
        assert len(moves) == 1
        frm, to, moved = moves[0]
        assert frm == TERRAIN_ROCK
        assert to == TERRAIN_FOLIAGE, "WATER only grazes the bit; FOLIAGE abuts it"
        assert moved.area == pytest.approx(bit.area)

    def test_printable_pieces_are_never_moved(self):
        layers, bit = self._layers()
        moves = fretted_bit_moves(layers, self._boundary(bit), BLOB, SCALE)
        assert all(m[2].area < (BLOB * SCALE) ** 2 for m in moves), \
            "the 10x10 blob must stay where it is"

    def test_a_small_piece_clear_of_the_seam_is_not_moved(self):
        """Only pieces the seam actually frets are candidates."""
        layers, bit = self._layers()
        far_seam = box(100, 100, 110, 110).exterior
        assert fretted_bit_moves(layers, far_seam, BLOB, SCALE) == []

    def test_locked_layers_are_neither_source_nor_recipient(self):
        layers, bit = self._layers()
        moves = fretted_bit_moves(layers, self._boundary(bit), BLOB, SCALE,
                                  locked=(TERRAIN_FOLIAGE,))
        assert all(to != TERRAIN_FOLIAGE for _f, to, _b in moves), \
            "a locked layer owns its outline and must not be grown"
        assert moves and moves[0][1] == TERRAIN_WATER, "it should fall to WATER instead"

    def test_locking_the_owner_stops_it_being_pared(self):
        layers, bit = self._layers()
        moves = fretted_bit_moves(layers, self._boundary(bit), BLOB, SCALE,
                                  locked=(TERRAIN_ROCK,))
        assert moves == []

    def test_moves_conserve_area(self):
        """The caller applies these as difference/union, so the partition must hold."""
        layers, bit = self._layers()
        before = sum(g.area for g in layers.values())
        moves = fretted_bit_moves(layers, self._boundary(bit), BLOB, SCALE)
        applied = dict(layers)
        for frm, to, piece in moves:
            applied[frm] = applied[frm].difference(piece)
            applied[to] = unary_union([applied[to], piece])
        assert sum(g.area for g in applied.values()) == pytest.approx(before)

    def test_empty_and_missing_layers_are_tolerated(self):
        layers, bit = self._layers()
        layers[TERRAIN_GLACIER] = Polygon()
        layers[TERRAIN_WATER] = None
        moves = fretted_bit_moves(layers, self._boundary(bit), BLOB, SCALE)
        assert [m[1] for m in moves] == [TERRAIN_FOLIAGE]


# --------------------------------------------------------------------------
# resolve_layers
# --------------------------------------------------------------------------

CUTOUT = box(0.0, 0.0, 100.0, 100.0)


class TestResolveLayersPartition:
    def test_rock_base_tiles_the_cutout(self):
        """The ordinary print: rock is both the leftover and the base plate."""
        base, inserts = resolve_layers(
            CUTOUT,
            {TERRAIN_GLACIER: box(10, 10, 40, 40), TERRAIN_FOLIAGE: box(60, 60, 90, 90)},
            base_class=TERRAIN_ROCK, scale_m_per_mm=SCALE)
        assert set(inserts) == {TERRAIN_GLACIER, TERRAIN_FOLIAGE}
        _assert_tiles_exactly(CUTOUT, base, inserts)

    def test_overlap_goes_to_the_higher_precedence_class(self):
        """GLACIER outranks FOLIAGE, so it keeps the shared area."""
        glacier, foliage = box(10, 10, 50, 50), box(30, 30, 70, 70)
        base, inserts = resolve_layers(
            CUTOUT, {TERRAIN_GLACIER: glacier, TERRAIN_FOLIAGE: foliage},
            base_class=TERRAIN_ROCK, scale_m_per_mm=SCALE)
        assert TERRAIN_PRECEDENCE.index(TERRAIN_GLACIER) \
            < TERRAIN_PRECEDENCE.index(TERRAIN_FOLIAGE), "fixture assumes this order"
        assert inserts[TERRAIN_GLACIER].area == pytest.approx(glacier.area)
        assert inserts[TERRAIN_FOLIAGE].area == pytest.approx(
            foliage.difference(glacier).area)
        _assert_tiles_exactly(CUTOUT, base, inserts)

    def test_base_spans_the_whole_cutout_minus_the_inserts(self):
        base, inserts = resolve_layers(
            CUTOUT, {TERRAIN_GLACIER: box(10, 10, 40, 40)},
            base_class=TERRAIN_ROCK, scale_m_per_mm=SCALE)
        expected = CUTOUT.difference(inserts[TERRAIN_GLACIER])
        assert base.symmetric_difference(expected).area == pytest.approx(0.0, abs=1e-6)

    def test_masks_are_clipped_to_the_cutout(self):
        """A satellite outline overhanging the rim must not escape it."""
        base, inserts = resolve_layers(
            CUTOUT, {TERRAIN_GLACIER: box(-50, -50, 40, 40)},
            base_class=TERRAIN_ROCK, scale_m_per_mm=SCALE)
        assert CUTOUT.buffer(1e-9).contains(inserts[TERRAIN_GLACIER])
        _assert_tiles_exactly(CUTOUT, base, inserts)

    def test_no_masks_leaves_one_solid_base(self):
        base, inserts = resolve_layers(CUTOUT, {}, base_class=TERRAIN_ROCK,
                                       scale_m_per_mm=SCALE)
        assert inserts == {}
        assert base.area == pytest.approx(CUTOUT.area)

    def test_a_non_rectangular_cutout_is_respected(self):
        disc = box(0, 0, 100, 100).centroid.buffer(45.0, quad_segs=64)
        base, inserts = resolve_layers(
            disc, {TERRAIN_GLACIER: box(30, 30, 70, 70)},
            base_class=TERRAIN_ROCK, scale_m_per_mm=SCALE)
        _assert_tiles_exactly(disc, base, inserts)


class TestResolveLayersInverted:
    """Ararat: FOLIAGE is the base, so ROCK becomes the raw-complement insert."""

    def test_foliage_base_tiles_the_cutout(self):
        base, inserts = resolve_layers(
            CUTOUT,
            {TERRAIN_GLACIER: box(10, 10, 40, 40), TERRAIN_FOLIAGE: box(50, 0, 100, 100)},
            base_class=TERRAIN_FOLIAGE, scale_m_per_mm=SCALE)
        assert TERRAIN_FOLIAGE not in inserts, "the base is not one of the inserts"
        assert TERRAIN_ROCK in inserts, "the leftover is an insert when it is not the base"
        _assert_tiles_exactly(CUTOUT, base, inserts)

    def test_vacated_sliver_is_absorbed_rather_than_left_as_a_gap(self):
        """The razor-fin regression.

        The rock leftover here is a 0.4 m strip pinched between the snow insert and
        the foliage base -- thinner than the 1 mm rule, so drop_unprintable removes
        it from the insert. That area must be picked up by a neighbour in the same
        breath: left to fall through, it becomes a hairline wall of base standing
        full height between two recessed pockets.
        """
        snow = box(0, 0, 100, 50)
        foliage = box(0, 50.4, 100, 100)          # leaves a 0.4 m rock strip at y=50
        base, inserts = resolve_layers(
            CUTOUT, {TERRAIN_GLACIER: snow, TERRAIN_FOLIAGE: foliage},
            base_class=TERRAIN_FOLIAGE, min_thickness_mm=THICK, min_blob_mm=BLOB,
            scale_m_per_mm=SCALE)

        assert TERRAIN_ROCK not in inserts, "a 0.4 m strip is not a printable insert"
        _assert_tiles_exactly(CUTOUT, base, inserts)
        # The strip specifically: it must be part of the base now.
        strip = box(0, 50.0, 100, 50.4)
        assert base.intersection(strip).area == pytest.approx(strip.area, rel=1e-6)

    def test_vacated_spike_returns_to_its_own_insert(self):
        """The other branch: a sliver whose neighbour is the insert, not the base.

        A thin spike off the rock blob, flanked by snow on both sides, borders the
        rock insert far more than the base. It goes back to rock -- restoring the
        original seam bit-exactly -- instead of being handed to the base, which
        would leave the fin the previous test describes.
        """
        snow = unary_union([box(40, 0, 100, 49.7), box(40, 50.3, 100, 100)])
        foliage = box(0, 90, 12, 100)
        base, inserts = resolve_layers(
            CUTOUT, {TERRAIN_GLACIER: snow, TERRAIN_FOLIAGE: foliage},
            base_class=TERRAIN_FOLIAGE, min_thickness_mm=THICK, min_blob_mm=BLOB,
            scale_m_per_mm=SCALE)

        _assert_tiles_exactly(CUTOUT, base, inserts)
        spike = box(40, 49.7, 100, 50.3)
        assert inserts[TERRAIN_ROCK].intersection(spike).area \
            > base.intersection(spike).area, "the spike belongs to the rock insert"
        # And the base is left with no orphan sliver where the spike was.
        base_bits = base.geoms if base.geom_type == "MultiPolygon" else [base]
        assert all(b.area >= (BLOB * SCALE) ** 2 for b in base_bits), \
            "the base must not keep an unprintable orphan piece"

    def test_satellite_inserts_are_not_re_opened(self):
        """Re-opening APCSF snow severs its interior necks and shaves it.

        The leftover insert IS opened; a satellite outline must not be, so a thin
        snow tendril has to survive intact.
        """
        tendril = box(40, 49.7, 80, 50.3)         # 0.6 m: under the 1 mm rule
        snow = unary_union([box(10, 30, 40, 70), tendril])
        base, inserts = resolve_layers(
            CUTOUT, {TERRAIN_GLACIER: snow, TERRAIN_FOLIAGE: box(0, 0, 100, 10)},
            base_class=TERRAIN_FOLIAGE, min_thickness_mm=THICK, min_blob_mm=BLOB,
            scale_m_per_mm=SCALE)
        assert inserts[TERRAIN_GLACIER].intersection(tendril).area \
            == pytest.approx(tendril.area, rel=1e-6), "the tendril was trimmed"
        _assert_tiles_exactly(CUTOUT, base, inserts)

    def test_inserts_stay_disjoint_after_the_seam_recut(self):
        """Shared seams are re-cut to ONE set of vertices, not two near-miss copies."""
        base, inserts = resolve_layers(
            CUTOUT,
            {TERRAIN_GLACIER: box(0, 0, 100, 50), TERRAIN_FOLIAGE: box(0, 60, 100, 100)},
            base_class=TERRAIN_FOLIAGE, min_thickness_mm=THICK, min_blob_mm=BLOB,
            scale_m_per_mm=SCALE)
        for a, b in itertools.combinations(inserts.values(), 2):
            assert a.intersection(b).area == pytest.approx(0.0, abs=1e-9)
        _assert_tiles_exactly(CUTOUT, base, inserts)


class TestResolveFoliageBase:
    def test_wrapper_matches_resolve_layers(self):
        snow, foliage = box(10, 10, 40, 40), box(50, 0, 100, 100)
        b1, rock, glacier = resolve_foliage_base(CUTOUT, foliage, snow,
                                                 scale_m_per_mm=SCALE)
        b2, inserts = resolve_layers(CUTOUT, {TERRAIN_GLACIER: snow,
                                              TERRAIN_FOLIAGE: foliage},
                                     base_class=TERRAIN_FOLIAGE, scale_m_per_mm=SCALE)
        assert b1.area == pytest.approx(b2.area)
        assert glacier.area == pytest.approx(inserts[TERRAIN_GLACIER].area)
        assert rock.area == pytest.approx(inserts.get(TERRAIN_ROCK, Polygon()).area)

    def test_missing_classes_come_back_empty_not_none(self):
        """The preview scripts index the result unconditionally."""
        _base, rock, glacier = resolve_foliage_base(CUTOUT, CUTOUT, Polygon(),
                                                    scale_m_per_mm=SCALE)
        assert glacier.is_empty and rock.is_empty

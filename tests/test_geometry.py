import numpy as np
import pytest

from pipespacing.geometry import (
    Camera, build_pipes, classify, locate_pipes, nearest_nps,
    neighbour_pairs, pair_spacing,
)

CAM = Camera.from_focal_px(1280, 960, 1000.0)
OD = 0.1143  # 4" NPS


def project_horizontal(y_m, z_m, x_from=-1.0, x_to=1.0, od=OD):
    """Image box of a horizontal pipe whose centreline is at (y, z)."""
    u1 = CAM.cx + CAM.fx * x_from / z_m
    u2 = CAM.cx + CAM.fx * x_to / z_m
    v = CAM.cy + CAM.fy * y_m / z_m
    r_px = CAM.fy * (od / 2) / z_m
    return (u1, v - r_px, u2, v + r_px)


def synthetic_depth(pipes_yz, od=OD):
    """Depth map with the near surface of each pipe painted in (background far)."""
    depth = np.full((960, 1280), 10.0)
    for y, z in pipes_yz:
        x1, y1, x2, y2 = project_horizontal(y, z, od=od)
        depth[int(y1):int(y2) + 1, int(x1):int(x2) + 1] = z - od / 2
    return depth


def test_classify():
    assert classify((0, 0, 400, 40))[0] == "horizontal"
    assert classify((0, 0, 40, 400)) == ("vertical", 40)
    assert classify((0, 0, 100, 90))[0] == "other"


def test_nearest_nps():
    assert nearest_nps(112) == '4"'
    assert nearest_nps(60) == '2"'


def test_reference_mode_recovers_spacing():
    # Two 4" pipes, 300 mm apart vertically, 3 m from the camera.
    boxes = [project_horizontal(0.0, 3.0), project_horizontal(0.3, 3.0)]
    pipes = build_pipes(boxes, [0.9, 0.9])
    assert locate_pipes(pipes, CAM, od_mm=114.3) == "reference"
    [(a, b)] = neighbour_pairs(pipes)
    pair = pair_spacing(a, b)
    assert pair.spacing_mm == pytest.approx(300, abs=1)
    assert pair.clearance_mm == pytest.approx(300 - 114.3, abs=1)
    assert not pair.flagged


def test_depth_mode_handles_pipes_at_different_depths():
    # Same rack row, but one pipe sits 0.4 m behind the other.
    yz = [(0.0, 3.0), (0.2, 3.4)]
    pipes = build_pipes([project_horizontal(*p) for p in yz], [0.9, 0.9])
    assert locate_pipes(pipes, CAM, depth=synthetic_depth(yz)) == "depth"
    for p in pipes:
        assert p.od_mm == pytest.approx(114.3, rel=0.03)
        assert p.nps == '4"'
    [(a, b)] = neighbour_pairs(pipes)
    expected = np.hypot(0.2, 0.4) * 1000
    assert pair_spacing(a, b).spacing_mm == pytest.approx(expected, rel=0.02)


def test_fused_mode_corrects_depth_scale():
    # Depth model reads 20% short; the known OD should pull it back.
    yz = [(0.0, 3.0), (0.25, 3.0)]
    pipes = build_pipes([project_horizontal(*p) for p in yz], [0.9, 0.9])
    assert locate_pipes(pipes, CAM, depth=synthetic_depth(yz) * 0.8, od_mm=114.3) == "fused"
    [(a, b)] = neighbour_pairs(pipes)
    assert pair_spacing(a, b).spacing_mm == pytest.approx(250, rel=0.02)


def test_tight_clearance_is_flagged():
    boxes = [project_horizontal(0.0, 2.0), project_horizontal(0.13, 2.0)]
    pipes = build_pipes(boxes, [0.9, 0.9])
    locate_pipes(pipes, CAM, od_mm=114.3)
    [(a, b)] = neighbour_pairs(pipes)
    pair = pair_spacing(a, b, min_clearance_mm=25)
    assert pair.clearance_mm == pytest.approx(130 - 114.3, abs=1)
    assert pair.flagged


def test_negative_clearance_is_overlap_not_flagged():
    # Centrelines 80 mm apart can't hold two 114 mm pipes: a detection error.
    boxes = [project_horizontal(0.0, 2.0), project_horizontal(0.08, 2.0)]
    pipes = build_pipes(boxes, [0.9, 0.9])
    locate_pipes(pipes, CAM, od_mm=114.3)
    [(a, b)] = neighbour_pairs(pipes)
    pair = pair_spacing(a, b)
    assert pair.clearance_mm < 0
    assert pair.status == "overlap" and not pair.flagged


def test_pipes_that_do_not_run_alongside_are_not_paired():
    boxes = [
        project_horizontal(0.0, 3.0, x_from=-1.5, x_to=-0.5),
        project_horizontal(0.3, 3.0, x_from=0.5, x_to=1.5),
    ]
    pipes = build_pipes(boxes, [0.9, 0.9])
    locate_pipes(pipes, CAM, od_mm=114.3)
    assert neighbour_pairs(pipes) == []


def test_only_adjacent_pipes_are_paired():
    boxes = [project_horizontal(y, 3.0) for y in (0.0, 0.3, 0.6)]
    pipes = build_pipes(boxes, [0.9] * 3)
    locate_pipes(pipes, CAM, od_mm=114.3)
    assert [(a.id, b.id) for a, b in neighbour_pairs(pipes)] == [(0, 1), (1, 2)]

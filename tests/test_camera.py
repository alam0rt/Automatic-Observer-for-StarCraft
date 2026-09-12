import numpy as np

from observer.camera import CameraConfig, VIEW_PX, best_view, camera_track, coverage, view_cells, view_scores
from observer.features import GRID
from observer.labels import CELL_PX


def one_hot(row, col):
    heat = np.zeros((GRID, GRID), np.float32)
    heat[row, col] = 1.0
    return heat


def test_view_scores_count_mass_inside_each_view():
    vw, vh = view_cells()
    heat = one_hot(20, 30)
    scores = view_scores(heat)
    assert scores.shape == (GRID - vh + 1, GRID - vw + 1)
    # Every view whose top-left is within (vh, vw) cells above/left of the hot cell contains it
    assert scores.sum() == vw * vh
    assert scores[20, 30] == 1 and scores[20 - vh + 1, 30 - vw + 1] == 1
    assert scores[20 - vh, 30] == 0


def test_best_view_contains_the_hot_cell():
    view, score = best_view(one_hot(40, 10))
    assert score == 1
    assert view[0] <= 10 * CELL_PX < view[0] + VIEW_PX[0]
    assert view[1] <= 40 * CELL_PX < view[1] + VIEW_PX[1]
    assert coverage(one_hot(40, 10), view) == 1


def test_camera_holds_before_switching_then_cuts_to_far_action():
    config = CameraConfig(min_hold_frames=72, switch_margin=1.5, pan_limit_px=640)
    frames = np.arange(0, 400, 8)
    heats = np.stack([one_hot(5, 5) if f < 40 else one_hot(50, 50) for f in frames])

    track = camera_track(heats, frames, (GRID * CELL_PX, GRID * CELL_PX), config)

    near_start = track[track["frame"] < 72]
    assert (near_start["vpx"] < 640).all(), "camera must not leave before min_hold_frames"
    end = track.iloc[-1]
    assert end["vpx"] <= 50 * CELL_PX < end["vpx"] + VIEW_PX[0]
    assert end["vpy"] <= 50 * CELL_PX < end["vpy"] + VIEW_PX[1]


def test_camera_stays_inside_the_map():
    frames = np.arange(0, 80, 8)
    heats = np.stack([one_hot(GRID - 1, GRID - 1)] * len(frames))
    map_px = (96 * 32, 96 * 32)
    track = camera_track(heats, frames, map_px)
    assert (track["vpx"] <= map_px[0] - VIEW_PX[0]).all()
    assert (track["vpy"] <= map_px[1] - VIEW_PX[1]).all()

"""Turn per-frame interest heatmaps into a watchable camera track.

The original repo's observer jumped to whatever scored highest each step. Here the
camera holds a shot for a minimum time and only cuts when a new spot is clearly
better, and pans rather than cuts when the new spot is close by.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .labels import CELL_PX

# StarCraft's 640x480 screen minus the HUD shows 640x384 pixels of map (20x12 tiles)
VIEW_PX = (640, 384)


@dataclass(frozen=True)
class CameraConfig:
    min_hold_frames: int = 72       # never cut away sooner than this
    switch_margin: float = 1.5      # a new view must score this many times the current one
    pan_limit_px: int = 640         # moves shorter than this pan; longer ones cut
    pan_rate: float = 0.35          # fraction of the remaining distance covered per step


def view_cells() -> tuple[int, int]:
    return max(1, round(VIEW_PX[0] / CELL_PX)), max(1, round(VIEW_PX[1] / CELL_PX))


def view_scores(heat: np.ndarray) -> np.ndarray:
    """Probability mass inside the view whose top-left cell is (row, col)."""
    vw, vh = view_cells()
    summed = np.pad(heat, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    return summed[vh:, vw:] - summed[:-vh, vw:] - summed[vh:, :-vw] + summed[:-vh, :-vw]


def best_view(heat: np.ndarray) -> tuple[np.ndarray, float]:
    """Top-left of the best view in pixels, and its score."""
    scores = view_scores(heat)
    row, col = np.unravel_index(np.argmax(scores), scores.shape)
    return np.array([col * CELL_PX, row * CELL_PX], float), float(scores[row, col])


def score_at(heat: np.ndarray, top_left_px: np.ndarray) -> float:
    scores = view_scores(heat)
    col = int(np.clip(round(top_left_px[0] / CELL_PX), 0, scores.shape[1] - 1))
    row = int(np.clip(round(top_left_px[1] / CELL_PX), 0, scores.shape[0] - 1))
    return float(scores[row, col])


def camera_track(heats: np.ndarray, frames: np.ndarray, map_size_px: tuple[int, int],
                 config: CameraConfig = CameraConfig()) -> pd.DataFrame:
    """heats: (T, GRID, GRID) distributions at `frames`. Returns frame, vpx, vpy (viewport top-left)."""
    max_xy = np.array([map_size_px[0] - VIEW_PX[0], map_size_px[1] - VIEW_PX[1]], float).clip(min=0)

    position, _ = best_view(heats[0])
    goal = position.copy()
    last_switch = frames[0]
    rows = []
    for heat, frame in zip(heats, frames):
        candidate, candidate_score = best_view(heat)
        current_score = score_at(heat, goal)
        if frame - last_switch >= config.min_hold_frames and candidate_score > config.switch_margin * max(current_score, 1e-6):
            goal = candidate
            last_switch = frame
            if np.linalg.norm(goal - position) > config.pan_limit_px:
                position = goal.copy()

        position = position + config.pan_rate * (goal - position)
        clamped = np.clip(position, 0, max_xy)
        rows.append((int(frame), int(clamped[0]), int(clamped[1])))

    return pd.DataFrame(rows, columns=["frame", "vpx", "vpy"])


def coverage(heat: np.ndarray, top_left_px: np.ndarray) -> float:
    """Fraction of the best achievable view mass that this view captures (1 = optimal)."""
    _, best = best_view(heat)
    return score_at(heat, top_left_px) / best if best > 0 else 1.0

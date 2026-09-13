"""Interest targets: where on the map a spectator should be looking.

There are no human camera tracks for bot games, so targets come from the game itself:

  hindsight  value actually destroyed in the next `horizon` frames, discounted by
             how far ahead it happens. Only known offline, which is fine for labels.
  predicted  FAP's estimate of value lost by each fight at this frame. Catches
             fights that are about to start even if nothing has died yet.

When nothing is happening the target falls back to where the armies are, which is
what human observers show between fights.
"""
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter

from .features import CELL_TILES, GRID, TILE_PX, Game

CELL_PX = TILE_PX * CELL_TILES


@dataclass(frozen=True)
class LabelConfig:
    horizon: int = 144          # frames of hindsight (6 game seconds)
    decay: float = 72.0         # e-folding time of the hindsight discount, frames
    fap_weight: float = 0.5
    blur_cells: float = 1.5
    quiet_threshold: float = 50.0  # total interest below this (in value units) counts as quiet


def _splat(points_x, points_y, weights) -> np.ndarray:
    out = np.zeros((GRID, GRID), np.float32)
    cx = np.clip(np.asarray(points_x) // CELL_PX, 0, GRID - 1).astype(int)
    cy = np.clip(np.asarray(points_y) // CELL_PX, 0, GRID - 1).astype(int)
    np.add.at(out, (cy, cx), np.asarray(weights, np.float32))
    return out


def interest(game: Game, index: int, config: LabelConfig = LabelConfig()) -> tuple[np.ndarray, float]:
    """Target distribution over the (GRID, GRID) map at game.frames[index].

    Returns (target summing to 1, interest magnitude in value units).
    """
    frame = int(game.frames[index])

    events = game.events
    deaths = events[(events["event"] == "destroy") & (events["frame"] > frame)
                    & (events["frame"] <= frame + config.horizon) & (events["x"] >= 0)]
    discount = np.exp(-(deaths["frame"].to_numpy() - frame) / config.decay)
    heat = _splat(deaths["x"], deaths["y"], deaths["value"].to_numpy() * discount)

    if len(game.fights):
        fights = game.fights[game.fights["frame"] == frame]
        # sc-extract counts losses per unit so they can't be negative, but games extracted
        # before that netted FAP's regeneration and healing against damage. A negative
        # cell would make the target not a distribution.
        loss = (fights["loss_a"].clip(lower=0) + fights["loss_b"].clip(lower=0)).to_numpy()
        heat += config.fap_weight * _splat(fights["x"], fights["y"], loss)

    magnitude = float(heat.sum())
    if magnitude < config.quiet_threshold:
        heat = army_positions(game, index)

    heat = gaussian_filter(heat, config.blur_cells)
    total = heat.sum()
    if total <= 0:
        heat = np.full((GRID, GRID), 1.0 / (GRID * GRID), np.float32)
    else:
        heat = heat / total
    return heat.astype(np.float32), magnitude


def army_positions(game: Game, index: int) -> np.ndarray:
    """Value of mobile non-worker units, for frames with no fighting."""
    units = game.units_at(index)
    types = game.unit_types
    mobile = ~types["building"].reindex(units["type"]).fillna(True).to_numpy(bool)
    worker = types["worker"].reindex(units["type"]).fillna(False).to_numpy(bool)
    army = units[mobile & ~worker & (units["x"] >= 0).to_numpy()]
    return _splat(army["x"], army["y"], army["value"])

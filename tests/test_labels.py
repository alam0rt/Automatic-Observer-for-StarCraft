from pathlib import Path

import numpy as np
import pandas as pd

from observer.features import CELL_TILES, DYNAMIC_CHANNELS, GRID, N_DYNAMIC, TILE_PX, Game
from observer.labels import CELL_PX, LabelConfig, interest

ZERGLING, DRONE, HATCHERY = 37, 41, 131

UNIT_COLUMNS = ["frame", "id", "player", "type", "x", "y", "hp", "shields", "energy", "value",
                "flying", "burrowed", "cloaked", "attacking", "under_attack", "completed", "order", "order_x", "order_y"]


def unit(frame, uid, player, type_, x, y, value, attacking=False):
    return [frame, uid, player, type_, x, y, 35, 0, 0, value, False, False, False, attacking, False, True, 0, -1, -1]


def make_game(units, events=(), fights=None, size=64):
    return Game(
        name="synthetic",
        path=Path("."),
        meta={"map_width_tiles": size, "map_height_tiles": size, "interval": 8},
        map={"ground_height": [0] * size * size, "walkable_minitiles": [16] * size * size, "buildable": [1] * size * size},
        unit_types=pd.DataFrame({
            "building": {ZERGLING: False, DRONE: False, HATCHERY: True},
            "worker": {ZERGLING: False, DRONE: True, HATCHERY: False},
        }),
        units=pd.DataFrame(units, columns=UNIT_COLUMNS),
        events=pd.DataFrame(list(events), columns=["frame", "event", "id", "player", "type", "x", "y", "value"]),
        fights=fights if fights is not None else pd.DataFrame(),
    )


def cell_of(heat):
    return np.unravel_index(np.argmax(heat), heat.shape)


def test_upcoming_deaths_draw_the_target():
    units = [unit(f, 1, 0, ZERGLING, 100, 100, 50) for f in (0, 8, 16)]
    units += [unit(f, 2, 1, ZERGLING, 1500, 900, 50) for f in (0, 8, 16)]
    events = [(40, "destroy", 9, 1, ZERGLING, 1500, 900, 400)]
    game = make_game(units, events)

    target, magnitude = interest(game, 0, LabelConfig(quiet_threshold=10))

    assert np.isclose(target.sum(), 1)
    assert magnitude > 10
    assert cell_of(target) == (900 // CELL_PX, 1500 // CELL_PX)


def test_quiet_frames_follow_the_armies_not_workers_or_buildings():
    units = [unit(0, 1, 0, ZERGLING, 1200, 400, 500),
             unit(0, 2, 0, DRONE, 200, 200, 5000),
             unit(0, 3, 0, HATCHERY, 300, 1800, 5000)]
    game = make_game(units)

    target, magnitude = interest(game, 0)

    assert magnitude == 0
    assert cell_of(target) == (400 // CELL_PX, 1200 // CELL_PX)


def test_dynamic_features_put_units_in_their_player_and_kind_channels():
    units = [unit(0, 1, 3, ZERGLING, 70, 140, 100, attacking=True),
             unit(0, 2, 3, ZERGLING, 70, 140, 100),
             unit(0, 3, 5, DRONE, 700, 300, 50)]
    game = make_game(units)

    features = game.dynamic_features(0)
    assert features.shape == (N_DYNAMIC, GRID, GRID)

    cell = TILE_PX * CELL_TILES
    slots = game.player_slots
    ground = slots[3] * len(DYNAMIC_CHANNELS) + DYNAMIC_CHANNELS.index("ground_army")
    engaged = slots[3] * len(DYNAMIC_CHANNELS) + DYNAMIC_CHANNELS.index("engaged")
    workers = slots[5] * len(DYNAMIC_CHANNELS) + DYNAMIC_CHANNELS.index("workers")
    assert np.isclose(features[ground, 140 // cell, 70 // cell], np.log1p(2.0))
    assert np.isclose(features[engaged, 140 // cell, 70 // cell], np.log1p(1.0))
    assert np.isclose(features[workers, 300 // cell, 700 // cell], np.log1p(1.0))
    assert features.sum() > 0 and np.count_nonzero(features) == 3

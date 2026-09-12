"""Load extracted games and rasterise their state onto a fixed-size map grid.

Maps range up to 256x256 tiles; everything is padded to GRID_TILES and pooled by
CELL_TILES, so a cell is CELL_TILES build tiles on a side.
"""
import json
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import numpy as np
import pandas as pd

TILE_PX = 32
GRID_TILES = 128
CELL_TILES = 2
GRID = GRID_TILES // CELL_TILES

# Per-player dynamic channels
DYNAMIC_CHANNELS = ("ground_army", "air_army", "workers", "buildings", "engaged")
PLAYER_SLOTS = 2
N_DYNAMIC = len(DYNAMIC_CHANNELS) * PLAYER_SLOTS

STATIC_CHANNELS = ("ground_height", "walkable", "buildable", "on_map")
N_STATIC = len(STATIC_CHANNELS)


# eq=False: field-wise equality would compare DataFrames, which raises
@dataclass(eq=False)
class Game:
    name: str
    path: Path
    meta: dict
    map: dict
    unit_types: pd.DataFrame
    units: pd.DataFrame
    events: pd.DataFrame
    fights: pd.DataFrame
    _frame_bounds: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        self.units = self.units.sort_values("frame", kind="stable").reset_index(drop=True)
        self._frame_bounds = np.searchsorted(self.units["frame"].to_numpy(), self.frames, side="left")

    @property
    def width(self) -> int:
        return self.meta["map_width_tiles"]

    @property
    def height(self) -> int:
        return self.meta["map_height_tiles"]

    @cached_property
    def frames(self) -> np.ndarray:
        """Sampled frames, every meta['interval'] frames up to the last frame with units."""
        return np.arange(0, self.units["frame"].max() + 1, self.meta["interval"])

    @cached_property
    def player_slots(self) -> dict[int, int]:
        """Map player id -> slot. The two players with the most unit-frames get slots 0 and 1."""
        counts = self.units["player"].value_counts()
        return {int(player): slot for slot, player in enumerate(counts.index[:PLAYER_SLOTS])}

    def units_at(self, index: int) -> pd.DataFrame:
        end = self._frame_bounds[index + 1] if index + 1 < len(self.frames) else len(self.units)
        return self.units.iloc[self._frame_bounds[index]:end]

    @cached_property
    def static_features(self) -> np.ndarray:
        """(N_STATIC, GRID, GRID) float32 terrain channels."""
        shape = (self.height, self.width)
        grids = [
            np.asarray(self.map["ground_height"], np.float32).reshape(shape) / 5.0,
            np.asarray(self.map["walkable_minitiles"], np.float32).reshape(shape) / 16.0,
            np.asarray(self.map["buildable"], np.float32).reshape(shape),
            np.ones(shape, np.float32),
        ]
        return np.stack([pool(pad(g)) / CELL_TILES**2 for g in grids])

    @cached_property
    def _unit_channel_columns(self) -> tuple[np.ndarray, np.ndarray]:
        """Per unit-row channel index (-1 to skip) and weight."""
        types = self.unit_types
        type_id = self.units["type"].to_numpy()
        building = types["building"].reindex(type_id).fillna(False).to_numpy(bool)
        worker = types["worker"].reindex(type_id).fillna(False).to_numpy(bool)
        flying = self.units["flying"].to_numpy(bool)
        engaged = (self.units["attacking"] | self.units["under_attack"]).to_numpy(bool)

        kind = np.select([building, worker, flying], [3, 2, 1], default=0)
        slot = self.units["player"].map(self.player_slots).fillna(-1).to_numpy(int)
        channel = np.where(slot >= 0, slot * len(DYNAMIC_CHANNELS) + kind, -1)
        engaged_channel = np.where((slot >= 0) & engaged, slot * len(DYNAMIC_CHANNELS) + 4, -1)

        value = self.units["value"].to_numpy(np.float32)
        weight = np.where(kind == 2, 1.0, value / 100.0).astype(np.float32)
        return np.stack([channel, engaged_channel]), np.stack([weight, value / 100.0]).astype(np.float32)

    def dynamic_features(self, index: int) -> np.ndarray:
        """(N_DYNAMIC, GRID, GRID) float32 unit channels at self.frames[index], log-scaled."""
        end = self._frame_bounds[index + 1] if index + 1 < len(self.frames) else len(self.units)
        rows = slice(self._frame_bounds[index], end)
        channels, weights = self._unit_channel_columns
        x = self.units["x"].to_numpy()[rows]
        y = self.units["y"].to_numpy()[rows]
        on_map = (x >= 0) & (y >= 0)
        cx = np.clip(x // (TILE_PX * CELL_TILES), 0, GRID - 1)
        cy = np.clip(y // (TILE_PX * CELL_TILES), 0, GRID - 1)

        out = np.zeros((N_DYNAMIC, GRID, GRID), np.float32)
        for channel, weight in zip(channels[:, rows], weights[:, rows]):
            keep = on_map & (channel >= 0)
            np.add.at(out, (channel[keep], cy[keep], cx[keep]), weight[keep])
        return np.log1p(out)


def pad(grid: np.ndarray) -> np.ndarray:
    out = np.zeros((GRID_TILES, GRID_TILES), grid.dtype)
    h, w = min(grid.shape[0], GRID_TILES), min(grid.shape[1], GRID_TILES)
    out[:h, :w] = grid[:h, :w]
    return out


def pool(grid: np.ndarray) -> np.ndarray:
    """Sum-pool a GRID_TILES square grid down to GRID cells."""
    return grid.reshape(GRID, CELL_TILES, GRID, CELL_TILES).sum(axis=(1, 3))


def load_game(path: Path) -> Game:
    path = Path(path)
    meta = json.loads((path / "meta.json").read_text())
    if meta["map_width_tiles"] > GRID_TILES or meta["map_height_tiles"] > GRID_TILES:
        raise ValueError(f"{path.name}: {meta['map_width_tiles']}x{meta['map_height_tiles']} map is larger than {GRID_TILES} tiles")

    unit_types = pd.DataFrame.from_dict(json.loads((path / "unit_types.json").read_text()), orient="index")
    unit_types.index = unit_types.index.astype(int)

    fights_path = path / "fights.parquet"
    return Game(
        name=path.name,
        path=path,
        meta=meta,
        map=json.loads((path / "map.json").read_text()),
        unit_types=unit_types,
        units=pd.read_parquet(path / "units.parquet"),
        events=pd.read_parquet(path / "events.parquet"),
        fights=pd.read_parquet(fights_path) if fights_path.exists() else pd.DataFrame(),
    )


def load_games(root: Path) -> list[Game]:
    return [load_game(p) for p in sorted(Path(root).iterdir()) if (p / "meta.json").exists()]

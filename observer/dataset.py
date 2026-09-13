"""PyTorch datasets: a short history of map features -> where the action will be."""
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info

from .features import N_DYNAMIC, N_STATIC, Game, fits_grid, load_game
from .labels import LabelConfig, interest


@dataclass(frozen=True)
class WindowConfig:
    steps: int = 4      # frames of history fed to the model
    spacing: int = 3    # sampled frames between history steps (x interval game frames)

    @property
    def span(self) -> int:
        return (self.steps - 1) * self.spacing

    def in_channels(self) -> int:
        return self.steps * N_DYNAMIC + N_STATIC


class ObserverDataset(Dataset):
    """Items are (features, target, weight).

    features  (steps * N_DYNAMIC + N_STATIC, GRID, GRID), oldest step first
    target    (GRID, GRID), sums to 1
    weight    scalar; frames with more going on count for more
    """

    def __init__(self, games: list[Game], window: WindowConfig = WindowConfig(),
                 labels: LabelConfig = LabelConfig(), stride: int = 1):
        self.games = games
        self.window = window
        self.labels = labels
        self.index = [(g, i) for g, game in enumerate(games)
                      for i in range(window.span, len(game.frames), stride)]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, item):
        g, i = self.index[item]
        game = self.games[g]
        history = [game.dynamic_features(i - k * self.window.spacing) for k in reversed(range(self.window.steps))]
        features = np.concatenate(history + [game.static_features])
        target, magnitude = interest(game, i, self.labels)
        weight = np.float32(1.0 + np.log1p(magnitude / 100.0))
        return torch.from_numpy(features), torch.from_numpy(target), torch.tensor(weight)


@dataclass(frozen=True)
class GameEntry:
    """An extracted game on disk, known well enough to plan an epoch without loading it."""
    path: Path
    frames: int  # len(Game.frames)

    @property
    def name(self) -> str:
        return self.path.name

    def samples(self, window: WindowConfig, stride: int) -> int:
        return len(range(window.span, self.frames, stride))


def index_games(root: Path) -> list[GameEntry]:
    """Every extracted game under root that load_game accepts, without loading any."""
    entries = []
    for path in sorted(Path(root).iterdir()):
        meta_path = path / "meta.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text())
        last = last_unit_frame(path / "units.parquet")
        if fits_grid(meta) and last is not None:
            entries.append(GameEntry(path, last // meta["interval"] + 1))
    return entries


def last_unit_frame(path: Path) -> int | None:
    """Largest units.frame, from the Parquet footer's statistics where it has them."""
    parquet = pq.ParquetFile(path)
    if parquet.metadata.num_rows == 0:
        return None
    column = parquet.schema_arrow.get_field_index("frame")
    stats = [parquet.metadata.row_group(i).column(column).statistics for i in range(parquet.metadata.num_row_groups)]
    if all(s is not None and s.has_min_max for s in stats):
        return int(max(s.max for s in stats))
    return int(parquet.read(columns=["frame"]).column("frame").to_numpy().max())


class StreamingObserverDataset(IterableDataset):
    """ObserverDataset over more games than fit in memory.

    Each DataLoader worker takes its share of the games in a shuffled order, loads
    games_in_memory of them at a time and yields their samples shuffled together, so
    memory stays at workers * games_in_memory games. Call set_epoch() before each
    epoch to reshuffle; that needs non-persistent workers, which copy the dataset
    when the epoch's iterator is created.
    """

    def __init__(self, entries: list[GameEntry], window: WindowConfig = WindowConfig(),
                 labels: LabelConfig = LabelConfig(), stride: int = 1, games_in_memory: int = 8,
                 shuffle: bool = True, seed: int = 0):
        self.entries = entries
        self.window = window
        self.labels = labels
        self.stride = stride
        self.games_in_memory = games_in_memory
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

    def __len__(self):
        return sum(entry.samples(self.window, self.stride) for entry in self.entries)

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __iter__(self):
        worker = get_worker_info()
        worker_id, workers = (worker.id, worker.num_workers) if worker else (0, 1)
        order = np.arange(len(self.entries))
        if self.shuffle:
            order = np.random.default_rng([self.seed, self.epoch]).permutation(order)
        order = order[worker_id::workers]
        rng = np.random.default_rng([self.seed, self.epoch, worker_id])

        for start in range(0, len(order), self.games_in_memory):
            games = []
            for k in order[start:start + self.games_in_memory]:
                try:
                    games.append(load_game(self.entries[k].path))
                except Exception as error:  # one bad game shouldn't end a long run
                    print(f"skipping {self.entries[k].name}: {error}", flush=True)
            chunk = ObserverDataset(games, self.window, self.labels, self.stride)
            for item in (rng.permutation(len(chunk)) if self.shuffle else range(len(chunk))):
                yield chunk[item]


def split_games(games: list, val_every: int = 5) -> tuple[list, list]:
    """Deterministic split by game name so no frames from a validation game are seen in training.

    Takes anything with a .name: Games or GameEntries.
    """
    ordered = sorted(games, key=lambda g: g.name)
    val = ordered[::val_every]
    val_names = {g.name for g in val}
    train = [g for g in ordered if g.name not in val_names]
    return train, val

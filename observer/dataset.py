"""PyTorch dataset: a short history of map features -> where the action will be."""
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

from .features import N_DYNAMIC, N_STATIC, Game
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


def split_games(games: list[Game], val_every: int = 5) -> tuple[list[Game], list[Game]]:
    """Deterministic split by game so no frames from a validation game are seen in training."""
    ordered = sorted(games, key=lambda g: g.name)
    val = ordered[::val_every]
    val_names = {g.name for g in val}
    train = [g for g in ordered if g.name not in val_names]
    return train, val

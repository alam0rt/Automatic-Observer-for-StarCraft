import json
from collections import Counter

import pandas as pd
from torch.utils.data import DataLoader

from observer.dataset import ObserverDataset, StreamingObserverDataset, WindowConfig, index_games, split_games
from observer.features import load_game

ZERGLING = 37
WINDOW = WindowConfig(steps=2, spacing=1)

UNIT_COLUMNS = ["frame", "id", "player", "type", "x", "y", "hp", "shields", "energy", "value",
                "flying", "burrowed", "cloaked", "attacking", "under_attack", "completed", "order", "order_x", "order_y"]
EVENT_COLUMNS = ["frame", "event", "id", "player", "type", "x", "y", "value"]


def write_game(root, name, frames, interval=8, size=64, offset=0):
    """An extracted game on disk, laid out as sc-extract + observer.extract write it.

    One zergling walks a cell per sampled frame, so every sample's features differ.
    """
    path = root / name
    path.mkdir()
    (path / "meta.json").write_text(json.dumps({"map_width_tiles": size, "map_height_tiles": size, "interval": interval}))
    (path / "map.json").write_text(json.dumps(
        {"ground_height": [0] * size * size, "walkable_minitiles": [16] * size * size, "buildable": [1] * size * size}))
    (path / "unit_types.json").write_text(json.dumps({str(ZERGLING): {"building": False, "worker": False}}))
    units = [[f * interval, 1, 0, ZERGLING, 64 * ((f + offset) % 32) + 8, 64 * offset + 8, 35, 0, 0, 50,
              0, 0, 0, 0, 0, 1, 2, 0, 0] for f in range(frames)]
    pd.DataFrame(units, columns=UNIT_COLUMNS).to_parquet(path / "units.parquet", index=False)
    events = [[interval * 2, "destroy", 1, 0, ZERGLING, 500, 500, 50]]
    pd.DataFrame(events, columns=EVENT_COLUMNS).to_parquet(path / "events.parquet", index=False)
    return path


def fingerprints(items):
    return [features.numpy().tobytes() + target.numpy().tobytes() for features, target, _ in items]


def make_games(tmp_path):
    for k, frames in enumerate((12, 20, 7, 15, 9)):
        write_game(tmp_path, f"game_{k}", frames, offset=k)
    return index_games(tmp_path)


def test_index_matches_loaded_games_and_skips_oversized_maps(tmp_path):
    write_game(tmp_path, "small", 20)
    write_game(tmp_path, "huge", 20, size=200)

    entries = index_games(tmp_path)

    assert [e.name for e in entries] == ["small"]
    assert entries[0].frames == len(load_game(entries[0].path).frames)


def test_loaded_games_keep_only_small_columns(tmp_path):
    game = load_game(write_game(tmp_path, "g", 10))

    assert list(game.units.columns) == ["frame", "player", "type", "x", "y", "value", "flying", "attacking", "under_attack"]
    assert game.units["x"].dtype == "int16" and game.units["flying"].dtype == bool


def test_streaming_yields_the_same_samples_as_loading_everything(tmp_path):
    entries = make_games(tmp_path)
    eager = ObserverDataset([load_game(e.path) for e in entries], WINDOW, stride=2)
    streaming = StreamingObserverDataset(entries, WINDOW, stride=2, games_in_memory=2)

    assert len(streaming) == len(eager)
    assert Counter(fingerprints(streaming)) == Counter(fingerprints(eager))


def test_workers_share_the_games_without_overlap(tmp_path):
    entries = make_games(tmp_path)
    streaming = StreamingObserverDataset(entries, WINDOW, games_in_memory=2)

    items = list(DataLoader(streaming, batch_size=None, num_workers=2))

    assert Counter(fingerprints(items)) == Counter(fingerprints(StreamingObserverDataset(entries, WINDOW)))


def test_epochs_reshuffle(tmp_path):
    streaming = StreamingObserverDataset(make_games(tmp_path), WINDOW, games_in_memory=2)

    first = fingerprints(streaming)
    streaming.set_epoch(1)
    second = fingerprints(streaming)

    assert first != second and Counter(first) == Counter(second)


def test_split_works_on_index_entries(tmp_path):
    train, val = split_games(make_games(tmp_path))

    assert [e.name for e in val] == ["game_0"]
    assert {e.name for e in train} == {"game_1", "game_2", "game_3", "game_4"}

"""Run a trained model over an extracted game and write a camera track.

    python -m observer.observe runs/baseline/model.pt data/extracted/GAME_438CA8EB --out 438CA8EB.rep.vpd

The .vpd output (frame, vpx, vpy; viewport top-left in pixels) is the format the
original evaluation scripts in src/ read.
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .camera import CameraConfig, camera_track, coverage
from .dataset import ObserverDataset, WindowConfig
from .features import TILE_PX, load_game
from .labels import LabelConfig
from .model import InterestNet, log_distribution


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("game", type=Path, help="an extracted game directory")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    window = WindowConfig(**checkpoint["window"])
    labels = LabelConfig(**checkpoint["labels"])
    model = InterestNet(window.in_channels(), width=checkpoint["width"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    game = load_game(args.game)
    dataset = ObserverDataset([game], window, labels)
    heats, targets = [], []
    with torch.no_grad():
        for features, target, _ in DataLoader(dataset, args.batch_size):
            heats.append(log_distribution(model(features.to(device)).float()).exp().cpu().numpy())
            targets.append(target.numpy())
    heats, targets = np.concatenate(heats), np.concatenate(targets)

    frames = game.frames[window.span:]
    track = camera_track(heats, frames, (game.width * TILE_PX, game.height * TILE_PX), CameraConfig())
    track.to_csv(args.out, index=False)

    achieved = np.mean([coverage(t, np.array([x, y])) for t, x, y in zip(targets, track["vpx"], track["vpy"])])
    print(f"wrote {len(track)} viewports to {args.out}; camera track coverage {achieved:.3f}")


if __name__ == "__main__":
    main()

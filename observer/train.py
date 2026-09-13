"""Train the interest heatmap model on extracted games.

    python -m observer.train data/extracted runs/baseline --epochs 8

Reports, on held-out games, how much of the best achievable view the model's chosen
view captures ("coverage"), next to a rule-based baseline that points the camera at
units currently attacking or under attack.
"""
import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .camera import best_view, coverage
from .dataset import StreamingObserverDataset, WindowConfig, index_games, split_games
from .features import DYNAMIC_CHANNELS, N_DYNAMIC, PLAYER_SLOTS
from .labels import LabelConfig
from .model import InterestNet, interest_loss, log_distribution


def heuristic_heat(features: torch.Tensor, window: WindowConfig) -> np.ndarray:
    """Rule-based observer: engaged units in the latest step, else armies."""
    last = (window.steps - 1) * N_DYNAMIC
    per_player = len(DYNAMIC_CHANNELS)
    engaged = sum(features[:, last + s * per_player + 4] for s in range(PLAYER_SLOTS))
    army = sum(features[:, last + s * per_player + k] for s in range(PLAYER_SLOTS) for k in (0, 1))
    quiet = engaged.flatten(1).sum(dim=1) <= 0
    heat = torch.where(quiet[:, None, None], army, engaged)
    return heat.cpu().numpy()


def autocast_dtype(precision: str, device: torch.device) -> torch.dtype | None:
    """The dtype to autocast to, or None for plain float32.

    auto never picks float16: on Turing GTX 16xx / T1000 cards cuDNN's float16 kernels
    turn this model's output into NaN from the first batch. bfloat16 (Ampere and newer)
    is safe and needs no gradient scaling.
    """
    if device.type != "cuda" or precision == "fp32":
        return None
    if precision == "auto":
        return torch.bfloat16 if torch.cuda.is_bf16_supported(including_emulation=False) else None
    return {"bf16": torch.bfloat16, "fp16": torch.float16}[precision]


@torch.no_grad()
def evaluate(model, loader, window, device, dtype=None) -> dict:
    model.eval()
    loss_sum = weight_sum = 0.0
    model_cover, rule_cover = [], []
    for features, target, weight in loader:
        features, target, weight = features.to(device), target.to(device), weight.to(device)
        with torch.autocast(device.type, dtype=dtype, enabled=dtype is not None):
            logits = model(features)
        loss_sum += interest_loss(logits.float(), target, weight).item() * weight.sum().item()
        weight_sum += weight.sum().item()

        probs = log_distribution(logits.float()).exp().cpu().numpy()
        targets = target.cpu().numpy()
        for p, h, t in zip(probs, heuristic_heat(features, window), targets):
            model_cover.append(coverage(t, best_view(p)[0]))
            rule_cover.append(coverage(t, best_view(h)[0]))

    return {
        "loss": loss_sum / max(weight_sum, 1e-9),
        "coverage": float(np.mean(model_cover)),
        "rule_coverage": float(np.mean(rule_cover)),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("data", type=Path, help="directory of extracted games")
    parser.add_argument("out", type=Path, help="run directory for checkpoints and metrics")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--steps", type=int, default=WindowConfig.steps)
    parser.add_argument("--spacing", type=int, default=WindowConfig.spacing)
    parser.add_argument("--stride", type=int, default=2, help="use every Nth sampled frame for training")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--games-in-memory", type=int, default=8, help="games each data worker holds at once")
    parser.add_argument("--val-games", type=int, default=200, help="cap on held-out games evaluated each epoch (0: all)")
    parser.add_argument("--precision", choices=("auto", "bf16", "fp16", "fp32"), default="auto",
                        help="auto: bf16 where the GPU supports it, else fp32")
    args = parser.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = autocast_dtype(args.precision, device)
    window = WindowConfig(steps=args.steps, spacing=args.spacing)
    labels = LabelConfig()

    # Games are streamed from disk: thousands of them don't fit in memory at once.
    train_games, val_games = split_games(index_games(args.data))
    if args.val_games:
        val_games = val_games[:args.val_games]
    print(f"{len(train_games)} training games, {len(val_games)} validation games, device {device}, "
          f"precision {str(dtype).removeprefix('torch.') if dtype else 'float32'}")
    train_set = StreamingObserverDataset(train_games, window, labels, stride=args.stride,
                                         games_in_memory=args.games_in_memory, shuffle=True)
    val_set = StreamingObserverDataset(val_games, window, labels, stride=4,
                                       games_in_memory=args.games_in_memory, shuffle=False)
    train_loader = DataLoader(train_set, args.batch_size, num_workers=args.workers,
                              pin_memory=device.type == "cuda", drop_last=True)
    val_loader = DataLoader(val_set, args.batch_size, num_workers=args.workers, persistent_workers=args.workers > 0)
    # Workers drop their own partial batches, so an epoch can come up a few batches
    # short of this; it is never longer.
    steps_per_epoch = len(train_set) // args.batch_size
    print(f"{len(train_set)} training samples ({steps_per_epoch} steps per epoch), {len(val_set)} validation samples")

    model = InterestNet(window.in_channels(), width=args.width).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimizer, args.lr, total_steps=args.epochs * steps_per_epoch)
    scaler = torch.amp.GradScaler(enabled=dtype == torch.float16)

    args.out.mkdir(parents=True, exist_ok=True)
    best = -1.0
    for epoch in range(args.epochs):
        model.train()
        train_set.set_epoch(epoch)
        start, running, steps = time.time(), 0.0, 0
        for step, (features, target, weight) in zip(range(steps_per_epoch), train_loader):
            features, target, weight = features.to(device), target.to(device), weight.to(device)
            with torch.autocast(device.type, dtype=dtype, enabled=dtype is not None):
                logits = model(features)
            loss = interest_loss(logits.float(), target, weight)
            if not torch.isfinite(loss) and dtype != torch.float16:  # fp16's scaler skips the odd overflow itself
                raise SystemExit(f"non-finite loss at epoch {epoch + 1} step {steps + 1}; try --precision fp32")

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            schedule.step()
            running += loss.item()
            steps += 1
            if steps % 1000 == 0:
                print(f"  epoch {epoch + 1} step {steps}/{steps_per_epoch} loss {running / steps:.4f} "
                      f"{steps * args.batch_size / (time.time() - start):.0f} samples/s", flush=True)

        metrics = {"epoch": epoch + 1, "train_loss": running / max(steps, 1), **evaluate(model, val_loader, window, device, dtype),
                   "seconds": round(time.time() - start, 1)}
        print(json.dumps(metrics))
        with open(args.out / "metrics.jsonl", "a") as f:
            f.write(json.dumps(metrics) + "\n")

        if metrics["coverage"] > best:
            best = metrics["coverage"]
            torch.save({
                "model": model.state_dict(),
                "width": args.width,
                "window": asdict(window),
                "labels": asdict(labels),
                "metrics": metrics,
            }, args.out / "model.pt")


if __name__ == "__main__":
    main()

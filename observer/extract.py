"""Run sc-extract over a directory of replays and convert its CSV output to Parquet.

    python -m observer.extract data/replays data/extracted --jobs 6
"""
import argparse
import os
import re
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

TABLES = ("units", "events", "fights")


def find_replays(root: Path) -> dict[str, Path]:
    """One replay per game, keyed by game name.

    sc-docker saves a replay per player (player_0.rep, player_1.rep) in each game
    directory. They record the same game, so only the first is used and the game
    is named after the directory. Any other replay (e.g. StarData, which shards
    many games into numbered directories) is its own game, named by file stem.
    """
    games = {}
    for replay in sorted(root.rglob("*.rep")):
        per_player = re.fullmatch(r"player_\d+", replay.stem) and replay.parent != root
        game = replay.parent.name if per_player else replay.stem
        games.setdefault(game, replay)
    return games


def extract(replay: Path, out: Path, extractor: str, extra_args: list[str]) -> str | None:
    """Extract one replay into out/. Returns an error message, or None on success."""
    partial = out.with_name(out.name + ".partial")
    shutil.rmtree(partial, ignore_errors=True)

    proc = subprocess.run([extractor, str(replay), str(partial), *extra_args], capture_output=True, text=True)
    if proc.returncode != 0:
        tail = "\n".join(proc.stderr.strip().splitlines()[-5:])
        return f"sc-extract exited {proc.returncode}: {tail}"

    for table in TABLES:
        csv = partial / f"{table}.csv"
        if csv.exists():
            pd.read_csv(csv).to_parquet(partial / f"{table}.parquet", index=False)
            csv.unlink()

    shutil.rmtree(out, ignore_errors=True)
    partial.rename(out)
    return None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("replays", type=Path, help="directory searched recursively for .rep files")
    parser.add_argument("out", type=Path, help="one subdirectory per game is written here")
    parser.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    parser.add_argument("--extractor", default=shutil.which("sc-extract") or "sc-extract")
    parser.add_argument("--force", action="store_true", help="re-extract games that already have output")
    parser.add_argument("extra", nargs="*", help="passed through to sc-extract, after --")
    args = parser.parse_args(argv)

    games = find_replays(args.replays)
    todo = {g: r for g, r in games.items() if args.force or not (args.out / g / "meta.json").exists()}
    print(f"{len(games)} games, {len(games) - len(todo)} already extracted, {len(todo)} to do")
    args.out.mkdir(parents=True, exist_ok=True)

    failures = 0
    with ThreadPoolExecutor(args.jobs) as pool:
        futures = {pool.submit(extract, replay, args.out / game, args.extractor, args.extra): game
                   for game, replay in todo.items()}
        for future in as_completed(futures):
            game = futures[future]
            error = future.result()
            if error:
                failures += 1
                print(f"FAIL {game}: {error}", file=sys.stderr)
            else:
                print(f"ok   {game}")

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

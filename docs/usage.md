# Usage: adding replays, extracting, training

Run everything from the repo root.

## 1. Enter the environment

```sh
nix develop
```

This provides Python with torch, `sc-extract`, and `OPENBW_MPQ_PATH` already set. The
first run downloads torch and the StarCraft 1.16.1 data; later runs start instantly.

## 2. Add replays

Put `.rep` files anywhere under `data/replays/`; subfolders are fine.

```sh
# from the bot ladder on sauron
rsync -a --include='*/' --include='*.rep' --include='result.json' --exclude='*' \
  sauron:/srv/data/bwapi/.scbw/games/ data/replays/

# or any other replays you've downloaded
cp ~/Downloads/*.rep data/replays/
```

Replays must be from patch 1.16.1: bot ladders, SSCAIT, or StarData. Replays saved by
StarCraft: Remastered won't load.

## 3. Extract

```sh
python -m observer.extract data/replays data/extracted --jobs 6
```

- **Re-runs are safe:** games already in `data/extracted/` are skipped, so run it again
  after adding replays. Use `--force` to redo everything.
- **One replay per game:** where a game folder has a replay from each player
  (`player_0.rep`, `player_1.rep`), only the first is used.
- **Failures** print as `FAIL <game>: <reason>` and the rest keep going.

To test a single replay:

```sh
sc-extract data/replays/GAME_438CA8EB/player_0.rep /tmp/test-game
```

## 4. Train

```sh
python -m observer.train data/extracted runs/baseline --epochs 8
```

- **Output:** one line per epoch. `coverage` is the model's score and `rule_coverage`
  is the rule-based camera's score on the same held-out games. The model is only
  useful once `coverage` beats `rule_coverage`.
- **Saved files:** the best checkpoint is `runs/baseline/model.pt`, and the history is
  `runs/baseline/metrics.jsonl`.
- **Useful flags:**
  - `--batch-size`: lower it if you run out of GPU memory
  - `--workers`: data-loading processes (roughly half your CPU threads)
  - `--steps` / `--spacing`: how much recent history the model sees
- **Start fresh:** use a new run directory each time, because `metrics.jsonl` appends
  rather than overwrites.

## 5. Generate a camera track

```sh
python -m observer.observe runs/baseline/model.pt data/extracted/GAME_438CA8EB --out GAME_438CA8EB.rep.vpd
```

This writes `frame,vpx,vpy` (the viewport's top-left corner in pixels), which is the
format the original `src/` evaluation scripts read.

## 6. Explore the data (optional)

```sh
nix shell nixpkgs#duckdb -c duckdb -c "
  SELECT count(*) FROM 'data/extracted/*/fights.parquet'"
```

Each game folder holds `units`, `events` and `fights` as Parquet, plus `meta.json`,
`map.json` and `unit_types.json`. See the README for what each contains.

## 7. Run tests

```sh
pytest tests
```

## Running on sauron

The same steps work there. The training set is the
[StarData](https://github.com/TorchCraft/StarData) replays, unpacked under
`/srv/share/public/games/StarCraft/stardata/` (the ZFS pool, not the root NVMe). Extract
next to them and train on the A1000:

```sh
S=/srv/share/public/games/StarCraft/stardata
python -m observer.extract $S/stardata_original_replays $S/extracted --jobs 32
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
  python -m observer.train $S/extracted runs/baseline --workers 16 --batch-size 128
```

The bot-ladder replays in `/srv/data/bwapi/.scbw/games` can be extracted the same way.

- Run training inside `tmux` so it survives a disconnect.
- If the `llama-cpp` service is running, stop it first, or it will compete for GPU
  memory.
- `CUDA_DEVICE_ORDER=PCI_BUS_ID` makes CUDA number the GPUs the same way `nvidia-smi`
  does.

## How much data

| Goal | Games |
| --- | --- |
| Debugging the pipeline | ~20 |
| First trustworthy baseline | 200–500, all six matchups, 10+ maps |
| Solid model | 2,000–5,000 |

Variety (maps, matchups, play styles) matters more than raw count. As data grows,
retrain on 25%, 50% and 100% of it against the same held-out games: if coverage is still
rising, get more replays; if it has flattened, the model or labels are the limit.

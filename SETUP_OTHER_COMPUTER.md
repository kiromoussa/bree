# Running BREE on another computer

Written 2026-10-07. Two repos hold everything except the large rendered data.

| Repo | What | Clone to |
|---|---|---|
| `github.com/kiromoussa/bree` (public) | the vision pipeline, benchmark, training and real footage scripts, trained weights | `~/bree-vision` |
| `github.com/kiromoussa/bree-workspace` (private) | the browser store simulator, camera layouts, Isaac Sim kit, research, real footage kit | `~/bree` |

Keep those two folder names. The benchmark renderer reads layouts from `~/bree/software/shared/layouts`.

## 1. Install

Needs: Python 3.11 or newer, [uv](https://github.com/astral-sh/uv), Node 18 or newer, Google Chrome, git.

```bash
git clone https://github.com/kiromoussa/bree.git ~/bree-vision
git clone https://github.com/kiromoussa/bree-workspace.git ~/bree
cd ~/bree-vision
make setup                      # venv, pinned dependencies, public pretrained models (YOLO26, pose)
sh scripts/restore_weights.sh   # puts the weights trained here (weights/) where the pipeline looks
mkdir -p ~/bree-tools && cd ~/bree-tools && npm init -y && npm install playwright
echo 'export BREE_PLAYWRIGHT=~/bree-tools/node_modules/playwright' >> ~/.zshrc && source ~/.zshrc
```

The re-ID model (DINOv2) is exported on first use and needs network once.

## 2. Check it works

```bash
cd ~/bree-vision
make test-fast                  # unit tests, no model weights needed
make test                       # full suite; the MOT16 guard needs data/mot16 (see section 5)
```

## 3. The browser simulator

```bash
cd ~/bree/software/sim-prototype && python3 -m http.server 8765
# open http://127.0.0.1:8765/  (needs network once: three.js loads from a CDN)
```

It opens with the recommended 45 camera layout. Layout tools: `tools/README.md` in that folder.

## 4. The simulated benchmark

The rendered clips are not in git (about 80 GB). Two ways to get them:

**Re-render (same seeds give the same scenarios; allow 10 to 45 minutes per clip):**

```bash
cd ~/bree-vision
.venv/bin/python scripts/bench/render_split.py dev      # 6 clips
.venv/bin/python scripts/bench/render_split.py test     # 6 clips
.venv/bin/python scripts/bench/render_split.py dev2     # 20 clips, the hard set
.venv/bin/python -m bree.sim.bench dev2                 # scorecard in results/bench_dev2.md
```

**Or copy them from this Mac over the local network (Remote Login on in System Settings, Sharing):**

```bash
rsync -a --progress kiromoussa@Kiros-MacBook-Pro.local:bree-vision/data/synth/bench/ ~/bree-vision/data/synth/bench/
```

Sizes on this Mac: dev 6.9 GB, test 7.1 GB, dev2 41 GB, checkpoint 15 GB, train 10 GB, allcams 2.4 GB. Start with `dev`.

Commands and the truth format: `scripts/bench/README.md`. Results so far: `REPORT.md` and `results/bench_*.md`. All of it is simulated.

## 5. Public datasets (optional, for the older benches)

`scripts/download_data.sh` fetches what can be fetched; `data/README.md` lists the ones that need a manual download and their licences. On this Mac: UCF-Crime 10 GB, RetailS 5.1 GB, MOT16 3.7 GB, MERL 3.4 GB.

## 6. Real footage

`~/bree/pilot/real-footage-kit/README.md` is the checklist. The scripts are in `scripts/realkit/`.

## 7. Isaac Sim (needs an NVIDIA RTX GPU)

`~/bree/software/isaac-sim/README.md`. Never run end to end; expect fixes on the first run.

## What is not in git

- Rendered clips and training tiles (`data/synth`, 123 GB) and run outputs (`out`, 69 GB): re-render or rsync.
- Public pretrained weights (`models/`): fetched by `make setup`.
- The virtual environment: rebuilt by `make setup`.

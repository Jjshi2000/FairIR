This repository contains the implementation of **“Inter-Group Knowledge Transfer and Representation Distillation for Fair Recommendation.”**
It includes:

* Source code for two backbones (**BPR** and **GCCF**)
* Pretrained **model weights** (managed by **Git LFS**)
* Datasets (under `data/`)

---

## Get the Code & Weights (Git LFS)

Because checkpoints are large, **Git LFS** is required.

```bash
# ── 1) Install Git LFS
git lfs install

# ── 2) Clone and fetch LFS objects
git clone https://github.com/Jjshi2000/FairIR.git
cd FairIR
git lfs pull
```

---

## Quick Start — Evaluate Released Models
```bash
# ==== MovieLens-1M ====
# BPR
( cd FairIR_BPR_ML && python test.py --runid FairIR_BPR_ML )
# GCCF
( cd FairIR_GCCF_ML && python test.py --runid FairIR_GCCF ) 

# ==== LastFM ====
# BPR
( cd FairIR_BPR_LastFM && python test.py --runid FairIR_BPR_lastfm )
# GCCF
( cd FairIR_GCCF_LastFM && python test.py --runid FairIR_GCCF_lastfm )

# ==== Tmall ====
# BPR
( cd FairIR_BPR_Tmall && python test.py --runid FairIR_BPR_Tmall )
# GCCF
( cd FairIR_GCCF_Tmall && python test.py --runid FairIR_GCCF_Tmall )
```

---

## Reproduce Experiments — Training

> Defaults in the code are tuned to strong settings. Adjust if your GPU memory is limited.

### MovieLens-1M (example provided)

```bash
# MovieLens-1M + BPR (example)
( cd FairIR_BPR_ML && python main.py )
```

### The other 5 training runs

```bash
# 1) MovieLens-1M + GCCF
( cd FairIR_GCCF_ML && python main.py )

# 2) LastFM + BPR
( cd FairIR_BPR_LastFM && python main.py )

# 3) LastFM + GCCF
( cd FairIR_GCCF_LastFM && python main.py )

# 4) Tmall + BPR
( cd FairIR_BPR_Tmall && python main.py )

# 5) Tmall + GCCF
( cd FairIR_GCCF_Tmall && python main.py )
```

---

## Troubleshooting

* **CUDA out of memory**
  Reduce the number of negative samples **`K`** (e.g., `--K 20` or lower, depending on your GPU).
  If `K` is not exposed via CLI, change its default in the corresponding code.
---

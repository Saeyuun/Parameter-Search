# Pitch Imperfect — Prosody-Based Audio Deepfake Detector
### Replication of Warren et al. (2025)

---

## Overview

This project replicates the methodology of **Warren et al. (2025) "Pitch Imperfect: Detecting Audio Deepfakes Through Acoustic Prosodic Analysis"**.

It extracts 6 prosodic features per audio window using **Praat** (via Parselmouth), then trains an **LSTM-based classifier** to distinguish bonafide from spoofed speech.

The key step is a **Praat parameter search** — finding the optimal combination of:
- `silence_threshold`
- `octave_cost`
- `octave_jump_cost`
- `voiced_unvoiced_cost`
- `window_size` (50 / 100 / 200 / 500 ms)

The search uses **Optuna TPE (Bayesian optimization)** and supports **sharding across multiple devices** so the 2,200-trial search can be parallelized without overlap.

---

## Dataset

**ASVspoof2019 Logical Access (LA)**

Download from: https://datashare.ed.ac.uk/handle/10283/3336

Expected folder structure:
```
J:\thesis\dataset\LA\
├── ASVspoof2019_LA_cm_protocols\
│   ├── ASVspoof2019.LA.cm.train.trn.txt
│   └── ASVspoof2019.LA.cm.dev.trl.txt
├── ASVspoof2019_LA_train\flac\   ← training audio (.flac)
└── ASVspoof2019_LA_dev\flac\     ← validation audio (.flac)
```

| Split | Bonafide | Spoof  |
|-------|----------|--------|
| Train | 2,580    | 22,800 |
| Dev   | 2,548    | 22,296 |

---

## Dependencies

### Python version
Python 3.9 or 3.10 recommended (TensorFlow 2.x compatibility).

### Install all dependencies
```bash
pip install tensorflow parselmouth optuna numpy pandas scikit-learn scipy joblib tqdm
```

### Package list

| Package | Purpose |
|---------|---------|
| `tensorflow` | LSTM model training |
| `parselmouth` | Python interface to Praat (feature extraction) |
| `optuna` | Bayesian (TPE) parameter search |
| `numpy` | Numerical operations |
| `pandas` | CSV checkpointing and results handling |
| `scikit-learn` | MinMaxScaler, train/test split, metrics |
| `scipy` | EER calculation (brentq, interp1d) |
| `joblib` | Parallel Praat feature extraction |
| `tqdm` | Progress bars |

---

## Running the Parameter Search

### Single device (no sharding)
```bash
python py_paramsearch.py
```
Runs all 2,200 trials on one machine. Saves checkpoint to `search_checkpoint_shard_0.csv` and best params to `best_praat_params_shard_0.json`.

---

### Multiple devices (sharding)

Assign each device a unique `--shard` number starting from 0. All devices must use the same `--num-shards` value.

**Example: 4 devices**

| Device | Command |
|--------|---------|
| Device A | `python py_paramsearch.py --shard 0 --num-shards 4` |
| Device B | `python py_paramsearch.py --shard 1 --num-shards 4` |
| Device C | `python py_paramsearch.py --shard 2 --num-shards 4` |
| Device D | `python py_paramsearch.py --shard 3 --num-shards 4` |

Each device explores a **different region** of the parameter space (different Optuna seed per shard). No two devices run the same trials.

Each device saves:
- `search_checkpoint_shard_N.csv` — all completed trials for that shard
- `stage2_results_shard_N.csv` — top-20 revalidated candidates
- `best_praat_params_shard_N.json` — best params found by that shard

**Resuming after a crash:** just re-run the same command. The script automatically detects the checkpoint file and resumes from the last completed trial.

---

### Merging results from all devices

Once all devices are done:

1. Copy all `search_checkpoint_shard_*.csv` files into one folder on any machine.
2. Run:

```bash
python py_paramsearch.py --merge --num-shards 4
```

This produces:
- `search_results_all_shards.csv` — all trials ranked by EER
- `best_praat_params.json` — the single best parameter set across all devices

---

### Skipping the search (use known params)

If you already have good parameters and want to skip straight to final model training:
```bash
python py_paramsearch.py --skip-search
```
Edit the fallback `best_params` dict inside the `__main__` block to set your known values.

---

## Command-line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--shard N` | `0` | Zero-based shard index for this device |
| `--num-shards K` | `1` | Total number of devices / shards |
| `--merge` | off | Merge all shard CSVs and exit |
| `--skip-search` | off | Skip search, use hardcoded fallback params |

---

## Output Files

| File | Description |
|------|-------------|
| `search_checkpoint_shard_N.csv` | Per-trial results for shard N (auto-saved after every trial) |
| `stage2_results_shard_N.csv` | Top-20 revalidated candidates for shard N |
| `best_praat_params_shard_N.json` | Best params found by shard N |
| `search_results_all_shards.csv` | Merged results from all shards (after `--merge`) |
| `best_praat_params.json` | Final best params (after `--merge`) |

---

## Search Configuration

These constants at the top of `py_paramsearch.py` control the search:

```python
N_SEARCH       = 2200   # Total number of trials across all shards
SEARCH_SAMPLES = 2000   # Audio files sampled per trial
SEARCH_EPOCHS  = 200    # Epochs per trial (proxy Model A)
SEARCH_SEED    = 42     # Base random seed
```

---

## Reference

Warren et al. (2025). *Pitch Imperfect: Detecting Audio Deepfakes Through Acoustic Prosodic Analysis.* arXiv:2502.14726.

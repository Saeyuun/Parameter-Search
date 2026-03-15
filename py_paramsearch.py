#!/usr/bin/env python
# coding: utf-8

# # Pitch Imperfect — Prosody-Based Audio Deepfake Detector
# ### Replication of Warren et al. (2025)
#
# **All fixes applied:**
# 1. Correct dataset: ASVspoof2019 LA train/dev splits
# 2. Praat parameter search — 150 candidates × 200 epochs (Option A: reduced candidates, full epoch count matching paper)
# 3. Speed optimisations: audio pre-loaded into RAM once, parallel Praat extraction via joblib
# 4. Voiced/unvoiced frame gating via Praat pitch object
# 5. Native Praat `Get mean` / `Get standard deviation` for HNR
# 6. Class imbalance handled via `class_weight` in final training
# 7. Validation set wired into training with `ModelCheckpoint` on val AUC
#
# **Note on search scale:** The paper ran 2,200 candidates on a compute cluster (~220 GPU-hours).
# Option A runs 150 candidates at the paper's full 200 epochs each (~15-20 GPU-hours across sessions).
# The Phase 1 grid scan tightens the parameter bounds first, so 150 candidates in a safe region
# covers the space well enough to find a near-optimal configuration.


# ── Cell 1: Dependencies ──────────────────────────────────────────────────────
import gc
import glob
import json
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import random
import warnings
from functools import lru_cache

warnings.filterwarnings("ignore")
import logging

logging.getLogger("tensorflow").setLevel(logging.ERROR)

import numpy as np
import optuna
import pandas as pd
import parselmouth
import tensorflow as tf
from joblib import Parallel, delayed
from parselmouth.praat import call
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tensorflow.keras import layers

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── Global seeds ──────────────────────────────────────────────────────────────
GLOBAL_SEED = 42
random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)
tf.random.set_seed(GLOBAL_SEED)

print("Setup complete. TensorFlow:", tf.__version__)
print(f"Global random seed: {GLOBAL_SEED}")


# ── Cell 2: Prosody Feature Extractor ────────────────────────────────────────
# Accepts Praat params as constructor args so the search can vary them.
# Unvoiced windows are fully zeroed (paper's frame-level methodology).
# HNR uses Praat's native Get mean / Get standard deviation.


class ProsodyFeatureExtractor:
    """
    Extracts 6 prosodic features per window:
        [mean_F0, std_F0, jitter, shimmer, mean_HNR, std_HNR]
    """

    def __init__(
        self,
        min_pitch=75,
        max_pitch=500,
        time_step=0.01,
        window_size=0.1,
        silence_threshold=0.03,
        octave_cost=0.01,
        octave_jump_cost=0.35,
        voiced_unvoiced_cost=0.14,
    ):
        self.min_pitch = min_pitch
        self.max_pitch = max_pitch
        self.time_step = time_step
        self.window_size = window_size
        self.silence_threshold = silence_threshold
        self.octave_cost = octave_cost
        self.octave_jump_cost = octave_jump_cost
        self.voiced_unvoiced_cost = voiced_unvoiced_cost

    def _extract_from_sound(self, snd):
        """Core extraction from a pre-loaded parselmouth.Sound object."""
        pitch = call(
            snd,
            "To Pitch (ac)",
            self.time_step,
            self.min_pitch,
            15.0,
            "no",
            self.silence_threshold,
            0.45,
            self.octave_cost,
            self.octave_jump_cost,
            self.voiced_unvoiced_cost,
            self.max_pitch,
        )
        point_process = call(
            snd, "To PointProcess (periodic, cc)", self.min_pitch, self.max_pitch
        )
        harmonicity = call(
            snd,
            "To Harmonicity (cc)",
            self.time_step,
            self.min_pitch,
            self.silence_threshold,
            1.0,
        )
        duration = call(snd, "Get total duration")
        n_windows = max(1, int(duration / self.window_size))

        features = []
        for i in range(n_windows):
            start = i * self.window_size
            end = min((i + 1) * self.window_size, duration)

            # Voiced/unvoiced gating via pitch object
            voiced = [
                call(pitch, "Get value at time", t, "Hertz", "Linear")
                for t in np.arange(start, end, self.time_step)
            ]
            voiced = [v for v in voiced if v and not np.isnan(v) and v > 0]

            if not voiced:
                features.append([0.0] * 6)
                continue

            mean_f0 = float(np.mean(voiced))
            std_f0 = float(np.std(voiced))

            try:
                jitter = call(
                    point_process, "Get jitter (local)", start, end, 0.0001, 0.02, 1.3
                )
                jitter = 0.0 if (jitter is None or np.isnan(jitter)) else jitter
            except:
                jitter = 0.0

            try:
                shimmer = call(
                    [snd, point_process],
                    "Get shimmer (local)",
                    start,
                    end,
                    0.0001,
                    0.02,
                    1.3,
                    1.6,
                )
                shimmer = 0.0 if (shimmer is None or np.isnan(shimmer)) else shimmer
            except:
                shimmer = 0.0

            try:
                mean_hnr = call(harmonicity, "Get mean", start, end)
                mean_hnr = 0.0 if (mean_hnr is None or np.isnan(mean_hnr)) else mean_hnr
            except:
                mean_hnr = 0.0

            try:
                std_hnr = call(harmonicity, "Get standard deviation", start, end)
                std_hnr = 0.0 if (std_hnr is None or np.isnan(std_hnr)) else std_hnr
            except:
                std_hnr = 0.0

            features.append([mean_f0, std_f0, jitter, shimmer, mean_hnr, std_hnr])

        return np.array(features, dtype=np.float32) if features else None

    def extract_features(self, audio_path):
        """Load from disk and extract. Used during final training/eval."""
        try:
            snd = parselmouth.Sound(audio_path)
            return self._extract_from_sound(snd)
        except Exception:
            return None


# ── Cell 3: Dataset Builder, DataProcessor, Model ────────────────────────────


def make_dataset(features, labels, batch_size=32, shuffle=True):
    """
    tf.data pipeline with dynamic per-batch padding.
    Uses from_generator + padded_batch to handle variable-length sequences
    without Keras shape-locking issues.
    """
    labels = np.array(labels, dtype=np.float32)
    n = len(features)

    def generator():
        indices = np.arange(n)
        if shuffle:
            np.random.shuffle(indices)
        for i in indices:
            yield features[i].astype(np.float32), labels[i]

    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(None, 6), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.float32),
        ),
    )
    dataset = dataset.padded_batch(
        batch_size,
        padded_shapes=([None, 6], []),
        padding_values=(
            tf.constant(0.0, dtype=tf.float32),
            tf.constant(0.0, dtype=tf.float32),
        ),
    )
    return dataset.prefetch(tf.data.AUTOTUNE)


class DataProcessor:
    def __init__(self):
        self.scaler = MinMaxScaler()

    def process_dataset(self, audio_paths, labels, extractor):
        all_features, valid_labels = [], []
        print(f"Extracting features from {len(audio_paths)} files...")
        for path, label in tqdm(zip(audio_paths, labels), total=len(audio_paths)):
            feats = extractor.extract_features(path)
            if feats is not None and len(feats) > 0:
                all_features.append(feats)
                valid_labels.append(label)
        print(f"  -> {len(all_features)} files processed successfully.")
        return all_features, np.array(valid_labels)

    def scale_features(self, features, fit_scaler=True):
        flat = np.vstack(features)
        if fit_scaler:
            self.scaler.fit(flat)
        return [self.scaler.transform(f) for f in features]


def build_model_b(input_shape=(None, 6)):
    """Model B — paper Appendix (final training model)."""
    model = keras.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.LSTM(100, return_sequences=True, dropout=0.2),
            layers.BatchNormalization(),
            layers.LSTM(50, dropout=0.2),
            layers.BatchNormalization(),
            layers.Dense(50, activation="relu"),
            layers.Dropout(0.2),
            layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.AUC(name="auc")],
    )
    return model


# ── Cell 4: ASVspoof Data Loader ─────────────────────────────────────────────
# Handles both ASVspoof2019 LA and ASVspoof2021 DF metadata formats.


class ASVspoofLoader:
    def __init__(self, metadata_path, audio_dirs):
        self.metadata_path = metadata_path
        self.audio_dirs = audio_dirs

    def load_data(self, bonafide_limit=None, spoof_limit=None):
        print("Parsing metadata...")
        file_to_label = {}
        with open(self.metadata_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                filename = parts[1]
                label_str = parts[4]
                if label_str == "spoof":
                    file_to_label[filename] = 1
                elif label_str == "bonafide":
                    file_to_label[filename] = 0
        print(f"  -> {len(file_to_label)} metadata entries loaded.")

        paths, labels = [], []
        bonafide_count = spoof_count = 0

        print("Scanning audio directories...")
        for d in self.audio_dirs:
            if not os.path.exists(d):
                print(f"  WARNING: directory not found: {d}")
                continue
            for fpath in glob.glob(os.path.join(d, "*.flac")):
                if (
                    bonafide_limit is not None and bonafide_count >= bonafide_limit
                ) and (spoof_limit is not None and spoof_count >= spoof_limit):
                    break
                fname = os.path.splitext(os.path.basename(fpath))[0]
                if fname not in file_to_label:
                    continue
                label = file_to_label[fname]
                if label == 0 and (
                    bonafide_limit is None or bonafide_count < bonafide_limit
                ):
                    paths.append(fpath)
                    labels.append(label)
                    bonafide_count += 1
                elif label == 1 and (spoof_limit is None or spoof_count < spoof_limit):
                    paths.append(fpath)
                    labels.append(label)
                    spoof_count += 1

        print(
            f"  -> {len(paths)} files | Bonafide: {bonafide_count} | Spoof: {spoof_count}"
        )
        return paths, labels


# ── Cell 5: Praat Parameter Search (Option A) ─────────────────────────────────
#
# Option A: 150 candidates × 200 epochs (paper's exact epoch count).
# Speed optimisations applied:
#   [1] Audio pre-loaded into RAM once — eliminates disk I/O per trial.
#   [2] Parallel Praat extraction via joblib threads.
#   [3] EarlyStopping(patience=15) so trials that plateau finish faster.
#
# Paper used 2,200 candidates on a compute cluster. 150 candidates with
# Phase 1 bound-tightening covers the safe parameter region well enough
# to find a near-optimal configuration on a single Kaggle GPU session.

# ── Search budget ─────────────────────────────────────────────────────────────
SEARCH_SEED = 42
N_SEARCH = 2200  # candidates  (paper: 2,200 on a cluster)
SEARCH_SAMPLES = 2500  # files per class for the search pool
SEARCH_EPOCHS = 200  # epochs per proxy — matches paper exactly

random.seed(SEARCH_SEED)
np.random.seed(SEARCH_SEED)
tf.random.set_seed(SEARCH_SEED)


# ── Helpers ───────────────────────────────────────────────────────────────────
def calculate_eer(y_true, y_scores):
    fpr, tpr, _ = roc_curve(y_true, y_scores, pos_label=1)
    eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
    return eer * 100


@lru_cache(maxsize=6000)  # must be >= SEARCH_SAMPLES*2 so no files are evicted mid-search
def _load_sound_cached(path):
    """Load a single audio file, cached in RAM (LRU evicts oldest when full)."""
    try:
        return parselmouth.Sound(path)
    except Exception:
        return None


def preload_sounds(paths):
    """Warm the LRU cache — maxsize=6000 holds the full pool with no evictions."""
    print(f"Warming sound cache for {len(paths)} files (maxsize=6000)...")
    n_ok = 0
    for p in tqdm(paths):
        if _load_sound_cached(p) is not None:
            n_ok += 1
    print(f"  -> {n_ok}/{len(paths)} loaded successfully.")

    # Return a dict-like proxy so existing sound_map[p] calls still work
    class _CacheProxy:
        def __getitem__(self, key):
            return _load_sound_cached(key)

    return _CacheProxy()


def _extract_one(snd, params):
    """Single-file extraction from a pre-loaded Sound — called in parallel."""
    if snd is None:
        return None
    try:
        ws = params["window_size"]
        st = params["silence_threshold"]
        oc = params["octave_cost"]
        ojc = params["octave_jump_cost"]
        vuc = params["voiced_unvoiced_cost"]

        pitch = call(
            snd, "To Pitch (ac)", 0.01, 75, 15.0, "no", st, 0.45, oc, ojc, vuc, 500
        )
        pp = call(snd, "To PointProcess (periodic, cc)", 75, 500)
        harm = call(snd, "To Harmonicity (cc)", 0.01, 75, st, 1.0)
        dur = call(snd, "Get total duration")
        nw = max(1, int(dur / ws))

        features = []
        for i in range(nw):
            s = i * ws
            e = min((i + 1) * ws, dur)
            voiced = [
                call(pitch, "Get value at time", t, "Hertz", "Linear")
                for t in np.arange(s, e, 0.01)
            ]
            voiced = [v for v in voiced if v and not np.isnan(v) and v > 0]
            if not voiced:
                features.append([0.0] * 6)
                continue
            mean_f0 = float(np.mean(voiced))
            std_f0 = float(np.std(voiced))
            try:
                jitter = call(pp, "Get jitter (local)", s, e, 0.0001, 0.02, 1.3)
                jitter = 0.0 if (jitter is None or np.isnan(jitter)) else jitter
            except:
                jitter = 0.0
            try:
                shimmer = call(
                    [snd, pp], "Get shimmer (local)", s, e, 0.0001, 0.02, 1.3, 1.6
                )
                shimmer = 0.0 if (shimmer is None or np.isnan(shimmer)) else shimmer
            except:
                shimmer = 0.0
            try:
                mean_hnr = call(harm, "Get mean", s, e)
                mean_hnr = 0.0 if (mean_hnr is None or np.isnan(mean_hnr)) else mean_hnr
            except:
                mean_hnr = 0.0
            try:
                std_hnr = call(harm, "Get standard deviation", s, e)
                std_hnr = 0.0 if (std_hnr is None or np.isnan(std_hnr)) else std_hnr
            except:
                std_hnr = 0.0
            features.append([mean_f0, std_f0, jitter, shimmer, mean_hnr, std_hnr])

        return np.array(features, dtype=np.float32) if features else None
    except Exception:
        return None


def parallel_extract(sound_map, paths, params, n_jobs=-1):
    """Parallel extraction across all paths using pre-loaded Sounds."""
    return Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_extract_one)(sound_map[p], params) for p in paths
    )


# ── Phase 1: grid scan → tighten parameter bounds ─────────────────────────────
def grid_scan_bounds(sound_map, scan_paths, nan_threshold=0.25):
    """
    Sweep each Praat parameter through 14 values in [0,1] while holding
    the others at defaults. Retain only values whose NaN rate is acceptable.
    Returns a tightened PARAM_SPACE dict.
    """
    defaults = {
        "silence_threshold": 0.03,
        "octave_cost": 0.01,
        "octave_jump_cost": 0.35,
        "voiced_unvoiced_cost": 0.14,
        "window_size": 0.1,
    }
    sweep_values = [
        0.0,
        0.02,
        0.05,
        0.08,
        0.10,
        0.15,
        0.20,
        0.30,
        0.40,
        0.50,
        0.60,
        0.70,
        0.85,
        1.00,
    ]

    tightened = {}
    print(
        f"\n── Phase 1: grid scan ({len(scan_paths)} files, "
        f"NaN threshold ≤ {nan_threshold:.0%}) ──"
    )

    for param in [
        "silence_threshold",
        "octave_cost",
        "octave_jump_cost",
        "voiced_unvoiced_cost",
    ]:
        valid_vals = []
        for val in sweep_values:
            test_params = {**defaults, param: val}
            results = parallel_extract(sound_map, scan_paths, test_params)
            n_ok = sum(1 for r in results if r is not None)
            nan_ratio = 1.0 - n_ok / max(len(scan_paths), 1)
            if nan_ratio <= nan_threshold:
                valid_vals.append(val)

        if valid_vals:
            lo, hi = min(valid_vals), max(valid_vals)
            margin = max((hi - lo) * 0.05, 0.01)
            lo, hi = max(0.0, lo - margin), min(1.0, hi + margin)
        else:
            lo, hi = 0.0, 1.0

        tightened[param] = (round(lo, 4), round(hi, 4))
        print(f"  {param:30s}  → ({lo:.4f}, {hi:.4f})")

    tightened["window_size"] = [0.05, 0.1, 0.2, 0.5]
    return tightened


# ── Proxy model: Model A (paper Appendix) ─────────────────────────────────────
def _build_proxy_model():
    """
    Paper Appendix Model A:
        LSTM(64) → BN → LSTM(32) → BN → Dense(32, ReLU) → Dropout(0.2) → Dense(1, sigmoid)
    Used as the proxy during parameter search.
    """
    model = keras.Sequential(
        [
            layers.Input(shape=(None, 6)),
            layers.LSTM(64, return_sequences=True, dropout=0.2),
            layers.BatchNormalization(),
            layers.LSTM(32, dropout=0.2),
            layers.BatchNormalization(),
            layers.Dense(32, activation="relu"),
            layers.Dropout(0.2),
            layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        loss="binary_crossentropy",
    )
    return model


# ── Main search: Optuna TPE + optional sharding across devices ────────────────
def run_param_search(train_paths, train_labels, shard_id=0, num_shards=1):
    """
    Bayesian (Optuna TPE) parameter search with optional sharding.

    Sharding
    --------
    Split 2,200 total trials across devices with --shard / --num-shards:
        device 0:  python py_paramsearch.py --shard 0 --num-shards 4
        device 1:  python py_paramsearch.py --shard 1 --num-shards 4
        ...
    Each device writes its results to search_checkpoint_shard_<N>.csv.
    After all devices finish, merge with:
        python py_paramsearch.py --merge --num-shards 4

    Why Optuna TPE instead of random search
    ----------------------------------------
    TPE builds a probabilistic model of the EER surface and proposes candidates
    in regions likely to improve — typically converges 10-20x faster than random
    search for the same number of trials.

    Resume behaviour
    ----------------
    On restart the shard checkpoint is reloaded and injected back into the
    Optuna study so TPE continues from where it left off.
    """
    n_trials = N_SEARCH // num_shards + (N_SEARCH % num_shards if shard_id == 0 else 0)
    checkpoint_path = f"search_checkpoint_shard_{shard_id}.csv"

    print(f"\nShard {shard_id}/{num_shards}  →  {n_trials} trials")
    print(f"Checkpoint: {checkpoint_path}\n")

    # ── Balanced sub-sample ───────────────────────────────────────────────────
    bona_paths  = [p for p, l in zip(train_paths, train_labels) if l == 0][:SEARCH_SAMPLES]
    spoof_paths = [p for p, l in zip(train_paths, train_labels) if l == 1][:SEARCH_SAMPLES]
    search_paths  = bona_paths  + spoof_paths
    search_labels = [0] * len(bona_paths) + [1] * len(spoof_paths)
    print(f"Search pool: {len(bona_paths)} bonafide + {len(spoof_paths)} spoof = {len(search_paths)} files")

    # ── Pre-load audio into RAM once for all trials ───────────────────────────
    sound_map = preload_sounds(search_paths)

    # ── Phase 1: grid scan → tighten parameter bounds ────────────────────────
    scan_slice  = bona_paths[:50] + spoof_paths[:50]
    param_space = grid_scan_bounds(sound_map, scan_slice)

    # ── Stratified 75/25 split (shard-specific seed) ──────────────────────────
    # Different shards evaluate on slightly different val sets, which reduces
    # correlated noise and makes the merged results more robust.
    s_train_p, s_val_p, s_train_l, s_val_l = train_test_split(
        search_paths, search_labels,
        test_size=0.25, stratify=search_labels,
        random_state=SEARCH_SEED + shard_id,
    )
    print(f"Split → train: {len(s_train_p)}, val: {len(s_val_p)}\n")

    # ── Build Optuna study (TPE sampler, shard-specific seed) ─────────────────
    sampler = optuna.samplers.TPESampler(
        seed=SEARCH_SEED + shard_id,
        n_startup_trials=25,   # random warm-up before TPE activates
    )
    study = optuna.create_study(direction="minimize", sampler=sampler)

    # ── Resume: inject prior trials so TPE continues learning ─────────────────
    completed_results = []
    if os.path.exists(checkpoint_path):
        prior_df = pd.read_csv(checkpoint_path)
        completed_results = prior_df.to_dict("records")
        print(f"Resuming shard {shard_id}: {len(completed_results)} prior trials loaded.")
        dists = {
            "silence_threshold":    optuna.distributions.FloatDistribution(*param_space["silence_threshold"]),
            "octave_cost":          optuna.distributions.FloatDistribution(*param_space["octave_cost"]),
            "octave_jump_cost":     optuna.distributions.FloatDistribution(*param_space["octave_jump_cost"]),
            "voiced_unvoiced_cost": optuna.distributions.FloatDistribution(*param_space["voiced_unvoiced_cost"]),
            "window_size":          optuna.distributions.CategoricalDistribution([0.05, 0.1, 0.2, 0.5]),
        }
        for row in prior_df.itertuples(index=False):
            study.add_trial(optuna.trial.create_trial(
                params={
                    "silence_threshold":    float(row.silence_threshold),
                    "octave_cost":          float(row.octave_cost),
                    "octave_jump_cost":     float(row.octave_jump_cost),
                    "voiced_unvoiced_cost": float(row.voiced_unvoiced_cost),
                    "window_size":          float(row.window_size),
                },
                distributions=dists,
                value=float(row.eer),
            ))
        print(f"  → TPE warmed with {len(prior_df)} prior trials.")

    trials_remaining = n_trials - len(completed_results)
    if trials_remaining <= 0:
        print(f"Shard {shard_id} already complete.")
        results_df  = pd.DataFrame(completed_results)
        best_row    = results_df.loc[results_df["eer"].idxmin()]
        best_params = {k: best_row[k] for k in ["silence_threshold", "octave_cost",
                                                  "octave_jump_cost", "voiced_unvoiced_cost",
                                                  "window_size"]}
        return best_params, results_df

    print(f"Running {trials_remaining} remaining trials ...\n")

    # ── Objective (called by Optuna for each trial) ───────────────────────────
    t_counter = [len(completed_results)]

    def objective(trial):
        t_idx = t_counter[0]
        t_counter[0] += 1

        params = {
            "silence_threshold":    trial.suggest_float("silence_threshold",    *param_space["silence_threshold"]),
            "octave_cost":          trial.suggest_float("octave_cost",          *param_space["octave_cost"]),
            "octave_jump_cost":     trial.suggest_float("octave_jump_cost",     *param_space["octave_jump_cost"]),
            "voiced_unvoiced_cost": trial.suggest_float("voiced_unvoiced_cost", *param_space["voiced_unvoiced_cost"]),
            # Categorical so TPE learns which window sizes produce better EER
            "window_size":          trial.suggest_categorical("window_size", [0.05, 0.1, 0.2, 0.5]),
        }

        # Feature extraction (parallel, pre-loaded sounds)
        tr_raw   = parallel_extract(sound_map, s_train_p, params)
        tr_feats = [f for f in tr_raw  if f is not None and len(f) > 0]
        tr_labs  = [l for f, l in zip(tr_raw,  s_train_l) if f is not None and len(f) > 0]
        nan_ratio = 1.0 - len(tr_feats) / max(len(s_train_p), 1)
        if nan_ratio > 0.30:
            print(f"  [s{shard_id}|t{t_idx+1:4d}]  SKIP (nan={nan_ratio:.0%})")
            raise optuna.exceptions.TrialPruned()

        va_raw   = parallel_extract(sound_map, s_val_p, params)
        va_feats = [f for f in va_raw  if f is not None and len(f) > 0]
        va_labs  = [l for f, l in zip(va_raw,  s_val_l) if f is not None and len(f) > 0]
        if len(va_feats) < 20:
            print(f"  [s{shard_id}|t{t_idx+1:4d}]  SKIP (too few val samples)")
            raise optuna.exceptions.TrialPruned()

        # Scale
        proc_tmp  = DataProcessor()
        tr_scaled = proc_tmp.scale_features(tr_feats, fit_scaler=True)
        va_scaled = proc_tmp.scale_features(va_feats, fit_scaler=False)
        tr_gen    = make_dataset(tr_scaled, np.array(tr_labs), batch_size=32, shuffle=True)
        va_gen    = make_dataset(va_scaled, np.array(va_labs), batch_size=32, shuffle=False)

        # Train proxy (Model A, paper Appendix) — fixed epochs, no class_weight
        tf.random.set_seed(SEARCH_SEED + shard_id * 10000 + t_idx)
        model    = _build_proxy_model()
        model.fit(tr_gen, epochs=SEARCH_EPOCHS, verbose=0)
        y_scores = model.predict(va_gen, verbose=0).flatten()

        keras.backend.clear_session()
        gc.collect()

        try:
            eer = calculate_eer(np.array(va_labs), y_scores)
        except Exception:
            raise optuna.exceptions.TrialPruned()

        # Save after every trial so any crash loses at most one result
        completed_results.append({**params, "eer": eer, "nan_ratio": nan_ratio,
                                   "n_train": len(tr_feats), "n_val": len(va_feats)})
        pd.DataFrame(completed_results).sort_values("eer").to_csv(checkpoint_path, index=False)

        best_so_far = min(r["eer"] for r in completed_results)
        marker = "  ← NEW BEST" if eer <= best_so_far else ""
        print(f"  [s{shard_id}|t{t_idx+1:4d}/{n_trials}]  EER={eer:.2f}%{marker}  {params}")
        return eer

    study.optimize(objective, n_trials=trials_remaining, gc_after_trial=True)

    # ── Stage 2: revalidate top-20 on a fresh split ───────────────────────────
    # TPE already focuses on good regions, but Stage-1 val (~25 % of pool) has
    # enough variance that the true best may sit anywhere in the top-20.
    # Stage 2 re-runs those candidates on a DIFFERENT split to cancel any
    # lucky-split effect and confirm the real winner.
    results_df = pd.DataFrame(completed_results)
    best_params, best_eer = None, float("inf")

    if len(results_df) >= 5:
        TOP_N = min(20, len(results_df))
        top_candidates = (
            results_df.sort_values("eer")
            .head(TOP_N)[["silence_threshold", "octave_cost",
                          "octave_jump_cost", "voiced_unvoiced_cost", "window_size"]]
            .to_dict("records")
        )
        s2_train_p, s2_val_p, s2_train_l, s2_val_l = train_test_split(
            search_paths, search_labels,
            test_size=0.25, stratify=search_labels,
            random_state=SEARCH_SEED + shard_id + 1,
        )
        print(f"\n── Stage 2: revalidating top-{TOP_N} candidates "
              f"(fresh split, {SEARCH_EPOCHS} epochs) ──")
        s2_results = []

        for rank, cand in enumerate(top_candidates):
            s2_tr_raw   = parallel_extract(sound_map, s2_train_p, cand)
            s2_tr_feats = [f for f in s2_tr_raw if f is not None and len(f) > 0]
            s2_tr_labs  = [l for f, l in zip(s2_tr_raw, s2_train_l)
                           if f is not None and len(f) > 0]
            s2_va_raw   = parallel_extract(sound_map, s2_val_p, cand)
            s2_va_feats = [f for f in s2_va_raw if f is not None and len(f) > 0]
            s2_va_labs  = [l for f, l in zip(s2_va_raw, s2_val_l)
                           if f is not None and len(f) > 0]

            if len(s2_tr_feats) < 50 or len(s2_va_feats) < 20:
                print(f"  Stage2 [{rank+1:2d}/{TOP_N}]  SKIP")
                continue

            proc2     = DataProcessor()
            s2_tr_sc  = proc2.scale_features(s2_tr_feats, fit_scaler=True)
            s2_va_sc  = proc2.scale_features(s2_va_feats, fit_scaler=False)
            s2_tr_gen = make_dataset(s2_tr_sc, np.array(s2_tr_labs), 32, shuffle=True)
            s2_va_gen = make_dataset(s2_va_sc, np.array(s2_va_labs), 32, shuffle=False)

            tf.random.set_seed(SEARCH_SEED + shard_id * 10000 + 9000 + rank)
            s2_model = _build_proxy_model()
            s2_model.fit(s2_tr_gen, epochs=SEARCH_EPOCHS, verbose=0)
            s2_scores = s2_model.predict(s2_va_gen, verbose=0).flatten()
            keras.backend.clear_session()
            gc.collect()

            try:
                s2_eer = calculate_eer(np.array(s2_va_labs), s2_scores)
            except Exception:
                continue

            s1_eer = results_df.sort_values("eer").iloc[rank]["eer"]
            s2_results.append({**cand, "s2_eer": s2_eer, "s1_eer": s1_eer})
            print(f"  Stage2 [{rank+1:2d}/{TOP_N}]  "
                  f"s2_EER={s2_eer:.2f}%  (s1_EER={s1_eer:.2f}%)  {cand}")

        if s2_results:
            best_s2     = min(s2_results, key=lambda x: x["s2_eer"])
            best_params = {k: best_s2[k] for k in ["silence_threshold", "octave_cost",
                                                     "octave_jump_cost", "voiced_unvoiced_cost",
                                                     "window_size"]}
            best_eer    = best_s2["s2_eer"]
            s2_path     = f"stage2_results_shard_{shard_id}.csv"
            pd.DataFrame(s2_results).sort_values("s2_eer").to_csv(s2_path, index=False)
            print(f"\nStage 2 best EER={best_eer:.2f}%  params: {best_params}")
            print(f"Saved to {s2_path}")

    if best_params is None:
        best_row    = results_df.loc[results_df["eer"].idxmin()]
        best_params = {k: best_row[k] for k in ["silence_threshold", "octave_cost",
                                                  "octave_jump_cost", "voiced_unvoiced_cost",
                                                  "window_size"]}
        best_eer    = float(best_row["eer"])

    print(f"\nShard {shard_id} complete.  Best EER = {best_eer:.2f}%")
    print(f"Best params: {best_params}")
    return best_params, results_df


# ── Merge shards utility ──────────────────────────────────────────────────────


def merge_shards(num_shards=None):
    """
    Combine all per-shard checkpoint CSVs into a single ranked table,
    pick the best EER candidate, and write best_praat_params.json.

    Parameters
    ----------
    num_shards : int or None
        How many shard files to look for.  If None, auto-discovers every
        file matching the pattern ``search_checkpoint_shard_*.csv``.
    """
    import glob as _glob

    if num_shards is not None:
        paths = [f"search_checkpoint_shard_{i}.csv" for i in range(num_shards)]
    else:
        paths = sorted(_glob.glob("search_checkpoint_shard_*.csv"))

    if not paths:
        print("No shard checkpoint files found.")
        return None, None

    frames = []
    for p in paths:
        if os.path.exists(p):
            df = pd.read_csv(p)
            df["source_shard"] = p
            frames.append(df)
            print(f"  Loaded {len(df)} trials from {p}")
        else:
            print(f"  WARNING: {p} not found — skipping")

    if not frames:
        print("No data loaded from any shard.")
        return None, None

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values("eer").reset_index(drop=True)

    combined.to_csv("search_results_all_shards.csv", index=False)
    print(f"\nMerged {len(combined)} trials total → search_results_all_shards.csv")

    best_row = combined.iloc[0]
    best_params = {
        k: float(best_row[k])
        for k in ["silence_threshold", "octave_cost",
                  "octave_jump_cost", "voiced_unvoiced_cost", "window_size"]
    }
    best_eer = float(best_row["eer"])

    with open("best_praat_params.json", "w") as f:
        json.dump(best_params, f, indent=2)

    print(f"Best EER across all shards: {best_eer:.4f}%")
    print(f"Best params: {best_params}")
    print("Saved to best_praat_params.json")
    print("\nTop 10 candidates:")
    print(combined.head(10).to_string(index=False))

    return best_params, combined


# ── Cell 6: Deepfake Detector (final training & evaluation) ───────────────────


class DeepfakeDetector:
    def __init__(self, extractor_params=None):
        params = extractor_params or {}
        self.extractor = ProsodyFeatureExtractor(**params)
        self.processor = DataProcessor()
        self.model = None

    def train(
        self,
        train_paths,
        train_labels,
        val_paths=None,
        val_labels=None,
        epochs=200,
        batch_size=32,
    ):

        print("Processing training data...")
        train_feats, y_train = self.processor.process_dataset(
            train_paths, train_labels, self.extractor
        )
        X_train = self.processor.scale_features(train_feats, fit_scaler=True)

        # Class weights for 9:1 imbalance
        n_bonafide = int(np.sum(y_train == 0))
        n_spoof = int(np.sum(y_train == 1))
        total = n_bonafide + n_spoof
        class_weight = {0: total / (2.0 * n_bonafide), 1: total / (2.0 * n_spoof)}
        print(
            f"Class weights: bonafide={class_weight[0]:.3f}, "
            f"spoof={class_weight[1]:.3f}"
        )

        train_gen = make_dataset(X_train, y_train, batch_size=batch_size, shuffle=True)

        val_gen = None
        if val_paths is not None and len(val_paths) > 0:
            print("Processing validation data...")
            val_feats, y_val = self.processor.process_dataset(
                val_paths, val_labels, self.extractor
            )
            X_val = self.processor.scale_features(val_feats, fit_scaler=False)
            val_gen = make_dataset(X_val, y_val, batch_size=batch_size, shuffle=False)

        self.model = build_model_b((None, 6))
        self.model.summary()

        monitor = "val_auc" if val_gen is not None else "auc"
        callbacks = [
            keras.callbacks.ModelCheckpoint(
                "best_model.keras",
                monitor=monitor,
                save_best_only=True,
                mode="max",
                verbose=1,
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss" if val_gen is not None else "loss",
                factor=0.5,
                patience=10,
                min_lr=1e-6,
                verbose=1,
            ),
        ]

        history = self.model.fit(
            train_gen,
            epochs=epochs,
            validation_data=val_gen,
            class_weight=class_weight,
            callbacks=callbacks,
            verbose=1,
        )
        return history

    def load_best_model(self, path="best_model.keras"):
        self.model = keras.models.load_model(path)
        print(f"Loaded best model from '{path}'")

    def calculate_eer(self, y_true, y_scores):
        fpr, tpr, _ = roc_curve(y_true, y_scores, pos_label=1)
        eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
        return eer * 100

    def evaluate(self, eval_paths, eval_labels, split_name="Evaluation"):
        print(f"\n[{split_name}] Processing data...")
        eval_feats, y_eval = self.processor.process_dataset(
            eval_paths, eval_labels, self.extractor
        )
        if len(eval_feats) == 0:
            print("No features extracted.")
            return {}
        X_eval = self.processor.scale_features(eval_feats, fit_scaler=False)
        eval_gen = make_dataset(X_eval, y_eval, batch_size=32, shuffle=False)
        print("Predicting...")
        y_proba = self.model.predict(eval_gen, verbose=0).flatten()
        y_pred = (y_proba > 0.5).astype(int)
        return {
            "accuracy": accuracy_score(y_eval, y_pred),
            "precision": precision_score(y_eval, y_pred, zero_division=0),
            "recall": recall_score(y_eval, y_pred, zero_division=0),
            "f1_score": f1_score(y_eval, y_pred, zero_division=0),
            "eer": self.calculate_eer(y_eval, y_proba),
        }


if __name__ == "__main__":
    import argparse
    import warnings

    warnings.filterwarnings("ignore", message="Your input ran out of data")

    parser = argparse.ArgumentParser(
        description="Praat parameter search — run a shard on this device, or merge results."
    )
    parser.add_argument(
        "--shard",
        type=int,
        default=0,
        help="Zero-based shard index for this device (default: 0).",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Total number of shards / devices (default: 1 = no sharding).",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Merge all shard checkpoints and exit (no search).",
    )
    parser.add_argument(
        "--skip-search",
        action="store_true",
        help="Skip search; use hard-coded fallback params and go straight to training.",
    )
    args = parser.parse_args()

    # ── Merge-only mode ───────────────────────────────────────────────────────
    if args.merge:
        merge_shards(num_shards=args.num_shards if args.num_shards > 1 else None)
        raise SystemExit(0)

    # ── Load data ─────────────────────────────────────────────────────────────
    TRAIN_METADATA = r"J:\thesis\dataset\LA\ASVspoof2019_LA_cm_protocols\ASVspoof2019.LA.cm.train.trn.txt"
    TRAIN_AUDIO    = r"J:\thesis\dataset\LA\ASVspoof2019_LA_train\flac"

    VAL_METADATA = r"J:\thesis\dataset\LA\ASVspoof2019_LA_cm_protocols\ASVspoof2019.LA.cm.dev.trl.txt"
    VAL_AUDIO    = r"J:\thesis\dataset\LA\ASVspoof2019_LA_dev\flac"

    TRAIN_BONAFIDE = 2580;  TRAIN_SPOOF = 22800
    VAL_BONAFIDE   = 2548;  VAL_SPOOF   = 22296

    train_loader = ASVspoofLoader(TRAIN_METADATA, [TRAIN_AUDIO])
    X_train_paths, y_train_labels = train_loader.load_data(
        bonafide_limit=TRAIN_BONAFIDE, spoof_limit=TRAIN_SPOOF
    )

    val_loader = ASVspoofLoader(VAL_METADATA, [VAL_AUDIO])
    X_val_paths, y_val_labels = val_loader.load_data(
        bonafide_limit=VAL_BONAFIDE, spoof_limit=VAL_SPOOF
    )

    print(f"\nTrain: {len(X_train_paths)} files")
    print(f"Val:   {len(X_val_paths)} files")
    print(f"Shard {args.shard} of {args.num_shards}\n")

    # ── Parameter search ─────────────────────────────────────────────────────
    if args.skip_search:
        best_params = {
            "silence_threshold": 0.03,
            "octave_cost": 0.01,
            "octave_jump_cost": 0.35,
            "voiced_unvoiced_cost": 0.14,
            "window_size": 0.1,
        }
        search_results_df = None
        print("Skipping search. Using fallback params:", best_params)
    else:
        best_params, search_results_df = run_param_search(
            X_train_paths,
            y_train_labels,
            shard_id=args.shard,
            num_shards=args.num_shards,
        )

        if search_results_df is not None and len(search_results_df) > 0:
            out_csv = f"search_results_shard_{args.shard}.csv"
            search_results_df.sort_values("eer").to_csv(out_csv, index=False)
            print(f"\nShard results saved to {out_csv} ({len(search_results_df)} trials).")
            print("\nTop 10 candidates by EER:")
            print(search_results_df.sort_values("eer").head(10).to_string(index=False))

    # ── Save per-shard best params ────────────────────────────────────────────
    out_json = f"best_praat_params_shard_{args.shard}.json"
    with open(out_json, "w") as f:
        json.dump(best_params, f, indent=2)
    print(f"\nFinal best_params for shard {args.shard}: {best_params}")
    print(f"Saved to {out_json}")
    print("\nWhen all shards are done, run:  python py_paramsearch.py --merge --num-shards", args.num_shards)

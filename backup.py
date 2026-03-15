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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tensorflow.keras import layers

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


def _sample_params(param_space, trial_index):
    # Round-robin over the 4 discrete window sizes so each gets exactly
    # N_SEARCH // 4 trials — guarantees equal coverage of 50/100/200/500 ms.
    # The four Praat parameters remain fully randomised per trial.
    window_sizes = param_space["window_size"]           # [0.05, 0.1, 0.2, 0.5]
    window_size  = window_sizes[trial_index % len(window_sizes)]
    return {
        "silence_threshold":    round(random.uniform(*param_space["silence_threshold"]),    4),
        "octave_cost":          round(random.uniform(*param_space["octave_cost"]),          4),
        "octave_jump_cost":     round(random.uniform(*param_space["octave_jump_cost"]),     4),
        "voiced_unvoiced_cost": round(random.uniform(*param_space["voiced_unvoiced_cost"]), 4),
        "window_size":          window_size,
    }


# ── Main search ───────────────────────────────────────────────────────────────
def run_param_search(train_paths, train_labels):
    random.seed(SEARCH_SEED)
    np.random.seed(SEARCH_SEED)
    tf.random.set_seed(SEARCH_SEED)

    # Balanced sub-sample
    bona_paths = [p for p, l in zip(train_paths, train_labels) if l == 0][
        :SEARCH_SAMPLES
    ]
    spoof_paths = [p for p, l in zip(train_paths, train_labels) if l == 1][
        :SEARCH_SAMPLES
    ]
    search_paths = bona_paths + spoof_paths
    search_labels = [0] * len(bona_paths) + [1] * len(spoof_paths)
    print(
        f"Search pool: {len(bona_paths)} bonafide + {len(spoof_paths)} spoof "
        f"= {len(search_paths)} files"
    )

    # Pre-load all audio into RAM (done ONCE for all trials)
    sound_map = preload_sounds(search_paths)

    # Phase 1: grid scan on 100-file slice
    scan_slice = bona_paths[:50] + spoof_paths[:50]
    param_space = grid_scan_bounds(sound_map, scan_slice)
    # bounds are used as-is from the data-driven grid scan — no overrides

    # Stratified 75/25 train-val split
    s_train_p, s_val_p, s_train_l, s_val_l = train_test_split(
        search_paths,
        search_labels,
        test_size=0.25,
        stratify=search_labels,
        random_state=SEARCH_SEED,
    )
    print(f"Split → train: {len(s_train_p)}, val: {len(s_val_p)}\n")

    best_eer, best_params = float("inf"), None
    results = []

    # Resume from checkpoint if available
    checkpoint_path = "search_checkpoint.csv"
    completed_trials = 0
    if os.path.exists(checkpoint_path):
        prior_df = pd.read_csv(checkpoint_path)
        results = prior_df.to_dict("records")
        completed_trials = len(results)
        if results:
            best_row = min(results, key=lambda x: x["eer"])
            best_eer = best_row["eer"]
            best_params = {
                k: best_row[k]
                for k in [
                    "silence_threshold",
                    "octave_cost",
                    "octave_jump_cost",
                    "voiced_unvoiced_cost",
                    "window_size",
                ]
            }
        print(f"Resuming from checkpoint: {completed_trials} trials already done.")
        print(f"Current best EER: {best_eer:.2f}%")

    remaining = N_SEARCH - completed_trials
    print(
        f"\n── Phase 2: {remaining} remaining candidates "
        f"× {SEARCH_EPOCHS} epochs (Option A) ──\n"
    )

    for trial in range(completed_trials, N_SEARCH):
        params = _sample_params(param_space, trial)

        # Parallel extraction from pre-loaded Sounds
        tr_raw = parallel_extract(sound_map, s_train_p, params)
        tr_feats = [f for f in tr_raw if f is not None and len(f) > 0]
        tr_labs = [l for f, l in zip(tr_raw, s_train_l) if f is not None and len(f) > 0]

        nan_ratio = 1.0 - len(tr_feats) / max(len(s_train_p), 1)
        if nan_ratio > 0.30:
            print(f"  [{trial + 1:4d}/{N_SEARCH}]  SKIP (nan={nan_ratio:.0%})")
            continue

        va_raw = parallel_extract(sound_map, s_val_p, params)
        va_feats = [f for f in va_raw if f is not None and len(f) > 0]
        va_labs = [l for f, l in zip(va_raw, s_val_l) if f is not None and len(f) > 0]

        if len(va_feats) < 20:
            print(f"  [{trial + 1:4d}/{N_SEARCH}]  SKIP (too few val samples)")
            continue

        # Scale
        proc_tmp = DataProcessor()
        tr_scaled = proc_tmp.scale_features(tr_feats, fit_scaler=True)
        va_scaled = proc_tmp.scale_features(va_feats, fit_scaler=False)

        tr_gen = make_dataset(tr_scaled, np.array(tr_labs), batch_size=32, shuffle=True)
        va_gen = make_dataset(
            va_scaled, np.array(va_labs), batch_size=32, shuffle=False
        )

        # Train proxy model — no EarlyStopping so every candidate trains for the
        # same fixed SEARCH_EPOCHS, making EER comparisons fair across trials.
        tf.random.set_seed(SEARCH_SEED + trial)
        model = _build_proxy_model()
        model.fit(tr_gen, epochs=SEARCH_EPOCHS, verbose=0)

        # Evaluate EER
        y_scores = model.predict(va_gen, verbose=0).flatten()
        try:
            eer = calculate_eer(np.array(va_labs), y_scores)
        except Exception:
            keras.backend.clear_session()
            continue

        results.append(
            {
                **params,
                "eer": eer,
                "nan_ratio": nan_ratio,
                "n_train": len(tr_feats),
                "n_val": len(va_feats),
            }
        )

        if eer < best_eer:
            best_eer, best_params = eer, params.copy()
            print(
                f"  [{trial + 1:4d}/{N_SEARCH}]  EER={eer:.2f}%  ← NEW BEST  {params}"
            )
        else:
            print(f"  [{trial + 1:4d}/{N_SEARCH}]  EER={eer:.2f}%")

        keras.backend.clear_session()
        del tr_raw, tr_feats, tr_labs, va_raw, va_feats, va_labs
        del tr_scaled, va_scaled, tr_gen, va_gen, model, y_scores
        gc.collect()

        # Checkpoint every 25 trials
        if (trial + 1) % 25 == 0 and results:
            pd.DataFrame(results).sort_values("eer").to_csv(
                checkpoint_path, index=False
            )
            print(f"  ── checkpoint saved ({len(results)} trials) ──")

    # ── Stage 2: revalidate top-20 on a fresh split ───────────────────────────
    # Stage-1 EER is measured on ~1,250 val files — noisy enough that the true
    # best candidate can be buried in the top-20.  Stage 2 re-runs only those
    # top-20 with a DIFFERENT 75/25 split of the same pool and 200 epochs,
    # breaking any "lucky split" effect and confirming the real winner.
    results_df = pd.DataFrame(results)
    if len(results_df) >= 5:
        TOP_N      = min(20, len(results_df))
        S2_EPOCHS  = 200   # paper's full epoch count for the confirmation run

        top_candidates = (
            results_df.sort_values("eer")
            .head(TOP_N)[
                ["silence_threshold", "octave_cost",
                 "octave_jump_cost", "voiced_unvoiced_cost", "window_size"]
            ]
            .to_dict("records")
        )

        # Fresh stratified split with a different seed so val files differ from Stage 1
        s2_train_p, s2_val_p, s2_train_l, s2_val_l = train_test_split(
            search_paths, search_labels,
            test_size=0.25, stratify=search_labels,
            random_state=SEARCH_SEED + 1,
        )

        print(
            f"\n── Stage 2: revalidating top-{TOP_N} candidates  "
            f"(fresh split, {S2_EPOCHS} epochs) ──"
        )
        s2_results = []

        for rank, cand in enumerate(top_candidates):
            s2_tr_raw   = parallel_extract(sound_map, s2_train_p, cand)
            s2_tr_feats = [f for f in s2_tr_raw   if f is not None and len(f) > 0]
            s2_tr_labs  = [l for f, l in zip(s2_tr_raw,  s2_train_l)
                           if f is not None and len(f) > 0]

            s2_va_raw   = parallel_extract(sound_map, s2_val_p, cand)
            s2_va_feats = [f for f in s2_va_raw   if f is not None and len(f) > 0]
            s2_va_labs  = [l for f, l in zip(s2_va_raw,  s2_val_l)
                           if f is not None and len(f) > 0]

            if len(s2_tr_feats) < 50 or len(s2_va_feats) < 20:
                print(f"  Stage2 [{rank+1:2d}/{TOP_N}]  SKIP (too few samples)")
                continue

            proc2     = DataProcessor()
            s2_tr_sc  = proc2.scale_features(s2_tr_feats, fit_scaler=True)
            s2_va_sc  = proc2.scale_features(s2_va_feats, fit_scaler=False)
            s2_tr_gen = make_dataset(s2_tr_sc, np.array(s2_tr_labs), 32, shuffle=True)
            s2_va_gen = make_dataset(s2_va_sc, np.array(s2_va_labs), 32, shuffle=False)

            tf.random.set_seed(SEARCH_SEED + 9999 + rank)
            s2_model = _build_proxy_model()
            s2_model.fit(s2_tr_gen, epochs=S2_EPOCHS, verbose=0)

            s2_scores = s2_model.predict(s2_va_gen, verbose=0).flatten()
            try:
                s2_eer = calculate_eer(np.array(s2_va_labs), s2_scores)
            except Exception:
                keras.backend.clear_session()
                continue

            s1_eer = results_df.sort_values("eer").iloc[rank]["eer"]
            s2_results.append({**cand, "s2_eer": s2_eer, "s1_eer": s1_eer})
            print(
                f"  Stage2 [{rank+1:2d}/{TOP_N}]  "
                f"s2_EER={s2_eer:.2f}%  (s1_EER={s1_eer:.2f}%)  {cand}"
            )

            keras.backend.clear_session()
            del s2_tr_raw, s2_tr_feats, s2_va_raw, s2_va_feats
            del s2_tr_sc, s2_va_sc, s2_tr_gen, s2_va_gen, s2_model, s2_scores
            gc.collect()

        if s2_results:
            best_s2     = min(s2_results, key=lambda x: x["s2_eer"])
            best_params = {k: best_s2[k] for k in [
                "silence_threshold", "octave_cost",
                "octave_jump_cost", "voiced_unvoiced_cost", "window_size",
            ]}
            best_eer = best_s2["s2_eer"]
            pd.DataFrame(s2_results).sort_values("s2_eer").to_csv(
                "stage2_results.csv", index=False
            )
            print(f"\nStage 2 best EER={best_eer:.2f}%")
            print(f"Stage 2 best params: {best_params}")
            print("Stage 2 results saved to stage2_results.csv")

    print(f"\nSearch complete.  Best EER={best_eer:.2f}%")
    print(f"Best params: {best_params}")
    return best_params, pd.DataFrame(results)


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
    # ── Cell 7: Load Data (ASVspoof2019 LA train + dev) ───────────────────────────

    TRAIN_METADATA = r"J:\thesis\dataset\LA\ASVspoof2019_LA_cm_protocols\ASVspoof2019.LA.cm.train.trn.txt"
    TRAIN_AUDIO = r"J:\thesis\dataset\LA\ASVspoof2019_LA_train\flac"

    VAL_METADATA = r"J:\thesis\dataset\LA\ASVspoof2019_LA_cm_protocols\ASVspoof2019.LA.cm.dev.trl.txt"
    VAL_AUDIO = r"J:\thesis\dataset\LA\ASVspoof2019_LA_dev\flac"

    # Paper's exact sample counts
    TRAIN_BONAFIDE = 2580
    TRAIN_SPOOF = 22800
    VAL_BONAFIDE = 2548
    VAL_SPOOF = 22296

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

    # ── Cell 8: Run Praat Parameter Search ───────────────────────────────────────
    #
    # SKIP_SEARCH = True  → paste previously found params and skip to training.
    # SKIP_SEARCH = False → run the full 150-candidate search.
    #
    # The search saves a checkpoint every 25 trials to 'search_checkpoint.csv'.
    # If the session times out, re-run this cell — it will automatically resume
    # from the last checkpoint.
    import warnings

    from tqdm import tqdm

    warnings.filterwarnings("ignore", message="Your input ran out of data")
    SKIP_SEARCH = False

    if SKIP_SEARCH:
        best_params = {
            "silence_threshold": 0.03,
            "octave_cost": 0.01,
            "octave_jump_cost": 0.35,
            "voiced_unvoiced_cost": 0.14,
            "window_size": 0.1,
        }
        search_results_df = None
        print("Skipping search. Using params:", best_params)

    else:
        best_params, search_results_df = run_param_search(X_train_paths, y_train_labels)

        if search_results_df is not None and len(search_results_df) > 0:
            search_results_df.sort_values("eer").to_csv(
                "search_results_full.csv", index=False
            )
            print(f"\nFull results saved ({len(search_results_df)} trials).")
            print("\nTop 10 candidates by EER:")
            print(search_results_df.sort_values("eer").head(10).to_string(index=False))
            # (plots skipped in script mode — see search_results_full.csv)

    with open("best_praat_params.json", "w") as f:
        json.dump(best_params, f, indent=2)
    print("\nFinal best_params:", best_params)
    print("Saved to best_praat_params.json")

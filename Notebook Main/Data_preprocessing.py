

import os
import psutil


import h5py
import joblib

import numpy as np
import pandas as pd

import tensorflow as tf

from sklearn.preprocessing import StandardScaler

import gc




# Zone marking
def xy_to_zone_vectorized(xs, ys, n_rows, n_cols, flip_y=False):
    xs = np.asarray(xs)
    ys = np.asarray(ys)

    if flip_y:
        ys = 1 - ys

    row = np.clip((ys * n_rows).astype(int), 0, n_rows - 1)
    col = np.clip((xs * n_cols).astype(int), 0, n_cols - 1)
    return row * n_cols + col



# Velocity and movement features calculated based on normalized coordinates and specific windows
def add_velocity_features(df):
    """Enhanced velocity and movement features for better coordinate prediction"""
    df = df.sort_values(["match_id", "player_id", "frame_number"])

    # === BASIC VELOCITY (existing) ===
    df["dx"] = df.groupby(["match_id", "player_id"])["x_normalized"].diff().fillna(0)
    df["dy"] = df.groupby(["match_id", "player_id"])["y_normalized"].diff().fillna(0)

    # Smooth velocities
    df["dx"] = df.groupby(["match_id", "player_id"])["dx"].rolling(3, min_periods=1).mean().reset_index(level=[0,1], drop=True)
    df["dy"] = df.groupby(["match_id", "player_id"])["dy"].rolling(3, min_periods=1).mean().reset_index(level=[0,1], drop=True)

    # === MULTI-WINDOW VELOCITY (NEW) ===
    for window in [3, 5, 10]:
        df[f"dx_avg_{window}"] = df.groupby(["match_id", "player_id"])["dx"].rolling(window, min_periods=1).mean().reset_index(level=[0,1], drop=True)
        df[f"dy_avg_{window}"] = df.groupby(["match_id", "player_id"])["dy"].rolling(window, min_periods=1).mean().reset_index(level=[0,1], drop=True)
        df[f"speed_avg_{window}"] = np.sqrt(df[f"dx_avg_{window}"]**2 + df[f"dy_avg_{window}"]**2)

    # === EXISTING FEATURES ===
    df["speed_normalized"] = np.sqrt(df["dx"]**2 + df["dy"]**2)
    df["acceleration"] = df.groupby(["match_id", "player_id"])["speed_normalized"].diff().fillna(0)
    df["movement_angle"] = np.arctan2(df["dy"], df["dx"])

    # === NEW: ACCELERATION TRENDS ===
    df["acceleration_trend"] = df.groupby(["match_id", "player_id"])["acceleration"].rolling(5, min_periods=1).mean().reset_index(level=[0,1], drop=True)

    # === NEW: DIRECTION PERSISTENCE ===
    df["angle_change"] = df.groupby(["match_id", "player_id"])["movement_angle"].diff().fillna(0)
    df["angle_stability"] = df.groupby(["match_id", "player_id"])["angle_change"].rolling(5, min_periods=1).std().reset_index(level=[0,1], drop=True).fillna(0)

    # === NEW: SPEED MOMENTUM ===
    df["speed_change_rate"] = df.groupby(["match_id", "player_id"])["speed_normalized"].diff().fillna(0)

    # === SPATIAL FEATURES (existing) ===
    df["distance_from_center"] = np.sqrt((df["x_normalized"] - 0.5)**2 + (df["y_normalized"] - 0.5)**2)
    df["distance_from_goal_home"] = df["x_normalized"]
    df["distance_from_goal_away"] = 1 - df["x_normalized"]
    df["distance_from_sideline"] = np.minimum(df["y_normalized"], 1 - df["y_normalized"])

    # === BALL & TEAM FEATURES (existing) ===
    if "team_spread" not in df.columns:
        df["team_spread"] = df.groupby(["match_id", "frame_number", "team_type"])["x_normalized"].transform("std").fillna(0)
    if "distance_to_ball" not in df.columns:
        df["distance_to_ball"] = 0.5
    if "ball_possession_proximity" not in df.columns:
        df["ball_possession_proximity"] = df["distance_to_ball"]

    print("\n📌 Enhanced velocity features added!")
    print(f"New multi-window features: dx_avg_3/5/10, dy_avg_3/5/10, speed_avg_3/5/10")
    print(f"New trend features: acceleration_trend, angle_change, angle_stability, speed_change_rate")

    return df


# Contextual features based on game state
def add_contextual_features(df):
    """Add contextual features based on game state"""
    # Time-based features
    df["period_progress"] = df["timestamp_seconds"] / df.groupby(["match_id", "period"])["timestamp_seconds"].transform("max")

    # Formation-based features (simplified)
    if "position_category" in df.columns:
        # Create position-based features
        position_encoding = {"goalkeeper": 0, "defender": 1, "midfielder": 2, "forward": 3, "unknown": 4}
        df["position_encoded"] = df["position_category"].map(position_encoding).fillna(4)
    else:
        df["position_encoded"] = 2  # default to midfielder
    print("\n📌 add_contextual_features sample:")
    print(df.head(5)[[
        "period_progress",
        "position_encoded"
    ]])

    return df


# Data splitting by match id,
def split_by_match_df(df, train_frac=0.7, val_frac=0.15, seed=42, fixed_split=False):
    """
    Splits a dataframe into train/val/test by unique match_id.
    Ensures no leakage across matches.

    If fixed_split=True:
        3 matches -> train
        1 match   -> val
        1 match   -> test
    """
    unique_matches = df["match_id"].unique()
    np.random.seed(seed)
    np.random.shuffle(unique_matches)

    n_total = len(unique_matches)

    if fixed_split:
        if n_total < 5:
            raise ValueError("Need at least 5 unique matches for fixed split (3/1/1).")

        train_matches = unique_matches[:3]
        val_matches   = unique_matches[3:4]
        test_matches  = unique_matches[4:5]

    else:
        n_train = max(1, int(train_frac * n_total))
        n_val   = max(1, int(val_frac * n_total))
        n_train = min(n_train, n_total - n_val)

        train_matches = unique_matches[:n_train]
        val_matches   = unique_matches[n_train:n_train + n_val]
        test_matches  = unique_matches[n_train + n_val:]

    train_df = df[df["match_id"].isin(train_matches)].reset_index(drop=True)
    val_df   = df[df["match_id"].isin(val_matches)].reset_index(drop=True)
    test_df  = df[df["match_id"].isin(test_matches)].reset_index(drop=True)

    print("Split matches:", len(train_matches), len(val_matches), len(test_matches))
    print("Rows per split:", len(train_df), len(val_df), len(test_df))

    return train_df, val_df, test_df



def keras_sequence_generator_with_scaler(df, feature_cols, scaler, seq_len, horizon,coord_cols,
                             coordinate_targets, n_rows=None, n_cols=None):
    """
    Generator that yields sequences one by one (or in small batches) instead of storing all.
    """
    for _, player_df in df.groupby(["match_id", "player_id"]):
        feats = player_df[feature_cols].values.astype(np.float32)
        if scaler is not None:
            feats = scaler.transform(feats)
        coords = player_df[list(coord_cols)].values.astype(np.float32)

        if coordinate_targets:
            # regression: dx, dy
            for i in range(len(player_df) - seq_len - horizon + 1):
                seq = feats[i:i+seq_len]
                t_curr = i + seq_len - 1
                t_fut = t_curr + horizon
                # delta = coords[t_fut] - coords[t_curr]
                # yield seq, delta.astype(np.float32)
                print("Keras_seq_gen_with_scaler done for regression")
                yield seq, coords[t_fut].astype(np.float32)

        else:
            zones = xy_to_zone_vectorized(
                player_df["x_normalized"].values,
                player_df["y_normalized"].values,
                n_rows, n_cols
            )
            for i in range(len(player_df) - seq_len - horizon + 1):
                seq = feats[i:i+seq_len]
                t_fut = i + seq_len - 1 + horizon
                zone = zones[t_fut]
                # yield seq, np.array([zone], dtype=np.int32)
                print("Keras_seq_gen_with_scaler done for classification")
                yield seq, np.int32(zone)



def compute_zone_counts_from_df(df, seq_len, horizon, n_rows, n_cols):
        """
        Count target-zone frequency using the same target index logic as
        keras_sequence_generator_with_scaler (classification mode).
        """
        counts = {}

        for _, player_df in df.groupby(["match_id", "player_id"]):
            player_df = player_df.sort_values("frame_number")
            n = len(player_df)
            if n < seq_len + horizon:
                continue

            zones = xy_to_zone_vectorized(
                player_df["x_normalized"].values,
                player_df["y_normalized"].values,
                n_rows, n_cols
            )

            # t_fut = i + seq_len - 1 + horizon
            start = seq_len - 1 + horizon
            fut_zones = zones[start:]
            vc = pd.Series(fut_zones).value_counts()

            for z, c in vc.items():
                counts[int(z)] = counts.get(int(z), 0) + int(c)
            return pd.Series(counts).sort_index()



# Weighted sampling generator
def weighted_generator(zone_weights_dict,feature_cols,coord_cols,CO_ORDINATES,df, SEQ_LEN, HORIZON_FRAMES, N_ROWS, N_COLS,):
    coordinate_targets=CO_ORDINATES
    zone_counts = compute_zone_counts_from_df(df, SEQ_LEN, HORIZON_FRAMES, N_ROWS, N_COLS)
    zone_weights = 1.0 / np.sqrt(zone_counts)
    zone_weights = zone_weights / zone_weights.mean()
    zone_weights_dict = zone_weights.to_dict()  # <--- this is what generator needs
    
    scaler= StandardScaler()
    for x, y in keras_sequence_generator_with_scaler(
        df, feature_cols, scaler, SEQ_LEN, HORIZON_FRAMES, coord_cols,
        coordinate_targets, N_ROWS, N_COLS
    ):
        if coordinate_targets:
            weight = 1.0
        else:
            weight = zone_weights_dict.get(int(y), 1.0)

        yield x, y, weight


def make_tf_dataset_with_scaler(df, batch_size,N_ROWS,N_COLS, SEQ_LEN,HORIZON_FRAMES,FEATURE_COLS,shuffle=False, coordinate_targets=False, repeat=False):

    min_frames_needed = SEQ_LEN + HORIZON_FRAMES
    df = df.groupby(["match_id", "player_id"]).filter(
        lambda x: len(x) >= min_frames_needed
    )
    """Optimized tf.data pipeline with parallel processing and known cardinality"""
    output_types = (
    tf.float32,
    tf.float32 if coordinate_targets else tf.int32,
    tf.float32   # sample_weight
)

    output_shapes = (
    (SEQ_LEN, len(FEATURE_COLS)),
    (2,) if coordinate_targets else (),
    ()   # weight scalar
)

    # 🔧 PRE-CALCULATE: Count sequences for progress bar
    total_sequences = 0
    for _, player_df in df.groupby(["match_id", "player_id"]):
        total_sequences += max(0, len(player_df) - SEQ_LEN - HORIZON_FRAMES + 1)

    expected_batches = total_sequences // batch_size
    print(f"   Dataset: {total_sequences:,} sequences → {expected_batches} batches")
    
    scaler=StandardScaler()
    ds = tf.data.Dataset.from_generator(
        lambda: keras_sequence_generator_with_scaler(
            df, FEATURE_COLS, scaler, SEQ_LEN, HORIZON_FRAMES,
            coordinate_targets, n_rows=N_ROWS, n_cols=N_COLS,coordinate_targets=coordinate_targets
        ),
        output_types=output_types,
        output_shapes=output_shapes
    )

    if shuffle:
        ds = ds.shuffle(4096)  # 🔧 INCREASED: Better randomization

    # 🔧 OPTIMIZATION: Batch before cache for memory efficiency
    # ensure labels are always correct shape

    def map_fn(x, y, w):
        if coordinate_targets:
            y = tf.reshape(y, [-1])
        return x, y, w

    ds = ds.map(map_fn, num_parallel_calls=tf.data.AUTOTUNE)
    # 🔧 FIXED: drop_remainder=True prevents 'ran out of data' warnings
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    # Only repeat if requested (e.g. for training)
    if repeat:
        ds = ds.repeat()
        
    print("make_tf_done_with scaler")
    return ds,expected_batches



def alpha_from_class_counts(y_train):
        classes, counts = np.unique(y_train, return_counts=True)
        # Compute per-class alpha for focal loss (inverse frequency)
        counts = np.maximum(counts, 1)
        total = counts.sum()
        alpha = total / counts
        alpha = np.sqrt(alpha)
        alpha = alpha / alpha.sum()
        print("Class counts:", dict(zip(classes, counts)))
        return alpha


files_dir={
    "train_df":'preprocessed_data/train_df.pkl',
    "val_df":'preprocessed_data/val_df.pkl',
    "test_df":'preprocessed_data/test_df.pkl',    
}
def check_preprocessed_exists():
    """Check if all required preprocessed files exist for the specified backend"""

    required_files = [
            f'preprocessed_data/train_df.pkl',
            f'preprocessed_data/val_df.pkl',
            f'preprocessed_data/test_df.pkl'
    ]
    missing_files = [f for f in required_files if not os.path.exists(f)]

    if missing_files:
        return False, f"Missing files for : {[os.path.basename(f) for f in missing_files]}"

    return True, f"All preprocessed files found"


def load_preprocessed_sequences_keras():
    """Load preprocessed sequences for Keras backend"""

    print("\n" + "="*80)
    print("📂 LOADING KERAS PREPROCESSED DATA")
    print("="*80)

    try:

        train_df = pd.read_pickle(files_dir['train_df'])
        val_df = pd.read_pickle(files_dir['val_df'])
        test_df = pd.read_pickle(files_dir['test_df'])
        
       
        print(f"✓ Loaded dataframes:")
        print(f"   Train: {len(train_df)} rows, Val: {len(val_df)} rows, Test: {len(test_df)} rows")
        return {
            'train_df': train_df,
            'val_df': val_df,
            'test_df': test_df,
            'df': pd.concat([train_df, val_df, test_df], ignore_index=True)
        }

    except Exception as e:
        print(f"\n❌ Error loading Keras preprocessed data: {e}")
        return None




def load_dataset_with_memory_optimization(file_path):
    chunk_size = 500_000  # Adjust based on your RAM
    chunks = []
    total_rows = 0
    
    try:
        # Load WITHOUT forcing dtypes - let pandas auto-detect
        for i, chunk in enumerate(pd.read_csv(file_path, chunksize=chunk_size, low_memory=False)):
            
            # Drop any timestamp string columns that aren't needed
            cols_to_drop = []
            for col in chunk.columns:
                # Check if column contains time strings like '0:11.20'
                if chunk[col].dtype == 'object':
                    sample_val = str(chunk[col].dropna().iloc[0]) if len(chunk[col].dropna()) > 0 else ""
                    if ':' in sample_val and col not in ['match_id', 'player_id']:
                        cols_to_drop.append(col)
                        print(f"  Dropping timestamp column: {col}")
            
            if cols_to_drop:
                chunk = chunk.drop(columns=cols_to_drop)
            
            # Downcast numeric columns AFTER loading to save memory
            for col in chunk.select_dtypes(include=['float64']).columns:
                chunk[col] = chunk[col].astype('float32')
            for col in chunk.select_dtypes(include=['int64']).columns:
                chunk[col] = pd.to_numeric(chunk[col], downcast='integer')
            
            chunks.append(chunk)
            total_rows += len(chunk)
            
            # Memory check
            mem_percent = psutil.virtual_memory().percent
            print(f"  Chunk {i+1}: {len(chunk):,} rows | Total: {total_rows:,} | RAM: {mem_percent:.1f}%", end='\r')
            
            # Stop if memory is getting too high
            if mem_percent > 95:
                print(f"\n⚠️  Memory threshold reached at {total_rows:,} rows. Using partial data.")
                break
        
        print(f"\n✓ Loaded {len(chunks)} chunks")
        
        # Concatenate chunks
        df = pd.concat(chunks, ignore_index=True)
        del chunks
        gc.collect()
        
        print(f"Dataset loaded successfully! Shape: {df.shape}")
        print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
        if 'timestamp_seconds' in df.columns:
            print(f"Time range: {df['timestamp_seconds'].min():.2f} to {df['timestamp_seconds'].max():.2f} seconds")

        
    except MemoryError:
        print("❌ MemoryError during loading. Try reducing chunk_size or using fewer matches.")
        raise
    
    # Display basic info
    print(f"Columns: {list(df.columns)}")
    print(f"Unique matches: {df['match_id'].nunique()}")
    print(f"Unique players: {df['player_id'].nunique()}")
    
 


def balance_train_sequence_oversample(
    train_df,
    seq_len,
    horizon_frames,
    n_rows,
    n_cols,
    target_ratio=0.5,
    max_oversample=5.0,
    seed=42):
    """
    Balance sequential training data by oversampling player-trajectory segments
    whose target positions (SEQ_LEN + HORIZON_FRAMES ahead) are underrepresented.

    Temporal continuity is preserved: full player-epoch segments are duplicated
    with synthetic IDs so the tf.data generator treats them as independent tracks.

    Args:
        train_df       : DataFrame with [match_id, player_id, frame_number,
                         x_normalized, y_normalized, ...]
        target_ratio   : Minority zones raised to target_ratio * mean_count.
                         0.5 = bring each zone up to 50% of the average.
        max_oversample : Hard cap -- no segment duplicated more than this many times.
    Returns:
        Balanced DataFrame ready to pass to make_tf_dataset().
    """
    np.random.seed(seed)
    min_frames = seq_len + horizon_frames

    # --- 1. Compute target zone for every valid sequence window ---
    records = []
    for (mid, pid), grp in train_df.groupby(["match_id", "player_id"]):
        grp = grp.sort_values("frame_number").reset_index(drop=True)
        if len(grp) < min_frames:
            continue
        xs = grp["x_normalized"].values
        ys = grp["y_normalized"].values
        for i in range(len(grp) - min_frames + 1):
            t_fut = i + seq_len + horizon_frames - 1
            zone = int(xy_to_zone_vectorized(
                np.array([xs[t_fut]]), np.array([ys[t_fut]]), n_rows, n_cols)[0])
            records.append((mid, pid, zone))

    wins = pd.DataFrame(records, columns=["match_id", "player_id", "zone"])

    # --- 2. Distribution before ---
    zone_counts = wins["zone"].value_counts()
    mean_count  = zone_counts.mean()
    target_min  = max(1, int(target_ratio * mean_count))
    total_zones = n_rows * n_cols

    print("=" * 60)
    print("SEQUENCE BALANCING -- ZONE DISTRIBUTION BEFORE")
    print("=" * 60)
    print(f"  Total valid sequences  : {len(wins):,}")
    print(f"  Zones with sequences   : {len(zone_counts)} / {total_zones}")
    print(f"  Mean sequences / zone  : {mean_count:.0f}")
    print(f"  Min count              : {zone_counts.min()} (zone {zone_counts.idxmin()})")
    print(f"  Max count              : {zone_counts.max()} (zone {zone_counts.idxmax()})")
    print(f"  Imbalance ratio        : {zone_counts.max() / max(zone_counts.min(), 1):.1f}x")
    print(f"  Target minimum / zone  : >= {target_min}")

    # --- 3. Oversample player segments for minority zones ---
    new_segments = []
    syn_id = 0
    total_new_seqs = 0
    minority_zones = [z for z in range(total_zones)
                      if zone_counts.get(z, 0) < target_min]
    print(f"\n  Minority zones (below target) : {len(minority_zones)}")

    for zone in minority_zones:
        current     = zone_counts.get(zone, 0)
        deficit     = target_min - current
        contributors = wins[wins["zone"] == zone][["match_id", "player_id"]].drop_duplicates()
        if len(contributors) == 0:
            continue

        # Cap: avoid extreme duplication of a single player
        n_samples = min(deficit, int(max_oversample * len(contributors)))
        sampled = contributors.sample(n=n_samples, replace=True, random_state=seed)

        for _, row in sampled.iterrows():
            seg = train_df[
                (train_df["match_id"] == row["match_id"]) &
                (train_df["player_id"] == row["player_id"])
            ].copy()
            # Synthetic IDs: generator groups by (match_id, player_id),
            # so each copy forms its own independent temporal track.
            seg["match_id"]  = "syn_" + str(row["match_id"])
            seg["player_id"] = "syn_" + str(syn_id)
            syn_id += 1
            total_new_seqs += max(0, len(seg) - min_frames + 1)
            new_segments.append(seg)

    # --- 4. Combine ---
    if new_segments:
        balanced = pd.concat([train_df] + new_segments, ignore_index=True)
    else:
        balanced = train_df.copy()
        print("  Already balanced -- no oversampling needed")

    # --- 5. Report after ---
    print(f"\nBALANCING COMPLETE")
    print(f"  Synthetic segments added   : {len(new_segments)}")
    print(f"  Approx. new sequences      : {total_new_seqs:,}")
    print(f"  train_df rows: {len(train_df):,} --> {len(balanced):,}")
    print(f"  Growth factor              : {len(balanced) / len(train_df):.2f}x")
    print("=" * 60)
    return balanced



def balance_train_sequences_downsample(
    train_df,
    seq_len,
    horizon_frames,
    n_rows,
    n_cols,
    target_ratio=0.8,
    seed=42):
    """
    Balance by DOWNSAMPLING majority zones.

    Removes excess sequences from overrepresented zones while keeping all
    minority sequences. Reduces dataset size for better memory efficiency.

    Args:
        train_df       : DataFrame with [match_id, player_id, frame_number, ...]
        target_ratio   : Upper bound ratio (default 0.8 = 80% of mean).
                         0.5 = aggressive balancing (smaller dataset)
                         1.0 = no downsampling (keep full dataset)
        seed           : Random seed for reproducibility
    Returns:
        Balanced DataFrame (same or smaller than input)
    """
    np.random.seed(seed)
    min_frames = seq_len + horizon_frames

    # --- 1. Scan all valid sequence windows ---
    records = []
    for (mid, pid), grp in train_df.groupby(["match_id", "player_id"]):
        grp = grp.sort_values("frame_number").reset_index(drop=True)
        if len(grp) < min_frames:
            continue
        xs = grp["x_normalized"].values
        ys = grp["y_normalized"].values
        for i in range(len(grp) - min_frames + 1):
            t_fut = i + seq_len + horizon_frames - 1
            zone = int(xy_to_zone_vectorized(
                np.array([xs[t_fut]]), np.array([ys[t_fut]]), n_rows, n_cols)[0])
            records.append((mid, pid, zone))

    # --- 2. Compute distribution BEFORE ---
    zone_counts = pd.Series([r[2] for r in records]).value_counts()
    mean_count = zone_counts.mean()
    max_allowed = int(target_ratio * mean_count)
    total_zones = n_rows * n_cols

    print("=" * 60)
    print("SEQUENCE BALANCING -- DOWNSAMPLING")
    print("=" * 60)
    print(f"  Total valid sequences  : {len(records):,}")
    print(f"  Zones with sequences   : {len(zone_counts)} / {total_zones}")
    print(f"  Mean sequences / zone  : {mean_count:.0f}")
    print(f"  Min count              : {zone_counts.min()} (zone {zone_counts.idxmin()})")
    print(f"  Max count              : {zone_counts.max()} (zone {zone_counts.idxmax()})")
    print(f"  Imbalance ratio        : {zone_counts.max() / max(zone_counts.min(), 1):.1f}x")
    print(f"\n  Target maximum / zone  : <= {max_allowed} ({target_ratio*100:.0f}% of mean)")

    # --- 3. Mark sequences to remove (from overrepresented zones) ---
    to_remove = set()  # (match_id, player_id, frame_start_idx)
    total_removed = 0

    for zone in range(total_zones):
        current = zone_counts.get(zone, 0)
        if current <= max_allowed:
            continue  # Zone is balanced

        excess = current - max_allowed

        # Find all sequences for this zone
        zone_seqs = [(r[0], r[1]) for r in records if r[2] == zone]
        zone_seqs = list(set(zone_seqs))  # Unique (mid, pid)

        if not zone_seqs:
            continue

        # Randomly select sequences to mark for removal
        selected = np.random.choice(len(zone_seqs), size=min(excess, len(zone_seqs)), replace=False)
        for idx in selected:
            mid, pid = zone_seqs[idx]
            to_remove.add((mid, pid))
            total_removed += 1

    # --- 4. Filter out marked sequences ---
    balanced_rows = []
    for (mid, pid), grp in train_df.groupby(["match_id", "player_id"]):
        if (mid, pid) not in to_remove:
            balanced_rows.append(grp)

    if balanced_rows:
        balanced = pd.concat(balanced_rows, ignore_index=True)
    else:
        balanced = train_df.copy()

    # --- 5. Report ---
    print(f"\nDOWNSAMPLING COMPLETE")
    print(f"  Trajectories removed   : {total_removed:,}")
    print(f"  train_df rows: {len(train_df):,} --> {len(balanced):,}")
    if len(balanced) < len(train_df):
        reduction = (1 - len(balanced) / len(train_df)) * 100
        print(f"  Data reduction         : {reduction:.1f}%")
    print("=" * 60)
    return balanced




def check_class_imbalance_by_df(df,label_col="zone"):
    class_counts = df[label_col].value_counts().sort_index()
    class_percent = df[label_col].value_counts(normalize=True).sort_index() * 100

    distribution_df = pd.DataFrame({
        "Count": class_counts,
        "Percentage (%)": class_percent.round(2)
    })

    print("📊 Class Distribution:")
    print(distribution_df)
    largest_class = class_percent.idxmax()
    smallest_class = class_percent.idxmin()

    print("\n📈 Imbalance Summary:")
    print(f"Largest Class:  {largest_class} ({class_percent.max():.2f}%)")
    print(f"Smallest Class: {smallest_class} ({class_percent.min():.2f}%)")

    imbalance_ratio = class_counts.max() / class_counts.min()
    print(f"Imbalance Ratio (max/min): {imbalance_ratio:.2f}")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12,5))
    class_counts.plot(kind='bar')
    plt.title("Zone Class Distribution")
    plt.xlabel("Zone")
    plt.ylabel("Count")
    plt.show()



def check_memory_usage(threshold=80):
    """Check if memory usage exceeds threshold (%)"""
    memory = psutil.virtual_memory()
    usage_percent = memory.percent
    if usage_percent > threshold:
        print(f"\n⚠️  MEMORY WARNING: {usage_percent:.1f}% used (threshold: {threshold}%)")
        return True
    return False

def save_preprocessed_sequences(output_dir='preprocessed_data', chunk_size=500, memory_threshold=80):
    """Save preprocessed sequences with memory monitoring"""

    if SKIP_DATA_LOADING:
        print("⚠️  Data already loaded from preprocessed files. Skipping save.")
        return

    os.makedirs(output_dir, exist_ok=True)

    print("\n💾 Saving preprocessed sequences (memory-efficient mode)...")
    print(f"Backend: {BACKEND}")
    print(f"Memory threshold: {memory_threshold}%")

    # 1. Save scaler
    joblib.dump(scaler, f'{output_dir}/scaler.pkl')
    print(f"✓ Saved scaler")

    # 2. Save config
    config = {
        'seq_len': SEQ_LEN,
        'horizon_frames': HORIZON_FRAMES,
        'horizon_seconds': HORIZON_SECONDS,
        'n_rows': N_ROWS,
        'n_cols': N_COLS,
        'feature_cols': FEATURE_COLS,
        'coordinate_targets': CO_ORDINATES,
        'batch_size': BATCH_SIZE,
        'backend': BACKEND
    }
    joblib.dump(config, f'{output_dir}/config.pkl')
    print(f"✓ Saved config")

    if BACKEND == "keras":
        print("Saving Keras sequences with memory monitoring...")

        # Save dataframes first
        train_df.to_pickle(f'{output_dir}/train_df.pkl')
        val_df.to_pickle(f'{output_dir}/val_df.pkl')
        test_df.to_pickle(f'{output_dir}/test_df.pkl')
        print(f"✓ Saved dataframe splits")

        with h5py.File(f'{output_dir}/sequences.h5', 'w') as f:
            for split_name, split_df in [('train', train_df), ('val', val_df), ('test', test_df)]:
                print(f"\nProcessing {split_name} split...")

                # Check memory before starting
                if check_memory_usage(memory_threshold):
                    print(f"❌ Memory threshold exceeded before {split_name}. Stopping save.")
                    return False

                gen = keras_sequence_generator(
                    split_df, FEATURE_COLS, scaler, SEQ_LEN, HORIZON_FRAMES,
                    coordinate_targets=CO_ORDINATES, n_rows=N_ROWS, n_cols=N_COLS
                )

                X_dataset = None
                y_dataset = None
                total_sequences = 0
                batch = []

                for seq, target in gen:
                    # Check memory every chunk
                    if len(batch) > 0 and len(batch) % chunk_size == 0:
                        if check_memory_usage(memory_threshold):
                            print(f"\n❌ Memory threshold exceeded at {total_sequences} sequences for {split_name}")
                            print(f"   Partial data saved. Consider:")
                            print(f"   1. Reducing chunk_size (current: {chunk_size})")
                            print(f"   2. Increasing memory_threshold (current: {memory_threshold}%)")
                            print(f"   3. Using smaller dataset or more RAM")
                            return False

                    batch.append((seq, target))

                    if len(batch) >= chunk_size:
                        X_chunk = np.array([b[0] for b in batch], dtype=np.float32)
                        y_chunk = np.array([b[1] for b in batch], dtype=np.float32 if CO_ORDINATES else np.int32)

                        if X_dataset is None:
                            y_shape = (0, 2) if CO_ORDINATES else (0,)
                            y_maxshape = (None, 2) if CO_ORDINATES else (None,)

                            X_dataset = f.create_dataset(
                                f'X_{split_name}',
                                shape=(0, SEQ_LEN, len(FEATURE_COLS)),
                                maxshape=(None, SEQ_LEN, len(FEATURE_COLS)),
                                dtype=np.float32,
                                compression='gzip',
                                compression_opts=4
                            )

                            y_dataset = f.create_dataset(
                                f'y_{split_name}',
                                shape=y_shape,
                                maxshape=y_maxshape,
                                dtype=np.float32 if CO_ORDINATES else np.int32,
                                compression='gzip',
                                compression_opts=4
                            )

                        X_dataset.resize((total_sequences + len(X_chunk), SEQ_LEN, len(FEATURE_COLS)))
                        X_dataset[total_sequences:total_sequences+len(X_chunk)] = X_chunk

                        y_dataset.resize((total_sequences + len(y_chunk),) + y_chunk.shape[1:])
                        y_dataset[total_sequences:total_sequences+len(y_chunk)] = y_chunk

                        total_sequences += len(X_chunk)

                        mem_percent = psutil.virtual_memory().percent
                        print(f"  {split_name}: {total_sequences:,} sequences | RAM: {mem_percent:.1f}%", end='\r')

                        batch = []
                        del X_chunk, y_chunk
                        import gc
                        gc.collect()

                # Save remaining
                if batch:
                    X_chunk = np.array([b[0] for b in batch], dtype=np.float32)
                    y_chunk = np.array([b[1] for b in batch], dtype=np.float32 if CO_ORDINATES else np.int32)

                    X_dataset.resize((total_sequences + len(X_chunk), SEQ_LEN, len(FEATURE_COLS)))
                    X_dataset[total_sequences:total_sequences+len(X_chunk)] = X_chunk

                    y_dataset.resize((total_sequences + len(y_chunk),) + y_chunk.shape[1:])
                    y_dataset[total_sequences:total_sequences+len(y_chunk)] = y_chunk

                    total_sequences += len(X_chunk)
                    del X_chunk, y_chunk, batch
                    import gc
                    gc.collect()

                print(f"\n✓ Saved {split_name}: {total_sequences:,} sequences")

        print(f"\n✅ All Keras sequences saved successfully")
        return True


    print(f"\n✅ All data saved to '{output_dir}/'")
    return True

# Call with memory monitoring
# success = save_preprocessed_sequences(chunk_size=200, memory_threshold=80)
# if not success:
#     print("\n⚠️  Saving stopped due to memory constraints")




# ============================================================================
# GRAPH SEQUENCE GENERATOR FOR GCN MODELS
# ============================================================================

# def keras_graph_sequence_generator(
#     df, 
#     feature_cols, 
#     scaler, 
#     seq_len, 
#     horizon,
#     coord_cols,
#     coordinate_targets=False,
#     n_rows=None, 
#     n_cols=None,
#     k_teammates=5,
#     k_opponents=5,
#     k_edges=3,
#     normalize_adjacency=True
# ):
#     ""''
#     Generator that yields graph-structured sequences for GCN-TCN models.
    
#     For each sequence:
#     - Extracts k-NN spatial context (teammates + opponents) at each frame
#     - Builds node features and adjacency matrices
#     - Returns sequence of graphs over time
    
#     Args:
#         df: DataFrame with player tracking data
#         feature_cols: List of feature column names
#         scaler: Fitted StandardScaler (or None)
#         seq_len: Sequence length (number of frames)
#         horizon: Prediction horizon
#         coord_cols: Coordinate column names
#         coordinate_targets: If True, regression mode; if False, classification
#         n_rows, n_cols: Grid dimensions for zone classification
#         k_teammates: Number of nearest teammates to include
#         k_opponents: Number of nearest opponents to include
#         k_edges: Number of edges per node in k-NN graph
#         normalize_adjacency: Apply GCN normalization to adjacency
        
#     Yields:
#         ((node_features_seq, adjacency_seq, mask_seq), target, sample_weight)
#         - node_features_seq: [seq_len, N_nodes, num_features]
#         - adjacency_seq: [seq_len, N_nodes, N_nodes]
#         - mask_seq: [seq_len, N_nodes] - binary mask for padding
#         - target: zone_id (int32) or coordinates (float32)
#         - sample_weight: float32
#     ""''
#     from graph_utils import build_graph_snapshot
    
#     N_nodes = 1 + k_teammates + k_opponents  # Fixed graph size
#     num_features = len(feature_cols)
    
#     for _, player_df in df.groupby(["match_id", "player_id"]):
#         player_df = player_df.sort_values("frame_number").reset_index(drop=True)
        
#         # Get match_id and player_id for this sequence
#         match_id = player_df["match_id"].iloc[0]
#         player_id = player_df["player_id"].iloc[0]
        
#         # Get target zones if classification mode
#         if not coordinate_targets:
#             zones = xy_to_zone_vectorized(
#                 player_df["x_normalized"].values,
#                 player_df["y_normalized"].values,
#                 n_rows, n_cols
#             )
        
#         # Generate sequences
#         for i in range(len(player_df) - seq_len - horizon + 1):
#             # Initialize arrays for sequence of graphs
#             node_features_seq = np.zeros((seq_len, N_nodes, num_features), dtype=np.float32)
#             adjacency_seq = np.zeros((seq_len, N_nodes, N_nodes), dtype=np.float32)
#             mask_seq = np.zeros((seq_len, N_nodes), dtype=np.float32)
            
#             # Build graph for each frame in sequence
#             for t in range(seq_len):
#                 frame_idx = i + t
#                 frame_num = player_df.iloc[frame_idx]["frame_number"]
                
#                 # Extract graph snapshot for this frame
#                 node_feat, adjacency, mask = build_graph_snapshot(
#                     df=df,
#                     match_id=match_id,
#                     player_id=player_id,
#                     frame_num=frame_num,
#                     feature_cols=feature_cols,
#                     k_teammates=k_teammates,
#                     k_opponents=k_opponents,
#                     k_edges=k_edges,
#                     pad_to_fixed_size=True,
#                     normalize_adjacency=normalize_adjacency
#                 )
                
#                 # Apply scaler to node features if provided
#                 if scaler is not None:
#                     # Only scale real nodes (use mask)
#                     real_nodes = mask > 0
#                     if np.any(real_nodes):
#                         node_feat[real_nodes] = scaler.transform(node_feat[real_nodes])
                
#                 # Store in sequence
#                 node_features_seq[t] = node_feat
#                 adjacency_seq[t] = adjacency
#                 mask_seq[t] = mask
            
#             # Determine target
#             if coordinate_targets:
#                 # Regression mode: predict coordinates
#                 t_curr = i + seq_len - 1
#                 t_fut = t_curr + horizon
#                 coords = player_df[list(coord_cols)].values.astype(np.float32)
#                 target = coords[t_fut]
#             else:
#                 # Classification mode: predict zone
#                 t_fut = i + seq_len - 1 + horizon
#                 target = np.int32(zones[t_fut])
            
#             # Sample weight (can be modified for class balancing)
#             sample_weight = np.float32(1.0)
            
#             yield (node_features_seq, adjacency_seq, mask_seq), target, sample_weight


# def create_graph_tf_dataset(
#     df,
#     feature_cols,
#     scaler,
#     seq_len,
#     horizon,
#     coord_cols,
#     coordinate_targets=False,
#     n_rows=None,
#     n_cols=None,
#     k_teammates=5,
#     k_opponents=5,
#     k_edges=3,
#     batch_size=32,
#     shuffle=True,
#     prefetch=2
# ):
#     ""''
#     Create TensorFlow Dataset from graph sequence generator.
    
#     Returns:
#         tf.data.Dataset with batched graph sequences
#     ""''
#     N_nodes = 1 + k_teammates + k_opponents
#     num_features = len(feature_cols)
    
#     # Define output signature
#     if coordinate_targets:
#         target_spec = tf.TensorSpec(shape=(2,), dtype=tf.float32)
#     else:
#         target_spec = tf.TensorSpec(shape=(), dtype=tf.int32)
    
#     output_signature = (
#         (
#             tf.TensorSpec(shape=(seq_len, N_nodes, num_features), dtype=tf.float32),  # node_features
#             tf.TensorSpec(shape=(seq_len, N_nodes, N_nodes), dtype=tf.float32),      # adjacency
#             tf.TensorSpec(shape=(seq_len, N_nodes), dtype=tf.float32),               # mask
#         ),
#         target_spec,
#         tf.TensorSpec(shape=(), dtype=tf.float32)  # sample_weight
#     )
    
#     # Create dataset from generator
#     dataset = tf.data.Dataset.from_generator(
#         lambda: keras_graph_sequence_generator(
#             df, feature_cols, scaler, seq_len, horizon, coord_cols,
#             coordinate_targets, n_rows, n_cols,
#             k_teammates, k_opponents, k_edges
#         ),
#         output_signature=output_signature
#     )
    
#     if shuffle:
#         dataset = dataset.shuffle(buffer_size=10000)
    
#     dataset = dataset.batch(batch_size, drop_remainder=False)
#     dataset = dataset.prefetch(prefetch)
    
#     return dataset

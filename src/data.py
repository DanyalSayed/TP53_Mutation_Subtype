"""Data loading, schema verification, and sequence encoding for the TP53
mutation-subtype CNN experiments (notebooks 01_main_experiment,
02_window_ablation).

Split CSVs are produced by notebooks/data_pipeline.ipynb (notebook 00) at
data/splits/window_{W}/{hotspot,rare}/{train,val,test}.csv with columns
exactly: position_id, Sequence, MutationType. The canonical position table
(data/processed/position_table.csv) has columns: position_id, gene_number,
cds_pos, n_instances, majority_subtype, n_distinct_subtypes, shannon_entropy,
hotspot_flag -- it does NOT carry a Sequence column, so position-level
sequence-derived flags (e.g. is_cpg) are computed by joining against a split
file's Sequence column rather than from the position table alone.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# Fixed class order used everywhere labels are turned into indices, so index
# i always means CLASSES[i] across data.py, model outputs, metrics, and the
# predictions.parquet "probabilities" column.
CLASSES = ['C>A', 'C>G', 'C>T', 'T>A', 'T>C', 'T>G']
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

# Fixed one-hot channel order, as specified: A=0, C=1, G=2, T=3.
NUCLEOTIDE_ORDER = 'ACGT'
_TRANSLATE_TABLE = str.maketrans('ACGT', '\x00\x01\x02\x03')


def load_split(path):
    """Load one split CSV and verify it has the expected schema.

    Returns the DataFrame unchanged (position_id, Sequence, MutationType),
    but raises if the schema or content doesn't match what notebook 00
    is documented to produce.
    """
    df = pd.read_csv(path, dtype={'position_id': str})

    expected_cols = {'position_id', 'Sequence', 'MutationType'}
    assert set(df.columns) == expected_cols, (
        f"{path}: expected columns {expected_cols}, got {set(df.columns)}"
    )
    assert df['Sequence'].str.contains('N').sum() == 0, (
        f"{path}: found 'N'-padded sequences -- notebook 00's 101bp filter "
        "should guarantee every smaller window is also N-free."
    )
    unknown_labels = set(df['MutationType'].unique()) - set(CLASSES)
    assert not unknown_labels, f"{path}: unexpected MutationType values {unknown_labels}"

    seq_lens = df['Sequence'].str.len().unique()
    assert len(seq_lens) == 1, f"{path}: mixed sequence lengths {seq_lens}"

    return df


def load_position_table(path):
    """Load the QC-only position-level table and verify its schema.

    No `hotspot_flag` here -- hotspot/rare is a cluster-level property (see
    load_cluster_table), since isoform-redundant COSMIC re-annotation means
    raw positions fragment a locus's true instance count. Join on
    `cluster_id` for a position's hotspot status.
    """
    df = pd.read_csv(path, dtype={'position_id': str, 'cluster_id': str})
    expected_cols = {
        'position_id', 'gene_number', 'cds_pos', 'cluster_id', 'n_instances',
        'majority_subtype', 'n_distinct_subtypes', 'shannon_entropy',
    }
    assert set(df.columns) == expected_cols, (
        f"{path}: expected columns {expected_cols}, got {set(df.columns)}"
    )
    return df


def load_cluster_table(path):
    """Load the canonical cluster-level table and verify its schema.

    One row per locus cluster (position_ids sharing identical sequence at
    the smallest window size). This -- not the position table -- is where
    `hotspot_flag`, and the counts it's derived from, are correctly defined.
    """
    df = pd.read_csv(path, dtype={'cluster_id': str})
    expected_cols = {
        'cluster_id', 'n_positions', 'n_instances', 'majority_subtype',
        'n_distinct_subtypes', 'shannon_entropy', 'hotspot_flag',
    }
    assert set(df.columns) == expected_cols, (
        f"{path}: expected columns {expected_cols}, got {set(df.columns)}"
    )
    return df


def encode_sequences(sequences):
    """One-hot encode a list/Series of equal-length ACGT strings.

    Returns a float32 array of shape (N, 4, L), channel order A=0, C=1,
    G=2, T=3. Raises if any character outside ACGT is present.
    """
    sequences = list(sequences)
    seq_len = len(sequences[0])
    translated = [s.translate(_TRANSLATE_TABLE).encode('latin1') for s in sequences]
    idx = np.frombuffer(b''.join(translated), dtype=np.uint8).reshape(len(sequences), seq_len)

    if idx.max() > 3:
        bad = np.where(idx > 3)
        raise ValueError(
            f"Non-ACGT character encountered while encoding sequences "
            f"(row {bad[0][0]}, position {bad[1][0]})."
        )

    one_hot = np.eye(4, dtype=np.float32)[idx]  # (N, L, 4)
    return one_hot.transpose(0, 2, 1)  # (N, 4, L)


def encode_labels(mutation_types):
    """Map MutationType strings to fixed integer indices per CLASSES."""
    return np.array([CLASS_TO_IDX[m] for m in mutation_types], dtype=np.int64)


class MutationDataset(Dataset):
    """One-hot encoded sequence + integer label pairs for SimpleCNN."""

    def __init__(self, df):
        x = encode_sequences(df['Sequence'].values)
        y = encode_labels(df['MutationType'].values)
        self.X = torch.from_numpy(x)
        self.y = torch.from_numpy(y)
        self.position_ids = df['position_id'].values

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def compute_is_cpg(sequences, center_idx=10):
    """CpG-context flag: sequence[center_idx] == 'C' and sequence[center_idx+1] == 'G'.

    center_idx=10 is the mutated base's position in a 21bp window
    (0-indexed), i.e. the CpG rule is only meaningful at window_size=21.
    """
    seq_array = np.asarray(sequences, dtype=str)
    return np.array([
        len(s) > center_idx + 1 and s[center_idx] == 'C' and s[center_idx + 1] == 'G'
        for s in seq_array
    ])


def position_majority_ceiling(df, id_col='position_id', label_col='MutationType'):
    """Bayes-optimal accuracy ceiling given the test set's own per-position
    label distribution: sum each position's majority-label count and divide
    by total instances (the instance-weighted average of each position's
    majority-label fraction).

    Computed from `df` directly (a test split) -- the canonical position
    table's majority_subtype/n_instances are aggregated over train+val+test,
    not the test set alone, so it can't substitute here.
    """
    majority_counts = df.groupby(id_col)[label_col].agg(lambda s: s.value_counts().max())
    return float(majority_counts.sum() / len(df))

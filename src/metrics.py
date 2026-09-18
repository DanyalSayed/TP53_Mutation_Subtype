"""Metrics, baselines, and statistical tests shared by
notebooks/01_main_experiment.ipynb and notebooks/02_window_ablation.ipynb.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, matthews_corrcoef,
    f1_score, precision_recall_fscore_support, confusion_matrix,
)
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from scipy.stats import wilcoxon
from statsmodels.stats.contingency_tables import mcnemar

from src.data import CLASSES, CLASS_TO_IDX, compute_is_cpg


# ---------------------------------------------------------------------------
# Core classification metrics
# ---------------------------------------------------------------------------

def bootstrap_accuracy_ci(y_true, y_pred, n_boot=1000, ci=0.95, seed=42):
    """Bootstrap resampling CI for accuracy. Returns (point_accuracy, low, high)."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)

    point_acc = accuracy_score(y_true, y_pred)
    boot_accs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_accs[i] = accuracy_score(y_true[idx], y_pred[idx])

    alpha = 1 - ci
    low, high = np.quantile(boot_accs, [alpha / 2, 1 - alpha / 2])
    return float(point_acc), float(low), float(high)


def bootstrap_accuracy_ci_clustered(y_true, y_pred, cluster_ids, n_boot=1000, ci=0.95, seed=42):
    """Bootstrap resampling CI for accuracy, resampling whole clusters (loci)
    with replacement rather than individual instances.

    Instances sharing a cluster are not independent draws -- they share
    identical (or near-identical) local sequence context by construction
    (see notebooks/data_pipeline.ipynb Stage 5) -- so instance-level
    resampling (bootstrap_accuracy_ci) understates the true uncertainty.
    Each bootstrap iteration resamples the set of clusters, then includes
    every instance belonging to each resampled cluster (with its multiplicity
    if a cluster is drawn more than once), preserving each cluster's own
    instance count and composition.
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    cluster_ids = np.asarray(cluster_ids)
    assert len(cluster_ids) == len(y_true), "cluster_ids must align 1:1 with y_true/y_pred"

    unique_clusters = np.unique(cluster_ids)
    n_clusters = len(unique_clusters)
    cluster_to_idx = {c: np.where(cluster_ids == c)[0] for c in unique_clusters}

    point_acc = accuracy_score(y_true, y_pred)
    boot_accs = np.empty(n_boot)
    for i in range(n_boot):
        sampled_clusters = rng.choice(unique_clusters, size=n_clusters, replace=True)
        idx = np.concatenate([cluster_to_idx[c] for c in sampled_clusters])
        boot_accs[i] = accuracy_score(y_true[idx], y_pred[idx])

    alpha = 1 - ci
    low, high = np.quantile(boot_accs, [alpha / 2, 1 - alpha / 2])
    return float(point_acc), float(low), float(high)


def compute_all_metrics(y_true, y_pred, classes=CLASSES, n_boot=1000, seed=42, cluster_ids=None):
    """Accuracy (+95% bootstrap CI), balanced accuracy, MCC, macro/weighted
    F1, per-class precision/recall/F1, and the confusion matrix.

    If `cluster_ids` is given (one cluster/locus id per instance, aligned to
    y_true/y_pred), also computes a cluster-resampled accuracy CI
    (accuracy_ci_low_clustered/accuracy_ci_high_clustered) alongside the
    instance-level one, since the latter understates uncertainty when many
    instances share a locus (see bootstrap_accuracy_ci_clustered).
    """
    acc, ci_low, ci_high = bootstrap_accuracy_ci(y_true, y_pred, n_boot=n_boot, seed=seed)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=np.arange(len(classes)), zero_division=0
    )
    per_class = {
        cls: {
            'precision': float(precision[i]),
            'recall': float(recall[i]),
            'f1': float(f1[i]),
            'support': int(support[i]),
        }
        for i, cls in enumerate(classes)
    }

    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(classes)))

    result = {
        'accuracy': acc,
        'accuracy_ci_low': ci_low,
        'accuracy_ci_high': ci_high,
        'balanced_accuracy': float(balanced_accuracy_score(y_true, y_pred)),
        'mcc': float(matthews_corrcoef(y_true, y_pred)),
        'macro_f1': float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
        'weighted_f1': float(f1_score(y_true, y_pred, average='weighted', zero_division=0)),
        'per_class': per_class,
        'confusion_matrix': cm.tolist(),
    }

    if cluster_ids is not None:
        _, cci_low, cci_high = bootstrap_accuracy_ci_clustered(
            y_true, y_pred, cluster_ids, n_boot=n_boot, seed=seed
        )
        result['accuracy_ci_low_clustered'] = cci_low
        result['accuracy_ci_high_clustered'] = cci_high

    return result


def plot_confusion_matrix(cm, classes, title, out_path):
    cm = np.asarray(cm)
    cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha='right')
    ax.set_yticklabels(classes)
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label='Row-normalized fraction')

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                     color='white' if cm_norm[i, j] > 0.5 else 'black', fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

def random_baseline(n, classes=CLASSES, seed=42):
    rng = np.random.default_rng(seed)
    return rng.integers(0, len(classes), size=n)


def majority_baseline(train_labels, n_test, classes=CLASSES):
    """Returns (predictions_array, majority_class_str)."""
    values, counts = np.unique(train_labels, return_counts=True)
    majority_idx = values[np.argmax(counts)]
    majority_class = classes[majority_idx]
    return np.full(n_test, majority_idx, dtype=np.int64), majority_class


def cpg_baseline(sequences, majority_idx, classes=CLASSES, center_idx=10):
    """Predict C>T where the window is in CpG context, else the majority class."""
    is_cpg = compute_is_cpg(sequences, center_idx=center_idx)
    preds = np.full(len(sequences), majority_idx, dtype=np.int64)
    preds[is_cpg] = CLASS_TO_IDX['C>T']
    return preds, is_cpg


def trinucleotide_features(train_sequences, test_sequences, k=3):
    """Char-ngram (k=3) frequency features, fit on train, applied to test."""
    vectorizer = CountVectorizer(analyzer='char', ngram_range=(k, k), lowercase=False)
    X_train_counts = vectorizer.fit_transform(train_sequences)
    X_test_counts = vectorizer.transform(test_sequences)

    X_train = X_train_counts.toarray().astype(np.float64)
    X_test = X_test_counts.toarray().astype(np.float64)
    X_train /= X_train.sum(axis=1, keepdims=True).clip(min=1)
    X_test /= X_test.sum(axis=1, keepdims=True).clip(min=1)
    return X_train, X_test, vectorizer


def logistic_regression_baseline(train_sequences, train_labels, test_sequences, seed=42):
    """Trinucleotide-frequency logistic regression, fit on train, predicted on test."""
    X_train, X_test, _ = trinucleotide_features(train_sequences, test_sequences)
    clf = LogisticRegression(max_iter=2000, random_state=seed)
    clf.fit(X_train, train_labels)
    return clf.predict(X_test)


# ---------------------------------------------------------------------------
# Statistical testing
# ---------------------------------------------------------------------------

def mcnemar_test(y_true, pred_a, pred_b):
    """McNemar's test comparing two paired classifiers' correctness on the
    same test set. Uses the chi-square approximation with continuity
    correction (appropriate for the large sample sizes here; the exact
    binomial variant is only tractable/needed for small N).
    """
    correct_a = (np.asarray(pred_a) == np.asarray(y_true))
    correct_b = (np.asarray(pred_b) == np.asarray(y_true))

    both_correct = int(np.sum(correct_a & correct_b))
    a_only = int(np.sum(correct_a & ~correct_b))
    b_only = int(np.sum(~correct_a & correct_b))
    both_wrong = int(np.sum(~correct_a & ~correct_b))

    table = [[both_correct, a_only], [b_only, both_wrong]]
    result = mcnemar(table, exact=False, correction=True)
    return {
        'statistic': float(result.statistic),
        'pvalue': float(result.pvalue),
        'contingency_table': table,
    }


def cluster_paired_bootstrap_test(y_true, pred_a, pred_b, cluster_ids, n_boot=1000, ci=0.95, seed=42):
    """Cluster-resampled paired bootstrap test for the accuracy difference
    between two classifiers on the same test set.

    Replaces mcnemar_test's role under locus clustering: mcnemar_test treats
    every instance as an independent paired trial, but instances sharing a
    cluster share (near-)identical sequence context and are not independent,
    which inflates its chi-square statistic and understates its p-value when
    a small number of loci contribute a large number of instances (e.g. the
    hotspot test set: 112,511 instances from only 70 loci). This resamples
    whole clusters with replacement instead, so the resampling unit matches
    the study's own defined independence unit (the locus).

    Returns the observed accuracy_a - accuracy_b difference, a percentile
    95% CI on that difference, and a two-sided bootstrap p-value (twice the
    smaller tail proportion on either side of zero, capped at 1.0).
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    pred_a = np.asarray(pred_a)
    pred_b = np.asarray(pred_b)
    cluster_ids = np.asarray(cluster_ids)
    assert len(cluster_ids) == len(y_true) == len(pred_a) == len(pred_b)

    unique_clusters = np.unique(cluster_ids)
    n_clusters = len(unique_clusters)
    cluster_to_idx = {c: np.where(cluster_ids == c)[0] for c in unique_clusters}

    correct_a = (pred_a == y_true)
    correct_b = (pred_b == y_true)
    observed_diff = float(correct_a.mean() - correct_b.mean())

    boot_diffs = np.empty(n_boot)
    for i in range(n_boot):
        sampled_clusters = rng.choice(unique_clusters, size=n_clusters, replace=True)
        idx = np.concatenate([cluster_to_idx[c] for c in sampled_clusters])
        boot_diffs[i] = correct_a[idx].mean() - correct_b[idx].mean()

    alpha = 1 - ci
    ci_low, ci_high = np.quantile(boot_diffs, [alpha / 2, 1 - alpha / 2])

    prop_le_zero = float(np.mean(boot_diffs <= 0))
    prop_ge_zero = float(np.mean(boot_diffs >= 0))
    pvalue = min(1.0, 2 * min(prop_le_zero, prop_ge_zero))

    return {
        'observed_diff': observed_diff,
        'ci_low': float(ci_low),
        'ci_high': float(ci_high),
        'pvalue': pvalue,
        'n_clusters': int(n_clusters),
        'n_boot': int(n_boot),
    }


def cluster_wilcoxon_paired_test(y_true, pred_a, pred_b, cluster_ids):
    """Wilcoxon signed-rank test on paired per-cluster accuracy (classifier
    A vs. B), as a secondary, non-parametric confirmatory check alongside
    cluster_paired_bootstrap_test -- same non-parametric, cluster-respecting
    style already used for the CpG comparison (see the position/cluster-level
    Mann-Whitney U test in notebooks/04_cpg_analysis.ipynb), extended here to
    a paired (same test set, two classifiers) rather than unpaired setting.

    Clusters where A and B tie exactly (zero-difference pairs) are dropped
    by scipy's default Wilcoxon handling; this is noted in the returned
    `n_clusters_used` vs. `n_clusters_total`.
    """
    y_true = np.asarray(y_true)
    pred_a = np.asarray(pred_a)
    pred_b = np.asarray(pred_b)
    cluster_ids = np.asarray(cluster_ids)

    correct_a = (pred_a == y_true).astype(np.float64)
    correct_b = (pred_b == y_true).astype(np.float64)

    unique_clusters = np.unique(cluster_ids)
    acc_a = np.array([correct_a[cluster_ids == c].mean() for c in unique_clusters])
    acc_b = np.array([correct_b[cluster_ids == c].mean() for c in unique_clusters])

    n_nonzero = int(np.sum(acc_a != acc_b))
    if n_nonzero == 0:
        return {
            'statistic': None,
            'pvalue': 1.0,
            'n_clusters_total': int(len(unique_clusters)),
            'n_clusters_used': 0,
            'note': 'every cluster tied exactly (accuracy_a == accuracy_b); no test performed',
        }

    statistic, pvalue = wilcoxon(acc_a, acc_b, zero_method='wilcox')
    return {
        'statistic': float(statistic),
        'pvalue': float(pvalue),
        'n_clusters_total': int(len(unique_clusters)),
        'n_clusters_used': n_nonzero,
    }

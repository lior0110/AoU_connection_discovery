import os
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy, wasserstein_distance
from sklearn.preprocessing import StandardScaler


# =============================================================================
# 1. DATA PREPROCESSING & CLEANING
# =============================================================================

def trim_by_quantile(
    s: pd.Series, 
    q_lower: float = 0.05, 
    q_upper: float = 0.95
) -> pd.Series:
    """Trims a pandas Series to retain values within specified quantile bounds.

    Parameters
    ----------
    s : pd.Series
        Input series to be trimmed.
    q_lower : float, default=0.05
        Lower quantile threshold (0.0 to 1.0).
    q_upper : float, default=0.95
        Upper quantile threshold (0.0 to 1.0).

    Returns
    -------
    pd.Series
        Trimmed pandas Series excluding missing values and extreme quantile outliers.
    """
    clean_s = s.dropna()
    if clean_s.empty:
        return clean_s

    q_min = clean_s.quantile(q_lower)
    q_max = clean_s.quantile(q_upper)

    return clean_s[(clean_s >= q_min) & (clean_s <= q_max)]


# =============================================================================
# 2. STATISTICAL DISTANCE & INFORMATION METRICS
# =============================================================================

def compute_kl_divergence(
    v1: np.ndarray, 
    v2: np.ndarray, 
    num_bins: int = 50, 
    eps: float = 1e-10
) -> float:
    """Computes symmetric Kullback-Leibler (KL) Divergence using histogram binning.

    Parameters
    ----------
    v1, v2 : np.ndarray
        1D numerical arrays representing empirical distributions.
    num_bins : int, default=50
        Number of histogram bins used for discrete probability estimation.
    eps : float, default=1e-10
        Epsilon smoothing value to prevent log(0) or division by zero.

    Returns
    -------
    float
        Symmetric KL divergence score calculated as 0.5 * (KL(P||Q) + KL(Q||P)).
    """
    min_val = min(v1.min(), v2.min())
    max_val = max(v1.max(), v2.max())

    if min_val == max_val:
        return 0.0

    bins = np.linspace(min_val, max_val, num_bins + 1)
    p_counts, _ = np.histogram(v1, bins=bins)
    q_counts, _ = np.histogram(v2, bins=bins)

    p_dist = (p_counts + eps) / np.sum(p_counts + eps)
    q_dist = (q_counts + eps) / np.sum(q_counts + eps)

    kl_pq = entropy(p_dist, q_dist)
    kl_qp = entropy(q_dist, p_dist)

    return float(0.5 * (kl_pq + kl_qp))


def compute_normalized_jsd(
    v1: np.ndarray, 
    v2: np.ndarray, 
    num_bins: int = 50
) -> float:
    """Computes Jensen-Shannon Divergence (JSD) bounded strictly in [0, 1].

    Parameters
    ----------
    v1, v2 : np.ndarray
        1D numerical arrays representing empirical distributions.
    num_bins : int, default=50
        Histogram resolution for discrete probability estimation.

    Returns
    -------
    float
        JSD score (0.0 = identical distributions, 1.0 = disjoint distributions).
    """
    min_val, max_val = min(v1.min(), v2.min()), max(v1.max(), v2.max())
    if min_val == max_val:
        return 0.0

    bins = np.linspace(min_val, max_val, num_bins + 1)
    p_counts, _ = np.histogram(v1, bins=bins)
    q_counts, _ = np.histogram(v2, bins=bins)

    eps = 1e-10
    p_dist = (p_counts + eps) / np.sum(p_counts + eps)
    q_dist = (q_counts + eps) / np.sum(q_counts + eps)

    # SciPy jensenshannon returns sqrt(JSD) using base 2
    js_distance = jensenshannon(p_dist, q_dist, base=2)
    return float(js_distance**2)


def get_columns_perplexity(df: pd.DataFrame, n_bins: int = -1) -> pd.Series:
    """Calculates the perplexity (exponential of Shannon entropy) of each column.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.
    n_bins : int, default=-1
        Equal-width bin count for continuous numerical columns. 
        If -1, evaluates original discrete values without binning.

    Returns
    -------
    pd.Series
        Perplexity score per column (1.0 indicates effectively constant feature).
    """
    results = {}

    for col in df.columns:
        s = df[col].dropna()

        if s.empty:
            results[col] = 0.0
            continue

        try:
            if n_bins > 0 and pd.api.types.is_numeric_dtype(s):
                s = pd.cut(s, bins=n_bins)

            probs = s.value_counts(normalize=True).to_numpy()
            probs = probs[probs > 0]  # Exclude empty bins
            shannon_entropy = -np.sum(probs * np.log(probs)) if len(probs) > 0 else 0.0
            results[col] = float(np.exp(shannon_entropy))
        except Exception as e:
            print(f"Error computing perplexity for column '{col}': {e}")
            results[col] = 0.0

    return pd.Series(results, dtype=float)


# =============================================================================
# 3. DISTRIBUTION CLOSENESS & LAB UNIT AUDITING
# =============================================================================

def evaluate_distribution_closeness(
    s1: pd.Series,
    s2: pd.Series,
    q_lower: float = 0.10,
    q_upper: float = 0.90,
    q_threshold_ratio: float = 0.25,
    s_threshold_ratio: float = 2.0,
    jsd_threshold: float = 0.25,
    num_bins: int = 50,
    use_rule: str = "one_of",
    trim_outliers: bool = True,
    quantile_trim_bounds: Tuple[float, float] = (0.05, 0.95),
) -> Dict[str, Union[float, bool]]:
    """Evaluates whether two distributions are close using dual-scale analysis.

    Compares distributions across Linear and Log scales using relative Earth 
    Mover's Distance (EMD) and Normalized Jensen-Shannon Divergence (JSD).

    Parameters
    ----------
    s1, s2 : pd.Series
        Numerical series containing measurement distributions.
    q_lower, q_upper : float, default=(0.10, 0.90)
        Percentile bounds for inter-percentile spread computation.
    q_threshold_ratio : float, default=0.25
        Maximum acceptable ratio of EMD to pooled spread.
    s_threshold_ratio : float, default=2.0
        Maximum acceptable ratio of EMD to self-EMD baseline noise floor.
    jsd_threshold : float, default=0.25
        Maximum acceptable normalized JSD score [0, 1].
    num_bins : int, default=50
        Histogram resolution for JSD evaluation.
    use_rule : str, default='one_of'
        Rule for closeness criteria ('one_of', 'all_of', or 'emd_and_jsd').
    trim_outliers : bool, default=True
        Whether to trim outliers before distance calculation.
    quantile_trim_bounds : Tuple[float, float], default=(0.05, 0.95)
        Quantile bounds used during outlier trimming.

    Returns
    -------
    dict
        Dictionary containing raw EMD, relative EMD, JSD, noise baselines, and closeness flags.
    """
    # Clean missing values and optionally trim to quantile band
    s1_clean = s1.dropna()
    s2_clean = s2.dropna()

    if trim_outliers:
        s1_trimmed = trim_by_quantile(s1_clean, quantile_trim_bounds[0], quantile_trim_bounds[1])
        s2_trimmed = trim_by_quantile(s2_clean, quantile_trim_bounds[0], quantile_trim_bounds[1])
    else:
        s1_trimmed = s1_clean
        s2_trimmed = s2_clean

    v1_lin = s1_trimmed.to_numpy()
    v2_lin = s2_trimmed.to_numpy()

    if len(v1_lin) == 0 or len(v2_lin) == 0:
        raise ValueError("One or both Series contain no valid observations after trimming.")

    # -------------------------------------------------------------------------
    # LINEAR SCALE EVALUATION
    # -------------------------------------------------------------------------
    emd_lin = wasserstein_distance(v1_lin, v2_lin)

    spread1_lin = s1_clean.quantile(q_upper) - s1_clean.quantile(q_lower)
    spread2_lin = s2_clean.quantile(q_upper) - s2_clean.quantile(q_lower)
    pooled_spread_lin = (spread1_lin + spread2_lin) / 2.0

    rel_emd_lin = emd_lin / pooled_spread_lin if pooled_spread_lin > 0 else np.nan
    jsd_lin = compute_normalized_jsd(v1_lin, v2_lin, num_bins=num_bins)

    # Self-noise baseline calculation via random split permutation
    rng = np.random.default_rng(seed=42)
    half_size = len(v1_lin) // 2
    shuffled_v1 = rng.permutation(v1_lin)
    self_emd_baseline = (
        wasserstein_distance(shuffled_v1[:half_size], shuffled_v1[half_size:])
        if half_size > 0
        else 0.0
    )

    if use_rule == "one_of":
        is_close_lin = (
            (rel_emd_lin < q_threshold_ratio)
            or (emd_lin < s_threshold_ratio * self_emd_baseline)
            or (jsd_lin < jsd_threshold)
        )
    elif use_rule == "all_of":
        is_close_lin = (
            (rel_emd_lin < q_threshold_ratio)
            and (emd_lin < s_threshold_ratio * self_emd_baseline)
            and (jsd_lin < jsd_threshold)
        )
    elif use_rule == "emd_and_jsd":
        is_close_lin = (
            ((rel_emd_lin < q_threshold_ratio) or (emd_lin < s_threshold_ratio * self_emd_baseline))
            and (jsd_lin < jsd_threshold)
        )
    else:
        raise ValueError(f"Invalid use_rule '{use_rule}'. Options: 'one_of', 'all_of', 'emd_and_jsd'.")

    # -------------------------------------------------------------------------
    # LOG SCALE EVALUATION
    # -------------------------------------------------------------------------
    v1_pos = v1_lin[v1_lin > 0]
    v2_pos = v2_lin[v2_lin > 0]

    if len(v1_pos) > 0 and len(v2_pos) > 0:
        v1_log = np.log(v1_pos)
        v2_log = np.log(v2_pos)

        emd_log = wasserstein_distance(v1_log, v2_log)

        s1_pos, s2_pos = s1_clean[s1_clean > 0], s2_clean[s2_clean > 0]
        spread1_log = np.log(s1_pos.quantile(q_upper)) - np.log(s1_pos.quantile(q_lower))
        spread2_log = np.log(s2_pos.quantile(q_upper)) - np.log(s2_pos.quantile(q_lower))
        pooled_spread_log = (spread1_log + spread2_log) / 2.0

        rel_emd_log = emd_log / pooled_spread_log if pooled_spread_log > 0 else np.nan
        jsd_log = compute_normalized_jsd(v1_log, v2_log, num_bins=num_bins)

        half_size_log = len(v1_log) // 2
        shuffled_v1_log = rng.permutation(v1_log)
        self_emd_log_baseline = (
            wasserstein_distance(shuffled_v1_log[:half_size_log], shuffled_v1_log[half_size_log:])
            if half_size_log > 0
            else 0.0
        )

        if use_rule == "one_of":
            is_close_log = (
                (rel_emd_log < q_threshold_ratio)
                or (emd_log < s_threshold_ratio * self_emd_log_baseline)
                or (jsd_log < jsd_threshold)
            )
        elif use_rule == "all_of":
            is_close_log = (
                (rel_emd_log < q_threshold_ratio)
                and (emd_log < s_threshold_ratio * self_emd_log_baseline)
                and (jsd_log < jsd_threshold)
            )
        elif use_rule == "emd_and_jsd":
            is_close_log = (
                ((rel_emd_log < q_threshold_ratio) or (emd_log < s_threshold_ratio * self_emd_log_baseline))
                and (jsd_log < jsd_threshold)
            )
        else:
            raise ValueError(f"Invalid use_rule '{use_rule}'. Options: 'one_of', 'all_of', 'emd_and_jsd'.")
    else:
        emd_log = pooled_spread_log = self_emd_log_baseline = rel_emd_log = jsd_log = np.nan
        is_close_log = False

    is_close_dual_scale = is_close_lin or is_close_log

    return {
        "raw_emd_linear": emd_lin,
        "pooled_spread_linear": pooled_spread_lin,
        "self_noise_baseline": self_emd_baseline,
        "relative_emd_linear": rel_emd_lin,
        "jsd_linear": jsd_lin,
        "is_close_linear": is_close_lin,
        "raw_emd_log": emd_log,
        "pooled_spread_log": pooled_spread_log,
        "self_noise_log_baseline": self_emd_log_baseline,
        "relative_emd_log": rel_emd_log,
        "jsd_log": jsd_log,
        "is_close_log": is_close_log,
        "is_close_dual_scale": is_close_dual_scale,
        "alert_mismatch": not is_close_dual_scale,
    }


def audit_lab_unit_distributions_dual_scale(
    df: pd.DataFrame,
    min_sample_size: int = 15,
    emd_q_threshold: float = 0.25,
    emd_s_threshold: float = 2.0,
    jsd_threshold: float = 0.25,
    use_rule: str = "one_of",
    trim_outliers: bool = True,
    quantile_trim_bounds: Tuple[float, float] = (0.05, 0.95),
) -> pd.DataFrame:
    """Audits unit distributions across both Linear and Log scales for clinical lab tests.

    An alert flag is triggered ONLY IF distributions fail closeness thresholds across both scales.

    Parameters
    ----------
    df : pd.DataFrame
        Laboratory DataFrame containing 'laboratory_test_name', 'raw_value', and 'raw_unit'.
    min_sample_size : int, default=15
        Minimum valid observations required per unit group to participate in audit.
    emd_q_threshold : float, default=0.25
        EMD to spread threshold ratio.
    emd_s_threshold : float, default=2.0
        EMD to baseline noise floor ratio.
    jsd_threshold : float, default=0.25
        JSD threshold.
    use_rule : str, default='one_of'
        Closeness decision rule ('one_of', 'all_of', or 'emd_and_jsd').
    trim_outliers : bool, default=True
        Whether to trim quantile outliers.
    quantile_trim_bounds : Tuple[float, float], default=(0.05, 0.95)
        Lower and upper quantile bounds for trimming.

    Returns
    -------
    pd.DataFrame
        Audit summary DataFrame containing metrics and mismatch alerts per test and unit pair.
    """
    audit_records = []

    for test_name, group in df.groupby("laboratory_test_name"):
        valid_group = group.dropna(subset=["raw_value", "raw_unit"])
        unit_counts = valid_group["raw_unit"].value_counts()

        if len(unit_counts) < 2:
            continue  # Need at least 2 distinct units to compare

        ref_unit = unit_counts.index[0]
        ref_series_lin = valid_group.loc[
            valid_group["raw_unit"] == ref_unit, "raw_value"
        ]

        if len(ref_series_lin) < min_sample_size:
            continue

        for comp_unit in unit_counts.index[1:]:
            comp_series_lin = valid_group.loc[
                valid_group["raw_unit"] == comp_unit, "raw_value"
            ]

            if len(comp_series_lin) < min_sample_size:
                audit_records.append({
                    "laboratory_test_name": test_name,
                    "reference_unit": ref_unit,
                    "ref_count": len(ref_series_lin),
                    "comparison_unit": comp_unit,
                    "comp_count": len(comp_series_lin),
                    "raw_emd_linear": np.nan,
                    "pooled_spread_linear": np.nan,
                    "self_noise_baseline": np.nan,
                    "relative_emd_linear": np.nan,
                    "jsd_linear": np.nan,
                    "is_close_linear": False,
                    "alert_linear": True,
                    "raw_emd_log": np.nan,
                    "pooled_spread_log": np.nan,
                    "self_noise_log_baseline": np.nan,
                    "relative_emd_log": np.nan,
                    "jsd_log": np.nan,
                    "is_close_log": False,
                    "alert_log": True,
                    "final_alert": True,
                })
                continue

            distribution_closeness = evaluate_distribution_closeness(
                ref_series_lin,
                comp_series_lin,
                q_threshold_ratio=emd_q_threshold,
                s_threshold_ratio=emd_s_threshold,
                jsd_threshold=jsd_threshold,
                use_rule=use_rule,
                trim_outliers=trim_outliers,
                quantile_trim_bounds=quantile_trim_bounds,
            )

            audit_records.append({
                "laboratory_test_name": test_name,
                "reference_unit": ref_unit,
                "ref_count": len(ref_series_lin),
                "comparison_unit": comp_unit,
                "comp_count": len(comp_series_lin),
                "raw_emd_linear": distribution_closeness["raw_emd_linear"],
                "pooled_spread_linear": distribution_closeness["pooled_spread_linear"],
                "self_noise_baseline": distribution_closeness["self_noise_baseline"],
                "relative_emd_linear": distribution_closeness["relative_emd_linear"],
                "jsd_linear": distribution_closeness["jsd_linear"],
                "is_close_linear": distribution_closeness["is_close_linear"],
                "alert_linear": not distribution_closeness["is_close_linear"],
                "raw_emd_log": distribution_closeness["raw_emd_log"],
                "pooled_spread_log": distribution_closeness["pooled_spread_log"],
                "self_noise_log_baseline": distribution_closeness["self_noise_log_baseline"],
                "relative_emd_log": distribution_closeness["relative_emd_log"],
                "jsd_log": distribution_closeness["jsd_log"],
                "is_close_log": distribution_closeness["is_close_log"],
                "alert_log": not distribution_closeness["is_close_log"],
                "final_alert": distribution_closeness["alert_mismatch"],
            })

    return pd.DataFrame(audit_records)


# =============================================================================
# 4. COLUMN CLUSTERING, VALIDATION & MERGING
# =============================================================================

def compute_cluster_emd_matrix(
    data_df: pd.DataFrame, 
    column_list: List[str], 
    normalize: bool = False
) -> pd.DataFrame:
    """Computes a pairwise Earth Mover's Distance (Wasserstein Distance) matrix.

    Parameters
    ----------
    data_df : pd.DataFrame
        Dataset containing candidate numerical features.
    column_list : List[str]
        List of target column names to compute distance matrix across.
    normalize : bool, default=False
        If True, applies Z-score standardization prior to calculating pairwise distance.

    Returns
    -------
    pd.DataFrame
        Symmetric pairwise EMD distance matrix.
    """
    valid_cols = [c for c in column_list if c in data_df.columns]

    if len(valid_cols) < 2:
        print(f"Warning: Need at least 2 matching columns in DataFrame. Found: {len(valid_cols)}")
        return pd.DataFrame()

    subset = data_df[valid_cols].copy()

    if normalize:
        scaler = StandardScaler()
        scaled_array = scaler.fit_transform(subset)
        subset = pd.DataFrame(scaled_array, columns=valid_cols, index=subset.index)

    n = len(valid_cols)
    emd_matrix = pd.DataFrame(np.zeros((n, n)), index=valid_cols, columns=valid_cols)

    for i in range(n):
        for j in range(i, n):
            col1 = valid_cols[i]
            col2 = valid_cols[j]

            v1 = subset[col1].dropna().values
            v2 = subset[col2].dropna().values

            if len(v1) == 0 or len(v2) == 0:
                emd_val = np.nan
            else:
                emd_val = wasserstein_distance(v1, v2)

            emd_matrix.loc[col1, col2] = emd_val
            emd_matrix.loc[col2, col1] = emd_val

    return emd_matrix


def validate_and_split_cluster(
    df: pd.DataFrame,
    cluster_cols: List[str],
    p_overlap_thresh: float = 0.01,
    q_lower: float = 0.10,
    q_upper: float = 0.90,
    emd_q_thresh_ratio: float = 0.15,
    emd_s_thresh_ratio: float = 1.5,
    jsd_threshold: float = 0.20,
    num_bins: int = 50,
    use_rule: str = "one_of",
    trim_outliers: bool = True,
    quantile_trim_bounds: Tuple[float, float] = (0.05, 0.95),
) -> Dict[str, Union[bool, List[List[str]], pd.DataFrame]]:
    """Validates cluster coherence using graph connectivity and splits disjoint columns.

    An edge is added between two columns in an adjacency graph if they satisfy both
    sample overlap constraints and distribution closeness. Connected components form
    the resulting valid sub-clusters.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset containing feature variables.
    cluster_cols : List[str]
        Column names in the candidate cluster group.
    p_overlap_thresh : float, default=0.01
        Minimum sample co-occurrence fraction relative to the shorter column (0.01 = 1%).
    q_lower, q_upper : float, default=(0.10, 0.90)
        Quantile bounds for distribution spread.
    emd_q_thresh_ratio : float, default=0.15
        Maximum acceptable ratio of EMD / pooled spread.
    emd_s_thresh_ratio : float, default=1.5
        Maximum acceptable ratio of EMD / self-noise baseline.
    jsd_threshold : float, default=0.20
        Maximum JSD score threshold.
    num_bins : int, default=50
        Histogram bin resolution.
    use_rule : str, default='one_of'
        Decision rule for distribution closeness.
    trim_outliers : bool, default=True
        Whether to trim outliers before distance calculation.
    quantile_trim_bounds : Tuple[float, float], default=(0.05, 0.95)
        Quantile bounds used for trimming.

    Returns
    -------
    dict
        - 'is_true_cluster' (bool): True if all columns form a fully connected graph.
        - 'sub_clusters' (List[List[str]]): List of column groups after splitting.
        - 'pairwise_report' (pd.DataFrame): Pairwise metrics and compatibility summary.
    """
    valid_cols = [c for c in cluster_cols if c in df.columns]
    n_cols = len(valid_cols)

    if n_cols <= 1:
        return {
            "is_true_cluster": True,
            "sub_clusters": [valid_cols],
            "pairwise_report": pd.DataFrame(),
        }

    G = nx.Graph()
    G.add_nodes_from(valid_cols)
    report_rows = []

    for i in range(n_cols):
        for j in range(i + 1, n_cols):
            col1, col2 = valid_cols[i], valid_cols[j]
            s1, s2 = df[col1], df[col2]

            n1 = s1.notna().sum()
            n2 = s2.notna().sum()
            shorter_count = min(n1, n2)

            overlap_count = (s1.notna() & s2.notna()).sum()
            overlap_pct = (overlap_count / shorter_count) if shorter_count > 0 else 0.0

            passes_overlap = overlap_pct >= p_overlap_thresh

            dist_res = evaluate_distribution_closeness(
                s1, s2,
                q_lower=q_lower,
                q_upper=q_upper,
                q_threshold_ratio=emd_q_thresh_ratio,
                s_threshold_ratio=emd_s_thresh_ratio,
                jsd_threshold=jsd_threshold,
                num_bins=num_bins,
                use_rule=use_rule,
                trim_outliers=trim_outliers,
                quantile_trim_bounds=quantile_trim_bounds,
            )

            is_close = dist_res["is_close_dual_scale"]
            are_compatible = passes_overlap and is_close

            if are_compatible:
                G.add_edge(col1, col2)

            report_rows.append({
                "column_1": col1,
                "column_2": col2,
                "n_col1": n1,
                "n_col2": n2,
                "overlap_count": overlap_count,
                "overlap_pct": overlap_pct,
                "passes_overlap": passes_overlap,
                "raw_emd_linear": dist_res["raw_emd_linear"],
                "pooled_spread_linear": dist_res["pooled_spread_linear"],
                "self_noise_baseline": dist_res["self_noise_baseline"],
                "relative_emd_linear": dist_res["relative_emd_linear"],
                "jsd_linear": dist_res["jsd_linear"],
                "is_close_linear": dist_res["is_close_linear"],
                "alert_linear": not dist_res["is_close_linear"],
                "raw_emd_log": dist_res["raw_emd_log"],
                "pooled_spread_log": dist_res["pooled_spread_log"],
                "self_noise_log_baseline": dist_res["self_noise_log_baseline"],
                "relative_emd_log": dist_res["relative_emd_log"],
                "jsd_log": dist_res["jsd_log"],
                "is_close_log": dist_res["is_close_log"],
                "alert_log": not dist_res["is_close_log"],
                "distributions_are_close": is_close,
                "compatible": are_compatible,
            })

    pairwise_df = pd.DataFrame(report_rows)

    expected_edges = (n_cols * (n_cols - 1)) // 2
    actual_edges = G.number_of_edges()
    is_true_cluster = (actual_edges == expected_edges)

    sub_clusters = [list(component) for component in nx.connected_components(G)]

    return {
        "is_true_cluster": is_true_cluster,
        "sub_clusters": sub_clusters,
        "pairwise_report": pairwise_df,
    }


def merge_clustered_columns(
    df_wide: pd.DataFrame, 
    final_clusters_df: pd.DataFrame,
    unclustered_cols: List[str],
    naming_strategy: str = "cluster_id"
) -> pd.DataFrame:
    """Merges clustered columns in df_wide by coalescing values based on non-null coverage.

    Prioritizes columns with fewer missing values (higher non-null count) during backfill.

    Parameters
    ----------
    df_wide : pd.DataFrame
        Original wide-format DataFrame containing patient lab measurements.
    final_clusters_df : pd.DataFrame
        Mapping DataFrame containing 'column_name' and 'new_cluster_id'.
    unclustered_cols : List[str]
        Metadata or unclustered column names to retain.
    naming_strategy : str, default='cluster_id'
        - 'top_column': Name merged feature after constituent column with highest coverage.
        - 'cluster_id': Name merged feature by joining constituent column names with ' | '.

    Returns
    -------
    pd.DataFrame
        Merged DataFrame with coalesced features and preserved metadata columns.
    """
    merged_dict = {}

    # 1. Preserve unclustered metadata columns
    for col in unclustered_cols:
        if col in df_wide.columns:
            merged_dict[col] = df_wide[col]

    # 2. Process each cluster group
    for cluster_id, group in final_clusters_df.groupby("new_cluster_id"):
        cols = [c for c in group["column_name"].tolist() if c in df_wide.columns]

        if not cols:
            continue

        if len(cols) == 1:
            merged_dict[cols[0]] = df_wide[cols[0]]
        else:
            # Sort columns descending by non-null count (fewest NaNs first)
            non_null_counts = df_wide[cols].notna().sum()
            ordered_cols = non_null_counts.sort_values(ascending=False).index.tolist()

            # Coalesce values using prioritized order
            coalesced_series = df_wide[ordered_cols].bfill(axis=1).iloc[:, 0]

            if naming_strategy == "top_column":
                merged_col_name = ordered_cols[0]
            else:
                merged_col_name = " | ".join(ordered_cols)

            merged_dict[merged_col_name] = coalesced_series

    return pd.DataFrame(merged_dict, index=df_wide.index)


# =============================================================================
# 5. VISUALIZATION & DIAGNOSTIC PLOTTING
# =============================================================================

def plot_top_correlations(
    target_col: str,
    stats_df: pd.DataFrame,
    data_df: pd.DataFrame,
    top_n: int = 5,
    max_len: int = 2000,
    fig_size: Union[float, Tuple[float, float]] = 10.0,
    plot_type: str = "kde",
    random_state: Optional[int] = None,
    save_dir: Optional[str] = None,
    experiment_name: str = "current_experiment",
    file_format: str = "png",
    dpi: int = 300,
) -> None:
    """Plots joint distribution relationships for top N correlated features against a target column.

    Parameters
    ----------
    target_col : str
        Primary variable name (e.g., target lab protein).
    stats_df : pd.DataFrame
        DataFrame indexed by feature names containing metric summary statistics.
    data_df : pd.DataFrame
        Main measurement DataFrame containing features and target_col.
    top_n : int, default=5
        Number of top features from stats_df to plot.
    max_len : int, default=2000
        Maximum sample size per plot. Downsamples if dataset exceeds this size.
    fig_size : Union[float, Tuple[float, float]], default=10.0
        Plot height or (width, height) tuple for Seaborn JointGrid.
    plot_type : str, default='kde'
        Central joint plot type ('kde' or 'scatter').
    random_state : Optional[int], default=None
        Random seed for downsampling.
    save_dir : Optional[str], default=None
        Directory path where plot image files should be saved.
    experiment_name : str, default='current_experiment'
        Subdirectory name for experiment tracking.
    file_format : str, default='png'
        Image format for exported files ('png', 'pdf', 'svg').
    dpi : int, default=300
        Resolution in dots per inch for exported files.
    """
    grid_height = (
        fig_size[1] if isinstance(fig_size, (tuple, list)) else fig_size
    )
    n_plots = min(top_n, len(stats_df))

    if save_dir:
        output_dir = os.path.join(save_dir, target_col, experiment_name)
        os.makedirs(output_dir, exist_ok=True)

    for i in range(n_plots):
        other_col = stats_df.index[i]

        if other_col not in data_df.columns:
            print(f"Skipping '{other_col}': Column not found in data_df.")
            continue

        pair_df = data_df[[target_col, other_col]].dropna()

        if len(pair_df) < 30:
            print(f"Skipping '{other_col}': Fewer than 30 valid data points.")
            continue

        if (
            pair_df[target_col].nunique() <= 1
            or pair_df[other_col].nunique() <= 1
        ):
            print(f"Skipping '{other_col}': Variable contains constant values (zero variance).")
            continue

        if max_len is not None and len(pair_df) > max_len:
            pair_df = pair_df.sample(n=max_len, random_state=random_state)

        row1_title = f"{target_col} vs. {other_col}"
        stats_series = stats_df.loc[other_col, :]
        row2_title = ", ".join(
            [
                (
                    f"{col}: {val:.3f}"
                    if isinstance(val, (float, int))
                    else f"{col}: {val}"
                )
                for col, val in stats_series.items()
            ]
        )
        full_title = f"{row1_title}\n{row2_title}"

        try:
            g = sns.JointGrid(
                data=pair_df, x=target_col, y=other_col, height=grid_height
            )

            if plot_type.lower() in ["kde", "density"]:
                g.plot_joint(sns.kdeplot, fill=True)
            elif plot_type.lower() in ["scatter", "scatterplot"]:
                g.plot_joint(sns.scatterplot, alpha=0.3, s=30)
            else:
                raise ValueError(
                    f"Invalid plot_type '{plot_type}'. Expected 'kde' or 'scatter'."
                )

            g.plot_marginals(sns.histplot, kde=True, bins=20)
            g.fig.suptitle(full_title, y=1.03, fontsize=11)
            g.fig.subplots_adjust(top=0.88)

            if save_dir:
                clean_target = (
                    str(target_col)
                    .replace("/", "_")
                    .replace("\\", "_")
                    .replace(" ", "_")
                )
                clean_other = (
                    str(other_col)
                    .replace("/", "_")
                    .replace("\\", "_")
                    .replace(" ", "_")
                )
                filename = f"{clean_target}_vs_{clean_other}.{file_format}"
                file_path = os.path.join(output_dir, filename)

                g.savefig(file_path, bbox_inches="tight", dpi=dpi)

            plt.show()

        except Exception as e:
            plt.close()
            print(f"Could not plot '{other_col}': {e}")
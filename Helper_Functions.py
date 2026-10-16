    
import os

import numpy as np
import pandas as pd
from scipy.stats import entropy, wasserstein_distance
from scipy.spatial.distance import jensenshannon
from sklearn.preprocessing import StandardScaler

import seaborn as sns

import networkx as nx

import matplotlib.pyplot as plt

def trim_by_quantile(s: pd.Series, q_lower: float = 0.05, q_upper: float = 0.95) -> pd.Series:
    """
    Trims a pandas Series to include only values within specified quantile bounds.
    """
    clean_s = s.dropna()
    if clean_s.empty:
        return clean_s
    
    q_min = clean_s.quantile(q_lower)
    q_max = clean_s.quantile(q_upper)
    
    return clean_s[(clean_s >= q_min) & (clean_s <= q_max)]

def compute_kl_divergence(
    v1: np.ndarray, v2: np.ndarray, num_bins: int = 50, eps: float = 1e-10
) -> float:
    """Computes symmetric KL Divergence using histogram binning."""
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
    v1: np.ndarray, v2: np.ndarray, num_bins: int = 50
) -> float:
    """Computes Jensen-Shannon Divergence bounded strictly in [0, 1].

    0.0 = Identical distributions 1.0 = Completely disjoint distributions
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
    js_divergence = float(js_distance**2)

    return js_divergence

def evaluate_distribution_closeness(
    s1: pd.Series,
    s2: pd.Series,
    q_lower: float = 0.10,
    q_upper: float = 0.90,
    q_threshold_ratio: float = 0.25,
    s_threshold_ratio: float = 2.0,
    jsd_threshold: float = 0.25,
    num_bins: int = 50,
    use_rule: str = "one_of",  # Default: 'either' (Options: 'one_of', 'all_of', or 'emd_and_jsd')
    trim_outliers: bool = True,
    quantile_trim_bounds: tuple[float, float] = (0.05, 0.95),
    ) -> dict:
    """Evaluates whether two lab distributions are close using dual-scale

    (Linear + Log) relative Earth Mover's Distance (EMD) and Normalized
    Jensen-Shannon Divergence (JSD).

    Parameters
    ----------
    s1, s2 : pd.Series
        The two pandas Series containing numerical lab values.
    q_lower : float, default=0.10
        Lower percentile bound for inter-percentile spread.
    q_upper : float, default=0.90
        Upper percentile bound for inter-percentile spread.
    q_threshold_ratio : float, default=0.25
        Maximum acceptable ratio of EMD / pooled_spread.
    s_threshold_ratio : float, default=2.0
        Maximum acceptable ratio of EMD / self_emd_baseline.
    jsd_threshold : float, default=0.25
        Maximum acceptable JSD score (bounded [0, 1]).
    num_bins : int, default=50
        Histogram resolution passed to compute_normalized_jsd.
    use_rule : str, default='one_of'
        Which rule(s) to evaluate closeness with:
          - 'one_of': Pass one of the EMD test or the JSD test.
          - 'all_of': Pass both the EMD test and the JSD test.
          - 'emd_and_jsd': Pass one of the EMD test and the JSD test.
    trim_outliers : bool, default=True
        Whether to trim outliers from the input series.
    quantile_trim_bounds : tuple[float, float], default=(0.05, 0.95)
        The lower and upper quantile bounds for trimming outliers.
    """
# 1. Clean missing values and optionally trim to quantile band
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

    # Spread computed on the trimmed or full data
    spread1_lin = s1_clean.quantile(q_upper) - s1_clean.quantile(q_lower)
    spread2_lin = s2_clean.quantile(q_upper) - s2_clean.quantile(q_lower)
    pooled_spread_lin = (spread1_lin + spread2_lin) / 2.0
    
    rel_emd_lin = (
        emd_lin / pooled_spread_lin if pooled_spread_lin > 0 else np.nan
    )

    jsd_lin = compute_normalized_jsd(v1_lin, v2_lin, num_bins=num_bins)

    # Baseline Noise Floor Calculation
    rng = np.random.default_rng(seed=42)
    half_size = len(v1_lin) // 2
    shuffled_v1 = rng.permutation(v1_lin)
    self_emd_baseline = (
        wasserstein_distance(shuffled_v1[:half_size], shuffled_v1[half_size:])
        if half_size > 0
        else 0.0
    )

    if use_rule == "one_of":
        is_close_lin = (rel_emd_lin < q_threshold_ratio) or (emd_lin < s_threshold_ratio * self_emd_baseline) or (jsd_lin < jsd_threshold)
    elif use_rule == "all_of":
        is_close_lin = (rel_emd_lin < q_threshold_ratio) and (emd_lin < s_threshold_ratio * self_emd_baseline) and (jsd_lin < jsd_threshold)
    elif use_rule == "emd_and_jsd":
        is_close_lin = ((rel_emd_lin < q_threshold_ratio) or (emd_lin < s_threshold_ratio * self_emd_baseline)) and (jsd_lin < jsd_threshold)
    else:
        raise ValueError(f"Invalid use_rule '{use_rule}'.")

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
        
        rel_emd_log = (
            emd_log / pooled_spread_log if pooled_spread_log > 0 else np.nan
        )

        jsd_log = compute_normalized_jsd(v1_log, v2_log, num_bins=num_bins)

        half_size_log = len(v1_log) // 2
        shuffled_v1_log = rng.permutation(v1_log)
        self_emd_log_baseline = (
            wasserstein_distance(shuffled_v1_log[:half_size_log], shuffled_v1_log[half_size_log:])
            if half_size_log > 0
            else 0.0
        )

        if use_rule == "one_of":
            is_close_log = (rel_emd_log < q_threshold_ratio) or (emd_log < s_threshold_ratio * self_emd_log_baseline) or (jsd_log < jsd_threshold)
        elif use_rule == "all_of":
            is_close_log = (rel_emd_log < q_threshold_ratio) and (emd_log < s_threshold_ratio * self_emd_log_baseline) and (jsd_log < jsd_threshold)
        elif use_rule == "emd_and_jsd":
            is_close_log = ((rel_emd_log < q_threshold_ratio) or (emd_log < s_threshold_ratio * self_emd_log_baseline)) and (jsd_log < jsd_threshold)
        else:
            raise ValueError(f"Invalid use_rule '{use_rule}'.")
    else:
        emd_log, pooled_spread_log, self_emd_log_baseline, rel_emd_log, jsd_log = np.nan, np.nan, np.nan, np.nan, np.nan
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

# -------------------------------------------------------------------------
# Dual-Scale Audit Loop (Linear AND Log Gate)
# -------------------------------------------------------------------------
def audit_lab_unit_distributions_dual_scale(
    df: pd.DataFrame,
    min_sample_size: int = 15,
    emd_q_threshold: float = 0.25,
    emd_s_threshold: float = 2,
    jsd_threshold: float = 0.25,
    use_rule: str = "one_of",  # Default: 'either' (Options: 'one_of', 'all_of', or 'emd_and_jsd')
    trim_outliers: bool = True,
    quantile_trim_bounds: tuple[float, float] = (0.05, 0.95),
) -> pd.DataFrame:
    """Audits unit distributions across both Linear and Log scales.

    An alert is flagged ONLY IF both scales fail the similarity threshold.
    use_rule : str, default='one_of'
        Which rule(s) to evaluate closeness with:
          - 'one_of': Pass one of the EMD test or the JSD test.
          - 'all_of': Pass both the EMD test and the JSD test.
          - 'emd_and_jsd': Pass one of the EMD test and the JSD test.
    """
    audit_records = []

    for test_name, group in df.groupby("laboratory_test_name"):
        valid_group = group.dropna(subset=["raw_value", "raw_unit"])
        unit_counts = valid_group["raw_unit"].value_counts()

        if len(unit_counts) < 2:
            continue  # Need at least 2 units to compare

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
                audit_records.append(
                {
                    "laboratory_test_name": test_name,
                    "reference_unit": ref_unit,
                    "ref_count": len(ref_series_lin),
                    "comparison_unit": comp_unit,
                    "comp_count": len(comp_series_lin),
                    # Linear Outputs
                    "raw_emd_linear": np.nan,
                    "pooled_spread_linear": np.nan,
                    "self_noise_baseline": np.nan,
                    "relative_emd_linear": np.nan,
                    "jsd_linear": np.nan,
                    "is_close_linear": False,
                    "alert_linear": True,
                    # Log Outputs
                    "raw_emd_log": np.nan,
                    "pooled_spread_log": np.nan,
                    "self_noise_log_baseline": np.nan,
                    "relative_emd_log": np.nan,
                    "jsd_log": np.nan,
                    "is_close_log": False,
                    "alert_log": True,
                    # Summary Flags
                    "final_alert": True,
                }
                )
                continue

            distribution_closeness = evaluate_distribution_closeness(
                ref_series_lin, comp_series_lin, 
                q_threshold_ratio=emd_q_threshold, 
                s_threshold_ratio=emd_s_threshold, 
                jsd_threshold=jsd_threshold, 
                use_rule=use_rule,
                trim_outliers=trim_outliers,
                quantile_trim_bounds=quantile_trim_bounds,
            )
            
            # -------------------------------------------------------------
            # 3. Decision Rule: Dual-Failure Requirement
            # -------------------------------------------------------------
            final_alert = distribution_closeness['alert_mismatch']
            # final_alert = alert_lin and alert_log


            audit_records.append(
                {
                    "laboratory_test_name": test_name,
                    "reference_unit": ref_unit,
                    "ref_count": len(ref_series_lin),
                    "comparison_unit": comp_unit,
                    "comp_count": len(comp_series_lin),
                    # Linear Outputs
                    "raw_emd_linear": distribution_closeness['raw_emd_linear'],
                    "pooled_spread_linear": distribution_closeness['pooled_spread_linear'],
                    "self_noise_baseline": distribution_closeness['self_noise_baseline'],
                    "relative_emd_linear": distribution_closeness['pooled_spread_linear'],
                    "jsd_linear": distribution_closeness['jsd_linear'],
                    "is_close_linear": distribution_closeness['is_close_linear'],
                    "alert_linear": not distribution_closeness['is_close_linear'],
                    # Log Outputs
                    "raw_emd_log": distribution_closeness['raw_emd_log'],
                    "pooled_spread_log": distribution_closeness['pooled_spread_log'],
                    "self_noise_log_baseline": distribution_closeness['self_noise_log_baseline'],
                    "relative_emd_log": distribution_closeness['relative_emd_log'],
                    "jsd_log": distribution_closeness['jsd_log'],
                    "is_close_log": distribution_closeness['is_close_log'],
                    "alert_log": not distribution_closeness['is_close_log'],
                    # Summary Flags
                    "final_alert": final_alert,
                }
            )

    return pd.DataFrame(audit_records)


def compute_cluster_emd_matrix(data_df: pd.DataFrame, column_list: list[str], normalize: bool = False) -> pd.DataFrame:
    """
    Computes a pairwise Earth Mover's Distance (Wasserstein Distance) matrix
    for a list of numerical columns in a dataset.
    """
    # Filter dataset to only include columns present in data_df
    valid_cols = [c for c in column_list if c in data_df.columns]
    
    if len(valid_cols) < 2:
        print(f"Warning: Need at least 2 matching columns in DataFrame. Found: {len(valid_cols)}")
        return pd.DataFrame()

    subset = data_df[valid_cols].copy()

    # Optional Z-score normalization to compare distribution shapes across different units
    if normalize:
        scaler = StandardScaler()
        scaled_array = scaler.fit_transform(subset)
        subset = pd.DataFrame(scaled_array, columns=valid_cols, index=subset.index)

    n = len(valid_cols)
    emd_matrix = pd.DataFrame(np.zeros((n, n)), index=valid_cols, columns=valid_cols)

    # Compute pairwise 1D Wasserstein distances
    for i in range(n):
        for j in range(i, n):
            col1 = valid_cols[i]
            col2 = valid_cols[j]

            # Extract non-null observations for each column independently
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
    cluster_cols: list[str],
    p_overlap_thresh: float = 0.01,  # Default 1% (0.01)
    q_lower: float = 0.10,
    q_upper: float = 0.90,
    emd_q_thresh_ratio: float = 0.15,
    emd_s_thresh_ratio: float = 1.5,
    jsd_threshold: float = 0.20,
    num_bins: int = 50,
    use_rule: str = "one_of",  # Default: 'either' (Options: 'one_of', 'all_of', or 'emd_and_jsd')
    trim_outliers: bool = True,
    quantile_trim_bounds: tuple[float, float] = (0.05, 0.95)
) -> dict:
    """
    Validates a candidate cluster and splits it if required.
    
    Two columns CANNOT be in the same cluster if:
      - Overlap < P% of the shorter column's count.
      - Distributions are NOT close (evaluate_distribution_closeness == False).
    
    Parameters
    ----------
    df : pd.DataFrame
        The full patient labs dataset.
    cluster_cols : list[str]
        List of column names belonging to the candidate cluster.
    p_overlap_thresh : float, default=0.01
        Minimum required overlap fraction relative to shorter column (0.01 = 1%).
    q_lower, q_upper : float
        Percentile bounds for scaling EMD in evaluate_distribution_closeness.
    emd_q_thresh_ratio : float, default=0.15
        Maximum acceptable ratio of EMD / pooled_spread.
    emd_s_thresh_ratio : float, default=1.5
        Maximum acceptable ratio of EMD / self_emd_baseline.
    jsd_threshold : float, default=0.20
        Maximum acceptable JSD score (bounded [0, 1]).
    num_bins : int, default=50
        Histogram resolution passed to compute_normalized_jsd.
    use_rule : str, default='one_of'
        Which rule(s) to evaluate closeness with:
          - 'one_of': Pass one of the EMD test or the JSD test.
          - 'all_of': Pass both the EMD test and the JSD test.
          - 'emd_and_jsd': Pass one of the EMD test and the JSD test.
        
    Returns
    -------
    dict:
        - 'is_true_cluster' (bool): True if all columns can be merged together.
        - 'sub_clusters' (list[list[str]]): List of column groups after splitting.
        - 'pairwise_report' (pd.DataFrame): Detailed metrics for every column pair.
    """
    # 1. Filter columns present in DataFrame
    valid_cols = [c for c in cluster_cols if c in df.columns]
    n_cols = len(valid_cols)
    
    if n_cols <= 1:
        return {
            "is_true_cluster": True,
            "sub_clusters": [valid_cols],
            "pairwise_report": pd.DataFrame()
        }

    # 2. Initialize Graph
    G = nx.Graph()
    G.add_nodes_from(valid_cols)
    report_rows = []

    # 3. Evaluate Pairwise Compatibility
    for i in range(n_cols):
        for j in range(i + 1, n_cols):
            col1, col2 = valid_cols[i], valid_cols[j]
            s1, s2 = df[col1], df[col2]
            
            # Non-null counts
            n1 = s1.notna().sum()
            n2 = s2.notna().sum()
            shorter_count = min(n1, n2)
            
            # Overlap count
            overlap_count = (s1.notna() & s2.notna()).sum()
            overlap_pct = (overlap_count / shorter_count) if shorter_count > 0 else 0.0
            
            # Criterion 1: Overlap Check
            passes_overlap = overlap_pct >= p_overlap_thresh
            
            # Criterion 2: Distribution Closeness Check

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
            # print(dist_res)
                
            
            # if overlap_count >= 0 and passes_overlap:
            #     dist_res = evaluate_distribution_closeness(
            #         s1, s2, 
            #         q_lower=q_lower, 
            #         q_upper=q_upper, 
            #         q_threshold_ratio=emd_q_thresh_ratio,
            #         s_threshold_ratio=emd_s_thresh_ratio,
            #         jsd_threshold=jsd_threshold,
            #         num_bins=num_bins,
            #         use_rule=use_rule,
            #         trim_outliers=trim_outliers,
            #         quantile_trim_bounds=quantile_trim_bounds,
            #     )

            #     is_close = dist_res["is_close_dual_scale"]
                
            # else:
            #     dist_res={
            #         # Linear Outputs
            #         "raw_emd_linear": np.nan,
            #         "pooled_spread_linear": np.nan,
            #         "self_noise_baseline": np.nan,
            #         "relative_emd_linear": np.nan,
            #         "jsd_linear": np.nan,
            #         "is_close_linear": False,
            #         # Log Outputs
            #         "raw_emd_log": np.nan,
            #         "pooled_spread_log": np.nan,
            #         "self_noise_log_baseline": np.nan,
            #         "relative_emd_log": np.nan,
            #         "jsd_log": np.nan,
            #         "is_close_log": False,
            #         # Summary Flags
            #         "is_close_dual_scale": False,
            #         "alert_mismatch": True,
            #     }
            #     is_close = dist_res["is_close_dual_scale"]

            # Both criteria MUST pass for an edge to exist
            are_compatible = passes_overlap and is_close
            
            if are_compatible:
                G.add_edge(col1, col2)

            report_rows.append({
                # overlap output
                "column_1": col1,
                "column_2": col2,
                "n_col1": n1,
                "n_col2": n2,
                "overlap_count": overlap_count,
                "overlap_pct": overlap_pct,
                "passes_overlap": passes_overlap,
                # Linear Outputs
                "raw_emd_linear": dist_res['raw_emd_linear'],
                "pooled_spread_linear": dist_res['pooled_spread_linear'],
                "self_noise_baseline": dist_res['self_noise_baseline'],
                "relative_emd_linear": dist_res['pooled_spread_linear'],
                "jsd_linear": dist_res['jsd_linear'],
                "is_close_linear": dist_res['is_close_linear'],
                "alert_linear": not dist_res['is_close_linear'],
                # Log Outputs
                "raw_emd_log": dist_res['raw_emd_log'],
                "pooled_spread_log": dist_res['pooled_spread_log'],
                "self_noise_log_baseline": dist_res['self_noise_log_baseline'],
                "relative_emd_log": dist_res['relative_emd_log'],
                "jsd_log": dist_res['jsd_log'],
                "is_close_log": dist_res['is_close_log'],
                "alert_log": not dist_res['is_close_log'],
                # Summary Flags
                "distributions_are_close": dist_res["is_close_dual_scale"],
                "compatible": are_compatible
            })

    pairwise_df = pd.DataFrame(report_rows)
    # print(pairwise_df)

    # 4. Check if fully connected (True Cluster)
    expected_edges = (n_cols * (n_cols - 1)) // 2
    actual_edges = G.number_of_edges()
    is_true_cluster = (actual_edges == expected_edges)

    # 5. Extract Sub-Clusters
    sub_clusters = [list(component) for component in nx.connected_components(G)]

    return {
        "is_true_cluster": is_true_cluster,
        "sub_clusters": sub_clusters,
        "pairwise_report": pairwise_df
    }

def merge_clustered_columns(
    df_wide: pd.DataFrame, 
    final_clusters_df: pd.DataFrame,
    unclustered_cols: list,
    naming_strategy: str = "cluster_id"
) -> pd.DataFrame:
    """
    Merges columns in df_wide based on final_clusters_df mappings.
    Priority is given to columns with fewer NaNs (higher non-null counts).
    
    Parameters
    ----------
    df_wide : pd.DataFrame
        The original wide-format dataset containing patient lab measurements.
    final_clusters_df : pd.DataFrame
        Mapping dataframe containing 'column_name' and 'new_cluster_id'.
    naming_strategy : str, default='top_column'
        - 'top_column': Name the merged column after the primary column (fewest NaNs).
        - 'cluster_id': Name the merged column 'cluster_<new_cluster_id>'.
        
    Returns
    -------
    pd.DataFrame
        Merged DataFrame with coalesced lab columns and unclustered metadata preserved.
    """
    merged_dict = {}
    
    # 1. Preserve non-lab metadata columns (e.g., patient_id, age, race)
    clustered_cols_set = set(final_clusters_df["column_name"])
    # unclustered_cols = [col for col in df_wide.columns if col not in clustered_cols_set]
    unclustered_cols = unclustered_cols
    
    for col in unclustered_cols:
        merged_dict[col] = df_wide[col]
        
    # 2. Process each cluster group
    for cluster_id, group in final_clusters_df.groupby("new_cluster_id"):
        # Filter to columns that actually exist in df_wide
        cols = [c for c in group["column_name"].tolist() if c in df_wide.columns]
        
        if not cols:
            continue
            
        if len(cols) == 1:
            # Single-column cluster: no merging required
            merged_dict[cols[0]] = df_wide[cols[0]]
        else:
            # Sort columns by non-null count descending (fewest NaNs first)
            non_null_counts = df_wide[cols].notna().sum()
            ordered_cols = non_null_counts.sort_values(ascending=False).index.tolist()
            
            # Coalesce: bfill across columns ordered by priority, then take the first column
            coalesced_series = df_wide[ordered_cols].bfill(axis=1).iloc[:, 0]
            
            # Select column name for the merged output
            if naming_strategy == "top_column":
                merged_col_name = ordered_cols[0]
            else:
                # merged_col_name = f"cluster_{cluster_id}"

                merged_col_name = ordered_cols[0]
                for m_col in ordered_cols[1:]:
                    merged_col_name = merged_col_name + " | " + m_col
                
            merged_dict[merged_col_name] = coalesced_series

    # 3. Assemble and return final DataFrame
    df_merged = pd.DataFrame(merged_dict, index=df_wide.index)
    return df_merged

def get_columns_perplexity(df: pd.DataFrame, n_bins: int = -1) -> pd.Series:
    """Calculates the perplexity of each column in a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    n_bins : int, default -1
        Number of equal-width bins for continuous numeric columns.
        If -1, keeps original values without binning.

    Returns
    -------
    pd.Series
        Perplexity score per column (1.0 indicates effectively constant).
    """
    results = {}

    for col in df.columns:
        s = df[col].dropna()

        if s.empty:
            results[col] = 0.0
            continue

        # Bin numerical columns into N equal-width intervals if requested
        try:
            if n_bins > 0 and pd.api.types.is_numeric_dtype(s):
                s = pd.cut(s, bins=n_bins)
    
            probs = s.value_counts(normalize=True).to_numpy()
            probs = probs[probs > 0]  # Exclude empty bins to avoid np.log(0)
            shannon_entropy = -np.sum(probs * np.log(probs)) if len(probs) > 0 else 0.0
            results[col] = float(np.exp(shannon_entropy))
        except Exception as e:
            print(f"The system said: {e}")
            results[col] = 0

    return pd.Series(results, dtype=float)



def plot_top_correlations(
    target_col: str,
    stats_df: pd.DataFrame,
    data_df: pd.DataFrame,
    top_n: int = 5,
    max_len: int = 2000,
    fig_size: float = 10.0,
    plot_type: str = "kde",
    random_state: int = None,
    save_dir: str = None,
    experiment_name: str = "current_experiment",
    file_format: str = "png",
    dpi: int = 300,
):
    """Plots KDE JointGrid distributions for the top N features against a target column.

    Parameters:
    -----------
    target_col : str
        The primary column name (e.g., target protein).
    stats_df : pd.DataFrame
        DataFrame indexed by variable names containing summary statistics/metrics.
    data_df : pd.DataFrame
        Main measurement DataFrame containing columns for target_col and stats_df indices.
    top_n : int, default=5
        Number of top features from stats_df to plot.
    max_len : int, default=2000
        Maximum sample size to plot per grid. Downsamples if length exceeds this.
    fig_size : float or tuple, default=10.0
        Height/size of the JointGrid plot. Accepts a float (height) or (width, height) tuple.
    plot_type : str, default="kde"
        Type of central plot. Options are 'kde' (density contour) or 'scatter' (scatterplot).
    random_state : int, default=None
        Seed used for reproducible downsampling.
    save_dir : str, optional
        path to directory where plot image files should be saved. If None, plots are not saved.
    experiment_name : str, default="current_experiment"
        Name of the experiment, used for subfolder creation.
    file_format : str, default="png"
        File format for saved images (e.g., 'png', 'pdf', 'svg').
    dpi : int, default=300
        Resolution in dots per inch for saved images.
    """
    grid_height = (
        fig_size[1] if isinstance(fig_size, (tuple, list)) else fig_size
    )
    n_plots = min(top_n, len(stats_df))

    # Construct subfolder path: save_dir / target_col / experiment_name
    if save_dir:
        output_dir = os.path.join(save_dir, target_col, experiment_name)
        os.makedirs(output_dir, exist_ok=True)

    for i in range(n_plots):
        other_col = stats_df.index[i]

        if other_col not in data_df.columns:
            print(f"Skipping '{other_col}': Column not found in data_df.")
            continue

        pair_df = data_df[[target_col, other_col]].dropna()

        # Check for empty datasets or zero-variance columns
        if len(pair_df) < 30:
            print(f"Skipping '{other_col}': Fewer than 30 valid data points.")
            continue

        if (
            pair_df[target_col].nunique() <= 1
            or pair_df[other_col].nunique() <= 1
        ):
            print(
                f"Skipping '{other_col}': Variable contains constant values (zero variance)."
            )
            continue

        # Downsample if needed
        if max_len is not None and len(pair_df) > max_len:
            pair_df = pair_df.sample(n=max_len, random_state=random_state)

        # 1. Format two-row title
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

        # 2. Render plot safely
        try:
            g = sns.JointGrid(
                data=pair_df, x=target_col, y=other_col, height=grid_height
            )

            # Toggle between KDE plot and Scatter plot
            if plot_type.lower() in ["kde", "density"]:
                g.plot_joint(sns.kdeplot, fill=True)
            elif plot_type.lower() in ["scatter", "scatterplot"]:
                g.plot_joint(sns.scatterplot, alpha=0.3, s=30)
            else:
                raise ValueError(
                    f"Invalid plot_type '{plot_type}'. Expected 'kde' or 'scatter'."
                )

            # Fixed bins parameter prevents NumPy auto-binning failures
            g.plot_marginals(sns.histplot, kde=True, bins=20)

            g.fig.suptitle(full_title, y=1.03, fontsize=11)
            g.fig.subplots_adjust(top=0.88)

            # 3. Save plot if output directory is provided
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
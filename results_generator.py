import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

try:
    from scipy.stats import wilcoxon
except Exception:
    wilcoxon = None

# =============================================================================
# Constants
# =============================================================================
BLOCK_ORDER = ["Va", "Vm", "Pg", "Qg"]

FAMILY_ORDER = [
    "baseline",
    "oracle_ac_primal_only",
    "oracle_ac_primal_dual",
    "oracle_ac_constraint_dual",
    "oracle_ac_bounds_dual",
    "oracle_ac_primal_dual_all_bounds",
    "oracle_ac_dual_only_constraint",
    "oracle_ac_dual_only_bounds",
    "oracle_ac_dual_only_full",
    "dc_seed",
]

FAMILY_LABEL = {
    "baseline": "Baseline",
    "oracle_ac_primal_only": "Oracle AC primal-only",
    "oracle_ac_primal_dual": "Oracle AC primal+dual",
    "oracle_ac_constraint_dual": "Oracle AC constraint-dual-only",
    "oracle_ac_bounds_dual": "Oracle AC bounds-dual-only",
    "oracle_ac_primal_dual_all_bounds": "Oracle AC primal+dual (all bounds)",
    "oracle_ac_dual_only_constraint": "Oracle dual-only (constraint)",
    "oracle_ac_dual_only_bounds": "Oracle dual-only (bounds)",
    "oracle_ac_dual_only_full": "Oracle dual-only (full)",
    "dc_seed": "DC-seeded",
}

# Oracle families that iterate over the 15 primal-block subsets
ORACLE_COMBO_FAMILIES = [
    "oracle_ac_primal_only",
    "oracle_ac_primal_dual",
    "oracle_ac_constraint_dual",
    "oracle_ac_bounds_dual",
    "oracle_ac_primal_dual_all_bounds",
]

# Dual-only families (no primal blocks)
DUAL_ONLY_FAMILIES = [
    "oracle_ac_dual_only_constraint",
    "oracle_ac_dual_only_bounds",
    "oracle_ac_dual_only_full",
]

# Families that use IPOPT warm-start settings
WARM_START_FAMILIES = {
    "oracle_ac_primal_dual",
    "oracle_ac_constraint_dual",
    "oracle_ac_bounds_dual",
    "oracle_ac_primal_dual_all_bounds",
    "oracle_ac_dual_only_constraint",
    "oracle_ac_dual_only_bounds",
    "oracle_ac_dual_only_full",
}

# Short labels for heatmap panels
DUAL_MODE_PANEL_LABELS = {
    "oracle_ac_primal_only": "Primal only\n(no duals)",
    "oracle_ac_primal_dual": "Primal+dual\n(block-matched)",
    "oracle_ac_constraint_dual": "Constraint\nduals only",
    "oracle_ac_bounds_dual": "Bound mults\nonly",
    "oracle_ac_primal_dual_all_bounds": "Primal+dual\n(all bounds)",
}

TABLE_PREFIXES = {
    "global_summary": "table_01_global_summary",
    "pairwise_tests": "table_02_pairwise_tests",
    "combo_rank_oracle": "table_03_combo_ranking_oracle",
    "combo_rank_dc": "table_05_combo_ranking_dc_seed",
    "block_effects": "table_06_block_effects",
    "largest_cases": "table_08_casewise_largest_cases",
    "combo_wins": "table_10_combo_wins",
    "family_wins": "table_12_family_wins",
    "all_warmstarts_matrix": "table_13_all_warmstarts_matrix",
    "dual_decomposition_summary": "table_14_dual_decomposition_summary",
}

FIG_PREFIXES = {
    "case_size_runtime": "fig_01_case_size_vs_runtime",
    "performance_profile": "fig_02_performance_profile",
    "casewise_speedups": "fig_03_casewise_speedups",
    "oracle_combo_heatmap_speedup": "fig_04_oracle_combo_heatmap_speedup",
    "block_effects": "fig_06_block_effects",
    "speedup_vs_case_size": "fig_07_speedup_vs_case_size",
    "speedup_distributions": "fig_08_speedup_distributions",
    "baseline_vs_best": "fig_09_baseline_vs_best_scatter",
    "dc_native_boxplot": "fig_12_dc_native_boxplot",
    "dual_decomposition_heatmap": "fig_13_dual_decomposition_heatmap",
    "dual_only_bar": "fig_14_dual_only_bar",
}

CASE_HEADER_MAP = {
    "pglib_opf_case5_pjm": "5",
    "pglib_opf_case14_ieee": "14",
    "pglib_opf_case24_ieee_rts": "24",
    "pglib_opf_case30_as": "30a",
    "pglib_opf_case30_ieee": "30b",
    "pglib_opf_case39_epri": "39",
    "pglib_opf_case57_ieee": "57",
    "pglib_opf_case60_c": "60",
    "pglib_opf_case73_ieee_rts": "73",
    "pglib_opf_case89_pegase": "89",
    "pglib_opf_case118_ieee": "118",
    "pglib_opf_case200_activ": "200",
    "pglib_opf_case240_pserc": "240",
    "pglib_opf_case300_ieee": "300",
    "pglib_opf_case500_ieee": "500",
    "pglib_opf_case1354_pegase": "1K",
    "pglib_opf_case2000_goc": "2K",
    "pglib_opf_case3022_goc": "3,022",
    "pglib_opf_case4020_goc": "4K",
    "pglib_opf_case5658_epigrids": "5,658",
    "pglib_opf_case6468_rte": "6K",
    "pglib_opf_case7336_epigrids": "7,336",
    "pglib_opf_case8387_pegase": "8,387",
    "pglib_opf_case9241_pegase": "9,241",
    "pglib_opf_case10000_goc": "10,000",
    "pglib_opf_case13659_pegase": "14K",
    "pglib_opf_case30000_goc": "30,000",
}


# =============================================================================
# Small helpers
# =============================================================================
def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def latex_escape(val) -> str:
    if val is None:
        return "--"
    s = str(val)
    repl = {
        "\\": r"\textbackslash{}",
        "_": r"\_",
        "%": r"\%",
        "&": r"\&",
        "#": r"\#",
        "$": r"\$",
        "{": r"\{",
        "}": r"\}",
    }
    for a, b in repl.items():
        s = s.replace(a, b)
    return s


def fmt_float(x, digits: int = 3, nan: str = "---") -> str:
    try:
        if x is None or pd.isna(x) or not np.isfinite(float(x)):
            return nan
        return f"{float(x):.{digits}f}"
    except Exception:
        return nan


def fmt_pct_number(x, digits: int = 1, nan: str = "---") -> str:
    try:
        if x is None or pd.isna(x) or not np.isfinite(float(x)):
            return nan
        return f"{float(x):.{digits}f}"
    except Exception:
        return nan


def fmt_int(x, nan: str = "---") -> str:
    try:
        if x is None or pd.isna(x):
            return nan
        return f"{int(round(float(x)))}"
    except Exception:
        return nan


def pval_str(x: float) -> str:
    if x is None or pd.isna(x) or not np.isfinite(x):
        return "---"
    if x < 1e-4:
        return r"$<$0.0001"
    return f"{x:.4f}"


def bool_mark(x) -> str:
    if x is None or pd.isna(x):
        return "---"
    return r"\checkmark" if bool(x) else r"$\times$"


def speedup_pct(baseline, new):
    if baseline is None or new is None:
        return np.nan
    try:
        if pd.isna(baseline) or pd.isna(new) or float(baseline) <= 0:
            return np.nan
    except Exception:
        return np.nan
    return 100.0 * (float(baseline) - float(new)) / float(baseline)


def canonicalize_blocks(blocks: str) -> str:
    """Normalize a raw combination string into one canonical order."""
    if not isinstance(blocks, str) or not blocks.strip():
        return ""

    parts = [p.strip() for p in blocks.split("+") if p and p.strip()]
    if not parts:
        return ""

    seen = set()
    parts = [p for p in parts if not (p in seen or seen.add(p))]

    ordered = [p for p in BLOCK_ORDER if p in parts]
    extras = sorted([p for p in parts if p not in BLOCK_ORDER])
    return "+".join(ordered + extras)


def combo_sort_key(blocks: str) -> Tuple[int, Tuple[int, ...], str]:
    """Stable sort key for combinations using BLOCK_ORDER."""
    canon = canonicalize_blocks(blocks)
    if not canon:
        return (0, (), "")

    parts = canon.split("+")
    idx = {b: i for i, b in enumerate(BLOCK_ORDER)}
    ordered_idx = tuple(idx[p] for p in parts if p in idx)
    return (len(parts), ordered_idx, canon)


def blocks_to_math(blocks: str) -> str:
    mapping = {"Pg": r"P_g", "Qg": r"Q_g", "Vm": r"V_m", "Va": r"V_a"}
    canon = canonicalize_blocks(blocks)
    if not canon:
        return r"---"
    parts = canon.split("+")
    return "$" + "{+}".join(mapping.get(p, latex_escape(p)) for p in parts) + "$"


def human_blocks(blocks: str) -> str:
    canon = canonicalize_blocks(blocks)
    if not canon:
        return "Baseline"
    return canon.replace("+", " + ")


def family_short_label(family: str) -> str:
    return {
        "baseline": "Baseline",
        "oracle_ac_primal_only": "O-PO",
        "oracle_ac_primal_dual": "O-PD",
        "oracle_ac_constraint_dual": "O-CD",
        "oracle_ac_bounds_dual": "O-BD",
        "oracle_ac_primal_dual_all_bounds": "O-PD-AB",
        "oracle_ac_dual_only_constraint": "DO-C",
        "oracle_ac_dual_only_bounds": "DO-B",
        "oracle_ac_dual_only_full": "DO-F",
        "dc_seed": "DC",
    }.get(family, family)


def objective_matched_rows(df: pd.DataFrame) -> pd.DataFrame:
    if "objective_matches_baseline" not in df.columns:
        return df.copy()
    matched = df[df["objective_matches_baseline"] == 1].copy()
    return matched if not matched.empty else df.copy()


def safe_series_mean(series: pd.Series):
    vals = pd.to_numeric(series, errors="coerce")
    vals = vals[np.isfinite(vals)]
    return float(vals.mean()) if len(vals) else np.nan


def safe_series_median(series: pd.Series):
    vals = pd.to_numeric(series, errors="coerce")
    vals = vals[np.isfinite(vals)]
    return float(np.median(vals)) if len(vals) else np.nan


def case_short_name(case_name: str, nb: Optional[float] = None) -> str:
    if case_name in CASE_HEADER_MAP:
        return CASE_HEADER_MAP[case_name]
    if nb is not None and pd.notna(nb):
        nb_int = int(round(float(nb)))
        return f"{nb_int:,}" if nb_int >= 1000 else str(nb_int)
    return str(case_name)


def has_extended_dual_data(runs: pd.DataFrame) -> bool:
    """Check whether extended dual decomposition families are present in the data."""
    extended_families = set(ORACLE_COMBO_FAMILIES[2:]) | set(DUAL_ONLY_FAMILIES)
    present = set(runs["family"].unique())
    return bool(present & extended_families)


# =============================================================================
# Loading and normalization
# =============================================================================
def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def load_study(study_dir: Path) -> Dict[str, pd.DataFrame]:
    all_runs_path = study_dir / "all_runs.csv"
    if not all_runs_path.exists():
        raise FileNotFoundError(f"Missing required file: {all_runs_path}")

    data = {
        "all_runs": _read_csv(all_runs_path),
        "all_cases": _read_csv(study_dir / "all_cases.csv"),
        "manifest": {},
    }

    manifest_path = study_dir / "manifest.json"
    if manifest_path.exists():
        try:
            data["manifest"] = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            data["manifest"] = {}

    if data["all_runs"].empty:
        raise ValueError(f"{all_runs_path} is empty")

    return data


def normalize_data(data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    runs = data["all_runs"].copy()

    numeric_cols = [
        "objective", "build_time", "solve_time", "total_time",
        "end_to_end_time", "full_pipeline_time_including_shared_case_prep",
        "iterations", "nb", "nl", "ng_total", "ng_active", "success",
        "objective_matches_baseline", "objective_gap_vs_baseline_abs",
        "objective_gap_vs_baseline_rel", "hit_max_iter",
    ]
    for c in numeric_cols:
        if c in runs.columns:
            runs[c] = pd.to_numeric(runs[c], errors="coerce")

    text_cols = [
        "family", "blocks", "case_name", "run_name", "seed_vm_policy",
        "seed_qg_policy", "dc_seed_mode", "termination", "status", "dual_mode",
    ]
    for c in text_cols:
        if c not in runs.columns:
            runs[c] = ""
        runs[c] = runs[c].fillna("").astype(str)

    if "success" in runs.columns and runs["success"].notna().any():
        runs["success"] = runs["success"].fillna(0).astype(int).astype(bool)
    else:
        term_ok = runs["termination"].isin(["optimal", "locallyOptimal", "globallyOptimal"])
        runs["success"] = runs["objective"].notna() | term_ok

    if "end_to_end_time" not in runs.columns or runs["end_to_end_time"].isna().all():
        runs["end_to_end_time"] = runs["total_time"]

    baseline_ref = runs[(runs["family"] == "baseline") & (runs["success"])].copy()
    baseline_ref = (
        baseline_ref.sort_values(["case_name", "solve_time", "total_time"])
        .groupby("case_name", as_index=False)
        .head(1)
        [["case_name", "solve_time", "total_time", "iterations", "objective", "nb", "nl"]]
        .rename(columns={
            "solve_time": "baseline_solve_time",
            "total_time": "baseline_total_time",
            "iterations": "baseline_iterations",
            "objective": "baseline_objective",
        })
    )

    runs = runs.merge(baseline_ref, on=["case_name", "nb", "nl"], how="left")
    runs["speedup_vs_baseline_pct"] = 100.0 * (runs["baseline_solve_time"] - runs["solve_time"]) / runs["baseline_solve_time"]
    runs["end_to_end_speedup_vs_baseline_total_pct"] = 100.0 * (runs["baseline_total_time"] - runs["end_to_end_time"]) / runs["baseline_total_time"]

    cases = data.get("all_cases", pd.DataFrame()).copy()
    if cases.empty:
        cases = baseline_ref[["case_name", "nb", "nl"]].copy()
    else:
        for c in cases.columns:
            if c != "case_name":
                cases[c] = pd.to_numeric(cases[c], errors="coerce")

    data["all_runs"] = runs
    data["all_cases"] = cases
    return data


# =============================================================================
# Derived analytics
# =============================================================================
def derive_combo_summary(runs: pd.DataFrame) -> pd.DataFrame:
    df = runs.copy()
    group_cols = ["family", "blocks", "seed_vm_policy", "seed_qg_policy"]

    all_rows = (
        df.groupby(group_cols, dropna=False)
        .agg(total_cases=("case_name", "nunique"))
        .reset_index()
    )

    success_df = df[df["success"]].copy()
    if success_df.empty:
        out = all_rows.copy()
        out["n"] = 0
        return out

    if "objective_matches_baseline" not in success_df.columns:
        success_df["objective_matches_baseline"] = np.nan

    success_agg = (
        success_df.groupby(group_cols, dropna=False)
        .agg(
            n=("case_name", "count"),
            avg_solve_time=("solve_time", "mean"),
            median_solve_time=("solve_time", "median"),
            avg_end_to_end_time=("end_to_end_time", "mean"),
            median_end_to_end_time=("end_to_end_time", "median"),
            avg_iterations=("iterations", "mean"),
            median_iterations=("iterations", "median"),
            n_matching_objective=("objective_matches_baseline",
                                  lambda s: int(pd.to_numeric(s, errors="coerce").fillna(0).sum())),
            share_matching_objective=("objective_matches_baseline",
                                      lambda s: safe_series_mean(pd.to_numeric(s, errors="coerce"))),
        )
        .reset_index()
    )

    out = all_rows.merge(success_agg, on=group_cols, how="left")
    out["n"] = out["n"].fillna(0).astype(int)
    return out


def best_combo_from_summary(combo_df, family, metric="median_solve_time", native_dc_only=False):
    sub = combo_df[(combo_df["family"] == family) & (combo_df["n"] > 0)].copy()
    if native_dc_only and family == "dc_seed":
        sub = sub[(sub["seed_vm_policy"] == "") & (sub["seed_qg_policy"] == "")]
    if sub.empty:
        return None

    sub["combo_order"] = sub["blocks"].apply(combo_sort_key)
    sub = sub.sort_values([metric, "avg_solve_time", "combo_order"], ascending=True)
    return sub.iloc[0]


def build_casewise_best_table(runs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    good = runs[runs["success"]].copy()

    # All oracle combo families present in data
    present_combo_families = [f for f in ORACLE_COMBO_FAMILIES if f in set(good["family"])]

    for case_name, grp in good.groupby("case_name"):
        base = grp[grp["family"] == "baseline"].sort_values("solve_time").head(1)
        if base.empty:
            continue
        base = base.iloc[0]

        def pick_best(df, metric="solve_time"):
            matched = objective_matched_rows(df)
            if matched.empty:
                return None
            return matched.sort_values([metric, "solve_time", "run_name"]).head(1).iloc[0]

        row = {
            "case_name": case_name,
            "nb": base.get("nb"),
            "nl": base.get("nl"),
            "baseline_solve_time": base.get("solve_time"),
            "baseline_total_time": base.get("total_time"),
            "baseline_iter": base.get("iterations"),
        }

        # Best for each oracle combo family
        for fam in present_combo_families:
            fam_grp = grp[grp["family"] == fam].copy()
            best = pick_best(fam_grp)
            short = family_short_label(fam).lower().replace("-", "_")
            row[f"best_{short}_run"] = None if best is None else best.get("run_name")
            row[f"best_{short}_blocks"] = None if best is None else best.get("blocks")
            row[f"best_{short}_time_s"] = None if best is None else best.get("solve_time")
            row[f"best_{short}_iter"] = None if best is None else best.get("iterations")
            row[f"best_{short}_speedup_pct"] = None if best is None else speedup_pct(base.get("solve_time"), best.get("solve_time"))

        # DC-seeded
        dc_grp = grp[(grp["family"] == "dc_seed") & (grp["seed_vm_policy"] == "") & (grp["seed_qg_policy"] == "")].copy()
        dc_solve = pick_best(dc_grp, "solve_time")
        dc_e2e = pick_best(dc_grp, "end_to_end_time")

        row["best_dc_solve_run"] = None if dc_solve is None else dc_solve.get("run_name")
        row["best_dc_solve_blocks"] = None if dc_solve is None else dc_solve.get("blocks")
        row["best_dc_solve_time_s"] = None if dc_solve is None else dc_solve.get("solve_time")
        row["best_dc_solve_iter"] = None if dc_solve is None else dc_solve.get("iterations")
        row["best_dc_solve_speedup_pct"] = None if dc_solve is None else speedup_pct(base.get("solve_time"), dc_solve.get("solve_time"))
        row["best_dc_e2e_run"] = None if dc_e2e is None else dc_e2e.get("run_name")
        row["best_dc_e2e_blocks"] = None if dc_e2e is None else dc_e2e.get("blocks")
        row["best_dc_e2e_time_s"] = None if dc_e2e is None else dc_e2e.get("end_to_end_time")
        row["best_dc_e2e_iter"] = None if dc_e2e is None else dc_e2e.get("iterations")
        row["best_dc_e2e_speedup_pct"] = None if dc_e2e is None else speedup_pct(base.get("total_time"), dc_e2e.get("end_to_end_time"))

        # Dual-only families
        for fam in DUAL_ONLY_FAMILIES:
            fam_grp = grp[grp["family"] == fam].copy()
            best = pick_best(fam_grp)
            short = family_short_label(fam).lower().replace("-", "_")
            row[f"best_{short}_time_s"] = None if best is None else best.get("solve_time")
            row[f"best_{short}_speedup_pct"] = None if best is None else speedup_pct(base.get("solve_time"), best.get("solve_time"))

        # Overall best across all non-baseline
        all_nonbase = objective_matched_rows(grp[grp["family"] != "baseline"].copy())
        overall = pick_best(all_nonbase) if not all_nonbase.empty else None
        row["best_overall_speedup_pct"] = None if overall is None else speedup_pct(base.get("solve_time"), overall.get("solve_time"))

        # Backward-compat aliases for existing code
        row["best_oracle_po_time_s"] = row.get("best_o_po_time_s")
        row["best_oracle_po_speedup_pct"] = row.get("best_o_po_speedup_pct")
        row["best_oracle_pd_time_s"] = row.get("best_o_pd_time_s")
        row["best_oracle_pd_speedup_pct"] = row.get("best_o_pd_speedup_pct")

        rows.append(row)

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["nb", "nl", "case_name"]).reset_index(drop=True)
    return out


def compute_combo_wins(runs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    good = runs[(runs["success"]) & (runs["family"] != "baseline")].copy()
    good = good[(good["family"] != "dc_seed") | ((good["seed_vm_policy"] == "") & (good["seed_qg_policy"] == ""))]

    present_families = [f for f in ORACLE_COMBO_FAMILIES + ["dc_seed"] if f in set(good["family"])]

    for fam in present_families:
        sub = objective_matched_rows(good[good["family"] == fam].copy())
        if sub.empty:
            continue
        metric = "end_to_end_time" if fam == "dc_seed" else "solve_time"
        winners = sub.sort_values(["case_name", metric, "solve_time", "run_name"]).groupby("case_name").head(1)
        counts = winners.groupby(["blocks"]).size().reset_index(name="case_wins")
        counts["family"] = fam
        rows.append(counts)

    if rows:
        return pd.concat(rows, ignore_index=True)
    return pd.DataFrame(columns=["family", "blocks", "case_wins"])


def compute_family_wins(casewise_best: pd.DataFrame) -> pd.DataFrame:
    if casewise_best.empty:
        return pd.DataFrame(columns=["family", "case_wins_by_best_solve_time"])

    # Collect all family time columns dynamically
    family_time_cols = {}
    for fam in ORACLE_COMBO_FAMILIES:
        short = family_short_label(fam).lower().replace("-", "_")
        col = f"best_{short}_time_s"
        if col in casewise_best.columns:
            family_time_cols[fam] = col
    if "best_dc_solve_time_s" in casewise_best.columns:
        family_time_cols["dc_seed"] = "best_dc_solve_time_s"

    counts = {fam: 0 for fam in family_time_cols}

    for _, row in casewise_best.iterrows():
        candidates = []
        for fam, col in family_time_cols.items():
            if pd.notna(row.get(col)):
                candidates.append((fam, float(row[col])))
        if candidates:
            fam, _ = min(candidates, key=lambda x: (x[1], x[0]))
            counts[fam] += 1

    rows = [{"family": fam, "case_wins_by_best_solve_time": c} for fam, c in counts.items()]
    return pd.DataFrame(rows)


def rebuild_block_effects_from_runs(runs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    oracle_families = [f for f in ORACLE_COMBO_FAMILIES + ["dc_seed"] if f in set(runs["family"])]

    for fam in oracle_families:
        fam_sub = runs[(runs["family"] == fam) & (runs["success"])].copy()
        if fam_sub.empty:
            continue

        if "objective_matches_baseline" in fam_sub.columns:
            matched = fam_sub[fam_sub["objective_matches_baseline"] == 1].copy()
            if not matched.empty:
                fam_sub = matched

        fam_sub["blocks"] = fam_sub["blocks"].fillna("").astype(str)
        fam_sub["solve_time"] = pd.to_numeric(fam_sub["solve_time"], errors="coerce")

        for blk in BLOCK_ORDER:
            has_blk = fam_sub["blocks"].str.split("+").apply(lambda xs: blk in xs if isinstance(xs, list) else False)
            with_vals = fam_sub.loc[has_blk, "solve_time"]
            without_vals = fam_sub.loc[~has_blk, "solve_time"]
            with_vals = with_vals[np.isfinite(with_vals)]
            without_vals = without_vals[np.isfinite(without_vals)]

            rows.append({
                "family": fam,
                "block": blk,
                "n_with": int(len(with_vals)),
                "n_without": int(len(without_vals)),
                "avg_solve_with": float(with_vals.mean()) if len(with_vals) else np.nan,
                "avg_solve_without": float(without_vals.mean()) if len(without_vals) else np.nan,
                "pct_better_when_included": (
                    100.0 * (without_vals.mean() - with_vals.mean()) / without_vals.mean()
                    if len(with_vals) and len(without_vals) and float(without_vals.mean()) > 0
                    else np.nan
                ),
            })

    return pd.DataFrame(rows)


def hodges_lehmann(diff: Sequence[float]) -> float:
    vals = np.asarray([d for d in diff if np.isfinite(d)], dtype=float)
    if vals.size == 0:
        return np.nan
    walsh = []
    for i in range(len(vals)):
        for j in range(i, len(vals)):
            walsh.append(0.5 * (vals[i] + vals[j]))
    return float(np.median(walsh)) if walsh else np.nan


def holm_adjust(pvals: Sequence[float]) -> List[float]:
    pvals = [float(p) if p is not None and np.isfinite(p) else np.nan for p in pvals]
    out = [np.nan] * len(pvals)
    finite = [(i, p) for i, p in enumerate(pvals) if np.isfinite(p)]
    if not finite:
        return out
    ordered = sorted(finite, key=lambda x: x[1])
    running = 0.0
    m = len(ordered)
    for rank, (idx, p) in enumerate(ordered):
        val = (m - rank) * p
        running = max(running, val)
        out[idx] = min(running, 1.0)
    return out


def bonf_adjust(pvals: Sequence[float]) -> List[float]:
    pvals = [float(p) if p is not None and np.isfinite(p) else np.nan for p in pvals]
    m = sum(np.isfinite(p) for p in pvals)
    return [min(p * m, 1.0) if np.isfinite(p) else np.nan for p in pvals]


def pairwise_wilcoxon_table(casewise_best: pd.DataFrame) -> pd.DataFrame:
    rows = []

    def add_row(name, col_a, col_b):
        if col_a not in casewise_best.columns or col_b not in casewise_best.columns:
            return
        sub = casewise_best[[col_a, col_b]].dropna()
        if sub.empty:
            return
        a = pd.to_numeric(sub[col_a], errors="coerce").to_numpy(dtype=float)
        b = pd.to_numeric(sub[col_b], errors="coerce").to_numpy(dtype=float)
        keep = np.isfinite(a) & np.isfinite(b)
        a, b = a[keep], b[keep]
        if len(a) == 0:
            return
        diff = b - a
        pval = np.nan
        if wilcoxon is not None and np.any((a - b) != 0.0):
            try:
                pval = float(wilcoxon(a, b, zero_method="wilcox", alternative="two-sided").pvalue)
            except Exception:
                pval = np.nan
        rows.append({
            "Comparison": name,
            "n": int(len(a)),
            "Med. A [s]": float(np.median(a)),
            "Med. B [s]": float(np.median(b)),
            "Wins A": int(np.sum(a < b)),
            "Wins B": int(np.sum(b < a)),
            "HL [s]": hodges_lehmann(diff),
            "Raw p": pval,
        })

    # Core comparisons
    add_row("Baseline vs. oracle primal-only [UB]", "baseline_solve_time", "best_o_po_time_s")
    add_row("Baseline vs. oracle primal+dual [UB]", "baseline_solve_time", "best_o_pd_time_s")
    add_row("Baseline solve vs. best DC solve [UB]", "baseline_solve_time", "best_dc_solve_time_s")
    add_row("Baseline total vs. best DC E2E [UB]", "baseline_total_time", "best_dc_e2e_time_s")
    add_row("Oracle primal+dual vs. DC solve [UB]", "best_o_pd_time_s", "best_dc_solve_time_s")
    add_row("Oracle primal+dual vs. DC E2E [UB]", "best_o_pd_time_s", "best_dc_e2e_time_s")

    # Extended dual decomposition comparisons
    add_row("Oracle primal+dual vs. constraint-dual [UB]", "best_o_pd_time_s", "best_o_cd_time_s")
    add_row("Oracle primal+dual vs. bounds-dual [UB]", "best_o_pd_time_s", "best_o_bd_time_s")
    add_row("Oracle primal+dual vs. primal+dual all-bounds [UB]", "best_o_pd_time_s", "best_o_pd_ab_time_s")
    add_row("Constraint-dual vs. bounds-dual [UB]", "best_o_cd_time_s", "best_o_bd_time_s")
    add_row("Baseline vs. constraint-dual [UB]", "baseline_solve_time", "best_o_cd_time_s")
    add_row("Baseline vs. bounds-dual [UB]", "baseline_solve_time", "best_o_bd_time_s")

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out["Holm p"] = holm_adjust(out["Raw p"].tolist())
    out["Bonf. p"] = bonf_adjust(out["Raw p"].tolist())
    out["Sig. (Holm)"] = out["Holm p"] <= 0.05
    out["Sig. (Bonf.)"] = out["Bonf. p"] <= 0.05
    return out


# =============================================================================
# Figure helpers
# =============================================================================
def setup_publication_style():
    mpl.rcParams.update({
        "figure.dpi": 200, "savefig.dpi": 300, "font.size": 10,
        "axes.titlesize": 11, "axes.labelsize": 10, "legend.fontsize": 8,
        "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.grid": True, "grid.alpha": 0.20, "grid.linestyle": "--",
        "grid.linewidth": 0.6, "lines.linewidth": 1.7,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.constrained_layout.use": True,
    })


def save_figure(fig, stem):
    paths = []
    for ext in (".pdf", ".png"):
        out = stem.with_suffix(ext)
        fig.savefig(out, bbox_inches="tight")
        paths.append(str(out))
    plt.close(fig)
    return paths


def fig_case_size_vs_runtime(runs, casewise_best, fig_dir):
    base = runs[(runs["family"] == "baseline") & (runs["success"])]
    if base.empty or casewise_best.empty:
        return []

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.scatter(base["nb"], base["solve_time"], s=28, alpha=0.70, label="Baseline solve time")

    for col, label in [
        ("best_o_pd_time_s", "Best oracle AC primal+dual"),
        ("best_dc_solve_time_s", "Best DC-seeded AC solve"),
        ("best_dc_e2e_time_s", "Best DC end-to-end"),
    ]:
        if col in casewise_best.columns and casewise_best[col].notna().any():
            ax.scatter(casewise_best["nb"], casewise_best[col], s=30, alpha=0.80, label=label)

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Number of buses"); ax.set_ylabel("Time [s]")
    ax.set_title("Case size versus runtime")
    ax.legend(frameon=False, ncol=2)
    return save_figure(fig, fig_dir / FIG_PREFIXES["case_size_runtime"])


def fig_performance_profile(runs, combo_summary, fig_dir):
    methods = []
    methods.append("baseline_default")
    for fam in ORACLE_COMBO_FAMILIES:
        best = best_combo_from_summary(combo_summary, fam)
        if best is not None:
            methods.append(f"{fam}__{str(best['blocks']).replace('+', '_')}")
    native_dc = combo_summary[
        (combo_summary["family"] == "dc_seed") & (combo_summary["seed_vm_policy"] == "") &
        (combo_summary["seed_qg_policy"] == "") & (combo_summary["n"] > 0)
    ].copy()
    for blocks in ["Pg", "Va", "Pg+Va"]:
        if blocks in set(native_dc["blocks"]):
            methods.append(f"dc_seed__{blocks.replace('+', '_')}")

    seen = set()
    methods = [m for m in methods if not (m in seen or seen.add(m))]

    sub = runs[(runs["run_name"].isin(methods)) & (runs["success"])][["case_name", "run_name", "end_to_end_time"]].copy()
    if sub.empty:
        return []

    wide = sub.pivot(index="case_name", columns="run_name", values="end_to_end_time").replace([np.inf, -np.inf], np.nan)
    denom = wide.min(axis=1, skipna=True)
    valid = denom.notna()
    wide, denom = wide.loc[valid], denom.loc[valid]
    if wide.empty:
        return []

    ratios = wide.div(denom, axis=0)
    max_ratio = np.nanmax(ratios.to_numpy(dtype=float))
    if not np.isfinite(max_ratio):
        return []

    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    taus = np.linspace(1.0, max(3.0, max_ratio * 1.02), 250)

    for method in methods:
        if method not in ratios.columns:
            continue
        vals = pd.to_numeric(ratios[method], errors="coerce").to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        profile = [float(np.mean(vals <= tau)) for tau in taus]
        label = method.replace("oracle_ac_", "oracle ").replace("dc_seed", "dc seed").replace("_", " ")
        ax.plot(taus, profile, label=label)

    ax.set_xlabel(r"Performance ratio $\tau$"); ax.set_ylabel("Fraction of cases")
    ax.set_title("Performance profile of representative static methods")
    ax.set_ylim(0, 1.02)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.legend(frameon=False, ncol=2, fontsize=6)
    return save_figure(fig, fig_dir / FIG_PREFIXES["performance_profile"])


def fig_casewise_speedups(casewise_best, fig_dir):
    sort_col = "best_o_pd_speedup_pct"
    if sort_col not in casewise_best.columns or casewise_best[sort_col].isna().all():
        return []

    df = casewise_best.dropna(subset=[sort_col]).sort_values(sort_col).reset_index(drop=True)
    if df.empty:
        return []

    x = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(7.8, 4.4))

    for col, label in [
        ("best_o_po_speedup_pct", "Best oracle AC primal-only"),
        ("best_o_pd_speedup_pct", "Best oracle AC primal+dual"),
        ("best_dc_solve_speedup_pct", "Best DC-seeded AC solve"),
        ("best_dc_e2e_speedup_pct", "Best DC end-to-end"),
    ]:
        if col in df.columns and df[col].notna().any():
            ax.plot(x, df[col], label=label)

    ax.axhline(0.0, linewidth=1.0)
    ax.set_xlabel("Cases sorted by best oracle AC primal+dual speedup")
    ax.set_ylabel("Speedup relative to baseline [%]")
    ax.set_title("Per-case speedup distribution")
    ax.legend(frameon=False, ncol=2)
    return save_figure(fig, fig_dir / FIG_PREFIXES["casewise_speedups"])


def _oracle_combo_heatmap(runs, family, value_col):
    combos = sorted(
        {b for b in runs.loc[runs["family"] == family, "blocks"].dropna().astype(str) if b != ""},
        key=combo_sort_key,
    )
    vals, labels = [], []
    for b in combos:
        sub = runs[(runs["family"] == family) & (runs["blocks"] == b) & (runs["success"])]
        vals.append(sub[value_col].median() if not sub.empty else np.nan)
        labels.append(b)
    return np.array(vals, dtype=float).reshape(-1, 1), labels


def fig_oracle_combo_heatmap(runs, fig_dir):
    """Heatmap with panels for primal-only and primal+dual."""
    families = ["oracle_ac_primal_only", "oracle_ac_primal_dual"]
    families = [f for f in families if f in set(runs["family"])]
    if not families:
        return []

    mats, labels = [], []
    for fam in families:
        mat, lbl = _oracle_combo_heatmap(runs, fam, "speedup_vs_baseline_pct")
        if mat.size == 0:
            return []
        mats.append(mat)
        labels.append(lbl)

    nrows = max(mat.shape[0] for mat in mats)
    fig, axes = plt.subplots(1, len(families), figsize=(4.0 * len(families), max(4.0, 0.18 * nrows + 1.7)), sharex=False)
    if len(families) == 1:
        axes = [axes]

    for ax, fam, mat, lbl in zip(axes, families, mats, labels):
        padded = np.full((nrows, 1), np.nan)
        padded[:mat.shape[0], 0] = mat[:, 0]
        im = ax.imshow(padded, aspect="auto", interpolation="nearest")
        ax.set_xticks([0])
        ax.set_xticklabels([FAMILY_LABEL.get(fam, fam)], fontsize=7)
        ax.set_yticks(np.arange(nrows))
        ax.set_yticklabels([human_blocks(x) if x else "" for x in lbl + [""] * (nrows - len(lbl))])
        ax.grid(False)

    cbar = fig.colorbar(im, ax=axes, shrink=0.95)
    cbar.set_label("Median speedup vs baseline [%]")
    fig.suptitle("Oracle AC combination heatmap")
    return save_figure(fig, fig_dir / FIG_PREFIXES["oracle_combo_heatmap_speedup"])


def fig_dual_decomposition_heatmap(runs, fig_dir):
    """Extended heatmap showing all 5 oracle dual-mode families side by side."""
    families = [f for f in ORACLE_COMBO_FAMILIES if f in set(runs["family"])]
    if len(families) < 3:
        return []

    mats, labels_list = [], []
    for fam in families:
        mat, lbl = _oracle_combo_heatmap(runs, fam, "speedup_vs_baseline_pct")
        if mat.size == 0:
            continue
        mats.append(mat)
        labels_list.append((fam, lbl))

    if len(mats) < 3:
        return []

    nrows = max(m.shape[0] for m in mats)
    ncols = len(mats)
    fig, axes = plt.subplots(1, ncols, figsize=(2.4 * ncols + 1.5, max(4.0, 0.22 * nrows + 1.5)), sharex=False, sharey=True)

    all_vals = np.concatenate([m.ravel() for m in mats])
    all_vals = all_vals[np.isfinite(all_vals)]
    vmin = np.nanmin(all_vals) if len(all_vals) else -100
    vmax = np.nanmax(all_vals) if len(all_vals) else 50

    for i, (ax, mat, (fam, lbl)) in enumerate(zip(axes, mats, labels_list)):
        padded = np.full((nrows, 1), np.nan)
        padded[:mat.shape[0], 0] = mat[:, 0]
        im = ax.imshow(padded, aspect="auto", interpolation="nearest", vmin=vmin, vmax=vmax, cmap="RdYlGn")
        ax.set_xticks([0])
        panel_label = DUAL_MODE_PANEL_LABELS.get(fam, family_short_label(fam))
        ax.set_xticklabels([panel_label], fontsize=7)
        if i == 0:
            ax.set_yticks(np.arange(nrows))
            ax.set_yticklabels([human_blocks(x) if x else "" for x in lbl + [""] * (nrows - len(lbl))], fontsize=7)
        else:
            ax.set_yticks([])
        ax.grid(False)

    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.85)
    cbar.set_label("Median speedup vs baseline [%]")
    fig.suptitle("Dual decomposition: median speedup by initialization block and dual mode", fontsize=10)
    return save_figure(fig, fig_dir / FIG_PREFIXES["dual_decomposition_heatmap"])


def fig_dual_only_bar(runs, fig_dir):
    """Bar chart for dual-only experiments (no primal blocks)."""
    dual_only_fams = [f for f in DUAL_ONLY_FAMILIES if f in set(runs["family"])]
    if not dual_only_fams:
        return []

    data = []
    for fam in dual_only_fams:
        sub = runs[(runs["family"] == fam) & (runs["success"])].copy()
        if sub.empty:
            continue
        data.append({
            "family": fam,
            "label": FAMILY_LABEL.get(fam, fam),
            "median_speedup": sub["speedup_vs_baseline_pct"].median(),
            "n_converged": len(sub),
            "n_total": len(runs[runs["family"] == fam]),
        })

    if not data:
        return []

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    labels = [d["label"].replace("Oracle ", "").replace("dual-only ", "") for d in data]
    values = [d["median_speedup"] for d in data]
    colors = ["#d32f2f" if v < 0 else "#388e3c" for v in values]
    bars = ax.bar(range(len(data)), values, color=colors, alpha=0.8)

    for bar, d in zip(bars, data):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{d['n_converged']}/{d['n_total']}",
                ha="center", va="bottom", fontsize=8)

    ax.set_xticks(range(len(data)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Median speedup vs baseline [%]")
    ax.set_title("Dual-only restarts (no primal blocks)")
    ax.axhline(0.0, linewidth=1.0, color="black")
    return save_figure(fig, fig_dir / FIG_PREFIXES["dual_only_bar"])


def fig_block_effects(runs, fig_dir):
    sub = rebuild_block_effects_from_runs(runs)
    if sub.empty:
        return []

    sub = sub.dropna(subset=["pct_better_when_included"]).copy()
    fams = [f for f in ORACLE_COMBO_FAMILIES + ["dc_seed"] if f in set(sub["family"])]
    if not fams:
        return []

    fig, ax = plt.subplots(figsize=(max(7.6, 1.8 * len(fams) + 2), 4.2))
    x = np.arange(len(BLOCK_ORDER))
    width = 0.8 / max(len(fams), 1)

    for i, fam in enumerate(fams):
        fam_sub = sub[sub["family"] == fam].set_index("block").reindex(BLOCK_ORDER)
        vals = fam_sub["pct_better_when_included"].to_numpy(dtype=float)
        ax.bar(x + (i - (len(fams) - 1) / 2) * width, vals, width=width,
               label=FAMILY_LABEL.get(fam, fam), alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(BLOCK_ORDER)
    ax.set_ylabel("Average solve-time benefit when included [%]")
    ax.set_title("Average benefit of each initialization block")
    ax.axhline(0.0, linewidth=1.0)
    ax.legend(frameon=False, fontsize=6, ncol=2)
    return save_figure(fig, fig_dir / FIG_PREFIXES["block_effects"])


def fig_speedup_vs_size(casewise_best, fig_dir):
    df = casewise_best.dropna(subset=["nb"]).copy()
    if df.empty:
        return []

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    plotted = 0
    for col, label in [
        ("best_o_pd_speedup_pct", "Best oracle AC primal+dual"),
        ("best_o_cd_speedup_pct", "Best oracle constraint-dual"),
        ("best_o_bd_speedup_pct", "Best oracle bounds-dual"),
        ("best_dc_solve_speedup_pct", "Best DC-seeded AC solve"),
        ("best_dc_e2e_speedup_pct", "Best DC end-to-end"),
    ]:
        if col in df.columns and df[col].notna().any():
            ax.scatter(df["nb"], df[col], s=28, alpha=0.8, label=label)
            plotted += 1

    if plotted == 0:
        plt.close(fig)
        return []

    ax.set_xscale("log")
    ax.set_xlabel("Number of buses"); ax.set_ylabel("Speedup relative to baseline [%]")
    ax.set_title("Speedup versus case size")
    ax.axhline(0.0, linewidth=1.0)
    ax.legend(frameon=False, ncol=2, fontsize=7)
    return save_figure(fig, fig_dir / FIG_PREFIXES["speedup_vs_case_size"])


def fig_speedup_boxplot(casewise_best, fig_dir):
    data, labels = [], []
    candidates = [
        ("best_o_po_speedup_pct", "Best oracle\nAC primal-only"),
        ("best_o_pd_speedup_pct", "Best oracle\nAC primal+dual"),
        ("best_o_cd_speedup_pct", "Best oracle\nconstraint-dual"),
        ("best_o_bd_speedup_pct", "Best oracle\nbounds-dual"),
        ("best_dc_solve_speedup_pct", "Best DC\nAC solve"),
        ("best_dc_e2e_speedup_pct", "Best DC\nend-to-end"),
    ]
    for col, label in candidates:
        if col not in casewise_best.columns:
            continue
        vals = pd.to_numeric(casewise_best[col], errors="coerce").to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if len(vals):
            data.append(vals)
            labels.append(label)

    if not data:
        return []

    fig, ax = plt.subplots(figsize=(max(7.2, 1.4 * len(data) + 1), 4.2))
    ax.boxplot(data, tick_labels=labels, showfliers=False)
    ax.axhline(0.0, linewidth=1.0)
    ax.set_ylabel("Speedup relative to baseline [%]")
    ax.set_title("Distribution of per-case speedups")
    ax.tick_params(axis="x", labelsize=7)
    return save_figure(fig, fig_dir / FIG_PREFIXES["speedup_distributions"])


def fig_baseline_vs_best(casewise_best, fig_dir):
    df = casewise_best.copy()
    if df.empty:
        return []

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.5), sharex=False, sharey=False)

    left_cols = ["baseline_solve_time", "best_o_pd_time_s", "best_dc_solve_time_s"]
    left = df.dropna(subset=[c for c in left_cols if c in df.columns])
    right_cols = ["baseline_total_time", "best_dc_e2e_time_s"]
    right = df.dropna(subset=[c for c in right_cols if c in df.columns])

    if not left.empty and "best_o_pd_time_s" in left.columns and "best_dc_solve_time_s" in left.columns:
        lo = min(left["baseline_solve_time"].min(), left["best_o_pd_time_s"].min(), left["best_dc_solve_time_s"].min())
        hi = max(left["baseline_solve_time"].max(), left["best_o_pd_time_s"].max(), left["best_dc_solve_time_s"].max())
        axes[0].scatter(left["baseline_solve_time"], left["best_o_pd_time_s"], s=28, alpha=0.80, label="Best oracle AC primal+dual")
        axes[0].scatter(left["baseline_solve_time"], left["best_dc_solve_time_s"], s=28, alpha=0.80, label="Best DC-seeded AC solve")
        axes[0].plot([lo, hi], [lo, hi], linewidth=1.2)
        axes[0].set_xscale("log"); axes[0].set_yscale("log")
        axes[0].set_xlabel("Baseline solve time [s]"); axes[0].set_ylabel("Warm-start solve time [s]")
        axes[0].set_title("Solve-time comparison"); axes[0].legend(frameon=False)
    else:
        axes[0].set_visible(False)

    if not right.empty and "best_dc_e2e_time_s" in right.columns:
        lo = min(right["baseline_total_time"].min(), right["best_dc_e2e_time_s"].min())
        hi = max(right["baseline_total_time"].max(), right["best_dc_e2e_time_s"].max())
        axes[1].scatter(right["baseline_total_time"], right["best_dc_e2e_time_s"], s=28, alpha=0.80)
        axes[1].plot([lo, hi], [lo, hi], linewidth=1.2)
        axes[1].set_xscale("log"); axes[1].set_yscale("log")
        axes[1].set_xlabel("Baseline total time [s]"); axes[1].set_ylabel("Best DC end-to-end time [s]")
        axes[1].set_title("Practical workflow comparison")
    else:
        axes[1].set_visible(False)

    return save_figure(fig, fig_dir / FIG_PREFIXES["baseline_vs_best"])


def fig_dc_native_boxplot(runs, fig_dir):
    dc = runs[
        (runs["family"] == "dc_seed") & (runs["blocks"].isin(["Pg", "Va", "Pg+Va"])) &
        (runs["success"]) & (runs["seed_vm_policy"] == "") & (runs["seed_qg_policy"] == "")
    ].copy()
    if dc.empty:
        return []

    data, labels = [], []
    for blk in ["Pg", "Va", "Pg+Va"]:
        vals = pd.to_numeric(dc.loc[dc["blocks"] == blk, "end_to_end_speedup_vs_baseline_total_pct"], errors="coerce").to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if len(vals):
            data.append(vals)
            labels.append(human_blocks(blk))

    if not data:
        return []

    fig, ax = plt.subplots(figsize=(6.6, 4.1))
    ax.boxplot(data, tick_labels=labels, showfliers=False)
    ax.axhline(0.0, linewidth=1.0)
    ax.set_ylabel("End-to-end speedup vs baseline total time [%]")
    ax.set_title("Native DC blocks as practical initializers")
    return save_figure(fig, fig_dir / FIG_PREFIXES["dc_native_boxplot"])


# =============================================================================
# Table writers — every builder emits both .csv and .tex
# =============================================================================
def write_text(path, text):
    path.write_text(text, encoding="utf-8")
    return str(path)


def _render_cell(v) -> str:
    if pd.isna(v) if isinstance(v, float) else (v is None):
        return ""
    return str(v)


def build_global_summary_table(runs, combo_summary, casewise_best, out_dir):
    baseline = runs[(runs["family"] == "baseline") & (runs["success"])].copy()
    total_cases = int(runs[runs["family"] == "baseline"]["case_name"].nunique())

    rows = [{
        "Method": "Baseline case-data start",
        "Cases": total_cases,
        "Converged": int(len(baseline)),
        "Median solve [s]": baseline["solve_time"].median(),
        "Median total [s]": baseline["total_time"].median(),
        "Median E2E [s]": baseline["end_to_end_time"].median(),
        "Median iter": baseline["iterations"].median(),
        "Median speedup [%]": 0.0,
        "group": "base",
    }]

    for fam in ORACLE_COMBO_FAMILIES + ["dc_seed"]:
        native_dc = (fam == "dc_seed")
        best = best_combo_from_summary(combo_summary, fam, native_dc_only=native_dc)
        if best is None:
            continue
        filter_cond = (runs["family"] == fam) & (runs["blocks"] == best["blocks"]) & (runs["success"])
        if native_dc:
            filter_cond = filter_cond & (runs["seed_vm_policy"] == "") & (runs["seed_qg_policy"] == "")
        sub = runs[filter_cond].copy()
        rows.append({
            "Method": f"Minimum-median static policy: {FAMILY_LABEL.get(fam, fam)} ({blocks_to_math(best['blocks'])})",
            "Cases": total_cases,
            "Converged": int(len(sub)),
            "Median solve [s]": sub["solve_time"].median(),
            "Median total [s]": sub["total_time"].median(),
            "Median E2E [s]": sub["end_to_end_time"].median(),
            "Median iter": sub["iterations"].median(),
            "Median speedup [%]": sub["speedup_vs_baseline_pct"].median(),
            "group": "static",
        })

    casewise_entries = [
        ("oracle_ac_primal_only", "best_o_po_time_s", "best_o_po_speedup_pct", "best_o_po_iter", "Case-wise best oracle AC primal-only"),
        ("oracle_ac_primal_dual", "best_o_pd_time_s", "best_o_pd_speedup_pct", "best_o_pd_iter", "Case-wise best oracle AC primal+dual"),
        ("oracle_ac_constraint_dual", "best_o_cd_time_s", "best_o_cd_speedup_pct", "best_o_cd_iter", "Case-wise best constraint-dual"),
        ("oracle_ac_bounds_dual", "best_o_bd_time_s", "best_o_bd_speedup_pct", "best_o_bd_iter", "Case-wise best bounds-dual"),
        ("oracle_ac_primal_dual_all_bounds", "best_o_pd_ab_time_s", "best_o_pd_ab_speedup_pct", "best_o_pd_ab_iter", "Case-wise best primal+dual all-bounds"),
    ]
    for fam, col_time, col_spd, col_iter, label in casewise_entries:
        if col_time not in casewise_best.columns or casewise_best[col_time].isna().all():
            continue
        rows.append({
            "Method": label,
            "Cases": total_cases,
            "Converged": int(casewise_best[col_time].notna().sum()),
            "Median solve [s]": casewise_best[col_time].median(),
            "Median total [s]": np.nan,
            "Median E2E [s]": np.nan,
            "Median iter": casewise_best[col_iter].median() if col_iter in casewise_best.columns else np.nan,
            "Median speedup [%]": casewise_best[col_spd].median() if col_spd in casewise_best.columns else np.nan,
            "group": "casewise",
        })

    for col_time, col_spd, col_iter, label in [
        ("best_dc_solve_time_s", "best_dc_solve_speedup_pct", "best_dc_solve_iter", "Case-wise best DC-seeded AC solve"),
        ("best_dc_e2e_time_s", "best_dc_e2e_speedup_pct", "best_dc_e2e_iter", "Case-wise best DC end-to-end"),
    ]:
        if col_time not in casewise_best.columns or casewise_best[col_time].isna().all():
            continue
        rows.append({
            "Method": label,
            "Cases": total_cases,
            "Converged": int(casewise_best[col_time].notna().sum()),
            "Median solve [s]": casewise_best[col_time].median(),
            "Median total [s]": np.nan,
            "Median E2E [s]": np.nan,
            "Median iter": casewise_best[col_iter].median() if col_iter in casewise_best.columns else np.nan,
            "Median speedup [%]": casewise_best[col_spd].median() if col_spd in casewise_best.columns else np.nan,
            "group": "casewise",
        })

    df = pd.DataFrame(rows)
    csv_path = out_dir / f"{TABLE_PREFIXES['global_summary']}.csv"
    df.to_csv(csv_path, index=False)

    lines = [
        r"\begin{table*}[!t]",
        r"\centering",
        r"\caption{Global runtime summary across the baseline case-data start, oracle AC warm starts, and practical DC-seeded warm starts. ``Converged'' is the number of successful (non-failure) runs. Median times are over converged runs only. Case-wise best rows are ex-post upper-bound diagnostics; statistical comparisons on those rows are descriptive rather than inferential (Section~\ref{sec:stats}).}",
        r"\label{tab:global_summary}",
        r"\small",
        r"\setlength{\tabcolsep}{6pt}",
        r"\begin{adjustbox}{width=\textwidth}",
        r"\begin{tabular}{p{6.2cm}rrrrrrr}",
        r"\toprule",
        r"Method & Cases & Converged & Median solve [s] & Median total [s] & Median E2E [s] & Median iter & Median speedup [\%] \\",
        r"\midrule",
    ]
    for _, row in df[df["group"] == "base"].iterrows():
        lines.append(
            f"{row['Method']} & {fmt_int(row['Cases'])} & {fmt_int(row['Converged'])} & "
            f"{fmt_float(row['Median solve [s]'])} & {fmt_float(row['Median total [s]'])} & "
            f"{fmt_float(row['Median E2E [s]'])} & {fmt_int(row['Median iter'])} & "
            f"{fmt_pct_number(row['Median speedup [%]'])} \\\\")
    lines.append(r"\midrule")
    lines.append(r"\multicolumn{8}{l}{\textit{Minimum-median static policy (one fixed policy applied to all cases; selection by minimum median AC solve time over converged cases can favor narrower convergence sets)}} \\")
    for _, row in df[df["group"] == "static"].iterrows():
        lines.append(
            f"{row['Method']} & {fmt_int(row['Cases'])} & {fmt_int(row['Converged'])} & "
            f"{fmt_float(row['Median solve [s]'])} & {fmt_float(row['Median total [s]'])} & "
            f"{fmt_float(row['Median E2E [s]'])} & {fmt_int(row['Median iter'])} & "
            f"{fmt_pct_number(row['Median speedup [%]'])} \\\\")
    lines.append(r"\midrule")
    lines.append(r"\multicolumn{8}{l}{\textit{Case-wise best (ex-post upper bound per family; descriptive only)}} \\")
    for _, row in df[df["group"] == "casewise"].iterrows():
        total = r"---\rlap{$^\dagger$}" if pd.isna(row["Median total [s]"]) else fmt_float(row["Median total [s]"])
        e2e = r"---\rlap{$^\dagger$}" if pd.isna(row["Median E2E [s]"]) else fmt_float(row["Median E2E [s]"])
        lines.append(
            f"{row['Method']} & {fmt_int(row['Cases'])} & {fmt_int(row['Converged'])} & "
            f"{fmt_float(row['Median solve [s]'])} & {total} & {e2e} & "
            f"{fmt_int(row['Median iter'])} & {fmt_pct_number(row['Median speedup [%]'])} \\\\")
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{adjustbox}",
        r"{\footnotesize $^\dagger$ For case-wise best envelopes, each case may select a different combination, so total/E2E columns are omitted.}",
        r"\end{table*}",
    ])
    tex_path = out_dir / f"{TABLE_PREFIXES['global_summary']}.tex"
    write_text(tex_path, "\n".join(lines))
    return [str(csv_path), str(tex_path)]


def build_pairwise_table(pairwise, out_dir):
    csv_path = out_dir / f"{TABLE_PREFIXES['pairwise_tests']}.csv"
    pairwise.to_csv(csv_path, index=False)

    lines = [
        r"\begin{table*}[!t]",
        r"\centering",
        r"\caption{Matched-case pairwise Wilcoxon signed-rank comparisons~\cite{wilcoxon1945}. Holm correction uses $\alpha/(m-k+1)$ with $m=" + str(len(pairwise)) + r"$, $\alpha=0.05$~\cite{holm1979}. The Hodges--Lehmann estimator $\widehat{\Delta}_{HL}$~\cite{hodges1963estimates} is the median of all Walsh averages of paired differences; negative values indicate method~B is faster. All rows involve case-wise best envelopes ([UB] = ex-post upper-bound comparison), so the reported $p$-values and multiplicity adjustments are descriptive diagnostics rather than formal inferential claims.}",
        r"\label{tab:pairwise_tests}",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{adjustbox}{width=\textwidth}",
        r"\begin{tabular}{p{5.4cm}rrrrrrrrr}",
        r"\toprule",
        r"Comparison & $n$ & Med.\ A [s] & Med.\ B [s] & Wins A & Wins B & $\widehat{\Delta}_{HL}$ [s] & Raw $p$ & Holm $p$ & Bonf.\ $p$ " + r"\\",
        r"\midrule",
    ]
    for _, row in pairwise.iterrows():
        lines.append(
            (
                f"{row['Comparison']} & {fmt_int(row['n'])} & {fmt_float(row['Med. A [s]'])} & {fmt_float(row['Med. B [s]'])} & "
                f"{fmt_int(row['Wins A'])} & {fmt_int(row['Wins B'])} & {fmt_float(row['HL [s]'])} & "
                f"{pval_str(row['Raw p'])} & {pval_str(row['Holm p'])} & {pval_str(row['Bonf. p'])} "
                + r"\\"
            )
        )
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{adjustbox}",
        r"\end{table*}",
    ])
    tex_path = out_dir / f"{TABLE_PREFIXES['pairwise_tests']}.tex"
    write_text(tex_path, "\n".join(lines))
    return [str(csv_path), str(tex_path)]

def build_oracle_combo_rank_table(combo_summary, combo_wins, out_dir):
    wins = combo_wins.copy()
    csv_rows = []
    present_families = [f for f in ORACLE_COMBO_FAMILIES if f in set(combo_summary["family"])]

    lines = [
        r"\begin{table}[!t]",
        r"\centering",
        r"\scriptsize",
        r"\caption{Full ranking of all oracle AC combinations by median AC solve time. Rows are grouped by family. ``Converged'' is the number of successful runs. Case wins are tallied within each family.}",
        r"\label{tab:combo_rank_oracle}",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Combination & Converged & Med.\ solve [s] & Mean solve [s] & Med.\ iter & Wins \\",
        r"\midrule",
    ]

    for fam in present_families:
        sub = combo_summary[(combo_summary["family"] == fam) & (combo_summary["n"] > 0)].copy()
        sub = sub.sort_values(["median_solve_time", "avg_solve_time", "blocks"])
        lines.append(rf"\multicolumn{{6}}{{l}}{{\textit{{{FAMILY_LABEL.get(fam, fam)}}}}} \\")
        for _, row in sub.iterrows():
            cw = wins[(wins["family"] == fam) & (wins["blocks"] == row["blocks"])]["case_wins"]
            cw_val = int(cw.iloc[0]) if not cw.empty else 0
            csv_rows.append({
                "family": FAMILY_LABEL.get(fam, fam),
                "blocks": row["blocks"],
                "Converged": row["n"],
                "Median solve [s]": row["median_solve_time"],
                "Mean solve [s]": row["avg_solve_time"],
                "Median iter": row["median_iterations"],
                "Mean iter": row["avg_iterations"],
                "Case wins": cw_val,
            })
            lines.append(
                f"{blocks_to_math(row['blocks'])} & {fmt_int(row['n'])} & "
                f"{fmt_float(row['median_solve_time'])} & {fmt_float(row['avg_solve_time'])} & "
                f"{fmt_int(row['median_iterations'])} & {fmt_int(cw_val)} \\\\")
        if fam != present_families[-1]:
            lines.append(r"\midrule")

    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    df = pd.DataFrame(csv_rows)
    csv_path = out_dir / f"{TABLE_PREFIXES['combo_rank_oracle']}.csv"
    df.to_csv(csv_path, index=False)
    tex_path = out_dir / f"{TABLE_PREFIXES['combo_rank_oracle']}.tex"
    write_text(tex_path, "\n".join(lines))
    return [str(csv_path), str(tex_path)]


def build_dc_rank_table(combo_summary, combo_wins, out_dir):
    sub = combo_summary[
        (combo_summary["family"] == "dc_seed") & (combo_summary["seed_vm_policy"] == "") &
        (combo_summary["seed_qg_policy"] == "") & (combo_summary["n"] > 0)
    ].copy().sort_values(["median_end_to_end_time", "median_solve_time", "blocks"])

    wins = combo_wins[combo_wins["family"] == "dc_seed"]
    tbl_rows = []
    for _, row in sub.iterrows():
        cw = wins[wins["blocks"] == row["blocks"]]["case_wins"]
        tbl_rows.append({
            "Combination": row["blocks"],
            "Converged": row["n"],
            "Median solve [s]": row["median_solve_time"],
            "Median E2E [s]": row["median_end_to_end_time"],
            "Median iter": row["median_iterations"],
            "Case wins": int(cw.iloc[0]) if not cw.empty else 0,
        })
    df = pd.DataFrame(tbl_rows)
    csv_path = out_dir / f"{TABLE_PREFIXES['combo_rank_dc']}.csv"
    df.to_csv(csv_path, index=False)

    lines = [
        r"\begin{table}[!t]",
        r"\centering",
        r"\caption{Static ranking of DC-seeded combinations by median E2E time. The DCOPF presolve cost is common to all three. The ordering is descriptive and should not be read as a blanket recommendation for one-shot workflows.}",
        r"\label{tab:combo_rank_dc}",
        r"\small",
        r"\setlength{\tabcolsep}{6pt}",
        r"\begin{adjustbox}{width=\columnwidth}",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Combination & Converged & Median solve [s] & Median E2E [s] & Median iter & Case wins \\",
        r"\midrule",
    ]
    for _, row in df.iterrows():
        lines.append(
            f"{blocks_to_math(row['Combination'])} & {fmt_int(row['Converged'])} & "
            f"{fmt_float(row['Median solve [s]'])} & {fmt_float(row['Median E2E [s]'])} & "
            f"{fmt_int(row['Median iter'])} & {fmt_int(row['Case wins'])} \\\\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}", r"\end{table}"])
    tex_path = out_dir / f"{TABLE_PREFIXES['combo_rank_dc']}.tex"
    write_text(tex_path, "\n".join(lines))
    return [str(csv_path), str(tex_path)]


def build_block_effects_table(runs, out_dir):
    sub = rebuild_block_effects_from_runs(runs)
    sub["family_label"] = sub["family"].map(FAMILY_LABEL).fillna(sub["family"])
    csv_path = out_dir / f"{TABLE_PREFIXES['block_effects']}.csv"
    sub.to_csv(csv_path, index=False)

    lines = [
        r"\begin{table}[!t]", r"\centering",
        r"\caption{Marginal AC solve-time statistics for each initialization block. ``Avg.\ reduction [\%]'' $= (t_{\text{excl}} - t_{\text{incl}})/t_{\text{excl}} \times 100$. \textbf{These marginals are secondary diagnostics}; the combination heatmap (Fig.~\ref{fig:oracle_heatmap}) is the reliable guide.}",
        r"\label{tab:block_effects}", r"\small", r"\setlength{\tabcolsep}{5pt}",
        r"\begin{adjustbox}{width=\columnwidth}",
        r"\begin{tabular}{p{2.7cm}lrrr}", r"\toprule",
        r"Family & Block & Avg.\ incl.\ [s] & Avg.\ excl.\ [s] & Avg.\ reduction [\%] \\", r"\midrule",
    ]
    prev_fam = None
    for _, row in sub.iterrows():
        if prev_fam is not None and row["family"] != prev_fam:
            lines.append(r"\midrule")
        prev_fam = row["family"]
        lines.append(f"{latex_escape(row['family_label'])} & {blocks_to_math(row['block'])} & {fmt_float(row['avg_solve_with'])} & {fmt_float(row['avg_solve_without'])} & {fmt_pct_number(row['pct_better_when_included'])} \\\\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}", r"\end{table}"])
    tex_path = out_dir / f"{TABLE_PREFIXES['block_effects']}.tex"
    write_text(tex_path, "\n".join(lines))
    return [str(csv_path), str(tex_path)]


def build_largest_cases_table(casewise_best, out_dir):
    largest = casewise_best.sort_values(["nb", "nl"], ascending=[False, False]).head(8).copy()
    largest = largest.sort_values(["nb", "nl"]).reset_index(drop=True)
    csv_path = out_dir / f"{TABLE_PREFIXES['largest_cases']}.csv"
    largest.to_csv(csv_path, index=False)

    lines = [
        r"\begin{table*}[!t]", r"\centering",
        r"\caption{Warm-start performance on the eight largest benchmark instances. All best oracle PD results use the full-vector $V_a{+}V_m{+}P_g{+}Q_g$ restart. ``Best DC solve blocks'' identifies the DC combination with the shortest AC solve time.}",
        r"\label{tab:largest_cases}", r"\scriptsize", r"\setlength{\tabcolsep}{4.0pt}",
        r"\begin{adjustbox}{width=\textwidth}",
        r"\begin{tabular}{lrrrrrrrr}", r"\toprule",
        r"Case & $n_b$ & $n_l$ & Baseline solve [s] & Best oracle PD [s] & Best DC solve blocks & Best DC solve [s] & Best DC E2E [s] & Best speedup (any method) [\%] \\",
        r"\midrule",
    ]
    for _, row in largest.iterrows():
        pd_time = row.get("best_o_pd_time_s", np.nan)
        dc_time = row.get("best_dc_solve_time_s", np.nan)
        dc_blocks = row.get("best_dc_solve_blocks", "")
        dc_e2e = row.get("best_dc_e2e_time_s", np.nan)
        overall = row.get("best_overall_speedup_pct", np.nan)
        dc_mark = r"\rlap{$^*$}" if (pd.notna(dc_time) and pd.notna(row.get("baseline_solve_time")) and float(dc_time) > float(row["baseline_solve_time"])) else ""
        lines.append(
            f"\\texttt{{{latex_escape(row['case_name'])}}} & {fmt_int(row['nb'])} & {fmt_int(row['nl'])} & "
            f"{fmt_float(row['baseline_solve_time'])} & {fmt_float(pd_time)} & {blocks_to_math(dc_blocks)} & "
            f"{fmt_float(dc_time)}{dc_mark} & {fmt_float(dc_e2e)} & {fmt_pct_number(overall)} \\\
"
        )
    lines.extend([
        r"\bottomrule",
        r"\multicolumn{9}{l}{\footnotesize $^*$ All three DC combinations are slower than the baseline on this case.}",
        r"\end{tabular}", r"\end{adjustbox}", r"\end{table*}"])
    tex_path = out_dir / f"{TABLE_PREFIXES['largest_cases']}.tex"
    write_text(tex_path, "\n".join(lines))
    return [str(csv_path), str(tex_path)]


def build_family_wins_table(casewise_best, out_dir):
    df = compute_family_wins(casewise_best)
    df["family_label"] = df["family"].map(FAMILY_LABEL).fillna(df["family"])
    csv_path = out_dir / f"{TABLE_PREFIXES['family_wins']}.csv"
    df.to_csv(csv_path, index=False)

    lines = [
        r"\begin{table}[!t]", r"\centering",
        r"\caption{Case-wise family win counts by best AC solve time.}",
        r"\label{tab:family_wins}", r"\small",
        r"\begin{tabular}{lr}", r"\toprule",
        r"Family & Case wins \\", r"\midrule",
    ]
    for _, row in df.iterrows():
        lines.append(f"{latex_escape(row['family_label'])} & {fmt_int(row['case_wins_by_best_solve_time'])} \\\\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    tex_path = out_dir / f"{TABLE_PREFIXES['family_wins']}.tex"
    write_text(tex_path, "\n".join(lines))
    return [str(csv_path), str(tex_path)]


def build_combo_wins_table(combo_wins, out_dir):
    df = combo_wins[combo_wins["case_wins"] > 0].copy()
    df["family_label"] = df["family"].map(FAMILY_LABEL)
    csv_path = out_dir / f"{TABLE_PREFIXES['combo_wins']}.csv"
    df.to_csv(csv_path, index=False)

    lines = [
        r"\begin{table}[!t]", r"\centering",
        r"\caption{Counts of case-wise winning combinations within each initialization family. A combination wins a case if it achieves the shortest AC solve time among all members of its family on that case. Failures are excluded.}",
        r"\label{tab:combo_wins}", r"\small",
        r"\begin{tabular}{llr}", r"\toprule",
        r"Family & Combination & Case wins \\", r"\midrule",
    ]
    present = [f for f in ORACLE_COMBO_FAMILIES + ["dc_seed"] if f in set(df["family"])]
    for idx, fam in enumerate(present):
        fam_sub = df[df["family"] == fam].sort_values(["case_wins", "blocks"], ascending=[False, True])
        for _, row in fam_sub.iterrows():
            lines.append(f"{latex_escape(row['family_label'])} & {blocks_to_math(row['blocks'])} & {fmt_int(row['case_wins'])} \\\\")
        if idx < len(present) - 1 and not fam_sub.empty:
            lines.append(r"\midrule")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    tex_path = out_dir / f"{TABLE_PREFIXES['combo_wins']}.tex"
    write_text(tex_path, "\n".join(lines))
    return [str(csv_path), str(tex_path)]


def build_dual_decomposition_summary_table(runs, out_dir):
    """Summary table comparing dual modes across all oracle families including dual-only."""
    families = [f for f in ORACLE_COMBO_FAMILIES if f in set(runs["family"])]
    if len(families) < 3:
        return []

    tbl_rows = []
    for fam in families:
        sub = runs[(runs["family"] == fam) & (runs["success"])].copy()
        n_total = len(runs[runs["family"] == fam])
        dual_mode = runs[runs["family"] == fam]["dual_mode"].iloc[0] if "dual_mode" in runs.columns and len(runs[runs["family"] == fam]) else ""
        tbl_rows.append({
            "Family": FAMILY_LABEL.get(fam, fam), "Short": family_short_label(fam),
            "Dual mode": dual_mode, "Total runs": n_total, "Converged": len(sub),
            "Conv. rate [%]": 100.0 * len(sub) / n_total if n_total else np.nan,
            "Median solve [s]": sub["solve_time"].median() if len(sub) else np.nan,
            "Median speedup [%]": sub["speedup_vs_baseline_pct"].median() if len(sub) else np.nan,
        })

    for fam in DUAL_ONLY_FAMILIES:
        if fam not in set(runs["family"]):
            continue
        sub = runs[(runs["family"] == fam) & (runs["success"])].copy()
        n_total = len(runs[runs["family"] == fam])
        dual_mode = runs[runs["family"] == fam]["dual_mode"].iloc[0] if "dual_mode" in runs.columns and len(runs[runs["family"] == fam]) else ""
        tbl_rows.append({
            "Family": FAMILY_LABEL.get(fam, fam), "Short": family_short_label(fam),
            "Dual mode": dual_mode, "Total runs": n_total, "Converged": len(sub),
            "Conv. rate [%]": 100.0 * len(sub) / n_total if n_total else np.nan,
            "Median solve [s]": sub["solve_time"].median() if len(sub) else np.nan,
            "Median speedup [%]": sub["speedup_vs_baseline_pct"].median() if len(sub) else np.nan,
        })

    df = pd.DataFrame(tbl_rows)
    csv_path = out_dir / f"{TABLE_PREFIXES['dual_decomposition_summary']}.csv"
    df.to_csv(csv_path, index=False)

    lines = [
        r"\begin{table*}[!t]", r"\centering",
        r"\caption{Dual decomposition summary: convergence and performance by dual-mode family. ``Conv.\ rate'' is the fraction of attempted runs that converge. The primal-only family (no duals) and full primal+dual family (block-matched bounds) are included as reference. Constraint-dual-only and bounds-dual-only families isolate the stationarity (Lemma~\ref{lem:residual}) and complementarity (Corollary~\ref{cor:compl}) channels respectively. Dual-only rows supply no primal blocks.}",
        r"\label{tab:dual_decomposition_summary}", r"\small", r"\setlength{\tabcolsep}{5pt}",
        r"\begin{adjustbox}{width=\textwidth}",
        r"\begin{tabular}{llrrrrr}", r"\toprule",
        r"Family & Dual mode & Total runs & Converged & Conv.\ rate [\%] & Median solve [s] & Median speedup [\%] \\",
        r"\midrule",
    ]
    for i, (_, row) in enumerate(df.iterrows()):
        if i > 0 and "dual-only" in str(row["Family"]).lower() and "dual-only" not in str(df.iloc[i - 1]["Family"]).lower():
            lines.append(r"\midrule")
        lines.append(
            f"{latex_escape(row['Family'])} & \\texttt{{{latex_escape(row['Dual mode'])}}} & {fmt_int(row['Total runs'])} & "
            f"{fmt_int(row['Converged'])} & {fmt_pct_number(row['Conv. rate [%]'])} & "
            f"{fmt_float(row['Median solve [s]'])} & {fmt_pct_number(row['Median speedup [%]'])} \\\
"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}", r"\end{table*}"])
    tex_path = out_dir / f"{TABLE_PREFIXES['dual_decomposition_summary']}.tex"
    write_text(tex_path, "\n".join(lines))
    return [str(csv_path), str(tex_path)]


def build_full_warmstarts_matrix(runs, out_dir, matrix_cell_mode="time_speedup_iter"):
    valid_modes = {"time_speedup", "time_iter", "iter_only", "time_speedup_iter"}
    if matrix_cell_mode not in valid_modes:
        raise ValueError(f"Unknown matrix_cell_mode={matrix_cell_mode!r}; expected one of {sorted(valid_modes)}")

    df = runs.copy()
    df = df[
        (df["family"].str.startswith("oracle_ac")) | (df["family"] == "baseline") |
        ((df["family"] == "dc_seed") & (df["seed_vm_policy"] == "") & (df["seed_qg_policy"] == ""))
    ].copy()

    df["case_short"] = df.apply(lambda r: case_short_name(r["case_name"], r.get("nb")), axis=1)
    case_meta = (
        df[["case_short", "case_name", "nb"]].dropna(subset=["case_short"]).drop_duplicates()
        .sort_values(["nb", "case_name"]).groupby("case_short", as_index=False).first()
    )
    case_sizes = case_meta.set_index("case_short")["nb"].to_dict()
    case_name_lookup = case_meta.set_index("case_short")["case_name"].to_dict()

    valid_modes = {"time_speedup", "time_iter", "iter_only", "time_speedup_iter"}

    def per_cell(row):
        if not bool(row.get("success", False)):
            return r"\textbf{F}"

        t = row.get("solve_time", np.nan)
        s = row.get("speedup_vs_baseline_pct", np.nan)
        it = row.get("iterations", np.nan)

        if matrix_cell_mode == "iter_only":
            return fmt_int(it)

        if matrix_cell_mode == "time_iter":
            return f"{fmt_float(t)}/{fmt_int(it)}"

        if matrix_cell_mode == "time_speedup_iter":
            return f"{fmt_float(t)}/{fmt_pct_number(s, 0)}/{fmt_int(it)}"

        if row["family"] == "dc_seed":
            return fmt_float(t)
        return f"{fmt_float(t)}/{fmt_pct_number(s, 0)}"

    df["cell"] = df.apply(per_cell, axis=1)
    df["FamilyShort"] = df["family"].map(family_short_label)
    df["blocks_display"] = df["blocks"].apply(canonicalize_blocks)
    df["Combination"] = df["blocks_display"].apply(lambda b: blocks_to_math(b) if b else "---")

    group_cols = ["family", "FamilyShort", "Combination", "blocks", "blocks_display"]
    summary = df.groupby(group_cols, dropna=False).agg(Cases=("case_name", "nunique"), OK=("success", "sum")).reset_index()
    success_df = df[df["success"]].copy()
    summary_ok = success_df.groupby(group_cols, dropna=False).agg(
        MedSolve=("solve_time", "median"),
        MedIter=("iterations", "median"),
        MedE2E=("end_to_end_time", "median"),
        MedSpd=("speedup_vs_baseline_pct", "median"),
    ).reset_index()
    wide = df.pivot_table(index=group_cols, columns="case_short", values="cell", aggfunc="first", observed=False).reset_index()
    out = summary.merge(summary_ok, on=group_cols, how="left").merge(wide, on=group_cols, how="left")

    fam_order = {f: i for i, f in enumerate(FAMILY_ORDER)}
    out["family_order"] = out["family"].map(fam_order).fillna(999)
    out["combo_order"] = out["blocks_display"].apply(combo_sort_key)
    out = out.sort_values(["family_order", "combo_order"]).reset_index(drop=True)

    used = set(group_cols + ["Cases", "OK", "MedSolve", "MedIter", "MedE2E", "MedSpd", "family_order", "combo_order"])
    case_cols = sorted([c for c in out.columns if c not in used], key=lambda c: (case_sizes.get(c, np.inf), str(c)))

    suffix = "" if matrix_cell_mode == "time_speedup" else f"_{matrix_cell_mode}"
    csv_path = out_dir / f"{TABLE_PREFIXES['all_warmstarts_matrix']}{suffix}.csv"
    out.to_csv(csv_path, index=False)

    case_desc = "; ".join(f"\\textbf{{{latex_escape(c)}}} = \\texttt{{{latex_escape(case_name_lookup.get(c, c))}}}" for c in case_cols)

    group_headers = {
        "baseline": r"\textit{Baseline case-data start}",
        "oracle_ac_primal_only": r"\textit{Oracle AC primal-only (O-PO): no duals}",
        "oracle_ac_primal_dual": r"\textit{Oracle AC primal+dual (O-PD): constraint duals + block-matched bound mults}",
        "oracle_ac_constraint_dual": r"\textit{Oracle AC constraint-dual (O-CD): constraint duals only, no bound mults}",
        "oracle_ac_bounds_dual": r"\textit{Oracle AC bounds-dual (O-BD): block-matched bound mults only, no constraint duals}",
        "oracle_ac_primal_dual_all_bounds": r"\textit{Oracle AC primal+dual all-bounds (O-PD-AB): constraint duals + ALL bound mults}",
        "oracle_ac_dual_only_constraint": r"\textit{Dual-only: constraint duals, no primal blocks}",
        "oracle_ac_dual_only_bounds": r"\textit{Dual-only: all bound mults, no primal blocks}",
        "oracle_ac_dual_only_full": r"\textit{Dual-only: constraint duals + all bound mults, no primal blocks}",
        "dc_seed": r"\textit{Practical DC-seeded (DC)}",
    }

    if matrix_cell_mode == "time_speedup":
        caption_mode = (
            r"For oracle rows, each cell reports AC solve time [s]\,/\,speedup [\%]. "
            r"DC rows report AC solve time [s]. \textbf{F} = solver failure. "
        )
    elif matrix_cell_mode == "time_iter":
        caption_mode = (
            r"Each successful cell reports AC solve time [s]\,/\,IPOPT iterations. "
            r"\textbf{F} = solver failure. "
        )
    elif matrix_cell_mode == "time_speedup_iter":
        caption_mode = (
            r"Each successful oracle cell reports AC solve time [s]\,/\,speedup [\%]\,/\,IPOPT iterations. "
            r"DC rows follow the same convention. \textbf{F} = solver failure. "
        )
    else:
        caption_mode = (
            r"Each successful cell reports IPOPT iterations only. "
            r"\textbf{F} = solver failure. "
        )

    total_cols = 8 + len(case_cols)

    lines = [
        r"\begingroup",
        r"\newgeometry{left=0.45cm,right=0.45cm,top=0.8cm,bottom=0.8cm}",
        r"\setlength{\tabcolsep}{1.2pt}",
        r"\renewcommand{\arraystretch}{0.82}",
        r"\fontsize{5.0}{5.4}\selectfont",
        r"\begin{landscape}",
        r"\begin{table}[p]", r"\centering",
        r"\caption{Complete warm-start results matrix. " + caption_mode + r"Column headers: " + case_desc + r".}",
        r"\label{tab:all_warmstarts_matrix}",
        r"\begin{adjustbox}{max width=\linewidth,max totalheight=0.90\textheight,keepaspectratio}",
        r"\begin{tabular}{p{1.55cm}p{2.00cm}rrrrrr*{" + str(len(case_cols)) + r"}{c}}",
        r"\toprule",
        "Family & Combination & Cases & OK & MedSolve[s] & MedIter & MedE2E[s] & MedSpd[\\%] & " + " & ".join(case_cols) + r" \\",
        r"\midrule",
    ]

    present_families = []
    for f in FAMILY_ORDER:
        if f in set(out["family"]):
            present_families.append(f)

    for idx, fam in enumerate(present_families):
        fam_rows = out[out["family"] == fam]
        if fam_rows.empty:
            continue
        lines.append(r"\multicolumn{" + str(total_cols) + r"}{l}{" + group_headers.get(fam, rf"\textit{{{latex_escape(fam)}}}") + r"} \\")
        for _, row in fam_rows.iterrows():
            cells = [_render_cell(row.get(c, "")) for c in case_cols]
            lines.append(
                f"{latex_escape(row['FamilyShort'])} & {row['Combination']} & {fmt_int(row['Cases'])} & {fmt_int(row['OK'])} & "
                f"{fmt_float(row['MedSolve'])} & {fmt_int(row['MedIter'])} & {fmt_float(row['MedE2E'])} & {fmt_pct_number(row['MedSpd'])} & "
                + " & ".join(cells) + r" \\"
            )
        if idx < len(present_families) - 1:
            lines.append(r"\midrule")

    lines.extend([
        r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}",
        r"\end{table}", r"\end{landscape}", r"\restoregeometry", r"\endgroup"])
    tex_path = out_dir / f"{TABLE_PREFIXES['all_warmstarts_matrix']}{suffix}.tex"
    write_text(tex_path, "\n".join(lines))
    return [str(csv_path), str(tex_path)]


# =============================================================================
# Main driver
# =============================================================================
def generate_tables(runs, out_dir, matrix_cell_mode="time_speedup_iter"):
    combo_summary = derive_combo_summary(runs)
    casewise_best = build_casewise_best_table(runs)
    combo_wins = compute_combo_wins(runs)
    pairwise = pairwise_wilcoxon_table(casewise_best)

    outputs = {}
    outputs["global_summary"] = build_global_summary_table(runs, combo_summary, casewise_best, out_dir)
    outputs["pairwise_tests"] = build_pairwise_table(pairwise, out_dir)
    outputs["combo_rank_oracle"] = build_oracle_combo_rank_table(combo_summary, combo_wins, out_dir)
    outputs["combo_rank_dc"] = build_dc_rank_table(combo_summary, combo_wins, out_dir)
    outputs["block_effects"] = build_block_effects_table(runs, out_dir)
    outputs["largest_cases"] = build_largest_cases_table(casewise_best, out_dir)
    outputs["combo_wins"] = build_combo_wins_table(combo_wins, out_dir)
    outputs["family_wins"] = build_family_wins_table(casewise_best, out_dir)
    outputs["all_warmstarts_matrix"] = build_full_warmstarts_matrix(runs, out_dir, matrix_cell_mode=matrix_cell_mode)

    if has_extended_dual_data(runs):
        outputs["dual_decomposition_summary"] = build_dual_decomposition_summary_table(runs, out_dir)

    return outputs


def write_report(out_dir, figures, tables):
    path = out_dir / "paper_outputs_README.md"
    lines = ["# Combined paper outputs", "",
             "Generated from `all_runs.csv` with extended dual decomposition support.", "", "## Figures"]
    for k, paths in figures.items():
        lines.append(f"- **{k}**")
        for p in (paths or []):
            lines.append(f"  - `{Path(p).name}`")
        if not paths:
            lines.append("  - not generated")

    lines += ["", "## Tables"]
    for k, paths in tables.items():
        lines.append(f"- **{k}**")
        for p in (paths or []):
            lines.append(f"  - `{Path(p).name}`")
        if not paths:
            lines.append("  - not generated")

    path.write_text("\n".join(lines), encoding="utf-8")
    return str(path)


def generate_all_outputs(study_dir, out_dir, matrix_cell_mode="time_speedup"):
    setup_publication_style()
    fig_dir = ensure_dir(out_dir / "figures")
    table_dir = ensure_dir(out_dir / "tables")

    data = normalize_data(load_study(study_dir))
    runs = data["all_runs"]

    combo_summary = derive_combo_summary(runs)
    casewise_best = build_casewise_best_table(runs)

    figures = {}
    figures["case_size_runtime"] = fig_case_size_vs_runtime(runs, casewise_best, fig_dir)
    figures["performance_profile"] = fig_performance_profile(runs, combo_summary, fig_dir)
    figures["casewise_speedups"] = fig_casewise_speedups(casewise_best, fig_dir)
    figures["oracle_combo_heatmap_speedup"] = fig_oracle_combo_heatmap(runs, fig_dir)
    figures["block_effects"] = fig_block_effects(runs, fig_dir)
    figures["speedup_vs_case_size"] = fig_speedup_vs_size(casewise_best, fig_dir)
    figures["speedup_distributions"] = fig_speedup_boxplot(casewise_best, fig_dir)
    figures["baseline_vs_best"] = fig_baseline_vs_best(casewise_best, fig_dir)
    figures["dc_native_boxplot"] = fig_dc_native_boxplot(runs, fig_dir)

    if has_extended_dual_data(runs):
        figures["dual_decomposition_heatmap"] = fig_dual_decomposition_heatmap(runs, fig_dir)
        figures["dual_only_bar"] = fig_dual_only_bar(runs, fig_dir)

    tables = generate_tables(runs, table_dir, matrix_cell_mode=matrix_cell_mode)
    report = write_report(out_dir, figures, tables)

    manifest = {
        "study_dir": str(study_dir),
        "out_dir": str(out_dir),
        "matrix_cell_mode": matrix_cell_mode,
        "has_extended_dual": has_extended_dual_data(runs),
        "figures": figures,
        "tables": tables,
        "report": report,
    }
    (out_dir / "paper_outputs_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def parse_args():
    p = argparse.ArgumentParser(description="Generate paper figures and tables from ACOPF study outputs.")
    p.add_argument("study_dir", nargs="?", default="acopf_acdc_paper_study")
    p.add_argument("out_dir", nargs="?", default=None)
    p.add_argument(
        "--matrix-cell-mode",
        choices=["time_speedup", "time_iter", "iter_only", "time_speedup_iter"],
        default="time_speedup_iter",
        help="How to render the per-case cells in the full warm-start matrix.",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    study_dir = Path(args.study_dir).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else (study_dir / "paper_outputs_combined")
    ensure_dir(out_dir)
    manifest = generate_all_outputs(
        study_dir,
        out_dir,
        matrix_cell_mode=args.matrix_cell_mode,
    )
    print(json.dumps(manifest, indent=2))

# ACOPF warm-start benchmark
# Author: Babak Taheri
import sys
import os
import argparse
import time
import math
import csv
import re
import json
import gzip
import hashlib
import traceback
import itertools
from statistics import mean, median
from datetime import datetime, timezone

import numpy as np
import scipy.io as sio
import scipy.sparse as sp
import pyomo.environ as pe
# =============================================================================
# CONSTANTS & INDICES
# =============================================================================
BUS_I, BUS_TYPE, PD, QD, GS, BS, BUS_AREA, VM, VA, BASE_KV, ZONE, VMAX, VMIN = range(13)
GEN_BUS, PG, QG, QMAX, QMIN, VG, MBASE, GEN_STATUS, PMAX, PMIN = range(10)
F_BUS, T_BUS, BR_R, BR_X, BR_B, RATE_A, RATE_B, RATE_C, TAP, SHIFT, BR_STATUS = range(11)
MODEL, STARTUP, SHUTDOWN, NCOST, COST = range(5)

DEFAULT_CASE_DIRS = ["test_cases", "testt_cases"]
DEFAULT_OUTPUT_DIR = "acopf_acdc_paper_study"
BLOCKS = ("Pg", "Qg", "Vm", "Va")
DC_NATIVE_BLOCKS = ("Pg", "Va")
DC_NATIVE_ONLY_COMBOS = (("Pg",), ("Va",), ("Pg", "Va"))
DC_VM_POLICIES = ("flat1", "case_vm", "gen_vg_flatload")
DC_QG_POLICIES = ("case_qg", "mid_q")
DEFAULT_INCLUDE_DC_COMPLETION_SENSITIVITY = False
DEFAULT_INCLUDE_EXTENDED_DUAL_DECOMPOSITION = True
DEFAULT_OBJECTIVE_MATCH_TOL = 1e-5

SUCCESS_TERMS = {pe.TerminationCondition.optimal, pe.TerminationCondition.locallyOptimal}
if hasattr(pe.TerminationCondition, "globallyOptimal"):
    SUCCESS_TERMS.add(pe.TerminationCondition.globallyOptimal)

DUAL_MODES = ("none", "full", "constraint_only", "bounds_only", "all_bounds_only", "full_all_bounds")

EXTENDED_DUAL_FAMILIES = [
    ("oracle_ac_constraint_dual", "constraint_only"),
    ("oracle_ac_bounds_dual", "bounds_only"),
    ("oracle_ac_primal_dual_all_bounds", "full_all_bounds"),
]

DUAL_ONLY_VARIANTS = [
    ("oracle_ac_dual_only_constraint", "constraint_only"),
    ("oracle_ac_dual_only_bounds", "all_bounds_only"),
    ("oracle_ac_dual_only_full", "full_all_bounds"),
]

# All families that require IPOPT warm-start parameter settings
WARM_START_FAMILIES = {
    "oracle_ac_primal_dual",
    "oracle_ac_constraint_dual",
    "oracle_ac_bounds_dual",
    "oracle_ac_primal_dual_all_bounds",
    "oracle_ac_dual_only_constraint",
    "oracle_ac_dual_only_bounds",
    "oracle_ac_dual_only_full",
}

# =============================================================================
# SMALL UTILITIES
# =============================================================================

def now_utc_iso():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def sha256_of_file(path, chunk_size=1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def json_gz_dump(obj, path):
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    return path

def row_is_successful(row):
    s = row.get("success", None)
    if isinstance(s, str):
        s = s.strip().lower()
        if s in {"1", "true", "yes"}:
            return True
        if s in {"0", "false", "no", ""}:
            return False
    if s in (1, True):
        return True
    if s in (0, False):
        return False
    return row.get("objective") is not None

def save_csv(rows, path, fieldnames=None):
    rows = list(rows)

    if fieldnames is not None:
        all_fields = list(fieldnames)
        seen = set(all_fields)
        for row in rows:
            for k in row.keys():
                if k not in seen:
                    seen.add(k)
                    all_fields.append(k)
    elif rows:
        all_fields = []
        seen = set()
        for row in rows:
            for k in row.keys():
                if k not in seen:
                    seen.add(k)
                    all_fields.append(k)
    else:
        all_fields = []

    with open(path, "w", newline="", encoding="utf-8") as f:
        if not all_fields:
            return path
        writer = csv.DictWriter(f, fieldnames=all_fields)
        writer.writeheader()
        for row in rows:
            clean = {}
            for k in all_fields:
                v = row.get(k)
                clean[k] = "+".join(v) if isinstance(v, tuple) else v
            writer.writerow(clean)
    return path


def clamp_to_bounds(var, val):
    lb = var.lb if var.lb is not None else -float("inf")
    ub = var.ub if var.ub is not None else float("inf")
    return min(max(float(val), lb), ub)


def safe_value(expr, default=None):
    try:
        return pe.value(expr)
    except Exception:
        return default


def suffix_get(sfx, comp, default=0.0):
    try:
        return float(pe.value(sfx[comp]))
    except Exception:
        return float(default)


def powerset_nonempty(items):
    for r in range(1, len(items) + 1):
        for combo in itertools.combinations(items, r):
            yield combo


def dc_meaningful_combos():
    return [c for c in powerset_nonempty(BLOCKS) if any(b in DC_NATIVE_BLOCKS for b in c)]


def dc_completion_combos():
    return [c for c in dc_meaningful_combos() if ("Vm" in c or "Qg" in c)]


def combo_name(blocks):
    return "_".join(blocks) if blocks else "none"


def make_run_name(family, blocks, vm_policy=None, qg_policy=None):
    if family == "baseline":
        return "baseline_default"
    name = f"{family}__{combo_name(blocks)}"
    if vm_policy is not None:
        name += f"__vm_{vm_policy}"
    if qg_policy is not None:
        name += f"__qg_{qg_policy}"
    return name


def success_termination(term):
    return term in SUCCESS_TERMS


def dict_abs_diff_stats(a, b):
    keys = sorted(set(a.keys()) & set(b.keys()))
    if not keys:
        return {"l2": None, "linf": None, "mean_abs": None, "median_abs": None, "max_abs": None}
    diffs = np.array([float(a[k]) - float(b[k]) for k in keys], dtype=float)
    ad = np.abs(diffs)
    return {
        "l2": float(np.linalg.norm(diffs)),
        "linf": float(np.max(ad)),
        "mean_abs": float(np.mean(ad)),
        "median_abs": float(np.median(ad)),
        "max_abs": float(np.max(ad)),
    }


def speedup_pct(baseline_time, new_time):
    if baseline_time is None or new_time is None or baseline_time <= 0:
        return None
    return 100.0 * (baseline_time - new_time) / baseline_time


def objective_matched_rows(rows):
    matched = [r for r in rows if r.get("objective_matches_baseline") == 1]
    return matched if matched else rows


def objective_match_counts(rows):
    comparable = [r for r in rows if r.get("objective_matches_baseline") in (0, 1)]
    n_matching = sum(r.get("objective_matches_baseline") == 1 for r in comparable)
    share = (n_matching / len(comparable)) if comparable else None
    return n_matching, share

def parse_ipopt_log(log_path):
    out = {
        "iterations": None,
        "objective": None,
        "dual_inf": None,
        "constr_viol": None,
        "var_bound_viol": None,
        "compl_inf": None,
        "overall_nlp_error": None,
        "ipopt_reported_total_sec": None,
        "ipopt_cpu_sec_no_func_eval": None,
        "ipopt_cpu_sec_func_eval": None,
        "ipopt_cpu_total_est": None,
        "restoration_used": False,
    }
    if not os.path.exists(log_path):
        return out
    try:
        txt = open(log_path, "r", encoding="utf-8", errors="ignore").read()
    except Exception:
        return out

    patterns = {
        "iterations": [r"Number of Iterations\.*:\s*(\d+)"],
        "objective": [r"Objective\.*:\s*([-+0-9.eEdD]+)", r"objective\s*=\s*([-+0-9.eEdD]+)"],
        "dual_inf": [r"Dual infeasibility\.*:\s*([-+0-9.eEdD]+)"],
        "constr_viol": [r"Constraint violation\.*:\s*([-+0-9.eEdD]+)"],
        "var_bound_viol": [r"Variable bound violation\.*:\s*([-+0-9.eEdD]+)"],
        "compl_inf": [r"Complementarity\.*:\s*([-+0-9.eEdD]+)"],
        "overall_nlp_error": [r"Overall NLP error\.*:\s*([-+0-9.eEdD]+)"],
        "ipopt_reported_total_sec": [r"Total seconds in IPOPT\s*=\s*([-+0-9.eEdD]+)"],
        "ipopt_cpu_sec_no_func_eval": [r"Total CPU secs in IPOPT \(w/o function evaluations\)\s*=\s*([-+0-9.eEdD]+)"],
        "ipopt_cpu_sec_func_eval": [r"Total CPU secs in NLP function evaluations\s*=\s*([-+0-9.eEdD]+)"],
    }
    for key, pats in patterns.items():
        for pat in pats:
            m = re.search(pat, txt)
            if m:
                val = m.group(1).replace("D", "E").replace("d", "e")
                out[key] = int(val) if key == "iterations" else float(val)
                break

    if out["ipopt_cpu_sec_no_func_eval"] is not None and out["ipopt_cpu_sec_func_eval"] is not None:
        out["ipopt_cpu_total_est"] = out["ipopt_cpu_sec_no_func_eval"] + out["ipopt_cpu_sec_func_eval"]
    elif out["ipopt_reported_total_sec"] is not None:
        out["ipopt_cpu_total_est"] = out["ipopt_reported_total_sec"]

    out["restoration_used"] = ("restoration phase" in txt.lower()) or (re.search(r"\n\s*\d+r\s", txt) is not None)
    return out


# =============================================================================
# COST MODEL HELPERS
# =============================================================================
def active_power_gencost_rows(gencost, ng_total):
    if gencost.ndim == 1:
        gencost = gencost.reshape(1, -1)
    if gencost.shape[0] < ng_total:
        raise ValueError(
            f"gencost has {gencost.shape[0]} row(s) but gen has {ng_total} row(s); "
            f"cannot map active-power costs to generators."
        )
    return gencost


def polynomial_cost_expr(pg_mw_expr, coeffs):
    degree = len(coeffs) - 1
    expr = 0
    for i, c in enumerate(coeffs):
        expr += float(c) * (pg_mw_expr ** (degree - i))
    return expr


def build_piecewise_segments(points):
    slopes = []
    intercepts = []
    for i in range(len(points) - 1):
        p1, f1 = points[i]
        p2, f2 = points[i + 1]
        if p2 <= p1:
            raise ValueError("Piecewise-linear cost points must have strictly increasing p values.")
        m = (f2 - f1) / (p2 - p1)
        b = f1 - m * p1
        slopes.append(float(m))
        intercepts.append(float(b))
    for i in range(len(slopes) - 1):
        if slopes[i + 1] + 1e-10 < slopes[i]:
            raise ValueError(
                "Nonconvex piecewise-linear generator costs are not supported by this implementation. "
                "Use convex PWL costs or reformulate with binaries/SOS2."
            )
    return slopes, intercepts


def build_generator_cost_specs(baseMVA, gen, gencost, active_gens, ng_total):
    active_cost_rows = active_power_gencost_rows(gencost, ng_total)
    specs = {}
    any_pwl = False
    max_poly_degree = 0
    any_quadratic = False

    for g in active_gens:
        row = active_cost_rows[g]
        model_type = int(row[MODEL])
        ncost = int(row[NCOST])
        if model_type == 2:
            coeffs = [float(x) for x in row[COST:COST + ncost]]
            if len(coeffs) != ncost:
                raise ValueError(f"Generator {g}: malformed polynomial gencost row.")
            degree = max(ncost - 1, 0)
            max_poly_degree = max(max_poly_degree, degree)
            any_quadratic = any_quadratic or degree == 2
            specs[g] = {
                "model": 2,
                "coeffs": coeffs,
                "poly_degree": degree,
            }
        elif model_type == 1:
            vals = [float(x) for x in row[COST:COST + 2 * ncost]]
            if len(vals) != 2 * ncost or ncost < 2:
                raise ValueError(f"Generator {g}: malformed piecewise-linear gencost row.")
            points = [(vals[2 * i], vals[2 * i + 1]) for i in range(ncost)]
            pmin_mw = float(gen[g, PMIN])
            pmax_mw = float(gen[g, PMAX])
            if pmin_mw < points[0][0] - 1e-8 or pmax_mw > points[-1][0] + 1e-8:
                raise ValueError(
                    f"Generator {g}: PMIN/PMAX [{pmin_mw}, {pmax_mw}] lies outside "
                    f"piecewise-linear cost domain [{points[0][0]}, {points[-1][0]}]."
                )
            slopes, intercepts = build_piecewise_segments(points)
            specs[g] = {
                "model": 1,
                "points": points,
                "slopes": slopes,
                "intercepts": intercepts,
                "poly_degree": 1,
            }
            any_pwl = True
        else:
            raise ValueError(f"Generator {g}: unsupported MATPOWER gencost MODEL={model_type}.")

    return {
        "active_rows": active_cost_rows,
        "specs": specs,
        "any_pwl": any_pwl,
        "max_poly_degree": max_poly_degree,
        "any_quadratic": any_quadratic,
    }


def add_generator_cost_structure(model, ctx, pg_accessor):
    specs = ctx["cost_specs"]["specs"]
    pwl_gens = [g for g in model.G if specs[g]["model"] == 1]
    model.PWLCostG = pe.Set(initialize=pwl_gens, ordered=True)
    pwl_seg_index = []
    for g in pwl_gens:
        for s in range(len(specs[g]["slopes"])):
            pwl_seg_index.append((int(g), int(s)))
    model.PWLSeg = pe.Set(dimen=2, initialize=pwl_seg_index, ordered=True)
    model.GenCost = pe.Var(model.PWLCostG, initialize=0.0)

    def pwl_rule(m, g, s):
        spec = specs[g]
        pg_mw = pg_accessor(m, g)
        return m.GenCost[g] >= spec["slopes"][s] * pg_mw + spec["intercepts"][s]

    model.PWLCostConstr = pe.Constraint(model.PWLSeg, rule=pwl_rule)

    def total_cost_rule(m):
        total = 0.0
        for g in m.G:
            spec = specs[g]
            pg_mw = pg_accessor(m, g)
            if spec["model"] == 2:
                total += polynomial_cost_expr(pg_mw, spec["coeffs"])
            elif spec["model"] == 1:
                total += m.GenCost[g]
            else:
                raise ValueError(f"Unsupported cost model for generator {g}.")
        return total

    model.Obj = pe.Objective(rule=total_cost_rule, sense=pe.minimize)


# =============================================================================
# CASE DISCOVERY
# =============================================================================
def discover_case_files(case_root=None):
    roots = [case_root] if case_root is not None else DEFAULT_CASE_DIRS
    out = []
    for root in roots:
        if root is None or not os.path.isdir(root):
            continue
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                if fn.lower().endswith(".mat"):
                    out.append(os.path.join(dirpath, fn))
    return sorted(set(out))


# =============================================================================
# LOAD PHYSICS
# =============================================================================
def load_and_build_physics(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    mat = sio.loadmat(file_path, squeeze_me=True, struct_as_record=False)
    mpc = mat.get("mpc") or mat.get("ppc")
    if mpc is None:
        for key in mat:
            if hasattr(mat[key], "bus") and hasattr(mat[key], "gen"):
                mpc = mat[key]
                break
    if mpc is None:
        raise ValueError("Invalid MATPOWER structure.")

    baseMVA = float(mpc.baseMVA)
    bus = np.array(mpc.bus, dtype=float)
    gen = np.array(mpc.gen, dtype=float)
    branch = np.array(mpc.branch, dtype=float)
    gencost = np.array(mpc.gencost, dtype=float)

    nb = len(bus)
    nl = len(branch)
    bus_map = {int(b_id): i for i, b_id in enumerate(bus[:, BUS_I])}

    f = np.array([bus_map[int(x)] for x in branch[:, F_BUS]])
    t = np.array([bus_map[int(x)] for x in branch[:, T_BUS]])

    r = branch[:, BR_R]
    x = branch[:, BR_X]
    b_ch = branch[:, BR_B]
    tap = branch[:, TAP].copy()
    tap[tap == 0] = 1.0
    shift = np.radians(branch[:, SHIFT])
    stat = branch[:, BR_STATUS]

    Zs = r + 1j * x
    Ys = 1 / Zs
    tap_ratio = tap * np.exp(1j * shift)

    Yff = (Ys + 1j * b_ch / 2) / (np.abs(tap_ratio) ** 2) * stat
    Yft = -Ys / np.conj(tap_ratio) * stat
    Ytf = -Ys / tap_ratio * stat
    Ytt = (Ys + 1j * b_ch / 2) * stat

    data_f = np.concatenate([Yff, Yft])
    row_f = np.concatenate([np.arange(nl), np.arange(nl)])
    col_f = np.concatenate([f, t])
    Yf = sp.coo_matrix((data_f, (row_f, col_f)), shape=(nl, nb)).tocsr()

    data_t = np.concatenate([Ytf, Ytt])
    Yt = sp.coo_matrix((data_t, (row_f, col_f)), shape=(nl, nb)).tocsr()

    Ybus_branch = (
        sp.coo_matrix((Yff, (f, f)), shape=(nb, nb))
        + sp.coo_matrix((Yft, (f, t)), shape=(nb, nb))
        + sp.coo_matrix((Ytf, (t, f)), shape=(nb, nb))
        + sp.coo_matrix((Ytt, (t, t)), shape=(nb, nb))
    )
    Ysh = (bus[:, GS] + 1j * bus[:, BS]) / baseMVA
    Ybus = Ybus_branch + sp.diags(Ysh, 0, shape=(nb, nb))

    return baseMVA, bus, gen, branch, gencost, bus_map, Ybus.tocsr(), Yf, Yt


def fast_sparse_rows_complex(matrix, num_rows):
    rows = [[] for _ in range(num_rows)]
    coo = matrix.tocoo()
    for r, c, d in zip(coo.row, coo.col, coo.data):
        rows[r].append((int(c), float(d.real), float(d.imag)))
    return rows


def fast_sparse_rows_real(matrix, num_rows):
    rows = [[] for _ in range(num_rows)]
    coo = matrix.tocoo()
    for r, c, d in zip(coo.row, coo.col, coo.data):
        rows[r].append((int(c), float(d)))
    return rows


def build_dc_matrices(branch, nb, nl, f_idx, t_idx):
    stat = branch[:, BR_STATUS].astype(float)
    x = branch[:, BR_X].astype(float).copy()
    x[np.abs(x) < 1e-12] = 1e-12
    tap = branch[:, TAP].astype(float).copy()
    tap[tap == 0.0] = 1.0
    shift = np.radians(branch[:, SHIFT].astype(float))

    b = stat / (x * tap)
    A = sp.coo_matrix(
        (np.r_[np.ones(nl), -np.ones(nl)], (np.r_[np.arange(nl), np.arange(nl)], np.r_[f_idx, t_idx])),
        shape=(nl, nb),
    ).tocsr()

    Bf = sp.diags(b) @ A
    Bbus = A.T @ Bf
    Pfinj = -b * shift
    Pbusinj = A.T @ Pfinj
    return Bbus.tocsr(), Bf.tocsr(), np.asarray(Pbusinj).ravel(), np.asarray(Pfinj).ravel()


def prepare_case_context(case_file):
    baseMVA, bus, gen, branch, gencost, bus_map, Ybus, Yf, Yt = load_and_build_physics(case_file)
    nb = len(bus)
    nl = len(branch)
    ng_total = len(gen)
    active_gens = [i for i in range(ng_total) if gen[i, GEN_STATUS] > 0]

    ref_candidates = np.where(bus[:, BUS_TYPE] == 3)[0]
    if len(ref_candidates) != 1:
        raise ValueError(
            f"Expected exactly one reference bus (BUS_TYPE=3), found {len(ref_candidates)} in case {case_file}."
        )
    ref_idx = int(ref_candidates[0])

    Ybus_rows = fast_sparse_rows_complex(Ybus, nb)
    Yf_rows = fast_sparse_rows_complex(Yf, nl)
    Yt_rows = fast_sparse_rows_complex(Yt, nl)

    bus_gens = [[] for _ in range(nb)]
    for g in active_gens:
        bus_gens[bus_map[int(gen[g, GEN_BUS])]].append(g)

    F_indices = [bus_map[int(x)] for x in branch[:, F_BUS]]
    T_indices = [bus_map[int(x)] for x in branch[:, T_BUS]]

    Bbus, Bf, Pbusinj, Pfinj = build_dc_matrices(branch, nb, nl, F_indices, T_indices)
    Bbus_rows = fast_sparse_rows_real(Bbus, nb)
    Bf_rows = fast_sparse_rows_real(Bf, nl)

    cost_specs = build_generator_cost_specs(baseMVA, gen, gencost, active_gens, ng_total)

    return {
        "case_file": case_file,
        "baseMVA": baseMVA,
        "bus": bus,
        "gen": gen,
        "branch": branch,
        "gencost": gencost,
        "active_power_gencost": cost_specs["active_rows"],
        "cost_specs": cost_specs,
        "bus_map": bus_map,
        "nb": nb,
        "nl": nl,
        "ng_total": ng_total,
        "ng_active": len(active_gens),
        "active_gens": active_gens,
        "bus_gens": bus_gens,
        "Ybus_rows": Ybus_rows,
        "Yf_rows": Yf_rows,
        "Yt_rows": Yt_rows,
        "Pd": bus[:, PD] / baseMVA,
        "Qd": bus[:, QD] / baseMVA,
        "F_indices": F_indices,
        "T_indices": T_indices,
        "RateA": branch[:, RATE_A] / baseMVA,
        "RateA_Sq": (branch[:, RATE_A] / baseMVA) ** 2,
        "ref_idx": ref_idx,
        "Bbus_rows": Bbus_rows,
        "Bf_rows": Bf_rows,
        "Pbusinj": Pbusinj,
        "Pfinj": Pfinj,
        "case_stats": {
            "baseMVA": baseMVA,
            "nb": nb,
            "nl": nl,
            "ng_total": ng_total,
            "ng_active": len(active_gens),
            "n_load_buses": int(np.sum((bus[:, PD] != 0.0) | (bus[:, QD] != 0.0))),
            "n_ref_buses": int(np.sum(bus[:, BUS_TYPE] == 3)),
            "n_pv_buses": int(np.sum(bus[:, BUS_TYPE] == 2)),
            "n_pq_buses": int(np.sum(bus[:, BUS_TYPE] == 1)),
            "total_pd_mw": float(np.sum(bus[:, PD])),
            "total_qd_mvar": float(np.sum(bus[:, QD])),
            "sum_pmax_mw": float(np.sum(gen[active_gens, PMAX])) if active_gens else 0.0,
            "sum_pmin_mw": float(np.sum(gen[active_gens, PMIN])) if active_gens else 0.0,
            "sum_qmax_mvar": float(np.sum(gen[active_gens, QMAX])) if active_gens else 0.0,
            "sum_qmin_mvar": float(np.sum(gen[active_gens, QMIN])) if active_gens else 0.0,
            "n_branch_with_limits": int(np.sum(branch[:, RATE_A] > 0.0)),
            "n_branch_out_of_service": int(np.sum(branch[:, BR_STATUS] == 0)),
            "gencost_any_pwl": int(cost_specs["any_pwl"]),
            "gencost_max_poly_degree": int(cost_specs["max_poly_degree"]),
            "gencost_any_quadratic": int(cost_specs["any_quadratic"]),
        },
    }


# =============================================================================
# WARM-START PACKAGE HELPERS
# =============================================================================
def extract_primal_solution(model):
    return {
        "Vm": {int(b): float(pe.value(model.Vm[b])) for b in model.B},
        "Va": {int(b): float(pe.value(model.Va[b])) for b in model.B},
        "Pg": {int(g): float(pe.value(model.Pg[g])) for g in model.G},
        "Qg": {int(g): float(pe.value(model.Qg[g])) for g in model.G},
    }


def extract_full_warmstart(model):
    return {
        "primal": extract_primal_solution(model),
        "dual": {
            "RefAngle": suffix_get(model.dual, model.RefAngle, 0.0),
            "EqP": {int(b): suffix_get(model.dual, model.EqP[b], 0.0) for b in model.B},
            "EqQ": {int(b): suffix_get(model.dual, model.EqQ[b], 0.0) for b in model.B},
            "FlowFrom": {int(l): suffix_get(model.dual, model.FlowFrom[l], 0.0) for l in model.FlowFrom},
            "FlowTo": {int(l): suffix_get(model.dual, model.FlowTo[l], 0.0) for l in model.FlowTo},
        },
        "zL": {
            "Vm": {int(b): suffix_get(model.ipopt_zL_out, model.Vm[b], 0.0) for b in model.B},
            "Va": {int(b): suffix_get(model.ipopt_zL_out, model.Va[b], 0.0) for b in model.B},
            "Pg": {int(g): suffix_get(model.ipopt_zL_out, model.Pg[g], 0.0) for g in model.G},
            "Qg": {int(g): suffix_get(model.ipopt_zL_out, model.Qg[g], 0.0) for g in model.G},
        },
        "zU": {
            "Vm": {int(b): suffix_get(model.ipopt_zU_out, model.Vm[b], 0.0) for b in model.B},
            "Va": {int(b): suffix_get(model.ipopt_zU_out, model.Va[b], 0.0) for b in model.B},
            "Pg": {int(g): suffix_get(model.ipopt_zU_out, model.Pg[g], 0.0) for g in model.G},
            "Qg": {int(g): suffix_get(model.ipopt_zU_out, model.Qg[g], 0.0) for g in model.G},
        },
    }


def build_warmstart_package(full_ws, blocks, dual_mode="none"):
    if dual_mode not in DUAL_MODES:
        raise ValueError(f"Unknown dual_mode: {dual_mode!r}. Expected one of {DUAL_MODES}")

    pkg = {
        "primal": {blk: full_ws["primal"][blk] for blk in blocks},
        "dual": {},
        "zL": {},
        "zU": {},
    }

    if dual_mode in ("full", "constraint_only", "full_all_bounds"):
        pkg["dual"] = full_ws["dual"]

    if dual_mode == "full":
        pkg["zL"] = {blk: full_ws["zL"][blk] for blk in blocks}
        pkg["zU"] = {blk: full_ws["zU"][blk] for blk in blocks}
    elif dual_mode == "bounds_only":
        pkg["zL"] = {blk: full_ws["zL"][blk] for blk in blocks}
        pkg["zU"] = {blk: full_ws["zU"][blk] for blk in blocks}
    elif dual_mode == "full_all_bounds":
        pkg["zL"] = {blk: full_ws["zL"][blk] for blk in BLOCKS}
        pkg["zU"] = {blk: full_ws["zU"][blk] for blk in BLOCKS}
    elif dual_mode == "all_bounds_only":
        pkg["zL"] = {blk: full_ws["zL"][blk] for blk in BLOCKS}
        pkg["zU"] = {blk: full_ws["zU"][blk] for blk in BLOCKS}
    # "constraint_only" and "none" leave zL/zU empty

    return pkg


def make_vm_policy(ctx, policy):
    bus = ctx["bus"]
    gen = ctx["gen"]
    active_gens = ctx["active_gens"]
    vm = {}
    if policy == "flat1":
        for b in range(ctx["nb"]):
            vm[b] = float(np.clip(1.0, bus[b, VMIN], bus[b, VMAX]))
    elif policy == "case_vm":
        for b in range(ctx["nb"]):
            vm[b] = float(np.clip(bus[b, VM], bus[b, VMIN], bus[b, VMAX]))
    elif policy == "gen_vg_flatload":
        for b in range(ctx["nb"]):
            vm[b] = float(np.clip(1.0, bus[b, VMIN], bus[b, VMAX]))
        for g in active_gens:
            b = ctx["bus_map"][int(gen[g, GEN_BUS])]
            vm[b] = float(np.clip(gen[g, VG], bus[b, VMIN], bus[b, VMAX]))
    else:
        raise ValueError(f"Unknown Vm policy: {policy}")
    return vm


def make_qg_policy(ctx, policy):
    gen = ctx["gen"]
    qg = {}
    for g in ctx["active_gens"]:
        qmin = float(gen[g, QMIN] / ctx["baseMVA"])
        qmax = float(gen[g, QMAX] / ctx["baseMVA"])
        if policy == "case_qg":
            val = float(gen[g, QG] / ctx["baseMVA"])
        elif policy == "mid_q":
            val = 0.5 * (qmin + qmax)
        else:
            raise ValueError(f"Unknown Qg policy: {policy}")
        qg[g] = min(max(val, qmin), qmax)
    return qg


def build_dc_seed_full_primal(ctx, dc_solution, vm_policy=None, qg_policy=None):
    full = {
        "Pg": dc_solution["Pg"],
        "Va": dc_solution["Va"],
    }
    if vm_policy is not None:
        full["Vm"] = make_vm_policy(ctx, vm_policy)
    if qg_policy is not None:
        full["Qg"] = make_qg_policy(ctx, qg_policy)
    return full


def build_dc_init_package(ctx, dc_solution, blocks, vm_policy=None, qg_policy=None):
    full = build_dc_seed_full_primal(ctx, dc_solution, vm_policy=vm_policy, qg_policy=qg_policy)
    missing = [blk for blk in blocks if blk not in full]
    if missing:
        raise ValueError(
            f"Requested DC init blocks {missing} but no DC completion policy was provided for them."
        )
    return {"primal": {blk: full[blk] for blk in blocks}, "dual": {}, "zL": {}, "zU": {}}


def apply_primal_start(model, init_package, ref_idx=None):
    if not init_package:
        return
    primal = init_package.get("primal", {})
    if "Vm" in primal:
        for b, val in primal["Vm"].items():
            if b in model.B:
                model.Vm[b].set_value(clamp_to_bounds(model.Vm[b], val))
    if "Va" in primal:
        for b, val in primal["Va"].items():
            if b in model.B:
                model.Va[b].set_value(clamp_to_bounds(model.Va[b], val))
    if "Pg" in primal:
        for g, val in primal["Pg"].items():
            if g in model.G:
                model.Pg[g].set_value(clamp_to_bounds(model.Pg[g], val))
    if "Qg" in primal:
        for g, val in primal["Qg"].items():
            if g in model.G:
                model.Qg[g].set_value(clamp_to_bounds(model.Qg[g], val))
    if ref_idx is not None and ref_idx in model.B:
        model.Va[ref_idx].set_value(0.0)


def apply_dual_start(model, init_package):
    if not init_package:
        return
    dual = init_package.get("dual", {})
    zL = init_package.get("zL", {})
    zU = init_package.get("zU", {})

    if "RefAngle" in dual:
        model.dual[model.RefAngle] = float(dual["RefAngle"])
    for b, val in dual.get("EqP", {}).items():
        if b in model.B:
            model.dual[model.EqP[b]] = float(val)
    for b, val in dual.get("EqQ", {}).items():
        if b in model.B:
            model.dual[model.EqQ[b]] = float(val)
    for l, val in dual.get("FlowFrom", {}).items():
        if l in model.FlowFrom:
            model.dual[model.FlowFrom[l]] = float(val)
    for l, val in dual.get("FlowTo", {}).items():
        if l in model.FlowTo:
            model.dual[model.FlowTo[l]] = float(val)

    for name, varcomp in [("Vm", model.Vm), ("Va", model.Va), ("Pg", model.Pg), ("Qg", model.Qg)]:
        for i, val in zL.get(name, {}).items():
            if i in varcomp:
                model.ipopt_zL_in[varcomp[i]] = float(val)
        for i, val in zU.get(name, {}).items():
            if i in varcomp:
                model.ipopt_zU_in[varcomp[i]] = float(val)


# =============================================================================
# MODEL BUILDERS
# =============================================================================
def build_acopf_model(ctx, init_package=None):
    baseMVA = ctx["baseMVA"]
    bus = ctx["bus"]
    gen = ctx["gen"]
    branch = ctx["branch"]
    nb = ctx["nb"]
    nl = ctx["nl"]
    active_gens = ctx["active_gens"]
    bus_gens = ctx["bus_gens"]
    Ybus_rows = ctx["Ybus_rows"]
    Yf_rows = ctx["Yf_rows"]
    Yt_rows = ctx["Yt_rows"]
    Pd = ctx["Pd"]
    Qd = ctx["Qd"]
    F_indices = ctx["F_indices"]
    T_indices = ctx["T_indices"]
    RateA_Sq = ctx["RateA_Sq"]
    ref_idx = ctx["ref_idx"]

    m = pe.ConcreteModel(name="ACOPF_Paper_Study")
    m.B = pe.RangeSet(0, nb - 1)
    m.L = pe.RangeSet(0, nl - 1)
    m.G = pe.Set(initialize=active_gens, ordered=True)

    m.ipopt_zL_out = pe.Suffix(direction=pe.Suffix.IMPORT)
    m.ipopt_zU_out = pe.Suffix(direction=pe.Suffix.IMPORT)
    m.ipopt_zL_in = pe.Suffix(direction=pe.Suffix.EXPORT)
    m.ipopt_zU_in = pe.Suffix(direction=pe.Suffix.EXPORT)
    m.dual = pe.Suffix(direction=pe.Suffix.IMPORT_EXPORT)

    m.Vm = pe.Var(
        m.B,
        initialize=lambda m, b: float(bus[b, VM]),
        bounds=lambda m, b: (float(bus[b, VMIN]), float(bus[b, VMAX])),
    )
    m.Va = pe.Var(
        m.B,
        initialize=lambda m, b: float(math.radians(bus[b, VA])),
        bounds=(-2.0 * math.pi, 2.0 * math.pi),
    )
    m.Pg = pe.Var(
        m.G,
        initialize=lambda m, g: float(gen[g, PG] / baseMVA),
        bounds=lambda m, g: (float(gen[g, PMIN] / baseMVA), float(gen[g, PMAX] / baseMVA)),
    )
    m.Qg = pe.Var(
        m.G,
        initialize=lambda m, g: float(gen[g, QG] / baseMVA),
        bounds=lambda m, g: (float(gen[g, QMIN] / baseMVA), float(gen[g, QMAX] / baseMVA)),
    )

    add_generator_cost_structure(m, ctx, lambda m, g: m.Pg[g] * baseMVA)

    m.RefAngle = pe.Constraint(expr=m.Va[ref_idx] == 0.0)

    def p_bal(m, b):
        p_inj = 0.0
        vb = m.Vm[b]
        tb = m.Va[b]
        for k, G, B in Ybus_rows[b]:
            td = tb - m.Va[k]
            p_inj += m.Vm[k] * (G * pe.cos(td) + B * pe.sin(td))
        p_gen = sum(m.Pg[g] for g in bus_gens[b]) if bus_gens[b] else 0.0
        return p_gen - Pd[b] == vb * p_inj

    def q_bal(m, b):
        q_inj = 0.0
        vb = m.Vm[b]
        tb = m.Va[b]
        for k, G, B in Ybus_rows[b]:
            td = tb - m.Va[k]
            q_inj += m.Vm[k] * (G * pe.sin(td) - B * pe.cos(td))
        q_gen = sum(m.Qg[g] for g in bus_gens[b]) if bus_gens[b] else 0.0
        return q_gen - Qd[b] == vb * q_inj

    m.EqP = pe.Constraint(m.B, rule=p_bal)
    m.EqQ = pe.Constraint(m.B, rule=q_bal)

    def Vre(m, b):
        return m.Vm[b] * pe.cos(m.Va[b])

    def Vim(m, b):
        return m.Vm[b] * pe.sin(m.Va[b])

    def I_re_im_from_row(m, l, rows):
        Ire = 0.0
        Iim = 0.0
        for k, G, B in rows[l]:
            vr = Vre(m, k)
            vi = Vim(m, k)
            Ire += G * vr - B * vi
            Iim += B * vr + G * vi
        return Ire, Iim

    def S_sq_from(m, l):
        Ire, Iim = I_re_im_from_row(m, l, Yf_rows)
        fb = F_indices[l]
        vr = Vre(m, fb)
        vi = Vim(m, fb)
        Pf = vr * Ire + vi * Iim
        Qf = vi * Ire - vr * Iim
        return Pf * Pf + Qf * Qf

    def S_sq_to(m, l):
        Ire, Iim = I_re_im_from_row(m, l, Yt_rows)
        tb = T_indices[l]
        vr = Vre(m, tb)
        vi = Vim(m, tb)
        Pt = vr * Ire + vi * Iim
        Qt = vi * Ire - vr * Iim
        return Pt * Pt + Qt * Qt

    def flow_from_limit(m, l):
        limit_sq = float(RateA_Sq[l])
        if limit_sq <= 0.0 or int(branch[l, BR_STATUS]) == 0:
            return pe.Constraint.Skip
        return S_sq_from(m, l) <= limit_sq

    def flow_to_limit(m, l):
        limit_sq = float(RateA_Sq[l])
        if limit_sq <= 0.0 or int(branch[l, BR_STATUS]) == 0:
            return pe.Constraint.Skip
        return S_sq_to(m, l) <= limit_sq

    m.FlowFrom = pe.Constraint(m.L, rule=flow_from_limit)
    m.FlowTo = pe.Constraint(m.L, rule=flow_to_limit)

    apply_primal_start(m, init_package, ref_idx=ref_idx)
    apply_dual_start(m, init_package)
    return m


def build_dcopf_model(ctx):
    baseMVA = ctx["baseMVA"]
    gen = ctx["gen"]
    active_gens = ctx["active_gens"]
    bus_gens = ctx["bus_gens"]
    Pd = ctx["Pd"]
    ref_idx = ctx["ref_idx"]
    RateA = ctx["RateA"]
    Bbus_rows = ctx["Bbus_rows"]
    Bf_rows = ctx["Bf_rows"]
    Pbusinj = ctx["Pbusinj"]
    Pfinj = ctx["Pfinj"]

    m = pe.ConcreteModel(name="DCOPF")
    m.B = pe.RangeSet(0, ctx["nb"] - 1)
    m.L = pe.RangeSet(0, ctx["nl"] - 1)
    m.G = pe.Set(initialize=active_gens, ordered=True)

    m.Theta = pe.Var(m.B, initialize=0.0, bounds=(-2 * math.pi, 2 * math.pi))
    m.Pg = pe.Var(
        m.G,
        initialize=lambda m, g: float(gen[g, PG] / baseMVA),
        bounds=lambda m, g: (float(gen[g, PMIN] / baseMVA), float(gen[g, PMAX] / baseMVA)),
    )

    add_generator_cost_structure(m, ctx, lambda m, g: m.Pg[g] * baseMVA)

    m.Ref = pe.Constraint(expr=m.Theta[ref_idx] == 0.0)

    def p_balance(m, b):
        theta_term = 0.0
        for k, Bij in Bbus_rows[b]:
            theta_term += Bij * m.Theta[k]
        p_gen = sum(m.Pg[g] for g in bus_gens[b]) if bus_gens[b] else 0.0
        return p_gen - Pd[b] == theta_term + float(Pbusinj[b])

    m.PBalance = pe.Constraint(m.B, rule=p_balance)

    def flow_expr(m, l):
        val = 0.0
        for k, Bij in Bf_rows[l]:
            val += Bij * m.Theta[k]
        return val + float(Pfinj[l])

    def flow_upper(m, l):
        if RateA[l] <= 0.0 or int(ctx["branch"][l, BR_STATUS]) == 0:
            return pe.Constraint.Skip
        return flow_expr(m, l) <= float(RateA[l])

    def flow_lower(m, l):
        if RateA[l] <= 0.0 or int(ctx["branch"][l, BR_STATUS]) == 0:
            return pe.Constraint.Skip
        return flow_expr(m, l) >= -float(RateA[l])

    m.FlowU = pe.Constraint(m.L, rule=flow_upper)
    m.FlowL = pe.Constraint(m.L, rule=flow_lower)
    return m


# =============================================================================
# POST-SOLVE METRICS
# =============================================================================
def collect_solution_metrics(model, ctx, baseline_solution=None):
    baseMVA = ctx["baseMVA"]
    bus = ctx["bus"]
    gen = ctx["gen"]
    branch = ctx["branch"]
    nl = ctx["nl"]

    Vm = {int(b): float(pe.value(model.Vm[b])) for b in model.B}
    Va = {int(b): float(pe.value(model.Va[b])) for b in model.B}
    Pg = {int(g): float(pe.value(model.Pg[g])) for g in model.G}
    Qg = {int(g): float(pe.value(model.Qg[g])) for g in model.G}

    vm_vals = np.array(list(Vm.values()), dtype=float)
    va_vals = np.array(list(Va.values()), dtype=float)

    Yf_rows = ctx["Yf_rows"]
    Yt_rows = ctx["Yt_rows"]
    F_indices = ctx["F_indices"]
    T_indices = ctx["T_indices"]
    RateA_Sq = ctx["RateA_Sq"]

    def Vre(b):
        return Vm[b] * math.cos(Va[b])

    def Vim_f(b):
        return Vm[b] * math.sin(Va[b])

    def current_from_row(l, rows):
        Ire = 0.0
        Iim = 0.0
        for k, G, B in rows[l]:
            vr = Vre(k)
            vi = Vim_f(k)
            Ire += G * vr - B * vi
            Iim += B * vr + G * vi
        return Ire, Iim

    s_from, s_to, util_from, util_to = [], [], [], []
    viol_from, viol_to = [], []
    for l in range(nl):
        if int(branch[l, BR_STATUS]) == 0:
            continue
        Ire, Iim = current_from_row(l, Yf_rows)
        fb = F_indices[l]
        vr = Vre(fb)
        vi = Vim_f(fb)
        Pf = vr * Ire + vi * Iim
        Qf = vi * Ire - vr * Iim
        Sf = math.sqrt(max(Pf * Pf + Qf * Qf, 0.0))
        s_from.append(Sf * baseMVA)

        Ire, Iim = current_from_row(l, Yt_rows)
        tb = T_indices[l]
        vr = Vre(tb)
        vi = Vim_f(tb)
        Pt = vr * Ire + vi * Iim
        Qt = vi * Ire - vr * Iim
        St = math.sqrt(max(Pt * Pt + Qt * Qt, 0.0))
        s_to.append(St * baseMVA)

        rate = float(branch[l, RATE_A])
        if rate > 0:
            util_from.append((Sf * baseMVA) / rate)
            util_to.append((St * baseMVA) / rate)
            viol_from.append(max((Sf * Sf) - float(RateA_Sq[l]), 0.0))
            viol_to.append(max((St * St) - float(RateA_Sq[l]), 0.0))

    tol = 1e-5
    n_pg_lb = sum(abs(Pg[g] - float(gen[g, PMIN] / baseMVA)) <= tol for g in model.G)
    n_pg_ub = sum(abs(Pg[g] - float(gen[g, PMAX] / baseMVA)) <= tol for g in model.G)
    n_qg_lb = sum(abs(Qg[g] - float(gen[g, QMIN] / baseMVA)) <= tol for g in model.G)
    n_qg_ub = sum(abs(Qg[g] - float(gen[g, QMAX] / baseMVA)) <= tol for g in model.G)
    n_vm_lb = sum(abs(Vm[b] - float(bus[b, VMIN])) <= tol for b in model.B)
    n_vm_ub = sum(abs(Vm[b] - float(bus[b, VMAX])) <= tol for b in model.B)

    out = {
        "objective_value": float(safe_value(model.Obj)) if safe_value(model.Obj) is not None else None,
        "total_pg_mw": float(sum(Pg.values()) * baseMVA),
        "total_qg_mvar": float(sum(Qg.values()) * baseMVA),
        "vm_min": float(np.min(vm_vals)),
        "vm_max": float(np.max(vm_vals)),
        "vm_mean": float(np.mean(vm_vals)),
        "vm_std": float(np.std(vm_vals)),
        "va_min_rad": float(np.min(va_vals)),
        "va_max_rad": float(np.max(va_vals)),
        "va_mean_rad": float(np.mean(va_vals)),
        "va_std_rad": float(np.std(va_vals)),
        "max_line_mva_from": float(max(s_from)) if s_from else None,
        "max_line_mva_to": float(max(s_to)) if s_to else None,
        "mean_line_mva_from": float(np.mean(s_from)) if s_from else None,
        "mean_line_mva_to": float(np.mean(s_to)) if s_to else None,
        "max_thermal_util_from": float(max(util_from)) if util_from else None,
        "max_thermal_util_to": float(max(util_to)) if util_to else None,
        "mean_thermal_util_from": float(np.mean(util_from)) if util_from else None,
        "mean_thermal_util_to": float(np.mean(util_to)) if util_to else None,
        "max_flow_limit_violation_sq_from": float(max(viol_from)) if viol_from else 0.0,
        "max_flow_limit_violation_sq_to": float(max(viol_to)) if viol_to else 0.0,
        "n_pg_at_lb": int(n_pg_lb),
        "n_pg_at_ub": int(n_pg_ub),
        "n_qg_at_lb": int(n_qg_lb),
        "n_qg_at_ub": int(n_qg_ub),
        "n_vm_at_lb": int(n_vm_lb),
        "n_vm_at_ub": int(n_vm_ub),
    }

    if baseline_solution is not None:
        for name, cur in [("Pg", Pg), ("Qg", Qg), ("Vm", Vm), ("Va", Va)]:
            stats = dict_abs_diff_stats(cur, baseline_solution[name])
            for k, v in stats.items():
                out[f"dist_to_baseline_{name}_{k}"] = v
    return out


def collect_initialization_quality(init_package, baseline_ws):
    row = {}
    baseline_primal = baseline_ws["primal"] if baseline_ws is not None else None
    if init_package is None or baseline_primal is None:
        for blk in BLOCKS:
            row[f"init_has_{blk}"] = 0
            for stat in ["l2", "linf", "mean_abs", "median_abs", "max_abs"]:
                row[f"init_{blk}_{stat}"] = None
        row["init_has_duals"] = 0
        row["init_has_constraint_duals"] = 0
        row["init_has_bound_mults"] = 0
        return row

    primal = init_package.get("primal", {})
    for blk in BLOCKS:
        row[f"init_has_{blk}"] = 1 if blk in primal else 0
        if blk in primal:
            stats = dict_abs_diff_stats(primal[blk], baseline_primal[blk])
            for stat, val in stats.items():
                row[f"init_{blk}_{stat}"] = val
        else:
            for stat in ["l2", "linf", "mean_abs", "median_abs", "max_abs"]:
                row[f"init_{blk}_{stat}"] = None
    row["init_has_duals"] = 1 if bool(init_package.get("dual")) else 0
    row["init_has_constraint_duals"] = 1 if bool(init_package.get("dual")) else 0
    row["init_has_bound_mults"] = 1 if bool(init_package.get("zL")) or bool(init_package.get("zU")) else 0
    return row


# =============================================================================
# SOLVER CONFIGURATION
# =============================================================================
def configure_ipopt(solver, use_dual_warmstart=False, max_iter=300, tol=1e-6, print_level=5):
    solver.options["tol"] = tol
    solver.options["max_iter"] = max_iter
    solver.options["print_level"] = print_level
    if use_dual_warmstart:
        solver.options["warm_start_init_point"] = "yes"
        solver.options["warm_start_bound_push"] = 1e-6
        solver.options["warm_start_bound_frac"] = 1e-6
        solver.options["warm_start_mult_bound_push"] = 1e-6
        solver.options["mu_init"] = 1e-6
        solver.options["least_square_init_duals"] = "no"

def solver_usable(name, problem_type="lp"):
    try:
        solver = pe.SolverFactory(name)
        if solver is None or not solver.available(False):
            return False

        m = pe.ConcreteModel()
        m.x = pe.Var(bounds=(0, 10))

        if problem_type == "lp":
            m.obj = pe.Objective(expr=2 * m.x)
            m.c = pe.Constraint(expr=m.x >= 1)
        elif problem_type == "qp":
            m.obj = pe.Objective(expr=(m.x - 1.5) ** 2 + m.x)
            m.c = pe.Constraint(expr=m.x >= 0.5)
        elif problem_type == "nlp":
            m.obj = pe.Objective(expr=(m.x - 1.25) ** 4 + m.x)
            m.c = pe.Constraint(expr=m.x >= 0.5)
        else:
            raise ValueError(f"Unknown solver test type: {problem_type}")

        results = solver.solve(m, tee=False)
        return success_termination(results.solver.termination_condition)
    except Exception:
        return False


def choose_dcopf_solver(ctx):
    max_degree = ctx["cost_specs"]["max_poly_degree"]
    any_quadratic = ctx["cost_specs"]["any_quadratic"]

    if max_degree > 2:
        candidates = [("ipopt", "nlp")]
    elif any_quadratic:
        candidates = [("highs", "qp"), ("cplex", "qp"), ("ipopt", "nlp")]
    else:
        candidates = [("highs", "lp"), ("cplex", "lp"), ("ipopt", "nlp")]

    for name, problem_type in candidates:
        if solver_usable(name, problem_type=problem_type):
            return name, problem_type
    raise RuntimeError("No usable solver found for DCOPF with the required problem structure.")


# =============================================================================
# SOLVE WRAPPERS
# =============================================================================
def solve_one_ac(ctx, family, blocks, init_package, case_output_dir, baseline_ws=None,
                 tee=False, max_iter=300, tol=1e-6, vm_policy=None, qg_policy=None,
                 dc_meta=None, shared_case_prep_time=0.0, baseline_objective=None,
                 objective_match_tol=DEFAULT_OBJECTIVE_MATCH_TOL, dual_mode="none"):
    run_name = make_run_name(family, blocks, vm_policy=vm_policy, qg_policy=qg_policy)
    case_file = ctx["case_file"]
    print(f"\n--- {os.path.basename(case_file)} | {run_name} ---")

    build_start = time.time()
    model = build_acopf_model(ctx, init_package=init_package)
    build_time = time.time() - build_start

    solver = pe.SolverFactory("ipopt")
    if solver is None or not solver.available(False):
        raise RuntimeError("Could not create usable IPOPT solver.")

    use_dual_warmstart = (family in WARM_START_FAMILIES)
    configure_ipopt(
        solver,
        use_dual_warmstart=use_dual_warmstart,
        max_iter=max_iter,
        tol=tol,
        print_level=5
    )

    logs_dir = ensure_dir(os.path.join(case_output_dir, "logs"))
    log_path = os.path.join(logs_dir, f"{run_name}.log")

    solve_start = time.time()
    try:
        results = solver.solve(model, tee=tee, logfile=log_path)
    except TypeError:
        results = solver.solve(model, tee=tee)
    solve_time = time.time() - solve_start
    total_time = build_time + solve_time

    term = results.solver.termination_condition
    term_str = str(term)
    status_str = str(results.solver.status)
    ok = bool(success_termination(term))

    ipopt_log = parse_ipopt_log(log_path)

    solver_iterations = ipopt_log.get("iterations", None)
    if solver_iterations is None:
        for attr in ("iterations", "iteration_count", "iter_count"):
            val = getattr(results.solver, attr, None)
            if val is not None:
                try:
                    solver_iterations = int(val)
                    break
                except Exception:
                    pass

    objective = None
    primal_solution = None
    full_warmstart = None
    solution_metrics = {}

    if ok:
        try:
            objective = float(pe.value(model.Obj))
        except Exception:
            objective = None

        try:
            primal_solution = extract_primal_solution(model)
        except Exception:
            primal_solution = None

        try:
            full_warmstart = extract_full_warmstart(model)
        except Exception:
            full_warmstart = None

        try:
            baseline_solution = baseline_ws["primal"] if baseline_ws is not None else None
            solution_metrics = collect_solution_metrics(model, ctx, baseline_solution=baseline_solution)
        except Exception:
            solution_metrics = {}

        print(
            f"[OK] {run_name}: {term_str} | "
            f"obj={objective if objective is not None else 'None'} | "
            f"iter={solver_iterations} | solve={solve_time:.3f}s"
        )
    else:
        print(f"[FAIL] {run_name}: {term_str} | iter={solver_iterations} | solve={solve_time:.3f}s")

    init_quality = collect_initialization_quality(init_package, baseline_ws) if baseline_ws is not None else {}
    dc_total_time = dc_meta.get("dc_total_time", 0.0) if dc_meta else 0.0
    full_pipeline_time = total_time + dc_total_time + shared_case_prep_time

    obj_gap_abs = None
    obj_gap_rel = None
    obj_match = None
    if objective is not None and baseline_objective is not None:
        obj_gap_abs = float(objective - baseline_objective)
        denom = max(1.0, abs(float(baseline_objective)))
        obj_gap_rel = float((objective - baseline_objective) / denom)
        obj_match = int(abs(obj_gap_abs) <= max(objective_match_tol, objective_match_tol * denom))

    row = {
        "timestamp_utc": now_utc_iso(),
        "case_file": case_file,
        "case_name": os.path.splitext(os.path.basename(case_file))[0],
        "run_name": run_name,
        "family": family,
        "dual_mode": dual_mode,
        "seed_source": "case_default" if family == "baseline" else ("ac_true" if family.startswith("oracle_ac") else "dcopf"),
        "is_oracle": 1 if family.startswith("oracle_ac") else 0,
        "is_practical": 0 if family.startswith("oracle_ac") else 1,
        "blocks": tuple(blocks),
        "num_blocks": len(blocks),
        "seed_vm_policy": vm_policy,
        "seed_qg_policy": qg_policy,
        "termination": term_str,
        "status": status_str,
        "success": int(ok),
        "objective": objective,
        "objective_gap_vs_baseline_abs": obj_gap_abs,
        "objective_gap_vs_baseline_rel": obj_gap_rel,
        "objective_matches_baseline": obj_match,
        "shared_case_prep_time": shared_case_prep_time,
        "build_time": build_time,
        "solve_time": solve_time,
        "total_time": total_time,
        "end_to_end_time": total_time + dc_total_time,
        "full_pipeline_time_including_shared_case_prep": full_pipeline_time,
        "log_path": log_path,
        "hit_max_iter": int(solver_iterations is not None and solver_iterations >= max_iter),
        **ipopt_log,
        **ctx["case_stats"],
        **init_quality,
        **solution_metrics,
    }

    row["iterations"] = solver_iterations

    if dc_meta:
        row.update(dc_meta)

    return {"row": row, "primal_solution": primal_solution, "full_warmstart": full_warmstart}

def solve_dcopf(ctx, case_output_dir):
    solver_name, solver_problem_type = choose_dcopf_solver(ctx)

    build_start = time.time()
    model = build_dcopf_model(ctx)
    build_time = time.time() - build_start

    solver = pe.SolverFactory(solver_name)
    if solver is None or not solver.available(False):
        raise RuntimeError(f"Could not create usable DC solver {solver_name}.")

    logs_dir = ensure_dir(os.path.join(case_output_dir, "logs"))
    log_path = os.path.join(logs_dir, "dcopf.log")

    t0 = time.time()
    try:
        results = solver.solve(model, tee=False, logfile=log_path)
    except TypeError:
        results = solver.solve(model, tee=False)
    solve_time = time.time() - t0
    total_time = build_time + solve_time

    term = results.solver.termination_condition
    ok = success_termination(term)
    obj = float(pe.value(model.Obj)) if ok else None
    sol = None
    if ok:
        sol = {
            "Pg": {int(g): float(pe.value(model.Pg[g])) for g in model.G},
            "Va": {int(b): float(pe.value(model.Theta[b])) for b in model.B},
            "objective": obj,
        }
    return {
        "success": ok,
        "termination": str(term),
        "status": str(results.solver.status),
        "solver": solver_name,
        "solver_problem_type": solver_problem_type,
        "build_time": build_time,
        "solve_time": solve_time,
        "total_time": total_time,
        "objective": obj,
        "log_path": log_path,
        "solution": sol,
    }


# =============================================================================
# CASE PAYLOAD STORAGE
# =============================================================================
def save_case_solution_npz(case_output_dir, case_name, raw_solutions):
    arrays = {}
    for run_name, sol in raw_solutions.items():
        for var in ["Vm", "Va", "Pg", "Qg"]:
            if var in sol:
                keys = np.array(sorted(sol[var].keys()), dtype=int)
                vals = np.array([sol[var][k] for k in keys], dtype=float)
                arrays[f"{run_name}__{var}__idx"] = keys
                arrays[f"{run_name}__{var}__val"] = vals
    path = os.path.join(case_output_dir, f"{case_name}__solutions.npz")
    np.savez_compressed(path, **arrays)
    return path


def save_case_json_payload(case_output_dir, case_name, payload):
    path = os.path.join(case_output_dir, f"{case_name}__payload.json.gz")
    json_gz_dump(payload, path)
    return path


# =============================================================================
# AGGREGATIONS
# =============================================================================
def aggregate_per_case_summary(all_case_results):
    out = []
    for case_name, case_rows in all_case_results.items():
        base = next((r for r in case_rows if r["family"] == "baseline" and row_is_successful(r)), None)

        oracle_po_all = [r for r in case_rows if r["family"] == "oracle_ac_primal_only" and row_is_successful(r)]
        oracle_pd_all = [r for r in case_rows if r["family"] == "oracle_ac_primal_dual" and row_is_successful(r)]
        dc_all = [r for r in case_rows if r["family"] == "dc_seed" and row_is_successful(r)]
        all_good_all = [r for r in case_rows if row_is_successful(r) and r["family"] != "baseline"]

        oracle_po = objective_matched_rows(oracle_po_all)
        oracle_pd = objective_matched_rows(oracle_pd_all)
        dc_rows = objective_matched_rows(dc_all)
        all_good = objective_matched_rows(all_good_all)

        best_po = min(oracle_po, key=lambda r: r["solve_time"]) if oracle_po else None
        best_pd = min(oracle_pd, key=lambda r: r["solve_time"]) if oracle_pd else None
        best_dc = min(dc_rows, key=lambda r: r["solve_time"]) if dc_rows else None
        best_dc_e2e = min(dc_rows, key=lambda r: r["end_to_end_time"]) if dc_rows else None
        best_overall = min(all_good, key=lambda r: r["solve_time"]) if all_good else None

        row = {
            "case_name": case_name,
            "baseline_solve_time": base["solve_time"] if base else None,
            "baseline_total_time": base["total_time"] if base else None,
            "baseline_iterations": base["iterations"] if base else None,
            "baseline_objective": base["objective"] if base else None,
            "baseline_hit_max_iter": base["hit_max_iter"] if base else None,

            "best_oracle_primal_only_run": best_po["run_name"] if best_po else None,
            "best_oracle_primal_only_solve_time": best_po["solve_time"] if best_po else None,
            "best_oracle_primal_only_speedup_pct": speedup_pct(base["solve_time"], best_po["solve_time"]) if base and best_po else None,
            "best_oracle_primal_only_obj_match": best_po.get("objective_matches_baseline") if best_po else None,

            "best_oracle_primal_dual_run": best_pd["run_name"] if best_pd else None,
            "best_oracle_primal_dual_solve_time": best_pd["solve_time"] if best_pd else None,
            "best_oracle_primal_dual_speedup_pct": speedup_pct(base["solve_time"], best_pd["solve_time"]) if base and best_pd else None,
            "best_oracle_primal_dual_obj_match": best_pd.get("objective_matches_baseline") if best_pd else None,

            "best_dc_run": best_dc["run_name"] if best_dc else None,
            "best_dc_solve_time": best_dc["solve_time"] if best_dc else None,
            "best_dc_speedup_pct": speedup_pct(base["solve_time"], best_dc["solve_time"]) if base and best_dc else None,
            "best_dc_obj_match": best_dc.get("objective_matches_baseline") if best_dc else None,

            "best_dc_end_to_end_run": best_dc_e2e["run_name"] if best_dc_e2e else None,
            "best_dc_end_to_end_time": best_dc_e2e["end_to_end_time"] if best_dc_e2e else None,
            "best_dc_end_to_end_speedup_pct": speedup_pct(base["total_time"], best_dc_e2e["end_to_end_time"]) if base and best_dc_e2e else None,
            "best_dc_end_to_end_obj_match": best_dc_e2e.get("objective_matches_baseline") if best_dc_e2e else None,

            "best_overall_run": best_overall["run_name"] if best_overall else None,
            "best_overall_solve_time": best_overall["solve_time"] if best_overall else None,
            "best_overall_speedup_pct": speedup_pct(base["solve_time"], best_overall["solve_time"]) if base and best_overall else None,
            "best_overall_obj_match": best_overall.get("objective_matches_baseline") if best_overall else None,
        }
        out.append(row)
    return out

def aggregate_global_summary(global_rows):
    families = sorted(set(r["family"] for r in global_rows))
    out = []
    for fam in families:
        rows = [r for r in global_rows if r["family"] == fam and row_is_successful(r)]
        if not rows:
            continue

        iter_vals = [r["iterations"] for r in rows if r.get("iterations") is not None]
        pipeline_vals = [
            r["full_pipeline_time_including_shared_case_prep"]
            for r in rows
            if r.get("full_pipeline_time_including_shared_case_prep") is not None
        ]
        objective_vals = [r["objective"] for r in rows if r.get("objective") is not None]

        n_matching, share_matching = objective_match_counts(rows)

        out.append({
            "family": fam,
            "n": len(rows),
            "avg_solve_time": mean(r["solve_time"] for r in rows),
            "median_solve_time": median(r["solve_time"] for r in rows),
            "avg_total_time": mean(r["total_time"] for r in rows),
            "median_total_time": median(r["total_time"] for r in rows),
            "avg_end_to_end_time": mean(r["end_to_end_time"] for r in rows),
            "median_end_to_end_time": median(r["end_to_end_time"] for r in rows),
            "avg_full_pipeline_time_including_shared_case_prep": mean(pipeline_vals) if pipeline_vals else None,
            "median_full_pipeline_time_including_shared_case_prep": median(pipeline_vals) if pipeline_vals else None,
            "avg_iterations": mean(iter_vals) if iter_vals else None,
            "median_iterations": median(iter_vals) if iter_vals else None,
            "avg_objective": mean(objective_vals) if objective_vals else None,
            "n_matching_objective": n_matching,
            "share_matching_objective": share_matching,
        })
    return out

def aggregate_combo_summary(global_rows):
    keys = set((r["family"], tuple(r["blocks"]), r.get("seed_vm_policy"), r.get("seed_qg_policy")) for r in global_rows)
    out = []
    for fam, blocks, vm_policy, qg_policy in sorted(keys):
        rows = [
            r for r in global_rows
            if r["family"] == fam
            and tuple(r["blocks"]) == tuple(blocks)
            and r.get("seed_vm_policy") == vm_policy
            and r.get("seed_qg_policy") == qg_policy
            and row_is_successful(r)
        ]
        if not rows:
            continue
        iter_vals = [r["iterations"] for r in rows if r.get("iterations") is not None]
        pipeline_vals = [r["full_pipeline_time_including_shared_case_prep"] for r in rows if r.get("full_pipeline_time_including_shared_case_prep") is not None]
        n_matching, share_matching = objective_match_counts(rows)
        out.append({
            "family": fam,
            "blocks": "+".join(blocks),
            "num_blocks": len(blocks),
            "seed_vm_policy": vm_policy,
            "seed_qg_policy": qg_policy,
            "n": len(rows),
            "avg_solve_time": mean(r["solve_time"] for r in rows),
            "median_solve_time": median(r["solve_time"] for r in rows),
            "avg_end_to_end_time": mean(r["end_to_end_time"] for r in rows),
            "median_end_to_end_time": median(r["end_to_end_time"] for r in rows),
            "avg_full_pipeline_time_including_shared_case_prep": mean(pipeline_vals) if pipeline_vals else None,
            "median_full_pipeline_time_including_shared_case_prep": median(pipeline_vals) if pipeline_vals else None,
            "avg_iterations": mean(iter_vals) if iter_vals else None,
            "median_iterations": median(iter_vals) if iter_vals else None,
            "n_matching_objective": n_matching,
            "share_matching_objective": share_matching,
        })
    return out

def aggregate_block_effects(global_rows, family):
    fam_all = [r for r in global_rows if r["family"] == family and r.get("objective") is not None]
    fam = objective_matched_rows(fam_all)
    out = []
    for blk in BLOCKS:
        with_blk = [r["solve_time"] for r in fam if blk in tuple(r["blocks"])]
        without_blk = [r["solve_time"] for r in fam if blk not in tuple(r["blocks"])]
        out.append({
            "family": family,
            "block": blk,
            "n_with": len(with_blk),
            "n_without": len(without_blk),
            "avg_solve_with": mean(with_blk) if with_blk else None,
            "avg_solve_without": mean(without_blk) if without_blk else None,
            "pct_better_when_included": 100.0 * (mean(without_blk) - mean(with_blk)) / mean(without_blk) if with_blk and without_blk and mean(without_blk) > 0 else None,
        })
    return out


def aggregate_dc_policy_summary(global_rows):
    dc = [
        r for r in global_rows
        if r["family"] == "dc_seed"
        and row_is_successful(r)
        and r.get("dc_seed_mode") == "completed_vm_qg"
        and r.get("seed_vm_policy") is not None
        and r.get("seed_qg_policy") is not None
    ]
    groups = {}
    for r in dc:
        k = (r.get("seed_vm_policy"), r.get("seed_qg_policy"))
        groups.setdefault(k, []).append(r)
    out = []
    for (vm_policy, qg_policy), rows in sorted(groups.items()):
        iter_vals = [r["iterations"] for r in rows if r.get("iterations") is not None]
        pipeline_vals = [r["full_pipeline_time_including_shared_case_prep"] for r in rows if r.get("full_pipeline_time_including_shared_case_prep") is not None]
        n_matching, share_matching = objective_match_counts(rows)
        out.append({
            "seed_vm_policy": vm_policy,
            "seed_qg_policy": qg_policy,
            "n": len(rows),
            "avg_solve_time": mean(r["solve_time"] for r in rows),
            "median_solve_time": median(r["solve_time"] for r in rows),
            "avg_end_to_end_time": mean(r["end_to_end_time"] for r in rows),
            "median_end_to_end_time": median(r["end_to_end_time"] for r in rows),
            "avg_full_pipeline_time_including_shared_case_prep": mean(pipeline_vals) if pipeline_vals else None,
            "median_full_pipeline_time_including_shared_case_prep": median(pipeline_vals) if pipeline_vals else None,
            "avg_iterations": mean(iter_vals) if iter_vals else None,
            "median_iterations": median(iter_vals) if iter_vals else None,
            "n_matching_objective": n_matching,
            "share_matching_objective": share_matching,
        })
    return out

def aggregate_paper_table_rows(per_case_summary):
    out = []
    for r in per_case_summary:
        out.append({
            "case_name": r["case_name"],
            "baseline_time_s": r["baseline_solve_time"],
            "baseline_iter": r["baseline_iterations"],
            "best_oracle_primal_only": r["best_oracle_primal_only_run"],
            "best_oracle_primal_only_time_s": r["best_oracle_primal_only_solve_time"],
            "best_oracle_primal_only_speedup_pct": r["best_oracle_primal_only_speedup_pct"],
            "best_oracle_primal_dual": r["best_oracle_primal_dual_run"],
            "best_oracle_primal_dual_time_s": r["best_oracle_primal_dual_solve_time"],
            "best_oracle_primal_dual_speedup_pct": r["best_oracle_primal_dual_speedup_pct"],
            "best_dc": r["best_dc_run"],
            "best_dc_time_s": r["best_dc_solve_time"],
            "best_dc_speedup_pct": r["best_dc_speedup_pct"],
            "best_dc_end_to_end": r["best_dc_end_to_end_run"],
            "best_dc_end_to_end_time_s": r["best_dc_end_to_end_time"],
            "best_dc_end_to_end_speedup_pct": r["best_dc_end_to_end_speedup_pct"],
            "best_overall": r["best_overall_run"],
            "best_overall_time_s": r["best_overall_solve_time"],
            "best_overall_speedup_pct": r["best_overall_speedup_pct"],
        })
    return out


def aggregate_family_wins(per_case_summary):
    rows = []
    for fam_key in ["oracle_ac_primal_only", "oracle_ac_primal_dual", "dc_seed"]:
        wins = 0
        for r in per_case_summary:
            candidates = []
            if r.get("best_oracle_primal_only_solve_time") is not None:
                candidates.append(("oracle_ac_primal_only", r["best_oracle_primal_only_solve_time"]))
            if r.get("best_oracle_primal_dual_solve_time") is not None:
                candidates.append(("oracle_ac_primal_dual", r["best_oracle_primal_dual_solve_time"]))
            if r.get("best_dc_solve_time") is not None:
                candidates.append(("dc_seed", r["best_dc_solve_time"]))
            if candidates and min(candidates, key=lambda x: x[1])[0] == fam_key:
                wins += 1
        rows.append({"family": fam_key, "case_wins_by_best_solve_time": wins})
    return rows


# =============================================================================
# READMEs
# =============================================================================
def write_readme(output_dir, case_root, n_cases, config):
    include_dc_completion = bool(config.get("include_dc_completion_sensitivity", False))
    include_dual_decomp = bool(config.get("include_extended_dual_decomposition", False))
    native_dc_runs = len(DC_NATIVE_ONLY_COMBOS)
    extra_dc_runs = len(dc_completion_combos()) * len(DC_VM_POLICIES) * len(DC_QG_POLICIES) if include_dc_completion else 0

    extended_dual_runs = 0
    if include_dual_decomp:
        extended_dual_runs = len(EXTENDED_DUAL_FAMILIES) * (2 ** len(BLOCKS) - 1) + len(DUAL_ONLY_VARIANTS)

    total_per_case = 1 + (2 * (2 ** len(BLOCKS) - 1)) + native_dc_runs + extra_dc_runs + extended_dual_runs

    if include_dc_completion:
        dc_family_desc = (
            f"`dc_seed`: {native_dc_runs} native-only runs (Pg, Va, Pg+Va) + "
            f"{extra_dc_runs} optional completion-sensitivity runs "
            f"(9 completion combos x {len(DC_VM_POLICIES)} Vm policies x {len(DC_QG_POLICIES)} Qg policies)"
        )
    else:
        dc_family_desc = f"`dc_seed`: {native_dc_runs} native-only practical DC runs = Pg, Va, and Pg+Va"

    dual_decomp_desc = ""
    if include_dual_decomp:
        dual_decomp_desc = f"""
## Extended Dual Decomposition Experiments
When `include_extended_dual_decomposition=True`, three additional oracle families are run
to isolate the contribution of constraint duals vs. bound multipliers:

- `oracle_ac_constraint_dual`: 15 runs (constraint duals lambda* only, no zL/zU)
- `oracle_ac_bounds_dual`: 15 runs (block-matched zL/zU only, no constraint duals)
- `oracle_ac_primal_dual_all_bounds`: 15 runs (constraint duals + ALL zL/zU for all 4 blocks)

Plus 3 dual-only edge cases (no primal blocks supplied):
- `oracle_ac_dual_only_constraint`: constraint duals only
- `oracle_ac_dual_only_bounds`: all bound multipliers only
- `oracle_ac_dual_only_full`: constraint duals + all bound multipliers

Total extended dual runs: {extended_dual_runs} per case.
"""

    text = f"""# ACOPF Oracle and DC-Seed Warm-Start Study

Created: {now_utc_iso()}

## Overview
This pipeline runs ACOPF experiments for every MATPOWER `.mat` case:
- baseline default ACOPF initialization
- oracle AC warm starts from the true AC solution
- practical DC-seeded AC warm starts from a DCOPF solve

The per-run timing metrics exclude shared case parsing / admittance preprocessing and report:
- `build_time`: optimization model build time for that run
- `solve_time`: solver execution time for that run
- `total_time = build_time + solve_time`
- `end_to_end_time = total_time + dc_total_time` for DC-seeded runs

The shared one-time case preparation cost is recorded separately as `shared_case_prep_time`.

## Families
- `baseline`: 1 run
- `oracle_ac_primal_only`: 15 runs over all non-empty subsets of {{Pg, Qg, Vm, Va}} (dual_mode=none)
- `oracle_ac_primal_dual`: 15 runs over all non-empty subsets (dual_mode=full, block-matched bounds)
- {dc_family_desc}

Total planned runs per successful case (base): {1 + 30 + native_dc_runs + extra_dc_runs}.
{dual_decomp_desc}
Total planned runs per successful case (all): {total_per_case}.

## Dual Modes
Each oracle run records a `dual_mode` field:
- `none`: no dual info supplied (IPOPT initializes from scratch)
- `full`: constraint duals + block-matched zL/zU
- `constraint_only`: constraint duals only, no zL/zU
- `bounds_only`: block-matched zL/zU only, no constraint duals
- `full_all_bounds`: constraint duals + ALL zL/zU (all 4 blocks)

## Configuration
- case_root = {case_root}
- n_cases = {n_cases}
- tol = {config['tol']}
- max_iter = {config['max_iter']}
- tee_baseline = {config['tee_baseline']}
- tee_others = {config['tee_others']}
- include_dc_completion_sensitivity = {config.get('include_dc_completion_sensitivity', False)}
- include_extended_dual_decomposition = {config.get('include_extended_dual_decomposition', False)}
- objective_match_tol = {config.get('objective_match_tol', DEFAULT_OBJECTIVE_MATCH_TOL)}
"""
    path = os.path.join(output_dir, "README.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return path


# =============================================================================
# CASE PIPELINE
# =============================================================================
def run_case_study(case_file, output_dir, tee_baseline=False, tee_others=False, max_iter=300, tol=1e-6,
                   include_dc_completion_sensitivity=DEFAULT_INCLUDE_DC_COMPLETION_SENSITIVITY,
                   include_extended_dual_decomposition=DEFAULT_INCLUDE_EXTENDED_DUAL_DECOMPOSITION,
                   objective_match_tol=DEFAULT_OBJECTIVE_MATCH_TOL):
    case_name = os.path.splitext(os.path.basename(case_file))[0]
    case_output_dir = ensure_dir(os.path.join(output_dir, "cases", case_name))

    ctx_start = time.time()
    ctx = prepare_case_context(case_file)
    shared_case_prep_time = time.time() - ctx_start
    case_sha256 = sha256_of_file(case_file)

    global_rows = []
    raw_solutions = {}
    raw_warmstarts = {}
    extra_payload = {}

    baseline_out = solve_one_ac(
        ctx=ctx,
        family="baseline",
        blocks=(),
        init_package=None,
        case_output_dir=case_output_dir,
        baseline_ws=None,
        tee=tee_baseline,
        max_iter=max_iter,
        tol=tol,
        shared_case_prep_time=shared_case_prep_time,
        baseline_objective=None,
        objective_match_tol=objective_match_tol,
        dual_mode="none",
    )
    baseline_row = baseline_out["row"]
    global_rows.append(baseline_row)
    if baseline_out["primal_solution"] is not None:
        raw_solutions[baseline_row["run_name"]] = baseline_out["primal_solution"]
    if baseline_out["full_warmstart"] is not None:
        raw_warmstarts[baseline_row["run_name"]] = baseline_out["full_warmstart"]

    baseline_ws = baseline_out["full_warmstart"]
    baseline_objective = baseline_row.get("objective")
    if baseline_ws is None:
        return {
            "case_name": case_name,
            "case_rows": global_rows,
            "case_manifest": {
                "case_name": case_name,
                "case_file": case_file,
                "case_sha256": case_sha256,
                "baseline_success": False,
                "shared_case_prep_time": shared_case_prep_time,
                "case_output_dir": case_output_dir,
            },
        }

    all_combos = list(powerset_nonempty(BLOCKS))

    for family, dual_mode in [("oracle_ac_primal_only", "none"), ("oracle_ac_primal_dual", "full")]:
        for blocks in all_combos:
            pkg = build_warmstart_package(baseline_ws, blocks, dual_mode=dual_mode)
            out = solve_one_ac(
                ctx=ctx,
                family=family,
                blocks=blocks,
                init_package=pkg,
                case_output_dir=case_output_dir,
                baseline_ws=baseline_ws,
                tee=tee_others,
                max_iter=max_iter,
                tol=tol,
                shared_case_prep_time=shared_case_prep_time,
                baseline_objective=baseline_objective,
                objective_match_tol=objective_match_tol,
                dual_mode=dual_mode,
            )
            global_rows.append(out["row"])
            if out["primal_solution"] is not None:
                raw_solutions[out["row"]["run_name"]] = out["primal_solution"]
            if out["full_warmstart"] is not None:
                raw_warmstarts[out["row"]["run_name"]] = out["full_warmstart"]

    if include_extended_dual_decomposition:
        for family, dual_mode in EXTENDED_DUAL_FAMILIES:
            for blocks in all_combos:
                pkg = build_warmstart_package(baseline_ws, blocks, dual_mode=dual_mode)
                out = solve_one_ac(
                    ctx=ctx,
                    family=family,
                    blocks=blocks,
                    init_package=pkg,
                    case_output_dir=case_output_dir,
                    baseline_ws=baseline_ws,
                    tee=tee_others,
                    max_iter=max_iter,
                    tol=tol,
                    shared_case_prep_time=shared_case_prep_time,
                    baseline_objective=baseline_objective,
                    objective_match_tol=objective_match_tol,
                    dual_mode=dual_mode,
                )
                global_rows.append(out["row"])
                if out["primal_solution"] is not None:
                    raw_solutions[out["row"]["run_name"]] = out["primal_solution"]

        for family, dual_mode in DUAL_ONLY_VARIANTS:
            pkg = build_warmstart_package(baseline_ws, (), dual_mode=dual_mode)
            out = solve_one_ac(
                ctx=ctx,
                family=family,
                blocks=(),
                init_package=pkg,
                case_output_dir=case_output_dir,
                baseline_ws=baseline_ws,
                tee=tee_others,
                max_iter=max_iter,
                tol=tol,
                shared_case_prep_time=shared_case_prep_time,
                baseline_objective=baseline_objective,
                objective_match_tol=objective_match_tol,
                dual_mode=dual_mode,
            )
            global_rows.append(out["row"])
            if out["primal_solution"] is not None:
                raw_solutions[out["row"]["run_name"]] = out["primal_solution"]

    dc_out = solve_dcopf(ctx, case_output_dir)
    extra_payload["dcopf"] = dc_out
    if dc_out["success"] and dc_out["solution"] is not None:
        dc_meta_base = {
            "dc_seed_mode": "native_only",
            "dc_solver": dc_out["solver"],
            "dc_solver_problem_type": dc_out["solver_problem_type"],
            "dc_termination": dc_out["termination"],
            "dc_status": dc_out["status"],
            "dc_build_time": dc_out["build_time"],
            "dc_solve_time": dc_out["solve_time"],
            "dc_total_time": dc_out["total_time"],
            "dc_objective": dc_out["objective"],
        }

        for blocks in DC_NATIVE_ONLY_COMBOS:
            pkg = build_dc_init_package(ctx, dc_out["solution"], blocks, vm_policy=None, qg_policy=None)
            out = solve_one_ac(
                ctx=ctx,
                family="dc_seed",
                blocks=blocks,
                init_package=pkg,
                case_output_dir=case_output_dir,
                baseline_ws=baseline_ws,
                tee=tee_others,
                max_iter=max_iter,
                tol=tol,
                vm_policy=None,
                qg_policy=None,
                shared_case_prep_time=shared_case_prep_time,
                baseline_objective=baseline_objective,
                objective_match_tol=objective_match_tol,
                dc_meta=dc_meta_base,
                dual_mode="none",
            )
            global_rows.append(out["row"])
            if out["primal_solution"] is not None:
                raw_solutions[out["row"]["run_name"]] = out["primal_solution"]

        if include_dc_completion_sensitivity:
            dc_meta_completed = dict(dc_meta_base, dc_seed_mode="completed_vm_qg")
            for vm_policy in DC_VM_POLICIES:
                for qg_policy in DC_QG_POLICIES:
                    for blocks in dc_completion_combos():
                        pkg = build_dc_init_package(ctx, dc_out["solution"], blocks, vm_policy=vm_policy, qg_policy=qg_policy)
                        out = solve_one_ac(
                            ctx=ctx,
                            family="dc_seed",
                            blocks=blocks,
                            init_package=pkg,
                            case_output_dir=case_output_dir,
                            baseline_ws=baseline_ws,
                            tee=tee_others,
                            max_iter=max_iter,
                            tol=tol,
                            vm_policy=vm_policy,
                            qg_policy=qg_policy,
                            shared_case_prep_time=shared_case_prep_time,
                            baseline_objective=baseline_objective,
                            objective_match_tol=objective_match_tol,
                            dc_meta=dc_meta_completed,
                            dual_mode="none",
                        )
                        global_rows.append(out["row"])
                        if out["primal_solution"] is not None:
                            raw_solutions[out["row"]["run_name"]] = out["primal_solution"]

    case_runs_csv = save_csv(global_rows, os.path.join(case_output_dir, f"{case_name}__runs.csv"))
    case_solutions_npz = save_case_solution_npz(case_output_dir, case_name, raw_solutions)
    payload = {
        "created_utc": now_utc_iso(),
        "case_name": case_name,
        "case_file": case_file,
        "case_sha256": case_sha256,
        "shared_case_prep_time": shared_case_prep_time,
        "baseline_warmstart": baseline_ws,
        "rows": global_rows,
        "raw_solutions": raw_solutions,
        "raw_warmstarts": raw_warmstarts,
        "extra": extra_payload,
    }
    case_payload_json = save_case_json_payload(case_output_dir, case_name, payload)

    manifest = {
        "case_name": case_name,
        "case_file": case_file,
        "case_sha256": case_sha256,
        "shared_case_prep_time": shared_case_prep_time,
        "baseline_success": True,
        "n_runs": len(global_rows),
        "n_successful_runs": sum(r.get("objective") is not None for r in global_rows),
        "n_runs_matching_baseline_objective": sum(r.get("objective_matches_baseline") == 1 for r in global_rows if r["family"] != "baseline"),
        "dc_success": bool(dc_out.get("success")),
        "case_output_dir": case_output_dir,
        "case_runs_csv": case_runs_csv,
        "case_solutions_npz": case_solutions_npz,
        "case_payload_json_gz": case_payload_json,
    }
    save_json(manifest, os.path.join(case_output_dir, f"{case_name}__manifest.json"))
    return {"case_name": case_name, "case_rows": global_rows, "case_manifest": manifest}


# =============================================================================
# BATCH PIPELINE
# =============================================================================
def run_batch_study(case_root=None, output_dir=DEFAULT_OUTPUT_DIR, tee_baseline=False, tee_others=False,
                    max_iter=300, tol=1e-6,
                    include_dc_completion_sensitivity=DEFAULT_INCLUDE_DC_COMPLETION_SENSITIVITY,
                    include_extended_dual_decomposition=DEFAULT_INCLUDE_EXTENDED_DUAL_DECOMPOSITION,
                    objective_match_tol=DEFAULT_OBJECTIVE_MATCH_TOL):
    ensure_dir(output_dir)
    cases = discover_case_files(case_root)
    if not cases:
        raise FileNotFoundError(f"No .mat cases found under {case_root if case_root else DEFAULT_CASE_DIRS}.")

    all_runs = []
    all_case_results = {}
    all_cases_rows = []
    manifests = []
    failed_cases = []

    print(f"Discovered {len(cases)} case(s).")
    if include_extended_dual_decomposition:
        print("Extended dual decomposition experiments ENABLED.")
    for i, case_file in enumerate(cases, 1):
        print("\n" + "=" * 100)
        print(f"[{i}/{len(cases)}] Running case: {case_file}")
        print("=" * 100)
        try:
            result = run_case_study(
                case_file,
                output_dir,
                tee_baseline=tee_baseline,
                tee_others=tee_others,
                max_iter=max_iter,
                tol=tol,
                include_dc_completion_sensitivity=include_dc_completion_sensitivity,
                include_extended_dual_decomposition=include_extended_dual_decomposition,
                objective_match_tol=objective_match_tol,
            )
            case_name = result["case_name"]
            case_rows = result["case_rows"]
            all_runs.extend(case_rows)
            all_case_results[case_name] = case_rows
            manifests.append(result["case_manifest"])

            if case_rows:
                first = case_rows[0]
                all_cases_rows.append({
                    "case_name": case_name,
                    "case_file": case_file,
                    "baseMVA": first.get("baseMVA"),
                    "nb": first.get("nb"),
                    "nl": first.get("nl"),
                    "ng_total": first.get("ng_total"),
                    "ng_active": first.get("ng_active"),
                    "n_load_buses": first.get("n_load_buses"),
                    "n_ref_buses": first.get("n_ref_buses"),
                    "n_pv_buses": first.get("n_pv_buses"),
                    "n_pq_buses": first.get("n_pq_buses"),
                    "total_pd_mw": first.get("total_pd_mw"),
                    "total_qd_mvar": first.get("total_qd_mvar"),
                    "sum_pmax_mw": first.get("sum_pmax_mw"),
                    "n_branch_with_limits": first.get("n_branch_with_limits"),
                    "n_branch_out_of_service": first.get("n_branch_out_of_service"),
                })
        except Exception as e:
            failed_cases.append({
                "case_file": case_file,
                "error": str(e),
                "traceback": traceback.format_exc(),
            })

    per_case_summary = aggregate_per_case_summary(all_case_results)
    global_summary = aggregate_global_summary(all_runs)
    combo_summary = aggregate_combo_summary(all_runs)

    # Block effects for all oracle families present in the data
    all_oracle_families = sorted({r["family"] for r in all_runs if r["family"].startswith("oracle_ac")})
    global_block_effects = []
    for fam in all_oracle_families:
        global_block_effects.extend(aggregate_block_effects(all_runs, fam))
    global_block_effects.extend(aggregate_block_effects(all_runs, "dc_seed"))

    dc_policy_summary = aggregate_dc_policy_summary(all_runs)
    paper_table_main = aggregate_paper_table_rows(per_case_summary)
    family_wins = aggregate_family_wins(per_case_summary)

    paths = {
        "all_runs_csv": save_csv(all_runs, os.path.join(output_dir, "all_runs.csv")),
        "all_cases_csv": save_csv(all_cases_rows, os.path.join(output_dir, "all_cases.csv"), fieldnames=[
            "case_name", "case_file", "baseMVA", "nb", "nl", "ng_total", "ng_active",
            "n_load_buses", "n_ref_buses", "n_pv_buses", "n_pq_buses", "total_pd_mw", "total_qd_mvar",
            "sum_pmax_mw", "n_branch_with_limits", "n_branch_out_of_service"
        ]),
        "per_case_summary_csv": save_csv(per_case_summary, os.path.join(output_dir, "per_case_summary.csv")),
        "global_summary_csv": save_csv(global_summary, os.path.join(output_dir, "global_summary.csv")),
        "combo_summary_csv": save_csv(combo_summary, os.path.join(output_dir, "combo_summary.csv")),
        "global_block_effects_csv": save_csv(global_block_effects, os.path.join(output_dir, "global_block_effects.csv")),
        "dc_policy_summary_csv": save_csv(dc_policy_summary, os.path.join(output_dir, "dc_policy_summary.csv")),
        "paper_table_main_csv": save_csv(paper_table_main, os.path.join(output_dir, "paper_table_main.csv")),
        "family_wins_csv": save_csv(family_wins, os.path.join(output_dir, "family_wins.csv")),
        "failed_cases_json": save_json(failed_cases, os.path.join(output_dir, "failed_cases.json")),
    }

    manifest = {
        "created_utc": now_utc_iso(),
        "case_root": case_root,
        "n_discovered_cases": len(cases),
        "n_successful_case_manifests": len(manifests),
        "n_failed_cases": len(failed_cases),
        "config": {
            "tee_baseline": tee_baseline,
            "tee_others": tee_others,
            "max_iter": max_iter,
            "tol": tol,
            "objective_match_tol": objective_match_tol,
            "dc_native_only_combos": [list(c) for c in DC_NATIVE_ONLY_COMBOS],
            "dc_vm_policies": list(DC_VM_POLICIES),
            "dc_qg_policies": list(DC_QG_POLICIES),
            "include_dc_completion_sensitivity": include_dc_completion_sensitivity,
            "include_extended_dual_decomposition": include_extended_dual_decomposition,
        },
        "counts": {
            "n_all_runs": len(all_runs),
            "n_successful_runs": sum(row_is_successful(r) for r in all_runs),
            "n_successful_baseline_runs": sum(r["family"] == "baseline" and row_is_successful(r) for r in all_runs),
            "n_successful_oracle_po_runs": sum(r["family"] == "oracle_ac_primal_only" and row_is_successful(r) for r in all_runs),
            "n_successful_oracle_pd_runs": sum(r["family"] == "oracle_ac_primal_dual" and row_is_successful(r) for r in all_runs),
            "n_successful_oracle_cd_runs": sum(r["family"] == "oracle_ac_constraint_dual" and row_is_successful(r) for r in all_runs),
            "n_successful_oracle_bd_runs": sum(r["family"] == "oracle_ac_bounds_dual" and row_is_successful(r) for r in all_runs),
            "n_successful_oracle_pd_ab_runs": sum(r["family"] == "oracle_ac_primal_dual_all_bounds" and row_is_successful(r) for r in all_runs),
            "n_successful_dual_only_runs": sum(r["family"].startswith("oracle_ac_dual_only") and row_is_successful(r) for r in all_runs),
            "n_successful_dc_runs": sum(r["family"] == "dc_seed" and row_is_successful(r) for r in all_runs),
        },
        "files": paths,
        "case_manifests": manifests,
    }

    manifest_path = save_json(manifest, os.path.join(output_dir, "manifest.json"))
    readme_path = write_readme(output_dir, case_root, len(cases), manifest["config"])

    print("\nBatch study completed.")
    print(f"Saved runs to: {paths['all_runs_csv']}")
    print(f"Saved manifest to: {manifest_path}")
    print(f"Saved README to: {readme_path}")
    print("Counts:")
    for k, v in manifest["counts"].items():
        print(f"  - {k}: {v}")

    return {
        "all_runs": all_runs,
        "per_case_summary": per_case_summary,
        "global_summary": global_summary,
        "combo_summary": combo_summary,
        "global_block_effects": global_block_effects,
        "dc_policy_summary": dc_policy_summary,
        "paper_table_main": paper_table_main,
        "family_wins": family_wins,
        "manifest": manifest,
    }


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Batch ACOPF warm-start study with oracle AC starts and DC-seeded practical starts."
    )
    parser.add_argument("case_root", nargs="?", default=None,
                        help="Root folder containing MATPOWER .mat cases. Defaults to searching test_cases and testt_cases.")
    parser.add_argument("output_dir", nargs="?", default=DEFAULT_OUTPUT_DIR,
                        help="Directory where outputs will be written.")
    parser.add_argument("--tee-baseline", action="store_true", default=False,
                        help="Print IPOPT output for baseline runs.")
    parser.add_argument("--tee-others", action="store_true", default=False,
                        help="Print solver output for non-baseline runs.")
    parser.add_argument("--max-iter", type=int, default=300,
                        help="IPOPT max_iter for ACOPF runs.")
    parser.add_argument("--tol", type=float, default=1e-6,
                        help="IPOPT tolerance for ACOPF runs.")
    parser.add_argument("--objective-match-tol", type=float, default=DEFAULT_OBJECTIVE_MATCH_TOL,
                        help="Tolerance used when flagging whether a run matches the baseline objective.")
    parser.add_argument(
        "--include-dc-completion-sensitivity",
        action="store_true",
        default=DEFAULT_INCLUDE_DC_COMPLETION_SENSITIVITY,
        help="Also run DC-seeded AC starts that override Vm and/or Qg using completion policies."
    )
    parser.add_argument(
        "--include-extended-dual-decomposition",
        action="store_true",
        default=DEFAULT_INCLUDE_EXTENDED_DUAL_DECOMPOSITION,
        help="Run extended dual decomposition experiments: constraint-duals-only, bounds-only, "
             "full-all-bounds, and dual-only (no primal) variants. Adds 48 runs per case."
    )

    args = parser.parse_args()

    run_batch_study(
        case_root=args.case_root,
        output_dir=args.output_dir,
        tee_baseline=args.tee_baseline,
        tee_others=args.tee_others,
        max_iter=args.max_iter,
        tol=args.tol,
        include_dc_completion_sensitivity=args.include_dc_completion_sensitivity,
        include_extended_dual_decomposition=args.include_extended_dual_decomposition,
        objective_match_tol=args.objective_match_tol,
    )
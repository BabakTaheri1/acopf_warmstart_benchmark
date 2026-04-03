import sys
import os
import time
import math
import csv
import re
import itertools
from statistics import mean

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

DEFAULT_CASE = "test_cases_2/pglib_opf_case2000_goc.mat"
DEFAULT_OUTPUT_DIR = "acopf_init_study"
BLOCKS = ("Pg", "Qg", "Vm", "Va")


# =============================================================================
# SMALL UTILITIES
# =============================================================================
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def clamp_to_bounds(var, val):
    lb = var.lb if var.lb is not None else -float("inf")
    ub = var.ub if var.ub is not None else float("inf")
    return min(max(float(val), lb), ub)


def suffix_get(sfx, comp, default=0.0):
    try:
        return float(pe.value(sfx[comp]))
    except Exception:
        return float(default)


def powerset_nonempty(items):
    for r in range(1, len(items) + 1):
        for combo in itertools.combinations(items, r):
            yield combo


def combo_name(blocks):
    return "_".join(blocks) if blocks else "none"


def make_run_name(family, blocks):
    if family == "baseline":
        return "baseline_default"
    return f"{family}__{combo_name(blocks)}"


def success_termination(term):
    return term in {
        pe.TerminationCondition.optimal,
        pe.TerminationCondition.locallyOptimal,
    }


def speedup_pct(baseline_time, new_time):
    if baseline_time is None or new_time is None or baseline_time <= 0:
        return None
    return 100.0 * (baseline_time - new_time) / baseline_time


def safe_float(x, default=None):
    try:
        return float(x)
    except Exception:
        return default


def parse_ipopt_iterations(log_path):
    if not os.path.exists(log_path):
        return None
    try:
        txt = open(log_path, "r", encoding="utf-8", errors="ignore").read()
    except Exception:
        return None

    patterns = [
        r"Number of Iterations\.*:\s*(\d+)",
        r"Number of iterations\.*:\s*(\d+)",
    ]
    for pat in patterns:
        m = re.search(pat, txt)
        if m:
            return int(m.group(1))
    return None


# =============================================================================
# DATA LOADER & PHYSICS
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


def prepare_case_context(case_file):
    baseMVA, bus, gen, branch, gencost, bus_map, Ybus, Yf, Yt = load_and_build_physics(case_file)

    nb = len(bus)
    nl = len(branch)
    active_gens = [i for i in range(len(gen)) if gen[i, GEN_STATUS] > 0]

    ref_candidates = np.where(bus[:, BUS_TYPE] == 3)[0]
    if len(ref_candidates) != 1:
        raise ValueError(
            f"Expected exactly one reference bus, found {len(ref_candidates)} in {case_file}."
        )
    ref_idx = int(ref_candidates[0])

    Ybus_rows = fast_sparse_rows_complex(Ybus, nb)
    Yf_rows = fast_sparse_rows_complex(Yf, nl)
    Yt_rows = fast_sparse_rows_complex(Yt, nl)

    bus_gens = [[] for _ in range(nb)]
    for g in active_gens:
        b = bus_map[int(gen[g, GEN_BUS])]
        bus_gens[b].append(g)

    return {
        "case_file": case_file,
        "baseMVA": baseMVA,
        "bus": bus,
        "gen": gen,
        "branch": branch,
        "gencost": gencost,
        "bus_map": bus_map,
        "nb": nb,
        "nl": nl,
        "active_gens": active_gens,
        "bus_gens": bus_gens,
        "Ybus_rows": Ybus_rows,
        "Yf_rows": Yf_rows,
        "Yt_rows": Yt_rows,
        "Pd": bus[:, PD] / baseMVA,
        "Qd": bus[:, QD] / baseMVA,
        "F_indices": [bus_map[int(x)] for x in branch[:, F_BUS]],
        "T_indices": [bus_map[int(x)] for x in branch[:, T_BUS]],
        "RateA_Sq": (branch[:, RATE_A] / baseMVA) ** 2,
        "ref_idx": ref_idx,
    }


# =============================================================================
# COST HELPERS
# =============================================================================
def parse_gencost_row(row):
    model = int(row[MODEL])
    ncost = int(row[NCOST])

    if model == 2:
        coeffs = [float(x) for x in row[COST:COST + ncost]]
        if len(coeffs) != ncost:
            raise ValueError(f"Malformed polynomial gencost row: expected {ncost} coefficients.")
        return {"model": 2, "ncost": ncost, "coeffs": coeffs}

    if model == 1:
        vals = [float(x) for x in row[COST:COST + 2 * ncost]]
        if len(vals) != 2 * ncost:
            raise ValueError(f"Malformed piecewise-linear gencost row: expected {2 * ncost} values.")
        pts = [(vals[2 * i], vals[2 * i + 1]) for i in range(ncost)]
        if len(pts) < 2:
            raise ValueError("Piecewise-linear gencost row needs at least two points.")
        for i in range(len(pts) - 1):
            if pts[i + 1][0] <= pts[i][0]:
                raise ValueError("Piecewise-linear gencost breakpoints must have strictly increasing p values.")
        slopes = []
        for i in range(len(pts) - 1):
            p1, f1 = pts[i]
            p2, f2 = pts[i + 1]
            slopes.append((f2 - f1) / (p2 - p1))
        for i in range(len(slopes) - 1):
            if slopes[i + 1] < slopes[i] - 1e-10:
                raise ValueError(
                    "Non-convex piecewise-linear gencost row detected. "
                    "This script supports convex PWL costs only."
                )
        segments = []
        for i in range(len(pts) - 1):
            p1, f1 = pts[i]
            p2, f2 = pts[i + 1]
            slope = (f2 - f1) / (p2 - p1)
            intercept = f1 - slope * p1
            segments.append((slope, intercept))
        return {"model": 1, "ncost": ncost, "points": pts, "segments": segments}

    raise ValueError(f"Unsupported gencost model: MODEL={model}")


def evaluate_cost_row_at_pg_mw(cost_spec, pg_mw):
    if cost_spec["model"] == 2:
        coeffs = cost_spec["coeffs"]
        degree = len(coeffs) - 1
        return sum(coeff * (pg_mw ** (degree - i)) for i, coeff in enumerate(coeffs))
    if cost_spec["model"] == 1:
        return max(slope * pg_mw + intercept for slope, intercept in cost_spec["segments"])
    raise ValueError(f"Unsupported cost spec model: {cost_spec['model']}")


# =============================================================================
# WARM-START HELPERS
# =============================================================================
def extract_primal_solution(m):
    return {
        "Vm": {int(b): float(pe.value(m.Vm[b])) for b in m.B},
        "Va": {int(b): float(pe.value(m.Va[b])) for b in m.B},
        "Pg": {int(g): float(pe.value(m.Pg[g])) for g in m.G},
        "Qg": {int(g): float(pe.value(m.Qg[g])) for g in m.G},
    }


def extract_full_warmstart(m):
    dual_eqp = {int(b): suffix_get(m.dual, m.EqP[b], 0.0) for b in m.B}
    dual_eqq = {int(b): suffix_get(m.dual, m.EqQ[b], 0.0) for b in m.B}
    dual_ff = {int(l): suffix_get(m.dual, m.FlowFrom[l], 0.0) for l in m.FlowFrom}
    dual_ft = {int(l): suffix_get(m.dual, m.FlowTo[l], 0.0) for l in m.FlowTo}

    return {
        "primal": {
            "Vm": {int(b): float(pe.value(m.Vm[b])) for b in m.B},
            "Va": {int(b): float(pe.value(m.Va[b])) for b in m.B},
            "Pg": {int(g): float(pe.value(m.Pg[g])) for g in m.G},
            "Qg": {int(g): float(pe.value(m.Qg[g])) for g in m.G},
        },
        "dual": {
            "RefAngle": suffix_get(m.dual, m.RefAngle, 0.0),
            "EqP": dual_eqp,
            "EqQ": dual_eqq,
            "FlowFrom": dual_ff,
            "FlowTo": dual_ft,
        },
        "zL": {
            "Vm": {int(b): suffix_get(m.ipopt_zL_out, m.Vm[b], 0.0) for b in m.B},
            "Va": {int(b): suffix_get(m.ipopt_zL_out, m.Va[b], 0.0) for b in m.B},
            "Pg": {int(g): suffix_get(m.ipopt_zL_out, m.Pg[g], 0.0) for g in m.G},
            "Qg": {int(g): suffix_get(m.ipopt_zL_out, m.Qg[g], 0.0) for g in m.G},
        },
        "zU": {
            "Vm": {int(b): suffix_get(m.ipopt_zU_out, m.Vm[b], 0.0) for b in m.B},
            "Va": {int(b): suffix_get(m.ipopt_zU_out, m.Va[b], 0.0) for b in m.B},
            "Pg": {int(g): suffix_get(m.ipopt_zU_out, m.Pg[g], 0.0) for g in m.G},
            "Qg": {int(g): suffix_get(m.ipopt_zU_out, m.Qg[g], 0.0) for g in m.G},
        },
    }


def build_warmstart_package(full_ws, blocks, include_duals=False):
    pkg = {
        "primal": {blk: full_ws["primal"][blk] for blk in blocks},
        "dual": {},
        "zL": {},
        "zU": {},
    }
    if include_duals:
        pkg["dual"] = full_ws["dual"]
        pkg["zL"] = {blk: full_ws["zL"][blk] for blk in blocks}
        pkg["zU"] = {blk: full_ws["zU"][blk] for blk in blocks}
    return pkg


def apply_primal_start(m, init_package, ref_idx=None):
    if not init_package:
        return

    primal = init_package.get("primal", {})

    if "Vm" in primal:
        for b, val in primal["Vm"].items():
            if b in m.B:
                m.Vm[b].set_value(clamp_to_bounds(m.Vm[b], val))

    if "Va" in primal:
        for b, val in primal["Va"].items():
            if b in m.B:
                m.Va[b].set_value(clamp_to_bounds(m.Va[b], val))

    if "Pg" in primal:
        for g, val in primal["Pg"].items():
            if g in m.G:
                m.Pg[g].set_value(clamp_to_bounds(m.Pg[g], val))

    if "Qg" in primal:
        for g, val in primal["Qg"].items():
            if g in m.G:
                m.Qg[g].set_value(clamp_to_bounds(m.Qg[g], val))

    if ref_idx is not None and ref_idx in m.B:
        m.Va[ref_idx].set_value(0.0)


def apply_dual_start(m, init_package):
    if not init_package:
        return

    dual = init_package.get("dual", {})
    zL = init_package.get("zL", {})
    zU = init_package.get("zU", {})

    if "RefAngle" in dual:
        m.dual[m.RefAngle] = float(dual["RefAngle"])

    for b, val in dual.get("EqP", {}).items():
        if b in m.B:
            m.dual[m.EqP[b]] = float(val)

    for b, val in dual.get("EqQ", {}).items():
        if b in m.B:
            m.dual[m.EqQ[b]] = float(val)

    for l, val in dual.get("FlowFrom", {}).items():
        if l in m.FlowFrom:
            m.dual[m.FlowFrom[l]] = float(val)

    for l, val in dual.get("FlowTo", {}).items():
        if l in m.FlowTo:
            m.dual[m.FlowTo[l]] = float(val)

    for name, varcomp in [("Vm", m.Vm), ("Va", m.Va), ("Pg", m.Pg), ("Qg", m.Qg)]:
        for i, val in zL.get(name, {}).items():
            if i in varcomp:
                m.ipopt_zL_in[varcomp[i]] = float(val)

    for name, varcomp in [("Vm", m.Vm), ("Va", m.Va), ("Pg", m.Pg), ("Qg", m.Qg)]:
        for i, val in zU.get(name, {}).items():
            if i in varcomp:
                m.ipopt_zU_in[varcomp[i]] = float(val)


# =============================================================================
# MODEL BUILDER
# =============================================================================
def build_acopf_model(ctx, init_package=None):
    baseMVA = ctx["baseMVA"]
    bus = ctx["bus"]
    gen = ctx["gen"]
    branch = ctx["branch"]
    gencost = ctx["gencost"]
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

    m = pe.ConcreteModel(name="ACOPF_WarmStart_Study")

    m.B = pe.RangeSet(0, nb - 1)
    m.L = pe.RangeSet(0, nl - 1)
    m.G = pe.Set(initialize=active_gens, ordered=True)

    # IPOPT warm-start suffixes
    m.ipopt_zL_out = pe.Suffix(direction=pe.Suffix.IMPORT)
    m.ipopt_zU_out = pe.Suffix(direction=pe.Suffix.IMPORT)
    m.ipopt_zL_in = pe.Suffix(direction=pe.Suffix.EXPORT)
    m.ipopt_zU_in = pe.Suffix(direction=pe.Suffix.EXPORT)
    m.dual = pe.Suffix(direction=pe.Suffix.IMPORT_EXPORT)

    # Variables
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
        bounds=lambda m, g: (
            float(gen[g, PMIN] / baseMVA),
            float(gen[g, PMAX] / baseMVA),
        ),
    )

    m.Qg = pe.Var(
        m.G,
        initialize=lambda m, g: float(gen[g, QG] / baseMVA),
        bounds=lambda m, g: (
            float(gen[g, QMIN] / baseMVA),
            float(gen[g, QMAX] / baseMVA),
        ),
    )

    # Cost support:
    # - polynomial costs of arbitrary degree
    # - convex piecewise-linear costs via epigraph variables
    cost_specs = {int(g): parse_gencost_row(gencost[g]) for g in active_gens}
    pwl_gens = [g for g in active_gens if cost_specs[g]["model"] == 1]

    if pwl_gens:
        m.PWLGenCost = pe.Var(
            pwl_gens,
            initialize=lambda m, g: evaluate_cost_row_at_pg_mw(
                cost_specs[g], float(gen[g, PG])
            )
        )
        m.PWLCostConstr = pe.ConstraintList()
        for g in pwl_gens:
            pg_mw = m.Pg[g] * baseMVA
            for slope, intercept in cost_specs[g]["segments"]:
                m.PWLCostConstr.add(m.PWLGenCost[g] >= slope * pg_mw + intercept)

    def obj_rule(m):
        cost = 0.0
        for g in m.G:
            spec = cost_specs[int(g)]
            pg_mw = m.Pg[g] * baseMVA
            if spec["model"] == 2:
                coeffs = spec["coeffs"]
                degree = len(coeffs) - 1
                cost += sum(coeff * (pg_mw ** (degree - i)) for i, coeff in enumerate(coeffs))
            elif spec["model"] == 1:
                cost += m.PWLGenCost[g]
            else:
                raise ValueError(f"Unsupported cost model for generator {g}: {spec['model']}")
        return cost

    m.Obj = pe.Objective(rule=obj_rule, sense=pe.minimize)

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

    def I_re_im_from_row(m, l, y_rows):
        Ire = 0.0
        Iim = 0.0
        for k, G, B in y_rows[l]:
            vr = Vre(m, k)
            vi = Vim(m, k)
            Ire += G * vr - B * vi
            Iim += B * vr + G * vi
        return Ire, Iim

    def S_sq_from(m, l):
        Ire, Iim = I_re_im_from_row(m, l, Yf_rows)
        fbus = F_indices[l]
        vr = Vre(m, fbus)
        vi = Vim(m, fbus)
        Pf = vr * Ire + vi * Iim
        Qf = vi * Ire - vr * Iim
        return Pf * Pf + Qf * Qf

    def S_sq_to(m, l):
        Ire, Iim = I_re_im_from_row(m, l, Yt_rows)
        tbus = T_indices[l]
        vr = Vre(m, tbus)
        vi = Vim(m, tbus)
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


# =============================================================================
# SOLVER
# =============================================================================
def configure_ipopt(solver, tol=1e-6, max_iter=300, print_level=5, use_dual_warmstart=False):
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


def solve_one(
    ctx,
    family,
    blocks,
    init_package,
    output_dir,
    tee=False,
    tol=1e-6,
    max_iter=300,
    baseline_objective=None,
    objective_match_abs_tol=1e-5,
    objective_match_rel_tol=1e-6,
):
    run_name = make_run_name(family, blocks)
    print(f"\n--- {run_name} ---")

    total_start = time.time()

    build_start = time.time()
    model = build_acopf_model(ctx, init_package=init_package)
    build_time = time.time() - build_start

    solver = pe.SolverFactory("ipopt")
    if solver is None:
        raise RuntimeError("Could not create IPOPT solver.")

    use_dual_warmstart = (family == "primal_dual")
    configure_ipopt(
        solver,
        tol=tol,
        max_iter=max_iter,
        print_level=5,
        use_dual_warmstart=use_dual_warmstart,
    )

    logs_dir = ensure_dir(os.path.join(output_dir, "logs"))
    log_path = os.path.join(logs_dir, f"{run_name}.log")

    solve_start = time.time()
    try:
        results = solver.solve(model, tee=tee, logfile=log_path)
    except TypeError:
        results = solver.solve(model, tee=tee)
    solve_time = time.time() - solve_start
    total_time = time.time() - total_start

    term = results.solver.termination_condition
    term_str = str(term)
    status_str = str(results.solver.status)
    iters = parse_ipopt_iterations(log_path)

    objective = None
    primal_solution = None
    full_warmstart = None
    objective_gap_abs = None
    objective_gap_rel = None
    objective_matches_baseline = None

    if success_termination(term):
        objective = float(pe.value(model.Obj))
        primal_solution = extract_primal_solution(model)
        full_warmstart = extract_full_warmstart(model)

        if baseline_objective is not None:
            objective_gap_abs = abs(objective - baseline_objective)
            denom = max(abs(baseline_objective), 1.0)
            objective_gap_rel = objective_gap_abs / denom
            objective_matches_baseline = int(
                objective_gap_abs <= objective_match_abs_tol + objective_match_rel_tol * max(1.0, abs(baseline_objective))
            )

        print(f"✅ {run_name}: {term_str}")
        print(f"   objective = {objective:.6f}")
        print(f"   iterations = {iters}")
        if objective_gap_abs is not None:
            print(
                f"   obj gap vs baseline = {objective_gap_abs:.6e} "
                f"(rel={objective_gap_rel:.6e}, match={objective_matches_baseline})"
            )
        print(f"   build = {build_time:.3f}s | solve = {solve_time:.3f}s | total = {total_time:.3f}s")
    else:
        print(f"⚠️ {run_name}: {term_str}")
        print(f"   iterations = {iters}")
        print(f"   build = {build_time:.3f}s | solve = {solve_time:.3f}s | total = {total_time:.3f}s")

    return {
        "run_name": run_name,
        "family": family,
        "blocks": tuple(blocks),
        "num_blocks": len(blocks),
        "termination": term_str,
        "status": status_str,
        "objective": objective,
        "objective_gap_abs_vs_baseline": objective_gap_abs,
        "objective_gap_rel_vs_baseline": objective_gap_rel,
        "objective_matches_baseline": objective_matches_baseline,
        "iterations": iters,
        "build_time": build_time,
        "solve_time": solve_time,
        "total_time": total_time,
        "primal_solution": primal_solution,
        "full_warmstart": full_warmstart,
        "log_path": log_path,
    }


# =============================================================================
# REPORTING
# =============================================================================
def save_results_csv(results, output_dir):
    path = os.path.join(output_dir, "acopf_init_study_results.csv")
    fields = [
        "run_name",
        "family",
        "blocks",
        "num_blocks",
        "termination",
        "status",
        "objective",
        "objective_gap_abs_vs_baseline",
        "objective_gap_rel_vs_baseline",
        "objective_matches_baseline",
        "iterations",
        "build_time",
        "solve_time",
        "total_time",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in results:
            row = {k: r.get(k) for k in fields}
            row["blocks"] = "+".join(r["blocks"]) if r["blocks"] else ""
            writer.writerow(row)
    return path


def compute_block_effects(results, family):
    rows = []
    fam = [
        r for r in results
        if r["family"] == family
        and r["objective"] is not None
        and r.get("objective_matches_baseline") in (None, 1)
    ]
    if not fam:
        return rows

    for blk in BLOCKS:
        with_blk = [r["solve_time"] for r in fam if blk in r["blocks"]]
        without_blk = [r["solve_time"] for r in fam if blk not in r["blocks"]]

        rows.append({
            "family": family,
            "block": blk,
            "n_with": len(with_blk),
            "n_without": len(without_blk),
            "avg_solve_with": mean(with_blk) if with_blk else None,
            "avg_solve_without": mean(without_blk) if without_blk else None,
            "delta_without_minus_with": (
                mean(without_blk) - mean(with_blk)
                if with_blk and without_blk else None
            ),
            "pct_better_when_included": (
                100.0 * (mean(without_blk) - mean(with_blk)) / mean(without_blk)
                if with_blk and without_blk and mean(without_blk) > 0 else None
            ),
        })
    return rows


def save_block_effects_csv(rows, output_dir):
    path = os.path.join(output_dir, "acopf_init_block_effects.csv")
    fields = [
        "family",
        "block",
        "n_with",
        "n_without",
        "avg_solve_with",
        "avg_solve_without",
        "delta_without_minus_with",
        "pct_better_when_included",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def print_summary(results):
    print("\n" + "=" * 150)
    print(
        f"{'RUN':<30} {'FAMILY':<12} {'ITER':>7} {'OBJ':>16} "
        f"{'OBJ_GAP':>12} {'MATCH':>7} {'SOLVE(s)':>12} {'TOTAL(s)':>12} {'TERM':>16}"
    )
    print("-" * 150)
    for r in results:
        obj = "None" if r["objective"] is None else f"{r['objective']:.6f}"
        iters = "None" if r["iterations"] is None else str(r["iterations"])
        gap = "None" if r["objective_gap_abs_vs_baseline"] is None else f"{r['objective_gap_abs_vs_baseline']:.2e}"
        match = "None" if r["objective_matches_baseline"] is None else str(r["objective_matches_baseline"])
        print(
            f"{r['run_name']:<30} "
            f"{r['family']:<12} "
            f"{iters:>7} "
            f"{obj:>16} "
            f"{gap:>12} "
            f"{match:>7} "
            f"{r['solve_time']:>12.4f} "
            f"{r['total_time']:>12.4f} "
            f"{r['termination']:>16}"
        )
    print("=" * 150)


def print_top_runs(results, baseline):
    good = [
        r for r in results
        if r["objective"] is not None
        and r["family"] != "baseline"
        and r.get("objective_matches_baseline") == 1
    ]
    if not good:
        print("\nNo successful non-baseline runs matched the baseline objective within tolerance.")
        return

    baseline_solve = baseline["solve_time"]

    good_sorted = sorted(good, key=lambda r: r["solve_time"])
    print("\nBest objective-matching runs by solve time:")
    for r in good_sorted[:10]:
        spd = speedup_pct(baseline_solve, r["solve_time"])
        blk = "+".join(r["blocks"])
        print(
            f"  {r['run_name']:<30} solve={r['solve_time']:.4f}s "
            f"iter={r['iterations']} speedup_vs_baseline={spd:.2f}% blocks={blk}"
        )


def print_family_summary(results, baseline):
    for fam in ("primal_only", "primal_dual"):
        fam_rows = [
            r for r in results
            if r["family"] == fam
            and r["objective"] is not None
            and r.get("objective_matches_baseline") == 1
        ]
        if not fam_rows:
            print(f"\nFamily: {fam}")
            print("  No successful runs matched the baseline objective within tolerance.")
            continue

        best = min(fam_rows, key=lambda r: r["solve_time"])
        avg_solve = mean(r["solve_time"] for r in fam_rows)
        avg_iters = (
            mean(r["iterations"] for r in fam_rows if r["iterations"] is not None)
            if any(r["iterations"] is not None for r in fam_rows)
            else None
        )
        spd = speedup_pct(baseline["solve_time"], best["solve_time"])
        print(f"\nFamily: {fam}")
        print(f"  best run        : {best['run_name']}")
        print(f"  best solve time : {best['solve_time']:.4f}s")
        print(f"  best iterations : {best['iterations']}")
        print(f"  speedup vs base : {spd:.2f}%")
        print(f"  avg solve time  : {avg_solve:.4f}s")
        print(f"  avg iterations  : {avg_iters}")


def print_block_effects(rows):
    if not rows:
        return
    rows = sorted(
        [r for r in rows if r["pct_better_when_included"] is not None],
        key=lambda x: x["pct_better_when_included"],
        reverse=True,
    )
    for fam in ("primal_only", "primal_dual"):
        fam_rows = [r for r in rows if r["family"] == fam]
        if not fam_rows:
            continue
        print(f"\nAverage inclusion effect for {fam}:")
        for r in fam_rows:
            print(
                f"  {r['block']}: avg_with={r['avg_solve_with']:.4f}s, "
                f"avg_without={r['avg_solve_without']:.4f}s, "
                f"benefit={r['pct_better_when_included']:.2f}%"
            )


def print_conclusion(results, block_rows):
    baseline = next(r for r in results if r["family"] == "baseline")
    good = [r for r in results if r["objective"] is not None]

    if len(good) <= 1:
        print("\nNo successful non-baseline runs, so no conclusion can be drawn.")
        return

    primal_only_rows = [
        r for r in good
        if r["family"] == "primal_only" and r.get("objective_matches_baseline") == 1
    ]
    primal_dual_rows = [
        r for r in good
        if r["family"] == "primal_dual" and r.get("objective_matches_baseline") == 1
    ]
    good_matching = [
        r for r in good
        if r["family"] != "baseline" and r.get("objective_matches_baseline") == 1
    ]

    print("\n" + "#" * 100)
    print("CONCLUSION")
    print("#" * 100)
    print(f"Baseline solve time: {baseline['solve_time']:.4f}s, iterations: {baseline['iterations']}")

    if primal_only_rows:
        best_po = min(primal_only_rows, key=lambda r: r["solve_time"])
        print(
            f"Best primal-only run : {best_po['run_name']} | "
            f"{best_po['solve_time']:.4f}s | iter={best_po['iterations']} | "
            f"speedup={speedup_pct(baseline['solve_time'], best_po['solve_time']):.2f}%"
        )

    if primal_dual_rows:
        best_pd = min(primal_dual_rows, key=lambda r: r["solve_time"])
        print(
            f"Best primal+dual run : {best_pd['run_name']} | "
            f"{best_pd['solve_time']:.4f}s | iter={best_pd['iterations']} | "
            f"speedup={speedup_pct(baseline['solve_time'], best_pd['solve_time']):.2f}%"
        )

    if good_matching:
        best_all = min(good_matching, key=lambda r: r["solve_time"])
        print(
            f"Best overall objective-matching non-baseline run : {best_all['run_name']} | "
            f"{best_all['solve_time']:.4f}s | iter={best_all['iterations']}"
        )
    else:
        print("No successful non-baseline run matched the baseline objective within tolerance.")

    for fam in ("primal_only", "primal_dual"):
        fam_rows = [
            r for r in block_rows
            if r["family"] == fam and r["pct_better_when_included"] is not None
        ]
        if fam_rows:
            best_blk = max(fam_rows, key=lambda r: r["pct_better_when_included"])
            print(
                f"Most helpful block on average in {fam}: "
                f"{best_blk['block']} ({best_blk['pct_better_when_included']:.2f}% average solve-time benefit)"
            )

    print("\nInterpretation tips:")
    print("  1) Prefer solve_time over total_time for restart-quality conclusions.")
    print("  2) Objective-match checks matter because ACOPF is nonconvex.")
    print("  3) If Vm/Va dominate, state initialization matters most.")
    print("  4) If primal_dual consistently beats primal_only, full IPOPT warm starts are helping.")


# =============================================================================
# FULL PIPELINE
# =============================================================================
def run_full_study(
    case_file,
    output_dir=DEFAULT_OUTPUT_DIR,
    tee_baseline=False,
    tee_others=False,
    tol=1e-6,
    max_iter=300,
    objective_match_abs_tol=1e-5,
    objective_match_rel_tol=1e-6,
):
    ensure_dir(output_dir)

    # Build case context once so per-run timing is not contaminated by repeated file I/O
    # and physics reconstruction.
    ctx = prepare_case_context(case_file)

    # -------------------------------------------------------------------------
    # Step 1: Baseline solve from default case-data initialization
    # -------------------------------------------------------------------------
    baseline = solve_one(
        ctx=ctx,
        family="baseline",
        blocks=(),
        init_package=None,
        output_dir=output_dir,
        tee=tee_baseline,
        tol=tol,
        max_iter=max_iter,
        baseline_objective=None,
        objective_match_abs_tol=objective_match_abs_tol,
        objective_match_rel_tol=objective_match_rel_tol,
    )
    results = [baseline]

    if baseline["full_warmstart"] is None:
        print("\nBaseline did not solve successfully. Aborting study.")
        print_summary(results)
        return results

    full_ws = baseline["full_warmstart"]
    baseline_objective = baseline["objective"]

    # -------------------------------------------------------------------------
    # Step 2: All non-empty subsets of {Pg, Qg, Vm, Va}
    # -------------------------------------------------------------------------
    combos = list(powerset_nonempty(BLOCKS))

    for blocks in combos:
        pkg = build_warmstart_package(full_ws, blocks, include_duals=False)
        r = solve_one(
            ctx=ctx,
            family="primal_only",
            blocks=blocks,
            init_package=pkg,
            output_dir=output_dir,
            tee=tee_others,
            tol=tol,
            max_iter=max_iter,
            baseline_objective=baseline_objective,
            objective_match_abs_tol=objective_match_abs_tol,
            objective_match_rel_tol=objective_match_rel_tol,
        )
        results.append(r)

    for blocks in combos:
        pkg = build_warmstart_package(full_ws, blocks, include_duals=True)
        r = solve_one(
            ctx=ctx,
            family="primal_dual",
            blocks=blocks,
            init_package=pkg,
            output_dir=output_dir,
            tee=tee_others,
            tol=tol,
            max_iter=max_iter,
            baseline_objective=baseline_objective,
            objective_match_abs_tol=objective_match_abs_tol,
            objective_match_rel_tol=objective_match_rel_tol,
        )
        results.append(r)

    # -------------------------------------------------------------------------
    # Step 3: Save and report
    # -------------------------------------------------------------------------
    results_csv = save_results_csv(results, output_dir)
    block_rows = (
        compute_block_effects(results, "primal_only")
        + compute_block_effects(results, "primal_dual")
    )
    block_csv = save_block_effects_csv(block_rows, output_dir)

    print_summary(results)
    print_top_runs(results, baseline)
    print_family_summary(results, baseline)
    print_block_effects(block_rows)
    print_conclusion(results, block_rows)

    print(f"\nSaved detailed results to: {results_csv}")
    print(f"Saved block-effect summary to: {block_csv}")
    print(f"Solver logs are in: {os.path.join(output_dir, 'logs')}")

    return results


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CASE
    run_full_study(
        case_file=target,
        output_dir=DEFAULT_OUTPUT_DIR,
        tee_baseline=False,
        tee_others=False,
        tol=1e-6,
        max_iter=300,
        objective_match_abs_tol=1e-5,
        objective_match_rel_tol=1e-6,
    )
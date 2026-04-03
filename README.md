# Not All Warm Starts Help: Benchmarking Primal-Dual Initialization in ACOPF

Benchmark suite for evaluating warm-start initialization strategies in AC Optimal Power Flow (ACOPF), accompanying the paper:

> **Not All Warm Starts Help: Benchmarking Primal-Dual Initialization in ACOPF**

## Overview

This code runs a systematic comparison of ACOPF initialization strategies across PGLib-OPF benchmark instances. The default configuration (with extended dual decomposition enabled) evaluates the following families:

- **Baseline**: native case-data initialization (the default MATPOWER starting point)
- **Oracle AC primal-only**: all 15 non-empty subsets of {Pg, Qg, Vm, Va} from a converged AC solution, with IPOPT initializing duals from scratch
- **Oracle AC primal-plus-dual**: same 15 primal subsets plus full dual variables from the converged solution
- **Oracle AC constraint-dual-only**: all 15 primal subsets with constraint duals only (bound multipliers initialized by IPOPT)
- **Oracle AC bounds-dual-only**: all 15 primal subsets with bound multipliers only (constraint duals initialized by IPOPT)
- **Oracle AC primal-dual (all bounds)**: all 15 primal subsets plus constraint duals and all bound multipliers
- **Oracle AC dual-only (constraint)**: empty primal set with constraint duals only
- **Oracle AC dual-only (bounds)**: empty primal set with all bound multipliers only
- **Oracle AC dual-only (full)**: empty primal set with full duals (constraint + all bounds)
- **DC-seeded**: Pg, Va, and Pg+Va from a DCOPF solve injected into the AC initial point

With extended dual decomposition enabled (the default), each case produces 82 solver runs
(1 baseline + 15 primal-only + 15 primal-plus-dual + 15 constraint-dual-only + 15 bounds-dual-only
+ 15 primal-dual-all-bounds + 3 dual-only variants + 3 DC-seeded).
## Requirements

- Python 3.9+
- NumPy
- SciPy (including `scipy.io` for MATPOWER `.mat` file loading)
- Pyomo 6.x
- IPOPT (built with MUMPS linear solver recommended)
- PGLib-OPF test cases in MATPOWER `.mat` format

Install Python dependencies:
```bash
pip install numpy scipy pyomo
```

IPOPT must be installed separately and available on your PATH. See the [IPOPT installation guide](https://coin-or.github.io/Ipopt/INSTALL.html).

## Test Cases

Download PGLib-OPF benchmark cases from [https://github.com/power-grid-lib/pglib-opf](https://github.com/power-grid-lib/pglib-opf) and convert to MATPOWER `.mat` format. Place `.mat` files in a directory (default search paths: `test_cases/`).

## Usage

Basic run with default settings (extended dual decomposition enabled):
```bash
python acopf_batch_study.py path/to/mat_cases/
```

Full options:
```bash
python acopf_batch_study.py path/to/mat_cases/ output_dir/ \
    --max-iter 300 \
    --tol 1e-6 \
    --objective-match-tol 1e-5 \
    --include-dc-completion-sensitivity \
    --include-extended-dual-decomposition \
    --tee-baseline \
    --tee-others
```


## Extended Dual Decomposition Experiments

The `--include-extended-dual-decomposition` flag (enabled by default) runs additional experiments investigating partial dual information:

- **Constraint-dual-only**: Provides constraint duals (λ*) with primal blocks, letting IPOPT initialize bound multipliers
- **Bounds-dual-only**: Provides bound multipliers (zL/zU) for supplied primal blocks, letting IPOPT initialize constraint duals
- **Primal-dual (all bounds)**: Provides constraint duals plus ALL bound multipliers (all 4 blocks, not just supplied blocks)
- **Dual-only variants**: Tests whether dual information alone (with no primal initialization) can help

These experiments run over all 15 non-empty primal-block subsets and add approximately 48 additional solver runs per case.

## Citation

If you use this code, please cite:
<!-- ```bibtex
@article{taheri2026warmstart,
  title={Not All Warm Starts Help: Benchmarking Primal-Dual Initialization in {ACOPF}},
  author={Taheri, Babak},
  year={2026}
} -->
```


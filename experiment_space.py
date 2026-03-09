"""
6-Dimensional Parameter Space Tensor
=====================================
Models a 6-param experiment where outcomes sit at each cell of a sparse tensor.

Shape: (n_p1, n_p2, n_p3, n_p4, n_p5, n_p6, n_outcomes)
Missing (unrun) cells are NaN.
"""

import numpy as np
import itertools
from typing import Any

# ── 1. Define your parameter space ──────────────────────────────────────────

PARAMS = {
    "Alpha": [0.1, 0.5, 1.0],
    "Beta": [10, 50, 100],
    "Gamma": ["low", "mid", "high"],
    "Delta": [0.01, 0.1],
    "Eps": ["A", "B", "C"],
    "Zeta": [1, 2, 4, 8],
}

OUTCOMES = ["Accuracy", "Speed", "Cost"]  # names of your outcome metrics

# ── 2. Build index maps (value → integer index along each axis) ──────────────

param_names = list(PARAMS.keys())
param_values = list(PARAMS.values())
param_dims = [len(v) for v in param_values]  # e.g. [3, 3, 3, 2, 3, 4]
n_outcomes = len(OUTCOMES)

# Map each parameter value to its axis index
index_maps = [{v: i for i, v in enumerate(vals)} for vals in param_values]

# Tensor shape: one axis per parameter + one axis for outcomes
tensor_shape = tuple(param_dims) + (n_outcomes,)
print(f"Parameter dims : {param_dims}")
print(f"Outcome count  : {n_outcomes}")
print(
    f"Full tensor    : {tensor_shape}  ({np.prod(param_dims)} cells × {n_outcomes} outcomes)"
)
print(f"Total cells    : {np.prod(tensor_shape)}\n")

# Initialise with NaN (unrun cells stay NaN)
T = np.full(tensor_shape, np.nan)


# ── 3. Helper: log a single experiment run ───────────────────────────────────


def log_run(param_dict: dict[str, Any], outcome_values: list[float]) -> None:
    """
    Record one experiment result into the tensor.

    param_dict     : {param_name: value, ...}  e.g. {"Alpha": 0.5, "Beta": 10, ...}
    outcome_values : list of floats, one per outcome, e.g. [0.88, 0.75, 8.1]
    """
    assert len(outcome_values) == n_outcomes, (
        f"Expected {n_outcomes} outcomes, got {len(outcome_values)}"
    )

    idx = tuple(index_maps[i][param_dict[name]] for i, name in enumerate(param_names))
    T[idx] = outcome_values


# ── 4. Helper: retrieve a run ────────────────────────────────────────────────


def get_run(param_dict: dict[str, Any]) -> np.ndarray | None:
    """Return outcome array for a given parameter combination, or None if unrun."""
    idx = tuple(index_maps[i][param_dict[name]] for i, name in enumerate(param_names))
    result = T[idx]
    return None if np.all(np.isnan(result)) else result


# ── 5. Helper: take a 2-D slice (fix all but two params) ────────────────────


def slice_2d(
    fixed: dict[str, Any], row_param: str, col_param: str, outcome: str
) -> np.ndarray:
    """
    Return a 2-D matrix for one outcome, fixing all other params.

    fixed     : values for the 4 fixed parameters
    row_param : name of the parameter that varies along rows
    col_param : name of the parameter that varies along columns
    outcome   : which outcome metric to show
    """
    outcome_idx = OUTCOMES.index(outcome)

    # Build full index, using slice(None) for the two free axes
    idx = []
    for name in param_names:
        if name == row_param or name == col_param:
            idx.append(slice(None))
        else:
            idx.append(index_maps[param_names.index(name)][fixed[name]])

    mat = T[tuple(idx)][..., outcome_idx]

    # Ensure row=row_param, col=col_param
    axes_order = [
        i for i, name in enumerate(param_names) if name in (row_param, col_param)
    ]
    if param_names.index(row_param) > param_names.index(col_param):
        mat = mat.T
    return mat


# ── 6. Sparsity report ───────────────────────────────────────────────────────


def sparsity_report() -> None:
    n_total = np.prod(param_dims)
    # A cell is "run" if at least one outcome is not NaN
    run_mask = ~np.all(np.isnan(T), axis=-1)
    n_run = int(run_mask.sum())
    print(
        f"Cells run      : {n_run} / {n_total}  ({100 * n_run / n_total:.1f}% coverage)"
    )


# ── 7. Demo ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Log some experiment results
    runs = [
        (
            {
                "Alpha": 0.1,
                "Beta": 10,
                "Gamma": "low",
                "Delta": 0.01,
                "Eps": "A",
                "Zeta": 1,
            },
            [0.82, 0.91, 12.3],
        ),
        (
            {
                "Alpha": 0.5,
                "Beta": 10,
                "Gamma": "mid",
                "Delta": 0.1,
                "Eps": "B",
                "Zeta": 2,
            },
            [0.88, 0.75, 8.1],
        ),
        (
            {
                "Alpha": 1.0,
                "Beta": 50,
                "Gamma": "high",
                "Delta": 0.01,
                "Eps": "A",
                "Zeta": 4,
            },
            [0.91, 0.60, 5.6],
        ),
        (
            {
                "Alpha": 0.5,
                "Beta": 100,
                "Gamma": "low",
                "Delta": 0.1,
                "Eps": "C",
                "Zeta": 8,
            },
            [0.76, 0.82, 15.2],
        ),
        (
            {
                "Alpha": 1.0,
                "Beta": 10,
                "Gamma": "mid",
                "Delta": 0.01,
                "Eps": "B",
                "Zeta": 1,
            },
            [0.79, 0.88, 9.4],
        ),
        (
            {
                "Alpha": 0.1,
                "Beta": 50,
                "Gamma": "high",
                "Delta": 0.1,
                "Eps": "C",
                "Zeta": 2,
            },
            [0.94, 0.55, 4.1],
        ),
    ]

    for params, outcomes in runs:
        log_run(params, outcomes)

    sparsity_report()

    # Retrieve a specific run
    result = get_run(
        {"Alpha": 0.5, "Beta": 10, "Gamma": "mid", "Delta": 0.1, "Eps": "B", "Zeta": 2}
    )
    print(f"\nRetrieved run  : {dict(zip(OUTCOMES, result))}")

    # 2-D slice: vary Alpha × Beta, fix the rest, look at Accuracy
    mat = slice_2d(
        fixed={"Gamma": "mid", "Delta": 0.01, "Eps": "A", "Zeta": 1},
        row_param="Alpha",
        col_param="Beta",
        outcome="Accuracy",
    )
    print(
        f"\nAccuracy slice (Alpha × Beta):\n"
        f"  rows={PARAMS['Alpha']}  cols={PARAMS['Beta']}\n{mat}"
    )

    # All combinations iterator (useful for planning which runs are missing)
    all_combos = list(itertools.product(*param_values))
    print(f"\nTotal possible combinations: {len(all_combos)}")

    # Find unrun cells
    unrun = []
    for combo in all_combos:
        pdict = dict(zip(param_names, combo))
        if get_run(pdict) is None:
            unrun.append(pdict)
    print(f"Unrun cells    : {len(unrun)}")
    if unrun:
        print(f"First unrun    : {unrun[0]}")

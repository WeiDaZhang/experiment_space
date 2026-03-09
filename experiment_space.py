"""
6-D Parameter Tensor with Matrix Outcomes & Derivatives
========================================================

Structure
---------
Each cell (p1..p6) stores:
  - raw outcome  : (N+1) x f  ndarray   (e.g. time/freq × features)
  - derived      : dict[str, scalar | ndarray]  computed from the raw matrix

Storage
-------
  raw_store  : dict  {param_tuple -> (N+1)xf ndarray}   sparse, only logged cells
  deriv_T    : ndarray  shape (*param_dims, n_derived)    dense, NaN for unrun cells

The derived tensor is cheap to store densely and supports the visualisation /
ranking / slicing from before. Raw matrices are accessed on demand.
"""

import numpy as np
from typing import Any, Callable

# ── 1. Parameter space (edit these) ─────────────────────────────────────────

PARAMS = {
    "Alpha": [0.1, 0.5, 1.0],
    "Beta": [10, 50, 100],
    "Gamma": ["low", "mid", "high"],
    "Delta": [0.01, 0.1],
    "Eps": ["A", "B", "C"],
    "Zeta": [1, 2, 4, 8],
}

# Raw outcome shape
N = 99  # so the matrix is (N+1) rows  e.g. 100 time steps
F = 4  # number of features / channels

# ── 2. Derivative definitions ────────────────────────────────────────────────
#
# Each entry: (name, function(matrix) -> scalar or 1-D array)
# The function receives the (N+1)×f matrix and must return a fixed-size result.
#
# Mixed scalars and vectors are fine — they are stored flattened into one row
# of deriv_T; DERIVATIVE_SLICES records which positions belong to which name.


def _col_means(m):
    return m.mean(axis=0)  # shape (f,)   — mean of each feature


def _col_maxs(m):
    return m.max(axis=0)  # shape (f,)   — max  of each feature


def _col_stds(m):
    return m.std(axis=0)  # shape (f,)   — std  of each feature


def _overall_mean(m):
    return np.array([m.mean()])  # shape (1,)   — grand mean


def _peak_row(m):
    return np.array([m.sum(axis=1).argmax() / len(m)])  # normalised peak index


def _slope(m):  # shape (f,)   — linear trend per feature
    x = np.arange(len(m))
    x = (x - x.mean()) / (x.std() + 1e-12)
    return np.array([np.polyfit(x, m[:, j], 1)[0] for j in range(m.shape[1])])


DERIVATIVES: list[tuple[str, Callable]] = [
    ("col_means", _col_means),
    ("col_maxs", _col_maxs),
    ("col_stds", _col_stds),
    ("overall_mean", _overall_mean),
    ("peak_row", _peak_row),
    ("slope", _slope),
]

# ── 3. Internal indexing ─────────────────────────────────────────────────────

param_names = list(PARAMS.keys())
param_values = list(PARAMS.values())
param_dims = [len(v) for v in param_values]

index_maps = [{v: i for i, v in enumerate(vals)} for vals in param_values]


def _param_idx(param_dict: dict) -> tuple:
    return tuple(index_maps[i][param_dict[name]] for i, name in enumerate(param_names))


# Work out flat layout of the derived vector
_deriv_sizes: list[int] = []
_test_mat = np.ones((N + 1, F))
for _name, _fn in DERIVATIVES:
    _out = np.atleast_1d(_fn(_test_mat))
    _deriv_sizes.append(_out.size)

DERIVATIVE_SLICES: dict[str, slice] = {}
_pos = 0
for (_name, _), _sz in zip(DERIVATIVES, _deriv_sizes):
    DERIVATIVE_SLICES[_name] = slice(_pos, _pos + _sz)
    _pos += _sz

n_derived = _pos

# Dense derived tensor: shape (*param_dims, n_derived)
deriv_T = np.full((*param_dims, n_derived), np.nan)

# Sparse raw store: param_tuple -> (N+1) x f ndarray
raw_store: dict[tuple, np.ndarray] = {}

print(f"Parameter dims   : {param_dims}  ({np.prod(param_dims)} cells)")
print(f"Raw matrix shape : ({N + 1}, {F})")
print(f"Derived vector   : {n_derived} values from {len(DERIVATIVES)} functions")
for name, sl in DERIVATIVE_SLICES.items():
    print(f"  [{sl.start}:{sl.stop}]  {name}")
print(f"deriv_T shape    : {deriv_T.shape}\n")


# ── 4. Log a run ─────────────────────────────────────────────────────────────


def log_run(param_dict: dict[str, Any], matrix: np.ndarray) -> dict[str, np.ndarray]:
    """
    Store a raw (N+1)×f matrix for one parameter combination and compute
    all derivatives automatically.

    Returns the dict of computed derivatives for inspection.
    """
    assert matrix.shape == (N + 1, F), f"Expected ({N + 1}, {F}), got {matrix.shape}"

    idx = _param_idx(param_dict)

    # Store raw
    raw_store[idx] = matrix.copy()

    # Compute and store derived values
    derived_vec = np.empty(n_derived)
    results = {}
    for name, fn in DERIVATIVES:
        val = np.atleast_1d(fn(matrix))
        derived_vec[DERIVATIVE_SLICES[name]] = val
        results[name] = val

    deriv_T[idx] = derived_vec
    return results


# ── 5. Retrieve ───────────────────────────────────────────────────────────────


def get_raw(param_dict: dict) -> np.ndarray | None:
    """Return the raw (N+1)×f matrix, or None if not yet run."""
    return raw_store.get(_param_idx(param_dict))


def get_derived(param_dict: dict) -> dict[str, np.ndarray] | None:
    """Return a dict of all derived quantities, or None if not yet run."""
    idx = _param_idx(param_dict)
    row = deriv_T[idx]
    if np.all(np.isnan(row)):
        return None
    return {name: row[sl] for name, sl in DERIVATIVE_SLICES.items()}


def get_derived_scalar(name: str) -> np.ndarray:
    """
    Return a sub-tensor of shape (*param_dims, size_of_derivative).
    NaN where the cell has not been run.
    Useful for heatmaps and slicing.
    """
    return deriv_T[..., DERIVATIVE_SLICES[name]]


# ── 6. Add a new derivative after the fact ───────────────────────────────────


def add_derivative(name: str, fn: Callable) -> None:
    """
    Compute a new derivative for all already-logged runs and append it to
    deriv_T.  Use this when your criteria evolve and you want a new metric.
    """
    global deriv_T, n_derived

    # Determine output size from first logged run
    sample = next(iter(raw_store.values()))
    sz = np.atleast_1d(fn(sample)).size

    # Extend the tensor
    extra = np.full((*param_dims, sz), np.nan)
    deriv_T = np.concatenate([deriv_T, extra], axis=-1)

    new_sl = slice(n_derived, n_derived + sz)
    DERIVATIVE_SLICES[name] = new_sl
    DERIVATIVES.append((name, fn))
    n_derived += sz

    # Back-fill for all existing runs
    for idx, mat in raw_store.items():
        deriv_T[idx + (new_sl,)] = np.atleast_1d(fn(mat))

    print(
        f"Added derivative '{name}'  [{new_sl.start}:{new_sl.stop}]  "
        f"→ deriv_T now shape {deriv_T.shape}"
    )


# ── 7. 2-D slice of a derived scalar ─────────────────────────────────────────


def slice_2d(
    fixed: dict[str, Any],
    row_param: str,
    col_param: str,
    deriv_name: str,
    deriv_component: int = 0,
) -> np.ndarray:
    """
    Fix 4 parameters, vary 2, return a 2-D matrix of one derived scalar.
    If the derivative is a vector (e.g. col_means has f components),
    use deriv_component to pick which one.
    """
    sl = DERIVATIVE_SLICES[deriv_name]
    sub = get_derived_scalar(deriv_name)  # shape (*param_dims, sz)
    if sub.shape[-1] > 1:
        sub = sub[..., deriv_component]  # shape (*param_dims,)
    else:
        sub = sub[..., 0]

    idx = []
    for name in param_names:
        if name in (row_param, col_param):
            idx.append(slice(None))
        else:
            idx.append(index_maps[param_names.index(name)][fixed[name]])

    mat = sub[tuple(idx)]
    if param_names.index(row_param) > param_names.index(col_param):
        mat = mat.T
    return mat


# ── 8. Sparsity ───────────────────────────────────────────────────────────────


def sparsity_report() -> None:
    n_total = np.prod(param_dims)
    run_mask = ~np.all(np.isnan(deriv_T), axis=-1)
    n_run = int(run_mask.sum())
    print(f"Cells run : {n_run} / {n_total}  ({100 * n_run / n_total:.1f}% coverage)")


# ── 9. Demo ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    rng = np.random.default_rng(42)

    demo_runs = [
        {
            "Alpha": 0.1,
            "Beta": 10,
            "Gamma": "low",
            "Delta": 0.01,
            "Eps": "A",
            "Zeta": 1,
        },
        {"Alpha": 0.5, "Beta": 10, "Gamma": "mid", "Delta": 0.1, "Eps": "B", "Zeta": 2},
        {
            "Alpha": 1.0,
            "Beta": 50,
            "Gamma": "high",
            "Delta": 0.01,
            "Eps": "A",
            "Zeta": 4,
        },
        {
            "Alpha": 0.5,
            "Beta": 100,
            "Gamma": "low",
            "Delta": 0.1,
            "Eps": "C",
            "Zeta": 8,
        },
    ]

    for p in demo_runs:
        # Simulate a (N+1)×F matrix — replace with your real data
        mat = (
            rng.random((N + 1, F)) * p["Alpha"] + rng.standard_normal((N + 1, F)) * 0.05
        )
        derivs = log_run(p, mat)
        print(
            f"Logged {p['Alpha']}/{p['Beta']}/{p['Gamma']}  "
            f"| overall_mean={derivs['overall_mean'][0]:.4f}  "
            f"| col_means={np.round(derivs['col_means'], 3)}"
        )

    print()
    sparsity_report()

    # Retrieve raw matrix
    raw = get_raw(
        {"Alpha": 0.5, "Beta": 10, "Gamma": "mid", "Delta": 0.1, "Eps": "B", "Zeta": 2}
    )
    print(f"\nRaw matrix shape : {raw.shape}")

    # Retrieve derived dict
    d = get_derived(
        {"Alpha": 0.5, "Beta": 10, "Gamma": "mid", "Delta": 0.1, "Eps": "B", "Zeta": 2}
    )
    print(f"Derived keys     : {list(d.keys())}")
    print(f"slope            : {np.round(d['slope'], 4)}")

    # 2-D slice
    mat2d = slice_2d(
        fixed={"Gamma": "mid", "Delta": 0.01, "Eps": "A", "Zeta": 1},
        row_param="Alpha",
        col_param="Beta",
        deriv_name="overall_mean",
    )
    print(f"\noverall_mean slice (Alpha × Beta):\n{mat2d}")

    # Add a new derivative on the fly (criteria evolve!)
    print()
    add_derivative("energy", lambda m: np.array([(m**2).sum()]))
    d2 = get_derived(
        {"Alpha": 0.5, "Beta": 10, "Gamma": "mid", "Delta": 0.1, "Eps": "B", "Zeta": 2}
    )
    print(f"energy           : {d2['energy'][0]:.2f}")

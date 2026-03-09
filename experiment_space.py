"""
Generalised N-Dimensional Parameter Tensor
===========================================

Parameters and outcomes are defined as dataclasses.
The tensor dimension is fully inferred from the parameter definitions —
no hardcoded 6 anywhere.

Structure
---------
  ParameterDef  : one axis of the tensor (name + discrete values)
  OutcomeDef    : one derivative function applied to the raw matrix
  ExperimentSpace : owns the tensor, raw store, and all operations
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Any, Callable


# ── Dataclasses ──────────────────────────────────────────────────────────────


@dataclass
class ParameterDef:
    """One parameter axis in the experiment space."""

    name: str
    values: list[Any]  # discrete values this param can take

    # Derived at post-init — not set by user
    _index_map: dict[Any, int] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self):
        self._index_map = {v: i for i, v in enumerate(self.values)}

    @property
    def n(self) -> int:
        return len(self.values)

    def index_of(self, value: Any) -> int:
        if value not in self._index_map:
            raise KeyError(
                f"Parameter '{self.name}': value {value!r} not in {self.values}"
            )
        return self._index_map[value]


@dataclass
class OutcomeDef:
    """
    One derived outcome computed from the raw (N+1)×f matrix.

    fn        : (N+1)×f ndarray  ->  scalar or 1-D ndarray (fixed size)
    direction : 'max' | 'min' | None  — used for scoring / ranking
    """

    name: str
    fn: Callable[[np.ndarray], np.ndarray | float]
    direction: str | None = "max"  # 'max', 'min', or None (informational)
    _size: int = field(default=0, init=False, repr=False)

    def compute(self, matrix: np.ndarray) -> np.ndarray:
        return np.atleast_1d(np.asarray(self.fn(matrix), dtype=float))


@dataclass
class RawShape:
    """Shape of the raw outcome matrix stored at each cell."""

    rows: int  # N+1
    cols: int  # f

    @property
    def shape(self) -> tuple[int, int]:
        return (self.rows, self.cols)


# ── Experiment Space ─────────────────────────────────────────────────────────


class ExperimentSpace:
    """
    An n-dimensional sparse tensor over a discrete parameter grid.

    Each cell may hold:
      - a raw (rows × cols) matrix
      - a flat vector of derived values (concatenation of all OutcomeDef outputs)

    Parameters
    ----------
    parameters : list of ParameterDef   — defines the tensor axes
    outcomes   : list of OutcomeDef     — defines what to compute from each raw matrix
    raw_shape  : RawShape               — shape of the raw outcome matrix
    """

    def __init__(
        self,
        parameters: list[ParameterDef],
        outcomes: list[OutcomeDef],
        raw_shape: RawShape,
    ):
        self.parameters = parameters
        self.outcomes = outcomes
        self.raw_shape = raw_shape

        self._param_dims: list[int] = [p.n for p in parameters]
        self._n_dims: int = len(parameters)
        self._raw_store: dict[tuple, np.ndarray] = {}
        self._slices: dict[str, slice] = {}
        self._deriv_size: int = 0

        self._build_slices()

        # Dense derived tensor — shape (*param_dims, n_derived), NaN = unrun
        self.deriv_T = np.full((*self._param_dims, self._deriv_size), np.nan)

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _build_slices(self) -> None:
        """Compute flat layout of the derived vector from a probe matrix."""
        probe = np.ones(self.raw_shape.shape)
        pos = 0
        for od in self.outcomes:
            out = od.compute(probe)
            od._size = out.size
            self._slices[od.name] = slice(pos, pos + od._size)
            pos += od._size
        self._deriv_size = pos

    def _param_idx(self, param_dict: dict[str, Any]) -> tuple[int, ...]:
        return tuple(p.index_of(param_dict[p.name]) for p in self.parameters)

    def _full_idx(self, param_dict: dict[str, Any]) -> tuple:
        """Index into deriv_T: (*param_idx, slice-all-derived)."""
        return self._param_idx(param_dict)

    # ── Public API ───────────────────────────────────────────────────────────

    def log_run(
        self,
        param_dict: dict[str, Any],
        matrix: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """
        Store a raw matrix and compute all derivatives for one parameter combination.

        Returns a dict {outcome_name: computed_array} for immediate inspection.
        """
        if matrix.shape != self.raw_shape.shape:
            raise ValueError(
                f"Expected matrix shape {self.raw_shape.shape}, got {matrix.shape}"
            )
        idx = self._param_idx(param_dict)
        self._raw_store[idx] = matrix.copy()

        results: dict[str, np.ndarray] = {}
        for od in self.outcomes:
            val = od.compute(matrix)
            self.deriv_T[idx][self._slices[od.name]] = val
            results[od.name] = val
        return results

    def get_raw(self, param_dict: dict[str, Any]) -> np.ndarray | None:
        """Return the raw matrix for a cell, or None if unrun."""
        return self._raw_store.get(self._param_idx(param_dict))

    def get_derived(self, param_dict: dict[str, Any]) -> dict[str, np.ndarray] | None:
        """Return all derived quantities for a cell, or None if unrun."""
        idx = self._param_idx(param_dict)
        row = self.deriv_T[idx]
        if np.all(np.isnan(row)):
            return None
        return {od.name: row[self._slices[od.name]] for od in self.outcomes}

    def get_outcome_tensor(self, outcome_name: str) -> np.ndarray:
        """
        Return a sub-tensor of shape (*param_dims, outcome_size).
        NaN where the cell is unrun. Useful for heatmaps and slicing.
        """
        sl = self._slices[outcome_name]
        return self.deriv_T[..., sl]

    def add_outcome(self, outcome_def: OutcomeDef) -> None:
        """
        Append a new OutcomeDef and back-fill it for all already-logged runs.
        Use this as your criteria evolve — no re-running needed.
        """
        probe = np.ones(self.raw_shape.shape)
        val = outcome_def.compute(probe)
        outcome_def._size = val.size

        new_sl = slice(self._deriv_size, self._deriv_size + outcome_def._size)
        self._slices[outcome_def.name] = new_sl
        self.outcomes.append(outcome_def)
        self._deriv_size += outcome_def._size

        # Extend deriv_T along the last axis
        extra = np.full((*self._param_dims, outcome_def._size), np.nan)
        self.deriv_T = np.concatenate([self.deriv_T, extra], axis=-1)

        # Back-fill existing runs
        for idx, mat in self._raw_store.items():
            self.deriv_T[idx + (new_sl,)] = outcome_def.compute(mat)

        print(
            f"Added outcome '{outcome_def.name}'  "
            f"[{new_sl.start}:{new_sl.stop}]  "
            f"→ deriv_T shape {self.deriv_T.shape}"
        )

    def slice_2d(
        self,
        fixed: dict[str, Any],
        row_param: str,
        col_param: str,
        outcome: str,
        component: int = 0,
    ) -> tuple[np.ndarray, list, list]:
        """
        Fix all but two parameters, return a 2-D matrix of one derived scalar.

        If the outcome is a vector (e.g. col_means with f components),
        `component` selects which element to show.

        Returns (matrix, row_values, col_values) for labelled plotting.
        """
        sub = self.get_outcome_tensor(outcome)  # (*param_dims, size)
        if sub.shape[-1] > 1:
            sub = sub[..., component]
        else:
            sub = sub[..., 0]

        param_name_list = [p.name for p in self.parameters]
        row_axis = param_name_list.index(row_param)
        col_axis = param_name_list.index(col_param)

        idx: list[Any] = []
        for p in self.parameters:
            if p.name in (row_param, col_param):
                idx.append(slice(None))
            else:
                idx.append(p.index_of(fixed[p.name]))

        mat = sub[tuple(idx)]
        if row_axis > col_axis:
            mat = mat.T

        row_vals = self.parameters[row_axis].values
        col_vals = self.parameters[col_axis].values
        return mat, row_vals, col_vals

    def sparsity(self) -> dict[str, Any]:
        n_total = int(np.prod(self._param_dims))
        n_run = int((~np.all(np.isnan(self.deriv_T), axis=-1)).sum())
        return {
            "n_cells": n_total,
            "n_run": n_run,
            "n_unrun": n_total - n_run,
            "coverage": n_run / n_total,
            "param_dims": self._param_dims,
            "n_params": self._n_dims,
            "deriv_T_shape": self.deriv_T.shape,
        }

    def __repr__(self) -> str:
        s = self.sparsity()
        params_str = " × ".join(f"{p.name}({p.n})" for p in self.parameters)
        return (
            f"ExperimentSpace[\n"
            f"  params  : {params_str}\n"
            f"  dims    : {s['param_dims']}  →  {s['n_cells']} cells\n"
            f"  raw     : {self.raw_shape.shape}  per cell\n"
            f"  derived : {self._deriv_size} values  {list(self._slices.keys())}\n"
            f"  run     : {s['n_run']} / {s['n_cells']}  "
            f"({100 * s['coverage']:.1f}% coverage)\n]"
        )


# ══════════════════════════════════════════════════════════════════════════════
# Demo — edit everything below to match your actual experiment
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # ── Define the parameter space ───────────────────────────────────────────

    parameters = [
        ParameterDef("Alpha", [0.1, 0.5, 1.0]),
        ParameterDef("Beta", [10, 50, 100]),
        ParameterDef("Gamma", ["low", "mid", "high"]),
        ParameterDef("Delta", [0.01, 0.1]),
        ParameterDef("Eps", ["A", "B", "C"]),
        ParameterDef("Zeta", [1, 2, 4, 8]),
    ]

    # ── Define the raw matrix shape ──────────────────────────────────────────

    raw_shape = RawShape(rows=100, cols=4)  # (N+1) x f

    # ── Define initial outcomes (derivatives of the raw matrix) ─────────────

    outcomes = [
        OutcomeDef("col_means", lambda m: m.mean(axis=0), direction="max"),
        OutcomeDef("col_maxs", lambda m: m.max(axis=0), direction="max"),
        OutcomeDef("col_stds", lambda m: m.std(axis=0), direction=None),
        OutcomeDef("overall_mean", lambda m: np.array([m.mean()]), direction="max"),
        OutcomeDef(
            "peak_row",
            lambda m: np.array([m.sum(axis=1).argmax() / len(m)]),
            direction=None,
        ),
        OutcomeDef(
            "slope",
            lambda m: np.array(
                [
                    np.polyfit((np.arange(len(m)) - len(m) / 2) / len(m), m[:, j], 1)[0]
                    for j in range(m.shape[1])
                ]
            ),
            direction=None,
        ),
    ]

    # ── Build the space ──────────────────────────────────────────────────────

    space = ExperimentSpace(parameters, outcomes, raw_shape)
    print(space)

    # ── Log some runs ────────────────────────────────────────────────────────

    rng = np.random.default_rng(0)

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

    print("\n── Logging runs ────────────────────────────────")
    for p in demo_runs:
        mat = rng.random(raw_shape.shape) * p["Alpha"]
        derivs = space.log_run(p, mat)
        print(
            f"  {p['Alpha']}/{p['Beta']}/{p['Gamma']}"
            f"  overall_mean={derivs['overall_mean'][0]:.4f}"
            f"  col_means={np.round(derivs['col_means'], 3)}"
        )

    print(f"\n{space}")

    # ── Retrieve ─────────────────────────────────────────────────────────────

    query = {
        "Alpha": 0.5,
        "Beta": 10,
        "Gamma": "mid",
        "Delta": 0.1,
        "Eps": "B",
        "Zeta": 2,
    }

    raw = space.get_raw(query)
    print(f"\n── Raw matrix : {raw.shape}")

    d = space.get_derived(query)
    print(f"── Derived    : {list(d.keys())}")
    print(f"   slope      : {np.round(d['slope'], 4)}")

    # ── 2-D slice ────────────────────────────────────────────────────────────

    mat2d, rows, cols = space.slice_2d(
        fixed={"Gamma": "low", "Delta": 0.01, "Eps": "A", "Zeta": 1},
        row_param="Alpha",
        col_param="Beta",
        outcome="overall_mean",
    )
    print(f"\n── overall_mean slice  (Alpha × Beta)")
    print(f"   rows={rows}  cols={cols}")
    print(f"   {mat2d}")

    # ── Add a new outcome on the fly ─────────────────────────────────────────

    print("\n── Adding new outcome 'energy' ─────────────────")
    space.add_outcome(
        OutcomeDef(
            name="energy",
            fn=lambda m: np.array([(m**2).sum()]),
            direction="min",
        )
    )

    d2 = space.get_derived(query)
    print(f"   energy = {d2['energy'][0]:.4f}")
    print(f"\n{space}")

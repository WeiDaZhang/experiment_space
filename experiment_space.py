"""
ExperimentSpace  —  Generalised Parameter Tensor
=================================================

Design
------
- AxisDef is the single shared description of any discrete axis: name, values,
  unit, scale.  Both parameter axes and outcome output axes are AxisDefs.
- ParameterDef inherits AxisDef and adds indexing / label helpers.
- OutcomeDef replaces the old parallel lists (dim_coords / dim_units /
  dim_scales) with dim_axes: list[AxisDef].  Scalar outcomes have dim_axes=[].
- Raw data per cell is an arbitrary Python object — no shape or type constraint.
- Each OutcomeDef owns a sub-tensor of shape (*param_dims, *outcome_dims).
  A scalar outcome (shape ()) adds no extra axes.
- select(filter={}) resolves axis names from parameters and the chosen outcome's dim_axes.
  Bare names work when unambiguous; 'p:name' / 'o:name' qualify when needed.
  Scalar value → axis collapsed; list of values → axis kept (restricted).

Key classes
-----------
  AxisDef         — shared axis description (name, values, unit, scale)
  ParameterDef    — AxisDef subclass for parameter axes
  OutcomeDef      — derived quantity with dim_axes: list[AxisDef]
  ExperimentSpace — owns raw_store, sub-tensors, and all operations
"""

from __future__ import annotations

import re
import warnings
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Callable

# Matches an optional sign, integer or decimal magnitude (incl. scientific
# notation), optional whitespace, then the rest as a unit.
# Examples: "1mm"  "1.5 mm"  "-10°C"  "0.01pF"  "1e3Hz"  "100 kHz"
_MAGNITUDE_RE = re.compile(r"^\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*(.+?)\s*$")


# ══════════════════════════════════════════════════════════════════════════════
# AxisDef  —  shared description of any discrete axis
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class AxisDef:
    """
    Description of one discrete axis, shared by parameters and outcome dimensions.

    name    : axis name, used in __repr__ and plot labels.
    values  : ordered list of coordinate values (any hashable type).
              Indexing always uses these bare values — unit is display only.
    unit    : physical unit, e.g. "mm", "Hz", "°C".  Display only.
    scale   : "linear" (default) or "log".  Hint for visualisers only.
    """

    name: str
    values: list[Any]
    unit: str | None = None
    scale: str = "linear"

    def __post_init__(self):
        if len(self.values) == 0:
            raise ValueError(f"AxisDef '{self.name}': values must be non-empty")
        if len(self.values) != len(set(map(str, self.values))):
            raise ValueError(f"AxisDef '{self.name}': values must be unique")
        if self.scale not in ("linear", "log"):
            raise ValueError(f"AxisDef '{self.name}': scale must be 'linear' or 'log'")

    # ── Size ─────────────────────────────────────────────────────────────────

    @property
    def n(self) -> int:
        return len(self.values)

    # ── Indexing ─────────────────────────────────────────────────────────────

    def index_of(self, value: Any) -> int:
        """
        Return the integer position of value in self.values.

        Accepts two input forms:
            Bare value      e.g. 1, "low", 10.0
                            Direct lookup — the normal case.
            Annotated string  e.g. "1mm", "10 Hz", "0.01pF"
                            Parsed with _MAGNITUDE_RE.  The extracted unit
                            must match self.unit exactly; the magnitude is
                            coerced to int if whole, then looked up in values.

        Raises KeyError with a diagnostic message for:
            - value not found (bare or after parsing)
            - annotated string on a no-unit axis
            - unit mismatch (e.g. "1cm" on a "mm" axis)
        """
        # ── Fast path: exact match (bare value, the normal case) ─────────────
        try:
            return self.values.index(value)
        except ValueError:
            pass

        # ── Slow path: try parsing as annotated string "1mm", "10Hz", etc. ──
        if isinstance(value, str):
            m = _MAGNITUDE_RE.match(value)
            if m:
                mag_str, unit_str = m.group(1), m.group(2)

                if self.unit is None:
                    raise KeyError(
                        f"AxisDef '{self.name}' has no unit but received "
                        f"annotated string {value!r}. "
                        f"Pass the bare value instead."
                    )
                if unit_str != self.unit:
                    raise KeyError(
                        f"AxisDef '{self.name}' (unit='{self.unit}'): "
                        f"unit mismatch — got '{unit_str}' in {value!r}. "
                        f"Pass '{self.unit}'-annotated strings or bare values."
                    )

                mag = float(mag_str)
                bare = int(mag) if mag == int(mag) else mag
                try:
                    return self.values.index(bare)
                except ValueError:
                    raise KeyError(
                        f"AxisDef '{self.name}': magnitude {bare!r} "
                        f"(parsed from {value!r}) not in {self.values}"
                    ) from None

        # ── Neither path worked ───────────────────────────────────────────────
        raise KeyError(
            f"AxisDef '{self.name}': {value!r} not in {self.values}"
            + (
                f"  (tip: annotated strings like '1 {self.unit}' also accepted)"
                if self.unit
                else ""
            )
        )

    # ── Display ───────────────────────────────────────────────────────────────

    def label(self, value: Any) -> str:
        """Single coordinate as a display string, e.g. '10 Hz'."""
        return f"{value} {self.unit}" if self.unit else str(value)

    @property
    def labels(self) -> list[str]:
        """All coordinates as display strings."""
        return [self.label(v) for v in self.values]

    @property
    def axis_label(self) -> str:
        """Axis header for plot labels, e.g. 'frequency (Hz)'."""
        return f"{self.name} ({self.unit})" if self.unit else self.name

    # ── Construction from annotated strings ─────────────────────────────────

    @classmethod
    def from_strings(
        cls,
        name: str,
        strings: list[str],
        scale: str = "linear",
    ) -> "AxisDef":
        """
        Parse a list of unit-annotated strings into an AxisDef (or subclass).

        Each string must be of the form  "<magnitude><unit>"  or
        "<magnitude> <unit>", e.g. "1mm", "0.01 pF", "100Hz", "-10°C".

        Rules
        -----
        - All strings must share the same unit; mixed units raise ValueError.
        - Magnitudes that are whole numbers become ints; otherwise floats.
        - If any string cannot be parsed (e.g. "low", "mid", "high"), a
          ParseError is raised.  Use the normal constructor for categorical axes.
        - `name` and `scale` are passed through unchanged.

        Returns an instance of cls (so ParameterDef.from_strings returns a
        ParameterDef, AxisDef.from_strings returns an AxisDef).

        Examples
        --------
            ParameterDef.from_strings("thickness",  ["1mm", "2mm", "5mm"])
            AxisDef.from_strings("frequency", ["10Hz", "100Hz", "1kHz"])
            ParameterDef.from_strings("cap", ["0.01pF", "0.1pF", "0.5pF"])
            ParameterDef.from_strings("temp", ["-10°C", "20°C", "37°C"])
        """
        magnitudes: list[int | float] = []
        units: list[str] = []

        for s in strings:
            m = _MAGNITUDE_RE.match(str(s))
            if m is None:
                raise ValueError(
                    f"AxisDef.from_strings ('{name}'): cannot parse {s!r} — "
                    f"expected '<number><unit>', e.g. '1mm' or '10 Hz'. "
                    f"For categorical axes use the normal constructor."
                )
            mag_str, unit_str = m.group(1), m.group(2)
            mag = float(mag_str)
            magnitudes.append(int(mag) if mag == int(mag) else mag)
            units.append(unit_str)

        # All units must be identical
        unique_units = set(units)
        if len(unique_units) > 1:
            raise ValueError(
                f"AxisDef.from_strings ('{name}'): mixed units {sorted(unique_units)} — "
                f"all values must share the same unit."
            )

        return cls(
            name=name,
            values=magnitudes,
            unit=units[0],
            scale=scale,
        )

    def __repr__(self) -> str:
        unit_str = f" [{self.unit}]" if self.unit else ""
        scale_str = f" ({self.scale})" if self.scale != "linear" else ""
        return f"AxisDef('{self.name}'{unit_str}, n={self.n}{scale_str})"


# ══════════════════════════════════════════════════════════════════════════════
# ParameterDef  —  AxisDef for a parameter (experiment input axis)
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class ParameterDef(AxisDef):
    """
    One parameter axis of the experiment space.  Inherits all of AxisDef —
    name, values, unit, scale, index_of, label, labels, axis_label.
    No new fields are added.

    Example
    -------
        ParameterDef("thickness", [1, 2, 5],       unit="mm")
        ParameterDef("frequency", [10, 100, 1000], unit="Hz", scale="log")
        ParameterDef("mode",      ["low", "mid", "high"])   # categorical
    """

    def __repr__(self) -> str:
        unit_str = f" [{self.unit}]" if self.unit else ""
        scale_str = f" ({self.scale})" if self.scale != "linear" else ""
        return f"ParameterDef('{self.name}'{unit_str}, n={self.n}{scale_str})"


# ══════════════════════════════════════════════════════════════════════════════
# OutcomeDef  —  one derived quantity with fully described output axes
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class OutcomeDef:
    """
    One derived outcome computed from raw cell data.

    fn
        Callable(raw: Any) -> np.ndarray
        May return any shape.  Shape () means scalar — adds no extra axes.
        Shape (f,) adds one axis.  Shape (m, f) adds two, etc.

    direction
        'max', 'min', or None.  Informational — used by scoring / ranking.

    dim_axes
        One AxisDef per output axis.  Scalar outcomes have dim_axes=[].
        Each AxisDef carries name, values (coordinates), unit, and scale —
        the same fields as a ParameterDef.
        Validated against fn's actual output shape on first compute() call.

    Example
    -------
        OutcomeDef(
            name      = "spectrum",
            fn        = lambda raw: np.abs(np.fft.rfft(raw)),
            direction = "max",
            dim_axes  = [
                AxisDef("frequency", [10, 20, 50, 100], unit="Hz", scale="log")
            ],
        )

        OutcomeDef(
            name      = "feature_stats",
            fn        = lambda raw: np.stack([raw.mean(0), raw.std(0)], axis=-1),
            direction = None,
            dim_axes  = [
                AxisDef("feature",   ["f0", "f1", "f2", "f3"]),
                AxisDef("statistic", ["mean", "std"]),
            ],
        )
    """

    name: str
    fn: Callable[[Any], np.ndarray]
    direction: str | None = "max"
    dim_axes: list[AxisDef] = field(default_factory=list)

    # Inferred on first compute() call — not set by user
    _out_shape: tuple[int, ...] = field(default=(), init=False, repr=False)
    _out_dtype: Any = field(default=None, init=False, repr=False)

    def __post_init__(self):
        if self.direction not in ("max", "min", None):
            raise ValueError(
                f"OutcomeDef '{self.name}': direction must be 'max', 'min', or None"
            )

    # ── Core ─────────────────────────────────────────────────────────────────

    def compute(self, raw: Any) -> np.ndarray:
        """
        Apply fn to raw data.  Returns a numpy array; shape () for scalars.
        On the first call, validates dim_axes against the actual output shape.
        """
        result = np.asarray(self.fn(raw))

        if self._out_shape == ():  # first call only
            if result.shape != ():  # non-scalar: validate
                if len(self.dim_axes) != result.ndim:
                    raise ValueError(
                        f"OutcomeDef '{self.name}': fn returned shape "
                        f"{result.shape} ({result.ndim}D) but dim_axes has "
                        f"{len(self.dim_axes)} entries"
                    )
                for axis_i, (size, ax) in enumerate(zip(result.shape, self.dim_axes)):
                    if ax.n != size:
                        raise ValueError(
                            f"OutcomeDef '{self.name}': output axis {axis_i} "
                            f"('{ax.name}') has size {size} but AxisDef has n={ax.n}"
                        )
            object.__setattr__(self, "_out_shape", result.shape)
            object.__setattr__(self, "_out_dtype", result.dtype)

        return result

    @property
    def out_shape(self) -> tuple[int, ...]:
        """Output array shape.  Available after the first compute() call."""
        return self._out_shape

    @property
    def out_dtype(self) -> "np.dtype | None":
        """Output array dtype.  Available after the first compute() call."""
        return self._out_dtype

    @property
    def is_scalar(self) -> bool:
        return self._out_shape == ()

    # ── Indexing (delegates to AxisDef) ──────────────────────────────────────

    def coord_index(self, axis: int, label: Any) -> int:
        """
        Integer index of a coordinate label along one output axis.
        axis is 0-based relative to this outcome's own output axes.
        Delegates to AxisDef.index_of — always bare value, never 'value unit'.
        """
        return self.dim_axes[axis].index_of(label)

    def __repr__(self) -> str:
        axes_str = (
            "scalar"
            if not self.dim_axes
            else "[" + ", ".join(repr(a) for a in self.dim_axes) + "]"
        )
        return (
            f"OutcomeDef('{self.name}', "
            f"shape={self._out_shape}, "
            f"dir={self.direction}, "
            f"axes={axes_str})"
        )


# ══════════════════════════════════════════════════════════════════════════════
# _CombinedAxisDef  —  internal: axis produced by squeeze() when folding >1 axes
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class _CombinedAxisDef(AxisDef):
    """
    AxisDef for the combined (folded) axis produced by SelectionResult.squeeze().

    values  : list of tuples, one per cartesian-product combination.
    fold_axes : the original AxisDefs that were folded together.

    label() joins the individual per-axis label strings with " / ".
    axis_label is "ax_a × ax_b × ..." with units if all axes share one unit.
    """

    fold_axes: list["AxisDef"] = field(default_factory=list)

    def __post_init__(self):
        # Bypass AxisDef.__post_init__ uniqueness check — tuple values are
        # always unique by construction (cartesian product).
        if self.scale not in ("linear", "log"):
            raise ValueError(f"_CombinedAxisDef: scale must be 'linear' or 'log'")

    def label(self, value: Any) -> str:
        """Join per-axis labels: (1, "A", 10) → '1 mm / A / 10 Hz'."""
        if isinstance(value, tuple):
            return " / ".join(ax.label(v) for ax, v in zip(self.fold_axes, value))
        # Single-axis fold (shouldn't reach here, but safe fallback)
        return str(value)

    @property
    def axis_label(self) -> str:
        return self.name


# ══════════════════════════════════════════════════════════════════════════════
# SelectionResult  —  named container returned by ExperimentSpace.select()
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class SelectionResult:
    """
    Container for a sub-tensor returned by ExperimentSpace.select().

    Bundles the numeric data with its full axis metadata so the two never
    become separated when passed to squeeze() or plotting code.

    Attributes
    ----------
    outcome : str
        Name of the OutcomeDef that produced this result.
    tensor  : np.ndarray
        The selected sub-tensor.  tensor.shape[i] corresponds to axes[i].
    axes    : list[AxisDef]
        One AxisDef per tensor dimension, in definition order.
        Scalar-fixed axes are absent; list-restricted axes carry the subset
        of values that was selected.

    Axis access
    -----------
    result[name]            AxisDef by name (bare or "p:"/"o:" qualified).
    result.dim(name)        Integer dimension index of a named axis.
    result.coords(dim)      Coordinate values along a dimension.
    result.labels(dim)      Display strings for coordinates along a dimension.
    result.axis_label(dim)  Axis title string, e.g. "frequency (Hz)".

    Reshaping
    ---------
    result.squeeze(axes)    Fold all but the named axes into a combined leading
                            axis, reordering to match the requested axis list.
                            Returns a new SelectionResult ready for plotting.
    """

    outcome: str
    tensor: np.ndarray
    axes: list["AxisDef"]

    # ── Shape ─────────────────────────────────────────────────────────────────

    @property
    def shape(self) -> tuple[int, ...]:
        return self.tensor.shape

    @property
    def ndim(self) -> int:
        return self.tensor.ndim

    # ── Axis access ───────────────────────────────────────────────────────────

    def dim(self, name: str) -> int:
        """
        Return the dimension index of an axis by name.
        Accepts bare names or "p:"/"o:" qualified names (prefix stripped).
        """
        bare = name[2:] if name.startswith(("p:", "o:")) else name
        for i, ax in enumerate(self.axes):
            if ax.name == bare:
                return i
        raise KeyError(
            f"SelectionResult: no axis named '{bare}' in {[a.name for a in self.axes]}"
        )

    def __getitem__(self, name: str) -> "AxisDef":
        """Return the AxisDef for the named axis."""
        return self.axes[self.dim(name)]

    def coords(self, dim: int) -> list:
        """Coordinate values along dimension dim."""
        return self.axes[dim].values

    def labels(self, dim: int) -> list[str]:
        """Display strings for coordinates along dimension dim."""
        return self.axes[dim].labels

    def axis_label(self, dim: int) -> str:
        """Axis title string for dimension dim, e.g. 'frequency (Hz)'."""
        return self.axes[dim].axis_label

    # ── Squeeze ───────────────────────────────────────────────────────────────

    def squeeze(self, axes: list[str]) -> "SelectionResult":
        """
        Reshape the tensor to (combined, d0, d1, ...) where d0, d1, ... are
        the axes named in the `axes` list, in that order.

        All dimensions NOT named in `axes` are folded into a single leading
        combined axis whose size is their product.  If every axis is named,
        no combined axis is produced and the tensor is simply reordered.

        Parameters
        ----------
        axes
            Ordered list of up to 3 axis names to keep as independent
            dimensions.  Order is explicit — the output tensor and
            SelectionResult.axes mirror this list exactly (after the
            combined axis).  "p:" / "o:" qualifiers work as in select():
            required only when two surviving axes share the same name
            (p: picks the first match in definition order, o: the last).

        Returns
        -------
        A new SelectionResult with tensor shape (N_combined, d0, d1, ...).

        The combined axis AxisDef has:
            name   = "a × b × ..."  (joined names of folded axes)
            values = list of tuples (cartesian product of folded axis values)
            labels = "v_a / v_b / ..."  per combination

        A single-axis fold uses bare values (not tuples) and inherits that
        axis's unit and scale for labelling.
        """
        from itertools import product as iproduct

        if len(axes) > 3:
            raise ValueError(
                f"squeeze: at most 3 axes may be kept independently, "
                f"got {len(axes)}: {axes}"
            )

        # ── Resolve names to positions in self.axes ───────────────────────────
        # self.axes is in definition order: params first, outcome axes after.
        # p: → pick first match (params precede outcome axes by construction)
        # o: → pick last match
        keep_positions: list[int] = []
        seen: set[int] = set()
        for name in axes:
            if name.startswith("p:"):
                bare, pick = name[2:], "first"
            elif name.startswith("o:"):
                bare, pick = name[2:], "last"
            else:
                bare, pick = name, None

            found = [i for i, ax in enumerate(self.axes) if ax.name == bare]
            if not found:
                raise KeyError(
                    f"squeeze: '{bare}' not found in axes {[a.name for a in self.axes]}"
                )
            if len(found) > 1:
                if pick is None:
                    raise ValueError(
                        f"squeeze: '{bare}' matches multiple axes at positions "
                        f"{found}. Use 'p:{bare}' or 'o:{bare}' to disambiguate."
                    )
                pos = found[0] if pick == "first" else found[-1]
            else:
                pos = found[0]

            if pos in seen:
                raise ValueError(f"squeeze: axis '{name}' listed more than once")
            keep_positions.append(pos)
            seen.add(pos)

        fold_positions = [i for i in range(self.tensor.ndim) if i not in seen]
        fold_axes = [self.axes[i] for i in fold_positions]
        keep_axes = [self.axes[i] for i in keep_positions]

        # ── Transpose: fold axes first, keep axes in requested order ──────────
        new_order = fold_positions + keep_positions
        t = np.transpose(self.tensor, new_order)

        keep_shape = t.shape[len(fold_positions) :]

        # ── No fold axes → pure reorder ───────────────────────────────────────
        if not fold_positions:
            return SelectionResult(
                outcome=self.outcome,
                tensor=t,
                axes=keep_axes,
            )

        # ── Build combined axis ───────────────────────────────────────────────
        N_combined = int(np.prod(t.shape[: len(fold_positions)]))
        t_reshaped = t.reshape(N_combined, *keep_shape)

        fold_value_lists = [ax.values for ax in fold_axes]
        combo_values = list(iproduct(*fold_value_lists))

        if len(fold_axes) == 1:
            fa = fold_axes[0]
            combo_values = [v[0] for v in combo_values]
            combined_ax = AxisDef(fa.name, combo_values, fa.unit, fa.scale)
        else:
            combined_ax = _CombinedAxisDef(
                name=" × ".join(ax.name for ax in fold_axes),
                values=combo_values,
                fold_axes=fold_axes,
            )

        return SelectionResult(
            outcome=self.outcome,
            tensor=t_reshaped,
            axes=[combined_ax] + keep_axes,
        )

    # ── Repr ──────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        axes_str = ", ".join(f"{ax.axis_label}({ax.n})" for ax in self.axes)
        return (
            f"SelectionResult(outcome='{self.outcome}', "
            f"shape={self.shape}, axes=[{axes_str}])"
        )


# ══════════════════════════════════════════════════════════════════════════════
# ExperimentSpace
# ══════════════════════════════════════════════════════════════════════════════


class ExperimentSpace:
    """
    An n-dimensional sparse experiment tensor.

    Parameters define the tensor axes (shape: *param_dims).
    Each OutcomeDef contributes a dense sub-tensor:
        shape (*param_dims,)              for scalar outcomes
        shape (*param_dims, *out_shape)   for array outcomes
    NaN marks unrun cells.

    Raw data (any Python object) is stored per cell when store_raw=True (default).
    New OutcomeDefs can be added at any time and are back-filled from stored raw.
    """

    def __init__(
        self,
        parameters: list[ParameterDef],
        outcomes: list[OutcomeDef],
        store_raw: bool = True,
    ):
        if not parameters:
            raise ValueError("ExperimentSpace requires at least one ParameterDef")
        names = [p.name for p in parameters]
        if len(names) != len(set(names)):
            raise ValueError("ParameterDef names must be unique")

        self.parameters: list[ParameterDef] = parameters
        self.outcomes: list[OutcomeDef] = outcomes
        self.store_raw: bool = store_raw

        self._param_dims: tuple[int, ...] = tuple(p.n for p in parameters)
        self._param_names: list[str] = [p.name for p in parameters]

        self._raw_store: dict[tuple, Any] = {}
        self._sub_tensors: dict[str, np.ndarray] = {}

    # ── Internal ──────────────────────────────────────────────────────────────

    def _param_idx(self, param_dict: dict[str, Any]) -> tuple[int, ...]:
        try:
            return tuple(p.index_of(param_dict[p.name]) for p in self.parameters)
        except KeyError as e:
            raise KeyError(f"param_dict missing or invalid key: {e}") from None

    def _ensure_sub_tensor(self, od: OutcomeDef) -> None:
        if od.name not in self._sub_tensors:
            shape = self._param_dims + od.out_shape
            dtype = od.out_dtype if od.out_dtype is not None else float
            nan_val = (
                complex(np.nan, np.nan)
                if np.issubdtype(dtype, np.complexfloating)
                else np.nan
            )
            self._sub_tensors[od.name] = np.full(shape, nan_val, dtype=dtype)

    def _write_cell(self, od: OutcomeDef, idx: tuple, value: np.ndarray) -> None:
        self._ensure_sub_tensor(od)
        self._sub_tensors[od.name][idx] = value

    # ── Core API ──────────────────────────────────────────────────────────────

    def log_run(
        self,
        param_dict: dict[str, Any],
        raw: Any,
    ) -> dict[str, np.ndarray]:
        """
        Record one experiment result.

        param_dict  : {param_name: value} for every ParameterDef.
        raw         : the raw measurement — any Python object.

        Returns {outcome_name: computed_array} for immediate inspection.
        """
        idx = self._param_idx(param_dict)
        if self.store_raw:
            self._raw_store[idx] = raw
        results: dict[str, np.ndarray] = {}
        for od in self.outcomes:
            val = od.compute(raw)
            self._write_cell(od, idx, val)
            results[od.name] = val
        return results

    def get_raw(self, param_dict: dict[str, Any]) -> Any | None:
        """Stored raw object for a cell, or None if unrun / not retained."""
        return self._raw_store.get(self._param_idx(param_dict))

    def get_derived(
        self,
        param_dict: dict[str, Any],
    ) -> dict[str, np.ndarray] | None:
        """
        All derived values for a cell as {outcome_name: array}.
        Returns None if the cell has not been run.
        Scalar outcomes return a shape-() array.
        """
        idx = self._param_idx(param_dict)
        if not self._sub_tensors:
            return None
        if not any(not np.all(np.isnan(t[idx])) for t in self._sub_tensors.values()):
            return None
        return {name: t[idx].copy() for name, t in self._sub_tensors.items()}

    def get_outcome_tensor(self, outcome_name: str) -> np.ndarray:
        """
        Full sub-tensor for one outcome.
        Shape (*param_dims,) for scalars, (*param_dims, *out_shape) otherwise.
        NaN where unrun.
        """
        if outcome_name not in self._sub_tensors:
            raise KeyError(
                f"No sub-tensor for '{outcome_name}' — has any run been logged?"
            )
        return self._sub_tensors[outcome_name]

    def add_outcome(self, outcome_def: OutcomeDef) -> None:
        """
        Add a new OutcomeDef and back-fill all already-logged runs.
        Requires store_raw=True (default); warns and skips back-fill otherwise.
        """
        if outcome_def.name in self._sub_tensors:
            raise ValueError(f"Outcome '{outcome_def.name}' already exists")
        if not self._raw_store:
            if not self.store_raw:
                warnings.warn(
                    f"Cannot back-fill '{outcome_def.name}': store_raw=False.",
                    stacklevel=2,
                )
            self.outcomes.append(outcome_def)
            return
        for idx, raw in self._raw_store.items():
            val = outcome_def.compute(raw)
            self._write_cell(outcome_def, idx, val)
        self.outcomes.append(outcome_def)
        print(
            f"Added '{outcome_def.name}'  "
            f"out_shape={outcome_def.out_shape}  "
            f"→ sub-tensor {self._sub_tensors[outcome_def.name].shape}"
        )

    # ── Axis name resolution ─────────────────────────────────────────────────

    def _resolve_axes(
        self,
        od: "OutcomeDef",
        filter: dict[str, Any],
    ) -> tuple[dict[int, Any], dict[int, Any]]:
        """
        Translate a name-keyed axes dict into two index-keyed dicts:
            param_sel  : {param_position    → value_or_list}
            outcome_sel: {outcome_axis_index → value_or_list}

        Name resolution rules
        ---------------------
        Bare name
            Look up in parameters first, then in od.dim_axes.
            If found in both, raise AmbiguousAxisError — caller must qualify.
        "p:name"
            Force lookup in parameters only.
        "o:name"
            Force lookup in od.dim_axes only.

        Value semantics (applied later by select())
        --------------------------------------------
        Scalar  → axis collapsed (dimensionality reduced by 1).
        List    → axis kept, restricted to those coordinates (even if len 1).
        """
        p_names = {p.name: i for i, p in enumerate(self.parameters)}
        o_names = {ax.name: i for i, ax in enumerate(od.dim_axes)}

        param_sel: dict[int, Any] = {}
        outcome_sel: dict[int, Any] = {}

        for key, val in filter.items():
            if key.startswith("p:"):
                name = key[2:]
                if name not in p_names:
                    raise KeyError(
                        f"select: 'p:{name}' — no parameter named '{name}'. "
                        f"Parameters: {list(p_names)}"
                    )
                param_sel[p_names[name]] = val

            elif key.startswith("o:"):
                name = key[2:]
                if name not in o_names:
                    raise KeyError(
                        f"select: 'o:{name}' — outcome '{od.name}' has no "
                        f"dim_axis named '{name}'. "
                        f"Outcome axes: {list(o_names)}"
                    )
                outcome_sel[o_names[name]] = val

            else:
                in_p = key in p_names
                in_o = key in o_names
                if in_p and in_o:
                    raise ValueError(
                        f"select: axis name '{key}' is ambiguous — it exists "
                        f"as both a parameter and a dim_axis of outcome "
                        f"'{od.name}'. Qualify with 'p:{key}' or 'o:{key}'."
                    )
                if not in_p and not in_o:
                    raise KeyError(
                        f"select: '{key}' not found in parameters "
                        f"{list(p_names)} or in dim_axes of outcome "
                        f"'{od.name}' {list(o_names)}."
                    )
                if in_p:
                    param_sel[p_names[key]] = val
                else:
                    outcome_sel[o_names[key]] = val

        return param_sel, outcome_sel

    # ── Slicing & selection ───────────────────────────────────────────────────

    def select(
        self,
        outcome: str,
        filter: dict[str, Any] | None = None,
    ) -> SelectionResult:
        """
        Return a sub-tensor of one outcome, filtered and/or sliced by axis name.

        Parameters
        ----------
        outcome
            Name of the OutcomeDef to query.

        filter
            Dict of {axis_name: value_or_list}.

            axis_name
                Bare name resolved against parameters first, then the outcome's
                dim_axes.  Prefix with "p:" to force parameter lookup, "o:" to
                force outcome dim_axis lookup.  Qualification is only required
                when the same name exists in both.

            value_or_list
                Scalar  — axis is fixed to that value and collapsed from the
                          output (dimensionality reduced by 1).
                List    — axis is restricted to those values and kept in the
                          output (even a single-element list keeps the axis).

        Returns
        -------
        (tensor, axes_out)
            tensor    : ndarray whose shape corresponds to the remaining axes.
            axes_out  : list of AxisDef, one per output dimension, in order.
                        Scalar-fixed axes are absent; list-restricted axes
                        appear with values restricted to the selected subset.

        Examples
        --------
            # Fix two params (collapsed), restrict spectrum frequencies (kept)
            result = space.select(
                "spectrum",
                filter={
                    "thickness": 2,            # scalar → collapsed
                    "mode":      "low",        # scalar → collapsed
                    "frequency": [10, 50, 100],# list   → kept (3 values)
                }
            )
            # result.shape == (3, 2, 3, 4, 3)  [remaining params × 3 freq bins]

            # Ambiguous name — qualify explicitly
            result = space.select(
                "spectrum",
                filter={
                    "p:frequency": [10, 100],  # the parameter axis
                    "o:frequency": [10, 50],   # the outcome dim_axis
                }
            )
        """
        t = self.get_outcome_tensor(outcome)
        od = self._outcome_by_name(outcome)

        if not filter:
            # Return full tensor; axes_out = all param axes + all outcome axes
            axes_out = list(self.parameters) + list(od.dim_axes)
            return SelectionResult(outcome=outcome, tensor=t, axes=axes_out)

        param_sel, outcome_sel = self._resolve_axes(od, filter)

        # ── Build index tuple and axes_out in definition order ────────────────
        # Output axis order always mirrors the sub-tensor storage order:
        # (*param_dims, *outcome_dims).  select() subsets values; it never
        # reorders.  Reordering is the responsibility of squeeze().

        full_idx: list[Any] = []
        axes_out: list[AxisDef] = []

        for pi, p in enumerate(self.parameters):
            if pi not in param_sel:
                full_idx.append(slice(None))
                axes_out.append(p)
            else:
                val = param_sel[pi]
                if isinstance(val, list):
                    full_idx.append([p.index_of(v) for v in val])
                    idxs = [p.index_of(v) for v in val]
                    bare_vals = [p.values[i] for i in idxs]
                    axes_out.append(AxisDef(p.name, bare_vals, p.unit, p.scale))
                else:
                    full_idx.append(p.index_of(val))  # scalar → collapsed

        for oi, ax in enumerate(od.dim_axes):
            if oi not in outcome_sel:
                full_idx.append(slice(None))
                axes_out.append(ax)
            else:
                val = outcome_sel[oi]
                if isinstance(val, list):
                    full_idx.append([ax.index_of(v) for v in val])
                    axes_out.append(AxisDef(ax.name, val, ax.unit, ax.scale))
                else:
                    full_idx.append(ax.index_of(val))  # scalar → collapsed

        # ── Apply index preserving axis order ─────────────────────────────────
        # Apply one entry at a time, tracking the live dimension position.
        # np.take(axis=dim) restricts a list index on a specific axis without
        # disturbing any other axis — avoids numpy advanced-indexing promotion
        # that would move fancy-indexed axes to the front.
        result = t
        dim = 0  # current dim in result as preceding scalars collapse axes
        for entry in full_idx:
            if isinstance(entry, int):
                # Scalar → collapse: index at current dim, dim does not advance
                result = result[(slice(None),) * dim + (entry,)]
            elif isinstance(entry, list):
                # List → restrict in place, axis stays at dim
                result = np.take(result, entry, axis=dim)
                dim += 1
            else:
                # slice(None) → keep all, advance
                dim += 1

        return SelectionResult(outcome=outcome, tensor=result, axes=axes_out)

    # ── Introspection ─────────────────────────────────────────────────────────

    def sparsity(self) -> dict[str, Any]:
        n_total = int(np.prod(self._param_dims))
        if not self._sub_tensors:
            n_run = 0
        else:
            first = next(iter(self._sub_tensors.values()))
            run_mask = ~np.all(
                np.isnan(first.reshape(self._param_dims + (-1,))), axis=-1
            )
            n_run = int(run_mask.sum())
        return {
            "n_params": len(self.parameters),
            "param_dims": self._param_dims,
            "n_cells": n_total,
            "n_run": n_run,
            "n_unrun": n_total - n_run,
            "coverage": n_run / n_total,
            "n_outcomes": len(self.outcomes),
            "store_raw": self.store_raw,
        }

    def _outcome_by_name(self, name: str) -> OutcomeDef:
        for od in self.outcomes:
            if od.name == name:
                return od
        raise KeyError(f"No OutcomeDef named '{name}'")

    def __repr__(self) -> str:
        s = self.sparsity()

        params_str = " × ".join(p.axis_label for p in self.parameters)

        def _outcome_line(od: OutcomeDef) -> str:
            if not od.dim_axes:
                axes_str = "scalar"
            else:
                axes_str = "[" + ", ".join(a.axis_label for a in od.dim_axes) + "]"
            return (
                f"    {od.name}  "
                f"shape={od.out_shape}  "
                f"dir={od.direction}  "
                f"axes={axes_str}"
            )

        outcomes_str = "\n".join(_outcome_line(od) for od in self.outcomes)
        return (
            f"ExperimentSpace[\n"
            f"  params   : {params_str}\n"
            f"  dims     : {list(s['param_dims'])}  →  {s['n_cells']} cells\n"
            f"  run      : {s['n_run']} / {s['n_cells']}  "
            f"({100 * s['coverage']:.1f}% coverage)\n"
            f"  store_raw: {s['store_raw']}\n"
            f"  outcomes :\n{outcomes_str}\n"
            f"]"
        )


# ══════════════════════════════════════════════════════════════════════════════
# Demo
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    rng = np.random.default_rng(42)

    N, F = 99, 4
    SAMPLE_RATE = 1000
    FREQ_BINS = list(np.fft.rfftfreq(N + 1, d=1.0 / SAMPLE_RATE).astype(int))

    def make_raw(thickness: float) -> dict:
        ts = rng.random((N + 1, F)) * thickness
        ts += rng.standard_normal((N + 1, F)) * 0.02
        return {"timeseries": ts, "thickness_mm": thickness}

    # ── Parameters — ParameterDef inherits AxisDef ───────────────────────────

    parameters = [
        ParameterDef("thickness", [1, 2, 5], unit="mm"),
        ParameterDef("frequency", [10, 100, 1000], unit="Hz", scale="log"),
        ParameterDef("mode", ["low", "mid", "high"]),
        ParameterDef("temperature", [20, 37], unit="°C"),
        ParameterDef("material", ["A", "B", "C"]),
        ParameterDef("sample_rate", [1000, 2000, 4000, 8000], unit="Hz", scale="log"),
    ]

    # ── Shared AxisDef objects reused across outcomes ─────────────────────────
    # (same type as parameter axes — identical interface)

    FEAT_AXIS = AxisDef("feature", [f"feat_{i}" for i in range(F)])
    FREQ_AXIS = AxisDef("frequency", FREQ_BINS, unit="Hz", scale="log")
    STAT_AXIS = AxisDef("statistic", ["mean", "std"])

    # ── Outcomes — dim_axes is now list[AxisDef], same type as parameters ─────

    outcomes = [
        OutcomeDef(
            name="grand_mean",
            fn=lambda raw: np.asarray(raw["timeseries"].mean()),
            direction="max",
            dim_axes=[],  # scalar
        ),
        OutcomeDef(
            name="feature_means",
            fn=lambda raw: raw["timeseries"].mean(axis=0),
            direction="max",
            dim_axes=[FEAT_AXIS],  # shape (F,)
        ),
        OutcomeDef(
            name="spectrum",
            fn=lambda raw: np.abs(np.fft.rfft(raw["timeseries"][:, 0])),
            direction="max",
            dim_axes=[FREQ_AXIS],  # shape (51,)
        ),
        OutcomeDef(
            name="feature_stats",
            fn=lambda raw: np.stack(
                [
                    raw["timeseries"].mean(axis=0),
                    raw["timeseries"].std(axis=0),
                ],
                axis=-1,
            ),
            direction=None,
            dim_axes=[FEAT_AXIS, STAT_AXIS],  # shape (F, 2)
        ),
    ]

    space = ExperimentSpace(parameters, outcomes, store_raw=True)

    # ── Log runs ──────────────────────────────────────────────────────────────

    demo_runs = [
        {
            "thickness": 1,
            "frequency": 10,
            "mode": "low",
            "temperature": 20,
            "material": "A",
            "sample_rate": 1000,
        },
        {
            "thickness": 2,
            "frequency": 10,
            "mode": "mid",
            "temperature": 37,
            "material": "B",
            "sample_rate": 2000,
        },
        {
            "thickness": 5,
            "frequency": 100,
            "mode": "high",
            "temperature": 20,
            "material": "A",
            "sample_rate": 4000,
        },
        {
            "thickness": 2,
            "frequency": 1000,
            "mode": "low",
            "temperature": 37,
            "material": "C",
            "sample_rate": 8000,
        },
        {
            "thickness": 1,
            "frequency": 100,
            "mode": "mid",
            "temperature": 20,
            "material": "B",
            "sample_rate": 1000,
        },
    ]

    print("── Logging runs ─────────────────────────────────")
    for p in demo_runs:
        result = space.log_run(p, make_raw(p["thickness"]))
        print(
            f"  {p['thickness']}mm / {p['frequency']}Hz / {p['mode']}"
            f"  grand_mean={result['grand_mean']:.4f}"
            f"  spectrum[:3]={np.round(result['spectrum'][:3], 2)}"
        )

    print(f"\n{space}\n")

    # ── from_strings classmethod ──────────────────────────────────────────────

    print("── AxisDef.from_strings ─────────────────────────")

    cases = [
        ("thickness", ["1mm", "2mm", "5mm"]),
        ("cap", ["0.01pF", "0.1pF", "0.5pF"]),
        ("freq", ["10Hz", "100Hz", "1000Hz"]),
        ("temp", ["-10°C", "20°C", "37°C"]),
        ("gap", ["1.5 mm", "2.5 mm", "5.0 mm"]),
        ("freq_log", ["10Hz", "100Hz", "1000Hz"], "log"),  # with scale
    ]

    for args in cases:
        name_, strings_, *rest = args
        scale_ = rest[0] if rest else "linear"
        ax = AxisDef.from_strings(name_, strings_, scale=scale_)
        print(f"  {ax!r}")
        print(f"    values={ax.values}  labels={ax.labels}")

    # ParameterDef.from_strings returns a ParameterDef, not a bare AxisDef
    print()
    p = ParameterDef.from_strings("thickness", ["1mm", "2mm", "5mm"])
    print(f"  type : {type(p).__name__}")
    print(f"  repr : {p!r}")
    print(f"  index_of(2) = {p.index_of(2)}")

    # Mixed units → ValueError
    print()
    try:
        AxisDef.from_strings("bad", ["1mm", "2cm"])
    except ValueError as e:
        print(f"  mixed units caught: {e}")

    # Unparseable string → ValueError
    try:
        AxisDef.from_strings("cat", ["low", "mid", "high"])
    except ValueError as e:
        print(f"  categorical caught: {e}")

    # ── Uniform axis interface across parameters and outcome axes ─────────────

    print("\n── All axes share the same AxisDef interface ────")
    all_axes: list[AxisDef] = list(parameters) + [
        ax for od in outcomes for ax in od.dim_axes
    ]
    for ax in all_axes:
        print(
            f"  {ax.axis_label:25s}  "
            f"n={ax.n:3d}  "
            f"scale={ax.scale:6s}  "
            f"labels[:2]={ax.labels[:2]}"
        )

    # ── Sub-tensor shapes ─────────────────────────────────────────────────────

    print("\n── Sub-tensor shapes ────────────────────────────")
    for name, t in space._sub_tensors.items():
        print(f"  {name:20s}  {t.shape}")

    # ── get_derived ───────────────────────────────────────────────────────────

    query = {
        "thickness": 2,
        "frequency": 10,
        "mode": "mid",
        "temperature": 37,
        "material": "B",
        "sample_rate": 2000,
    }

    d = space.get_derived(query)
    print(f"\n── get_derived  2mm / 10Hz / mid")
    for k, v in d.items():
        print(
            f"  {k:20s}  shape={str(v.shape):10s}"
            f"  val={np.round(v.flat[:3], 3)}{'...' if v.size > 3 else ''}"
        )

    # ── select() — name-based, unified parameter + outcome axes ─────────────

    print("\n── select: fix params (scalar), restrict freq (list) ──")
    r1 = space.select(
        "spectrum",
        filter={
            "mode": "low",  # scalar → axis collapsed
            "temperature": 20,  # scalar → axis collapsed
            "material": "A",  # scalar → axis collapsed
            "sample_rate": 1000,  # scalar → axis collapsed
            "p:frequency": 10,  # param freq — scalar → collapsed
            "o:frequency": [10, 50, 100],  # outcome freq bins — list → kept
        },
    )
    print(f"   {r1!r}")
    print(f"   dim('thickness') = {r1.dim('thickness')}")
    print(f"   coords(1)        = {r1.coords(1)}")
    print(f"   labels(1)        = {r1.labels(1)}")
    # → (thickness, frequency[outcome=3]) remaining

    print("\n── select: restrict both a param and outcome axis ──")
    r2 = space.select(
        "spectrum",
        filter={
            "thickness": [1, 5],  # list → kept, 2 thicknesses
            "mode": "low",
            "temperature": 20,
            "material": "A",
            "sample_rate": 1000,
            "p:frequency": 10,  # fix param freq
            "o:frequency": [10, 50, 100],  # list → kept, 3 freq bins
        },
    )
    print(f"   {r2!r}")
    print(f"   r2['thickness'].labels = {r2['thickness'].labels}")
    print(f"   r2['frequency'].labels = {r2['frequency'].labels}")

    print("\n── select: scalar collapse on outcome axis ──────────")
    r3 = space.select(
        "feature_stats",
        filter={
            "feature": "feat_2",  # outcome dim_axis — scalar → collapsed
            "statistic": "std",  # outcome dim_axis — scalar → collapsed
        },
    )
    print(f"   {r3!r}")

    print("\n── select: list on outcome axis, keep axis ──────────")
    r4 = space.select(
        "feature_stats",
        filter={
            "statistic": ["mean", "std"],  # list → kept
        },
    )
    print(f"   {r4!r}")

    print("\n── select: ambiguous name → qualify with p: / o: ───")
    # "frequency" exists as both a parameter and spectrum's dim_axis
    try:
        space.select("spectrum", filter={"frequency": [10, 50]})
    except ValueError as e:
        print(f"   ambiguous: {e}")

    r5 = space.select(
        "spectrum",
        filter={
            "p:frequency": [10, 100],  # the parameter axis
            "o:frequency": [10, 50],  # the spectrum dim_axis (freq bins)
            "mode": "low",
            "temperature": 20,
            "material": "A",
            "sample_rate": 1000,
        },
    )
    print(f"   {r5!r}")

    # ── index_of with annotated strings ──────────────────────────────────────

    print("── index_of: annotated string tolerance ────────────")
    thick_p = next(p for p in parameters if p.name == "thickness")

    # All of these should resolve to index 1  (bare value 2, unit mm)
    for v in [2, "2mm", "2 mm"]:
        print(f"   index_of({v!r:8}) = {thick_p.index_of(v)}")

    # log_run with annotated strings in param_dict
    print()
    print("   log_run with annotated param_dict:")
    annotated_run = {
        "thickness": "5mm",  # annotated string
        "frequency": "100Hz",  # annotated string
        "mode": "high",  # bare (categorical — no unit)
        "temperature": "20°C",  # annotated string
        "material": "A",  # bare (categorical)
        "sample_rate": "4000Hz",  # annotated string
    }
    result = space.log_run(annotated_run, make_raw(5))
    print(f"   grand_mean = {result['grand_mean']:.4f}")

    # get_derived with annotated strings — same cell
    d_ann = space.get_derived(annotated_run)
    d_bare = space.get_derived(
        {
            "thickness": 5,
            "frequency": 100,
            "mode": "high",
            "temperature": 20,
            "material": "A",
            "sample_rate": 4000,
        }
    )
    print(
        f"   annotated get_derived matches bare: "
        f"{np.allclose(d_ann['grand_mean'], d_bare['grand_mean'])}"
    )

    # Unit mismatch → hard error
    print()
    try:
        thick_p.index_of("2cm")
    except KeyError as e:
        print(f"   unit mismatch: {e}")

    # Annotated string on no-unit axis → hard error
    try:
        mode_p = next(p for p in parameters if p.name == "mode")
        mode_p.index_of("1low")
    except KeyError as e:
        print(f"   no-unit axis:  {e}")

    # ── squeeze() ─────────────────────────────────────────────────────────────

    print("\n── squeeze() ────────────────────────────────────")

    # Start with a selection that has multiple free axes
    r_multi = space.select(
        "spectrum",
        filter={
            "p:frequency": 10,  # fix drive frequency
            "o:frequency": FREQ_BINS[:10],  # keep first 10 freq bins
        },
    )
    print(f"   before squeeze: {r_multi!r}")
    # shape: (thickness=3, mode=3, temperature=2, material=3, sample_rate=4, freq_bins=10)

    # Keep only freq bins as independent; fold everything else into combined
    sq1 = r_multi.squeeze(["frequency"])
    print(f"\n   squeeze(['frequency']):")
    print(f"   shape               : {sq1.shape}")
    print(f"   axes                : {[a.name for a in sq1.axes]}")
    print(f"   combined n          : {sq1.axes[0].n}")
    print(f"   combined values[:2] : {sq1.axes[0].values[:2]}")
    print(f"   combined labels[:2] : {sq1.axes[0].labels[:2]}")
    print(f"   freq bin labels[:4] : {sq1.axes[1].labels[:4]}")

    # Keep two independent axes: thickness and freq bins
    sq2 = r_multi.squeeze(["thickness", "frequency"])
    print(f"\n   squeeze(['thickness', 'frequency']):")
    print(f"   shape               : {sq2.shape}")
    print(f"   axes                : {[a.name for a in sq2.axes]}")
    print(f"   combined n          : {sq2.axes[0].n}")
    print(f"   combined vals[:2]   : {sq2.axes[0].values[:2]}")
    print(f"   combined lbl[:2]    : {sq2.axes[0].labels[:2]}")
    print(f"   thickness labels    : {sq2.axes[1].labels}")
    print(f"   freq bin labels[:4] : {sq2.axes[2].labels[:4]}")

    # All axes named — reorder only, no combined axis
    r_small = space.select(
        "spectrum",
        filter={
            "p:frequency": 10,
            "mode": "low",
            "temperature": 20,
            "material": "A",
            "sample_rate": 1000,
            "o:frequency": FREQ_BINS[:5],
        },
    )
    print(f"\n   before reorder: {r_small!r}")
    sq3 = r_small.squeeze(["frequency", "thickness"])  # reorder: freq first
    print(f"   squeeze(['frequency','thickness']) — reorder only:")
    print(f"   shape : {sq3.shape}")
    print(f"   axes  : {[a.name for a in sq3.axes]}")

    # ── complex dtype ────────────────────────────────────────────────────────

    print("\n── complex dtype ────────────────────────────────")

    space.add_outcome(
        OutcomeDef(
            name="fft_complex",
            fn=lambda raw: np.fft.rfft(raw["timeseries"][:, 0]),
            direction=None,
            dim_axes=[FREQ_AXIS],
        )
    )

    d_cx = space.get_derived(query)
    cx = d_cx["fft_complex"]
    print(f"   dtype  : {cx.dtype}")
    print(f"   shape  : {cx.shape}")
    print(f"   val[:3]: {np.round(cx[:3], 2)}")
    print(f"   sub-tensor dtype: {space._sub_tensors['fft_complex'].dtype}")

    # Real and complex outcomes coexist — check real outcome unaffected
    print(f"   grand_mean dtype: {space._sub_tensors['grand_mean'].dtype}")

    # ── add_outcome on the fly ────────────────────────────────────────────────

    print("\n── add_outcome 'energy' (scalar) ────────────────")
    space.add_outcome(
        OutcomeDef(
            name="energy",
            fn=lambda raw: np.asarray((raw["timeseries"] ** 2).sum()),
            direction="min",
            dim_axes=[],
        )
    )
    d2 = space.get_derived(query)
    print(f"   energy = {d2['energy']:.4f}")

    print(f"\n{space}")

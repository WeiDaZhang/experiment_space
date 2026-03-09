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
- select() resolves axis names from parameters and the chosen outcome's dim_axes.
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
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
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
        result = np.asarray(self.fn(raw), dtype=float)

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

        return result

    @property
    def out_shape(self) -> tuple[int, ...]:
        """Output array shape.  Available after the first compute() call."""
        return self._out_shape

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
# SelectionResult  —  named container returned by ExperimentSpace.select()
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class SelectionResult:
    """
    Container for a sub-tensor returned by ExperimentSpace.select().

    Bundles the numeric data with its full axis metadata so the two never
    become separated when passed between functions or plotting code.

    Attributes
    ----------
    outcome : str
        Name of the OutcomeDef that produced this result.
    tensor  : np.ndarray
        The selected sub-tensor.  tensor.shape[i] corresponds to axes[i].
    axes    : list[AxisDef]
        One AxisDef per tensor dimension, in order.
        Scalar-fixed axes are absent; list-restricted axes carry the subset
        of values that was selected.

    Axis access
    -----------
    result[name]          AxisDef by name (bare or "p:"/"o:" qualified).
    result.dim(name)      Integer dimension index of a named axis.
    result.coords(dim)    Coordinate values along a dimension (int or name).
    result.labels(dim)    Display strings for coordinates along a dimension.
    result.axis_label(dim)  Axis title string, e.g. "frequency (Hz)".

    Plotting
    --------
    result.plot()         Auto-dispatch: line (1D), heatmap (2D), subplots (3D).
                          Returns the matplotlib Figure.
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

    # ── Plotting ──────────────────────────────────────────────────────────────

    def plot(
        self,
        title: str | None = None,
        figsize: tuple[float, float] | None = None,
        cmap: str = "viridis",
    ) -> "plt.Figure":
        """
        Auto-dispatch plot based on tensor dimensionality.

        1-D  → line plot.
        2-D  → heatmap (imshow).
        3-D  → row of heatmaps, one per slice along axis 0.
        >3-D → raises NotImplementedError; reduce with select() first.

        Parameters
        ----------
        title   : overall figure title; defaults to the outcome name.
        figsize : passed to plt.figure(); auto-sized if None.
        cmap    : matplotlib colormap name for heatmaps.

        Returns the matplotlib Figure.
        """
        ndim = self.tensor.ndim
        if ndim == 0:
            raise ValueError(
                "SelectionResult.plot(): tensor is scalar (0-D). Nothing to plot."
            )
        if ndim > 3:
            raise NotImplementedError(
                f"SelectionResult.plot(): tensor is {ndim}-D. "
                f"Reduce to ≤3-D with select() before plotting."
            )

        fig_title = title or self.outcome

        if ndim == 1:
            return self._plot_1d(fig_title, figsize)
        elif ndim == 2:
            return self._plot_2d(fig_title, figsize, cmap)
        else:
            return self._plot_3d(fig_title, figsize, cmap)

    # ── Internal plot helpers ─────────────────────────────────────────────────

    @staticmethod
    def _apply_scale(ax_mpl, axis: "AxisDef", which: str) -> None:
        """Apply log/linear scale and label to a matplotlib axis."""
        set_scale = ax_mpl.set_xscale if which == "x" else ax_mpl.set_yscale
        set_label = ax_mpl.set_xlabel if which == "x" else ax_mpl.set_ylabel
        if axis.scale == "log":
            set_scale("log")
        set_label(axis.axis_label)

    def _plot_1d(self, title: str, figsize) -> "plt.Figure":
        ax_def = self.axes[0]
        xs = ax_def.values
        ys = self.tensor

        fig, ax = plt.subplots(figsize=figsize or (7, 4))
        ax.plot(xs, ys, marker="o", linewidth=1.5, markersize=4)
        ax.set_xticks(xs)
        ax.set_xticklabels(ax_def.labels, rotation=45, ha="right")
        self._apply_scale(ax, ax_def, "x")
        ax.set_ylabel(self.outcome)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig

    def _plot_2d(self, title: str, figsize, cmap: str) -> "plt.Figure":
        row_def = self.axes[0]  # y axis
        col_def = self.axes[1]  # x axis
        data = self.tensor  # shape (n_rows, n_cols)

        fig, ax = plt.subplots(figsize=figsize or (7, 5))
        im = ax.imshow(
            data,
            aspect="auto",
            origin="lower",
            cmap=cmap,
            interpolation="nearest",
        )
        fig.colorbar(im, ax=ax, label=self.outcome)

        ax.set_xticks(range(len(col_def.values)))
        ax.set_xticklabels(col_def.labels, rotation=45, ha="right")
        ax.set_yticks(range(len(row_def.values)))
        ax.set_yticklabels(row_def.labels)
        ax.set_xlabel(col_def.axis_label)
        ax.set_ylabel(row_def.axis_label)
        ax.set_title(title)
        fig.tight_layout()
        return fig

    def _plot_3d(self, title: str, figsize, cmap: str) -> "plt.Figure":
        slice_def = self.axes[0]  # one panel per value on this axis
        row_def = self.axes[1]
        col_def = self.axes[2]
        n_slices = len(slice_def.values)

        fig, axes_mpl = plt.subplots(
            1,
            n_slices,
            figsize=figsize or (4 * n_slices, 4),
            squeeze=False,
        )
        vmin = np.nanmin(self.tensor)
        vmax = np.nanmax(self.tensor)

        for si, (val, ax) in enumerate(zip(slice_def.values, axes_mpl[0])):
            im = ax.imshow(
                self.tensor[si],
                aspect="auto",
                origin="lower",
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                interpolation="nearest",
            )
            ax.set_xticks(range(len(col_def.values)))
            ax.set_xticklabels(col_def.labels, rotation=45, ha="right")
            ax.set_yticks(range(len(row_def.values)))
            ax.set_yticklabels(row_def.labels if si == 0 else [])
            ax.set_xlabel(col_def.axis_label)
            if si == 0:
                ax.set_ylabel(row_def.axis_label)
            ax.set_title(f"{slice_def.axis_label} = {slice_def.label(val)}")

        fig.colorbar(im, ax=axes_mpl[0, -1], label=self.outcome)
        fig.suptitle(title, y=1.02)
        fig.tight_layout()
        return fig

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
            self._sub_tensors[od.name] = np.full(shape, np.nan)

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
        axes: dict[str, Any],
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

        for key, val in axes.items():
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
        axes: dict[str, Any] | None = None,
    ) -> SelectionResult:
        """
        Return a sub-tensor of one outcome, filtered and/or sliced by axis name.

        Parameters
        ----------
        outcome
            Name of the OutcomeDef to query.

        axes
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
            t, axes_out = space.select(
                "spectrum",
                axes={
                    "thickness": 2,            # scalar → collapsed
                    "mode":      "low",        # scalar → collapsed
                    "frequency": [10, 50, 100],# list   → kept (3 values)
                }
            )
            # t.shape == (3, 2, 3, 4, 3)  [remaining params × 3 freqs]

            # Ambiguous name — qualify explicitly
            t, axes_out = space.select(
                "spectrum",
                axes={
                    "p:frequency": [10, 100],  # the parameter axis
                    "o:frequency": [10, 50],   # the outcome dim_axis
                }
            )
        """
        t = self.get_outcome_tensor(outcome)
        od = self._outcome_by_name(outcome)

        if not axes:
            # Return full tensor; axes_out = all param axes + all outcome axes
            axes_out = list(self.parameters) + list(od.dim_axes)
            return SelectionResult(outcome=outcome, tensor=t, axes=axes_out)

        param_sel, outcome_sel = self._resolve_axes(od, axes)

        n_param = len(self._param_dims)

        # ── Build numpy index tuple ───────────────────────────────────────────
        # We index the full sub-tensor shape: (*param_dims, *out_shape)
        # For each axis position:
        #   not in sel  → slice(None)   keep all
        #   scalar sel  → int           collapse axis
        #   list sel    → list[int]     fancy index, keep axis

        full_idx: list[Any] = []
        axes_out: list[AxisDef] = []

        for pi, p in enumerate(self.parameters):
            if pi not in param_sel:
                full_idx.append(slice(None))
                axes_out.append(p)
            else:
                val = param_sel[pi]
                if isinstance(val, list):
                    idxs = [p.index_of(v) for v in val]
                    full_idx.append(idxs)
                    # Restricted AxisDef: same metadata, subset of values
                    axes_out.append(AxisDef(p.name, val, p.unit, p.scale))
                else:
                    full_idx.append(p.index_of(val))
                    # scalar → axis collapsed, not added to axes_out

        for oi, ax in enumerate(od.dim_axes):
            if oi not in outcome_sel:
                full_idx.append(slice(None))
                axes_out.append(ax)
            else:
                val = outcome_sel[oi]
                if isinstance(val, list):
                    idxs = [ax.index_of(v) for v in val]
                    full_idx.append(idxs)
                    axes_out.append(AxisDef(ax.name, val, ax.unit, ax.scale))
                else:
                    full_idx.append(ax.index_of(val))
                    # scalar → collapsed

        # numpy fancy indexing with mixed slice/int/list requires np.ix_
        # for the list dimensions; we use a two-pass approach:
        # first apply scalar/slice indices, then apply fancy indices.
        result = t

        # Pass 1: apply scalars and slices (preserves axis order)
        scalar_idx = tuple(
            (slice(None) if isinstance(i, list) else i) for i in full_idx
        )
        result = result[scalar_idx]

        # Pass 2: apply list (fancy) indices.
        # After pass 1, scalar axes are gone; rebuild position mapping.
        surviving = [i for i in full_idx if not isinstance(i, int)]
        list_positions = [pos for pos, i in enumerate(surviving) if isinstance(i, list)]
        if list_positions:
            # Build np.ix_ grid over the list dimensions only
            list_idx_values = [surviving[pos] for pos in list_positions]
            # Construct full index: slice(None) for non-list dims, ix_ for list dims
            grid = np.ix_(*list_idx_values)
            grid_idx = [slice(None)] * result.ndim
            for pos, g in zip(list_positions, grid):
                grid_idx[pos] = g
            result = result[tuple(grid_idx)]

        return SelectionResult(outcome=outcome, tensor=result, axes=axes_out)

    def slice_2d(
        self,
        outcome: str,
        row: str,
        col: str,
        fixed: dict[str, Any] | None = None,
    ) -> SelectionResult:
        """
        Convenience wrapper around select() for the common 2-D heatmap case.

        Fix all axes except two (row and col), return a matrix and the two
        AxisDef objects so the caller has labels, unit, and scale directly.

        row / col may be parameter names or outcome dim_axis names (with the
        same "p:" / "o:" qualification rules as select()).

        fixed
            Values for every axis *not* named in row or col.  Scalars only
            (each fixes and collapses one axis).  May be omitted if row and
            col together account for all axes.

        Returns
        -------
        (matrix, row_axisdef, col_axisdef)
            matrix is 2-D; NaN where no run has been logged.
        """
        all_axes = {**(fixed or {}), row: None, col: None}
        # Build axes dict: fixed axes get scalar values (collapsed),
        # row/col get None as sentinels — we pass slice(None) for them.
        axes_dict: dict[str, Any] = {}
        if fixed:
            axes_dict.update(fixed)  # scalar → will be collapsed

        # Call select with only the fixed axes; row and col remain free
        sel = self.select(outcome, axes_dict if axes_dict else None)
        t, axes_out = sel.tensor, sel.axes

        # Find row and col positions in axes_out
        def _find(name: str) -> int:
            bare = name[2:] if name.startswith(("p:", "o:")) else name
            for i, ax in enumerate(axes_out):
                if ax.name == bare:
                    return i
            raise KeyError(
                f"slice_2d: '{name}' not found in remaining axes "
                f"{[a.name for a in axes_out]} after fixing {list(fixed or {})}"
            )

        ri = _find(row)
        ci = _find(col)

        if ri == ci:
            raise ValueError(f"slice_2d: row and col refer to the same axis '{row}'")

        # Collapse all remaining axes except row and col by taking index 0
        keep = {ri, ci}
        idx: list[Any] = []
        for i in range(t.ndim):
            if i in keep:
                idx.append(slice(None))
            else:
                idx.append(0)
        mat = t[tuple(idx)]

        # Ensure row is axis 0, col is axis 1
        # After collapsing, row and col are the only two axes left;
        # figure out their new positions
        surviving = sorted(keep)
        new_ri = surviving.index(ri)
        new_ci = surviving.index(ci)
        if new_ri > new_ci:
            mat = mat.T

        return SelectionResult(
            outcome=outcome,
            tensor=mat,
            axes=[axes_out[ri], axes_out[ci]],
        )

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
        axes={
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
        axes={
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
        axes={
            "feature": "feat_2",  # outcome dim_axis — scalar → collapsed
            "statistic": "std",  # outcome dim_axis — scalar → collapsed
        },
    )
    print(f"   {r3!r}")

    print("\n── select: list on outcome axis, keep axis ──────────")
    r4 = space.select(
        "feature_stats",
        axes={
            "statistic": ["mean", "std"],  # list → kept
        },
    )
    print(f"   {r4!r}")

    print("\n── select: ambiguous name → qualify with p: / o: ───")
    # "frequency" exists as both a parameter and spectrum's dim_axis
    try:
        space.select("spectrum", axes={"frequency": [10, 50]})
    except ValueError as e:
        print(f"   ambiguous: {e}")

    r5 = space.select(
        "spectrum",
        axes={
            "p:frequency": [10, 100],  # the parameter axis
            "o:frequency": [10, 50],  # the spectrum dim_axis (freq bins)
            "mode": "low",
            "temperature": 20,
            "material": "A",
            "sample_rate": 1000,
        },
    )
    print(f"   {r5!r}")

    # ── slice_2d — now uses axis names for row/col ────────────────────────────

    print("\n── slice_2d: grand_mean, thickness × frequency ──────")
    s1 = space.slice_2d(
        outcome="grand_mean",
        row="thickness",
        col="frequency",
        fixed={"mode": "low", "temperature": 20, "material": "A", "sample_rate": 1000},
    )
    print(f"   {s1!r}")
    print(f"   {s1.axis_label(0)}: {s1.labels(0)}")
    print(f"   {s1.axis_label(1)}: {s1.labels(1)}")
    print(f"   matrix:\n   {np.round(s1.tensor, 4)}")

    print("\n── slice_2d: spectrum, thickness × freq bin ─────────")
    # Row = thickness (param), col = frequency (outcome dim_axis)
    # "frequency" is ambiguous — qualify
    s2 = space.slice_2d(
        outcome="spectrum",
        row="thickness",
        col="o:frequency",  # outcome dim_axis, not param
        fixed={
            "p:frequency": 10,
            "mode": "low",
            "temperature": 20,
            "material": "A",
            "sample_rate": 1000,
        },
    )
    print(f"   {s2!r}")
    print(f"   matrix (first 5 cols):\n   {np.round(s2.tensor[:, :5], 4)}")

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

    # ── SelectionResult.plot() ───────────────────────────────────────────────

    print("\n── plot() demonstrations ────────────────────────")

    # 1-D: feature_means for a single cell — all features along one run
    r_1d = space.select(
        "feature_means",
        axes={
            "thickness": 2,
            "p:frequency": 10,
            "mode": "mid",
            "temperature": 37,
            "material": "B",
            "sample_rate": 2000,
        },
    )
    print(f"   1-D: {r_1d!r}")
    fig1 = r_1d.plot(title="Feature means — 2mm/10Hz/mid")
    plt.show()
    # fig1.savefig("/mnt/user-data/outputs/plot_1d.png", dpi=120, bbox_inches="tight")
    print("   saved plot_1d.png")

    # 2-D: spectrum outcome, thickness × frequency bins (restricted)
    r_2d = space.select(
        "spectrum",
        axes={
            "p:frequency": 10,
            "mode": "low",
            "temperature": 20,
            "material": "A",
            "sample_rate": 1000,
            "o:frequency": FREQ_BINS[:20],  # first 20 freq bins
        },
    )
    print(f"   2-D: {r_2d!r}")
    fig2 = r_2d.plot(title="Spectrum — thickness × frequency bins")
    plt.show()
    # fig2.savefig("/mnt/user-data/outputs/plot_2d.png", dpi=120, bbox_inches="tight")
    print("   saved plot_2d.png")

    # 3-D: feature_stats — thickness × feature × statistic, fix remaining params
    r_3d = space.select(
        "feature_stats",
        axes={
            "thickness": [1, 2, 5],  # list → kept (slices)
            "p:frequency": 10,
            "mode": "low",
            "temperature": 20,
            "material": "A",
            "sample_rate": 1000,
            # keep both outcome axes: feature(4) and statistic(2)
        },
    )
    print(f"   3-D: {r_3d!r}")
    fig3 = r_3d.plot(title="Feature stats — thickness × feature × statistic")
    plt.show()
    # fig3.savefig("/mnt/user-data/outputs/plot_3d.png", dpi=120, bbox_inches="tight")
    print("   saved plot_3d.png")

    plt.close("all")

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

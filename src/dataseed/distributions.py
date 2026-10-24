"""Marginal distributions used by the generator.

Every marginal exposes ``ppf(u)`` (quantile function, ``u`` in (0, 1)) so that the same object
can be driven by independent uniforms or by a Gaussian copula. ``values(u)`` additionally
handles rounding and bounds. Bounded sampling is done by *truncating the distribution* (mapping
``u`` into the allowed probability interval) rather than clipping, so no artificial piles of
values appear at the boundaries. Integers use the interval [min-0.5, max+0.5] before rounding so
that every integer in the range receives its fair share of probability.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats

from .exceptions import GenerationError
from .schema import ColumnSpec, parse_bound

_EPS = 1e-12


def to_internal(value, col_type: str) -> float | None:
    """Convert a bound/value to the numeric scale used internally (days / seconds / number)."""
    if value is None:
        return None
    seconds = parse_bound(value, col_type) if col_type in ("date", "datetime") else None
    if col_type == "date":
        return float(round(seconds / 86400.0))
    if col_type == "datetime":
        return float(round(seconds))
    return float(value)


def from_internal(values: np.ndarray, col_type: str) -> pd.Series:
    """Convert internal numeric values back to dates/datetimes (or return numbers)."""
    if col_type == "date":
        return pd.Series(pd.to_datetime(values.astype("int64"), unit="D"))
    if col_type == "datetime":
        return pd.Series(pd.to_datetime(values.astype("int64"), unit="s"))
    return pd.Series(values)


class Marginal:
    """Base class. Subclasses implement :meth:`_ppf`."""

    kind = "base"

    def __init__(self, integer: bool, lo: float | None, hi: float | None, decimals: int | None = None):
        self.integer, self.lo, self.hi, self.decimals = integer, lo, hi, decimals
        self.notes: list[str] = []

    # continuous support (integers extend by half a unit so rounding is unbiased)
    @property
    def clo(self) -> float:
        if self.lo is None:
            return -np.inf
        return math.ceil(self.lo) - 0.5 if self.integer else self.lo

    @property
    def chi(self) -> float:
        if self.hi is None:
            return np.inf
        return math.floor(self.hi) + 0.5 if self.integer else self.hi

    def _ppf(self, u: np.ndarray) -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError

    def ppf(self, u: np.ndarray) -> np.ndarray:
        return self._ppf(np.clip(u, _EPS, 1 - _EPS))

    def values(self, u: np.ndarray) -> np.ndarray:
        x = self.ppf(u)
        if self.integer:
            x = np.rint(x)
        elif self.decimals is not None:
            x = np.round(x, self.decimals)
        lo = math.ceil(self.lo) if (self.integer and self.lo is not None) else self.lo
        hi = math.floor(self.hi) if (self.integer and self.hi is not None) else self.hi
        return np.clip(x, lo if lo is not None else -np.inf, hi if hi is not None else np.inf)

    def describe(self) -> str:
        return self.kind


class Constant(Marginal):
    kind = "constant"

    def __init__(self, value: float, integer: bool, decimals: int | None = None):
        super().__init__(integer, value, value, decimals)
        self.value = value

    def _ppf(self, u):
        return np.full(u.shape, self.value, dtype=float)

    def describe(self) -> str:
        return f"constant value {self.value:g}"


class Uniform(Marginal):
    kind = "uniform"

    def _ppf(self, u):
        return self.clo + u * (self.chi - self.clo)

    def describe(self) -> str:
        return f"uniform on [{self.lo:g}, {self.hi:g}]"


class TruncatedNormal(Marginal):
    kind = "normal"

    def __init__(self, mean, std, integer, lo, hi, decimals=None):
        super().__init__(integer, lo, hi, decimals)
        self.mean, self.std = float(mean), float(std)
        za = (self.clo - self.mean) / self.std
        zb = (self.chi - self.mean) / self.std
        self.pa, self.pb = stats.norm.cdf(za), stats.norm.cdf(zb)
        self.degenerate = (self.pb - self.pa) < 1e-9
        if self.degenerate:
            self.notes.append("normal mass inside the bounds is negligible; fell back to uniform within bounds")

    def _ppf(self, u):
        if self.degenerate:
            lo = self.clo if np.isfinite(self.clo) else self.mean
            hi = self.chi if np.isfinite(self.chi) else self.mean
            return lo + u * (hi - lo)
        return self.mean + self.std * stats.norm.ppf(self.pa + u * (self.pb - self.pa))

    def describe(self) -> str:
        b = "" if self.lo is None and self.hi is None else f" truncated to [{self.lo}, {self.hi}]"
        return f"normal(mean={self.mean:g}, std={self.std:g}){b}"


class TruncatedLogNormal(Marginal):
    kind = "lognormal"

    def __init__(self, mean, std, integer, lo, hi, decimals=None):
        super().__init__(integer, lo, hi, decimals)
        s2 = math.log(1 + (std / mean) ** 2)
        self.sigma, self.mu = math.sqrt(s2), math.log(mean) - s2 / 2
        self.mean, self.std = mean, std

        def cdf(x):
            return 0.0 if x <= 0 else stats.norm.cdf((math.log(x) - self.mu) / self.sigma) if np.isfinite(x) else 1.0
        self.pa, self.pb = cdf(self.clo), cdf(self.chi)
        self.degenerate = (self.pb - self.pa) < 1e-9
        if self.degenerate:
            self.notes.append("lognormal mass inside the bounds is negligible; fell back to uniform within bounds")

    def _ppf(self, u):
        if self.degenerate:
            return self.clo + u * (self.chi - self.clo)
        return np.exp(self.mu + self.sigma * stats.norm.ppf(self.pa + u * (self.pb - self.pa)))

    def describe(self) -> str:
        return f"lognormal(mean={self.mean:g}, std={self.std:g}) bounded to [{self.lo}, {self.hi}]"


class Empirical(Marginal):
    """Smoothed empirical distribution: linear interpolation of stored sample quantiles."""

    kind = "empirical"

    def __init__(self, quantile_values, integer, lo, hi, decimals=None):
        super().__init__(integer, lo, hi, decimals)
        self.q = np.sort(np.asarray(quantile_values, dtype=float))
        self.p = np.linspace(0, 1, len(self.q))

    def _ppf(self, u):
        return np.interp(u, self.p, self.q)

    def describe(self) -> str:
        return f"empirical (interpolated between {len(self.q)} fitted quantiles)"


class Discrete(Marginal):
    kind = "discrete"

    def __init__(self, values, probs, integer, decimals=None):
        order = np.argsort(values)
        self.v = np.asarray(values, dtype=float)[order]
        p = np.asarray(probs, dtype=float)[order]
        self.cum = np.cumsum(p / p.sum())
        super().__init__(integer, float(self.v.min()), float(self.v.max()), decimals)

    def _ppf(self, u):
        idx = np.minimum(np.searchsorted(self.cum, u, side="right"), len(self.v) - 1)
        return self.v[idx]

    def values(self, u):
        return self._ppf(np.clip(u, _EPS, 1 - _EPS))

    def describe(self) -> str:
        return f"discrete choice among {len(self.v)} allowed values"


def build_marginal(col: ColumnSpec) -> Marginal:
    """Create the marginal distribution for a numeric/date/datetime column spec."""
    t = col.type
    integer = t in ("integer", "date", "datetime")
    decimals = col.decimals if t == "float" else None
    lo, hi = to_internal(col.min, t), to_internal(col.max, t)
    if col.categories is not None and t in ("integer", "float"):
        probs = col.probabilities or [1.0 / len(col.categories)] * len(col.categories)
        m = Discrete(col.categories, probs, integer, decimals)
        return m
    dist = col.distribution
    if dist == "auto":
        if col.quantiles:
            dist = "empirical"
        elif col.mean is not None and col.std is not None:
            dist = "normal"
        elif lo is not None and hi is not None:
            dist = "uniform"
        elif col.default is not None:
            dist = "constant"
    if dist == "constant":
        value = to_internal(col.default if col.default is not None else col.min, t)
        return Constant(value, integer, decimals)
    if dist == "uniform":
        if lo is None or hi is None:
            raise GenerationError(f"Column {col.name!r}: uniform distribution needs min and max.")
        if lo == hi:
            return Constant(lo, integer, decimals)
        return Uniform(integer, lo, hi, decimals)
    if dist == "normal":
        mean, std, notes = col.mean, col.std, []
        if mean is None or std is None:
            mean, std = (lo + hi) / 2.0, (hi - lo) / 6.0
            notes.append(f"mean/std not given; assumed mean={mean:g}, std={std:g} (range/6)")
        if std <= 0:
            return Constant(mean, integer, decimals)
        m = TruncatedNormal(mean, std, integer, lo, hi, decimals)
        m.notes += notes
        return m
    if dist == "lognormal":
        return TruncatedLogNormal(col.mean, col.std, integer, lo, hi, decimals)
    if dist == "empirical":
        q = [to_internal(v, t) for v in col.quantiles]
        if q[0] == q[-1]:
            return Constant(q[0], integer, decimals)
        return Empirical(q, integer, lo if lo is not None else q[0], hi if hi is not None else q[-1], decimals)
    raise GenerationError(f"Column {col.name!r}: cannot build a distribution from the given properties.")

"""SI dimensions and quantity literals.

A dimension is a vector of integer exponents over the SI base units we need:
(m, kg, s, A, K). Dimension mismatches are static errors, the physics
analogue of a type error.
"""
from __future__ import annotations

from dataclasses import dataclass

BASE = ("m", "kg", "s", "A", "K")


@dataclass(frozen=True)
class Dim:
    exp: tuple[int, ...] = (0,) * len(BASE)

    def __mul__(self, o: "Dim") -> "Dim":
        return Dim(tuple(a + b for a, b in zip(self.exp, o.exp)))

    def __truediv__(self, o: "Dim") -> "Dim":
        return Dim(tuple(a - b for a, b in zip(self.exp, o.exp)))

    def __pow__(self, n: int) -> "Dim":
        return Dim(tuple(a * n for a in self.exp))

    @property
    def dimensionless(self) -> bool:
        return all(e == 0 for e in self.exp)

    def __str__(self) -> str:
        if self.dimensionless:
            return "1"
        for name in PREFERRED:
            if UNITS[name][1] == self:
                return name
        return "·".join(f"{b}^{e}" if e != 1 else b for b, e in zip(BASE, self.exp) if e)


def _base(i: int) -> Dim:
    return Dim(tuple(1 if j == i else 0 for j in range(len(BASE))))


ONE = Dim()
METER, KG, SECOND, AMPERE, KELVIN = (_base(i) for i in range(len(BASE)))
VOLT = KG * METER**2 / SECOND**3 / AMPERE
WATT = VOLT * AMPERE
OHM = VOLT / AMPERE
FARAD = SECOND / OHM
HENRY = OHM * SECOND
TESLA = VOLT * SECOND / METER**2
PASCAL = KG / METER / SECOND**2

# name -> (scale to SI, dimension)
UNITS: dict[str, tuple[float, Dim]] = {
    "V": (1.0, VOLT),
    "A": (1.0, AMPERE),
    "W": (1.0, WATT),
    "Ohm": (1.0, OHM),
    "ohm": (1.0, OHM),
    "Ω": (1.0, OHM),
    "F": (1.0, FARAD),
    "H": (1.0, HENRY),
    "T": (1.0, TESLA),
    "s": (1.0, SECOND),
    "Hz": (1.0, ONE / SECOND),
    "K": (1.0, KELVIN),
    "Pa": (1.0, PASCAL),
    "m": (1.0, METER),
    "kg": (1.0, KG),
    # Celsius: a temperature *difference* of 1 K. Absolute readings
    # (ranges) get the 273.15 K offset in the parser; see ABSOLUTE_OFFSET.
    "degC": (1.0, KELVIN),
    "°C": (1.0, KELVIN),
}
ABSOLUTE_OFFSET = {"degC": 273.15, "°C": 273.15}
PREFERRED = ("V", "A", "W", "Ohm", "F", "H", "T", "s", "K", "Pa")
PREFIXES = {"p": 1e-12, "n": 1e-9, "u": 1e-6, "µ": 1e-6, "m": 1e-3, "k": 1e3, "M": 1e6}

# What a sensor or part kind measures / is.
KIND_DIM = {
    "voltage": VOLT,
    "current": AMPERE,
    "temperature": KELVIN,
    "magnetic_field": TESLA,
    "pressure": PASCAL,
    "humidity": ONE,  # relative humidity as a fraction
    "resistor": OHM,
    "capacitor": FARAD,
    "inductor": HENRY,
}
SENSOR_KINDS = ("voltage", "current", "temperature", "magnetic_field", "pressure", "humidity")
SOURCE_KINDS = ("voltage", "current")
PART_KINDS = ("resistor", "capacitor", "inductor")


def parse_unit(name: str) -> tuple[float, Dim] | None:
    if name in UNITS:
        return UNITS[name]
    if len(name) > 1 and name[0] in PREFIXES and name[1:] in UNITS and name[1:] not in ABSOLUTE_OFFSET:
        scale, dim = UNITS[name[1:]]
        return PREFIXES[name[0]] * scale, dim
    return None


def fmt(value: float, dim: Dim) -> str:
    unit = str(dim)
    if value == 0 or unit not in PREFERRED or unit == "K":
        return f"{value:.4g} {unit}"
    for p, s in (("M", 1e6), ("k", 1e3), ("", 1.0), ("m", 1e-3), ("u", 1e-6), ("n", 1e-9), ("p", 1e-12)):
        if abs(value) >= s:
            return f"{value / s:.4g} {p}{unit}"
    return f"{value:.4g} {unit}"

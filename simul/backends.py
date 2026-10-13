"""Hardware backends.

Every backend speaks the same four operations, which map 1:1 onto the
JSON-lines wire protocol in PROTOCOL.md:

    hello()                                  -> {"device", "fw", "channels"}
    set(ch, value, settle=0)                 -> drive a source (SI units), wait `settle` s
    read(chs, n)                             -> {ch: mean of n readings}
    step(ch, lo, hi, chs, dt, n)             -> {"t": [...], ch: [...]}

The simulator solves whatever netlist the rig declares (modified nodal
analysis, backward-Euler transients) and adds physics that the rig does
NOT state, so there is something real to discover: self-heating,
temperature coefficients by material, coil fields, a drifting room.
Its instruments err within (never beyond) their declared accuracy.
"""
from __future__ import annotations

import json
import math
import random
from typing import Protocol

from .check import GROUND
from .lang import Rig

T_REF = 293.15  # part nominal values are specified at 20 °C

# Hidden truths, per material: (temperature coefficient 1/K, thermal resistance K/W)
MATERIALS = {
    "metal_film": (50e-6, 150.0),
    "carbon": (-400e-6, 150.0),
    "copper": (3.93e-3, 25.0),
    "nichrome": (0.17e-3, 20.0),
    "constantan": (-0.03e-3, 30.0),
}
FIELD_PER_AMP = 0.0126  # T/A inside the coil: μ0·n for 1000 turns over 10 cm
EARTH_FIELD = 48e-6  # T, along the sensor axis

# Physics the rig files never state. Each effect can be switched on or off, which is what lets the
# benchmark (bench.py) score a discovery by asking: would this claim still hold without the effect?
EFFECTS = {
    "self_heating": "resistors warm by power x thermal resistance of their material (value: scale)",
    "earth_field": "a constant background field at every field sensor (value: T)",
    "leaky_capacitor": "every capacitor has a parallel leakage resistance (value: Ohm)",
    "source_offset": "every voltage source outputs this much more than commanded (value: V)",
    "nonlinear_sensor": "one sensor reads x * (1 + k * x / full scale) (value: [sensor, k])",
    "mains_hum": "50 Hz pickup on voltage sensors during fast sampling (value: amplitude V)",
}
DEFAULT_EFFECTS = {"self_heating": 1.0, "earth_field": EARTH_FIELD}
HUM_HZ, HUM_PHASE = 50.0, 0.7


class Backend(Protocol):
    def hello(self) -> dict: ...
    def describe(self) -> list[dict]: ...
    def set(self, ch: str, value: float, settle: float = 0.0) -> None: ...
    def read(self, chs: list[str], n: int = 16) -> dict[str, float]: ...
    def step(self, ch: str, lo: float, hi: float, chs: list[str], dt: float, n: int) -> dict[str, list[float]]: ...


def solve(a: list[list[float]], b: list[float]) -> list[float]:
    """Gaussian elimination with partial pivoting."""
    n = len(b)
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(m[r][c]))
        if abs(m[p][c]) < 1e-30:
            raise ArithmeticError("singular circuit (floating node or source loop?)")
        m[c], m[p] = m[p], m[c]
        for r in range(c + 1, n):
            f = m[r][c] / m[c][c]
            if f:
                for k in range(c, n + 1):
                    m[r][k] -= f * m[c][k]
    x = [0.0] * n
    for r in range(n - 1, -1, -1):
        x[r] = (m[r][n] - sum(m[r][k] * x[k] for k in range(r + 1, n))) / m[r][r]
    return x


class Circuit:
    """MNA over the rig's netlist. Unknowns: node voltages, then currents of
    voltage sources and inductors (positive from their first node to their second, through the element)."""

    def __init__(self, rig: Rig, values: dict[str, float], leak: float | None = None):
        self.rig, self.values, self.leak = rig, values, leak
        nodes = {n for el in [*rig.parts.values(), *rig.sources.values()] for n in el.nodes} - {GROUND}
        self.idx = {n: i for i, n in enumerate(sorted(nodes))}
        branch = [s.name for s in rig.sources.values() if s.kind == "voltage"]
        branch += [p.name for p in rig.parts.values() if p.kind == "inductor"]
        self.br = {name: len(self.idx) + i for i, name in enumerate(branch)}
        self.size = len(self.idx) + len(branch)

    def run(self, drive: dict[str, float], h: float | None, prev: dict | None) -> dict:
        """One solve. h=None: DC operating point. Otherwise a backward-Euler step of size h from `prev`."""
        n = self.size
        A = [[0.0] * n for _ in range(n)]
        z = [0.0] * n
        ix = self.idx.get

        def stamp_g(a, b, g):
            for p, q, s in ((a, a, 1), (b, b, 1), (a, b, -1), (b, a, -1)):
                if ix(p) is not None and ix(q) is not None:
                    A[ix(p)][ix(q)] += s * g

        def inject(a, b, i):  # current i pushed into node a, drawn from node b
            if ix(a) is not None:
                z[ix(a)] += i
            if ix(b) is not None:
                z[ix(b)] -= i

        def stamp_branch(name, a, b):
            k = self.br[name]
            for node, s in ((a, 1), (b, -1)):
                if ix(node) is not None:
                    A[ix(node)][k] += s
                    A[k][ix(node)] += s
            return k

        for i in range(len(self.idx)):
            A[i][i] += 1e-12  # gmin: keeps capacitor-only nodes solvable at DC
        for p in self.rig.parts.values():
            a, b = p.nodes
            v = self.values[p.name]
            if p.kind == "resistor":
                stamp_g(a, b, 1 / v)
            elif p.kind == "capacitor":
                if self.leak:
                    stamp_g(a, b, 1 / self.leak)
                if h is not None:
                    stamp_g(a, b, v / h)
                    inject(a, b, v / h * prev["v"][p.name])
            elif p.kind == "inductor":
                k = stamp_branch(p.name, a, b)
                if h is not None:
                    A[k][k] -= v / h
                    z[k] -= v / h * prev["i"][p.name]
        for s in self.rig.sources.values():
            a, b = s.nodes
            if s.kind == "voltage":
                z[stamp_branch(s.name, a, b)] = drive[s.name]
            else:
                inject(a, b, drive[s.name])

        x = solve(A, z)
        volt = lambda node: 0.0 if node == GROUND else x[self.idx[node]]
        out = {"v": {}, "i": {}}
        for p in self.rig.parts.values():
            a, b = p.nodes
            vab = volt(a) - volt(b)
            out["v"][p.name] = vab
            if p.kind == "resistor":
                out["i"][p.name] = vab / self.values[p.name]
            elif p.kind == "capacitor":
                out["i"][p.name] = 0.0 if h is None else self.values[p.name] * (vab - prev["v"][p.name]) / h
            else:
                out["i"][p.name] = x[self.br[p.name]]
        for s in self.rig.sources.values():
            a, b = s.nodes
            out["v"][s.name] = volt(a) - volt(b)
            # current delivered out of the + terminal into the circuit
            out["i"][s.name] = -x[self.br[s.name]] if s.kind == "voltage" else drive[s.name]
        return out


class SimBackend:
    def __init__(self, rig: Rig, seed: int = 0, truth: dict[str, float] | None = None,
                 effects: dict | None = None):
        self.rig = rig
        self.fx = dict(DEFAULT_EFFECTS if effects is None else effects)
        unknown = set(self.fx) - set(EFFECTS)
        if unknown:
            raise ValueError(f"unknown effects {sorted(unknown)} (known: {', '.join(EFFECTS)})")
        rng = random.Random(seed)
        # Actual part values at 20 °C: somewhere inside tolerance unless overridden.
        self.nominal = {p.name: p.nominal * (1 + rng.uniform(-1, 1) * p.tol) for p in rig.parts.values()}
        self.nominal.update(truth or {})
        # Instrument errors: fixed gain/offset error + bounded noise, jointly within spec.
        self.err = {
            name: (rng.uniform(-0.5, 0.5) * s.acc.rel, rng.uniform(-0.5, 0.5) * s.acc.abs)
            for name, s in [*rig.sources.items(), *rig.sensors.items()]
        }
        self.rng = rng
        self.ambient = 295.15 + rng.uniform(-2, 2)
        self.cmd = {name: 0.0 for name in rig.sources}
        self.temp = {p: self.ambient for p in rig.parts}
        self.calls = 0

    # -- physics
    def _drift(self):
        self.calls += 1
        self.ambient += self.rng.gauss(0, 0.01)

    def _values(self) -> dict[str, float]:
        """Part values at their current temperatures."""
        out = {}
        for p in self.rig.parts.values():
            alpha = MATERIALS.get(p.material or "metal_film", MATERIALS["metal_film"])[0] if p.kind == "resistor" else 0.0
            out[p.name] = self.nominal[p.name] * (1 + alpha * (self.temp[p.name] - T_REF))
        return out

    def _circuit(self) -> Circuit:
        return Circuit(self.rig, self._values(), self.fx.get("leaky_capacitor"))

    def _actual_drive(self, cmd: dict[str, float]) -> dict[str, float]:
        return {s: self._source(s, v) for s, v in cmd.items()}

    def _source(self, s: str, v: float) -> float:
        offset = self.fx.get("source_offset", 0.0) if self.rig.sources[s].kind == "voltage" else 0.0
        return self._err(s, v) + offset

    def _thermal_equilibrium(self, drive):
        """Iterate electrical <-> thermal coupling to the steady state."""
        for _ in range(30):
            sol = self._circuit().run(drive, None, None)
            for p in self.rig.parts.values():
                rth = MATERIALS.get(p.material or "metal_film", MATERIALS["metal_film"])[1] * self.fx.get("self_heating", 0.0)
                power = abs(sol["v"][p.name] * sol["i"][p.name]) if p.kind == "resistor" else 0.0
                self.temp[p.name] = self.ambient + power * rth
        return sol

    def _quantity(self, ch: str, sol: dict) -> float:
        s = self.rig.sensors[ch]
        if s.where == "across":
            return sol["v"][s.target]
        if s.where == "through":
            return sol["i"][s.target]
        if s.where == "at":
            if s.kind == "temperature":
                return self.temp.get(s.target, self.ambient)
            if s.kind == "magnetic_field":
                coil = self.rig.parts.get(s.target)
                inside = FIELD_PER_AMP * sol["i"][s.target] if coil and coil.kind == "inductor" else 0.0
                return inside + self.fx.get("earth_field", 0.0)
        # ambient
        return {"temperature": self.ambient, "magnetic_field": self.fx.get("earth_field", 0.0),
                "pressure": 101_325.0, "humidity": 0.45}.get(s.kind, 0.0)

    # -- instruments
    def _sense(self, ch: str, sol: dict, t: float | None = None) -> float:
        """What sensor `ch` reads: the quantity, any hidden instrument effects, then its in-spec error."""
        x = self._quantity(ch, sol)
        s = self.rig.sensors[ch]
        if "nonlinear_sensor" in self.fx and self.fx["nonlinear_sensor"][0] == ch:
            x *= 1 + self.fx["nonlinear_sensor"][1] * x / max(abs(s.lo), abs(s.hi))
        if t is not None and s.kind == "voltage" and "mains_hum" in self.fx:
            x += self.fx["mains_hum"] * math.sin(2 * math.pi * HUM_HZ * t + HUM_PHASE)
        return self._err(ch, x)

    def _err(self, name: str, true: float) -> float:
        gain, offset = self.err[name]
        acc = (self.rig.sources | self.rig.sensors)[name].acc
        bound = acc.bound(true)
        noise = max(-0.3 * bound, min(0.3 * bound, self.rng.gauss(0, 0.1 * bound)))
        return true * (1 + gain) + offset + noise

    # -- protocol
    def hello(self) -> dict:
        return {"device": self.rig.device or "sim", "fw": "sim-0.2", "channels": [*self.rig.sources, *self.rig.sensors]}

    def describe(self) -> list[dict]:
        out = []
        for s in self.rig.sources.values():
            out.append({"name": s.name, "role": "source", "kind": s.kind, "range": [s.lo, s.hi],
                        "accuracy": [s.acc.rel, s.acc.abs], "module": "sim", "calibrated": None})
        for s in self.rig.sensors.values():
            out.append({"name": s.name, "role": "sensor", "kind": s.kind, "where": s.where, "target": s.target,
                        "range": [s.lo, s.hi], "accuracy": [s.acc.rel, s.acc.abs], "module": "sim", "calibrated": None})
        return out

    def set(self, ch: str, value: float, settle: float = 0.0) -> None:
        self.cmd[ch] = value

    def read(self, chs: list[str], n: int = 16) -> dict[str, float]:
        self._drift()
        sol = self._thermal_equilibrium(self._actual_drive(self.cmd))
        reads = [{c: self._sense(c, sol) for c in chs} for _ in range(n)]  # averaging removes hum
        return {c: sum(r[c] for r in reads) / n for c in chs}

    def step(self, ch, lo, hi, chs, dt, n):
        self._drift()
        self.cmd[ch] = lo
        before = self._actual_drive(self.cmd)
        sol = self._thermal_equilibrium(before)  # settled at `lo`
        self.cmd[ch] = hi
        after = {**before, ch: self._source(ch, hi)}
        circuit = self._circuit()  # thermal time constants >> electrical: hold temperatures
        sub = 20
        h = dt / sub
        out = {"t": [], **{c: [] for c in chs}}
        for k in range(n):
            if k:
                for _ in range(sub):
                    sol = circuit.run(after, h, sol)
            elif n:
                sol = circuit.run(after, 1e-12 * dt, sol)  # the instant after the edge
            out["t"].append(k * dt)
            for c in chs:
                out[c].append(self._sense(c, sol, k * dt))
        return out


class SerialBackend:
    """A real device speaking PROTOCOL.md over a serial port."""

    def __init__(self, port: str, baud: int = 115200, timeout: float = 30.0):
        import serial  # pyserial

        self.port = serial.Serial(port, baud, timeout=timeout)
        self.port.reset_input_buffer()

    def _call(self, **msg) -> dict:
        self.port.write((json.dumps(msg) + "\n").encode())
        line = self.port.readline()
        if not line:
            raise TimeoutError(f"device did not answer {msg['op']!r}")
        reply = json.loads(line)
        if not reply.get("ok"):
            raise RuntimeError(f"device error on {msg['op']!r}: {reply.get('error')}")
        return reply

    def hello(self) -> dict:
        return self._call(op="hello")

    def describe(self) -> list[dict]:
        return self._call(op="describe")["channels"]

    def set(self, ch, value, settle=0.0):
        self._call(op="set", ch=ch, value=value, settle_ms=int(settle * 1000))

    def read(self, chs, n=16):
        return self._call(op="read", ch=chs, n=n)["values"]

    def step(self, ch, lo, hi, chs, dt, n):
        return self._call(op="step", ch=ch, lo=lo, hi=hi, sample=chs, dt=dt, n=n)["data"]


def open_backend(spec: str, rig: Rig, truth: dict[str, float] | None = None, effects: dict | None = None) -> Backend:
    """'sim', 'sim:<seed>' or 'serial:<port>'."""
    kind, _, arg = spec.partition(":")
    if kind == "sim":
        return SimBackend(rig, seed=int(arg or 0), truth=truth, effects=effects)
    if kind == "serial":
        return SerialBackend(arg)
    raise ValueError(f"unknown backend {spec!r}")

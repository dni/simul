"""Law discovery: from an experiment's data to candidate claims.

This is the *untrusted* half of the loop, in the tradition of BACON
(Langley et al.), Eureqa (Schmidt & Lipson 2009) and the Robot Scientist
(King et al. 2009). It proposes; the kernel disposes:

  1. search   simplest models that fit every row within the instruments'
              declared worst-case error (not least-squares "good enough")
  2. explain  name each fitted constant in terms of the rig description:
              1.03 kΩ -> R, 1.02 ms -> L / (R + Rc), 0.97 -> R / (R + Rc)
              constants with no explanation become `param`s: new physics,
              or something the rig description is missing
  3. emit     a claim in Simul syntax, with the best simpler model that
              failed as its `against` rival, to be proven on FRESH data

Model families:
  static (sweep):  y = c1*f1 + c2*f2, features f from {1, x, x^2, 1/x,
                   sqrt(x), x*z, x/z, x*(z - 293.15 K)} over observed columns
  transient (step): y = a + b*exp(-t/tau) and y = a + b*t
"""
from __future__ import annotations

import itertools
import math
import zlib
from dataclasses import dataclass, field

from .kernel import uncertainties
from .lang import Claim, Rig, Step, Sweep
from .units import KIND_DIM, ONE, PREFERRED, SECOND, UNITS, Dim, fmt

T_REF = 293.15


# ---------------------------------------------------------------- features

@dataclass
class Feature:
    text: str  # Simul expression
    fn: object  # row -> float
    dim: Dim
    cost: int
    vars: tuple[str, ...]


def features(cols: dict[str, Dim], target: str) -> list[Feature]:
    xs = [c for c in cols if c != target]
    out = [Feature("1", lambda r: 1.0, ONE, 0, ())]
    for x in xs:
        d = cols[x]
        out += [
            Feature(x, lambda r, x=x: r[x], d, 1, (x,)),
            Feature(f"{x}^2", lambda r, x=x: r[x] ** 2, d**2, 2, (x,)),
            Feature(f"{x}^-1", lambda r, x=x: 1 / r[x], ONE / d, 2, (x,)),
            Feature(f"sqrt({x})", lambda r, x=x: math.sqrt(max(r[x], 0)), Dim(tuple(e // 2 for e in d.exp)), 3, (x,)),
        ]
        if any(e % 2 for e in d.exp):
            out.pop()  # sqrt of an odd dimension is not expressible
    for x, z in itertools.permutations(xs, 2):
        dx, dz = cols[x], cols[z]
        if x < z:
            out.append(Feature(f"{x} * {z}", lambda r, x=x, z=z: r[x] * r[z], dx * dz, 2, (x, z)))
        out.append(Feature(f"{x} / {z}", lambda r, x=x, z=z: r[x] / r[z], dx / dz, 2, (x, z)))
        if cols[z] == KIND_DIM["temperature"]:
            out.append(Feature(f"{x} * ({z} - {T_REF} K)", lambda r, x=x, z=z: r[x] * (r[z] - T_REF), dx * dz, 3, (x, z)))
    return out


# ---------------------------------------------------------------- linear fits

def lstsq(rows, fs, y, w):
    """Weighted least squares for y ~ sum c_k f_k (normal equations; k <= 3)."""
    k = len(fs)
    A = [[sum(w[i] ** 2 * fs[a](r) * fs[b](r) for i, r in enumerate(rows)) for b in range(k)] for a in range(k)]
    z = [sum(w[i] ** 2 * fs[a](r) * y[i] for i, r in enumerate(rows)) for a in range(k)]
    from .backends import solve
    return solve(A, z)


def worst_ratio(model, rows, target, us) -> float:
    """max over rows of |y - model| / (u_y + sum |d model/dx| u_x)."""
    worst = 0.0
    for r, u in zip(rows, us):
        try:
            pred = model(r)
            bound = u[target]
            for x, ux in u.items():
                if x == target or ux == 0:
                    continue
                h = max(abs(r[x]) * 1e-6, 1e-15)
                bound += abs((model({**r, x: r[x] + h}) - model({**r, x: r[x] - h})) / (2 * h)) * ux
            ratio = abs(r[target] - pred) / bound if bound else float("inf")
        except (ArithmeticError, ValueError):
            return float("inf")
        worst = max(worst, ratio)
    return worst


def consistent_range(make, coeffs, i, rows, target, us, max_rel=1.0) -> tuple[float, float]:
    """Interval of coefficient i (others fixed) over which the model stays consistent."""
    c = coeffs[i]
    scale = abs(c) or 1.0

    def ok(v):
        cs = list(coeffs)
        cs[i] = v
        return worst_ratio(make(cs), rows, target, us) <= 1.0

    def edge(sign):
        lo, hi = 0.0, max_rel * scale
        if ok(c + sign * hi):
            return c + sign * hi
        for _ in range(30):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if ok(c + sign * mid) else (lo, mid)
        return c + sign * lo

    return edge(-1), edge(+1)


# ---------------------------------------------------------------- explanation

@dataclass
class Named:
    text: str
    lo: float
    hi: float
    dim: Dim
    cost: int


def _iv(op, a: Named, b: Named) -> tuple[float, float] | None:
    if op == "+":
        return a.lo + b.lo, a.hi + b.hi
    if op == "-":
        return a.lo - b.hi, a.hi - b.lo
    if op == "*":
        ps = [a.lo * b.lo, a.lo * b.hi, a.hi * b.lo, a.hi * b.hi]
        return min(ps), max(ps)
    if b.lo <= 0 <= b.hi:
        return None
    qs = [a.lo / b.lo, a.lo / b.hi, a.hi / b.lo, a.hi / b.hi]
    return min(qs), max(qs)


def vocabulary(rig: Rig, row: dict, constants: dict[str, tuple[float, float]]) -> list[Named]:
    """Expressions the rig description lets us name, with value intervals."""
    atoms = [Named(p.name, *p.bounds, KIND_DIM[p.kind], 1) for p in rig.parts.values()]
    for name, (lo, hi) in constants.items():
        kind = (rig.sources.get(name.removesuffix("_0")) or rig.sensors.get(name)).kind
        atoms.append(Named(name, lo, hi, KIND_DIM[kind], 1))
    # Only drive levels (V, V_0) combine multiplicatively with parts (V / R is a current);
    # a steady sensor reading such as room temperature may only appear additively.
    drive = {n for n in constants if n.removesuffix("_0") in rig.sources}
    level2 = []
    for a, b in itertools.permutations(atoms, 2):
        if a.text in constants and a.text not in drive or b.text in constants and b.text not in drive:
            continue
        both_readings = a.text in constants and b.text in constants
        for op in "" if both_readings else "*/":  # V / V_0 is not physics; V / R is
            if op == "*" and a.text > b.text:
                continue
            iv = _iv(op, a, b)
            if iv:
                level2.append(Named(f"{a.text} {op} {b.text}", *iv, a.dim * b.dim if op == "*" else a.dim / b.dim, 2))
        if a.dim == b.dim and a.text < b.text:
            level2.append(Named(f"({a.text} + {b.text})", *_iv("+", a, b), a.dim, 2))
        if a.dim == b.dim and a.text in constants and b.text in constants:
            level2.append(Named(f"({a.text} - {b.text})", *_iv("-", a, b), a.dim, 2))
    for p in rig.parts.values():  # conductance-like reciprocals: Ir == Vr / R reads as (1 / R) * Vr
        lo, hi = p.bounds
        level2.append(Named(f"1 / {p.name}", 1 / hi, 1 / lo, ONE / KIND_DIM[p.kind], 2))
    level3 = []
    for a in atoms:
        if a.text in constants:
            continue
        for b in level2:
            if b.text.startswith("("):  # a / (x + y), a * (x + y)
                for op in "*/":
                    iv = _iv(op, a, b)
                    if iv:
                        level3.append(Named(f"{a.text} {op} {b.text}", *iv, a.dim * b.dim if op == "*" else a.dim / b.dim, 3))
    vocab = [Named("1", 1.0, 1.0, ONE, 0), *atoms, *level2, *level3]
    # dimensionless factors times source-like quantities: R / (R + Rc) * V
    ratios = [n for n in vocab if n.dim == ONE and n.text != "1"]
    sources = [n for n in vocab if n.text in constants or n.text.startswith("(") and any(c in n.text for c in constants) and "-" in n.text]
    for r in ratios:
        for s in sources:
            iv = _iv("*", r, s)
            vocab.append(Named(f"{r.text} * {s.text}", *iv, s.dim, r.cost + s.cost))
    return vocab


def explain(lo: float, hi: float, dim: Dim, vocab: list[Named]) -> Named | None:
    """Simplest named expression whose interval overlaps the coefficient's consistent range."""
    best = None
    mid = (lo + hi) / 2
    for n in vocab:
        if n.dim != dim or n.hi < lo or n.lo > hi:
            continue
        nmid = (n.lo + n.hi) / 2
        key = (n.cost, abs(math.log(abs(nmid / mid))) if mid and nmid and nmid / mid > 0 else 9)
        if best is None or key < best[0]:
            best = (key, n)
    return best[1] if best else None


def unit_text(d: Dim) -> str:
    """Render a dimension as a unit expression the parser accepts (for `param` declarations)."""
    if d == ONE:
        return "1"
    names = [u for u in PREFERRED]
    for u in names:
        if UNITS[u][1] == d:
            return u
    for a, b in itertools.product(["1", *names], names):
        if (ONE if a == "1" else UNITS[a][1]) / UNITS[b][1] == d:
            return f"{a}/{b}"
        if a != "1" and UNITS[a][1] * UNITS[b][1] == d and a <= b:
            return f"{a}*{b}"
    for a, b, c in itertools.product(["1", *names], names, names):
        da = ONE if a == "1" else UNITS[a][1]
        if da / UNITS[b][1] / UNITS[c][1] == d and b <= c:
            return f"{a}/{b}/{c}"
    return "*".join(f"{u}^{e}" for u, e in zip(("m", "kg", "s", "A", "K"), d.exp) if e)


# ---------------------------------------------------------------- search

@dataclass
class Candidate:
    target: str
    terms: list[str]  # feature texts, or "exp(-t / tau)" etc.
    coeffs: list[float]
    ranges: list[tuple[float, float]]
    dims: list[Dim]
    ratio: float
    cost: int
    names: list[Named | None] = field(default_factory=list)
    tau: tuple[float, float, float] | None = None  # (value, lo, hi) for exponentials
    tau_name: Named | None = None

    @property
    def unexplained(self) -> int:
        return sum(n is None for n in self.names) + (self.tau is not None and self.tau_name is None)

    def law(self, params: dict[str, str]) -> str:
        parts = []
        for term, c, n in zip(self.terms, self.coeffs, self.names):
            coef = n.text if n else params[term]
            if "tau" in term:
                tau = self.tau_name.text if self.tau_name else params["tau"]
                term = term.replace("tau", f"({tau})")
            if term == "1":
                parts.append(coef)
            elif coef == "1":
                parts.append(term)
            else:
                parts.append(f"{coef} * {term}")
        return f"{self.target} == " + " + ".join(parts)


def static_candidates(rows, us, cols, target, max_terms=2) -> list[Candidate]:
    fs = features(cols, target)
    y = [r[target] for r in rows]
    w = [1 / u[target] for u in us]
    out = []
    for k in range(1, max_terms + 1):
        for combo in itertools.combinations(fs, k):
            if target in {v for f in combo for v in f.vars}:
                continue
            try:
                c = lstsq(rows, [f.fn for f in combo], y, w)
            except ArithmeticError:
                continue
            make = lambda cs, combo=combo: (lambda r: sum(ci * f.fn(r) for ci, f in zip(cs, combo)))
            ratio = worst_ratio(make(c), rows, target, us)
            if ratio <= 1.0:
                out.append(Candidate(target, [f.text for f in combo], c, [], [cols[target] / f.dim for f in combo],
                                     ratio, sum(f.cost for f in combo) + 2 * k))
                out[-1]._make = make
    return out


def transient_candidates(rows, us, target, dt, duration) -> list[Candidate]:
    y = [r[target] for r in rows]
    w = [1 / u[target] for u in us]
    out = []
    one, tt = (lambda r: 1.0), (lambda r: r["t"])
    # linear: y = a + b t
    a, b = lstsq(rows, [one, tt], y, w)
    lin = lambda cs: (lambda r: cs[0] + cs[1] * r["t"])
    out.append(Candidate(target, ["1", "t"], [a, b], [], [], worst_ratio(lin([a, b]), rows, target, us), 4))
    out[-1]._make = lin
    # exponential: y = a + b exp(-t/tau); scan tau on a log grid, refine
    def fit(tau):
        e = lambda r: math.exp(-r["t"] / tau)
        cs = lstsq(rows, [one, e], y, w)
        mk = lambda cs, tau=tau: (lambda r: cs[0] + cs[1] * math.exp(-r["t"] / tau))
        return worst_ratio(mk(cs), rows, target, us), cs, mk
    grid = [2 * dt * (5 * duration / (2 * dt)) ** (i / 80) for i in range(81)]
    best_tau = min(grid, key=lambda tau: fit(tau)[0])
    lo_t, hi_t = best_tau / 1.2, best_tau * 1.2
    for _ in range(40):  # golden-section on the worst ratio
        m1, m2 = lo_t + (hi_t - lo_t) * 0.382, lo_t + (hi_t - lo_t) * 0.618
        lo_t, hi_t = (lo_t, m2) if fit(m1)[0] < fit(m2)[0] else (m1, hi_t)
    tau = (lo_t + hi_t) / 2
    ratio, cs, mk = fit(tau)
    if ratio <= 1.0:
        # consistent tau interval: refit a, b at each tau
        def ok(t):
            return fit(t)[0] <= 1.0
        def edge(direction):
            good, bad = tau, tau * (4 if direction > 0 else 0.25)
            if ok(bad):
                return bad
            for _ in range(30):
                mid = math.sqrt(good * bad)
                good, bad = (mid, bad) if ok(mid) else (good, mid)
            return good
        c = Candidate(target, ["1", "exp(-t / tau)"], cs, [], [], ratio, 6, tau=(tau, edge(-1), edge(+1)))
        c._make = mk
        out.append(c)
    return [c for c in out if c.ratio <= 1.0] + [c for c in out if c.ratio > 1.0]


# ---------------------------------------------------------------- driver

def varying(rows, us, col) -> bool:
    vals = [r[col] for r in rows]
    return max(vals) - min(vals) > 3 * max(u[col] for u in us)


def discover(claim: Claim, rig: Rig, rows: list[dict], top: int = 3) -> list[dict]:
    """Return, per observed quantity, the best candidate laws as ready-to-prove claim text."""
    proc = claim.procedure
    us = [uncertainties(rig, r) for r in rows]
    chans = [c for c in rows[0] if c in rig.sensors]
    transient = isinstance(proc, Step)
    var_cols = {c: KIND_DIM[(rig.sensors.get(c) or rig.sources[c]).kind]
                for c in [*chans, proc.source] if c in rows[0] and varying(rows, us, c)}
    # quantities that stay constant during the run become nameable constants
    constants = {}
    for c in rows[0]:
        if c == "t" or c in var_cols:
            continue
        v = sum(r[c] for r in rows) / len(rows)
        u = max(x[c] for x in us)
        constants[c] = (v - u, v + u)
    vocab = vocabulary(rig, rows[0], constants)

    results = []
    for target in [c for c in chans if c in var_cols or transient]:
        if transient:
            if target not in proc.sample:
                continue
            cands = transient_candidates(rows, us, target, proc.dt, proc.duration)
        else:
            cands = static_candidates(rows, us, var_cols, target)
        good = []
        for c in (c for c in cands if c.ratio <= 1.0):
            c.ranges = [consistent_range(c._make, c.coeffs, i, rows, target, us) for i in range(len(c.coeffs))]
            if any(lo <= 0 <= hi for (lo, hi), t in zip(c.ranges, c.terms) if t != "1"):
                continue  # a term that may be zero is unsupported by the data: the simpler model covers it
            good.append(c)
            dims = c.dims or ([KIND_DIM[rig.sensors[target].kind]] * 2)
            c.dims = dims
            c.names = [explain(lo, hi, d, vocab) if not (lo <= 0 <= hi and t == "1") else Named("0", 0, 0, d, 0)
                       for (lo, hi), d, t in zip(c.ranges, dims, c.terms)]
            if c.tau:
                c.tau_name = explain(c.tau[1], c.tau[2], SECOND, vocab)
        good.sort(key=lambda c: (c.unexplained, c.cost, c.ratio))
        seen = set()
        for c in good:
            r = render(c, claim, rig)
            if r["law"] in seen:
                continue
            seen.add(r["law"])
            results.append(r)
            if len([r for r in results if r["target"] == target]) >= top:
                break
    results.sort(key=lambda r: (r["unexplained"], r["cost"]))
    return results


def render(c: Candidate, claim: Claim, rig: Rig) -> dict:
    params, decls = {}, []
    for i, (term, n, (lo, hi), d) in enumerate(zip(c.terms, c.names, c.ranges, c.dims)):
        if n is None:
            pname = f"c{i + 1}"
            params[term] = pname
            decls.append(f"  param {pname} : {unit_text(d)} in {lo:.6g} .. {hi:.6g}")
    if c.tau and not c.tau_name:
        params["tau"] = "tau"
        decls.append(f"  param tau : s in {c.tau[1]:.6g} .. {c.tau[2]:.6g}")
    # drop terms explained as zero
    keep = [i for i, n in enumerate(c.names) if not (n and n.text == "0")]
    shown = Candidate(c.target, [c.terms[i] for i in keep], [c.coeffs[i] for i in keep], [c.ranges[i] for i in keep],
                      [c.dims[i] for i in keep], c.ratio, c.cost, [c.names[i] for i in keep], c.tau, c.tau_name)
    law = shown.law(params)
    how = procedure_text(claim.procedure, rig)
    name = f"{claim.name}_{c.target}_{zlib.crc32(law.encode()) % 1000:03d}"
    text = f"claim {name} on {rig.name} {{\n" + "\n".join(decls + [how, f"  law {law}"]) + "\n}\n"
    return {"target": c.target, "law": law, "claim": text, "unexplained": shown.unexplained,
            "cost": c.cost, "ratio": c.ratio,
            "notes": [f"tau ≈ {fmt(c.tau[0], SECOND)}" + (f" = {c.tau_name.text}" if c.tau_name else " (unexplained)")] if c.tau else []}


def procedure_text(proc: Sweep | Step, rig: Rig) -> str:
    """A procedure as Simul source lines (indented for a claim body)."""
    src = rig.sources[proc.source]
    sd = KIND_DIM[src.kind]
    if isinstance(proc, Sweep):
        settle = f" settle {fmt(proc.settle, SECOND)}" if proc.settle else ""
        return (f"  sweep {src.name} from {fmt(proc.lo, sd)} to {fmt(proc.hi, sd)} steps {proc.steps}{settle}\n"
                f"  observe {', '.join(proc.observe)}")
    return (f"  step {src.name} from {fmt(proc.lo, sd)} to {fmt(proc.hi, sd)}\n"
            f"  sample {', '.join(proc.sample)} every {fmt(proc.dt, SECOND)} for {fmt(proc.duration, SECOND)}")

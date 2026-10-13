"""Experiment design: choose the experiment that decides between rival laws.

Robot Scientist Adam chose experiments to separate its hypotheses, not at
random. This module does the same for a claim with `against` rivals.

Two laws `y == f(x, w)` and `y == g(x, v)` can both survive an experiment
iff some data could be consistent with both: there exist witnesses w, v
(each inside its own declared ranges, chosen independently, as the kernel
does) such that at every row

    |f - g|  <=  U_f + U_g          (U = the kernel's error bound for that law)

So the experiment's *separation* is

    S  =  min over (w, v)  of  max over rows  |f - g| / (U_f + U_g)

S > 1 means the experiment is decisive in the worst case: whatever the
hardware does, at least one of the two laws will be refuted. S <= 1 means
there is an outcome both laws survive, and the claim may come back
INCONCLUSIVE however carefully it is run.

Predictions need every input the laws mention. Controls (the source, `t`,
`<src>_0`) are set by the experiment; other readings (a current, a
temperature) come from a *pilot* run, interpolated in the source level,
and are never extrapolated.

This is advisory, untrusted code: S comes from a numerical search, which
may miss the witnesses that make two laws agree. The kernel's verdict on
the real run is what counts.

When no experiment on the rig is decisive, `design` reports which declared
tolerance or instrument accuracy, tightened by how much, would make one.
"""
from __future__ import annotations

import copy
import random
from dataclasses import dataclass, replace

from .check import MAX_SAMPLES, compile_expr, existentials, free_vars
from .discover import procedure_text
from .kernel import uncertainties
from .lang import Accuracy, Claim, Law, Rig, SimulError, Step, Sweep, Var
from .units import KIND_DIM, SECOND, fmt

MAX_ROWS = 48  # rows used to score a candidate (subsampled: scores are conservative)
MARGIN = 1.0


# ---------------------------------------------------------------- predictions

def target_of(claim: Claim, rig: Rig) -> str:
    if claim.law is None or not claim.against:
        raise SimulError(f"{claim.name}: experiment design needs a law and at least one `against` rival")
    names = set()
    for law in [claim.law, *claim.against]:
        if not isinstance(law.lhs, Var) or law.lhs.name not in rig.sensors or law.lhs.name in free_vars(law.rhs):
            raise SimulError(f"{claim.name}: write each law as `<sensor> == <expression without it>` to design for it")
        names.add(law.lhs.name)
    if len(names) > 1:
        raise SimulError(f"{claim.name}: the law and its rivals must predict the same sensor, got {sorted(names)}")
    return names.pop()


def needed_columns(claim: Claim, rig: Rig) -> list[str]:
    """Readings the claim's laws mention that its experiment does not set."""
    proc = claim.procedure
    if claim.law is None:
        return []
    controls = {proc.source, "t", f"{proc.source}_0"}
    if isinstance(claim.law.lhs, Var):
        controls.add(claim.law.lhs.name)
    used = set().union(*(free_vars(l.rhs) for l in [claim.law, *claim.against]))
    return sorted(n for n in used if n in rig.sensors and n not in controls)


class Context:
    """Readings the laws need but the experiment does not set, from a pilot run."""

    def __init__(self, claim: Claim, rig: Rig, target: str, pilot: list[dict] | None):
        proc = claim.procedure
        self.needed = needed_columns(claim, rig)
        self.src = proc.source
        self.lo, self.hi = rig.sources[proc.source].lo, rig.sources[proc.source].hi
        self.table: list[dict] = []
        if not self.needed:
            return
        if not pilot or any(n not in pilot[0] for n in self.needed):
            raise SimulError(f"the laws mention {', '.join(self.needed)}, which an experiment does not set: "
                             f"give a pilot run (notebook entry) that observed them")
        if isinstance(proc, Step):
            slow = [n for n in self.needed if n not in rig.monitors]
            if slow:
                raise SimulError(f"in a step experiment only monitors can be predicted, not {slow}")
            self.const = {n: sum(r[n] for r in pilot) / len(pilot) for n in self.needed}
            return
        if self.src not in pilot[0] or "t" in pilot[0]:
            raise SimulError(f"the pilot run must be a sweep of {self.src}")
        self.table = sorted(pilot, key=lambda r: r[self.src])
        self.lo, self.hi = self.table[0][self.src], self.table[-1][self.src]

    def at(self, v: float) -> dict:
        if not self.needed:
            return {}
        if not self.table:
            return dict(self.const)
        rows = self.table
        for a, b in zip(rows, rows[1:]):
            if a[self.src] <= v <= b[self.src]:
                span = b[self.src] - a[self.src]
                f = (v - a[self.src]) / span if span else 0.0
                return {n: a[n] + f * (b[n] - a[n]) for n in self.needed}
        raise ValueError(f"{v} outside the pilot range")


def rows_for(proc: Sweep | Step, ctx: Context) -> list[dict]:
    if isinstance(proc, Sweep):
        idx = range(proc.steps)
        if proc.steps > MAX_ROWS:
            idx = sorted({round(k * (proc.steps - 1) / (MAX_ROWS - 1)) for k in range(MAX_ROWS)})
        vals = [proc.lo + (proc.hi - proc.lo) * k / (proc.steps - 1) for k in idx]
        return [{proc.source: v, **ctx.at(v)} for v in vals]
    n = int(round(proc.duration / proc.dt))
    # the early transient carries most information: keep the first samples, then spread out
    idx = sorted({*range(min(n, MAX_ROWS // 3)), *(round(k * (n - 1) / (MAX_ROWS - 1)) for k in range(MAX_ROWS))})
    env = ctx.at(proc.hi)
    return [{"t": k * proc.dt, proc.source: proc.hi, f"{proc.source}_0": proc.lo, **env} for k in idx]


# ---------------------------------------------------------------- separation

class Rivals:
    """The claim's law against one rival: how well can an experiment tell them apart?"""

    def __init__(self, claim: Claim, rig: Rig, rival: Law, target: str):
        self.claim, self.target = claim, target
        self.laws = [claim.law, rival]
        self.fns = [compile_expr(l.rhs) for l in self.laws]
        self.uses = [free_vars(l.rhs) for l in self.laws]
        self.set_rig(rig)

    def set_rig(self, rig: Rig) -> None:
        self.rig = rig
        self.boxes = [existentials(l, self.claim, rig) for l in self.laws]
        self.keys = [(i, n) for i, box in enumerate(self.boxes) for n in box]

    def witnesses(self, z: list[float]) -> list[dict[str, float]]:
        out: list[dict[str, float]] = [{}, {}]
        for (i, n), x in zip(self.keys, z):
            lo, hi, _ = self.boxes[i][n]
            out[i][n] = lo + (hi - lo) * x
        return out

    def _predict(self, i: int, env: dict, us: dict) -> tuple[float, float]:
        """Law i's prediction for the target and its error bound U at this row."""
        f, law = self.fns[i], self.laws[i]
        y = f(env)
        u = self.rig.sensors[self.target].acc.bound(y) + law.slack * abs(y)
        for x in self.uses[i]:
            ux = us.get(x, 0.0)
            if ux:
                h = max(abs(env[x]) * 1e-6, 1e-15)
                hi, lo = dict(env), dict(env)
                hi[x] += h
                lo[x] -= h
                u += abs((f(hi) - f(lo)) / (2 * h)) * ux
        return y, u

    def ratios(self, rows: list[dict], us: list[dict], z: list[float]) -> list[float]:
        ws = self.witnesses(z)
        out = []
        for row, u in zip(rows, us):
            try:
                (ya, ua), (yb, ub) = self._predict(0, {**row, **ws[0]}, u), self._predict(1, {**row, **ws[1]}, u)
                out.append(abs(ya - yb) / (ua + ub) if ua + ub else float("inf"))
            except (ArithmeticError, ValueError):
                out.append(float("inf"))  # a law that cannot be evaluated here is refuted here
        return out

    def separation(self, rows: list[dict], stop_below: float | None = None, seed: int = 0) -> tuple[float, list[float]]:
        """min over witness pairs of the worst-row ratio, by multistart pattern search."""
        us = [uncertainties(self.rig, r) for r in rows]
        obj = lambda z: max(self.ratios(rows, us, z))
        d = len(self.keys)
        if d == 0:
            return obj([]), []
        rng = random.Random(seed)
        starts = [[0.5] * d] + [[rng.random() for _ in range(d)] for _ in range(24 + 8 * d)]
        scored = sorted(((obj(z), z) for z in starts), key=lambda p: p[0])
        best = scored[0]
        for value, z in scored[:3]:
            cur, step = (value, z), 0.2
            while step > 2e-4:
                if stop_below is not None and cur[0] <= stop_below:
                    return cur
                moves = [[(s if j == i else 0.0) for j in range(d)] for i in range(d) for s in (step, -step)]
                moves += [[rng.uniform(-step, step) for _ in range(d)] for _ in range(d)]
                improved = False
                for m in moves:
                    q = [min(1.0, max(0.0, a + b)) for a, b in zip(cur[1], m)]
                    v = obj(q)
                    if v < cur[0]:
                        cur, improved = (v, q), True
                if not improved:
                    step /= 2
            best = min(best, cur, key=lambda p: p[0])
        return best

    def where(self, rows: list[dict], z: list[float]) -> dict:
        """The row where the two laws differ most, given the witnesses that make them agree best."""
        us = [uncertainties(self.rig, r) for r in rows]
        rs = self.ratios(rows, us, z)
        i = max(range(len(rows)), key=lambda k: rs[k])
        return rows[i]


# ---------------------------------------------------------------- candidates

def candidates(proc: Sweep | Step, rig: Rig, ctx: Context) -> tuple[list[tuple[float, Sweep | Step]], list[Sweep | Step]]:
    """(cost, procedure) pairs to try, and the 'maximal' experiments that bound what any candidate can do."""
    src = rig.sources[proc.source]
    out = []
    if isinstance(proc, Sweep):
        a, b = max(src.lo, ctx.lo), min(src.hi, ctx.hi)
        fr = [0, 0.25, 0.5, 0.75, 0.9, 1]
        per_point = proc.settle + 0.1
        for lo_f in fr:
            for hi_f in fr:
                if hi_f <= lo_f:
                    continue
                for steps in (2, 3, 4, 6, 8, 12, 16, 24):
                    out.append((steps * per_point, replace(proc, lo=a + (b - a) * lo_f, hi=a + (b - a) * hi_f, steps=steps)))
        return out, [replace(proc, lo=a, hi=b, steps=max(proc.steps, 33))]
    levels = {(proc.lo, proc.hi), (src.lo, src.hi), (src.hi, src.lo)}
    maximal = []
    for lo, hi in sorted(levels):
        for kd in (0.25, 0.5, 1, 2, 4):
            for kt in (0.125, 0.25, 0.5, 1, 2, 4):
                dt, dur = proc.dt * kd, proc.duration * kt
                if dur <= dt or dur / dt > MAX_SAMPLES:
                    continue
                out.append((dur + 1e-5 * dur / dt, replace(proc, lo=lo, hi=hi, dt=dt, duration=dur)))
        dur = proc.duration * 4
        maximal.append(replace(proc, lo=lo, hi=hi, duration=dur, dt=max(proc.dt / 4, dur / MAX_SAMPLES)))
    return out, maximal


# ---------------------------------------------------------------- driver

@dataclass
class Rated:
    proc: Sweep | Step
    seps: list[float]
    z: list[list[float]]

    @property
    def worst(self) -> float:
        return min(self.seps)


def rate(proc, rivals: list[Rivals], ctx: Context, stop_below=None) -> Rated:
    rows = rows_for(proc, ctx)
    seps, zs = [], []
    for r in rivals:
        s, z = r.separation(rows, stop_below)
        seps.append(s)
        zs.append(z)
        if stop_below is not None and s <= stop_below:
            break
    return Rated(proc, seps + [0.0] * (len(rivals) - len(seps)), zs)


def best_maximal(maximal, rivals, ctx) -> Rated:
    return max((rate(p, rivals, ctx) for p in maximal), key=lambda r: r.worst)


def tightened(rig: Rig, item: str, factor: float) -> Rig:
    out = copy.deepcopy(rig)
    if item in out.parts:
        out.parts[item].tol *= factor
    else:
        inst = out.sources.get(item) or out.sensors[item]
        inst.acc = Accuracy(inst.acc.rel * factor, inst.acc.abs * factor)
    return out


def upgrades(claim: Claim, rig: Rig, rivals: list[Rivals], ctx: Context, maximal) -> list[dict]:
    """Which single declared tolerance/accuracy, tightened how far, makes some experiment decisive."""
    items = sorted({n for r in rivals for box in r.boxes for n in box if n in rig.parts})
    items += [claim.procedure.source, rivals[0].target, *ctx.needed]

    def decisive(rig2: Rig) -> bool:
        for r in rivals:
            r.set_rig(rig2)
        try:
            return any(rate(p, rivals, ctx, stop_below=MARGIN).worst > MARGIN for p in maximal)
        finally:
            for r in rivals:
                r.set_rig(rig)

    advice = []
    for item in items:
        if not decisive(tightened(rig, item, 0.1)):
            continue
        factor = 0.1
        for f in (0.5, 0.3):
            if decisive(tightened(rig, item, f)):
                factor = f
                break
        advice.append({"item": item, "factor": factor, "text": describe_upgrade(rig, item, factor)})
    if not advice:  # no single upgrade suffices: tighten everything, then loosen back what is not needed
        need = list(items)
        if not decisive(_tighten_all(rig, need)):
            return [{"item": "*", "factor": None,
                     "text": "even 10x better parts and instruments would not decide it on this rig: the laws "
                             "make nearly the same predictions here; a different rig or experiment kind is needed"}]
        for item in items:
            trial = [i for i in need if i != item]
            if decisive(_tighten_all(rig, trial)):
                need = trial
        advice.append({"item": "+".join(need), "factor": 0.1,
                       "text": "all of: " + "; ".join(describe_upgrade(rig, i, 0.1) for i in need)})
    return advice


def _tighten_all(rig: Rig, items: list[str]) -> Rig:
    for item in items:
        rig = tightened(rig, item, 0.1)
    return rig


def describe_upgrade(rig: Rig, item: str, factor: float) -> str:
    if item in rig.parts:
        p = rig.parts[item]
        return f"{item}: tolerance ±{p.tol * 100:.3g} % -> ±{p.tol * factor * 100:.3g} % (measure it on a calibration rig)"
    inst = rig.sources.get(item) or rig.sensors[item]
    d = KIND_DIM[inst.kind]
    return (f"{item}: accuracy {inst.acc.rel * 100:.3g} % + {fmt(inst.acc.abs, d)} -> "
            f"{inst.acc.rel * factor * 100:.3g} % + {fmt(inst.acc.abs * factor, d)}")


def law_text(source: str, law: Law) -> str:
    line = source.splitlines()[law.line - 1].strip()
    return line.split(None, 1)[1].split("#")[0].strip() if line.split(None, 1)[0] in ("law", "against") else line


def rewrite(source: str, claim: Claim, rig: Rig, proc, suffix: str = "_d") -> str:
    """The claim's own text with its procedure replaced (and a new name, so it can sit next to the original)."""
    lines = source.splitlines()
    start = claim.line - 1
    end = next(i for i in range(start, len(lines)) if lines[i].strip() == "}")
    body, inserted = [], False
    for ln in lines[start + 1:end]:
        if ln.split() and ln.split()[0] in ("sweep", "observe", "step", "sample"):
            if not inserted:
                body.append(procedure_text(proc, rig))
                inserted = True
            continue
        body.append(ln)
    head = lines[start].replace(f"claim {claim.name} ", f"claim {claim.name}{suffix} ", 1)
    return "\n".join([head, *body, "}"]) + "\n"


def design(claim: Claim, rig: Rig, source: str, pilot: list[dict] | None = None) -> dict:
    """Find the cheapest experiment (same kind as the claim's) that decides the claim against its rivals."""
    target = target_of(claim, rig)
    ctx = Context(claim, rig, target, pilot)
    rivals = [Rivals(claim, rig, law, target) for law in claim.against]
    cands, maximal = candidates(claim.procedure, rig, ctx)
    current = rate(claim.procedure, rivals, ctx) if _inside(claim.procedure, ctx) else None
    best_max = best_maximal(maximal, rivals, ctx)
    out = {
        "claim": claim.name,
        "target": target,
        "rivals": [law_text(source, l) for l in claim.against],
        "current": current and {"separation": current.seps, "decisive": current.worst > MARGIN},
        "best_possible": {"separation": best_max.seps, "decisive": best_max.worst > MARGIN,
                          "procedure": procedure_text(best_max.proc, rig)},
    }
    if best_max.worst <= MARGIN:
        out["decisive"] = False
        out["upgrades"] = upgrades(claim, rig, rivals, ctx, [best_max.proc])
        return out
    cost, r = None, best_max
    for c, proc in sorted(cands, key=lambda c: c[0]):
        if cost is not None and c > cost:
            break  # among equally cheap decisive experiments, keep the most decisive
        rated = rate(proc, rivals, ctx, stop_below=MARGIN)
        if rated.worst > MARGIN and (cost is None or rated.worst > r.worst):
            cost, r = c, rated
    rows = rows_for(r.proc, ctx)
    where = [_row_text(rv.where(rows, z), rig) for rv, z in zip(rivals, r.z)]
    out.update({
        "decisive": True,
        "procedure": procedure_text(r.proc, rig),
        "separation": r.seps,
        "differ_most_at": where,
        "cost_s": cost,
        "claim_text": rewrite(source, claim, rig, r.proc),
    })
    return out


def _inside(proc, ctx: Context) -> bool:
    return not isinstance(proc, Sweep) or not ctx.needed or ctx.lo <= proc.lo and proc.hi <= ctx.hi


def _row_text(row: dict, rig: Rig) -> str:
    parts = []
    for k, v in row.items():
        if k == "t":
            parts.append(f"t = {fmt(v, SECOND)}")
        elif k in rig.sources:
            parts.append(f"{k} = {fmt(v, KIND_DIM[rig.sources[k].kind])}")
    return ", ".join(parts)

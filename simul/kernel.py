"""The verification kernel.

Semantics of a law `A == B within s %` on a dataset D:

    CONSISTENT  iff  there exist part values p, each within its declared
                     tolerance, such that for every data row x in D
                       |A(x,p) - B(x,p)|  <=  sum_i |d(A-B)/dx_i| * u_i(x_i)
                                              + s * max(|A|, |B|)
                     where u_i is the worst-case error bound the rig declares
                     for instrument i.

The part values p are the *witness*. Finding one is search (expensive,
untrusted); checking one is a single pass over the data (cheap, trusted).
That is the same split as proof search vs. proof checking in Lean.

Verdicts:
    CONFIRMED     the law is consistent and every `against` law is not
    INCONCLUSIVE  the law and some competing law are both consistent:
                  this rig cannot tell them apart
    REFUTED       no witness was found for the law
    DATA          an `experiment`: no law, the data itself is the product
"""
from __future__ import annotations

import hashlib
import itertools
import json
from dataclasses import dataclass

from .backends import Backend
from .check import check_claim, evaluate, existentials, observed
from .lang import Claim, Law, Program, Rig, SimulError, Step, Sweep, parse

CERT_VERSION = 1


@dataclass
class Fit:
    ratio: float  # worst residual / allowed bound; <= 1 means consistent
    witness: dict[str, float]
    worst_row: int

    @property
    def consistent(self) -> bool:
        return self.ratio <= 1.0


# ---------------------------------------------------------------- experiment

def run_experiment(claim: Claim, rig: Rig, dev: Backend) -> list[dict[str, float]]:
    """Run the procedure. Every row also carries the rig's monitor sensors."""
    proc = claim.procedure
    hello = dev.hello()
    if rig.device and hello["device"] != rig.device:
        raise SimulError(f"rig {rig.name} describes device {rig.device!r}, but {hello['device']!r} is connected")
    chans = observed(claim, rig)
    missing = [c for c in [proc.source, *chans] if c not in hello["channels"]]
    if missing:
        raise SimulError(f"device lacks channels {missing}")
    rows = []
    if isinstance(proc, Sweep):
        for k in range(proc.steps):
            v = proc.lo + (proc.hi - proc.lo) * k / (proc.steps - 1)
            dev.set(proc.source, v, proc.settle)
            rows.append({proc.source: v, **dev.read(chans)})
    else:
        fast = [c for c in proc.sample]
        slow = [m for m in chans if m not in fast]
        dev.set(proc.source, proc.lo)
        env = dev.read(slow) if slow else {}  # monitors: read once around the transient
        n = int(round(proc.duration / proc.dt))
        data = dev.step(proc.source, proc.lo, proc.hi, fast, proc.dt, n)
        rows = [{"t": data["t"][i], proc.source: proc.hi, f"{proc.source}_0": proc.lo, **env,
                 **{c: data[c][i] for c in fast}} for i in range(n)]
    for i, row in enumerate(rows):
        for ch in chans:
            s = rig.sensors[ch]
            # a reading may stray past the range by its own error bound (e.g. offset near 0 V)
            if not s.lo - s.acc.bound(s.lo) <= row[ch] <= s.hi + s.acc.bound(s.hi):
                raise SimulError(f"row {i}: {ch} = {row[ch]:.4g} is outside the sensor range; data is invalid")
    return rows


# ---------------------------------------------------------------- checking

def uncertainties(rig: Rig, row: dict[str, float]) -> dict[str, float]:
    u = {}
    for name, x in row.items():
        inst = rig.sources.get(name.removesuffix("_0")) or rig.sensors.get(name)
        u[name] = inst.acc.bound(x) if inst else 0.0  # `t`: clock error not modelled yet
    return u


def row_ratio(law: Law, rig: Rig, row: dict[str, float], witness: dict[str, float]) -> float:
    env = {**row, **witness}
    a, b = evaluate(law.lhs, env), evaluate(law.rhs, env)
    bound = law.slack * max(abs(a), abs(b))
    for name, u in uncertainties(rig, row).items():
        if u == 0:
            continue
        h = max(abs(row[name]) * 1e-6, 1e-15)
        hi, lo = dict(env), dict(env)
        hi[name] += h
        lo[name] -= h
        deriv = ((evaluate(law.lhs, hi) - evaluate(law.rhs, hi)) - (evaluate(law.lhs, lo) - evaluate(law.rhs, lo))) / (2 * h)
        bound += abs(deriv) * u
    resid = abs(a - b)
    if bound == 0:
        return 0.0 if resid == 0 else float("inf")
    return resid / bound


def check_witness(law: Law, claim: Claim, rig: Rig, rows: list[dict[str, float]], witness: dict[str, float]) -> Fit:
    """The trusted check. Cheap: one pass over the data."""
    bounds = existentials(law, claim, rig)
    if set(witness) != set(bounds):
        return Fit(float("inf"), witness, -1)
    for name, v in witness.items():
        lo, hi, _ = bounds[name]
        if not lo <= v <= hi:
            return Fit(float("inf"), witness, -1)
    worst, worst_i = 0.0, 0
    for i, row in enumerate(rows):
        try:
            r = row_ratio(law, rig, row, witness)
        except (ArithmeticError, ValueError):
            r = float("inf")
        if r > worst:
            worst, worst_i = r, i
    return Fit(worst, witness, worst_i)


def search_witness(law: Law, claim: Claim, rig: Rig, rows: list[dict[str, float]], grid: int = 7, rounds: int = 8) -> Fit:
    """Untrusted search: zooming grid over the tolerance box, minimizing the worst ratio.

    NOTE: a grid search can miss a witness, so REFUTED currently means
    "no witness found", not "none exists". Sound refutation needs interval
    branch-and-bound; see README roadmap.
    """
    bounds = existentials(law, claim, rig)
    names = list(bounds)
    box = {n: (lo, hi) for n, (lo, hi, _) in bounds.items()}
    best = check_witness(law, claim, rig, rows, {n: g for n, (_, _, g) in bounds.items()})
    if not names:
        return best
    if len(names) > 3:
        grid = 5
    for _ in range(rounds):
        axes = [[lo + (hi - lo) * k / (grid - 1) for k in range(grid)] for lo, hi in box.values()]
        for point in itertools.product(*axes):
            fit = check_witness(law, claim, rig, rows, dict(zip(names, point)))
            if fit.ratio < best.ratio:
                best = fit
        if best.consistent:
            break
        for n in names:  # zoom in around the best point, staying inside tolerance
            lo, hi = box[n]
            half = (hi - lo) * 0.3
            plo, phi, _ = bounds[n]
            box[n] = (max(plo, best.witness[n] - half), min(phi, best.witness[n] + half))
    return best


def verdict(claim: Claim, rig: Rig, rows: list[dict[str, float]]) -> dict:
    if claim.law is None:
        return {"verdict": "DATA", "law": None, "against": []}
    main = search_witness(claim.law, claim, rig, rows)
    rivals = [search_witness(law, claim, rig, rows) for law in claim.against]
    if not main.consistent:
        v = "REFUTED"
    elif any(r.consistent for r in rivals):
        v = "INCONCLUSIVE"
    else:
        v = "CONFIRMED"
    return {
        "verdict": v,
        "law": {"ratio": main.ratio, "witness": main.witness, "worst_row": main.worst_row},
        "against": [{"ratio": r.ratio, "witness": r.witness, "consistent": r.consistent} for r in rivals],
    }


# ---------------------------------------------------------------- certificates

def prove(prog: Program, claim_name: str, dev: Backend) -> dict:
    claim = prog.claims[claim_name]
    rig = check_claim(claim, prog)
    rows = run_experiment(claim, rig, dev)
    return {
        "simul_cert": CERT_VERSION,
        "claim": claim_name,
        "source_sha256": hashlib.sha256(prog.source.encode()).hexdigest(),
        "source": prog.source,
        "device": dev.hello(),
        "data": rows,
        **verdict(claim, rig, rows),
    }


def recheck(cert: dict) -> tuple[bool, str]:
    """Re-verify a certificate offline: no hardware, no trust in whoever made it."""
    if cert.get("simul_cert") != CERT_VERSION:
        return False, "unsupported certificate version"
    if hashlib.sha256(cert["source"].encode()).hexdigest() != cert["source_sha256"]:
        return False, "source hash mismatch"
    prog = parse(cert["source"])
    claim = prog.claims[cert["claim"]]
    rig = check_claim(claim, prog)
    rows = cert["data"]
    if cert["verdict"] in ("CONFIRMED", "INCONCLUSIVE"):
        fit = check_witness(claim.law, claim, rig, rows, cert["law"]["witness"])
        if not fit.consistent:
            return False, f"witness does not hold (ratio {fit.ratio:.3g} at row {fit.worst_row})"
    redo = verdict(claim, rig, rows)  # negative parts have no short witness: search again
    if redo["verdict"] != cert["verdict"]:
        return False, f"verdict mismatch: certificate says {cert['verdict']}, recheck gives {redo['verdict']}"
    return True, cert["verdict"]


def dumps(cert: dict) -> str:
    return json.dumps(cert, indent=2, ensure_ascii=False)

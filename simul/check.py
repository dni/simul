"""Static checks: name resolution, dimensional analysis, rig capability.

Everything here runs before any hardware is touched. A claim that passes is
well-formed: its law is dimensionally consistent and its experiment is
physically executable on the declared rig.
"""
from __future__ import annotations

import math

from .lang import Bin, Call, Claim, Expr, Law, Neg, Num, Pow, Program, Rig, SimulError, Step, Sweep, Var
from .units import KIND_DIM, ONE, PART_KINDS, SECOND, SENSOR_KINDS, SOURCE_KINDS, Dim, fmt

FUNCS = {"exp": math.exp, "ln": math.log, "sqrt": math.sqrt, "sin": math.sin, "cos": math.cos}
MAX_SAMPLES = 10_000
GROUND = "gnd"


def dim_of(e: Expr, env: dict[str, Dim]) -> Dim:
    match e:
        case Num(dim=d):
            return d
        case Var(name=n, line=line):
            if n not in env:
                raise SimulError(f"unknown or unobserved name {n!r} (in scope: {', '.join(sorted(env))})", line)
            return env[n]
        case Neg(arg=a):
            return dim_of(a, env)
        case Pow(base=b, exp=k):
            return dim_of(b, env) ** k
        case Call(fn=f, arg=a, line=line):
            d = dim_of(a, env)
            if f in ("exp", "ln", "sin", "cos"):
                if not d.dimensionless:
                    raise SimulError(f"argument of {f}() must be dimensionless, got {d}", line)
                return ONE
            if f == "sqrt":
                if any(x % 2 for x in d.exp):
                    raise SimulError(f"sqrt() of {d} has no integer dimension", line)
                return Dim(tuple(x // 2 for x in d.exp))
            raise SimulError(f"unknown function {f!r} (known: {', '.join(FUNCS)})", line)
        case Bin(op=op, left=l, right=r):
            dl, dr = dim_of(l, env), dim_of(r, env)
            if op in "+-":
                if dl != dr:
                    raise SimulError(f"cannot {'add' if op == '+' else 'subtract'} {dl} and {dr}")
                return dl
            return dl * dr if op == "*" else dl / dr
    raise TypeError(e)


def evaluate(e: Expr, env: dict[str, float]) -> float:
    match e:
        case Num(value=v):
            return v
        case Var(name=n):
            return env[n]
        case Neg(arg=a):
            return -evaluate(a, env)
        case Pow(base=b, exp=k):
            return evaluate(b, env) ** k
        case Call(fn=f, arg=a):
            return FUNCS[f](evaluate(a, env))
        case Bin(op=op, left=l, right=r):
            x, y = evaluate(l, env), evaluate(r, env)
            return x + y if op == "+" else x - y if op == "-" else x * y if op == "*" else x / y
    raise TypeError(e)


def compile_expr(e: Expr):
    """The same function as `evaluate`, compiled to Python for speed (untrusted tools only: design, search)."""
    def py(e: Expr) -> str:
        match e:
            case Num(value=v):
                return repr(v)
            case Var(name=n):
                return f"env[{n!r}]"
            case Neg(arg=a):
                return f"(-{py(a)})"
            case Pow(base=b, exp=k):
                return f"({py(b)} ** {k})"
            case Call(fn=f, arg=a):
                return f"_f[{f!r}]({py(a)})"
            case Bin(op=op, left=l, right=r):
                return f"({py(l)} {op} {py(r)})"
        raise TypeError(e)
    return eval(f"lambda env: {py(e)}", {"_f": FUNCS})


def free_vars(e: Expr) -> set[str]:
    match e:
        case Var(name=n):
            return {n}
        case Num():
            return set()
        case Neg(arg=a) | Pow(base=a) | Call(arg=a):
            return free_vars(a)
        case Bin(left=l, right=r):
            return free_vars(l) | free_vars(r)
    raise TypeError(e)


def check_rig(rig: Rig) -> None:
    def bad(name: str, msg: str):
        raise SimulError(f"rig {rig.name}, {name}: {msg}")

    nodes: dict[str, int] = {}
    for s in rig.sources.values():
        if s.kind not in SOURCE_KINDS:
            bad(s.name, f"source kind must be one of {SOURCE_KINDS}, not {s.kind!r}")
        if s.lo >= s.hi:
            bad(s.name, "empty range")
    for s in rig.sensors.values():
        if s.kind not in SENSOR_KINDS:
            bad(s.name, f"sensor kind must be one of {SENSOR_KINDS}, not {s.kind!r}")
        if s.lo >= s.hi:
            bad(s.name, "empty range")
        if s.where in ("across", "through") and s.kind not in ("voltage", "current"):
            bad(s.name, f"a {s.kind} sensor is placed 'at <part>' or 'ambient', not '{s.where}'")
        if s.where == "across" and s.kind != "voltage" or s.where == "through" and s.kind != "current":
            bad(s.name, "'across' measures voltage, 'through' measures current")
        if s.where in ("across", "through", "at") and s.target not in rig.parts and s.target not in rig.sources:
            bad(s.name, f"placed at unknown element {s.target!r}")
    for p in rig.parts.values():
        if p.kind not in PART_KINDS:
            bad(p.name, f"part kind must be one of {PART_KINDS}, not {p.kind!r}")
        if not 0 <= p.tol < 1:
            bad(p.name, "tolerance must be in [0, 100) %")
        if p.nominal <= 0:
            bad(p.name, "nominal value must be positive")
    for el in [*rig.parts.values(), *rig.sources.values()]:
        a, b = el.nodes
        if a == b:
            bad(el.name, f"both terminals on node {a!r} (shorted)")
        for n in (a, b):
            nodes[n] = nodes.get(n, 0) + 1
    if rig.parts and GROUND not in nodes:
        raise SimulError(f"rig {rig.name}: netlist has no '{GROUND}' node")
    for n, count in nodes.items():
        if count < 2:
            raise SimulError(f"rig {rig.name}: node {n!r} has only one connection (open circuit?)")
    for m in rig.monitors:
        if m not in rig.sensors:
            raise SimulError(f"rig {rig.name}: monitor {m!r} is not a sensor")


def observed(claim: Claim, rig: Rig) -> list[str]:
    proc = claim.procedure
    chans = proc.observe if isinstance(proc, Sweep) else proc.sample
    return chans + [m for m in rig.monitors if m not in chans]


def scope(claim: Claim, rig: Rig) -> dict[str, Dim]:
    """Names a law may mention, with their dimensions."""
    proc = claim.procedure
    env = {p.name: KIND_DIM[p.kind] for p in rig.parts.values()}
    env |= {p.name: p.dim for p in claim.params.values()}
    src = rig.sources[proc.source]
    env[src.name] = KIND_DIM[src.kind]
    for n in observed(claim, rig):
        env[n] = KIND_DIM[rig.sensors[n].kind]
    if isinstance(proc, Step):
        env["t"] = SECOND
        env[f"{src.name}_0"] = KIND_DIM[src.kind]  # level before the step; `V` is the level after
    return env


def check_claim(claim: Claim, prog: Program) -> Rig:
    where = f"{'experiment' if claim.experiment else 'claim'} {claim.name}"
    if claim.rig not in prog.rigs:
        raise SimulError(f"{where}: unknown rig {claim.rig!r}", claim.line)
    rig = prog.rigs[claim.rig]
    proc = claim.procedure
    if proc is None:
        raise SimulError(f"{where}: needs a procedure (sweep or step)", claim.line)
    if claim.law is None and not claim.experiment:
        raise SimulError(f"{where}: needs a law (or declare it as an `experiment`)", claim.line)
    if proc.source not in rig.sources:
        raise SimulError(f"{where}: {proc.source!r} is not a source of rig {rig.name}", claim.line)
    src = rig.sources[proc.source]
    if proc.dim != KIND_DIM[src.kind]:
        raise SimulError(f"{where}: {src.name} is a {src.kind} source, but the procedure is in {proc.dim}", claim.line)
    for v in (proc.lo, proc.hi):
        if not src.lo <= v <= src.hi:
            d = KIND_DIM[src.kind]
            raise SimulError(
                f"{where}: {fmt(v, d)} is outside the range of {src.name} "
                f"[{fmt(src.lo, d)} .. {fmt(src.hi, d)}]", claim.line)
    chans = proc.observe if isinstance(proc, Sweep) else proc.sample
    if not chans:
        raise SimulError(f"{where}: observes nothing", claim.line)
    for n in chans:
        if n not in rig.sensors:
            raise SimulError(f"{where}: {n!r} is not a sensor of rig {rig.name}", claim.line)
    if isinstance(proc, Sweep) and proc.steps < 2:
        raise SimulError(f"{where}: a sweep needs at least 2 steps", claim.line)
    if isinstance(proc, Step):
        if proc.dt <= 0 or proc.duration <= proc.dt:
            raise SimulError(f"{where}: need 0 < sample interval < duration", claim.line)
        if proc.duration / proc.dt > MAX_SAMPLES:
            raise SimulError(f"{where}: more than {MAX_SAMPLES} samples", claim.line)
    for p in claim.params.values():
        if p.name in rig.parts or p.name in rig.sensors or p.name in rig.sources:
            raise SimulError(f"{where}: param {p.name!r} shadows a rig element", claim.line)
        if not p.lo < p.hi:
            raise SimulError(f"{where}: param {p.name!r} has an empty range", claim.line)

    env = scope(claim, rig)
    for law in [claim.law, *claim.against] if claim.law else []:
        dl, dr = dim_of(law.lhs, env), dim_of(law.rhs, env)
        if dl != dr:
            raise SimulError(f"{where}: law is dimensionally inconsistent: {dl} == {dr}", law.line)
        if not (free_vars(law.lhs) | free_vars(law.rhs)) & set(observed(claim, rig)):
            raise SimulError(f"{where}: law mentions no observed quantity, nothing to test", law.line)
    return rig


def check_program(prog: Program) -> None:
    for rig in prog.rigs.values():
        check_rig(rig)
    for claim in prog.claims.values():
        check_claim(claim, prog)


def existentials(law: Law, claim: Claim, rig: Rig) -> dict[str, tuple[float, float, float]]:
    """Parts and params a law mentions: name -> (lo, hi, starting guess).

    These are existentially quantified: the claim says *some* value in
    [lo, hi] makes the law hold. Narrower ranges make stronger claims.
    """
    used = free_vars(law.lhs) | free_vars(law.rhs)
    out = {}
    for name in sorted(used):
        if name in rig.parts:
            p = rig.parts[name]
            out[name] = (*p.bounds, p.nominal)
        elif name in claim.params:
            p = claim.params[name]
            out[name] = (p.lo, p.hi, (p.lo + p.hi) / 2)
    return out

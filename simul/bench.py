"""Hidden-physics benchmark: measure how much an AI scientist finds.

Each scenario is a rig file plus physics the rig never mentions (contact
with the outside world: Earth's field, a leaky capacitor, mains hum...).
The solver sees only the rig blocks (`blind`: no example claims, no
comments) and an experiment budget.

Scoring needs no judge, human or AI. A certified claim *detects* a hidden
effect iff, on a fresh simulator with the same seed,

    the claim is CONFIRMED with the effect on, and REFUTED with it off.

That is a counterfactual test: the claim says something that is true only
because the effect is there. Words in a report do not score; certified
claims do.

Solvers:
    baseline   the scripted loop: explore with the procedures a human wrote
               for this rig, discover candidate laws, certify every one
    agent      Claude, through simul.agent (needs an API key; costs money)
"""
from __future__ import annotations

import pathlib
import time
from dataclasses import dataclass

from . import notebook
from .backends import SimBackend
from .discover import procedure_text
from .kernel import prove
from .lab import Lab
from .lang import SimulError, Sweep, parse

EXAMPLES = pathlib.Path(__file__).resolve().parent.parent / "examples"


@dataclass
class Scenario:
    name: str
    rig_file: str
    effects: dict
    answer: str  # what a human would call it; never shown to the solver


SCENARIOS = [
    Scenario("rl_background", "rl.phy", {"earth_field": 150e-6},
             "a 150 uT background field (Earth plus a magnet on the bench) adds to the coil's field"),
    Scenario("heat_joule", "heat.phy", {"self_heating": 1.0}, "the coil heats itself: T rises with power, R with T"),
    Scenario("rc_leak", "rc.phy", {"leaky_capacitor": 100e3}, "the capacitor leaks through ~100 kOhm"),
    Scenario("ohm_offset", "ohm.phy", {"source_offset": 0.06}, "the source outputs 60 mV more than commanded"),
    Scenario("ohm_nonlinear", "ohm.phy", {"nonlinear_sensor": ["Ir", 0.5]},
             "the current sensor reads up to 16 % high at high current (quadratic error)"),
    Scenario("rc_hum", "rc.phy", {"mains_hum": 0.15}, "150 mV of 50 Hz mains pickup on the fast ADC"),
    # two effects at once: one can hide the other
    Scenario("rc_leak_hum", "rc.phy", {"leaky_capacitor": 100e3, "mains_hum": 0.15},
             "a capacitor leaking through ~100 kOhm, and 150 mV of 50 Hz pickup"),
    Scenario("rl_field_sensor", "rl.phy", {"earth_field": 150e-6, "nonlinear_sensor": ["B", 0.3]},
             "a 150 uT background field, and a Hall sensor that reads high at strong fields"),
    Scenario("heat_offset", "heat.phy", {"self_heating": 1.0, "source_offset": 0.04},
             "self-heating, and a source that outputs 40 mV more than commanded"),
]


def detects(entry: dict, effects: dict, effect: str, seed: int) -> bool:
    """Counterfactual check: does this certified claim depend on `effect`?"""
    prog = parse(entry["source"])
    rig = prog.rigs[prog.claims[entry["claim"]].rig]
    try:
        on = prove(prog, entry["claim"], SimBackend(rig, seed, effects=effects))
        off = prove(prog, entry["claim"], SimBackend(rig, seed, effects={k: v for k, v in effects.items() if k != effect}))
    except SimulError:
        return False
    return on["verdict"] == "CONFIRMED" and off["verdict"] == "REFUTED"


def score(lab: Lab, sc: Scenario, seed: int) -> dict:
    entries = notebook.entries(lab.notebook)
    confirmed = [e for e in entries if e["verdict"] == "CONFIRMED"]
    found = {}
    for effect in sc.effects:
        for e in confirmed:
            if detects(e, sc.effects, effect, seed):
                found[effect] = {"entry": e["id"], "claim": e["claim"], "source": _claim_block(e)}
                break
    return {"scenario": sc.name, "effects": list(sc.effects), "found": found,
            "missed": [k for k in sc.effects if k not in found], "runs": lab.attempts,
            "certified": len(confirmed), "answer": sc.answer}


def _claim_block(entry: dict) -> str:
    lines = entry["source"].splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.startswith(f"claim {entry['claim']} "))
    end = next(i for i in range(start, len(lines)) if lines[i].strip() == "}")
    return "\n".join(lines[start:end + 1])


# ---------------------------------------------------------------- solvers

def baseline(lab: Lab, sc: Scenario, budget: int, log=print) -> None:
    """Discover from the human-written procedures for this rig, then certify every candidate on fresh data."""
    original = parse((EXAMPLES / sc.rig_file).read_text(encoding="utf-8"))
    explorations, seen = [], set()
    for c in original.claims.values():
        rig = lab.base.rigs[c.rig]
        proc = c.procedure
        if isinstance(proc, Sweep):  # observe everything the rig can see
            proc = Sweep(proc.source, proc.lo, proc.hi, proc.steps, [s for s in rig.sensors if s not in rig.monitors],
                         proc.settle, proc.dim)
        text = procedure_text(proc, rig)
        if (c.rig, text) not in seen:
            seen.add((c.rig, text))
            explorations.append(f"experiment x{len(explorations) + 1} on {c.rig} {{\n{text}\n}}\n")
    used = 0
    for text in explorations:
        if used >= budget:
            return
        name = text.split()[1]
        used += 1
        try:
            _, cands = lab.discover(text, name, by="baseline", why="explore")
        except SimulError as e:  # e.g. a reading outside the sensor's range: the run is spent, the data void
            log(f"  error: {e}")
            continue
        for cand in cands:
            if used >= budget:
                return
            cname = cand["claim"].split()[1]
            cert = lab.run(cand["claim"], cname, by="baseline", why=f"certify {cand['law']}")
            used += 1
            log(f"  {cert['verdict']:<12} {cand['law']}")


def agent(lab: Lab, sc: Scenario, budget: int, log=print) -> None:
    from .agent import run_agent
    run_agent(lab, "Find and certify the laws that govern this hardware. Pay special attention to anything the "
                   "rig description does not account for: certify it with a claim that would fail without it.", budget)


SOLVERS = {"baseline": baseline, "agent": agent}


def run(solver: str = "baseline", names: list[str] | None = None, budget: int = 12, seed: int = 0,
        out_dir: str | None = None, log=print) -> list[dict]:
    out = pathlib.Path(out_dir or f"lab/bench/{time.strftime('%Y%m%d-%H%M%S')}-{solver}")
    results = []
    for sc in SCENARIOS:
        if names and sc.name not in names:
            continue
        log(f"[{sc.name}]")
        lab = Lab([str(EXAMPLES / sc.rig_file)], f"sim:{seed}", str(out / f"{sc.name}.jsonl"),
                  effects=sc.effects, blind=True)
        SOLVERS[solver](lab, sc, budget, log)
        res = score(lab, sc, seed)
        log(f"  found {len(res['found'])}/{len(sc.effects)} using {res['runs']} runs"
            + "".join(f"\n  ✓ {k}: entry #{v['entry']} {v['claim']}" for k, v in res["found"].items())
            + "".join(f"\n  ✗ {k} ({sc.answer})" for k in res["missed"]))
        results.append(res)
    total = sum(len(r["found"]) for r in results), sum(len(r["effects"]) for r in results)
    log(f"\nscore: {total[0]}/{total[1]} hidden effects certified ({solver}, budget {budget}, seed {seed})")
    return results

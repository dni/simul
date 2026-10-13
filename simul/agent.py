"""An AI experimentalist: Claude drives the lab through the same tools a human uses.

Trust boundary: Claude proposes experiments and claims and interprets results.
It never decides a verdict. Verdicts come from the kernel, every run is logged
in the notebook, and rig files (the hardware facts) cannot be changed from here.
"""
from __future__ import annotations

import json

import anthropic

from . import lang, notebook
from .lab import Lab
from .lang import SimulError

MODEL = "claude-opus-5-5"

SYSTEM = f"""You are an experimental physicist operating an automated electronics lab through the Simul language.

The lab's hardware is described by rig files (call describe_hardware). Rigs are fixed facts written by humans: \
you cannot change the wiring or add instruments, but you can design any experiment the rigs allow and state any \
claim about them.

How the lab decides what is true:
- A claim is CONFIRMED when some values of its parts and params, within their declared ranges, fit EVERY data row \
within the instruments' worst-case error bounds, and every `against` rival does not. REFUTED when no such values \
exist. INCONCLUSIVE when a rival fits too: the rig cannot tell them apart.
- A verdict says what is true of THIS hardware within the stated precision, nothing more. Report it that way.

Method:
1. Explore: run `experiment` blocks via discover() to get data and candidate laws. Vary what the rigs let you vary \
(source range, step vs sweep, sample rate, duration).
2. When two laws compete, call design() before spending runs: it says whether ANY experiment on this rig can \
tell them apart, gives the cheapest one that can, and otherwise names the hardware upgrade that would.
3. Never trust a fit to the data it came from. Certify every law with run() on a NEW claim, which takes fresh data.
4. Make claims strong: narrow param ranges, and an `against` rival you actually believe someone might hold.
5. A constant the rig description cannot explain is the interesting part: new physics, or something the hardware \
description is missing. Form a hypothesis about what it is and test it with a different experiment.
6. If the instruments are too coarse to decide something, say exactly which instrument and what precision would decide it.

Each hardware run (run or discover) costs one unit of a limited budget; check() and design() are free, so use them before running.

When done, write a final report: certified laws (with notebook entry numbers), refuted ideas, unexplained constants \
and your best hypothesis for each, and concrete hardware changes that would let the next round decide more.

Simul language reference:
{lang.__doc__}
Laws use + - * / ^ (integer exponents), exp(), ln(), sqrt(), sin(), cos(), names from the rig, params, `t` (step experiments) and \
`<source>_0` (level before the step). Quantities need units: 3.3 V, 20 us, 10 kOhm, 293.15 K. Write temperatures \
in laws in K.
"""

TOOLS = [
    {
        "name": "describe_hardware",
        "description": "Return the rig files: the exact wiring (netlist), sources, sensors with ranges and accuracies, "
                       "part tolerances, and example claims/experiments.",
        "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
        "strict": True,
    },
    {
        "name": "check",
        "description": "Statically check Simul claims/experiments against the rigs (units, names, ranges). Free: no hardware run.",
        "input_schema": {
            "type": "object",
            "properties": {"source": {"type": "string", "description": "One or more claim/experiment blocks."}},
            "required": ["source"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "name": "run",
        "description": "Run one claim or experiment on the hardware (fresh data), get the kernel's verdict and a data "
                       "summary. Costs one budget unit. Logged in the notebook.",
        "input_schema": {
            "type": "object",
            "properties": {
                "source": {"type": "string", "description": "Simul text containing the claim/experiment."},
                "name": {"type": "string", "description": "Which claim/experiment in `source` to run."},
                "why": {"type": "string", "description": "One sentence: the hypothesis this run tests."},
            },
            "required": ["source", "name", "why"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "name": "discover",
        "description": "Run an `experiment` and propose candidate laws that fit its data within instrument error, with "
                       "constants named from the rig where possible. Returns ready-to-prove claim text. Costs one "
                       "budget unit. Candidates are NOT certified until you run() them as new claims.",
        "input_schema": {
            "type": "object",
            "properties": {
                "source": {"type": "string", "description": "Simul text containing the experiment."},
                "name": {"type": "string"},
                "why": {"type": "string"},
            },
            "required": ["source", "name", "why"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "name": "design",
        "description": "Plan the experiment that decides a claim against its `against` rivals. Free: no hardware run. "
                       "Reports the separation S per rival (S > 1: every possible outcome refutes at least one of the "
                       "two laws), the cheapest decisive procedure with a rewritten claim ready to run(), or, if no "
                       "experiment on this rig is decisive, which tolerances/accuracies would have to improve. Laws "
                       "must have the form `<sensor> == <expression>`. Readings the laws need but an experiment does "
                       "not set (e.g. a temperature) are taken from a pilot run in the notebook.",
        "input_schema": {
            "type": "object",
            "properties": {
                "source": {"type": "string", "description": "Simul text containing the claim (law + against)."},
                "name": {"type": "string", "description": "The claim to design for."},
                "pilot_entry": {"type": "integer", "description": "Notebook entry to take other readings from "
                                                                  "(default: latest suitable)."},
            },
            "required": ["source", "name"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "name": "notebook",
        "description": "List past runs in the lab notebook (all sessions, humans and agents): verdicts, witnesses, reasons.",
        "input_schema": {
            "type": "object",
            "properties": {"limit": {"type": "integer", "description": "How many recent entries (default 30)."}},
            "required": [],
            "additionalProperties": False,
        },
        "strict": True,
    },
]


def compact_rows(rows: list[dict], max_rows: int = 25) -> list[dict]:
    step = max(1, len(rows) // max_rows)
    return [{k: float(f"{v:.5g}") for k, v in r.items()} for r in rows[::step]]


class Session:
    def __init__(self, lab: Lab, budget: int):
        self.lab, self.budget = lab, budget

    def call(self, name: str, args: dict) -> str:
        if name == "describe_hardware":
            return self.lab.rigs()
        if name == "notebook":
            return json.dumps(notebook.summary(self.lab.notebook, args.get("limit") or 30))
        if name == "check":
            return f"ok: {', '.join(self.lab.check(args['source']))}"
        if name == "design":
            return json.dumps(self.lab.design(args["source"], args["name"], args.get("pilot_entry")))
        if self.budget <= 0:
            raise SimulError("experiment budget exhausted: write your final report now")
        self.budget -= 1
        left = f"budget left: {self.budget}"
        if name == "run":
            cert = self.lab.run(args["source"], args["name"], by="agent", why=args["why"])
            return json.dumps({"verdict": cert["verdict"], "law": cert["law"], "against": cert["against"],
                               "entry": cert["entry"], "rows": len(cert["data"]),
                               "data": compact_rows(cert["data"]), "note": left})
        if name == "discover":
            cert, cands = self.lab.discover(args["source"], args["name"], by="agent", why=args["why"])
            return json.dumps({"entry": cert["entry"], "rows": len(cert["data"]), "data": compact_rows(cert["data"]),
                               "candidates": [{k: c[k] for k in ("law", "claim", "unexplained", "notes")} for c in cands],
                               "note": left})
        raise SimulError(f"unknown tool {name!r}")


def run_agent(lab: Lab, goal: str, budget: int = 12, max_turns: int = 60) -> str:
    client = anthropic.Anthropic()
    session = Session(lab, budget)
    messages = [{"role": "user", "content": f"Goal: {goal}\nExperiment budget: {budget} hardware runs."}]
    for _ in range(max_turns):
        response = client.beta.messages.create(
            model=MODEL,
            max_tokens=16000,
            system=[{"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}],
            tools=TOOLS,
            messages=messages,
            thinking={"type": "adaptive", "display": "summarized"},
            output_config={"effort": "high"},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
        # append-only history: keep every block (thinking included) exactly as returned
        messages.append({"role": "assistant", "content": response.content})
        for block in response.content:
            if block.type == "thinking" and block.thinking:
                print(f"\033[2m· {block.thinking.strip()[:400]}\033[0m")
            elif block.type == "text":
                print(block.text)
        if response.stop_reason == "refusal":
            print("[the model declined to continue]")
            return ""
        if response.stop_reason != "tool_use":
            return next((b.text for b in reversed(response.content) if b.type == "text"), "")
        results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            print(f"\033[36m→ {block.name}\033[0m {block.input.get('name', '')} {block.input.get('why', '')}")
            try:
                out, err = session.call(block.name, block.input), False
            except (SimulError, ArithmeticError, KeyError, ValueError) as e:
                out, err = f"error: {e}", True
            results.append({"type": "tool_result", "tool_use_id": block.id, "content": out, "is_error": err})
        messages.append({"role": "user", "content": results})
    return ""

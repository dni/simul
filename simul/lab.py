"""A lab session: rig descriptions + connected devices + notebook.

Shared by the CLI and the agent. Rig files are human-written hardware
facts; claims and experiments are added on top of them per request, so a
caller (human or AI) can never redefine what the hardware is.
"""
from __future__ import annotations

import pathlib
import re

from .backends import Backend, open_backend
from .check import check_claim, check_program
from .design import design, needed_columns
from .discover import discover
from .kernel import prove, recheck
from .lang import Program, SimulError, Sweep, parse
from . import notebook


class Lab:
    def __init__(self, rig_files: list[str], backend: str = "sim", notebook_path: str | None = None,
                 truth: dict[str, float] | None = None, effects: dict | None = None, blind: bool = False):
        self.rig_source = "\n".join(pathlib.Path(f).read_text(encoding="utf-8") for f in rig_files)
        if blind:  # rig blocks only: no example claims, no comments that might hint at the physics
            self.rig_source = rigs_only(self.rig_source)
        self.base = parse(self.rig_source)
        check_program(self.base)
        self.backend_spec, self.truth, self.effects = backend, truth, effects
        self.attempts = 0  # hardware runs started, including ones whose data turned out invalid
        self.devices: dict[str, Backend] = {}
        self.notebook = pathlib.Path(notebook_path) if notebook_path else notebook.DEFAULT

    def device(self, rig: str) -> Backend:
        """One connection per rig for the whole session: same physical (or simulated) hardware every run."""
        if rig not in self.devices:
            self.devices[rig] = open_backend(self.backend_spec, self.base.rigs[rig], self.truth, self.effects)
        return self.devices[rig]

    def program(self, extra: str = "") -> Program:
        """Rig files + extra claims. Extra text may not declare rigs."""
        if extra:
            for name in parse(extra).rigs:
                raise SimulError(f"rig {name!r}: rigs are hardware facts and come from rig files only")
        prog = parse(self.rig_source + "\n" + extra)
        check_program(prog)
        return prog

    def check(self, text: str) -> list[str]:
        prog = self.program(text)
        return [n for n in parse(text).claims if n in prog.claims]

    def run(self, text: str, name: str, by: str = "human", why: str = "") -> dict:
        prog = self.program(text)
        if name not in prog.claims:
            raise SimulError(f"no claim or experiment named {name!r}")
        claim = prog.claims[name]
        self.attempts += 1
        cert = prove(prog, name, self.device(claim.rig))
        cert["entry"] = notebook.append(cert, self.notebook, by=by, why=why)
        return cert

    def discover(self, text: str, name: str, by: str = "human", why: str = "") -> tuple[dict, list[dict]]:
        cert = self.run(text, name, by, why)
        prog = self.program(text)
        claim = prog.claims[name]
        return cert, discover(claim, check_claim(claim, prog), cert["data"])

    def design(self, text: str, name: str, pilot_entry: int | None = None) -> dict:
        """Plan the experiment that decides `name` against its rivals. Free: no hardware run.

        Readings the laws need but an experiment does not set come from a pilot run: the given
        notebook entry, or else the latest one on the same rig that observed them.
        """
        prog = self.program(text)
        if name not in prog.claims:
            raise SimulError(f"no claim named {name!r}")
        claim = prog.claims[name]
        rig = check_claim(claim, prog)
        pilot = None
        needed = needed_columns(claim, rig)
        if needed:
            for e in reversed(notebook.entries(self.notebook)):
                if pilot_entry is not None and e["id"] != pilot_entry:
                    continue
                rows = e.get("data") or []
                if rows and all(c in rows[0] for c in needed) and parse(e["source"]).claims[e["claim"]].rig == rig.name:
                    if isinstance(claim.procedure, Sweep) and (claim.procedure.source not in rows[0] or "t" in rows[0]):
                        continue
                    pilot = rows
                    break
        out = design(claim, rig, prog.source, pilot)
        return out

    def rigs(self) -> str:
        return self.rig_source

    @staticmethod
    def recheck(cert: dict) -> tuple[bool, str]:
        return recheck(cert)


def rigs_only(text: str) -> str:
    """Just the rig blocks of a source text, without comments."""
    blocks = re.findall(r"^rig\s+\w+\s*\{.*?^\}", text, re.M | re.S)
    return "\n\n".join(re.sub(r"[ \t]*#.*", "", b) for b in blocks) + "\n"

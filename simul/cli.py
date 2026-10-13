"""simul check | prove | discover | recheck | notebook | agent"""
from __future__ import annotations

import argparse
import json
import sys

from .check import check_program
from .kernel import dumps, recheck
from .lab import Lab
from .lang import SimulError, parse
from . import notebook
from .units import parse_unit

COLOR = {"CONFIRMED": "\033[32m", "REFUTED": "\033[31m", "INCONCLUSIVE": "\033[33m", "DATA": "\033[36m"}


def parse_truth(items: list[str]) -> dict[str, float]:
    """--truth R=150Ohm  -> overrides a simulated part's real value (to inject faults)."""
    out = {}
    for item in items:
        name, _, q = item.partition("=")
        num = q.rstrip("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZΩµ")
        unit = q[len(num):]
        scale = parse_unit(unit)[0] if unit else 1.0
        out[name] = float(num) * scale
    return out


def show(name: str, cert: dict) -> None:
    v = cert["verdict"]
    if v == "DATA":
        print(f"{COLOR[v]}{v:<12}\033[0m {name:<24} {len(cert['data'])} rows  (notebook entry {cert.get('entry')})")
        return
    wit = ", ".join(f"{k}={x:.4g}" for k, x in cert["law"]["witness"].items()) or "none needed"
    print(f"{COLOR[v]}{v:<12}\033[0m {name:<24} worst residual {cert['law']['ratio']:.2f}× bound  witness: {wit}")
    for i, a in enumerate(cert["against"]):
        print(f"{'':13}against #{i + 1}: {'consistent' if a['consistent'] else 'excluded'} ({a['ratio']:.2f}× bound)")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="simul", description="Verify and discover physical laws on specified hardware.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def lab_args(p):
        p.add_argument("--backend", default="sim", help="sim | sim:<seed> | serial:<port>")
        p.add_argument("--truth", action="append", default=[], help="simulator only: force a part value, e.g. R=150Ohm")
        p.add_argument("--notebook", help=f"lab notebook path (default {notebook.DEFAULT})")

    c = sub.add_parser("check", help="static check only (units, names, netlist, rig capability)")
    c.add_argument("file")
    p = sub.add_parser("prove", help="run claims/experiments, log them, issue certificates")
    p.add_argument("file")
    p.add_argument("claim", nargs="?", help="default: every claim in the file")
    p.add_argument("-o", "--out", help="also write certificate(s) to this JSON file")
    lab_args(p)
    d = sub.add_parser("discover", help="run an experiment and propose laws; --verify proves them on fresh data")
    d.add_argument("file")
    d.add_argument("experiment")
    d.add_argument("--verify", type=int, default=0, metavar="N", help="prove the top N candidates on a fresh run")
    lab_args(d)
    r = sub.add_parser("recheck", help="re-verify a certificate offline")
    r.add_argument("cert")
    n = sub.add_parser("notebook", help="list past runs")
    n.add_argument("--notebook")
    n.add_argument("--limit", type=int, default=30)
    ds = sub.add_parser("describe", help="ask a device to describe itself and print a rig skeleton")
    ds.add_argument("backend", help="serial:<port> (or sim:<rigfile>:<rig> to test)")
    ds.add_argument("--name", default="bench")
    a = sub.add_parser("agent", help="let Claude run experiments and look for laws on its own")
    a.add_argument("rigs", nargs="+", help="rig files (hardware facts)")
    a.add_argument("--goal", default="Find and certify the laws that govern this hardware, including any effect the rig description does not account for.")
    a.add_argument("--budget", type=int, default=12, help="max experiments (hardware runs)")
    lab_args(a)
    args = ap.parse_args(argv)

    try:
        if args.cmd == "recheck":
            certs = json.load(open(args.cert))
            ok_all = True
            for cert in certs if isinstance(certs, list) else [certs]:
                ok, msg = recheck(cert)
                ok_all &= ok
                print(f"{'ok  ' if ok else 'FAIL'} {cert['claim']}: {msg}")
            return 0 if ok_all else 1
        if args.cmd == "notebook":
            for e in notebook.summary(notebook.DEFAULT if not args.notebook else __import__("pathlib").Path(args.notebook), args.limit):
                print(f"#{e['id']:<4} {e['time']}  {e['verdict']:<12} {e['claim']:<28} by {e['by']}: {e['why'][:60]}")
            return 0
        if args.cmd == "describe":
            from .backends import SerialBackend, SimBackend
            from .describe import rig_text
            if args.backend.startswith("sim:"):
                _, f, r = args.backend.split(":")
                dev = SimBackend(parse(open(f, encoding="utf-8").read()).rigs[r])
            else:
                dev = SerialBackend(args.backend.removeprefix("serial:"))
            print(rig_text(args.name, dev.hello()["device"], dev.describe()))
            return 0
        if args.cmd == "check":
            prog = parse(open(args.file, encoding="utf-8").read())
            check_program(prog)
            print(f"ok: {len(prog.rigs)} rig(s), {len(prog.claims)} claim(s)/experiment(s) well-formed")
            return 0
        if args.cmd == "agent":
            from .agent import run_agent
            lab = Lab(args.rigs, args.backend, args.notebook, parse_truth(args.truth))
            run_agent(lab, args.goal, args.budget)
            return 0

        lab = Lab([args.file], args.backend, args.notebook, parse_truth(args.truth))
        if args.cmd == "prove":
            certs = []
            for name in [args.claim] if args.claim else list(lab.base.claims):
                cert = lab.run("", name, by="cli")
                certs.append(cert)
                show(name, cert)
            if args.out:
                with open(args.out, "w", encoding="utf-8") as f:
                    f.write(dumps(certs[0] if len(certs) == 1 else certs))
                print(f"certificate written to {args.out}")
            return 0

        # discover
        cert, cands = lab.discover("", args.experiment, by="cli")
        show(args.experiment, cert)
        if not cands:
            print("no law fits within the instruments' error bounds (or nothing varied enough to measure)")
        for i, cand in enumerate(cands, 1):
            tag = "derived from rig" if not cand["unexplained"] else f"{cand['unexplained']} new constant(s)"
            print(f"\n[{i}] {cand['law']}    ({tag}{'; ' + '; '.join(cand['notes']) if cand['notes'] else ''})")
        for cand in cands[: args.verify]:
            print(f"\nverifying on fresh data:\n{cand['claim']}")
            name = cand["claim"].split()[1]
            show(name, lab.run(cand["claim"], name, by="cli", why=f"verify discovery from entry {cert['entry']}"))
        return 0
    except SimulError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())

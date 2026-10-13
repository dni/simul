"""Lexer, parser and AST for the Simul claim language.

    rig <name> {
      device <ident>                 # binds this description to one physical board
      source <name> : <kind> between <node> <node> range <qty> .. <qty> accuracy <pct> % + <qty>
      sensor <name> : <kind> (across <name> | through <name> | at <name> | ambient)
                      range <qty> .. <qty> accuracy <pct> % + <qty>
      part   <name> : <kind> <qty> ± <pct> % between <node> <node> [material <ident>]
      monitor <sensor>, ...          # logged with every data row of every experiment
    }

    claim <name> on <rig> {                               # `experiment` = a claim without a law,
      param <name> : <unit> in <num> .. <num>             # run for discovery
      sweep <source> from <qty> to <qty> steps <int> [settle <qty>]   # quasi-static
      observe <sensor>, ...
    | step <source> from <qty> to <qty>                   # transient experiment, defines `t`
      sample <sensor>, ... every <qty> for <qty>
      law <expr> == <expr> [within <pct> %]
      against <expr> == <expr> [within <pct> %]           # competing hypothesis, optional
    }
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .units import ABSOLUTE_OFFSET, KIND_DIM, SECOND, Dim, ONE, parse_unit


class SimulError(Exception):
    def __init__(self, msg: str, line: int | None = None):
        super().__init__(f"line {line}: {msg}" if line else msg)


# ---------------------------------------------------------------- AST

@dataclass
class Num:
    value: float
    dim: Dim = ONE


@dataclass
class Var:
    name: str
    line: int = 0


@dataclass
class Bin:
    op: str
    left: "Expr"
    right: "Expr"


@dataclass
class Neg:
    arg: "Expr"


@dataclass
class Pow:
    base: "Expr"
    exp: int


@dataclass
class Call:
    fn: str
    arg: "Expr"
    line: int = 0


Expr = Num | Var | Bin | Neg | Pow | Call


@dataclass
class Accuracy:
    rel: float  # fraction of reading
    abs: float  # SI units

    def bound(self, x: float) -> float:
        return self.rel * abs(x) + self.abs


@dataclass
class Source:
    name: str
    kind: str
    lo: float
    hi: float
    acc: Accuracy
    nodes: tuple[str, str] = ("", "")  # (+, -)


@dataclass
class Sensor:
    name: str
    kind: str
    where: str  # "across" | "through" | "at" | "ambient"
    target: str | None
    lo: float
    hi: float
    acc: Accuracy


@dataclass
class Part:
    name: str
    kind: str
    nominal: float
    tol: float  # fraction
    nodes: tuple[str, str] = ("", "")
    material: str | None = None

    @property
    def bounds(self) -> tuple[float, float]:
        return self.nominal * (1 - self.tol), self.nominal * (1 + self.tol)


@dataclass
class Rig:
    name: str
    device: str | None = None
    sources: dict[str, Source] = field(default_factory=dict)
    sensors: dict[str, Sensor] = field(default_factory=dict)
    parts: dict[str, Part] = field(default_factory=dict)
    monitors: list[str] = field(default_factory=list)


@dataclass
class Param:
    """An unknown constant the claim asserts exists within [lo, hi] (SI units)."""
    name: str
    dim: Dim
    lo: float
    hi: float
    unit: str = ""


@dataclass
class Law:
    lhs: Expr
    rhs: Expr
    slack: float = 0.0  # declared model idealization, fraction
    line: int = 0


@dataclass
class Sweep:
    source: str
    lo: float
    hi: float
    steps: int
    observe: list[str] = field(default_factory=list)
    settle: float = 0.0
    dim: Dim = ONE


@dataclass
class Step:
    source: str
    lo: float
    hi: float
    sample: list[str] = field(default_factory=list)
    dt: float = 0.0
    duration: float = 0.0
    dim: Dim = ONE


@dataclass
class Claim:
    name: str
    rig: str
    procedure: Sweep | Step | None = None
    law: Law | None = None
    against: list[Law] = field(default_factory=list)
    params: dict[str, Param] = field(default_factory=dict)
    line: int = 0
    experiment: bool = False  # declared with `experiment`: data collection, no law


@dataclass
class Program:
    rigs: dict[str, Rig]
    claims: dict[str, Claim]
    source: str


# ---------------------------------------------------------------- lexer

TOKEN = re.compile(
    r"""
    (?P<ws>[ \t\r]+|\#[^\n]*)
  | (?P<nl>\n)
  | (?P<num>\d+(\.\d+)?([eE][+-]?\d+)?)
  | (?P<ident>°C|[A-Za-zΩµ_][A-Za-z0-9Ωµ_]*)
  | (?P<sym>\.\.|==|\+-|±|[{}:,()+\-*/^%])
    """,
    re.VERBOSE,
)


@dataclass
class Tok:
    kind: str
    text: str
    line: int


def lex(src: str) -> list[Tok]:
    toks, pos, line = [], 0, 1
    while pos < len(src):
        m = TOKEN.match(src, pos)
        if not m:
            raise SimulError(f"unexpected character {src[pos]!r}", line)
        kind = m.lastgroup
        if kind == "nl":
            line += 1
        elif kind != "ws":
            text = m.group()
            toks.append(Tok("sym" if text in ("+-", "±") else kind, "±" if text == "+-" else text, line))
        pos = m.end()
    toks.append(Tok("eof", "", line))
    return toks


# ---------------------------------------------------------------- parser

class Parser:
    def __init__(self, src: str):
        self.src = src
        self.toks = lex(src)
        self.i = 0

    # helpers
    @property
    def tok(self) -> Tok:
        return self.toks[self.i]

    def next(self) -> Tok:
        t = self.toks[self.i]
        self.i += 1
        return t

    def at(self, text: str) -> bool:
        return self.tok.text == text and self.tok.kind in ("sym", "ident")

    def expect(self, text: str) -> Tok:
        if not self.at(text):
            raise SimulError(f"expected {text!r}, got {self.tok.text or 'end of file'!r}", self.tok.line)
        return self.next()

    def ident(self) -> str:
        if self.tok.kind != "ident":
            raise SimulError(f"expected a name, got {self.tok.text!r}", self.tok.line)
        return self.next().text

    def number(self) -> float:
        if self.tok.kind != "num":
            raise SimulError(f"expected a number, got {self.tok.text!r}", self.tok.line)
        return float(self.next().text)

    def qty(self, absolute: bool = False) -> tuple[float, Dim]:
        """absolute=True: a reading (20 °C = 293.15 K); otherwise a difference (1 °C = 1 K)."""
        line = self.tok.line
        neg = self.at("-") and self.next()
        v = -self.number() if neg else self.number()
        if self.tok.kind == "ident" and (u := parse_unit(self.tok.text)):
            unit = self.next().text
            offset = ABSOLUTE_OFFSET.get(unit, 0.0) if absolute else 0.0
            return v * u[0] + offset, u[1]
        return v, ONE  # dimensionless; callers check it against what they expect

    def signed(self) -> float:
        neg = self.at("-") and self.next()
        return -self.number() if neg else self.number()

    def percent(self) -> float:
        v = self.number()
        self.expect("%")
        return v / 100

    # grammar
    def program(self) -> Program:
        rigs, claims = {}, {}
        while self.tok.kind != "eof":
            if self.at("rig"):
                r = self.rig()
                rigs[r.name] = r
            elif self.at("claim") or self.at("experiment"):
                c = self.claim()
                claims[c.name] = c
            else:
                raise SimulError(f"expected 'rig', 'claim' or 'experiment', got {self.tok.text!r}", self.tok.line)
        return Program(rigs, claims, self.src)

    def typed(self, want: Dim, what: str, absolute: bool = False) -> float:
        line = self.tok.line
        v, d = self.qty(absolute)
        if d != want:
            raise SimulError(f"{what} must be in units of {want}, got {d}", line)
        return v

    def accuracy(self, want: Dim) -> Accuracy:
        self.expect("accuracy")
        rel = self.percent()
        self.expect("+")
        return Accuracy(rel, self.typed(want, "accuracy"))

    def range_(self, want: Dim) -> tuple[float, float]:
        self.expect("range")
        lo = self.typed(want, "range", absolute=True)
        self.expect("..")
        return lo, self.typed(want, "range", absolute=True)

    def rig(self) -> Rig:
        self.expect("rig")
        rig = Rig(self.ident())
        self.expect("{")
        while not self.at("}"):
            line = self.tok.line
            word = self.ident()
            if word == "device":
                rig.device = self.ident()
                continue
            if word == "monitor":
                rig.monitors += self.names()
                continue
            name = self.ident()
            if name in rig.sources or name in rig.sensors or name in rig.parts:
                raise SimulError(f"duplicate name {name!r}", line)
            self.expect(":")
            kind = self.ident()
            if kind not in KIND_DIM:
                raise SimulError(f"unknown kind {kind!r} (known: {', '.join(KIND_DIM)})", line)
            want = KIND_DIM[kind]
            if word == "source":
                nodes = self.between()
                lo, hi = self.range_(want)
                rig.sources[name] = Source(name, kind, lo, hi, self.accuracy(want), nodes)
            elif word == "sensor":
                where = self.ident()
                if where not in ("across", "through", "at", "ambient"):
                    raise SimulError("sensor needs 'across X', 'through X', 'at X' or 'ambient'", line)
                target = None if where == "ambient" else self.ident()
                lo, hi = self.range_(want)
                rig.sensors[name] = Sensor(name, kind, where, target, lo, hi, self.accuracy(want))
            elif word == "part":
                nominal = self.typed(want, f"value of {name}")
                self.expect("±")
                tol = self.percent()
                part = rig.parts[name] = Part(name, kind, nominal, tol, self.between())
                if self.at("material"):
                    self.next()
                    part.material = self.ident()
            else:
                raise SimulError(f"unknown rig entry {word!r}", line)
        self.expect("}")
        return rig

    def names(self) -> list[str]:
        out = [self.ident()]
        while self.at(","):
            self.next()
            out.append(self.ident())
        return out

    def law(self) -> Law:
        line = self.tok.line
        lhs = self.expr()
        self.expect("==")
        rhs = self.expr()
        slack = 0.0
        if self.at("within"):
            self.next()
            slack = self.percent()
        return Law(lhs, rhs, slack, line)

    def unit_expr(self) -> tuple[float, Dim, str]:
        """`1/K`, `T/A`, `V*s`, `Ohm` ... used to declare a param's unit."""
        start, scale, dim = self.i, 1.0, ONE
        op = "*"
        while True:
            if self.tok.kind == "num" and self.tok.text == "1":
                self.next()
                s, d = 1.0, ONE
            else:
                line = self.tok.line
                u = parse_unit(self.ident())
                if not u:
                    raise SimulError(f"unknown unit {self.toks[self.i - 1].text!r}", line)
                s, d = u
            if self.at("^"):
                self.next()
                neg = self.at("-") and self.next()
                k = int(self.number()) * (-1 if neg else 1)
                s, d = s**k, d**k
            scale, dim = (scale * s, dim * d) if op == "*" else (scale / s, dim / d)
            if not (self.at("*") or self.at("/")):
                return scale, dim, "".join(t.text for t in self.toks[start:self.i])
            op = self.next().text

    def between(self) -> tuple[str, str]:
        self.expect("between")
        return self.ident(), self.ident()

    def claim(self) -> Claim:
        kw = self.next()
        name = self.ident()
        self.expect("on")
        c = Claim(name, self.ident(), line=kw.line, experiment=kw.text == "experiment")
        self.expect("{")
        while not self.at("}"):
            line = self.tok.line
            word = self.ident()
            if word in ("sweep", "step"):
                if c.procedure:
                    raise SimulError("a claim has exactly one procedure (sweep or step)", line)
                src = self.ident()
                self.expect("from")
                lo, dim = self.qty(absolute=True)
                self.expect("to")
                hi, dim2 = self.qty(absolute=True)
                if dim != dim2:
                    raise SimulError(f"{word} from {dim} to {dim2}: units differ", line)
                if word == "sweep":
                    self.expect("steps")
                    c.procedure = Sweep(src, lo, hi, int(self.number()), dim=dim)
                    if self.at("settle"):
                        self.next()
                        c.procedure.settle = self.typed(SECOND, "settle time")
                else:
                    c.procedure = Step(src, lo, hi, dim=dim)
            elif word == "observe":
                if not isinstance(c.procedure, Sweep):
                    raise SimulError("'observe' belongs after a 'sweep'", line)
                c.procedure.observe += self.names()
            elif word == "sample":
                if not isinstance(c.procedure, Step):
                    raise SimulError("'sample' belongs after a 'step'", line)
                c.procedure.sample += self.names()
                self.expect("every")
                c.procedure.dt = self.typed(SECOND, "sample interval")
                self.expect("for")
                c.procedure.duration = self.typed(SECOND, "duration")
            elif word == "param":
                pname = self.ident()
                self.expect(":")
                scale, dim, unit = self.unit_expr()
                self.expect("in")
                lo = self.signed()
                self.expect("..")
                c.params[pname] = Param(pname, dim, lo * scale, self.signed() * scale, unit)
            elif word == "law":
                c.law = self.law()
            elif word == "against":
                c.against.append(self.law())
            else:
                raise SimulError(f"unknown claim entry {word!r}", line)
        self.expect("}")
        return c

    # expressions: sum > product > unary > power > atom
    def expr(self) -> Expr:
        e = self.term()
        while self.at("+") or self.at("-"):
            op = self.next().text
            e = Bin(op, e, self.term())
        return e

    def term(self) -> Expr:
        e = self.unary()
        while self.at("*") or self.at("/"):
            op = self.next().text
            e = Bin(op, e, self.unary())
        return e

    def unary(self) -> Expr:
        if self.at("-"):
            self.next()
            return Neg(self.unary())
        return self.power()

    def power(self) -> Expr:
        base = self.atom()
        if self.at("^"):
            self.next()
            neg = self.at("-") and self.next()
            n = self.number()
            if n != int(n):
                raise SimulError("only integer exponents are supported", self.tok.line)
            return Pow(base, -int(n) if neg else int(n))
        return base

    def atom(self) -> Expr:
        t = self.tok
        if t.kind == "num":
            v = self.number()
            if self.tok.kind == "ident" and (u := parse_unit(self.tok.text)):
                if self.tok.text in ABSOLUTE_OFFSET:
                    raise SimulError("write temperatures in laws in K (°C is ambiguous: reading or difference?)", t.line)
                self.next()
                return Num(v * u[0], u[1])
            return Num(v)
        if t.kind == "ident":
            self.next()
            if self.at("("):
                self.next()
                arg = self.expr()
                self.expect(")")
                return Call(t.text, arg, t.line)
            return Var(t.text, t.line)
        if self.at("("):
            self.next()
            e = self.expr()
            self.expect(")")
            return e
        raise SimulError(f"unexpected {t.text or 'end of file'!r} in expression", t.line)


def parse(src: str) -> Program:
    return Parser(src).program()

"""Generate the instrument half of a rig file from what the device reports about itself.

Each hardware module stores its own descriptor and calibration (kind, placement,
range, accuracy, calibration date) and reports it over the `describe` op. The
generated text is the part of a rig that should never be typed by hand; the human
adds the netlist (`between` nodes for sources and parts) for the circuit under test.
"""
from __future__ import annotations

from .units import KIND_DIM, KELVIN, ONE, fmt


def q(value: float, kind: str, absolute: bool = False) -> str:
    dim = KIND_DIM[kind]
    if dim == ONE:
        return f"{value:.6g}"
    if dim == KELVIN and absolute:
        return f"{value - 273.15:.6g} °C"
    return fmt(value, dim)


def rig_text(name: str, device: str, channels: list[dict]) -> str:
    lines = [f"rig {name} {{", f"  device {device}"]
    for c in channels:
        rel, ab = c["accuracy"]
        rng = f"range {q(c['range'][0], c['kind'], True)} .. {q(c['range'][1], c['kind'], True)}"
        acc = f"accuracy {rel * 100:.3g} % + {q(ab, c['kind'])}"
        cal = f"   # module {c.get('module', '?')}, calibrated {c.get('calibrated') or 'never: datasheet figures'}"
        if c["role"] == "source":
            lines.append(f"  source {c['name']} : {c['kind']} between <+node> <-node> {rng} {acc}{cal}")
        else:
            where = "ambient" if c["where"] == "ambient" else f"{c['where']} {c['target'] or '<part>'}"
            lines.append(f"  sensor {c['name']} : {c['kind']} {where} {rng} {acc}{cal}")
    lines += ["  # parts of the circuit under test, e.g.:", "  # part R : resistor 1 kOhm ± 1 % between <node> <node>", "}"]
    return "\n".join(lines) + "\n"

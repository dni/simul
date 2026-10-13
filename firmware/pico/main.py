# Simul device firmware v0.2 for Raspberry Pi Pico (MicroPython).
# UNTESTED ON HARDWARE: a starting point that implements PROTOCOL.md.
#
# Every channel carries its own descriptor (kind, placement, range, accuracy,
# calibration). `describe` reports them so `simul describe` can write the
# instrument half of the rig file: the hardware describes itself.
import json
import sys
import time

from machine import ADC, PWM, Pin

VREF = 3.3


# ---------------------------------------------------------------- drivers

class GpioSource:
    """Digital pin: only 0 V or VREF. Fast edge, ideal for step experiments."""

    def __init__(self, pin):
        self.pin = Pin(pin, Pin.OUT, value=0)

    def set(self, v):
        if abs(v) > 1e-9 and abs(v - VREF) > 1e-9:
            raise ValueError("gpio source supports only 0 V and %g V" % VREF)
        self.pin.value(1 if v > VREF / 2 else 0)


class PwmSource:
    """PWM through an RC low-pass and an op-amp buffer. Slow, for sweeps."""

    def __init__(self, pin, freq=100_000):
        self.pwm = PWM(Pin(pin))
        self.pwm.freq(freq)
        self.pwm.duty_u16(0)

    def set(self, v):
        if not 0 <= v <= VREF:
            raise ValueError("out of range")
        self.pwm.duty_u16(int(v / VREF * 65535))


class AdcVoltage:
    """ADC pin (GP26..28) referenced to ground. gain/offset come from calibration."""

    def __init__(self, pin, gain=1.0, offset=0.0):
        self.adc = ADC(Pin(pin))
        self.gain, self.offset = gain, offset

    def read(self):
        return self.adc.read_u16() * VREF / 65535 * self.gain + self.offset


class AdcDiff:
    """Voltage between two ADC nodes, e.g. across a resistor above a shunt."""

    def __init__(self, hi, lo):
        self.hi, self.lo = hi, lo

    def read(self):
        return self.hi.read() - self.lo.read()


class ShuntCurrent:
    """Current through a known shunt resistor to ground."""

    def __init__(self, adc, ohms):
        self.adc, self.ohms = adc, ohms

    def read(self):
        return self.adc.read() / self.ohms


class HallField:
    """Ratiometric analog Hall sensor (e.g. DRV5055): out = VREF/2 + sensitivity * B."""

    def __init__(self, adc, volts_per_tesla, zero=VREF / 2):
        self.adc, self.sens, self.zero = adc, volts_per_tesla, zero

    def read(self):
        return (self.adc.read() - self.zero) / self.sens


class DS18B20:
    """1-Wire temperature sensor, ±0.5 K. Slow (750 ms per conversion): use as a monitor."""

    def __init__(self, pin, index=0):
        import ds18x20
        import onewire

        self.ds = ds18x20.DS18X20(onewire.OneWire(Pin(pin)))
        roms = self.ds.scan()
        if not roms:
            raise RuntimeError("no DS18B20 on pin %d" % pin)
        self.rom = roms[index]

    def read(self):
        self.ds.convert_temp()
        time.sleep_ms(750)
        return self.ds.read_temp(self.rom) + 273.15  # SI: kelvin


# ---------------------------------------------------------------- board configuration
# name: (driver, descriptor). Ranges and accuracy in SI units. Replace datasheet
# figures with calibrated ones and set "calibrated" to the calibration date.

DEVICE = "pico_rc_1"

SOURCES = {
    "V": (GpioSource(15), {"kind": "voltage", "range": [0, VREF], "accuracy": [0.02, 0.02],
                           "module": "gpio15", "calibrated": None}),
}
SENSORS = {
    "Vc": (AdcVoltage(26), {"kind": "voltage", "where": "across", "target": "C", "range": [0, VREF],
                            "accuracy": [0.01, 0.015], "module": "adc0", "calibrated": None}),
}

# Example additions (uncomment what is wired):
# SENSORS["B"] = (HallField(AdcVoltage(27), volts_per_tesla=100.0),     # DRV5055A4: 12.5 mV/mT at 3.3 V
#                 {"kind": "magnetic_field", "where": "at", "target": "L", "range": [-0.0165, 0.0165],
#                  "accuracy": [0.02, 20e-6], "module": "drv5055", "calibrated": None})
# SENSORS["Tw"] = (DS18B20(22), {"kind": "temperature", "where": "at", "target": "Rw",
#                  "range": [218.15, 398.15], "accuracy": [0, 0.5], "module": "ds18b20", "calibrated": None})


# ---------------------------------------------------------------- protocol

def handle(req):
    op = req["op"]
    if op == "hello":
        return {"device": DEVICE, "fw": "0.2", "channels": list(SOURCES) + list(SENSORS)}
    if op == "describe":
        chans = [dict(d, name=n, role="source") for n, (_, d) in SOURCES.items()]
        chans += [dict(d, name=n, role="sensor") for n, (_, d) in SENSORS.items()]
        return {"channels": chans}
    if op == "set":
        SOURCES[req["ch"]][0].set(req["value"])
        time.sleep_ms(max(20, req.get("settle_ms", 0)))
        return {}
    if op == "read":
        n = req.get("n", 16)
        return {"values": {c: sum(SENSORS[c][0].read() for _ in range(n)) / n for c in req["ch"]}}
    if op == "step":
        src, chans = SOURCES[req["ch"]][0], [SENSORS[c][0] for c in req["sample"]]
        dt_us, n = int(req["dt"] * 1e6), req["n"]
        src.set(req["lo"])
        time.sleep_ms(req.get("settle_ms", 100))
        ts, vals = [], [[] for _ in chans]
        src.set(req["hi"])
        t0 = time.ticks_us()
        for k in range(n):
            target = k * dt_us
            while time.ticks_diff(time.ticks_us(), t0) < target:
                pass
            ts.append(time.ticks_diff(time.ticks_us(), t0) * 1e-6)
            for i, ch in enumerate(chans):
                vals[i].append(ch.read())
            if ts[-1] * 1e6 - target > dt_us / 2:
                raise RuntimeError("cannot keep up with dt=%dus" % dt_us)
        return {"data": dict([("t", ts)] + [(c, v) for c, v in zip(req["sample"], vals)])}
    raise ValueError("unknown op " + op)


def main():
    while True:
        line = sys.stdin.readline()
        if not line.strip():
            continue
        try:
            reply = handle(json.loads(line))
            reply["ok"] = True
        except Exception as e:
            reply = {"ok": False, "error": str(e)}
        sys.stdout.write(json.dumps(reply) + "\n")


main()

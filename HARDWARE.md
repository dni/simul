# Hardware: modular, self-describing lab devices

Goal: an AI can iterate quickly on real physics without a human rewiring anything, and every
device can describe itself precisely enough that a verdict means something.

## Principles

1. **The description is part of the instrument.** Each module stores its own descriptor and
   calibration (kind, range, worst-case accuracy, calibration date, serial number) in a small
   I2C EEPROM. `describe` reports it, and `simul describe` writes the rig's instrument lines from
   it. Humans never type an accuracy figure.
2. **Accuracy is measured, not assumed.** A calibration module carries references (a 0.01 %
   resistor, a 2.5 V ±0.05 % voltage reference). Calibration runs are ordinary Simul claims on
   that reference rig. A module whose calibration claim is REFUTED is flagged before it can
   produce data.
3. **Refuse, never approximate.** A module that cannot do what is asked (voltage out of range,
   sample rate too high) answers with an error. Silent clipping or mistiming would make every
   verdict built on it worthless.
4. **Reconfigurable by software.** The circuit under test sits on a switch matrix, so the agent
   can change the circuit, not just the drive level. That is what makes iteration fast: one
   board, many experiments, no hands.
5. **Fast experiments first.** Electrical transients take milliseconds and thermal equilibria
   take minutes. The agent should learn most from fast runs and spend slow runs deliberately.

## Architecture

```
            USB (PROTOCOL.md)
  host ─────────────────────── hub: Raspberry Pi Pico 2 (12-bit ADC, ~500 kS/s, PIO for timing)
  (simul, agent)                 │  I2C bus on Qwiic / STEMMA QT connectors (plug-and-play)
                                 ├── source module      DAC + buffer amp (sweeps), GPIO/MOSFET edge (steps)
                                 ├── V/I module         INA226: 16-bit, shunt-based current + bus voltage
                                 ├── temperature module DS18B20 (±0.5 K) or MAX31865 + PT100 (±0.1 K)
                                 ├── field module       DRV5055 analog Hall (fast) or TMAG5273 3-axis (I2C)
                                 ├── environment module BME280: room temperature, humidity, pressure
                                 ├── precision ADC      ADS1115 16-bit for slow, precise readings
                                 ├── switch matrix      analog switches (signal) / relays (power) around the DUT
                                 └── calibration module reference resistor + voltage reference
  every module: 24C02 EEPROM with descriptor + calibration
```

The parts above are suggestions to start from, not a validated design. Check each datasheet's
accuracy specification against what your claims need before building.

## Build order

1. **Bench v0** (about 10 €): a Pico, an RC board and an RL board on a breadboard
   (`examples/rc.phy`, `examples/rl.phy`). Flash `firmware/pico/main.py` and run
   `simul describe serial:/dev/ttyACM0`, then `simul prove examples/rc.phy --backend serial:/dev/ttyACM0`.
2. **Bench v1**: add INA226, DS18B20 and BME280, then the copper-coil heating board
   (`examples/heat.phy`). This is the first rig where the agent can find something the description
   leaves out.
3. **Bench v2**: descriptor EEPROMs plus the calibration module. From here on, accuracy figures
   are measured.
4. **Bench v3**: the switch matrix. This needs a language extension: `switch S1 between a b`
   as a controllable element, and `toggle`/`configure` steps in experiments.

## What the simulator already models (so you can develop before building)

`SimBackend` solves the declared netlist (modified nodal analysis, the method SPICE uses) and
adds physics the rig file does not state: resistor self-heating with a temperature coefficient
per `material`, the field inside a coil, Earth's field, and a drifting room temperature. Its
instruments err within, never beyond, their declared accuracy. Real hardware will still surprise
it (contact resistance, ADC nonlinearity, interference); finding those surprises is what the real
bench is for.

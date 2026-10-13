# Simul: discover and certify physical laws on precisely described hardware

Simul is a language and a lab system for doing physics the way Lean does mathematics: claims
are checked by a small trusted kernel, not by whoever made them. Because physics has no axioms
to derive from, Simul replaces "proof" with a closed loop: an AI scientist *derives* laws from the
device, and the kernel *certifies* them on fresh measurements.

```
          ┌──────────── untrusted: proposes ────────────┐      ┌──── trusted: decides ────┐
 rig  ──▶ │ experiment ──▶ data ──▶ discover ──▶ claims │ ──▶  │ fresh run ──▶ kernel ──▶ │ ──▶ certificate
 (exact   │   (agent designs it)   (finds laws, names  │      │  CONFIRMED / REFUTED /   │     + notebook
 hardware │                         constants from rig) │      │  INCONCLUSIVE            │
 descr.)  └─────────────────────────────────────────────┘      └──────────────────────────┘
```

A certified statement reads: *on this device, as described, within the instruments' calibrated
error, every measurement fits this law and rules out the named rivals.* Every assumption is in
the source file, so the statement is exactly as strong as what it says.

## Prior work this builds on

- **BACON** (Langley, Simon, Bradshaw and Zytkow, 1970s–80s) rediscovered Kepler's third law and
  Ohm's law from data tables by searching for invariant combinations of variables.
- **Robot Scientist Adam** ([King et al., 2009](https://en.wikipedia.org/wiki/Ross_D._King)): Ross
  King, a Fellow of the Alan Turing Institute, built the first machine to autonomously discover
  scientific knowledge. It originates hypotheses, devises and *physically runs* experiments with lab
  robotics, interprets the results and repeats the cycle. Simul is that closed loop for physics
  benches.
- **Nobel Turing Challenge** ([Kitano, 2021](https://www.physics.ox.ac.uk/events/nobel-turing-challenge)),
  co-organized by King: AI systems doing autonomous science at Nobel level by 2050, judged by
  independent verification. Simul's kernel and certificates are that verification layer.
- **Distilling free-form natural laws from experimental data**
  ([Schmidt & Lipson, Science 2009](https://news.cornell.edu/node/272917)): symbolic regression
  recovered Hamiltonians and conservation laws from motion-tracked pendulums. Simul's discovery is
  much smaller in scope but differs in two ways: a model must fit *every* point within *declared
  worst-case instrument error* (not a goodness-of-fit score), and constants are explained from a
  formal description of the hardware.

## Quick start (no hardware needed: the simulator solves the declared netlist)

```sh
python3 -m simul discover examples/heat.phy heat_explore --verify 3   # derive laws, certify top 3 on fresh data
python3 -m simul prove examples/heat.phy        # naive Ohm REFUTED on a self-heating coil, R(T) CONFIRMED
python3 -m simul design examples/heat.phy ohm_thermal --pilot heat_explore   # cheapest decisive experiment
python3 -m simul bench                          # hidden-physics benchmark (scripted baseline)
python3 -m simul notebook                       # everything that ever ran, with who and why
python3 -m simul describe sim:examples/rl.phy:bench_rl   # device self-description -> rig skeleton
pip install anthropic && python3 -m simul agent examples/heat.phy --budget 12   # Claude runs the lab
python3 -m unittest discover -s tests
```

What discovery finds on the examples, from raw data alone:

| rig | law found | constants |
|---|---|---|
| RC | `Vc == V + (V_0 - V) * exp(-t / (C * R))` | τ named as R·C from the rig |
| RL | `Vr == R/(R+Rc)*V + R/(R+Rc)*(V_0 - V)*exp(-t/(L/R))` | all from the rig |
| RL, field | `B == c1 + c2 * Vr` | c1 ≈ 40 µT: Earth's field, **not in the description** |
| copper coil | `Tw == Ta + c2 * Ir * Vr` | c2 ≈ 25 K/W: thermal resistance, **new constant** |
| copper coil | `Ir == V/Rw + c2 * V * (Tw - 293.15 K)` | c2 = −α/Rw: copper's temperature coefficient |

Constants the rig cannot explain are the interesting output: either new physics, or something
the hardware description is missing.

## Choosing the experiment that decides

Adam chose experiments to separate its hypotheses. `simul design` does that for any claim with
`against` rivals, without touching hardware. Two laws can both survive an experiment if some
data could fit both, so an experiment's *separation* is

```
S = min over witnesses (each law its own, as the kernel allows)  max over rows  |f − g| / (U_f + U_g)
```

S > 1 means every possible outcome refutes at least one of the two laws. Readings an experiment
cannot set (a temperature, a current) come from a pilot run in the notebook and are never
extrapolated.

| claim | what design says |
|---|---|
| copper coil: R(T) vs plain Ohm | 2 points incl. 6 V decide it: 4 min instead of 24 |
| RC: charging vs no capacitance | one sample right after the edge decides it |
| RL: τ = L/(R+Rc) vs τ = L/R | **no experiment on this rig can**: L's ±10 % lets one impersonate the other. It names the upgrades that would (measure L, R, Rc on a calibration rig; better source and ADC) |

The last row is the useful kind of answer: it says what to build next. The search is untrusted;
the kernel's verdict on the real run is what counts.

## Benchmark: how much does the AI scientist find?

`simul bench` runs a solver against rigs whose simulator has physics the rig never states, and
the solver sees only the rig blocks (no example claims, no comments). Scoring needs no judge: a
certified claim *detects* a hidden effect if, on a fresh simulator with the same seed, it is
CONFIRMED with the effect on and REFUTED with it off. A sentence in a report scores nothing.

| scenario | hidden | baseline |
|---|---|---|
| rl_background | 150 µT background field | ✓ |
| heat_joule | self-heating | ✓ |
| rc_leak | capacitor leaking through 100 kΩ | ✓ |
| ohm_offset | source outputs 60 mV extra | ✓ |
| ohm_nonlinear | current sensor up to 16 % high (quadratic) | ✓ |
| rc_hum | 150 mV of 50 Hz pickup | ✗ (pushes readings out of range; needs a sine law) |
| rc_leak_hum | leak + hum | 1/2 (the leak's constant absorbs the hum) |
| rl_field_sensor | background + nonlinear Hall sensor | 1/2 |
| heat_offset | self-heating + source offset | 2/2 |

The scripted baseline (explore with the human-written procedures, discover, certify every
candidate) scores **9/12** with a budget of 12 runs. `simul bench --solver agent` puts Claude in
its place. Every scenario is solvable: for instance, a claim with `a * sin(314.159 Hz * t + phi)`
stepping to 2.5 V over 40 ms detects the hum.

## Describing hardware exactly

A rig is the full, checkable description of one physical device: the netlist (every part and
source sits `between` two nodes), every instrument with its placement, range and worst-case
accuracy, part tolerances and materials, monitored environment, and the `device` it describes.

```
rig bench_rl {
  source V  : voltage between in gnd range 0 V .. 3.3 V   accuracy 2 % + 20 mV
  part   L  : inductor 100 mH ± 10 % between in wind
  part   Rc : resistor 4 Ohm   ± 25 % between wind mid material copper   # the coil's own wire
  part   R  : resistor 100 Ohm ± 1 %  between mid gnd
  sensor Vr : voltage across R range 0 V .. 3.3 V        accuracy 1 % + 15 mV
  sensor B  : magnetic_field at L range -20 mT .. 20 mT  accuracy 2 % + 20 uT
}
```

Before any hardware runs, the checker enforces that every quantity's unit matches its kind, that
the netlist is closed (no dangling nodes, no shorted elements), that laws are dimensionally
consistent (`exp()` of anything with units is rejected), that experiments stay inside source
ranges, and that the connected board is the one the rig describes. The instrument lines should
come from the device itself (`describe`), with calibrated accuracy. See [HARDWARE.md](HARDWARE.md).

## The language

```
claim ohm_thermal on bench_heat {
  param alpha : 1/K in 0.0035 .. 0.0043        # "some α in this range exists": narrower = stronger
  sweep V from 0.5 V to 6 V steps 12 settle 120 s
  observe Vr, Ir
  law Vr == Rw * (1 + alpha * (Tw - 293.15 K)) * Ir
  against Vr == Rw * Ir                        # the rival this experiment must rule out
}

experiment heat_explore on bench_heat {        # no law: data for discovery
  sweep V from 0.5 V to 6 V steps 12 settle 120 s
  observe Vr, Ir, Tw
}
```

What it means for `A == B within s %` to hold: there exist values for the mentioned parts and
params, inside their declared ranges, such that for every data row
`|A − B| ≤ Σᵢ |∂(A−B)/∂xᵢ|·uᵢ + s·max(|A|,|B|)`, where uᵢ is instrument i's declared worst-case
error. Those values are the **witness**: finding it is search, while checking it is one cheap
pass over the data, the same split as proof search vs. proof checking in Lean. Certificates
(source, hash, raw data, device, witness, verdict) are re-checked offline by `simul recheck`.

## The AI scientist

`simul agent` gives Claude (`claude-opus-5-5`) the same tools a human has: `describe_hardware`,
`check` and `design` (free), `run` and `discover` (one budget unit each), and `notebook`. It is told to explore,
never trust a fit on the data it came from, make claims strong, chase unexplained constants, and
say which instrument upgrade would decide what it could not. The trust boundary is built into
`Lab`: the agent cannot declare or alter rigs, every run is logged, and only the kernel issues
verdicts. Refusals fall back to another model server-side (`fallbacks: "default"`).

## Layout

| file | role |
|---|---|
| `simul/lang.py`, `check.py`, `units.py` | language, static checks, units (SI + °C) |
| `simul/kernel.py` | experiments, witness search and check, verdicts, certificates |
| `simul/discover.py` | law discovery and explanation from the rig |
| `simul/design.py` | experiment design: separation of rival laws, upgrade advice |
| `simul/bench.py` | hidden-physics scenarios, counterfactual scoring, baseline solver |
| `simul/backends.py` | netlist simulator (MNA plus switchable hidden effects), serial driver |
| `simul/lab.py`, `notebook.py` | sessions, one device connection per rig, append-only notebook |
| `simul/agent.py` | Claude as experimentalist |
| `firmware/pico/main.py`, `PROTOCOL.md` | device side (untested on hardware) |

## Known gaps / roadmap

- **REFUTED is not yet sound.** It means "grid search found no witness". Interval branch-and-bound
  over the parameter box would make it a proof.
- **Discovery's model space is small:** two-term linear-in-coefficient models and single
  exponentials. Laws may use `sin`/`cos`, but discovery does not propose them yet. Next steps are
  sums of a transient and a sinusoid (hum), products of exponentials (RLC), power laws with
  fitted exponents, and agent-proposed model families.
- **Run the benchmark with the agent** and compare to the baseline's 9/12; then use it to tune
  the agent's prompt and tools. Add a run-time cost (settle times) to the budget, not just a count.
- **Design is greedy about cost** (steps × settle, or duration) and does not model device limits
  such as the maximum sample rate.
- **Calibration certificates** (claims on a reference rig) should produce the accuracy figures.
- **Switch matrices** (`switch` elements) so the agent can change the circuit, not just the drive.
- **Timing uncertainty** is not modelled; `t` is treated as exact.
- **`d/dt` in laws**, to state induction differentially (`V_L == L * d(I)/dt`).
- **Derivations:** certified laws as premises for new claims. This is where it becomes a proof
  language rather than a test language.

import json
import pathlib
import tempfile
import unittest

from simul import SimulError, parse
from simul import notebook
from simul.backends import Circuit, SimBackend
from simul.bench import detects
from simul.check import check_claim, check_program
from simul.design import design
from simul.discover import discover
from simul.kernel import prove, recheck, run_experiment
from simul.lab import Lab

EXAMPLES = pathlib.Path(__file__).parent.parent / "examples"

RIG = """
rig b {
  source V  : voltage between top gnd range 0 V .. 5 V accuracy 1 % + 10 mV
  part   R  : resistor 1 kOhm ± 5 % between top gnd
  sensor Vr : voltage across R range 0 V .. 5 V accuracy 1 % + 10 mV
  sensor Ir : current through R range 0 A .. 10 mA accuracy 1 % + 10 uA
}
"""


def claim(body: str, rig: str = RIG) -> str:
    return rig + "claim c on b {\n" + body + "\n}"


def run_claim(src, name, seed=0, truth=None):
    prog = parse(src)
    rig = prog.rigs[prog.claims[name].rig]
    return prove(prog, name, SimBackend(rig, seed=seed, truth=truth))


class StaticChecks(unittest.TestCase):
    def rejects(self, src, fragment):
        with self.assertRaises(SimulError) as cm:
            check_program(parse(src))
        self.assertIn(fragment, str(cm.exception))

    def test_examples_are_well_formed(self):
        for f in EXAMPLES.glob("*.phy"):
            with self.subTest(f=f.name):
                check_program(parse(f.read_text()))

    def test_dimension_mismatch(self):
        self.rejects(claim("sweep V from 1 V to 4 V steps 5\nobserve Vr, Ir\nlaw Vr == Ir"), "dimensionally inconsistent")

    def test_exp_needs_dimensionless(self):
        self.rejects(claim("sweep V from 1 V to 4 V steps 5\nobserve Vr\nlaw Vr == V * exp(R)"), "dimensionless")

    def test_out_of_source_range(self):
        self.rejects(claim("sweep V from 1 V to 9 V steps 5\nobserve Vr\nlaw Vr == V"), "outside the range")

    def test_procedure_unit_must_match_source(self):
        self.rejects(claim("sweep V from 1 mA to 4 mA steps 5\nobserve Vr\nlaw Vr == V"), "voltage source")

    def test_unobserved_sensor(self):
        self.rejects(claim("sweep V from 1 V to 4 V steps 5\nobserve Vr\nlaw Vr == R * Ir"), "unobserved")

    def test_time_only_in_step(self):
        self.rejects(claim("sweep V from 1 V to 4 V steps 5\nobserve Vr\nlaw Vr == V * t / 1 s"), "unknown or unobserved name 't'")

    def test_rig_quantity_units_checked(self):
        with self.assertRaises(SimulError) as cm:
            parse(RIG.replace("range 0 A .. 10 mA", "range 0 V .. 10 mV"))
        self.assertIn("units of A", str(cm.exception))

    def test_open_node_rejected(self):
        self.rejects(RIG.replace("between top gnd\n", "between top mid\n"), "only one connection")

    def test_param_range_and_dims(self):
        ok = claim("param g : 1/Ohm in 0.0009 .. 0.0011\nsweep V from 1 V to 4 V steps 5\nobserve Vr, Ir\nlaw Ir == g * Vr")
        check_program(parse(ok))
        self.rejects(ok.replace("1/Ohm", "Ohm"), "dimensionally inconsistent")

    def test_celsius_is_absolute_in_ranges(self):
        rig = parse(RIG.replace("}", "  sensor Ta : temperature ambient range 0 °C .. 50 °C accuracy 0 % + 1 K\n}")).rigs["b"]
        self.assertAlmostEqual(rig.sensors["Ta"].lo, 273.15)
        self.assertAlmostEqual(rig.sensors["Ta"].acc.abs, 1.0)


class Circuits(unittest.TestCase):
    def test_divider(self):
        rig = parse("""rig d {
          source V : voltage between a gnd range 0 V .. 10 V accuracy 0 % + 0 V
          part R1 : resistor 1 kOhm ± 0 % between a m
          part R2 : resistor 3 kOhm ± 0 % between m gnd
        }""").rigs["d"]
        sol = Circuit(rig, {"R1": 1000.0, "R2": 3000.0}).run({"V": 8.0}, None, None)
        self.assertAlmostEqual(sol["v"]["R2"], 6.0, places=6)
        self.assertAlmostEqual(sol["i"]["V"], 0.002, places=9)


class Verdicts(unittest.TestCase):
    def test_true_claims_confirmed_across_seeds(self):
        for f, name in [("ohm", "ohms_law"), ("rc", "rc_charge"), ("rl", "self_induction"), ("heat", "ohm_thermal")]:
            src = (EXAMPLES / f"{f}.phy").read_text()
            for seed in range(3):
                with self.subTest(claim=name, seed=seed):
                    self.assertEqual(run_claim(src, name, seed)["verdict"], "CONFIRMED")

    def test_false_claims_refuted(self):
        for f, name in [("ohm", "constant_drop"), ("rc", "rc_linear"), ("heat", "ohm_naive")]:
            with self.subTest(claim=name):
                self.assertEqual(run_claim((EXAMPLES / f"{f}.phy").read_text(), name)["verdict"], "REFUTED")

    def test_out_of_spec_part_refutes(self):
        src = (EXAMPLES / "ohm.phy").read_text()
        self.assertEqual(run_claim(src, "ohms_law", truth={"R": 1200.0})["verdict"], "REFUTED")

    def test_indistinguishable_rival_is_inconclusive(self):
        src = claim("sweep V from 1 V to 4 V steps 5\nobserve Vr, Ir\nlaw Vr == R * Ir\nagainst Vr == V")
        self.assertEqual(run_claim(src, "c")["verdict"], "INCONCLUSIVE")

    def test_wrong_device_rejected(self):
        prog = parse((EXAMPLES / "rc.phy").read_text())
        rig = prog.rigs["bench_rc"]
        dev = SimBackend(rig)
        dev.hello = lambda: {"device": "some_other_board", "channels": ["V", "Vc"]}
        with self.assertRaises(SimulError):
            prove(prog, "rc_charge", dev)


class Discovery(unittest.TestCase):
    def discover(self, f, name):
        prog = parse((EXAMPLES / f).read_text())
        c = prog.claims[name]
        rig = check_claim(c, prog)
        return discover(c, rig, run_experiment(c, rig, SimBackend(rig, seed=1)))

    def test_rc_law_derived_from_data(self):
        laws = [r["law"] for r in self.discover("rc.phy", "rc_charge")]
        self.assertIn("Vc == V + (V_0 - V) * exp(-t / (C * R))", laws)

    def test_ohms_law_derived_from_data(self):
        laws = [r["law"] for r in self.discover("ohm.phy", "explore")]
        self.assertIn("Vr == R * Ir", laws)

    def test_joule_heating_found_as_new_constant(self):
        found = [r for r in self.discover("heat.phy", "heat_explore") if r["law"] == "Tw == Ta + c2 * Ir * Vr"]
        self.assertTrue(found)
        self.assertEqual(found[0]["unexplained"], 1)  # thermal resistance: not in the rig description

    def test_discovered_claim_certifies_on_fresh_data(self):
        with tempfile.TemporaryDirectory() as d:
            lab = Lab([str(EXAMPLES / "heat.phy")], notebook_path=f"{d}/nb.jsonl")
            _, cands = lab.discover("", "heat_explore")
            text = next(c["claim"] for c in cands if c["law"] == "Tw == Ta + c2 * Ir * Vr")
            cert = lab.run(text, text.split()[1])
            self.assertEqual(cert["verdict"], "CONFIRMED")
            self.assertEqual(cert["entry"], 2)


RL_TAU = """
claim tau_rivals on bench_rl {
  step V from 0 V to 3.3 V
  sample Vr every 20 us for 6 ms
  law Vr == R / (R + Rc) * (V + (V_0 - V) * exp(-t * (R + Rc) / L))
  against Vr == R / (R + Rc) * (V + (V_0 - V) * exp(-t * R / L))
}
"""


class Design(unittest.TestCase):
    def test_cheapest_decisive_experiment_is_found_and_confirms(self):
        with tempfile.TemporaryDirectory() as d:
            lab = Lab([str(EXAMPLES / "heat.phy")], notebook_path=f"{d}/nb.jsonl")
            lab.run("", "heat_explore")  # pilot: the laws mention Tw and Ir, which no experiment sets
            plan = lab.design("", "ohm_thermal")
            self.assertTrue(plan["decisive"])
            self.assertIn("steps 2 ", plan["procedure"])  # two points instead of twelve
            self.assertIn("to 6 V", plan["procedure"])  # where heating makes the laws differ most
            cert = lab.run(plan["claim_text"], "ohm_thermal_d")
            self.assertEqual(cert["verdict"], "CONFIRMED")

    def test_undecidable_rivals_get_upgrade_advice(self):
        src = (EXAMPLES / "rl.phy").read_text() + RL_TAU
        prog = parse(src)
        plan = design(prog.claims["tau_rivals"], prog.rigs["bench_rl"], src)
        self.assertFalse(plan["decisive"])  # L's ±10 % lets L/R impersonate L/(R+Rc)
        self.assertIn("L", plan["upgrades"][0]["item"].split("+"))

    def test_needs_pilot_for_unset_readings(self):
        src = (EXAMPLES / "heat.phy").read_text()
        prog = parse(src)
        with self.assertRaises(SimulError):
            design(prog.claims["ohm_thermal"], prog.rigs["bench_heat"], src)


LEAK = """claim leak on bench_rc {
  param Rl : kOhm in 85 .. 115
  step V from 0 V to 3.3 V
  sample Vc every 20 us for 5 ms
  law Vc == Rl / (R + Rl) * (V + (V_0 - V) * exp(-t * (R + Rl) / (R * Rl * C)))
}"""


class HiddenPhysics(unittest.TestCase):
    def test_blind_lab_shows_only_rig_blocks(self):
        lab = Lab([str(EXAMPLES / "heat.phy")], blind=True)
        self.assertNotIn("claim", lab.rigs())
        self.assertNotIn("#", lab.rigs())
        self.assertEqual(lab.rigs().count("rig "), 1)

    def test_counterfactual_scoring(self):
        fx = {"leaky_capacitor": 100e3}
        with tempfile.TemporaryDirectory() as d:
            lab = Lab([str(EXAMPLES / "rc.phy")], notebook_path=f"{d}/nb.jsonl", effects=fx, blind=True)
            self.assertEqual(lab.run(LEAK, "leak")["verdict"], "CONFIRMED")
            textbook = (EXAMPLES / "rc.phy").read_text().split("claim rc_charge", 1)[1].split("}", 1)[0]
            lab.run("claim rc_charge" + textbook.replace("against Vc == V", "") + "}", "rc_charge")
            leak, plain = notebook.entries(lab.notebook)
            self.assertTrue(detects(leak, fx, "leaky_capacitor", 0))  # true only because of the leak
            self.assertFalse(detects(plain, fx, "leaky_capacitor", 0))

    def test_unknown_effect_rejected(self):
        with self.assertRaises(ValueError):
            SimBackend(parse(RIG).rigs["b"], effects={"gremlins": 1})

    def test_effects_off_means_ideal_physics(self):
        prog = parse((EXAMPLES / "heat.phy").read_text())
        cert = prove(prog, "ohm_naive", SimBackend(prog.rigs["bench_heat"], effects={}))
        self.assertEqual(cert["verdict"], "CONFIRMED")  # no self-heating: plain Ohm's law holds


class LabBoundary(unittest.TestCase):
    def test_rigs_cannot_be_declared_by_callers(self):
        with tempfile.TemporaryDirectory() as d:
            lab = Lab([str(EXAMPLES / "ohm.phy")], notebook_path=f"{d}/nb.jsonl")
            with self.assertRaises(SimulError):
                lab.check(RIG)


class Certificates(unittest.TestCase):
    def setUp(self):
        prog = parse((EXAMPLES / "ohm.phy").read_text())
        self.cert = json.loads(json.dumps(prove(prog, "ohms_law", SimBackend(prog.rigs["bench_ohm"]))))

    def test_roundtrip(self):
        self.assertEqual(recheck(self.cert), (True, "CONFIRMED"))

    def test_tampered_data_fails(self):
        self.cert["data"][3]["Vr"] *= 1.2
        self.assertFalse(recheck(self.cert)[0])

    def test_tampered_witness_fails(self):
        self.cert["law"]["witness"]["R"] = 2000.0
        self.assertFalse(recheck(self.cert)[0])

    def test_tampered_source_fails(self):
        self.cert["source"] = self.cert["source"].replace("± 5 %", "± 50 %")
        self.assertEqual(recheck(self.cert), (False, "source hash mismatch"))


if __name__ == "__main__":
    unittest.main()

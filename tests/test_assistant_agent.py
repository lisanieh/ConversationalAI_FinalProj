"""Unit tests for the assistant agent. Run:  python -m unittest discover tests"""

import json
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np

from assistant_agent import (Heard, MockASRChannel, RepairOutcomeModel, RuleExtractor,
                             SlotErrorEstimator, TextChannel, U2RAgent, make_policy,
                             score_dialogue, slot_features)
from assistant_agent.estimator import auroc
from assistant_agent.policies import U2R
from assistant_agent.repair import decode_confirm, decode_spelled, is_cannot_type
from assistant_agent.scoring import classify_error
from assistant_agent.values import canonical, parse_time, values_match
from robot_customer import GoalBuilder, RobotCustomer, ScriptedBrain, run_dialogue, spell_out
from robot_customer.prompts import CANNOT_TYPE_LINE
from robot_customer.router import VALID_ACTIONS


class TestValues(unittest.TestCase):
    def test_time(self):
        for s, want in [("19:30", "19:30"), ("7:30 pm", "19:30"), ("seven thirty p m", "19:30"),
                        ("half past seven", "19:30"), ("noon", "12:00"), ("11:30", "11:30"),
                        ("seven oh five pm", "19:05"), ("1830", "18:30"), ("eight", "20:00")]:
            self.assertEqual(parse_time(s), want, s)

    def test_formats(self):
        self.assertEqual(canonical("phone", "four oh four, five five five, one two three four"), "404 555 1234")
        self.assertEqual(canonical("phone", "(404) 555-1234"), "404 555 1234")
        self.assertEqual(canonical("email", "varun dot k at gmail dot com"), "varun.k@gmail.com")
        self.assertEqual(canonical("people", "three"), "3")
        self.assertEqual(canonical("area", "the city center"), "centre")
        self.assertEqual(canonical("type", "a guest house"), "guesthouse")
        self.assertEqual(canonical("pricerange", "no preference"), "dontcare")
        self.assertTrue(values_match("name", "varun kamath", "Varun Kamath"))
        self.assertFalse(values_match("area", "nort", "north"))      # format only, never a fix

    def test_error_types(self):
        self.assertEqual(classify_error("name", "Varun Kamath", "varun kamath"), "correct")
        self.assertEqual(classify_error("name", "Varun Kamath", "varun"), "deletion")
        self.assertEqual(classify_error("name", "Varun Kamath", "varun ka math"), "substitution")
        self.assertEqual(classify_error("email", "vk@gmail.com", "vk@gmail.comm"), "insertion")


class TestRepairDecoding(unittest.TestCase):
    def test_spell_round_trip(self):
        """Whatever the robot customer spells, the agent must decode back exactly."""
        b = GoalBuilder()
        for seed in range(300):
            g = b.build(seed)
            for slot, v in g.slots.items():
                got = decode_spelled(spell_out(v, slot), slot)
                self.assertTrue(values_match(slot, got, v), f"{slot}: {v!r} -> {spell_out(v, slot)!r} -> {got!r}")

    def test_spell_odd_names_and_whisper_styles(self):
        for v in ["Sean O'Connor", "Anna Smith-Jones", "Li Na"]:
            self.assertTrue(values_match("name", decode_spelled(spell_out(v, "name"), "name"), v), v)
        self.assertEqual(decode_spelled("V-A-R-U-N, K-A-M-A-T-H", "name"), "varun kamath")
        self.assertEqual(decode_spelled("v a r u n dot k at gmail dot com", "email"), "varun.k@gmail.com")
        self.assertEqual(decode_spelled("varun.k@gmail.com", "email"), "varun.k@gmail.com")
        self.assertEqual(decode_spelled("4 0 4, 5 5 5, 1 2 3 4", "phone"), "404 555 1234")

    def test_confirm(self):
        r = RuleExtractor()
        self.assertEqual(decode_confirm("Yes, that's right.", "day", "friday", r), ("yes", "friday"))
        self.assertEqual(decode_confirm("No, it's Saturday.", "day", "friday", r), ("no", "saturday"))
        self.assertEqual(decode_confirm("No, it's 404 555 1234.", "phone", "404 555 1233", r),
                         ("no", "404 555 1234"))
        self.assertEqual(decode_confirm("No, that's wrong.", "name", "varun", r), ("no", None))

    def test_cannot_type(self):
        self.assertTrue(is_cannot_type(CANNOT_TYPE_LINE))
        self.assertFalse(is_cannot_type("A guesthouse, please."))


class TestRuleExtractor(unittest.TestCase):
    def test_openers_and_answers(self):
        r = RuleExtractor()
        self.assertEqual(r.extract("Hi, I'm looking for a guesthouse in the north."),
                         ("hotel", {"area": "north", "type": "guesthouse"}))
        dom, v = r.extract("Hi, I'd like to book a table at a thai place in the centre.")
        self.assertEqual((dom, v), ("restaurant", {"area": "centre", "food": "thai"}))
        self.assertEqual(r.extract("It's Varun Kamath.", asked_slot="name")[1], {"name": "Varun Kamath"})
        self.assertEqual(r.extract("For 3 people.", asked_slot="people")[1], {"people": "3"})
        self.assertEqual(r.extract("At 19:30.", asked_slot="time")[1], {"time": "19:30"})
        self.assertEqual(r.extract("My number is 404 555 1234.", asked_slot="phone")[1],
                         {"phone": "404 555 1234"})
        self.assertEqual(r.extract("I don't have a preference.", asked_slot="area")[1], {"area": "dontcare"})


class TestLLMExtractor(unittest.TestCase):
    class FakeLLM:
        def __init__(self, reply):
            self.reply, self.seen = reply, None

        def chat(self, messages, max_tokens=200):
            self.seen = messages
            return self.reply

    def test_parses_and_filters(self):
        from assistant_agent import LLMExtractor
        llm = self.FakeLLM('Sure! ```json\n{"domain": "Hotel", "slots": {"area": "nort", '
                           '"stars": "4", "people": null}}\n```')
        dom, vals = LLMExtractor(llm).extract("a place in the nort", asked_slot="area")
        self.assertEqual((dom, vals), ("hotel", {"area": "nort"}))     # kept as heard; unknown slot dropped
        self.assertIn("The assistant just asked about: area", llm.seen[0]["content"])

    def test_falls_back_to_rules_on_garbage(self):
        from assistant_agent import LLMExtractor
        dom, vals = LLMExtractor(self.FakeLLM("I think north")).extract("In the north, please.", "area")
        self.assertEqual(vals, {"area": "north"})


class TestPolicies(unittest.TestCase):
    TABLE = {"confirm|*": {"r": 0.5, "e": 0.02, "n_wrong": 100, "n_right": 100},
             "spell|*": {"r": 0.8, "e": 0.05, "n_wrong": 100, "n_right": 100},
             "type|*": {"r": 0.95, "e": 0.01, "n_wrong": 100, "n_right": 100}}

    def test_u2r_choices(self):
        u = U2R(RepairOutcomeModel(self.TABLE))
        for p, want in [(0.05, "accept"), (0.3, "confirm"), (0.9, "type")]:
            self.assertEqual(u.choose({"p_hat": p, "slot_type": "name"}), want)
        self.assertNotEqual(u.choose({"p_hat": 0.9, "slot_type": "name", "can_type": False}), "type")

    def test_p_star_is_the_crossing(self):
        u = U2R(RepairOutcomeModel(self.TABLE))
        p = u.p_star("name")
        c = u.expected_costs(p, "name")
        self.assertAlmostEqual(c["spell"], c["type"], places=9)

    def test_outcome_smoothing_and_pooling(self):
        rows = ([{"policy": "spell", "mode": "speech", "slot_type": "name", "true_value": "x",
                  "committed_value": "x", "first_correct": False, "committed_correct": True}] * 3 +
                [{"policy": "spell", "mode": "speech", "slot_type": "name", "true_value": "x",
                  "committed_value": "x", "first_correct": True, "committed_correct": True}] * 5)
        m = RepairOutcomeModel(min_n=20).fit(rows)
        self.assertAlmostEqual(m.table["spell|*"]["r"], 4 / 5)         # (3+1)/(3+2)
        self.assertAlmostEqual(m.table["spell|*"]["e"], 1 / 7)         # (0+1)/(5+2)
        self.assertEqual(m.get("spell", "name"), m.get("spell", "email"))  # small cell -> pooled
        self.assertEqual(m.get("accept", "name"), (0.0, 0.0))

    def test_all_specs_build(self):
        o = RepairOutcomeModel(self.TABLE)
        for spec in ["accept", "confirm", "spell", "type", "typed", "threshold:-0.5", "llm",
                     "u2r", "u2r:5", "u2r-notype"]:
            make_policy(spec, o)


class TestEstimator(unittest.TestCase):
    def _data(self, n, rng):
        rows, y = [], []
        for _ in range(n):
            wrong = rng.random() < 0.2
            rows.append({"min_logprob": rng.gauss(-2.0 if wrong else -0.3, 0.6),
                         "mean_logprob": rng.gauss(-1.0 if wrong else -0.1, 0.3),
                         "agreement": rng.uniform(0, 0.6) if wrong else rng.uniform(0.5, 1),
                         "no_speech_prob": rng.uniform(0, 0.1), "compression_ratio": rng.uniform(1, 1.5),
                         "snr_est": rng.gauss(10, 5), "slot_type": rng.choice(["name", "category"])})
            y.append(int(wrong))
        return rows, y

    def test_learns_calibrates_and_saves(self):
        rng = random.Random(0)
        tr, ytr = self._data(800, rng)
        cal, ycal = self._data(300, rng)
        te, yte = self._data(300, rng)
        est = SlotErrorEstimator().fit(tr, ytr).calibrate(cal, ycal, "temperature")
        self.assertGreater(auroc(yte, est.predict(te)), 0.9)
        self.assertGreater(est.coefficients()["agreement"], -100)
        with tempfile.TemporaryDirectory() as d:
            est.save(Path(d) / "e.json")
            est2 = SlotErrorEstimator.load(Path(d) / "e.json")
        np.testing.assert_allclose(est.predict(te), est2.predict(te))
        p0 = est.predict_one(None, "name")                   # no features -> slot-type prior
        self.assertTrue(0 < p0 < 1)
        est.calibrate(cal, ycal, "platt")
        self.assertGreater(auroc(yte, est.predict(te)), 0.9)

    def test_auroc_matches_pair_count(self):
        rng = np.random.default_rng(1)
        y, p = rng.integers(0, 2, 60), rng.integers(0, 5, 60) / 4
        pos, neg = p[y == 1], p[y == 0]
        brute = np.mean([(a > b) + 0.5 * (a == b) for a in pos for b in neg])
        self.assertAlmostEqual(auroc(y, p), brute)


def _run(seed, spec, channel=None, outcomes=None, estimator=None, cannot_type_rate=0.2):
    goal = GoalBuilder(cannot_type_rate=cannot_type_rate).build(seed)
    customer = RobotCustomer(goal, ScriptedBrain())
    agent = U2RAgent(make_policy(spec, outcomes), RuleExtractor(), estimator)
    channel = channel or TextChannel()
    run_dialogue(customer, lambda pkg: agent.step(channel(pkg) if pkg else None), max_turns=40)
    return customer, agent


class TestAgentDialogues(unittest.TestCase):
    def test_labels_are_valid_for_the_router(self):
        for spec in ["accept", "confirm", "spell", "type", "typed"]:
            customer, _ = _run(3, spec)
            for h in customer.history:
                if h["speaker"] == "assistant":
                    self.assertIn(h["action"], VALID_ACTIONS)

    def test_perfect_channel_fills_every_slot(self):
        for spec in ["accept", "confirm", "spell"]:
            for seed in range(25):
                customer, agent = _run(seed, spec)
                rows, dlg = score_dialogue(agent, customer, {"policy": spec})
                self.assertTrue(dlg["task_success"], (spec, seed, dlg["wrong_slots"], agent.final_state()))
                self.assertEqual(dlg["n_missed"] + dlg["n_spurious"], 0)

    def test_cannot_type_falls_back_to_spelling_once(self):
        seed = next(s for s in range(100) if not GoalBuilder().build(s).can_type)
        customer, agent = _run(seed, "type")
        paths = [p.path for p in customer.packages]
        self.assertEqual(paths.count("cannot_type"), 1)              # TYPE is never asked again
        self.assertIn("spell", paths)
        self.assertTrue(any(a["repair_channel"] == "spell_fallback" for a in agent.attempts))
        rows, dlg = score_dialogue(agent, customer, {"policy": "type"})
        self.assertTrue(dlg["joint_goal_correct"])                   # spelling has no typos

    def test_typed_baseline_asks_by_typing(self):
        seed = next(s for s in range(100) if GoalBuilder().build(s).can_type)
        customer, agent = _run(seed, "typed")
        asks = [h for h in customer.history if h["speaker"] == "assistant" and h["slot"]]
        self.assertTrue(asks and all(h["action"] == "type" for h in asks))
        self.assertTrue(all(a["action"] == "type" for a in agent.attempts if a["mode"] == "typed"))

    def test_mock_asr_features_and_u2r(self):
        o = RepairOutcomeModel(TestPolicies.TABLE)
        rng = random.Random(0)
        rows, y = TestEstimator()._data(400, rng)
        est = SlotErrorEstimator().fit(rows, y)
        customer, agent = _run(5, "u2r", MockASRChannel("noise", 5, seed=5), o, est)
        spoken = [a for a in agent.attempts if a["mode"] == "speech"]
        self.assertTrue(spoken and all(a["p_hat"] is not None and a["features"] for a in spoken))
        self.assertTrue(all(a["expected_costs"] for a in spoken))

    def test_slot_features(self):
        h = Heard("my name is varun kamath", words=[("my", .99), ("name", .99), ("is", .99),
                                                     ("varun", .5), ("kamath", .2)],
                  samples=["my name is varun kamath", "my name is baron kamath"])
        f = slot_features(h, "varun kamath")
        self.assertAlmostEqual(f["min_logprob"], float(np.log(0.2)))
        self.assertAlmostEqual(f["agreement"], 0.5)
        self.assertIsNone(slot_features(Heard("x", mode="typed"), "x"))


class TestEndToEnd(unittest.TestCase):
    def test_fit_train_eval(self):
        import contextlib
        import io

        import run_agent
        from assistant_agent import train
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):
            common = ["--channel", "mock", "--n", "6", "--out", f"{d}/runs"]
            run_agent.main(["--phase", "fit", "--conditions", "clean", "noise:5",
                            "--logdir", f"{d}/fit", *common])
            train.main(["--attempts", f"{d}/fit/attempts.jsonl", "--out", f"{d}/models", "--min-n", "5"])
            run_agent.main(["--policies", "u2r", "accept", "--conditions", "music:5", "--logdir", f"{d}/eval",
                            "--estimator", f"{d}/models/estimator.json",
                            "--outcomes", f"{d}/models/outcomes.json", *common])
            dlg = [json.loads(line) for line in open(f"{d}/eval/dialogues.jsonl")]
            self.assertEqual(len(dlg), 12)
            self.assertTrue((Path(d) / "eval" / "agent_prompts.txt").exists())


if __name__ == "__main__":
    unittest.main()

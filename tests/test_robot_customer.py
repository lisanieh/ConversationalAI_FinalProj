"""Unit tests for the robot customer. Run:  python -m unittest discover tests"""

import random
import tempfile
import unittest
from pathlib import Path

from robot_customer import (AssistantTurn, GoalBuilder, Path as RoutePath, RobotCustomer,
                            ScriptedBrain, add_typos, read_jsonl, route, run_dialogue,
                            spell_out, to_spoken)
from robot_customer.brain import build_messages, clean_llm_output
from robot_customer.mock_assistant import MockAssistant
from robot_customer.speech_text import email_spoken, number_to_words, phone_spoken


class TestGoalBuilder(unittest.TestCase):
    def test_reproducible(self):
        b = GoalBuilder()
        self.assertEqual(b.build(7).to_dict(), b.build(7).to_dict())
        self.assertNotEqual(b.build(7).slots, b.build(8).slots)

    def test_has_contact_slots(self):
        g = GoalBuilder().build(1)
        for k in ("name", "email", "phone", "day", "people", "area"):
            self.assertIn(k, g.slots)
        self.assertRegex(g.slots["phone"], r"^\d{3} \d{3} \d{4}$")
        self.assertIn("@", g.slots["email"])

    def test_cannot_type_rate(self):
        b = GoalBuilder(cannot_type_rate=0.2)
        share = sum(not b.build(s).can_type for s in range(2000)) / 2000
        self.assertAlmostEqual(share, 0.2, delta=0.03)

    def test_multiwoz_loader(self):
        import json
        data = {"MUL001.json": {"goal": {
            "hotel": {"info": {"area": "north", "pricerange": "cheap", "type": "hotel"},
                      "book": {"day": "friday", "people": "3", "stay": "3"}},
            "restaurant": {}}}}
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "data.json"
            p.write_text(json.dumps(data))
            g = GoalBuilder(multiwoz_path=p).build(0)
        self.assertEqual(g.source, "multiwoz:MUL001.json")
        self.assertEqual(g.slots["day"], "friday")
        self.assertEqual(g.slots["stay"], "3")


class TestSpeechText(unittest.TestCase):
    def test_email(self):
        self.assertEqual(email_spoken("varun.k@gmail.com"), "varun dot k at gmail dot com")

    def test_phone(self):
        self.assertEqual(phone_spoken("404 555 1234"), "four oh four, five five five, one two three four")

    def test_sentence(self):
        self.assertEqual(to_spoken("Around 3pm on Fri for 5 people"),
                         "Around three p m on friday for five people")
        self.assertEqual(to_spoken("At 18:30."), "At six thirty p m.")

    def test_numbers(self):
        self.assertEqual(number_to_words(42), "forty-two")
        self.assertEqual(number_to_words(105), "one hundred five")

    def test_spell(self):
        self.assertEqual(spell_out("varun.k@gmail.com", "email"),
                         "V. A. R. U. N. dot. K. at. G. M. A. I. L. dot. com.")
        self.assertEqual(spell_out("404 555 1234", "phone"),
                         "four. oh. four, five. five. five, one. two. three. four.")


class TestTypos(unittest.TestCase):
    def test_zero_rate_is_identity(self):
        self.assertEqual(add_typos("varun.k@gmail.com", 0.0).typed, "varun.k@gmail.com")

    def test_reproducible(self):
        a = add_typos("varun.k@gmail.com", 0.3, random.Random(5))
        b = add_typos("varun.k@gmail.com", 0.3, random.Random(5))
        self.assertEqual(a.typed, b.typed)
        self.assertTrue(a.edits)

    def test_rate_roughly_respected(self):
        rng = random.Random(0)
        text = "a" * 10000
        res = add_typos(text, 0.02, rng)
        self.assertAlmostEqual(len(res.edits) / len(text), 0.02, delta=0.006)


class TestRouter(unittest.TestCase):
    def setUp(self):
        self.goal = GoalBuilder().build(3)

    def test_paths(self):
        g = self.goal
        self.assertEqual(route(AssistantTurn("?", "normal", "area"), g), RoutePath.SPEAK)
        self.assertEqual(route(AssistantTurn("?", "confirm", "email"), g), RoutePath.SPEAK)
        self.assertEqual(route(AssistantTurn("?", "spell", "email"), g), RoutePath.SPELL)
        g.can_type = True
        self.assertEqual(route(AssistantTurn("?", "type", "email"), g), RoutePath.TYPE)
        g.can_type = False
        self.assertEqual(route(AssistantTurn("?", "type", "email"), g), RoutePath.CANNOT_TYPE)

    def test_unknown_action(self):
        with self.assertRaises(ValueError):
            route(AssistantTurn("?", "shout", "email"), self.goal)


class TestBrainHelpers(unittest.TestCase):
    def test_clean(self):
        self.assertEqual(clean_llm_output('Customer: "Hi, a hotel please." *smiles*'), "Hi, a hotel please.")
        self.assertEqual(clean_llm_output("Thanks! [DONE]"), "Thanks! [DONE]")
        self.assertEqual(clean_llm_output("Yes, thank you. DONE."), "Yes, thank you. [DONE]")
        self.assertEqual(clean_llm_output("I'm done with dinner."), "I'm done with dinner.")
        # The LLM writes the assistant's turn too: its [DONE] must not count.
        self.assertEqual(clean_llm_output(
            "I'm looking for a cheap guesthouse.\nAssistant: Booked!\nCustomer: Thanks [DONE]"),
            "I'm looking for a cheap guesthouse.")
        self.assertEqual(clean_llm_output("A guesthouse please.\n\nGreat, thanks! [DONE]"),
                         "A guesthouse please.")
        self.assertEqual(clean_llm_output("It's (404) 555-0100."), "It's (404) 555-0100.")

    def test_roles(self):
        g = GoalBuilder().build(0)
        msgs = build_messages(g, [{"speaker": "assistant", "text": "Hi"},
                                  {"speaker": "customer", "text": "Hello"}])
        self.assertEqual([m["role"] for m in msgs], ["system", "user", "assistant"])
        self.assertIn(g.slots["email"], msgs[0]["content"])


class TestCustomer(unittest.TestCase):
    def test_spell_and_type_use_exact_goal_value(self):
        g = GoalBuilder().build(11)
        g.can_type = True
        c = RobotCustomer(g, ScriptedBrain(), typo_rate=0.0)
        c.respond({"text": "Hello", "action": "normal", "slot": None})
        sp = c.respond({"text": "Spell your email", "action": "spell", "slot": "email"})
        ty = c.respond({"text": "Type your email", "action": "type", "slot": "email"})
        self.assertEqual(sp.true_text, g.slots["email"])
        self.assertEqual(sp.spoken_text, spell_out(g.slots["email"], "email"))
        self.assertEqual(sp.mode, "speech")
        self.assertEqual(ty.mode, "typed")
        self.assertEqual(ty.typed_text, g.slots["email"])
        self.assertIsNone(ty.audio_file)

    def test_cannot_type(self):
        g = GoalBuilder().build(11)
        g.can_type = False
        c = RobotCustomer(g, ScriptedBrain())
        p = c.respond({"text": "Type your email", "action": "type", "slot": "email"})
        self.assertEqual(p.path, "cannot_type")
        self.assertEqual(p.mode, "speech")
        self.assertIn("driving", p.true_text)

    def test_tts_hook(self):
        g = GoalBuilder().build(2)
        seen = []
        c = RobotCustomer(g, ScriptedBrain(), tts=lambda text, voice, path: seen.append((text, voice)) or 1.5)
        p = c.respond({"text": "Hello", "action": "normal", "slot": None})
        self.assertEqual(p.audio_seconds, 1.5)
        self.assertEqual(seen[0][1], g.voice)

    def test_full_dialogue_and_save(self):
        with tempfile.TemporaryDirectory() as d:
            g = GoalBuilder().build(4)
            c = RobotCustomer(g, ScriptedBrain(), typo_rate=0.0, out_dir=d)
            a = MockAssistant(g.domain, policy="mixed")
            res = run_dialogue(c, a.step, max_turns=30)
            self.assertTrue(res["done"])
            self.assertEqual(a.final_state(), g.slots)
            out = c.save(a.final_state())
            pk = read_jsonl(out / "turns.jsonl")
            self.assertEqual(len(pk), res["turns"])
            self.assertTrue((out / "prompts.txt").exists())

    def test_early_done_is_ignored(self):
        """Reproduces the PACE run: Qwen appended [DONE] to its first sentence."""
        class EagerBrain:
            last_raw = None
            def reply(self, goal, history):
                self.last_raw = "I'm looking for a cheap guesthouse. [DONE]"
                return "I'm looking for a cheap guesthouse. [DONE]"
        c = RobotCustomer(GoalBuilder().build(19), EagerBrain())
        p0 = c.respond({"text": "Hello", "action": "normal", "slot": None})
        self.assertFalse(p0.done)
        self.assertTrue(p0.done_ignored)
        self.assertEqual(p0.llm_raw, "I'm looking for a cheap guesthouse. [DONE]")
        p1 = c.respond({"text": "Your phone is 614 854 1763?", "action": "confirm", "slot": "phone"})
        self.assertFalse(p1.done)
        p2 = c.respond({"text": "Your booking is confirmed.", "action": "normal", "slot": None})
        self.assertTrue(p2.done)

    def test_turn_limit(self):
        g = GoalBuilder().build(4)
        c = RobotCustomer(g, ScriptedBrain())
        res = run_dialogue(c, lambda pkg: {"text": "Hmm?", "action": "normal", "slot": None}, max_turns=5)
        self.assertTrue(res["hit_turn_limit"])
        self.assertEqual(res["turns"], 5)


if __name__ == "__main__":
    unittest.main()

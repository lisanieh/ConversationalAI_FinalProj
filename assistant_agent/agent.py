"""The assistant agent: dialogue state + slot error estimate + U2R repair.

    customer turn -> channel -> Heard (Whisper transcript or typed text)
       -> extract slot values (LLM)          {slot: raw value}
       -> features -> p_hat                  P(value is wrong)
       -> policy (U2R)                       accept / confirm / spell / type
       -> next turn {"text", "action", "slot"}   (the robot customer reads the label)
       -> decode the repair answer, commit the value

Every value the agent hears becomes one *attempt* record (``self.attempts``).
The agent never sees the goal: ``scoring.score_dialogue`` adds the truth later.

Rules the agent follows:
  * Typed input does not go through Whisper. It is read back before it is
    committed ("I have ..., is that right?"); on a "no" the customer is asked
    to type it again, at most ``max_retypes`` times (``type_readback=False``
    commits typed input as is, the old behaviour).
  * One question per turn: several heard values queue their repairs.
  * "No" to a confirmation without a new value -> spell (name/email/phone) or
    a plain re-ask (other slots). That follow-up is committed as it is heard.
  * "No, it's X" for name/email/phone: X was heard by the same recognizer, so
    it is read back once ("recheck"); a second "no" -> spell. Other slots take X.
  * Once the customer says they cannot type, TYPE is never asked again; a
    TYPE choice becomes SPELL (logged as ``spell_fallback``).
  * A slot that gets no value after ``max_asks`` questions is left empty.
  * SPELL / TYPE are only used for name, email and phone (``allowed_actions``);
    for other slots a policy's SPELL / TYPE becomes CONFIRM
    (logged as ``requested_action``).
"""

from __future__ import annotations

from dataclasses import dataclass

from .channel import Heard, slot_features
from .config import DOMAINS, EFFORT, ENTITY_SLOTS, allowed_actions, schema, slot_type
from .extract import RuleExtractor
from .llm import parse_json
from .policies import Policy
from .prompts import CHOOSE_PROMPT, MISHEAR_PROMPT
from .repair import (ASK_DOMAIN, BOOKED, GREETING, decode_confirm, decode_spelled, is_cannot_type,
                     request)
from .values import DONTCARE, canonical


@dataclass
class Pending:
    slot: str
    action: str                 # confirm | spell | type | reask
    attempt: dict | None        # the attempt being repaired (None for a plain question)


class U2RAgent:
    def __init__(self, policy: Policy, extractor=None, estimator=None, llm=None,
                 ask_llm: bool = False, max_asks: int = 2, type_readback: bool = True,
                 max_retypes: int = 1):
        self.policy, self.extractor = policy, extractor or RuleExtractor()
        self.type_readback, self.max_retypes = type_readback, max_retypes
        self.rules = RuleExtractor()
        self.estimator, self.llm, self.ask_llm, self.max_asks = estimator, llm, ask_llm, max_asks
        self.domain: str | None = None
        self.state: dict[str, str] = {}              # includes values still waiting for a repair answer
        self.committed: dict[str, str] = {}          # only values the agent has settled on
        self.turn_log: list[dict] = []               # per customer turn: committed state and new commits
        self._commits_now: dict[str, str] = {}
        self.attempts: list[dict] = []
        self.queue: list[Pending] = []
        self.pending: Pending | None = None          # what the last agent turn asked
        self.asks: dict[str, int] = {}
        self.user_cannot_type = False
        self.finished = False
        self.turns: list[dict] = []                  # what the agent sent, for the transcript

    # ============================================================== main step
    def step(self, heard: Heard | None) -> dict:
        if heard is not None:
            self._hear(heard)
            self.turn_log.append({"turn": heard.turn, "belief": dict(self.committed),
                                  "commits": self._commits_now})
            self._commits_now = {}
        out = self._next()
        self.turns.append(out)
        return out

    def _hear(self, heard: Heard) -> None:
        p, self.pending = self.pending, None
        if p is None or p.action in ("normal", "ask_type", "domain"):
            if p is not None and p.action == "ask_type" and is_cannot_type(heard.text):
                self.user_cannot_type = True
                self.asks[p.slot] = 0                # ask again, by voice this time
                return
            if p is not None and p.action == "ask_type" and heard.mode == "typed":
                att = self._new_value(p.slot, heard.text, heard)
                if att is not None:                  # typing was the first channel, not a repair
                    att.update(action="type", repair_channel="ask_type" + ("+readback" if self.type_readback else ""),
                               effort_prior_s=EFFORT["type"], repair_turns=[heard.turn])
                return
            self._take_in(heard, asked=p.slot if p else None)
        else:
            self._resolve(p, heard)

    # ======================================================= new information
    def _take_in(self, heard: Heard, asked: str | None) -> None:
        dom, values = self.extractor.extract(heard.text, asked_slot=asked, domain=self.domain)
        if self.domain is None and dom in DOMAINS:
            self.domain = dom
        allowed = schema(self.domain) if self.domain else None
        for slot, raw in values.items():
            if allowed is not None and slot not in allowed:
                continue
            self._new_value(slot, raw, heard)

    def _new_value(self, slot: str, raw: str, heard: Heard) -> dict | None:
        hyp = canonical(slot, raw)
        if not hyp:
            return None
        if hyp == DONTCARE:
            self.state[slot] = self.committed[slot] = DONTCARE
            return None
        if self.state.get(slot) == hyp:
            return None                               # repeated, nothing new
        st = slot_type(slot)
        att = {"slot": slot, "slot_type": st, "turn": heard.turn, "mode": heard.mode,
               "raw": raw, "hypothesis": hyp, "transcript": heard.text,
               "features": None, "p_hat": None, "llm_p_wrong": None, "expected_costs": None,
               "action": "accept", "repair_channel": "", "repair_received": None,
               "committed_value": None, "effort_prior_s": 0.0, "extra_turns": 0,
               "repair_turns": []}
        self.attempts.append(att)
        if heard.mode == "typed":
            self._typed(att, hyp)
            self.state[slot] = hyp
            return att

        feats = slot_features(heard, raw)
        att["features"] = feats
        if self.estimator is not None:
            att["p_hat"] = self.estimator.predict_one(feats, st)
        if self.ask_llm and self.llm is not None:
            att["llm_p_wrong"] = self.llm_p_wrong(heard.text, slot, hyp)
        allowed = allowed_actions(slot)
        ctx = {"p_hat": att["p_hat"], "slot_type": st, "features": feats, "agent": self,
               "transcript": heard.text, "slot": slot, "hyp": hyp, "can_type": not self.user_cannot_type,
               "allowed": allowed}
        action = self.policy.choose(ctx)
        if action not in allowed:                     # e.g. always_spell on "day": confirm instead
            att["requested_action"] = action
            action = "confirm"
        att["expected_costs"] = ctx.get("expected_costs")
        att["action"] = action
        if action == "accept":
            self._commit(att, hyp)
        elif action == "type" and self.user_cannot_type:
            att["repair_channel"] = "spell_fallback"
            self.queue.append(Pending(slot, "spell", att))
        else:
            att["repair_channel"] = action
            self.queue.append(Pending(slot, action, att))
        self.state[slot] = hyp                        # provisional until the repair answer
        return att

    # ============================================================== repairs
    def _resolve(self, p: Pending, heard: Heard) -> None:
        att = p.attempt
        att["extra_turns"] += 1
        att["repair_turns"].append(heard.turn)
        att["effort_prior_s"] += EFFORT[p.action]
        att["repair_received"] = heard.text
        hyp = att["hypothesis"]

        if p.action in ("type", "retype"):
            if heard.mode == "typed":
                self._typed(att, canonical(p.slot, heard.text) or hyp)
            elif is_cannot_type(heard.text):
                self.user_cannot_type = True
                att["repair_channel"] = "spell_fallback"
                self.queue.insert(0, Pending(p.slot, "spell", att))
            else:                                     # answered by voice anyway
                _, vals = self.rules.extract(heard.text, asked_slot=p.slot)
                self._commit(att, canonical(p.slot, vals.get(p.slot)) or hyp)
        elif p.action == "spell":
            self._commit(att, decode_spelled(heard.text, p.slot) or hyp)
        elif p.action == "confirm":
            verdict, value = decode_confirm(heard.text, p.slot, hyp, self.rules)
            if verdict == "yes":
                self._commit(att, hyp)
            elif value and p.slot in ENTITY_SLOTS:
                # "No, it's ..." went through the same recognizer that just got it wrong:
                # read the correction back once before trusting it.
                att["correction"] = value
                att["repair_channel"] = "confirm+recheck"
                self.queue.insert(0, Pending(p.slot, "recheck", att))
            elif value:
                self._commit(att, value)
            else:
                self._follow_up(p.slot, att)
        elif p.action == "recheck":
            verdict, _ = decode_confirm(heard.text, p.slot, att["correction"], self.rules)
            if verdict == "yes":
                self._commit(att, att["correction"])
            else:                                     # still wrong: stop using the voice channel
                self._follow_up(p.slot, att, prefix="confirm+recheck")
        elif p.action == "reask":
            _, vals = self.extractor.extract(heard.text, asked_slot=p.slot, domain=self.domain)
            self._commit(att, canonical(p.slot, vals.get(p.slot)) or hyp)
        elif p.action == "readback":
            typed = att["typed_value"]
            verdict, _ = decode_confirm(heard.text, p.slot, typed, self.rules)
            if verdict == "yes" or att["retypes"] >= self.max_retypes:
                self._commit(att, typed)
            else:                                     # "no": a spoken correction would go through
                att["retypes"] += 1                   # ASR again, so ask to type it once more
                self.queue.insert(0, Pending(p.slot, "retype", att))

    def _follow_up(self, slot: str, att: dict, prefix: str = "confirm") -> None:
        """After a "no": spell an entity, plainly re-ask anything else."""
        follow = "spell" if slot in ENTITY_SLOTS else "reask"
        att["repair_channel"] = f"{prefix}+{follow}"
        self.queue.insert(0, Pending(slot, follow, att))

    def _typed(self, att: dict, value: str) -> None:
        """A typed value: read it back before committing, so a typo can be caught."""
        if not self.type_readback:
            self._commit(att, value)
            return
        att["typed_value"] = value
        att.setdefault("retypes", 0)
        if not att["repair_channel"].endswith("+readback"):
            att["repair_channel"] += "+readback"
        self.queue.insert(0, Pending(att["slot"], "readback", att))

    def _commit(self, att: dict, value: str) -> None:
        att["committed_value"] = value
        self.state[att["slot"]] = self.committed[att["slot"]] = value
        self._commits_now[att["slot"]] = value

    # ============================================================ next turn
    def _next(self) -> dict:
        if self.queue:
            p = self.queue.pop(0)
            if p.action == "type" and self.user_cannot_type:     # queued before the customer said so
                p.action = "spell"
                p.attempt["repair_channel"] = "spell_fallback"
            self.pending = p
            value = ((p.attempt.get("typed_value") if p.action == "readback" else p.attempt["hypothesis"])
                     if p.attempt else "")
            if p.action == "recheck":
                value = p.attempt["correction"]
            label = {"reask": "normal", "readback": "confirm", "retype": "type",
                     "recheck": "confirm"}.get(p.action, p.action)
            return {"text": request(p.action, p.slot, value), "action": label, "slot": p.slot}
        if not self.turns:
            return self._ask_none(GREETING, "greet")
        if self.domain is None:
            self.asks["_domain"] = self.asks.get("_domain", 0) + 1
            if self.asks["_domain"] <= self.max_asks:
                return self._ask_none(ASK_DOMAIN, "domain")
            self.domain = DOMAINS[0]
        for slot in schema(self.domain):
            if slot not in self.state and self.asks.get(slot, 0) < self.max_asks:
                self.asks[slot] = self.asks.get(slot, 0) + 1
                typed = self.policy.ask_action == "type" and not self.user_cannot_type
                self.pending = Pending(slot, "ask_type" if typed else "normal", None)
                return {"text": request("ask_type" if typed else "normal", slot),
                        "action": "type" if typed else "normal", "slot": slot}
        self.finished = True
        return self._ask_none(BOOKED, "done")

    def _ask_none(self, text: str, kind: str) -> dict:
        self.pending = Pending("", "domain", None) if kind == "domain" else None
        return {"text": text, "action": "normal", "slot": None}

    def final_state(self) -> dict:
        return dict(self.state)

    # ============================================================ LLM extras
    def llm_p_wrong(self, transcript: str, slot: str, value: str) -> float | None:
        """RQ4: the LLM's own estimate that it misheard the value."""
        msg = MISHEAR_PROMPT.format(transcript=transcript, slot=slot, value=value)
        p = parse_json(self.llm.chat([{"role": "user", "content": msg}], max_tokens=30)).get("p_wrong")
        try:
            return min(1.0, max(0.0, float(p) / 100))
        except (TypeError, ValueError):
            return None

    def llm_choose(self, transcript: str, slot: str, value: str) -> str:
        """Baseline: the LLM picks the repair itself."""
        if self.llm is None:
            raise ValueError("the 'llm' policy needs an agent LLM (--agent-extractor llm)")
        msg = CHOOSE_PROMPT.format(transcript=transcript, slot=slot, value=value)
        a = str(parse_json(self.llm.chat([{"role": "user", "content": msg}], max_tokens=30))
                .get("action", "confirm")).lower()
        return a if a in ("accept", "confirm", "spell", "type") else "confirm"

"""Export runs in the format error_scripts/evaluate.py and policy_eval.py read.

run_agent.py writes these next to its own logs, in <logdir>/for_eval/:

  goals.json            {dialogue_id: {slot: value}}                      (episode_metrics.py)
  eval_turns.jsonl      one row per customer turn: gold_state, belief_state,
                        repairs [{slot, action, p_hat, p_llm, effort_s, typed_chars}],
                        commits, final                                     (episode_metrics.py)
  eval_slots.jsonl      one row per heard value: ref_text, transcript, ref_value,
                        hyp_value, p_hat, p_llm, Whisper features, and y / error_type
                        from error_scripts/align.label_slot                 (evaluate.py --slots)
  repairs.jsonl         one row per executed repair: slot_type, condition, action,
                        was_wrong, final_wrong, effort                       (outcome_model.py)
  replay_outcomes.jsonl save-point replays: final_wrong_<A>, effort_<A>    (policy_eval.py)

Names follow error_scripts: policies "U2R", "no_repair", "always_spell", ...;
actions upper case; conditions "noise_5", "music_10", "clean". The belief state
holds committed values only ("dontcare" dropped, since goals never contain it).

Merge several runs (for example the fit and eval phases, or several seeds run
as separate jobs) into one folder:

  python -m assistant_agent.export logs/eval_s0 logs/eval_s1 logs/eval_s2 --out eval_input/
  cd error_scripts && python evaluate.py --turns ../eval_input/eval_turns.jsonl \\
      --goals ../eval_input/goals.json --slots ../eval_input/eval_slots.jsonl --out ../results
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path

from .scoring import repair_seconds
from .values import DONTCARE, canonical, normalize, values_match

_ALIGN = Path(__file__).resolve().parents[1] / "error_scripts" / "align.py"
_align = None


def _label_slot():
    """error_scripts/align.label_slot, loaded by path so the team uses ONE labeller."""
    global _align
    if _align is None and _ALIGN.exists():
        spec = importlib.util.spec_from_file_location("error_scripts_align", _ALIGN)
        _align = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_align)
    return _align.label_slot if _align else None


POLICY_NAMES = {"accept": "no_repair", "confirm": "always_confirm", "spell": "always_spell",
                "type": "always_type", "typed": "typed_input", "llm": "llm_choice",
                "u2r": "U2R", "u2r-notype": "U2R_notype"}
FEATURE_NAMES = {"agreement": "agree_k", "snr_est": "snr_est_db", "value_found": "span_found"}


def policy_name(spec: str) -> str:
    if spec in POLICY_NAMES:
        return POLICY_NAMES[spec]
    if spec.startswith("threshold"):
        return "raw_conf_threshold"
    if spec.startswith("u2r:"):
        return f"U2R_Cw{spec.split(':')[1]}"
    return spec


def condition_name(condition: str, snr_db: float | None) -> str:
    return condition if snr_db is None else f"{condition}_{snr_db:g}"


def _gold_said(goal, packages, upto: int) -> dict:
    """Goal slots the customer has said (or spelled/typed) by turn ``upto``."""
    said = {}
    for pkg in packages[:upto + 1]:
        if pkg.slot in goal.slots and pkg.path in ("spell", "type"):
            said[pkg.slot] = goal.slots[pkg.slot]
        text = f" {normalize(pkg.true_text)} "
        for slot, v in goal.slots.items():
            if normalize(v) and f" {normalize(v)} " in text:
                said[slot] = v
    return {s: canonical(s, v) for s, v in said.items()}


def episode_rows(agent, customer, meta: dict, effort) -> tuple[dict, list[dict]]:
    goal = customer.goal
    pol, cond, seed = policy_name(meta["policy"]), condition_name(meta["condition"], meta.get("snr_db")), meta.get("seed", 0)
    repairs_at: dict[int, list] = {}
    for a in agent.attempts:
        if a["action"] == "accept" or a["committed_value"] is None or not a.get("repair_turns"):
            continue
        secs, chars = repair_seconds(a, agent, customer, effort)
        repairs_at.setdefault(a["repair_turns"][-1], []).append(
            {"slot": a["slot"], "action": a["action"].upper(), "channel": a["repair_channel"],
             "p_hat": a["p_hat"], "p_llm": a["llm_p_wrong"], "effort_s": secs, "typed_chars": chars})
    logs = agent.turn_log
    rows = []
    for i, t in enumerate(logs):
        rows.append({"dialogue_id": goal.dialogue_id, "seed": seed, "condition": cond, "policy": pol,
                     "turn_id": t["turn"],
                     "gold_state": _gold_said(goal, customer.packages, t["turn"]),
                     "belief_state": {k: v for k, v in t["belief"].items() if v != DONTCARE},
                     "repairs": repairs_at.get(t["turn"], []),
                     "commits": {k: v for k, v in t["commits"].items() if v != DONTCARE},
                     "final": i == len(logs) - 1})
    return {goal.dialogue_id: {s: canonical(s, v) for s, v in goal.slots.items()}}, rows


def slot_rows(attempt_rows: list[dict], customer) -> list[dict]:
    """eval_slots.jsonl rows from scored attempt rows (spoken values and missed slots)."""
    label = _label_slot()
    out = []
    for r in attempt_rows:
        if r.get("mode") == "typed" or r.get("spurious"):
            continue
        k = r.get("turn")
        ref_text = customer.packages[k].true_text if k is not None and k < len(customer.packages) else ""
        ref_v = canonical(r["slot"], r["true_value"])
        hyp_v = canonical(r["slot"], r.get("hypothesis")) if r.get("hypothesis") else None
        row = {"dialogue_id": r["dialogue_id"], "seed": r.get("seed", 0),
               "condition": condition_name(r["condition"], r.get("snr_db")), "policy": policy_name(r["policy"]),
               "turn": k, "slot": r["slot"], "slot_type": r["slot_type"], "ref_text": ref_text,
               "transcript": r.get("transcript") or "", "ref_value": ref_v, "hyp_value": hyp_v,
               "p_hat": r.get("p_hat"), "p_llm": r.get("llm_p_wrong"), "missed": r["missed"],
               "action": (r.get("action") or "").upper() or None}
        f = r.get("features") or {}
        for key, v in f.items():
            row[FEATURE_NAMES.get(key, key)] = v
        if f.get("mean_logprob") is not None:
            row["raw_conf"] = math.exp(f["mean_logprob"])
        if label:
            row.update(label(ref_text, row["transcript"], ref_v, hyp_v))
            row["y"] = int(not values_match(r["slot"], hyp_v, ref_v))   # same truth as our first_correct
            if row["y"] == 0:
                row["error_type"] = "correct"
            elif row["error_type"] == "correct":        # differs only in symbols align.normalize drops
                row["error_type"] = "substitution"
        else:
            row.update(y=int(not r["first_correct"]), error_type=r["error_type"])
        out.append(row)
    return out


def repair_rows(attempt_rows: list[dict]) -> list[dict]:
    """outcome_model.py input: one row per executed repair (forced or chosen)."""
    return [{"dialogue_id": r["dialogue_id"], "seed": r.get("seed", 0), "slot_type": r["slot_type"],
             "condition": condition_name(r["condition"], r.get("snr_db")), "policy": policy_name(r["policy"]),
             "action": r["action"].upper(), "channel": r.get("repair_channel"),
             "was_wrong": int(not r["first_correct"]), "final_wrong": int(not r["committed_correct"]),
             "effort": r.get("effort_s", 0.0)}
            for r in attempt_rows
            if r.get("mode") == "speech" and not r["missed"] and not r["spurious"]
            and r.get("action") not in (None, "accept") and r.get("committed_value") is not None]


def replay_rows(replays: list[dict]) -> list[dict]:
    out = []
    for r in replays:
        row = {k: v for k, v in r.items() if k != "features"}
        row["condition"] = condition_name(r["condition"], r.get("snr_db"))
        row["policy"] = policy_name(r["policy"])
        for key, v in (r.get("features") or {}).items():
            row[FEATURE_NAMES.get(key, key)] = v
        out.append(row)
    return out


class EvalWriter:
    """Streams the export files for one run into <logdir>/for_eval/."""

    def __init__(self, folder: str | Path):
        self.dir = Path(folder)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.goals: dict = {}
        self.files = {n: open(self.dir / n, "w", encoding="utf-8")
                      for n in ("eval_turns.jsonl", "eval_slots.jsonl", "repairs.jsonl", "replay_outcomes.jsonl")}

    def _write(self, name: str, rows) -> None:
        for r in rows:
            self.files[name].write(json.dumps(r, ensure_ascii=False) + "\n")

    def add(self, agent, customer, meta, effort, attempt_rows, replays=()) -> None:
        goal, turns = episode_rows(agent, customer, meta, effort)
        self.goals.update(goal)
        self._write("eval_turns.jsonl", turns)
        self._write("eval_slots.jsonl", slot_rows(attempt_rows, customer))
        self._write("repairs.jsonl", repair_rows(attempt_rows))
        self._write("replay_outcomes.jsonl", replay_rows(replays))

    def close(self) -> None:
        for f in self.files.values():
            f.close()
        (self.dir / "goals.json").write_text(json.dumps(self.goals, indent=1), encoding="utf-8")


def merge(logdirs: list[str], out: str) -> None:
    """Concatenate several runs' for_eval/ folders into one."""
    o = Path(out)
    o.mkdir(parents=True, exist_ok=True)
    goals = {}
    names = ("eval_turns.jsonl", "eval_slots.jsonl", "repairs.jsonl", "replay_outcomes.jsonl")
    with_handles = {n: open(o / n, "w", encoding="utf-8") for n in names}
    for d in logdirs:
        src = Path(d) / "for_eval" if (Path(d) / "for_eval").exists() else Path(d)
        goals.update(json.loads((src / "goals.json").read_text(encoding="utf-8")))
        for n in names:
            if (src / n).exists():
                with_handles[n].write((src / n).read_text(encoding="utf-8"))
    for f in with_handles.values():
        f.close()
    (o / "goals.json").write_text(json.dumps(goals, indent=1), encoding="utf-8")
    print(f"merged {len(logdirs)} run(s), {len(goals)} goals -> {o}/")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Merge runs' for_eval/ folders for error_scripts.")
    ap.add_argument("logdirs", nargs="+")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    merge(a.logdirs, a.out)


if __name__ == "__main__":
    main()

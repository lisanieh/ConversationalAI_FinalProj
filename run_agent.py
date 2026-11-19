"""Run the robot customer against the U2R assistant agent and log every slot.

  customer (robot_customer) -> channel (text | mock ASR | Whisper) -> agent (assistant_agent)

Offline plumbing test (no model, no GPU; NEVER report mock numbers):
  python run_agent.py --phase fit  --channel mock --n 40 --logdir logs/fit
  python -m assistant_agent.train --attempts logs/fit/attempts.jsonl --out models/
  python run_agent.py --phase eval --channel mock --n 40 --logdir logs/eval \
         --estimator models/estimator.json --outcomes models/outcomes.json

Read a few dialogues (perfect recognition, one policy):
  python run_agent.py --policies confirm --conditions text --channel text --n 3 --show

Real models: customer on Qwen, agent on Llama (two vLLM servers on one node):
  vllm serve Qwen/Qwen2.5-7B-Instruct          --port 8000 --gpu-memory-utilization 0.3
  vllm serve meta-llama/Llama-3.1-8B-Instruct  --port 8001 --gpu-memory-utilization 0.3
  python run_agent.py --brain llm --customer-url http://localhost:8000/v1 \
         --agent-extractor llm --agent-url http://localhost:8001/v1 ...
The Whisper channel plugs in through assistant_agent.channel.ASRChannel.

Each (goal seed) gets the same goal, voice and can-type trait under every
policy and condition, so policy comparisons are paired. Log files are
OVERWRITTEN per run, so give every run its own --logdir.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from assistant_agent import (ChatLLM, LLMExtractor, MockASRChannel, RepairOutcomeModel,
                             RuleExtractor, SlotErrorEstimator, TextChannel, U2RAgent,
                             make_policy, score_dialogue)
from assistant_agent import prompts as agent_prompts
from robot_customer import GoalBuilder, LLMBrain, RobotCustomer, ScriptedBrain, run_dialogue

PHASES = {
    # fit: forced policies on clean + noise -> training data for the estimator and r_a / e_a
    "fit": dict(policies=["accept", "confirm", "spell", "type"],
                conditions=["clean", "noise:20", "noise:10", "noise:5", "noise:0"]),
    # eval: every policy on HELD-OUT conditions (never used for fitting)
    "eval": dict(policies=["typed", "accept", "confirm", "spell", "type", "threshold:-0.5",
                           "u2r", "u2r-notype", "u2r:5", "u2r:20"],
                 conditions=["music:10", "music:5", "sidetalk:10", "sidetalk:5"]),
}


def parse_condition(spec: str) -> tuple[str, float | None]:
    name, _, snr = spec.partition(":")
    return name, float(snr) if snr else None


def make_channel(kind: str, condition: str, snr: float | None, seed: int):
    if kind == "text":
        return TextChannel()
    if kind == "mock":
        return MockASRChannel(condition, snr, seed=seed)
    raise SystemExit("--channel whisper: wrap the audio step in assistant_agent.channel.ASRChannel "
                     "and pass it here (see the README)")


def show(customer, agent, dlg) -> None:
    g = customer.goal
    print(f"\n=== {g.dialogue_id} | {g.domain} | can_type={g.can_type} | policy={dlg['policy']} "
          f"| {dlg['condition']}")
    print("goal:", g.slots)
    for h in customer.history:
        if h["speaker"] == "assistant":
            print(f"  A [{h['action']}:{h['slot']}] {h['text']}")
        else:
            print(f"  C <{h['path']}> {h['typed_text'] if h['mode'] == 'typed' else h['text']}")
    for a in agent.attempts:
        p = f"{a['p_hat']:.2f}" if a["p_hat"] is not None else "-"
        print(f"    heard {a['slot']}={a['hypothesis']!r} p_hat={p} -> {a['action']}"
              f"{'/' + a['repair_channel'] if a['repair_channel'] not in ('', a['action']) else ''}"
              f" -> {a['committed_value']!r}")
    print(f"  -> joint_goal={dlg['joint_goal_correct']} success={dlg['task_success']} "
          f"turns={dlg['n_turns']} effort={dlg['total_effort']:.1f} wrong={dlg['wrong_slots']}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", choices=list(PHASES), default=None)
    ap.add_argument("--policies", nargs="+", default=None)
    ap.add_argument("--conditions", nargs="+", default=None, help="clean | noise:10 | music:5 | text")
    ap.add_argument("--channel", choices=["text", "mock", "whisper"], default="mock")
    ap.add_argument("--n", type=int, default=10, help="goals (dialogues per policy and condition)")
    ap.add_argument("--seed", type=int, default=0, help="first goal seed")
    ap.add_argument("--multiwoz", default=None)
    ap.add_argument("--cannot-type-rate", type=float, default=0.20)
    ap.add_argument("--typo-rate", type=float, default=0.02)
    ap.add_argument("--max-turns", type=int, default=40)
    # customer
    ap.add_argument("--brain", choices=["scripted", "llm"], default="scripted")
    ap.add_argument("--customer-url", default="http://localhost:11434/v1")
    ap.add_argument("--customer-model", default="qwen2.5:7b-instruct")
    # agent
    ap.add_argument("--agent-extractor", choices=["rule", "llm"], default="rule")
    ap.add_argument("--agent-url", default="http://localhost:8001/v1")
    ap.add_argument("--agent-model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--estimator", default=None)
    ap.add_argument("--outcomes", default=None)
    ap.add_argument("--ask-llm", action="store_true", help="RQ4: also log the LLM's own p_wrong")
    # output
    ap.add_argument("--logdir", default="logs/run")
    ap.add_argument("--out", default="runs", help="per-dialogue folders from the robot customer")
    ap.add_argument("--show", action="store_true", help="print every dialogue")
    a = ap.parse_args(argv)

    preset = PHASES.get(a.phase, {})
    policies = a.policies or preset.get("policies") or ["accept"]
    conditions = a.conditions or preset.get("conditions") or (["text"] if a.channel == "text" else ["clean"])
    estimator = SlotErrorEstimator.load(a.estimator) if a.estimator else None
    outcomes = RepairOutcomeModel.load(a.outcomes) if a.outcomes else None
    needs_llm = a.agent_extractor == "llm" or a.ask_llm or "llm" in policies
    if any(p.startswith("u2r") for p in policies) and estimator is None:
        raise SystemExit("u2r policies need --estimator and --outcomes (train on the fit phase first)")

    logdir = Path(a.logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    (logdir / "agent_prompts.txt").write_text(
        f"PROMPT_VERSION: {agent_prompts.PROMPT_VERSION}\n\n" +
        "\n".join(f"=== {k} ===\n{v}\n" for k, v in agent_prompts.ALL_PROMPTS.items()), encoding="utf-8")
    (logdir / "config.json").write_text(json.dumps(vars(a), indent=2), encoding="utf-8")
    f_att = open(logdir / "attempts.jsonl", "w", encoding="utf-8")
    f_dlg = open(logdir / "dialogues.jsonl", "w", encoding="utf-8")

    builder = GoalBuilder(multiwoz_path=a.multiwoz, cannot_type_rate=a.cannot_type_rate)
    summary = {}
    for cond in conditions:
        cname, snr = parse_condition(cond)
        for spec in policies:
            for seed in range(a.seed, a.seed + a.n):
                goal = builder.build(seed)
                brain = (ScriptedBrain() if a.brain == "scripted" else
                         LLMBrain(a.customer_url, a.customer_model, seed=seed))
                customer = RobotCustomer(goal, brain, typo_rate=a.typo_rate,
                                         out_dir=Path(a.out) / spec.replace(":", "_") / cond.replace(":", "_"))
                llm = ChatLLM(a.agent_url, a.agent_model, seed=seed) if needs_llm else None
                extractor = LLMExtractor(llm) if a.agent_extractor == "llm" else RuleExtractor()
                agent = U2RAgent(make_policy(spec, outcomes), extractor, estimator, llm, ask_llm=a.ask_llm)
                channel = make_channel(a.channel, cname, snr, seed)

                run_dialogue(customer, lambda pkg: agent.step(channel(pkg) if pkg else None),
                             max_turns=a.max_turns)
                meta = {"run_id": logdir.name, "policy": spec, "condition": cname, "snr_db": snr,
                        "channel": a.channel}
                rows, dlg = score_dialogue(agent, customer, meta)
                for r in rows:
                    f_att.write(json.dumps(r, ensure_ascii=False) + "\n")
                f_dlg.write(json.dumps(dlg, ensure_ascii=False) + "\n")
                customer.save(final_state=agent.final_state())
                s = summary.setdefault((cond, spec), [0, 0, 0.0, 0])
                s[0] += dlg["task_success"]
                s[1] += dlg["n_wrong_commits"]
                s[2] += dlg["total_effort"]
                s[3] += 1
                if a.show:
                    show(customer, agent, dlg)
    f_att.close()
    f_dlg.close()

    print(f"\n{'condition':12s} {'policy':16s} {'success':>8s} {'wrong/dlg':>10s} {'effort/dlg':>11s}")
    for (cond, spec), (succ, wrong, eff, n) in summary.items():
        print(f"{cond:12s} {spec:16s} {succ / n:8.0%} {wrong / n:10.2f} {eff / n:11.1f}")
    print(f"\nLogs: {logdir / 'attempts.jsonl'}, {logdir / 'dialogues.jsonl'}")
    if a.channel == "mock":
        print("Mock ASR: these numbers test the plumbing only. Do not report them.")


if __name__ == "__main__":
    main()

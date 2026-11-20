"""Run the robot customer against the U2R assistant agent and log every slot.

  customer (robot_customer) -> channel (text | mock ASR | Whisper) -> agent (assistant_agent)

Offline plumbing test (no model, no GPU; NEVER report mock numbers):
  python run_agent.py --phase fit  --channel mock --n 40 --seeds 0 1 2 --logdir logs/fit
  python -m assistant_agent.train --attempts logs/fit/attempts.jsonl \
         --replays logs/fit/replays.jsonl --out models/
  python run_agent.py --phase eval --channel mock --n 40 --seeds 0 1 2 --logdir logs/eval \
         --estimator models/estimator.json --outcomes models/outcomes.json
  cd error_scripts && python evaluate.py --turns ../logs/eval/for_eval/eval_turns.jsonl \
         --goals ../logs/eval/for_eval/goals.json --slots ../logs/eval/for_eval/eval_slots.jsonl \
         --out ../results

Read a few dialogues (perfect recognition, one policy):
  python run_agent.py --policies confirm --conditions text --channel text --n 3 --show

Real models: customer on Qwen, agent on Llama (two vLLM servers on one node):
  vllm serve Qwen/Qwen2.5-7B-Instruct          --port 8000 --gpu-memory-utilization 0.3
  vllm serve meta-llama/Llama-3.1-8B-Instruct  --port 8001 --gpu-memory-utilization 0.3
  python run_agent.py --brain llm --customer-url http://localhost:8000/v1 \
         --agent-extractor llm --agent-url http://localhost:8001/v1 ...

Real audio (Kokoro TTS -> MUSAN noise -> Whisper, assistant_agent/audio.py):
  python run_agent.py --channel whisper --musan datasets/musan --whisper-model small.en --K 5          --multiwoz datasets/multiwoz/data/MultiWOZ_2.2/train --phase fit ...
Goals: --multiwoz takes MultiWOZ 2.2 (a split folder, the release folder, or the
download_datasets.py JSON) or 2.1 (data.json); without it, goals are sampled
from the ontology. Use train goals for the fit phase and test goals for eval.

Design
  * --n goals x --seeds runs. A goal (dialogue_id) is the same under every
    policy, condition and seed; the run seed changes the randomness only
    (customer LLM sampling, typos, recognizer noise). Comparisons are paired on
    (dialogue_id, seed).
  * --replay (on by default in the fit phase) saves the state before each turn
    with new spoken values and tries all four repairs on each value
    (assistant_agent/replay.py). These give r_a / e_a / c_a for U2R and the
    factorial table for error_scripts/policy_eval.py.
  * Effort is in seconds of the customer's time (assistant_agent/effort.py;
    --effort models/effort.json to use the team's own timings).
  * Outputs in --logdir (OVERWRITTEN, one folder per run): attempts.jsonl,
    dialogues.jsonl, replays.jsonl, agent_prompts.txt, config.json, and
    for_eval/ in the format error_scripts reads.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from assistant_agent import (ChatLLM, LLMExtractor, MockASRChannel, RepairOutcomeModel,
                             RuleExtractor, SlotErrorEstimator, TextChannel, U2RAgent,
                             make_policy, score_dialogue)
from assistant_agent import prompts as agent_prompts
from assistant_agent.effort import EffortModel
from assistant_agent.export import EvalWriter
from assistant_agent.replay import drive, replay_turn
from robot_customer import GoalBuilder, LLMBrain, RobotCustomer, ScriptedBrain

PHASES = {
    # fit: clean + noise -> training data for the estimator, and save-point replays for r/e/c
    "fit": dict(policies=["accept"], conditions=["clean", "noise:20", "noise:15", "noise:10", "noise:5"],
                replay=True),
    # eval: every policy on HELD-OUT conditions (never used for fitting). C_w sweep in seconds.
    "eval": dict(policies=["typed", "accept", "confirm", "spell", "type", "threshold:-0.5",
                           "u2r", "u2r-notype", "u2r:30", "u2r:120"],
                 conditions=["music:10", "music:5", "sidetalk:10", "sidetalk:5"], replay=False),
}


def parse_condition(spec: str) -> tuple[str, float | None]:
    name, _, snr = spec.partition(":")
    return name, float(snr) if snr else None


def run_seed_of(goal_seed: int, run_seed: int) -> int:
    """One integer per (goal, repetition) for LLM sampling and recognizer noise."""
    return goal_seed * 1009 + run_seed


def make_audio(a) -> dict:
    """Load Kokoro, Whisper and MUSAN once per run (GPU node)."""
    from assistant_agent.audio import KokoroTTS, NoiseMixer, WhisperASR
    save = Path(a.save_audio) if a.save_audio else None
    tts = KokoroTTS(device=a.audio_device, save_dir=save / "clean" if save else None)
    asr = WhisperASR(a.whisper_model, device=a.audio_device, K=a.K, sample_temperature=a.sample_temperature)
    mixer = NoiseMixer(a.musan) if a.musan else None
    return {"tts": tts, "asr": asr, "mixer": mixer, "save_dir": save}


def make_channel(kind: str, condition: str, snr: float | None, seed: int, audio: dict | None = None):
    if kind == "text":
        return TextChannel()
    if kind == "mock":
        return MockASRChannel(condition, snr, seed=seed)
    from assistant_agent.audio import AudioChannel
    return AudioChannel(audio["tts"], audio["asr"], audio["mixer"], condition, snr, seed=seed,
                        save_dir=audio["save_dir"])


def show(customer, agent, dlg) -> None:
    g = customer.goal
    print(f"\n=== {g.dialogue_id} seed={dlg['seed']} | {g.domain} | can_type={g.can_type} "
          f"| policy={dlg['policy']} | {dlg['condition']}")
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
          f"turns={dlg['n_turns']} repair effort={dlg['total_effort_s']:.1f}s "
          f"call={dlg['total_user_time_s']:.0f}s wrong={dlg['wrong_slots']}")


def run_all(a, conditions, policies, builder, estimator, outcomes, effort, needs_llm, do_replay,
            logdir, files, writer, summary, audio=None) -> None:
    f_att, f_dlg, f_rep = files
    for cond in conditions:
        cname, snr = parse_condition(cond)
        for spec in policies:
            for run_seed in a.seeds:
                for gs in range(a.first_goal, a.first_goal + a.n):
                    goal = builder.build(gs)
                    rs = run_seed_of(gs, run_seed)
                    brain = (ScriptedBrain() if a.brain == "scripted" else
                             LLMBrain(a.customer_url, a.customer_model, seed=rs))
                    out_dir = (None if a.no_save_dialogues else
                               Path(a.out) / spec.replace(":", "_") / cond.replace(":", "_") / f"seed{run_seed}")
                    customer = RobotCustomer(goal, brain, typo_rate=a.typo_rate, out_dir=out_dir,
                                             run_seed=run_seed, tts=audio["tts"] if audio else None)
                    llm = ChatLLM(a.agent_url, a.agent_model, seed=rs) if needs_llm else None
                    extractor = LLMExtractor(llm) if a.agent_extractor == "llm" else RuleExtractor()
                    agent = U2RAgent(make_policy(spec, outcomes), extractor, estimator, llm, ask_llm=a.ask_llm)
                    channel = make_channel(a.channel, cname, snr, rs, audio)
                    meta = {"run_id": logdir.name, "policy": spec, "condition": cname, "snr_db": snr,
                            "channel": a.channel, "seed": run_seed}

                    replays: list[dict] = []
                    hook = ((lambda pkg: replays.extend(replay_turn(customer, agent, channel, pkg, effort, meta)))
                            if do_replay else None)
                    drive(customer, agent, channel, max_turns=a.max_turns, on_turn=hook)

                    rows, dlg = score_dialogue(agent, customer, meta, effort)
                    for r in rows:
                        f_att.write(json.dumps(r, ensure_ascii=False) + "\n")
                    f_dlg.write(json.dumps(dlg, ensure_ascii=False) + "\n")
                    for r in replays:
                        f_rep.write(json.dumps(r, ensure_ascii=False) + "\n")
                    writer.add(agent, customer, meta, effort, rows, replays)
                    if out_dir is not None:
                        customer.save(final_state=agent.final_state())
                    s = summary.setdefault((cond, spec), [0, 0, 0.0, 0, 0])
                    s[0] += dlg["task_success"]
                    s[1] += dlg["n_wrong_commits"]
                    s[2] += dlg["total_effort_s"]
                    s[3] += 1
                    s[4] += len(replays)
                    if a.show:
                        show(customer, agent, dlg)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", choices=list(PHASES), default=None)
    ap.add_argument("--policies", nargs="+", default=None)
    ap.add_argument("--conditions", nargs="+", default=None, help="clean | noise:10 | music:5 | text")
    ap.add_argument("--channel", choices=["text", "mock", "whisper"], default="mock")
    ap.add_argument("--n", type=int, default=10, help="number of goals")
    ap.add_argument("--first-goal", type=int, default=0, help="first goal seed")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0], help="run seeds (repetitions of each goal)")
    ap.add_argument("--replay", dest="replay", action="store_true", default=None,
                    help="save-point replays of all four repairs (default: on for --phase fit)")
    ap.add_argument("--no-replay", dest="replay", action="store_false")
    ap.add_argument("--multiwoz", default=None,
                    help="MultiWOZ 2.2 split folder / release folder / HF json, or 2.1 data.json")
    ap.add_argument("--cannot-type-rate", type=float, default=0.20)
    ap.add_argument("--typo-rate", type=float, default=0.02)
    ap.add_argument("--max-turns", type=int, default=40)
    ap.add_argument("--effort", default=None, help="effort model JSON (python -m assistant_agent.effort)")
    # audio (--channel whisper)
    ap.add_argument("--musan", default="datasets/musan", help="MUSAN root with noise/ music/ speech/")
    ap.add_argument("--whisper-model", default="small.en")
    ap.add_argument("--K", type=int, default=5, help="sampled Whisper transcripts per turn (agreement)")
    ap.add_argument("--sample-temperature", type=float, default=0.7)
    ap.add_argument("--audio-device", default=None, help="cuda | cpu (default: cuda if available)")
    ap.add_argument("--save-audio", default=None, help="folder to keep clean and mixed WAVs")
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
    ap.add_argument("--no-save-dialogues", action="store_true", help="skip the per-dialogue folders")
    ap.add_argument("--show", action="store_true", help="print every dialogue")
    a = ap.parse_args(argv)

    preset = PHASES.get(a.phase, {})
    policies = a.policies or preset.get("policies") or ["accept"]
    conditions = a.conditions or preset.get("conditions") or (["text"] if a.channel == "text" else ["clean"])
    do_replay = a.replay if a.replay is not None else preset.get("replay", False)
    estimator = SlotErrorEstimator.load(a.estimator) if a.estimator else None
    outcomes = RepairOutcomeModel.load(a.outcomes) if a.outcomes else None
    effort = EffortModel.load(a.effort)
    needs_llm = a.agent_extractor == "llm" or a.ask_llm or "llm" in policies
    if any(p.startswith("u2r") for p in policies) and (estimator is None or outcomes is None):
        raise SystemExit("u2r policies need --estimator and --outcomes (train on the fit phase first)")

    logdir = Path(a.logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    (logdir / "agent_prompts.txt").write_text(
        f"PROMPT_VERSION: {agent_prompts.PROMPT_VERSION}\n\n" +
        "\n".join(f"=== {k} ===\n{v}\n" for k, v in agent_prompts.ALL_PROMPTS.items()), encoding="utf-8")
    (logdir / "config.json").write_text(json.dumps({**vars(a), "replay": do_replay, "effort_model": vars(effort)},
                                                   indent=2), encoding="utf-8")
    f_att = open(logdir / "attempts.jsonl", "w", encoding="utf-8")
    f_dlg = open(logdir / "dialogues.jsonl", "w", encoding="utf-8")
    f_rep = open(logdir / "replays.jsonl", "w", encoding="utf-8")
    writer = EvalWriter(logdir / "for_eval")

    builder = GoalBuilder(multiwoz_path=a.multiwoz, cannot_type_rate=a.cannot_type_rate)
    summary: dict = {}
    audio = None
    if a.channel == "whisper":
        if all(c.split(":")[0] == "clean" for c in conditions):
            a.musan = None                      # no noise needed
        audio = make_audio(a)
    try:
        run_all(a, conditions, policies, builder, estimator, outcomes, effort, needs_llm, do_replay,
                logdir, (f_att, f_dlg, f_rep), writer, summary, audio)
    finally:                                   # close the logs even if a dialogue crashes
        for f in (f_att, f_dlg, f_rep):
            f.close()
        writer.close()

    print(f"\n{'condition':12s} {'policy':16s} {'success':>8s} {'wrong/dlg':>10s} {'repair s/dlg':>13s}"
          + ("  replayed values" if do_replay else ""))
    for (cond, spec), (succ, wrong, eff, n, nrep) in summary.items():
        print(f"{cond:12s} {spec:16s} {succ / n:8.0%} {wrong / n:10.2f} {eff / n:13.1f}"
              + (f"  {nrep:>15d}" if do_replay else ""))
    print(f"\nLogs: {logdir}/ (attempts, dialogues{', replays' if do_replay else ''}); "
          f"error_scripts input: {logdir / 'for_eval'}/")
    if a.channel == "mock":
        print("Mock ASR: these numbers test the plumbing only. Do not report them.")


if __name__ == "__main__":
    main()

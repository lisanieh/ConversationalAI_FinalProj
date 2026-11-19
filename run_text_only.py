"""Check 1 from the plan: run the robot customer against the mock assistant with
no audio, and print the conversations so you can read them.

Examples
  # No model needed (rule-based customer), just checks wiring:
  python run_text_only.py --brain scripted --n 5

  # Real customer brain with Ollama on a laptop:
  ollama pull qwen2.5:7b-instruct
  python run_text_only.py --brain llm --n 20

  # Real customer brain with vLLM on PACE:
  vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000
  python run_text_only.py --brain llm --base-url http://<node>:8000/v1 \
      --model Qwen/Qwen2.5-7B-Instruct --n 20

Outputs go to runs/<dialogue_id>/ : goal.json, turns.jsonl (the hand-off
packages), transcript.json, final_state.json, prompts.txt.
"""

import argparse

from robot_customer import GoalBuilder, LLMBrain, RobotCustomer, ScriptedBrain, run_dialogue
from robot_customer.mock_assistant import FIXED_POLICIES, MockAssistant


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--brain", choices=["scripted", "llm"], default="scripted")
    ap.add_argument("--base-url", default="http://localhost:11434/v1")
    ap.add_argument("--model", default="qwen2.5:7b-instruct")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--multiwoz", default=None, help="path to MultiWOZ 2.1 data.json (optional)")
    ap.add_argument("--policy", choices=list(FIXED_POLICIES), default="mixed")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0, help="first dialogue seed")
    ap.add_argument("--max-turns", type=int, default=20)
    ap.add_argument("--typo-rate", type=float, default=0.02)
    ap.add_argument("--out", default="runs")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    builder = GoalBuilder(multiwoz_path=args.multiwoz)
    n_done = n_correct = 0
    for seed in range(args.seed, args.seed + args.n):
        goal = builder.build(seed)
        brain = (ScriptedBrain() if args.brain == "scripted" else
                 LLMBrain(args.base_url, args.model, temperature=args.temperature, seed=seed))
        customer = RobotCustomer(goal, brain, typo_rate=args.typo_rate, out_dir=args.out)
        assistant = MockAssistant(goal.domain, policy=args.policy)
        result = run_dialogue(customer, assistant.step, max_turns=args.max_turns)
        final = assistant.final_state()
        correct = all(final.get(k) == v for k, v in goal.slots.items())
        customer.save(final_state=final)
        n_done += result["done"]
        n_correct += correct

        if not args.quiet:
            print(f"\n=== {goal.dialogue_id} | {goal.domain} | can_type={goal.can_type} | voice={goal.voice}")
            print("goal:", goal.slots)
            for h in customer.history:
                if h["speaker"] == "assistant":
                    print(f"  A [{h['action']}:{h['slot']}] {h['text']}")
                else:
                    shown = h["typed_text"] if h["mode"] == "typed" else h["text"]
                    print(f"  C <{h['path']}> {shown}")
            pk = [p for p in customer.packages if p.path == "spell"]
            for p in pk:
                print(f"    spoken (spell): {p.spoken_text}")
            print(f"  -> done={result['done']} turns={result['turns']} all_slots_correct={correct}")

    print(f"\n{n_done}/{args.n} finished with [DONE]; {n_correct}/{args.n} final forms match the goal "
          f"(mismatches are expected from typos).")


if __name__ == "__main__":
    main()

"""Run a task set against a model and report success rate.

    python -m tiny_api_agent.eval --spec examples/library.json --tasks examples/library_tasks.json \
        --model anthropic:claude-haiku-5-5 --verbosity 3

    python -m tiny_api_agent.eval ... --dry-run      # print the first prompt and exit
    python -m tiny_api_agent.eval ... --show         # print each transcript
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from .drift import no_drift
from .models import load_model
from .reward import compute_reward
from .rollout import SYSTEM_PROMPT, build_user_prompt, run_episode
from .schema import ApiSpec
from .server import MockServer


def load_tasks(path: str) -> list[dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data["tasks"] if isinstance(data, dict) else data


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--spec", required=True)
    p.add_argument("--tasks", required=True)
    p.add_argument("--model", default="anthropic:claude-haiku-5-5")
    p.add_argument("--verbosity", type=int, default=3, help="server error verbosity 0..3")
    p.add_argument("--budget", type=int, default=5, help="max requests per episode")
    p.add_argument("--limit", type=int, default=None, help="only run the first N tasks")
    p.add_argument("--task-id", default=None, help="run a single task by id")
    p.add_argument("--show", action="store_true", help="print transcripts")
    p.add_argument("--dry-run", action="store_true", help="print the first prompt and exit")
    p.add_argument("--out", default=None, help="JSONL path for episodes (default runs/<ts>.jsonl)")
    args = p.parse_args(argv)

    spec = ApiSpec.load(args.spec)
    tasks = load_tasks(args.tasks)
    if args.task_id:
        tasks = [t for t in tasks if t.get("id") == args.task_id]
    if args.limit:
        tasks = tasks[: args.limit]
    if not tasks:
        print("no tasks selected", file=sys.stderr)
        return 2

    spec_for_model, _ = no_drift(spec)

    if args.dry_run:
        token = spec.auth.token if spec.auth else None
        print("=== SYSTEM ===")
        print(SYSTEM_PROMPT.format(budget=args.budget))
        print("=== USER ===")
        print(build_user_prompt(spec_for_model, tasks[0], token))
        return 0

    model = load_model(args.model)
    out_path = Path(args.out) if args.out else Path("runs") / f"{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    server = MockServer(spec, verbosity=args.verbosity)
    results = []
    with out_path.open("w", encoding="utf-8") as fh:
        for task in tasks:
            server.reset()
            t0 = time.time()
            ep = run_episode(model, spec, spec_for_model, task, server, budget=args.budget)
            reward = compute_reward(spec, task, ep, server.snapshot())
            dt = time.time() - t0
            results.append(reward)
            mark = "PASS" if reward.success else "FAIL"
            print(f"{mark}  {task.get('id', '?'):<22} reqs={reward.n_requests}  score={reward.score:+.2f}  {dt:4.1f}s")
            if args.show:
                for turn in ep.turns:
                    tag = "MODEL " if turn.role == "model" else "SERVER"
                    print(f"    [{tag}] {turn.text.strip()[:600]}")
                print()
            fh.write(json.dumps({"episode": ep.to_json(), "reward": reward.as_dict(), "model": model.name,
                                 "verbosity": args.verbosity}) + "\n")

    n = len(results)
    passed = sum(r.success for r in results)
    mean_score = sum(r.score for r in results) / n
    mean_reqs = sum(r.n_requests for r in results) / n
    print()
    print(f"model={model.name} verbosity={args.verbosity}  success {passed}/{n} ({100 * passed / n:.0f}%)  "
          f"mean score {mean_score:+.3f}  mean requests {mean_reqs:.2f}")
    print(f"episodes -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

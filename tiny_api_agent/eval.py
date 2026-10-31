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

from dotenv import load_dotenv

from .env.drift import no_drift
from .models import load_model
from .reward.executor import Reward, compute_reward, expected_outcome
from .roles.executor import DEFAULT_BUDGET, build_user_prompt, run_episode, system_prompt

# Run-file format. Every JSONL row carries this. Two row shapes share it:
#   episode row: {"schema_version", "episode", "reward", "model", "verbosity"}
#   error row:   {"schema_version", "task", "error", "model", "verbosity", "episode"?}  (episode present
#                when the model finished but scoring failed). Readers branch on "error" in row.
RUN_SCHEMA_VERSION = 2
from .env.schema import ApiSpec
from .env.server import MockServer


def _positive_int(s: str) -> int:
    v = int(s)
    if v < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return v


def _task_label(task: dict) -> str:
    return str(task.get("id") or "?")


def load_tasks(path: str) -> list[dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data["tasks"] if isinstance(data, dict) else data


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--spec", required=True)
    p.add_argument("--tasks", required=True)
    p.add_argument("--model", default="anthropic:claude-haiku-5-5")
    p.add_argument("--verbosity", type=int, default=3, choices=range(4), help="server error verbosity 0..3")
    p.add_argument("--budget", type=_positive_int, default=DEFAULT_BUDGET, help="max turns of requests per episode")
    p.add_argument("--limit", type=_positive_int, default=None, help="only run the first N tasks")
    p.add_argument("--task-id", default=None, help="run a single task by id")
    p.add_argument("--show", action="store_true", help="print transcripts")
    p.add_argument("--dry-run", action="store_true", help="print the first prompt and exit")
    p.add_argument("--out", default=None, help="JSONL path for episodes (default runs/<ts>.jsonl)")
    args = p.parse_args(argv)

    load_dotenv()  # reads .env from the project root if present
    for stream in (sys.stdout, sys.stderr):  # transcripts carry any unicode; a cp1252 console must not kill the run
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
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
        print(system_prompt(args.budget))
        print("=== USER ===")
        print(build_user_prompt(spec_for_model, tasks[0], token))
        return 0

    model = load_model(args.model)
    out_path = Path(args.out) if args.out else Path("runs") / f"{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Validate every task (prompt and golden path) before spending a single model call.
    token = spec.auth.token if spec.auth else None
    for task in tasks:
        try:
            build_user_prompt(spec_for_model, task, token)
            expected_outcome(spec, task)
        except Exception as e:
            print(f"task {_task_label(task)}: invalid task for this spec: {e}", file=sys.stderr)
            return 2

    server = MockServer(spec, verbosity=args.verbosity)
    results: list[Reward] = []
    errors = 0
    with out_path.open("w", encoding="utf-8") as fh:
        for task in tasks:
            server.reset()
            t0 = time.time()
            ep = None
            try:
                ep = run_episode(model, spec, spec_for_model, task, server, budget=args.budget)
                reward = compute_reward(spec, task, ep, server.snapshot())
            except Exception as e:  # one failed episode (API error, bad task) must not lose the run
                errors += 1
                print(f"ERROR {_task_label(task):<22} {type(e).__name__}: {e}")
                row = {"schema_version": RUN_SCHEMA_VERSION, "task": task, "error": f"{type(e).__name__}: {e}",
                       "model": model.name, "verbosity": args.verbosity}
                if ep is not None:  # the model finished; keep its transcript even though scoring failed
                    row["episode"] = ep.to_json()
                fh.write(json.dumps(row) + "\n")
                continue
            dt = time.time() - t0
            results.append(reward)
            mark = "PASS" if reward.success else "FAIL"
            rep = ""
            if reward.repair_candidate:
                rep = " repaired" if reward.success else " unrepaired"
            print(f"{mark}  {_task_label(task):<22} turns={reward.n_turns} reqs={reward.n_requests}  score={reward.score:+.2f}  "
                  f"stop={ep.stop_reason}  {dt:4.1f}s{rep}")
            if args.show:
                for turn in ep.turns:
                    tag = "MODEL " if turn.role == "model" else "SERVER"
                    print(f"    [{tag}] {turn.text.strip()[:600]}")
                print()
            fh.write(json.dumps({"schema_version": RUN_SCHEMA_VERSION, "episode": ep.to_json(), "reward": reward.as_dict(),
                                 "model": model.name, "verbosity": args.verbosity}) + "\n")

    print()
    if not results:
        print(f"model={model.name} verbosity={args.verbosity}  no episode completed ({errors} errors)")
        print(f"episodes -> {out_path}")
        return 1
    s = summarize(results)
    print(f"model={model.name} verbosity={args.verbosity}  success {s['passed']}/{s['n']} ({100 * s['success_rate']:.0f}%)  "
          f"mean score {s['mean_score']:+.3f}  mean turns {s['mean_turns']:.2f}  mean requests {s['mean_requests']:.2f}"
          + (f"  errors {errors}" if errors else ""))
    if s["repair_candidates"]:
        print(f"repair-only: {s['repaired']}/{s['repair_candidates']} doable tasks with a rejected request still succeeded "
              f"({100 * s['repair_rate']:.0f}%); no-fit tasks excluded")
    else:
        print("repair-only: no doable task had a rejected request (nothing to repair in this run)")
    print(f"episodes -> {out_path}")
    return 1 if errors else 0  # a half-failed run is not a success for a pipeline reading the exit code


def summarize(results: list[Reward]) -> dict:
    """The numbers the summary line prints. Pure, so it can be tested with hand-built rewards."""
    n = len(results)
    candidates = [r for r in results if r.repair_candidate]
    repaired = sum(r.success for r in candidates)
    return {
        "n": n,
        "passed": sum(r.success for r in results),
        "success_rate": sum(r.success for r in results) / n,
        "mean_score": sum(r.score for r in results) / n,
        "mean_turns": sum(r.n_turns for r in results) / n,
        "mean_requests": sum(r.n_requests for r in results) / n,
        "repair_candidates": len(candidates),
        "repaired": repaired,
        "repair_rate": repaired / len(candidates) if candidates else None,
    }


if __name__ == "__main__":
    raise SystemExit(main())

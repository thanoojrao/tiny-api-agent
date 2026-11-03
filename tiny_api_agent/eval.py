"""Run a task set against a model and report success rate.

    python -m tiny_api_agent.eval --spec examples/library.json --tasks examples/library_tasks.json \
        --model anthropic:claude-haiku-5-5 --verbosity 3

    python -m tiny_api_agent.eval ... --dry-run      # print the first prompt and exit
    python -m tiny_api_agent.eval ... --show         # print each transcript
    python -m tiny_api_agent.eval ... --drift 2 --drift-kinds rename_param,move_endpoint --seed 7
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

from .env.drift import DriftRecord, apply_drift, golden_endpoints, parse_kinds
from .models import load_model
from .reward.executor import Reward, compute_reward, expected_outcome
from .roles.executor import DEFAULT_BUDGET, build_user_prompt, run_episode, system_prompt

# Run-file format. Every JSONL row carries this. Two row shapes share it:
#   episode row: {"schema_version", "episode", "reward", "drift", "model", "verbosity"}
#   error row:   {"schema_version", "task", "error", "drift", "model", "verbosity", "episode"?}  (episode present
#                when the model finished but scoring failed). Readers branch on "error" in row.
# "drift" is the DriftRecord as JSON: {"applied": [{"kind", "endpoint", ...}]}, empty list when the docs were true.
RUN_SCHEMA_VERSION = 3
from .env.schema import ApiSpec
from .env.server import MockServer


def _positive_int(s: str) -> int:
    v = int(s)
    if v < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return v


def _non_negative_int(s: str) -> int:
    v = int(s)
    if v < 0:
        raise argparse.ArgumentTypeError("must be 0 or more")
    return v


def _kinds(s: str) -> list[str]:
    try:
        return parse_kinds(s)
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e)) from None


def drift_for_task(spec: ApiSpec, task: dict, index: int, n: int, kinds: list[str], seed: int) -> tuple[ApiSpec, DriftRecord]:
    """The docs copy for one task: `n` lies aimed at its golden endpoints, seeded per task so adding a task
    elsewhere in the file does not change this one's drift."""
    rng = random.Random(f"{seed}:{index}:{task.get('id')}")
    return apply_drift(spec, n, rng, kinds=kinds, targets=golden_endpoints(spec, task))


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
    p.add_argument("--drift", type=_non_negative_int, default=0, help="lies per task in the docs the model reads (0 = true docs)")
    p.add_argument("--drift-kinds", type=_kinds, default=parse_kinds(None), metavar="KINDS",
                   help="comma-separated subset of: " + ", ".join(parse_kinds(None)))
    p.add_argument("--seed", type=int, default=0, help="seed for drift placement")
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

    token = spec.auth.token if spec.auth else None

    if args.dry_run:
        spec_for_model, record = drift_for_task(spec, tasks[0], 0, args.drift, args.drift_kinds, args.seed)
        print("=== SYSTEM ===")
        print(system_prompt(args.budget))
        print("=== USER ===")
        print(build_user_prompt(spec_for_model, tasks[0], token))
        if record:
            print("=== DRIFT ===")
            print(json.dumps(record.to_json(), indent=2))
        return 0

    model = load_model(args.model)
    out_path = Path(args.out) if args.out else Path("runs") / f"{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Validate every task (prompt and golden path) before spending a single model call.
    for task in tasks:
        try:
            build_user_prompt(spec, task, token)
            expected_outcome(spec, task)
        except Exception as e:
            print(f"task {_task_label(task)}: invalid task for this spec: {e}", file=sys.stderr)
            return 2

    server = MockServer(spec, verbosity=args.verbosity)
    results: list[Reward] = []
    drifted: list[bool] = []
    errors = 0
    with out_path.open("w", encoding="utf-8") as fh:
        for index, task in enumerate(tasks):
            server.reset()
            t0 = time.time()
            ep = None
            spec_for_model, record = drift_for_task(spec, task, index, args.drift, args.drift_kinds, args.seed)
            try:
                ep = run_episode(model, spec, spec_for_model, task, server, budget=args.budget)
                reward = compute_reward(spec, task, ep, server.snapshot())
            except Exception as e:  # one failed episode (API error, bad task) must not lose the run
                errors += 1
                print(f"ERROR {_task_label(task):<22} {type(e).__name__}: {e}")
                row = {"schema_version": RUN_SCHEMA_VERSION, "task": task, "error": f"{type(e).__name__}: {e}",
                       "drift": record.to_json(), "model": model.name, "verbosity": args.verbosity}
                if ep is not None:  # the model finished; keep its transcript even though scoring failed
                    row["episode"] = ep.to_json()
                fh.write(json.dumps(row) + "\n")
                continue
            dt = time.time() - t0
            results.append(reward)
            drifted.append(bool(record))
            mark = "PASS" if reward.success else "FAIL"
            rep = ""
            if reward.repair_candidate:
                rep = " repaired" if reward.success else " unrepaired"
            drift_note = f"  drift={','.join(record.kinds)}" if record else ""
            print(f"{mark}  {_task_label(task):<22} turns={reward.n_turns} reqs={reward.n_requests}  score={reward.score:+.2f}  "
                  f"stop={ep.stop_reason}  {dt:4.1f}s{rep}{drift_note}")
            if args.show:
                for turn in ep.turns:
                    tag = "MODEL " if turn.role == "model" else "SERVER"
                    print(f"    [{tag}] {turn.text.strip()[:600]}")
                print()
            fh.write(json.dumps({"schema_version": RUN_SCHEMA_VERSION, "episode": ep.to_json(), "reward": reward.as_dict(),
                                 "drift": record.to_json(), "model": model.name, "verbosity": args.verbosity}) + "\n")

    print()
    if not results:
        print(f"model={model.name} verbosity={args.verbosity} drift={args.drift}  no episode completed ({errors} errors)")
        print(f"episodes -> {out_path}")
        return 1
    s = summarize(results, drifted)
    print(f"model={model.name} verbosity={args.verbosity} drift={args.drift}  success {s['passed']}/{s['n']} ({100 * s['success_rate']:.0f}%)  "
          f"mean score {s['mean_score']:+.3f}  mean turns {s['mean_turns']:.2f}  mean requests {s['mean_requests']:.2f}"
          + (f"  errors {errors}" if errors else ""))
    if s["drifted"]:
        clean = f"clean {s['clean_passed']}/{s['clean']}" if s["clean"] else "clean: none"
        print(f"{clean}  drifted {s['drifted_passed']}/{s['drifted']} ({100 * s['drifted_rate']:.0f}%)")
    if s["repair_candidates"]:
        print(f"repair-only: {s['repaired']}/{s['repair_candidates']} doable tasks with a rejected request still succeeded "
              f"({100 * s['repair_rate']:.0f}%); no-fit tasks excluded")
    else:
        print("repair-only: no doable task had a rejected request (nothing to repair in this run)")
    print(f"episodes -> {out_path}")
    return 1 if errors else 0  # a half-failed run is not a success for a pipeline reading the exit code


def summarize(results: list[Reward], drifted: list[bool] | None = None) -> dict:
    """The numbers the summary lines print. Pure, so it can be tested with hand-built rewards.

    `drifted[i]` says whether episode i ran on lying docs; omitted means every episode was clean."""
    n = len(results)
    drifted = list(drifted) if drifted is not None else [False] * n
    if len(drifted) != n:
        raise ValueError("drifted must have one flag per result")
    candidates = [r for r in results if r.repair_candidate]
    repaired = sum(r.success for r in candidates)
    on_drift = [r for r, d in zip(results, drifted) if d]
    on_clean = [r for r, d in zip(results, drifted) if not d]
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
        "clean": len(on_clean),
        "clean_passed": sum(r.success for r in on_clean),
        "drifted": len(on_drift),
        "drifted_passed": sum(r.success for r in on_drift),
        "drifted_rate": sum(r.success for r in on_drift) / len(on_drift) if on_drift else None,
    }


if __name__ == "__main__":
    raise SystemExit(main())

# Reading list

Ordered the way the pieces depend on each other. Skip anything already familiar.

## 1. GRPO itself

- [DeepSeekMath](https://arxiv.org/abs/2402.03300), the section that introduces GRPO. The whole algorithm is one equation: sample a group of completions per prompt, normalize rewards within the group to get advantages, apply a PPO-style clipped update plus a KL penalty to a reference model. The point is dropping the value model.
- [DAPO](https://arxiv.org/abs/2503.14476) and [Understanding R1-Zero-Like Training](https://arxiv.org/abs/2503.20783) (Dr. GRPO). Fixes to vanilla GRPO: length bias from per-sequence normalization, the std normalization issue, clip-higher for exploration, dynamic sampling to drop groups where every sample got the same reward. TRL exposes most of these as flags.
- [Nathan Lambert's RLHF book](https://rlhfbook.com), policy gradient and GRPO chapters, for a plain-language grounding.

## 2. Build intuition by running it

- Run a GSM8K GRPO script (Will Brown's gist or the Unsloth notebook) with a 0.5B to 1.5B Qwen model.
- Watch reward per group, completion length, and KL. Change the KL coefficient and the number of generations per prompt and see what moves.
- Break it on purpose: use a binary reward that is almost always zero and watch training stall. That is the failure mode a drift curriculum will produce if the steps are too large.

## 3. Reward design and RLVR practice

- [ToolRL: Reward is All Tool Learning Needs](https://arxiv.org/abs/2504.13958). Decomposed format and correctness rewards. Read the ablations, not just the headline. Also the source of "cold-start GRPO beat SFT-then-GRPO on held-out tools."
- [MUA-RL](https://arxiv.org/pdf/2508.18669). The opposing view: dense format and matching rewards discourage exploration after errors in multi-turn settings; a binary task-completion reward did better. Testing this is part of the project.
- One reward-hacking write-up, to recognize the symptoms: length inflation, format gaming, exploiting a lenient verifier.

## 4. Multi-turn tool RL

- [When the Loop Closes: Diagnosing and Repairing RLVR for Small Tool-Use Agents](https://neurips.cc/virtual/2026/171163) (NeurIPS 2026). Single-turn RLVR gives small agents no gain over SFT; the failure mode is looping until the turn budget runs out, which single-turn rewards never see. Executing the loop during rollouts moved a 2B model from roughly 4% to 59% trajectory completion.
- Mechanical detail to understand before writing a trainer: in multi-turn rollouts, tool responses are in the context but must be masked out of the loss. Only model-generated tokens are trained on. See how verl's multi-turn agent loop or TRL's environment support handles this.
- [PROVE](https://www.alphaxiv.org/abs/2606.03892) and [TIER](https://www.opentrain.ai/papers/tier-trajectory-invariant-execution-rewards-for-multi-step-tool-composition--arxiv-2605.16790/) for execution-based rewards against live or schema-derived environments.

## 5. Stack

- [TRL GRPOTrainer](https://huggingface.co/docs/trl/grpo_trainer), especially the reward function interface and generation parameters.
- [verl](https://github.com/volcengine/verl) architecture overview. Likely the destination once multi-turn rollouts outgrow TRL.
- [Berkeley Function Calling Leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html) for eval categories, including relevance detection (the no-fit case).
- [APIGen](https://arxiv.org/abs/2406.18518) for how to synthesize diverse, verified function-calling data.

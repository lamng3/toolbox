# Roadmap

The toolbox grows by adding components behind the existing interfaces. Planned next:

## Post-training and RL

- [ ] **Rollout engine with weight sync**: generate GRPO rollouts with SGLang/vLLM and push
      updated policy weights after each step, so generation isn't bound by in-process `generate`.
- [ ] Mini-batched policy updates and sequence packing for long completions.
- [ ] KTO (unpaired preferences), ORPO and SimPO (reference-free) objectives.
- [ ] PPO for LLMs with a value head and token-level GAE.
- [ ] Multi-turn / tool-using agent rollouts. Each turn becomes an `ActionRecord` and the whole
      episode is credited.
- [ ] Process reward models (step-level rewards).

## Decision models and agents

- [ ] Adaptive rotation budget: read fewer rotations, calibrated without labels.
- [ ] Escalation to chain-of-thought when the one-pass decision is uncertain; distil the CoT
      answers back into one-pass decisions (Tacit-style self-distillation loop).
- [ ] An OpenTelemetry exporter for `ActionRecord`s.
- [ ] Vector-database `RecordRepository` backends for memory.

## Infrastructure

- [ ] KV cache + continuous batching in the reference `torch` engine (an inference-optimisation
      teaching track).
- [ ] DeepSpeed ZeRO strategy; FSDP2.
- [ ] Hydra multirun + submitit sweeps.
- [ ] Evaluation harness integration (lm-eval-harness tasks as evaluators).

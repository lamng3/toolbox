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
- [ ] Escalation to chain-of-thought when the one-pass decision is uncertain, with the CoT
      answers distilled back into one-pass decisions (Tacit-style loop on top of
      [`examples/jev`](examples/jev.md)).
- [ ] Vector-database `RecordRepository` backends for memory.

## More standard libraries as backends

- [x] PyTorch Geometric layers and models, TRL trainers, AnyJev readout, `torch.optim` /
      `transformers` schedules, sentence-transformers embeddings, any `nn.Module`.
- [ ] Accelerate / DeepSpeed ZeRO as distribution strategies; FSDP2.
- [ ] OpenTelemetry exporter for agent traces.
- [ ] lm-evaluation-harness tasks as evaluators.
- [ ] Math verifiers (e.g. `math-verify`) as rewards.

## Infrastructure

- [ ] KV cache + continuous batching in the reference `torch` engine (an inference-optimisation
      teaching track).
- [ ] Hydra multirun + submitit sweeps.

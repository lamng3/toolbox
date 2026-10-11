# Agent tracing

Agents make many small decisions: route a ticket, call a tool, pick an option, write a reply.
`toolbox.tracing` records each one as an **atomic action** with a uniform schema. That gives you
(1) an audit and profiling trail, and (2) a training set.

## The action record

| Field | Meaning |
| --- | --- |
| `kind` | `decision` · `generation` · `tool_call` · `env_step` · `step` |
| `agent` | logical actor name (`router`, `planner`, `search`) |
| `state` | what the agent saw (prompt, context, observation) |
| `action` | what it did (chosen option, text, tool arguments, return value) |
| `options`, `probs` | candidates and the model's probability for each (decisions) |
| `reward`, `outcome` | feedback, attached immediately or later |
| `trace_id`, `parent_id` | episode → turn → tool-call tree |
| `latency_ms`, `tokens`, `model`, `error` | profiling and failure information |
| `metadata` | anything else (prompt version, temperature, git SHA) |

## Recording

```python
from toolbox.tracing import JsonlTraceStore, Tracer

tracer = Tracer(JsonlTraceStore("runs/agent/traces"))

with tracer.span("step", agent="assistant", state=user_message):        # parent
    with tracer.span("decision", agent="router", state=user_message,
                     options=["billing", "fraud"]) as rec:               # child
        decision = reader.decide(user_message, question)
        rec.action, rec.probs = decision.answer, decision.probs

    @tracer.traced("tool_call", agent="search")
    def search(query: str) -> list[str]: ...
    search("refund policy")                                              # child, auto-logged

# later, when the outcome is known
tracer.feedback(rec.id, reward=1.0, outcome="resolved")
```

Spans nest automatically through `contextvars`, so this works across threads and asyncio
tasks. Errors raised inside a span are recorded (`error="TimeoutError: ..."`) and re-raised.
Latency is measured for every span.

Ready-made wrappers:

* `TracedEngine(engine, tracer)` logs every LLM call made through any inference engine.
* `TracedDecisionReader(reader, tracer)` logs every decision with its probabilities and
  returns the action id for later feedback.
* The `trace_rollouts` callback logs every RL rollout with its reward.

## From traces to training

```bash
toolbox traces runs/agent/traces            # counts, error rate, mean reward, latency per agent/kind
```

| `data.mode=` | Built from | Train with |
| --- | --- | --- |
| `sft` | `state → action`, optionally only `reward ≥ min_reward` | `objective=sft` (behaviour cloning / expert iteration) |
| `decision` | decision records (with `probs` as soft targets) | `decision_ce`, `decision_distill` |
| `preference` | same-state actions with different rewards → (chosen, rejected) | `dpo`, `reward_bt` |

```bash
toolbox train experiment=sft_from_traces data.path=runs/agent/traces data.min_reward=1
toolbox train experiment=decision_tiny data=traces data.path=runs/agent/traces data.mode=decision
toolbox train experiment=dpo_tiny data=traces data.path=runs/agent/traces data.mode=preference
```

`python -m examples.recipes.agent_tracing` runs the whole loop: act, give feedback, train on the rewarded
decisions.

## Storage

`TraceStore` is a repository over records: JSONL on disk by default, in memory for tests, or
any `RecordRepository` you implement (S3, a database). The log is **append-only**. Feedback
re-emits the record with the same `id`, and `latest_by_id()` resolves the newest version, so
concurrent writers never rewrite files.

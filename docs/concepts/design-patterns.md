# Design patterns

toolbox is meant to be read as well as used. Each pattern below appears where it solves a real
problem in a training stack. The links go to the code.

| # | Pattern | Where | The problem it solves |
| --- | --- | --- | --- |
| 1 | Replace conditional with **Strategy** | [`objectives/`](https://github.com/lamng3/toolbox/tree/main/src/toolbox/objectives), [`rl/advantage.py`](https://github.com/lamng3/toolbox/blob/main/src/toolbox/rl/advantage.py), [`distributed/`](https://github.com/lamng3/toolbox/tree/main/src/toolbox/distributed), [`rewards/`](https://github.com/lamng3/toolbox/tree/main/src/toolbox/rewards) | No `if loss_type == "dpo": ... elif ...` chains in trainers. Losses, advantages, rewards and distribution are swappable objects. |
| 2 | **Refactor a long function** | [`trainer/base.py`](https://github.com/lamng3/toolbox/blob/main/src/toolbox/trainer/base.py) `fit()` | A typical 300-line training loop becomes `_prepare_run → _run_one_step → _maybe_evaluate → _maybe_save → _finish`, each small and testable. |
| 3 | **Remove magic numbers** | [`core/constants.py`](https://github.com/lamng3/toolbox/blob/main/src/toolbox/core/constants.py) | `-100`, `1e-8`, `10000.0` (RoPE base), `0.2` (clip) have names and docstrings explaining them. |
| 4 | **Factory** | [`core/registry.py`](https://github.com/lamng3/toolbox/blob/main/src/toolbox/core/registry.py) `Registry.build`, [`trainer/optim.py`](https://github.com/lamng3/toolbox/blob/main/src/toolbox/trainer/optim.py) | Config `{"name": "adamw", "lr": 3e-4}` → object, with no import paths in configs. |
| 5 | **Dependency injection** | [`experiment.py`](https://github.com/lamng3/toolbox/blob/main/src/toolbox/experiment.py), trainer constructors | Trainers receive the model, optimizer, data, callbacks and strategy. They never construct them, so tests can pass fakes. `Registry.build(inject=...)` passes a dependency only if the factory accepts it. |
| 6 | Replace conditional with **polymorphism** | `CausalLM`, `InferenceEngine`, `Env`, `Tokenizer` | A from-scratch GPT and a Hugging Face model, or SGLang and vLLM, are interchangeable behind one interface. |
| 7 | **Retry logic refactor** | [`core/retry.py`](https://github.com/lamng3/toolbox/blob/main/src/toolbox/core/retry.py) | One `RetryPolicy` (exponential backoff + jitter) for Hub downloads, SLURM submission and HTTP inference, instead of ad-hoc loops. The `sleep` function is injectable, so tests run instantly. |
| 8 | **Observer** | [`core/events.py`](https://github.com/lamng3/toolbox/blob/main/src/toolbox/core/events.py), [`trainer/callbacks.py`](https://github.com/lamng3/toolbox/blob/main/src/toolbox/trainer/callbacks.py) | The trainer emits `STEP_END`, `EVAL_END`, `ROLLOUT_END`, ...; loggers, early stopping, profilers, trace and memory writers subscribe. New side effects need no loop edits. |
| 9 | **Plugin architecture** | `Registry` + `toolbox.plugins` entry points | External packages add models, losses or rewards by being installed. Nothing in toolbox changes. |
| 10 | **Repository** | [`data/repository.py`](https://github.com/lamng3/toolbox/blob/main/src/toolbox/data/repository.py), [`trainer/checkpoint.py`](https://github.com/lamng3/toolbox/blob/main/src/toolbox/trainer/checkpoint.py), [`tracing/tracer.py`](https://github.com/lamng3/toolbox/blob/main/src/toolbox/tracing/tracer.py) `TraceStore`, [`memory/`](https://github.com/lamng3/toolbox/tree/main/src/toolbox/memory) | Datasets, checkpoints, traces and memories go through one interface: in-memory, JSONL, HF Hub, or your own S3. |
| 11 | **Singleton** | [`core/logging.py`](https://github.com/lamng3/toolbox/blob/main/src/toolbox/core/logging.py) `RunContext`, [`tracing/tracer.py`](https://github.com/lamng3/toolbox/blob/main/src/toolbox/tracing/tracer.py) `get_tracer` | One run context and one default tracer per process. Both are thread-safe and resettable, which avoids the usual test-pollution problem of singletons. |
| 12 | **Decorator** | `@register`, `@retry`, `@timed`, `@rank_zero_only`, `@tracer.traced`, `TracedEngine`, `TracedDecisionReader` | Cross-cutting behaviour (registration, retries, timing, rank filtering, tracing) wraps code without editing it. |
| 13 | **Adapter** | [`nn/models/hf.py`](https://github.com/lamng3/toolbox/blob/main/src/toolbox/nn/models/hf.py), [`serving/engines.py`](https://github.com/lamng3/toolbox/blob/main/src/toolbox/serving/engines.py), `GymnasiumAdapter`, `pyg_to_graph`, `HFTokenizer` | Translates transformers / SGLang / vLLM / Gymnasium / PyG APIs into toolbox interfaces. |
| 14 | **Builder** | [`experiment.py`](https://github.com/lamng3/toolbox/blob/main/src/toolbox/experiment.py) `ExperimentBuilder` | Assembles a dozen interdependent components step by step, either fluently from Python or from a config. |
| 15 | **Template method** | [`trainer/base.py`](https://github.com/lamng3/toolbox/blob/main/src/toolbox/trainer/base.py) `BaseTrainer.fit` | The loop skeleton is fixed. Subclasses override `compute_loss` or `train_step` (SFT vs DPO vs GRPO vs PPO). |
| 16 | **Command** | [`launch/__init__.py`](https://github.com/lamng3/toolbox/blob/main/src/toolbox/launch/__init__.py) | `LocalTrainCommand`, `TorchrunCommand` and `SlurmSubmitCommand` support `describe()` (dry run), `execute()` and `undo()` (`scancel`). `CommandInvoker` keeps history for sweeps. |

Composite also appears, in `WeightedSumReward`, which builds child rewards from config.

## Examples

=== "Strategy"

    ```python
    # trainer code: no knowledge of which loss is active
    out = self.objective(logprobs, old_logprobs, advantages, mask, ref_logprobs)

    # config decides
    # objective=grpo  | objective=ppo_clip  | objective=reinforce
    ```

=== "Template method"

    ```python
    class MyTrainer(BaseTrainer):
        def compute_loss(self, batch):          # the only thing that differs
            return my_loss(self.model, batch)
    ```

=== "Observer"

    ```python
    class PrintReward(Callback):
        def on_rollout_end(self, trainer, rollout, rewards):
            print(trainer.step, rewards.mean())
    ```

=== "Command"

    ```python
    cmd = SlurmSubmitCommand(["experiment=sft_qwen"], partition="gpu")
    print(cmd.describe())    # dry run
    cmd.execute()            # sbatch
    cmd.undo()               # scancel
    ```

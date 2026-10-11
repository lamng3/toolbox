"""TRL as a post-training backend (pattern: *Adapter*).

`TRL <https://github.com/huggingface/trl>`_ is the standard library for post-training Hugging
Face models. This trainer hands the optimisation to TRL's own ``SFTTrainer``, ``DPOTrainer``,
``GRPOTrainer`` or ``RewardTrainer``. Toolbox still provides everything around it: configs and
overrides, data sources (JSONL, ``builtin://``, ``hf://``), toolbox rewards for GRPO, LoRA
settings from the model spec, toolbox callbacks (console, JSONL, W&B ...) and the launchers
(torchrun, SLURM).

.. code-block:: yaml

    model:   {name: hf_causal_lm, model_name_or_path: Qwen/Qwen2.5-0.5B-Instruct, lora: {r: 16}}
    data:    {name: prompts, path: "hf://openai/gsm8k?config=main&split=train",
              prompt_key: question, answer_key: answer}
    reward:  {name: exact_match}
    trainer: {name: trl, algorithm: grpo, max_steps: 500, batch_size: 8,
              args: {num_generations: 8, max_completion_length: 256, beta: 0.04}}

``args`` is passed to the TRL config class unchanged, so every TRL option is available.
Toolbox's own trainers (``supervised``, ``preference``, ``grpo``) stay available as readable,
hackable reference implementations.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import fields
from typing import TYPE_CHECKING, Any

from toolbox.core.constants import DEFAULT_SEED
from toolbox.core.decorators import require
from toolbox.core.events import Event, EventBus
from toolbox.core.logging import get_logger
from toolbox.core.registry import TRAINERS
from toolbox.core.types import Metrics
from toolbox.data.datasets.base import train_eval_split
from toolbox.data.repository import Record, load_records
from toolbox.trainer.callbacks import Callback
from toolbox.trainer.config import TrainerConfig

if TYPE_CHECKING:
    from toolbox.experiment import ExperimentBuilder
    from toolbox.rewards import Reward

log = get_logger("trl")

HF_MODEL = "hf_causal_lm"
DEFAULT_EVAL_FRACTION = 0.1


class Algorithm:
    """What each TRL algorithm needs: its trainer/config classes and record format."""

    def __init__(
        self, trainer: str, config: str, columns: Callable[[Record, Mapping[str, Any]], Record]
    ):
        self.trainer, self.config, self.columns = trainer, config, columns


def _text(record: Record, key: str) -> str:
    return str(record[key])


def _sft_columns(r: Record, data: Mapping[str, Any]) -> Record:
    prompt = _text(r, data.get("prompt_key", "prompt"))
    response = _text(r, data.get("response_key", "response"))
    if data.get("chat_template"):
        return {
            "prompt": [{"role": "user", "content": prompt}],
            "completion": [{"role": "assistant", "content": response}],
        }
    return {"prompt": prompt, "completion": response}


def _pair_columns(r: Record, data: Mapping[str, Any]) -> Record:
    prompt = _text(r, data.get("prompt_key", "prompt"))
    if data.get("chat_template"):
        return {
            "prompt": [{"role": "user", "content": prompt}],
            "chosen": [{"role": "assistant", "content": _text(r, "chosen")}],
            "rejected": [{"role": "assistant", "content": _text(r, "rejected")}],
        }
    return {"prompt": prompt, "chosen": _text(r, "chosen"), "rejected": _text(r, "rejected")}


def _prompt_columns(r: Record, data: Mapping[str, Any]) -> Record:
    prompt = _text(r, data.get("prompt_key", "prompt"))
    answer = str(r.get(data.get("answer_key", "answer"), ""))
    if data.get("chat_template"):
        return {"prompt": [{"role": "user", "content": prompt}], "answer": answer}
    return {"prompt": prompt, "answer": answer}


ALGORITHMS: dict[str, Algorithm] = {
    "sft": Algorithm("SFTTrainer", "SFTConfig", _sft_columns),
    "dpo": Algorithm("DPOTrainer", "DPOConfig", _pair_columns),
    "reward": Algorithm("RewardTrainer", "RewardConfig", _pair_columns),
    "grpo": Algorithm("GRPOTrainer", "GRPOConfig", _prompt_columns),
}


def trl_reward(reward: Reward, name: str) -> Callable[..., list[float]]:
    """Adapt a toolbox :class:`~toolbox.rewards.Reward` to TRL's reward-function signature.

    TRL calls ``fn(prompts, completions, **dataset_columns)``. The dataset's ``answer`` column
    becomes the reference, and chat-format completions are reduced to their text.
    """

    def as_text(item: Any) -> str:
        if isinstance(item, list):  # conversational: [{"role": ..., "content": ...}, ...]
            return "".join(str(m.get("content", "")) for m in item)
        return str(item)

    def reward_fn(prompts: Sequence[Any], completions: Sequence[Any], **kwargs: Any) -> list[float]:
        references = kwargs.get("answer") or [""] * len(completions)
        scores = reward(
            [as_text(p) for p in prompts],
            [as_text(c) for c in completions],
            [str(r) for r in references],
        )
        return [float(s) for s in scores]

    reward_fn.__name__ = name
    return reward_fn


@TRAINERS.register("trl")
class TRLTrainer:
    """Runs a TRL trainer from toolbox components. Call :meth:`fit` like any toolbox trainer."""

    def __init__(
        self,
        algorithm: str,
        model: Any,
        train_records: list[Record],
        eval_records: list[Record],
        config: TrainerConfig,
        args: Mapping[str, Any] | None = None,
        reward_funcs: Sequence[Callable[..., list[float]]] = (),
        peft_config: Any = None,
        tokenizer: Any = None,
        callbacks: Sequence[Callback] = (),
        learning_rate: float | None = None,
        seed: int = DEFAULT_SEED,
    ) -> None:
        if algorithm not in ALGORITHMS:
            raise ValueError(f"algorithm must be one of {sorted(ALGORITHMS)}, got {algorithm!r}")
        self.algorithm, self.config = algorithm, config
        self.model, self.peft_config, self.tokenizer = model, peft_config, tokenizer
        self.train_records, self.eval_records = train_records, eval_records
        self.reward_funcs = list(reward_funcs)
        self.user_args = dict(args or {})
        self.learning_rate, self.seed = learning_rate, seed
        self.events = EventBus()
        for callback in callbacks:
            callback.attach(self.events)
        self.step = 0
        self.last_metrics: Metrics = {}
        self.run_config: dict[str, Any] = {}

    @property
    def is_main_process(self) -> bool:
        return True

    # -- assembly from toolbox config ------------------------------------------------------
    @classmethod
    def from_builder(
        cls,
        b: ExperimentBuilder,
        config: TrainerConfig,
        algorithm: str = "sft",
        args: Mapping[str, Any] | None = None,
    ) -> TRLTrainer:
        model_spec = dict(b.specs["model"] or {})
        data_spec = dict(b.specs["data"] or {})
        model, peft_config = cls._model(model_spec)
        records = [
            ALGORITHMS[algorithm].columns(r, data_spec) for r in load_records(data_spec["path"])
        ]
        train, eval_ = train_eval_split(
            records, float(data_spec.get("eval_fraction", DEFAULT_EVAL_FRACTION)), b.seed
        )
        rewards = (
            [trl_reward(b.reward(), str(b.specs["reward"]["name"]))] if b.specs["reward"] else []
        )
        optimizer = b.specs.get("optimizer") or {}
        trainer = cls(
            algorithm,
            model,
            train,
            eval_,
            config,
            args=args,
            reward_funcs=rewards,
            peft_config=peft_config,
            callbacks=b.callbacks(),
            learning_rate=optimizer.get("lr"),
            seed=b.seed,
        )
        trainer.run_config = b.run_config
        return trainer

    @staticmethod
    def _model(spec: dict[str, Any]) -> tuple[Any, Any]:
        """HF model specs are passed to TRL by name (TRL loads them); LoRA becomes a PEFT config."""
        if spec.get("name") != HF_MODEL:
            raise ValueError("the trl trainer needs `model: {name: hf_causal_lm, ...}`")
        lora = spec.get("lora")
        peft_config = require("peft", extra="llm").LoraConfig(**lora) if lora else None
        return spec["model_name_or_path"], peft_config

    # -- running ---------------------------------------------------------------------------
    def _trl_args(self, trl: Any) -> Any:
        config_cls = getattr(trl, ALGORITHMS[self.algorithm].config)
        known = {f.name for f in fields(config_cls)}
        c = self.config
        base: dict[str, Any] = {
            "output_dir": c.output_dir,
            "max_steps": c.max_steps,
            "per_device_train_batch_size": c.batch_size,
            "gradient_accumulation_steps": c.grad_accum_steps,
            "max_grad_norm": c.grad_clip,
            "logging_steps": max(c.log_every, 1),
            "save_strategy": "steps" if c.save_every else "no",
            "save_steps": c.save_every or c.max_steps,
            "bf16": c.precision == "bf16",
            "fp16": c.precision == "fp16",
            "use_cpu": c.device == "cpu",
            "seed": self.seed,
            "report_to": "none",  # logging goes through toolbox callbacks
        }
        if self.learning_rate is not None:
            base["learning_rate"] = self.learning_rate
        if self.algorithm == "grpo":  # TRL needs the generation batch to divide into groups
            base["num_generations"] = self.user_args.get("num_generations", c.batch_size)
        base.update(self.user_args)
        unknown = sorted(set(base) - known)
        if unknown:
            raise ValueError(f"{config_cls.__name__} has no option(s) {unknown}")
        return config_cls(**base)

    def _bridge(self, transformers: Any) -> Any:
        """A ``transformers`` callback that forwards TRL's logs to toolbox callbacks."""
        outer = self

        class ToolboxBridge(transformers.TrainerCallback):
            def on_log(
                self, args: Any, state: Any, control: Any, logs: Any = None, **kw: Any
            ) -> None:
                metrics = {
                    k: float(v) for k, v in (logs or {}).items() if isinstance(v, int | float)
                }
                outer.step = int(state.global_step)
                outer.last_metrics = {**outer.last_metrics, **metrics}
                event = (
                    Event.EVAL_END
                    if any(k.startswith("eval_") for k in metrics)
                    else Event.STEP_END
                )
                outer.events.emit(event, trainer=outer, step=outer.step, metrics=metrics)

        return ToolboxBridge()

    def fit(self) -> Metrics:
        trl = require("trl", extra="llm")
        datasets = require("datasets", extra="llm")
        transformers = require("transformers", extra="llm")
        algo = ALGORITHMS[self.algorithm]
        kwargs: dict[str, Any] = {
            "model": self.model,
            "args": self._trl_args(trl),
            "train_dataset": datasets.Dataset.from_list(self.train_records),
            "callbacks": [self._bridge(transformers)],
            "peft_config": self.peft_config,
        }
        if self.algorithm == "grpo":
            if not self.reward_funcs:
                raise ValueError("algorithm=grpo needs a `reward` section")
            kwargs["reward_funcs"] = self.reward_funcs
        trainer = getattr(trl, algo.trainer)(**kwargs)
        self.events.emit(Event.FIT_START, trainer=self)
        trainer.train()
        trainer.save_model(f"{self.config.output_dir}/final")
        self.events.emit(Event.FIT_END, trainer=self, metrics=self.last_metrics)
        log.info("TRL %s finished; model saved to %s/final", self.algorithm, self.config.output_dir)
        return self.last_metrics

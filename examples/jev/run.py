"""Train a Jev-style decision model: AnyJev for readout and evaluation, toolbox for training.

    python -m examples.jev.run                                   # Qwen2.5-0.5B-Instruct, L0 teacher
    python -m examples.jev.run --teacher cot --samples 4         # chain-of-thought teacher
    python -m examples.jev.run --student-model Qwen/Qwen2.5-0.5B-Instruct \\
        --teacher-model Qwen/Qwen2.5-7B-Instruct --data data/processed/decisions.jsonl

Pipeline (each stage caches its output under ``--out``; delete a file to redo that stage):

    1. teacher   label every item with a decision distribution, no human labels needed
                 (teacher.jsonl)
    2. baseline  evaluate the base model through AnyJev: raw (1 forward) and L0 (K forwards)
    3. train     distill the teacher into the student with toolbox (LoRA, decision_distill,
                 AnyJev prompts, random option permutations)
    4. evaluate  the trained student through AnyJev, raw and L0
    5. report    table + results.json

The goal of the student: raw (one forward pass) ≈ teacher's quality, with ``K``× fewer passes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from anyjev.backends.hf import HFBackend
from rich.console import Console
from rich.table import Table

import examples.jev.data  # noqa: F401  registers the `anyjev_decisions` dataset
from examples.jev.evaluate import evaluate
from examples.jev.teacher import cot_teacher, l0_teacher
from toolbox.core.env import resolve_device
from toolbox.data.datasets.base import train_eval_split
from toolbox.data.repository import JsonlRepository, Record, load_records
from toolbox.experiment import ExperimentBuilder, load_config

HERE = Path(__file__).parent
EXPERIMENT = "jev_student"
DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
EVAL_FRACTION = 0.2
SEED = 42


def backend(model: str, device: str) -> HFBackend:
    return HFBackend(model, device=device, dtype="float32")


def save_student(trainer: Any, tokenizer: Any, path: Path) -> None:
    """Merge LoRA weights (if any) and save a plain Hugging Face model AnyJev can load."""
    hf_model = trainer.unwrapped_model.model
    if hasattr(hf_model, "merge_and_unload"):
        hf_model = hf_model.merge_and_unload()
    hf_model.save_pretrained(path)
    tokenizer.save_pretrained(path)


def stage_teacher(args: argparse.Namespace, records: list[Record], device: str) -> list[Record]:
    path = args.out / "teacher.jsonl"
    if path.exists():
        return load_records(path)
    teacher = backend(args.teacher_model or args.student_model, device)
    if args.teacher == "l0":
        labelled = l0_teacher(records, teacher)
    else:
        labelled = cot_teacher(records, teacher.model, teacher.tokenizer, samples=args.samples)
    JsonlRepository(args.out).write("teacher", labelled)
    return labelled


def stage_train(args: argparse.Namespace, device: str) -> Path:
    student_dir = args.out / "student"
    if (student_dir / "config.json").exists():
        return student_dir
    cfg = load_config(
        [
            f"experiment={EXPERIMENT}",
            f"output_dir={args.out / 'train'}",
            f"data.path={args.out / 'teacher.jsonl'}",
            f"data.eval_fraction={EVAL_FRACTION}",
            f"model.model_name_or_path={args.student_model}",
            f"tokenizer.name_or_path={args.student_model}",
            f"trainer.device={device}",
            f"trainer.max_steps={args.steps}",
            *args.overrides,
        ],
        config_dir=HERE / "configs",
    )
    builder = ExperimentBuilder.from_config(cfg)
    trainer = builder.build()
    trainer.fit()
    save_student(trainer, builder.tokenizer().hf_tokenizer, student_dir)
    return student_dir


def report(rows: list[tuple[str, dict[str, float]]], out: Path) -> None:
    cols = ["accuracy", "teacher_agreement", "flip_rate", "ece", "forward_passes"]
    table = Table(title="Jev decision model: AnyJev readouts before and after distillation")
    table.add_column("model / readout")
    for c in cols:
        table.add_column(c, justify="right")
    for name, metrics in rows:
        table.add_row(name, *[f"{metrics[c]:.3f}" if c in metrics else "-" for c in cols])
    Console(width=160).print(table)
    (out / "results.json").write_text(json.dumps(dict(rows), indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("overrides", nargs="*", help="toolbox overrides for the student training")
    parser.add_argument("--data", default="builtin://decisions.jsonl")
    parser.add_argument("--student-model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--teacher-model", default=None, help="default: the student model (self-distillation)"
    )
    parser.add_argument("--teacher", choices=("l0", "cot"), default="l0")
    parser.add_argument("--samples", type=int, default=4, help="CoT samples per item (cot teacher)")
    parser.add_argument("--limit", type=int, default=200, help="use the first N items")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", type=Path, default=Path("runs/jev"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    device = str(resolve_device(args.device))

    records = load_records(args.data)[: args.limit]
    labelled = stage_teacher(args, records, device)
    _, held_out = train_eval_split(labelled, EVAL_FRACTION, SEED)  # same split as training

    base = backend(args.student_model, device)
    rows = [
        ("base · raw (1 forward)", evaluate(base, held_out, "raw")),
        ("base · L0 (K forwards)", evaluate(base, held_out, "L0")),
    ]
    del base

    student = backend(str(stage_train(args, device)), device)
    rows += [
        ("student · raw (1 forward)", evaluate(student, held_out, "raw")),
        ("student · L0 (K forwards)", evaluate(student, held_out, "L0")),
    ]
    report(rows, args.out)
    print(f"student → {args.out / 'student'}   results → {args.out / 'results.json'}")


if __name__ == "__main__":
    main()

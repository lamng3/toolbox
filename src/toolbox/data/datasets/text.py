"""Language-model datasets: pretraining text, SFT pairs, preference pairs, RL prompts."""

from __future__ import annotations

from typing import Any

from toolbox.core.constants import DEFAULT_SEED, IGNORE_INDEX
from toolbox.core.registry import DATASETS
from toolbox.data.collators import PaddingCollator, list_collate
from toolbox.data.datasets.base import DataBundle, ListDataset, train_eval_split
from toolbox.data.repository import Record, load_records, resolve_path
from toolbox.data.tokenization import Tokenizer

DOC_SEPARATOR = "\n\n"


@DATASETS.register("text_lm")
def text_lm(
    tokenizer: Tokenizer,
    path: str = "builtin://tiny_corpus.txt",
    block_size: int = 128,
    eval_fraction: float = 0.1,
    seed: int = DEFAULT_SEED,
) -> DataBundle:
    """Pack a plain-text corpus into fixed-length blocks for next-token prediction.

    Documents (separated by blank lines) are wrapped in BOS/EOS, concatenated, and cut into
    non-overlapping windows of ``block_size`` tokens — the standard GPT pretraining recipe.
    """
    text = resolve_path(path).read_text(encoding="utf-8")
    stream: list[int] = []
    for doc in filter(None, (d.strip() for d in text.split(DOC_SEPARATOR))):
        stream.extend(tokenizer.encode(doc, add_bos=True, add_eos=True))
    blocks = [
        {"input_ids": stream[i : i + block_size], "labels": stream[i : i + block_size]}
        for i in range(0, max(len(stream) - block_size, 1), block_size)
    ]
    train, eval_ = train_eval_split(blocks, eval_fraction, seed)
    return DataBundle(
        ListDataset(train), ListDataset(eval_) or None, PaddingCollator(tokenizer.pad_token_id)
    )


def _render_prompt(tokenizer: Tokenizer, record: Record, prompt_key: str, chat: bool) -> str:
    prompt = str(record[prompt_key])
    return tokenizer.apply_chat_template([{"role": "user", "content": prompt}]) if chat else prompt


def encode_prompt_response(
    tokenizer: Tokenizer, prompt: str, response: str, max_length: int
) -> dict[str, list[int]]:
    """Tokenise a prompt/response pair; prompt tokens are masked out of the loss."""
    prompt_ids = tokenizer.encode(prompt, add_bos=True)
    response_ids = tokenizer.encode(response, add_eos=True)
    input_ids = (prompt_ids + response_ids)[:max_length]
    labels = ([IGNORE_INDEX] * len(prompt_ids) + response_ids)[:max_length]
    return {"input_ids": input_ids, "labels": labels}


@DATASETS.register("sft")
def sft(
    tokenizer: Tokenizer,
    path: str = "builtin://sft.jsonl",
    prompt_key: str = "prompt",
    response_key: str = "response",
    max_length: int = 256,
    chat_template: bool = False,
    eval_fraction: float = 0.1,
    seed: int = DEFAULT_SEED,
) -> DataBundle:
    """Supervised fine-tuning on ``{"prompt": ..., "response": ...}`` JSONL records."""
    examples = [
        encode_prompt_response(
            tokenizer,
            _render_prompt(tokenizer, r, prompt_key, chat_template),
            str(r[response_key]),
            max_length,
        )
        for r in load_records(path)
    ]
    train, eval_ = train_eval_split(examples, eval_fraction, seed)
    return DataBundle(
        ListDataset(train), ListDataset(eval_) or None, PaddingCollator(tokenizer.pad_token_id)
    )


@DATASETS.register("preference")
def preference(
    tokenizer: Tokenizer,
    path: str = "builtin://preference.jsonl",
    max_length: int = 256,
    chat_template: bool = False,
    eval_fraction: float = 0.1,
    seed: int = DEFAULT_SEED,
) -> DataBundle:
    """Preference pairs ``{"prompt", "chosen", "rejected"}`` for DPO / reward modelling."""
    examples: list[dict[str, Any]] = []
    for r in load_records(path):
        prompt = _render_prompt(tokenizer, r, "prompt", chat_template)
        chosen = encode_prompt_response(tokenizer, prompt, str(r["chosen"]), max_length)
        rejected = encode_prompt_response(tokenizer, prompt, str(r["rejected"]), max_length)
        examples.append(
            {
                "chosen_input_ids": chosen["input_ids"],
                "chosen_labels": chosen["labels"],
                "rejected_input_ids": rejected["input_ids"],
                "rejected_labels": rejected["labels"],
            }
        )
    train, eval_ = train_eval_split(examples, eval_fraction, seed)
    return DataBundle(
        ListDataset(train), ListDataset(eval_) or None, PaddingCollator(tokenizer.pad_token_id)
    )


@DATASETS.register("prompts")
def prompts(
    tokenizer: Tokenizer,
    path: str = "builtin://arithmetic.jsonl",
    prompt_key: str = "prompt",
    answer_key: str = "answer",
    chat_template: bool = False,
    eval_fraction: float = 0.1,
    seed: int = DEFAULT_SEED,
) -> DataBundle:
    """Prompts (+ reference answers) for online RL; batches are lists of strings."""
    examples = [
        {
            "prompt": _render_prompt(tokenizer, r, prompt_key, chat_template),
            "answer": str(r.get(answer_key, "")),
        }
        for r in load_records(path)
    ]
    train, eval_ = train_eval_split(examples, eval_fraction, seed)
    return DataBundle(ListDataset(train), ListDataset(eval_) or None, list_collate)

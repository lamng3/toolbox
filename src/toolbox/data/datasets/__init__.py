"""Built-in dataset factories (registered in ``DATASETS``)."""

from toolbox.data.datasets.base import DataBundle, ListDataset, train_eval_split
from toolbox.data.datasets.graph import planetoid, stochastic_block_model, synthetic_graph
from toolbox.data.datasets.text import encode_prompt_response, preference, prompts, sft, text_lm

__all__ = [
    "DataBundle",
    "ListDataset",
    "encode_prompt_response",
    "planetoid",
    "preference",
    "prompts",
    "sft",
    "stochastic_block_model",
    "synthetic_graph",
    "text_lm",
    "train_eval_split",
]

"""Tokenizers behind one small interface.

:class:`ByteTokenizer` needs no download and no training, which makes every tiny experiment
runnable offline on a laptop. :class:`HFTokenizer` adapts any Hugging Face tokenizer to the same
interface (pattern: *Adapter*).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

from toolbox.core.constants import BYTE_VOCAB_SIZE
from toolbox.core.decorators import require
from toolbox.core.registry import TOKENIZERS


class Tokenizer(ABC):
    """Minimal tokenizer contract used by datasets, rollouts and serving."""

    pad_token_id: int
    bos_token_id: int | None
    eos_token_id: int | None

    @property
    @abstractmethod
    def vocab_size(self) -> int: ...

    @abstractmethod
    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]: ...

    @abstractmethod
    def decode(self, ids: Sequence[int], skip_special_tokens: bool = True) -> str: ...

    def apply_chat_template(self, messages: list[dict[str, str]]) -> str:
        """Render chat messages to a prompt string. Override for model-specific templates."""
        rendered = "".join(f"<|{m['role']}|>\n{m['content']}\n" for m in messages)
        return rendered + "<|assistant|>\n"

    def batch_decode(self, batch: Sequence[Sequence[int]]) -> list[str]:
        return [self.decode(ids) for ids in batch]


@TOKENIZERS.register("byte")
class ByteTokenizer(Tokenizer):
    """UTF-8 bytes as tokens (ids 0–255) plus ``<pad>``, ``<bos>``, ``<eos>``."""

    def __init__(self) -> None:
        self._bos, self._eos = BYTE_VOCAB_SIZE + 1, BYTE_VOCAB_SIZE + 2
        self.pad_token_id = BYTE_VOCAB_SIZE
        self.bos_token_id = self._bos
        self.eos_token_id = self._eos
        self._special = {self.pad_token_id, self._bos, self._eos}

    @property
    def vocab_size(self) -> int:
        return BYTE_VOCAB_SIZE + len(self._special)

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        ids = list(text.encode("utf-8"))
        if add_bos:
            ids.insert(0, self._bos)
        if add_eos:
            ids.append(self._eos)
        return ids

    def decode(self, ids: Sequence[int], skip_special_tokens: bool = True) -> str:
        if skip_special_tokens:
            ids = [i for i in ids if i < BYTE_VOCAB_SIZE]
        else:
            ids = [i if i < BYTE_VOCAB_SIZE else ord("?") for i in ids]
        return bytes(ids).decode("utf-8", errors="replace")


@TOKENIZERS.register("hf")
class HFTokenizer(Tokenizer):
    """Adapter over ``transformers.AutoTokenizer``."""

    def __init__(self, name_or_path: str, trust_remote_code: bool = False) -> None:
        transformers = require("transformers", extra="llm")
        self._tok = transformers.AutoTokenizer.from_pretrained(
            name_or_path, trust_remote_code=trust_remote_code
        )
        if self._tok.pad_token_id is None:
            self._tok.pad_token = self._tok.eos_token
        self.pad_token_id = self._tok.pad_token_id
        self.bos_token_id = self._tok.bos_token_id
        self.eos_token_id = self._tok.eos_token_id

    @property
    def vocab_size(self) -> int:
        return len(self._tok)

    @property
    def hf_tokenizer(self) -> Any:
        """The wrapped ``transformers`` tokenizer, for libraries that need it directly."""
        return self._tok

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        ids: list[int] = self._tok.encode(text, add_special_tokens=False)
        if add_bos and self.bos_token_id is not None:
            ids.insert(0, self.bos_token_id)
        if add_eos and self.eos_token_id is not None:
            ids.append(self.eos_token_id)
        return ids

    def decode(self, ids: Sequence[int], skip_special_tokens: bool = True) -> str:
        return str(self._tok.decode(list(ids), skip_special_tokens=skip_special_tokens))

    def apply_chat_template(self, messages: list[dict[str, str]]) -> str:
        if getattr(self._tok, "chat_template", None):
            return str(
                self._tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            )
        return super().apply_chat_template(messages)

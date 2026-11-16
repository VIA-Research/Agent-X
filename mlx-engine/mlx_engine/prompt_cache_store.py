"""
On-disk store of precomputed prompt KV caches (PromptWeaver, Agent-X).

PromptWeaver rebuilds agent prompts so that most of their tokens form static
prefixes (all tool descriptions and guidelines, followed by a fixed ordering of
clustered tool-use examples). The KV caches of a selected set of such prefixes
are computed offline and kept on storage. At request time, the stored prefix
that shares the longest token prefix with the incoming prompt is loaded and
trimmed to the shared length, so only the remaining tokens are prefilled.
"""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import mlx.core as mx
import numpy as np
from mlx_lm.models.cache import (
    can_trim_prompt_cache,
    load_prompt_cache,
    save_prompt_cache,
    trim_prompt_cache,
)

from mlx_engine.logging import log_info

LOG_PREFIX = "PromptCacheStore"


def _common_prefix_length(a: Sequence[int], b: Sequence[int]) -> int:
    n = min(len(a), len(b))
    if n == 0:
        return 0
    mismatch = np.flatnonzero(np.asarray(a[:n]) != np.asarray(b[:n]))
    return int(mismatch[0]) if mismatch.size else n


class PromptCacheStore:
    """
    A directory of prompt KV caches (``*.safetensors``) and an ``index.json``
    holding the prompt text and token ids of each entry.

    Args:
        cache_dir (str | Path): Directory to keep the caches in.
        model_id (Optional[str]): Identifier of the model the caches belong to.
            Opening a directory created for a different model raises an error.
    """

    INDEX_FILE = "index.json"

    def __init__(self, cache_dir: str | Path, model_id: Optional[str] = None):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.model_id = model_id
        self._entries: Dict[str, Dict[str, Any]] = {}

        index_path = self.cache_dir / self.INDEX_FILE
        if index_path.exists():
            index = json.loads(index_path.read_text())
            stored_model = index.get("model")
            if model_id is not None and stored_model not in (None, model_id):
                raise ValueError(
                    f"Prompt caches in {self.cache_dir} were computed for model "
                    f"'{stored_model}', not '{model_id}'. Reset the store or use "
                    "another cache directory."
                )
            self._entries = index.get("entries", {})
        log_info(
            prefix=LOG_PREFIX,
            message=f"{len(self._entries)} prompt cache(s) in {self.cache_dir}",
        )

    def _write_index(self) -> None:
        index = {"model": self.model_id, "entries": self._entries}
        (self.cache_dir / self.INDEX_FILE).write_text(json.dumps(index))

    def find_longest_prefix(self, tokens: Sequence[int]) -> Tuple[Optional[str], int]:
        """
        Find the entry sharing the longest token prefix with ``tokens``.

        At least one token of ``tokens`` is always left uncached so that the
        model has an input to process.

        Returns:
            Tuple[Optional[str], int]: The entry name (or ``None``) and the
                number of reusable tokens.
        """
        best_name, best_len = None, 0
        for name, entry in self._entries.items():
            n = _common_prefix_length(entry["tokens"], tokens)
            if n > best_len:
                best_name, best_len = name, n
        best_len = min(best_len, len(tokens) - 1)
        if best_len <= 0:
            return None, 0
        return best_name, best_len

    def load(self, name: str, num_tokens: int) -> List[Any]:
        """
        Load the KV cache of entry ``name`` and trim it to its first
        ``num_tokens`` tokens.
        """
        prompt_cache = load_prompt_cache(str(self.cache_dir / name))
        # Materialize the arrays so the read from storage happens here
        mx.eval([c.state for c in prompt_cache])
        num_to_trim = prompt_cache[0].offset - num_tokens
        if num_to_trim > 0:
            if not can_trim_prompt_cache(prompt_cache):
                raise ValueError(f"Prompt cache {name} cannot be trimmed.")
            trim_prompt_cache(prompt_cache, num_to_trim)
        return prompt_cache

    def save(
        self, prompt_text: str, tokens: Sequence[int], prompt_cache: List[Any]
    ) -> str:
        """
        Save the KV cache of ``tokens`` (the tokenization of ``prompt_text``).

        Returns:
            str: The entry name.
        """
        digest = hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()[:16]
        name = f"{digest}.safetensors"
        save_prompt_cache(
            str(self.cache_dir / name),
            prompt_cache,
            metadata={"num_tokens": str(len(tokens))},
        )
        self._entries[name] = {"prompt": prompt_text, "tokens": list(tokens)}
        self._write_index()
        return name

    def entries(self) -> List[Dict[str, Any]]:
        """Summaries of the stored entries."""
        return [
            {
                "name": name,
                "num_tokens": len(entry["tokens"]),
                "size_bytes": (self.cache_dir / name).stat().st_size,
                "prompt_preview": entry["prompt"][:80],
            }
            for name, entry in self._entries.items()
        ]

    def reset(self) -> None:
        """Delete all stored caches."""
        for name in self._entries:
            (self.cache_dir / name).unlink(missing_ok=True)
        self._entries = {}
        self._write_index()

# Copyright © 2026 Agent-X contributors

"""
N-gram lookup table (LUT) used as the draft model of ExSpec.

ExSpec (Agent-X, MobiSys '26) replaces the draft LLM of speculative decoding
with an n-gram LUT that is built on the fly from the few-shot examples and the
user query in the prompt. Each key is a context of ``n - 1`` tokens, and its
value is the token that most frequently follows that context.
"""

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple


@dataclass
class NGramLUT:
    """
    An n-gram lookup table mapping an ``(n - 1)``-token context to the most
    frequent next token.

    Args:
        n (int): The order of the n-gram model (e.g. 3 for a trigram model).
        table (Dict[Tuple[int, ...], int]): Context -> most frequent next token.
    """

    n: int
    table: Dict[Tuple[int, ...], int]

    @property
    def context_size(self) -> int:
        return self.n - 1

    def lookup(self, history: Sequence[int]) -> Optional[int]:
        """
        Return the draft token that follows the last ``n - 1`` tokens of
        ``history``, or ``None`` if the context is not in the table.
        """
        k = self.context_size
        if len(history) < k:
            return None
        key = tuple(history[len(history) - k :]) if k > 0 else ()
        return self.table.get(key)


def build_ngram_lut(tokens: Sequence[int], n: int = 3) -> NGramLUT:
    """
    Build an n-gram LUT by sliding a window of ``n`` tokens over ``tokens``.

    Ties between equally frequent next tokens are broken in favor of the token
    that appeared first in ``tokens``.

    Args:
        tokens (Sequence[int]): The token stream to build the table from.
        n (int): The order of the n-gram model. Default: ``3``.

    Returns:
        NGramLUT: The lookup table.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}.")
    tokens = list(tokens)
    k = n - 1
    counts: Dict[Tuple[int, ...], Counter] = defaultdict(Counter)
    for i in range(len(tokens) - k):
        counts[tuple(tokens[i : i + k])][tokens[i + k]] += 1
    table = {ctx: max(c, key=c.get) for ctx, c in counts.items()}
    return NGramLUT(n=n, table=table)

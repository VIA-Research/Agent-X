"""
Builds the n-gram draft model of ExSpec (Agent-X) from a prompt.
"""

from typing import Literal

from mlx_lm.ngram_draft import NGramLUT, build_ngram_lut

from mlx_engine.logging import log_warn
from mlx_engine.model_kit.model_kit import ModelKit

# Marker preceding the few-shot examples in TinyAgent planner/arbiter prompts
FEW_SHOT_MARKER = "Here are some examples:"

NGramSource = Literal["few-shot", "all"]


def extract_ngram_source(prompt_text: str, source: NGramSource) -> str:
    """
    Select the part of the prompt to build the n-gram table from.

    ``"few-shot"`` uses the text after the first occurrence of
    ``FEW_SHOT_MARKER``, i.e. the few-shot examples and the user query.
    ``"all"`` uses the whole prompt.
    """
    if source == "all":
        return prompt_text
    if source == "few-shot":
        if FEW_SHOT_MARKER not in prompt_text:
            log_warn(
                f"'{FEW_SHOT_MARKER}' not found in prompt, "
                "building the n-gram table from the whole prompt"
            )
            return prompt_text
        return prompt_text.split(FEW_SHOT_MARKER, 1)[1]
    raise ValueError(f"Unsupported n-gram source: {source}")


def create_ngram_lut(
    model_kit: ModelKit,
    prompt_text: str,
    n: int = 3,
    source: NGramSource = "few-shot",
) -> NGramLUT:
    """
    Tokenize the selected part of the prompt and build an n-gram table from it.
    """
    tokens = model_kit.tokenize(extract_ngram_source(prompt_text, source))
    return build_ngram_lut(tokens, n)

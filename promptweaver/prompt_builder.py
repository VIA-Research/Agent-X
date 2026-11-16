"""
PromptWeaver prompt layout for the TinyAgent planner (Agent-X, MobiSys '26).

A PromptWeaver planner prompt consists of, in order:
  1. a static prefix with the descriptions and guidelines of *all* tools,
  2. the tool-use examples of the activated tool clusters, in a fixed order,
  3. the dynamic part: the selected tools, the date, single-tool examples of
     the selected tools, the top-K examples retrieved by ToolRAG, and the query.

Parts 1 and 2 are shared across queries that activate the same clusters, so
their KV caches are precomputed offline for a selected set of cluster
combinations (see select_cluster_combinations.py) and reused at runtime.

Usage (write the cache prompts of the selected cluster combinations):
    python prompt_builder.py --combinations 1234 123678 12368 127 134 1368 2 8 \
        --out-dir tinyagent/cache_prompts
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

TINYAGENT_DIR = Path(__file__).resolve().parent / "tinyagent"

EXAMPLE_SEPARATOR = "###\n"
TOOL_LIST_HEADER = "\n\nYou should only plan using these provided action types:\n"
RETRIEVED_EXAMPLES_HEADER = (
    "With this in mind, refer to these examples for planning:\n\n"
)


@dataclass
class Cluster:
    id: int
    tools: List[str]
    example: str


@dataclass
class PromptWeaverConfig:
    planner_prefix: str
    clusters: List[Cluster]
    single_tool_examples: Dict[str, str]

    @classmethod
    def load(cls, config_dir: Path = TINYAGENT_DIR) -> "PromptWeaverConfig":
        config_dir = Path(config_dir)
        examples = json.loads((config_dir / "tool_examples.json").read_text())
        return cls(
            planner_prefix=(config_dir / "planner_prefix.txt").read_text(),
            clusters=[Cluster(**c) for c in examples["clusters"]],
            single_tool_examples=examples["single_tool_examples"],
        )

    def cluster(self, cluster_id: int) -> Cluster:
        return next(c for c in self.clusters if c.id == cluster_id)

    def activated_clusters(self, selected_tools: Sequence[str]) -> List[Cluster]:
        """Clusters containing at least one selected tool, in the fixed order."""
        return [c for c in self.clusters if any(t in c.tools for t in selected_tools)]


def _clustered_examples(clusters: Sequence[Cluster]) -> str:
    return EXAMPLE_SEPARATOR.join(c.example for c in clusters) + "\n"


def build_cache_prompt(config: PromptWeaverConfig, clusters: Sequence[Cluster]) -> str:
    """
    The cacheable part of the planner prompt for a combination of clusters. It
    ends with the first characters of the tool list, which are the same for
    every query.
    """
    return (
        config.planner_prefix + _clustered_examples(clusters) + TOOL_LIST_HEADER + " - "
    )


def build_planner_prompt(
    config: PromptWeaverConfig,
    selected_tools: Sequence[str],
    retrieved_examples: Sequence[str],
    query: str,
    date: str,
) -> str:
    """
    Build the planner prompt for a query.

    Args:
        config: The PromptWeaver configuration.
        selected_tools: Tools selected by the ToolRAG classifier.
        retrieved_examples: Top-K tool-use examples retrieved by ToolRAG
            (K = 1 in the paper), each in the "Question: ...\\n1. ...\\n" format.
        query: The user query.
        date: The current date, e.g. "Monday 2025-06-02 20:18".
    """
    prompt = config.planner_prefix
    prompt += _clustered_examples(config.activated_clusters(selected_tools))

    prompt += TOOL_LIST_HEADER
    for tool in selected_tools:
        prompt += f" - {tool}\n"
    prompt += " - join\n"
    prompt += f" - Today's date is {date}\n\n"

    # Single-tool examples of the selected tools, then the retrieved examples
    retrieved = EXAMPLE_SEPARATOR.join(retrieved_examples) + EXAMPLE_SEPARATOR
    single = [
        config.single_tool_examples[t]
        for t in selected_tools
        if t in config.single_tool_examples
        and config.single_tool_examples[t] not in retrieved
    ]
    if single:
        retrieved = (
            EXAMPLE_SEPARATOR.join(single) + EXAMPLE_SEPARATOR + retrieved + "\n"
        )
    prompt += RETRIEVED_EXAMPLES_HEADER + retrieved

    return prompt + f"\n\nQuestion: {query}"


def main():
    parser = argparse.ArgumentParser(
        description="Write the cache prompts of the selected cluster combinations."
    )
    parser.add_argument(
        "--combinations",
        nargs="+",
        required=True,
        help="Cluster combinations as strings of cluster ids, e.g. 1234 127 2",
    )
    parser.add_argument("--config-dir", type=Path, default=TINYAGENT_DIR)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    config = PromptWeaverConfig.load(args.config_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for combination in args.combinations:
        clusters = [config.cluster(int(i)) for i in combination]
        path = args.out_dir / f"planner_clusters_{combination}.txt"
        path.write_text(build_cache_prompt(config, clusters))
        print(f"wrote {path}")


if __name__ == "__main__":
    main()

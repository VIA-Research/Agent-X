"""
Cluster combination selection (PromptWeaver offline phase, Algorithm 1).

Each training sample activates the clusters that contain at least one of the
tools in its ground-truth plan; the activated clusters, in their fixed order,
form the sample's cluster sequence. Storing the KV cache of a cluster prefix
(e.g. clusters 1-3) lets every sample whose sequence starts with that prefix
reuse it. Under a budget of N cached prefixes, the algorithm greedily adds the
prefix (a single cluster, or a one-cluster extension of an already cached
prefix) that most increases the coverage: the sum, over samples, of the length
of the longest cached prefix of the sample's sequence.

Only the maximal cached prefixes need to be stored, as shorter prefixes are
reused by trimming the KV cache.

Usage:
    python select_cluster_combinations.py \
        --dataset /path/to/TinyAgent-dataset/training_data.json --budget 15
"""

import argparse
from typing import List, Sequence, Set, Tuple

from dataset import load_plans
from prompt_builder import TINYAGENT_DIR, PromptWeaverConfig

Prefix = Tuple[int, ...]


def compute_coverage(sequences: Sequence[Sequence[int]], owned: Set[Prefix]) -> int:
    total = 0
    for seq in sequences:
        best = 0
        for length in range(1, len(seq) + 1):
            if tuple(seq[:length]) in owned:
                best = length
        total += best
    return total


def greedy_prefix_selection(
    sequences: Sequence[Sequence[int]], budget: int
) -> Set[Prefix]:
    all_prefixes = {
        tuple(seq[:length]) for seq in sequences for length in range(1, len(seq) + 1)
    }
    max_coverage = sum(len(seq) for seq in sequences)
    owned: Set[Prefix] = set()
    coverage = 0

    for step in range(1, budget + 1):
        candidates = [p for p in all_prefixes - owned if len(p) == 1 or p[:-1] in owned]
        best_gain, best_prefix = 0, None
        for p in candidates:
            gain = compute_coverage(sequences, owned | {p}) - coverage
            # Ties: prefer longer, then lexicographically smaller prefixes
            if gain > best_gain or (
                gain == best_gain
                and best_prefix is not None
                and (
                    len(p) > len(best_prefix)
                    or (len(p) == len(best_prefix) and p < best_prefix)
                )
            ):
                best_gain, best_prefix = gain, p
        if best_prefix is None or best_gain == 0:
            print("No further coverage gain possible.")
            break
        owned.add(best_prefix)
        coverage += best_gain
        print(
            f"Step {step}: added {list(best_prefix)} | gain: {best_gain} | "
            f"coverage: {coverage}/{max_coverage} ({coverage / max_coverage:.1%})"
        )
    return owned


def maximal_prefixes(prefixes: Set[Prefix]) -> List[Prefix]:
    return sorted(
        p
        for p in prefixes
        if not any(len(p) < len(q) and q[: len(p)] == p for q in prefixes)
    )


def main():
    parser = argparse.ArgumentParser(
        description="Select cluster combinations to cache."
    )
    parser.add_argument("--dataset", required=True, help="Path to training_data.json")
    parser.add_argument(
        "--budget", type=int, default=15, help="Number of cached prefixes"
    )
    parser.add_argument("--config-dir", default=TINYAGENT_DIR)
    args = parser.parse_args()

    config = PromptWeaverConfig.load(args.config_dir)
    sequences = [
        [c.id for c in config.activated_clusters(plan)]
        for plan in load_plans(args.dataset)
    ]
    owned = greedy_prefix_selection(sequences, args.budget)

    combinations = ["".join(map(str, p)) for p in maximal_prefixes(owned)]
    print(f"\nCluster combinations to cache: {' '.join(combinations)}")
    print(
        "Write their cache prompts with:\n"
        f"  python prompt_builder.py --combinations {' '.join(combinations)} "
        "--out-dir tinyagent/cache_prompts"
    )


if __name__ == "__main__":
    main()

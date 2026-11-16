"""
Co-activation locality-based tool clustering (PromptWeaver offline phase).

Builds a tool co-activation matrix from the ground-truth plans of the planner's
training dataset (how often each pair of tools is called in the same plan) and
factorizes it with non-negative matrix factorization (NMF). Each tool is
assigned to its top latent component, and also to its second one if that
weight is at least `threshold` times the top weight.

The resulting clusters are candidates, which are then refined by hand so that
each cluster has one tool-use example that uses exactly its tools. The default
settings give the 6 candidate clusters from which the 8 clusters of the paper
(tinyagent/tool_examples.json) were derived; see the README.

Usage:
    python cluster_tools.py --dataset /path/to/TinyAgent-dataset/training_data.json
"""

import argparse
import collections
from itertools import combinations
from typing import Dict, List, Sequence, Tuple

import numpy as np
from sklearn.decomposition import NMF

from dataset import load_plans

TINYAGENT_TOOLS = sorted(
    [
        "create_calendar_event",
        "get_phone_number",
        "get_email_address",
        "open_and_get_file_path",
        "summarize_pdf",
        "compose_new_email",
        "reply_to_email",
        "forward_email",
        "maps_open_location",
        "maps_show_directions",
        "create_note",
        "open_note",
        "append_note_content",
        "create_reminder",
        "send_sms",
        "get_zoom_meeting_link",
    ]
)


def build_cooccurrence_matrix(
    tools: Sequence[str], tool_sets: Dict[Tuple[str, ...], int]
) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    Args:
        tools: Tool names.
        tool_sets: Set of tools used in a plan (sorted tuple) -> number of plans.

    Returns:
        The symmetric co-occurrence matrix (diagonal: number of plans using the
        tool) and the tool -> row index mapping.
    """
    tool_index = {tool: i for i, tool in enumerate(tools)}
    A = np.zeros((len(tools), len(tools)), dtype=float)
    for key, count in tool_sets.items():
        for t1, t2 in combinations(key, 2):
            i, j = tool_index[t1], tool_index[t2]
            A[i, j] += count
            A[j, i] += count
        for t in key:
            A[tool_index[t], tool_index[t]] += count
    return A, tool_index


def cluster_tools(
    tools: Sequence[str],
    tool_sets: Dict[Tuple[str, ...], int],
    n_clusters: int = 6,
    threshold: float = 0.4,
    random_state: int = 0,
    max_iter: int = 200,
) -> Tuple[Dict[int, List[str]], Dict[str, List[int]]]:
    """
    Returns:
        The clusters (cluster index -> tools) and the memberships
        (tool -> cluster indices).
    """
    A, tool_index = build_cooccurrence_matrix(tools, tool_sets)
    model = NMF(
        n_components=n_clusters,
        init="nndsvda",
        random_state=random_state,
        max_iter=max_iter,
    )
    # numpy on Apple Accelerate emits spurious floating-point warnings in matmul
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        W = model.fit_transform(A)

    clusters = {c: [] for c in range(n_clusters)}
    memberships = {}
    for tool, i in tool_index.items():
        weights = W[i]
        ranked = np.argsort(-weights)
        # Top cluster, plus the second if its weight is large enough
        members = [int(ranked[0])]
        if weights[ranked[1]] >= threshold * weights[ranked[0]]:
            members.append(int(ranked[1]))
        memberships[tool] = members
        for c in members:
            clusters[c].append(tool)
    return clusters, memberships


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dataset", required=True, help="Path to training_data.json")
    parser.add_argument("--n-clusters", type=int, default=6)
    parser.add_argument("--threshold", type=float, default=0.4)
    parser.add_argument("--random-state", type=int, default=0)
    args = parser.parse_args()

    tool_sets = collections.Counter(
        tuple(sorted(set(plan))) for plan in load_plans(args.dataset)
    )
    clusters, memberships = cluster_tools(
        TINYAGENT_TOOLS,
        tool_sets,
        n_clusters=args.n_clusters,
        threshold=args.threshold,
        random_state=args.random_state,
    )

    print("Clusters (cluster index -> tools):")
    for c, members in clusters.items():
        print(f"  {c}: {members}")
    print("\nMemberships (tool -> cluster indices):")
    for tool, m in memberships.items():
        print(f"  {tool}: {m}")


if __name__ == "__main__":
    main()

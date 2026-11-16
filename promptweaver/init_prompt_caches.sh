#!/usr/bin/env bash
# Precompute and store the prompt KV caches used by PromptWeaver on a running
# Agent-X server: the planner prompts of the selected cluster combinations and
# the two static Arbiter prompts.
#
# Usage: ./init_prompt_caches.sh [server_url]
set -euo pipefail

SERVER="${1:-http://127.0.0.1:8000}"
DIR="$(cd "$(dirname "$0")" && pwd)/tinyagent"

store() {
    echo "storing $(basename "$1")"
    python3 -c 'import json, sys; print(json.dumps({"prompt": open(sys.argv[1]).read()}))' "$1" |
        curl -sSf -X POST "$SERVER/v1/prompt_cache" -H "Content-Type: application/json" -d @-
    echo
}

curl -sSf -X DELETE "$SERVER/v1/prompt_cache" > /dev/null
for f in "$DIR"/cache_prompts/*.txt; do
    store "$f"
done
store "$DIR/arbiter_prefix.txt"
store "$DIR/arbiter_final_prefix.txt"

# Agent-X: Full Pipeline Acceleration of On-device AI Agents

Agent-X is a software-only framework that accelerates both the prefill and the
decode stages of on-device LLM agents without changing their outputs.
**PromptWeaver** reorganizes the agent prompt into a static prefix of tool
descriptions and clustered tool-use examples, whose KV caches are precomputed
offline and reused from SSD. **ExSpec** performs speculative decoding without a
draft model, taking draft tokens from an n-gram table built on the fly from the
few-shot examples in the prompt. Agent-X is built on
[MLX-LM](https://github.com/ml-explore/mlx-lm) and
[MLX-engine](https://github.com/lmstudio-ai/mlx-engine).

For more details, see our [paper](https://doi.org/10.1145/3745756.3809195)
published at MobiSys 2026.

## Table of Contents

- [Setup](#setup)
- [Project Structure](#project-structure)
- [Example](#example)
- [PromptWeaver Offline Phase](#promptweaver-offline-phase)
- [Acknowledgment](#acknowledgment)
- [Citation](#citation)

## Setup

Agent-X runs on a Mac with Apple silicon. From the root of this repository:

```bash
conda create -n agentx python=3.12
conda activate agentx
bash setup.sh
```

Download the [TinyAgent-7B checkpoint](https://huggingface.co/squeeze-ai-lab/TinyAgent-7B)
to `$MODEL_DIR/TinyAgent-7B/`. The checkpoint has no tokenizer files; add those
of [Mistral-7B-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1), on which
TinyAgent-7B is based. The paper uses the float32 checkpoint as is.

```bash
export MODEL_DIR=/path/to/model_checkpoints
huggingface-cli download squeeze-ai-lab/TinyAgent-7B --local-dir $MODEL_DIR/TinyAgent-7B
huggingface-cli download mistralai/Mistral-7B-v0.1 tokenizer.model tokenizer.json tokenizer_config.json special_tokens_map.json --local-dir $MODEL_DIR/TinyAgent-7B
```

## Project Structure

```
setup.sh                              # Environment setup
mlx-lm/                               # MLX-LM v0.25.1 with ExSpec
  mlx_lm/ngram_draft.py               # n-gram draft table
  mlx_lm/generate.py                  # ExSpec decoding loop
mlx-engine/                           # MLX-engine (ecc2cf4) with PromptWeaver and ExSpec
  mlx_engine/prompt_cache_store.py    # Stored prompt KV caches
  mlx_engine/utils/ngram_lut.py       # n-gram table from the few-shot examples
  mlx_engine/generate.py              # Agent-X options of create_generator
  server.py                           # HTTP server
promptweaver/                         # PromptWeaver offline phase
  tinyagent/                          # TinyAgent prompts and cache prompts
examples/                             # Example TinyAgent planner prompt
```

`mlx-lm/` and `mlx-engine/` keep the MIT licenses of their upstream projects.
The TinyAgent prompts in `promptweaver/tinyagent/` and `examples/` come from
[TinyAgent](https://github.com/SqueezeAILab/TinyAgent) (MIT License,
`promptweaver/tinyagent/LICENSE`).

## Example

```bash
python mlx-engine/server.py --model $MODEL_DIR/TinyAgent-7B --prompt-dir examples

# In another terminal: store the PromptWeaver KV caches (once; kept in ./prompt_caches)
bash promptweaver/init_prompt_caches.sh

# Agent-X (PromptWeaver + ExSpec)
curl -N http://127.0.0.1:8000/v1/completions -H "Content-Type: application/json" -d '{"prompt_file": "planner_prompt.txt", "return_stats": true}'
# PromptWeaver only
curl -N http://127.0.0.1:8000/v1/completions -H "Content-Type: application/json" -d '{"prompt_file": "planner_prompt.txt", "ngram": 0}'
# ExSpec only
curl -N http://127.0.0.1:8000/v1/completions -H "Content-Type: application/json" -d '{"prompt_file": "planner_prompt.txt", "use_prompt_cache": false}'
# Baseline
curl -N http://127.0.0.1:8000/v1/completions -H "Content-Type: application/json" -d '{"prompt_file": "planner_prompt.txt", "use_prompt_cache": false, "ngram": 0}'
```

All four generate the same plan. With `"return_stats": true`, the response ends
with the prefill and decode times, the number of cached prompt tokens, and the
draft acceptance statistics.

Prompts can also be sent inline as `"prompt"`. ExSpec works with any prompt.
PromptWeaver speeds up prompts that start with a stored cache prompt; build
TinyAgent planner prompts with `build_planner_prompt` in
`promptweaver/prompt_builder.py` (TinyAgent Arbiter prompts need no change).

### CLI reference

```bash
python mlx-engine/server.py --help
```

| Argument | Default |
| --- | --- |
| `--model` | required |
| `--cache-dir` | `prompt_caches` |
| `--prompt-dir` | none |
| `--max-kv-size` | `8192` |
| `--host` | `127.0.0.1` |
| `--port` | `8000` |

`--max-kv-size` must exceed the longest prompt plus the generated tokens.

### Request fields

`POST /v1/completions` streams the generated text.

| Field | Default | Description |
| --- | --- | --- |
| `prompt` / `prompt_file` | required | Prompt, or a file in `--prompt-dir` |
| `max_tokens` | `512` | |
| `temperature` | `0` | Must be 0 with ExSpec |
| `use_prompt_cache` | `true` | PromptWeaver: reuse the stored KV cache with the longest shared prefix |
| `ngram` | `3` | ExSpec: n-gram order; `0` disables ExSpec |
| `ngram_source {few-shot,all}` | `few-shot` | Build the n-gram table from the text after `Here are some examples:`, or from the whole prompt |
| `num_draft_tokens` | `10` | |
| `selective` | `true` | Fall back to autoregressive decoding when the n-gram table has no draft |
| `return_stats` | `false` | |

`POST /v1/prompt_cache` with `{"prompt": ...}` stores the KV cache of a prompt,
`GET /v1/prompt_cache` lists the stored caches, and `DELETE /v1/prompt_cache`
deletes them. From Python, the same options are keyword arguments of
`mlx_engine.generate.create_generator`.

## PromptWeaver Offline Phase

`promptweaver/tinyagent/cache_prompts/` already contains the cache prompts of
the paper (8 tool clusters, budget of 15 cached cluster prefixes). To derive
them again or for another budget, download
[TinyAgent-dataset](https://huggingface.co/datasets/squeeze-ai-lab/TinyAgent-dataset)
to `$DATA_DIR/TinyAgent-dataset/`.

```bash
cd promptweaver
# Candidate tool clusters (see Tool clusters below)
python cluster_tools.py --dataset $DATA_DIR/TinyAgent-dataset/training_data.json
# Cluster combinations to cache (Algorithm 1)
python select_cluster_combinations.py --dataset $DATA_DIR/TinyAgent-dataset/training_data.json --budget 15
# Cache prompts of the selected combinations
python prompt_builder.py --combinations 1234 123678 12368 127 134 1368 2 8 --out-dir tinyagent/cache_prompts
```

### Tool clusters

`cluster_tools.py` factorizes the tool co-activation matrix of the training
plans with NMF (6 components) and assigns each tool to its top cluster, and to
its second cluster if that weight is at least 0.4 times the top weight. The 8
clusters of the paper (`tinyagent/tool_examples.json`) were derived from these 6
candidate clusters by hand, so that the tools of each cluster are exactly those
used by one training example, which becomes the example of the cluster:

| Candidate cluster | Clusters of the paper |
| --- | --- |
| `append_note_content` `compose_new_email` `create_calendar_event` `forward_email` `get_email_address` `get_zoom_meeting_link` `open_and_get_file_path` `open_note` | Split into 1 (`compose_new_email` `create_calendar_event` `forward_email` `get_email_address` `get_zoom_meeting_link` `open_and_get_file_path`) and 6 (`append_note_content` `open_and_get_file_path` `open_note`) |
| `compose_new_email` `create_note` `create_reminder` `maps_open_location` `maps_show_directions` `summarize_pdf` | Split into 2 (`compose_new_email` `get_email_address` `maps_open_location` `maps_show_directions`) and 8 (`create_note` `create_reminder` `open_and_get_file_path` `summarize_pdf`) |
| `append_note_content` `reply_to_email` `summarize_pdf` | 3, adding `open_and_get_file_path` |
| `get_phone_number` `reply_to_email` `send_sms` | 4 |
| `get_phone_number` `send_sms` | 5 |
| `create_calendar_event` `get_zoom_meeting_link` `maps_open_location` | 7 |

## Acknowledgment

This research is funded by the generous support from the following organizations:

- Institute of Information & communications Technology Planning & Evaluation
  (IITP) grant funded by the Korea government(MSIT) (No.RS-2024-00395134,
  DPU-Centric Datacenter Architecture for Next-Generation AI Devices),
  (No.RS-2024-00438851, (SW Starlab) High-performance Privacy-preserving Machine
  Learning System and System Software), (No. RS-2024-00457882, AI Research Hub
  Project), (No.RS-2025-02214652, Development of SoC Technology for AI
  Semiconductor-Converged Pooled Storage/Memory)
- Samsung Electronics Co., Ltd. (IO251210-14212-01)

## Citation

```bibtex
@inproceedings{chung2026agentx,
  title={Agent-X: Full Pipeline Acceleration of On-device AI Agents},
  author={Chung, Jinha and Shin, Byeongjun and Kim, Jiin and Rhu, Minsoo},
  booktitle={Proceedings of the 24th Annual International Conference on Mobile Systems, Applications and Services (MobiSys '26)},
  year={2026},
  doi={10.1145/3745756.3809195}
}
```

"""
HTTP server for Agent-X on top of mlx-engine.

Run:
    python server.py --model /path/to/TinyAgent-7B --cache-dir ./prompt_caches

Endpoints:
    POST   /v1/completions    Generate text (streams plain text).
    POST   /v1/prompt_cache   Precompute and store the KV cache of a prompt (PromptWeaver).
    GET    /v1/prompt_cache   List stored prompt caches.
    DELETE /v1/prompt_cache   Delete all stored prompt caches.
"""

import argparse
import threading
import time
from pathlib import Path
from typing import Literal, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, model_validator
from starlette.responses import StreamingResponse

from mlx_engine.generate import (
    AgentXStats,
    create_generator,
    load_model,
    store_prompt_cache,
    tokenize,
)
from mlx_engine.prompt_cache_store import PromptCacheStore

app = FastAPI()
model_kit = None
prompt_cache_store: Optional[PromptCacheStore] = None
prompt_dir: Optional[Path] = None
# The model serves one request at a time
model_lock = threading.Lock()


class PromptRequest(BaseModel):
    # Exactly one of `prompt` and `prompt_file` (relative to --prompt-dir)
    prompt: Optional[str] = None
    prompt_file: Optional[str] = None

    @model_validator(mode="after")
    def one_of_prompt_or_file(self):
        if (self.prompt is None) == (self.prompt_file is None):
            raise ValueError("Supply exactly one of prompt and prompt_file")
        return self

    def read_prompt(self) -> str:
        if self.prompt is not None:
            return self.prompt
        if prompt_dir is None:
            raise HTTPException(400, "prompt_file requires --prompt-dir")
        path = (prompt_dir / self.prompt_file).resolve()
        if prompt_dir not in path.parents or not path.is_file():
            raise HTTPException(400, f"Prompt file not found: {self.prompt_file}")
        return path.read_text(encoding="utf-8")


class CompletionRequest(PromptRequest):
    max_tokens: int = 512
    temperature: float = 0.0
    # PromptWeaver: reuse stored prompt KV caches
    use_prompt_cache: bool = True
    # ExSpec: n-gram speculative decoding (0 disables it)
    ngram: int = 3
    ngram_source: Literal["few-shot", "all"] = "few-shot"
    num_draft_tokens: int = 10
    selective: bool = True
    # Append a statistics block to the generated text
    return_stats: bool = False


@app.post("/v1/completions")
def completions(req: CompletionRequest):
    if req.ngram < 0 or req.num_draft_tokens < 1:
        raise HTTPException(400, "ngram must be >= 0 and num_draft_tokens >= 1")
    if req.ngram > 0 and req.temperature > 0:
        raise HTTPException(400, "N-gram speculative decoding requires temperature 0")
    prompt_text = req.read_prompt()
    stats = AgentXStats()

    def stream():
        with model_lock:
            for result in create_generator(
                model_kit,
                tokenize(model_kit, prompt_text),
                prompt_text=prompt_text,
                prompt_cache_store=prompt_cache_store if req.use_prompt_cache else None,
                ngram=req.ngram,
                ngram_source=req.ngram_source,
                ngram_selective=req.selective,
                num_draft_tokens=req.num_draft_tokens,
                max_tokens=req.max_tokens,
                temp=req.temperature,
                agent_x_stats=stats,
            ):
                yield result.text
        if req.return_stats:
            yield "\n\n" + stats.format()

    return StreamingResponse(stream(), media_type="text/plain")


@app.post("/v1/prompt_cache")
def add_prompt_cache(req: PromptRequest):
    prompt_text = req.read_prompt()
    start = time.perf_counter()
    with model_lock:
        num_tokens = store_prompt_cache(model_kit, prompt_text, prompt_cache_store)
    return {"num_tokens": num_tokens, "seconds": time.perf_counter() - start}


@app.get("/v1/prompt_cache")
def list_prompt_caches():
    return prompt_cache_store.entries()


@app.delete("/v1/prompt_cache")
def reset_prompt_caches():
    with model_lock:
        prompt_cache_store.reset()
    return {"status": "ok"}


def main():
    global model_kit, prompt_cache_store, prompt_dir

    parser = argparse.ArgumentParser(description="Agent-X server")
    parser.add_argument("--model", required=True, help="Path to the MLX model")
    parser.add_argument(
        "--cache-dir",
        default="prompt_caches",
        help="Directory of the stored prompt KV caches",
    )
    parser.add_argument(
        "--prompt-dir",
        default=None,
        help="Directory that `prompt_file` in requests is relative to",
    )
    parser.add_argument(
        "--max-kv-size",
        type=int,
        default=8192,
        help="Maximum KV cache size; must exceed the longest prompt plus output",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    model_kit = load_model(args.model, max_kv_size=args.max_kv_size)
    model_id = Path(args.model).resolve().name
    prompt_cache_store = PromptCacheStore(args.cache_dir, model_id=model_id)
    if args.prompt_dir is not None:
        prompt_dir = Path(args.prompt_dir).resolve()

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()

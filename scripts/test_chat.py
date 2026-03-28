#!/usr/bin/env python3
"""Simple reusable OpenAI-compatible chat test for local model servers."""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from typing import Any


def http_json(url: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    data = None
    headers = {"Content-Type": "application/json"}
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")

    request = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(request, timeout=300) as response:
        return json.loads(response.read().decode("utf-8"))


def resolve_model(base_url: str, model: str | None) -> str:
    if model:
        return model

    models_url = f"{base_url.rstrip('/')}/v1/models"
    response = http_json(models_url)
    items = response.get("data") or []
    if not items:
        raise RuntimeError(f"No models returned by {models_url}")
    return str(items[0]["id"])


def build_payload(model: str, prompt: str, max_tokens: int) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
    }


def extract_answer(response: dict[str, Any]) -> tuple[str | None, str | None]:
    choices = response.get("choices") or []
    if not choices:
        return None, None

    message = choices[0].get("message") or {}
    content = message.get("content")
    reasoning_content = message.get("reasoning_content")
    return content, reasoning_content


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Send a simple chat query to a local OpenAI-compatible server")
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000",
        help="Server base URL, default: http://127.0.0.1:8000",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model ID. If omitted, the script fetches /v1/models and uses the first one.",
    )
    parser.add_argument(
        "--prompt",
        default="英国的首都在哪里？请只用一句中文回答。",
        help="Prompt to send.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2480,
        help="Maximum completion tokens.",
    )
    return parser


def main() -> int:
    parser = make_parser()
    args = parser.parse_args()

    try:
        model = resolve_model(args.base_url, args.model)
        payload = build_payload(
            model=model,
            prompt=args.prompt,
            max_tokens=args.max_tokens,
        )
        response = http_json(f"{args.base_url.rstrip('/')}/v1/chat/completions", payload)
        content, reasoning_content = extract_answer(response)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        print(f"HTTP error {exc.code}: {body}", file=sys.stderr)
        return 1
    except urllib.error.URLError as exc:
        print(f"Request failed: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001
        print(f"Request failed: {exc}", file=sys.stderr)
        return 1

    print(f"Model: {model}")
    print(f"Prompt: {args.prompt}")
    if content is not None:
        print(f"Answer: {content}")
    else:
        print("Answer: <empty>")

    if reasoning_content:
        print("Reasoning:")
        print(reasoning_content)

    usage = response.get("usage")
    if usage:
        print("Usage:")
        print(json.dumps(usage, ensure_ascii=False, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

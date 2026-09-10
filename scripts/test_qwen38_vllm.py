#!/usr/bin/env python3
"""Exercise Qwen3.8 text, reasoning, tool parsing, and image input over HTTP."""
from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import json
from pathlib import Path
import socket
import struct
import time
import urllib.error
import urllib.request
import zlib


def request(url, payload=None, timeout=300):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            body = response.read()
            return json.loads(body) if body else {'http_status': response.status}
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f'HTTP {exc.code}: {exc.read().decode(errors="replace")}') from exc


def red_image_url():
    """A deterministic 224x224 red PNG fixture, with no downloads or Pillow dependency."""
    def chunk(kind, data):
        return struct.pack('!I', len(data)) + kind + data + struct.pack(
            '!I', zlib.crc32(kind + data) & 0xffffffff
        )
    size = 224
    rows = (b'\x00' + b'\xff\x00\x00' * size) * size
    png = b'\x89PNG\r\n\x1a\n'
    png += chunk(b'IHDR', struct.pack('!IIBBBBB', size, size, 8, 2, 0, 0, 0))
    png += chunk(b'IDAT', zlib.compress(rows)) + chunk(b'IEND', b'')
    return 'data:image/png;base64,' + base64.b64encode(png).decode()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', default='http://127.0.0.1:8000')
    parser.add_argument('--model', default='qwen3.8-27b')
    parser.add_argument('--wait-seconds', type=int, default=1200)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[1]
                        / 'run/qwen38-vllm-validation.json')
    args = parser.parse_args()
    base = args.base_url.rstrip('/')
    report = {
        'started_at': datetime.now(timezone.utc).isoformat(),
        'host': socket.gethostname(), 'base_url': base, 'model': args.model,
        'status': 'running', 'checks': {},
    }

    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')

    def chat(name, messages, *, thinking=False, **extra):
        payload = {
            'model': args.model, 'messages': messages, 'temperature': 0,
            'max_tokens': 512 if thinking else 256,
            'chat_template_kwargs': {'enable_thinking': thinking},
            **extra,
        }
        if thinking:
            payload['chat_template_kwargs']['reasoning_effort'] = 'low'
        start = time.monotonic()
        response = request(base + '/v1/chat/completions', payload)
        entry = {'elapsed_seconds': round(time.monotonic() - start, 3),
                 'request': payload, 'response': response}
        report['checks'][name] = entry
        save()
        choice = response['choices'][0]
        if choice.get('finish_reason') == 'length':
            raise AssertionError(f'{name}: response exhausted its token budget')
        return entry, choice['message']

    save()
    try:
        deadline = time.monotonic() + args.wait_seconds
        while True:
            try:
                report['checks']['health'] = request(base + '/health', timeout=5)
                break
            except (OSError, RuntimeError) as exc:
                if time.monotonic() >= deadline:
                    raise TimeoutError(f'Server not ready: {exc}') from exc
                print('Waiting for server health...', flush=True)
                time.sleep(5)
        models = request(base + '/v1/models')
        assert args.model in [item['id'] for item in models['data']], models
        report['checks']['models'] = models
        print('Health and model listing passed.', flush=True)

        entry, message = chat('text', [{'role': 'user', 'content':
            'What is the capital of the United Arab Emirates? Answer in one short sentence.'}])
        assert 'abu dhabi' in (message.get('content') or '').lower(), message
        entry['passed'] = True
        print('Text:', message['content'], flush=True)

        entry, message = chat('reasoning', [{'role': 'user', 'content':
            'Calculate 17 times 19. Give the exact answer.'}], thinking=True)
        assert '323' in (message.get('content') or ''), message
        reasoning = message.get('reasoning') or message.get('reasoning_content')
        assert reasoning, 'Reasoning was not separated from the answer'
        entry['passed'] = True
        print('Reasoning parser passed; answer:', message['content'], flush=True)

        tools = [{'type': 'function', 'function': {
            'name': 'get_weather', 'description': 'Get the current weather for a city.',
            'parameters': {'type': 'object', 'properties': {
                'city': {'type': 'string', 'description': 'The city name'}},
                'required': ['city'], 'additionalProperties': False},
        }}]
        entry, message = chat('tool_call', [{'role': 'user', 'content':
            'Call get_weather for Abu Dhabi. Do not guess the weather.'}],
            tools=tools, tool_choice='auto')
        calls = message.get('tool_calls') or []
        assert len(calls) == 1 and calls[0]['function']['name'] == 'get_weather', message
        arguments = json.loads(calls[0]['function']['arguments'])
        assert arguments.get('city', '').lower() == 'abu dhabi', arguments
        entry['passed'] = True
        print('Tool parser:', calls[0]['function'], flush=True)

        entry, message = chat('image', [{'role': 'user', 'content': [
            {'type': 'text', 'text': 'What is the dominant color in this image? Answer with one color word.'},
            {'type': 'image_url', 'image_url': {'url': red_image_url()}},
        ]}])
        assert 'red' in (message.get('content') or '').lower(), message
        entry['passed'] = True
        print('Image:', message['content'], flush=True)
        report['status'] = 'passed'
    except Exception as exc:
        report['status'] = 'failed'
        report['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        report['finished_at'] = datetime.now(timezone.utc).isoformat()
        save()
        print(f'Validation report: {args.output}', flush=True)


if __name__ == '__main__':
    main()

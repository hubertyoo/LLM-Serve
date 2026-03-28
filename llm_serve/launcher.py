from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml


RUNTIME_VLLM = "vllm"
RUNTIME_SGLANG = "sglang"
SUPPORTED_RUNTIMES = {RUNTIME_VLLM, RUNTIME_SGLANG}

ROOT_DIR = Path(__file__).resolve().parent.parent
LOGS_DIR = ROOT_DIR / "logs"
RUN_DIR = ROOT_DIR / "run"
STATE_DIR = RUN_DIR / "instances"


class LauncherError(RuntimeError):
    """Raised when the launcher cannot complete a requested action."""


@dataclass(slots=True)
class ModelConfig:
    config_path: Path
    runtime: str
    model_name: str
    model_path: str
    host: str
    port: int
    env: dict[str, str]
    runtime_args: dict[str, Any]


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _safe_slug(value: str) -> str:
    chars = [char.lower() if char.isalnum() else "_" for char in value]
    slug = "".join(chars).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug or "model"


def _ensure_dirs() -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    STATE_DIR.mkdir(parents=True, exist_ok=True)


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
    except FileNotFoundError as exc:
        raise LauncherError(f"Config file not found: {path}") from exc
    except yaml.YAMLError as exc:
        raise LauncherError(f"Invalid YAML in {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise LauncherError(f"Config must be a mapping: {path}")
    return data


def load_model_config(config_path: str | Path) -> ModelConfig:
    path = Path(config_path).expanduser().resolve()
    data = _load_yaml(path)

    runtime = str(data.get("runtime", "")).strip().lower()
    if runtime not in SUPPORTED_RUNTIMES:
        raise LauncherError(
            f"Unsupported runtime {runtime!r}. Supported: {', '.join(sorted(SUPPORTED_RUNTIMES))}"
        )

    model_name = str(data.get("model_name", "")).strip()
    model_path = str(data.get("model_path", "")).strip()
    if not model_name:
        raise LauncherError(f"'model_name' is required in {path}")
    if not model_path:
        raise LauncherError(f"'model_path' is required in {path}")

    server = data.get("server", {})
    if server is None:
        server = {}
    if not isinstance(server, dict):
        raise LauncherError(f"'server' must be a mapping in {path}")

    env = data.get("env", {})
    if env is None:
        env = {}
    if not isinstance(env, dict):
        raise LauncherError(f"'env' must be a mapping in {path}")

    runtime_args = data.get("runtime_args", {})
    if runtime_args is None:
        runtime_args = {}
    if not isinstance(runtime_args, dict):
        raise LauncherError(f"'runtime_args' must be a mapping in {path}")

    return ModelConfig(
        config_path=path,
        runtime=runtime,
        model_name=model_name,
        model_path=model_path,
        host=str(server.get("host", "0.0.0.0")),
        port=int(server.get("port", 8000)),
        env={str(key): str(value) for key, value in env.items()},
        runtime_args=runtime_args,
    )


def _flag_name(name: str) -> str:
    return f"--{name.replace('_', '-')}"


def _append_flag(command: list[str], key: str, value: Any) -> None:
    if value is None:
        return

    flag = _flag_name(key)
    if isinstance(value, bool):
        if value:
            command.append(flag)
        return

    if isinstance(value, (list, tuple)):
        for item in value:
            command.extend([flag, str(item)])
        return

    if isinstance(value, dict):
        command.extend([flag, json.dumps(value, separators=(",", ":"))])
        return

    command.extend([flag, str(value)])


def build_command(config: ModelConfig) -> list[str]:
    if config.runtime == RUNTIME_VLLM:
        command = [
            "vllm",
            "serve",
            config.model_path,
            "--host",
            config.host,
            "--port",
            str(config.port),
        ]
    elif config.runtime == RUNTIME_SGLANG:
        command = [
            sys.executable,
            "-m",
            "sglang.launch_server",
            "--model-path",
            config.model_path,
            "--host",
            config.host,
            "--port",
            str(config.port),
        ]
    else:
        raise LauncherError(f"Unsupported runtime {config.runtime}")

    for key, value in config.runtime_args.items():
        _append_flag(command, key, value)

    return command


def _state_files() -> list[Path]:
    if not STATE_DIR.exists():
        return []
    return sorted(STATE_DIR.glob("*.json"))


def _read_state(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (json.JSONDecodeError, FileNotFoundError):
        return None


def _write_state(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)


def _remove_state(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def _process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _cleanup_stale_states() -> None:
    for state_file in _state_files():
        payload = _read_state(state_file)
        if not payload:
            _remove_state(state_file)
            continue
        pid = payload.get("pid")
        if not isinstance(pid, int) or not _process_alive(pid):
            _remove_state(state_file)


def _matching_states(model_name: str) -> list[tuple[Path, dict[str, Any]]]:
    target = model_name.strip()
    normalized_target = _safe_slug(target)
    matches: list[tuple[Path, dict[str, Any]]] = []
    for state_file in _state_files():
        payload = _read_state(state_file)
        if not payload:
            continue
        state_name = str(payload.get("model_name", "")).strip()
        if state_name == target or _safe_slug(state_name) == normalized_target:
            matches.append((state_file, payload))
    return matches


def _kill_process_group(pid: int, grace_period: float) -> str:
    try:
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        return "already_exited"

    deadline = time.time() + grace_period
    while time.time() < deadline:
        if not _process_alive(pid):
            return "terminated"
        time.sleep(0.25)

    try:
        os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        return "terminated"
    return "killed"


def start_model(config_path: str | Path) -> dict[str, Any]:
    _ensure_dirs()
    _cleanup_stale_states()

    config = load_model_config(config_path)
    command = build_command(config)

    timestamp = _utc_now().strftime("%Y%m%dT%H%M%SZ")
    model_slug = _safe_slug(config.model_name)
    instance_id = f"{model_slug}_{config.runtime}_{config.port}_{timestamp}"
    log_dir = LOGS_DIR / model_slug
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{instance_id}.log"
    state_path = STATE_DIR / f"{instance_id}.json"

    env = os.environ.copy()
    env.update(config.env)

    with log_path.open("a", encoding="utf-8") as log_handle:
        process = subprocess.Popen(  # noqa: S603
            command,
            cwd=ROOT_DIR,
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )

    payload = {
        "instance_id": instance_id,
        "pid": process.pid,
        "model_name": config.model_name,
        "model_slug": model_slug,
        "runtime": config.runtime,
        "config_path": str(config.config_path),
        "log_path": str(log_path),
        "port": config.port,
        "started_at": _utc_now().isoformat(),
        "command": command,
    }
    _write_state(state_path, payload)
    return payload


def stop_model(model_name: str, grace_period: float = 15.0) -> list[dict[str, Any]]:
    _ensure_dirs()
    _cleanup_stale_states()

    results: list[dict[str, Any]] = []
    matches = _matching_states(model_name)
    for state_file, payload in matches:
        pid = payload.get("pid")
        status = "invalid_state"
        if isinstance(pid, int):
            status = _kill_process_group(pid, grace_period)
        _remove_state(state_file)
        results.append(
            {
                "instance_id": payload.get("instance_id"),
                "model_name": payload.get("model_name"),
                "runtime": payload.get("runtime"),
                "pid": pid,
                "status": status,
                "log_path": payload.get("log_path"),
            }
        )
    return results


def list_models() -> list[dict[str, Any]]:
    _ensure_dirs()
    _cleanup_stale_states()
    rows: list[dict[str, Any]] = []
    for state_file in _state_files():
        payload = _read_state(state_file)
        if payload:
            rows.append(payload)
    return rows


def _render_payload(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, ensure_ascii=True)


def _format_list(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "No running model servers."

    lines = []
    for row in rows:
        lines.append(
            " | ".join(
                [
                    f"model={row.get('model_name')}",
                    f"runtime={row.get('runtime')}",
                    f"pid={row.get('pid')}",
                    f"port={row.get('port')}",
                    f"log={row.get('log_path')}",
                ]
            )
        )
    return "\n".join(lines)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch and manage vLLM/SGLang model servers")
    subparsers = parser.add_subparsers(dest="command", required=True)

    start_parser = subparsers.add_parser("start", help="Start a model server from a YAML config")
    start_parser.add_argument("config", help="Path to the YAML config")
    start_parser.add_argument("--json", action="store_true", help="Print JSON output")

    stop_parser = subparsers.add_parser("stop", help="Stop running servers by model name")
    stop_parser.add_argument("model_name", help="Model name from YAML")
    stop_parser.add_argument(
        "--grace-period",
        type=float,
        default=15.0,
        help="Seconds to wait after SIGTERM before SIGKILL",
    )
    stop_parser.add_argument("--json", action="store_true", help="Print JSON output")

    list_parser = subparsers.add_parser("list", help="List running servers")
    list_parser.add_argument("--json", action="store_true", help="Print JSON output")

    render_parser = subparsers.add_parser("render-command", help="Render the launch command for a config")
    render_parser.add_argument("config", help="Path to the YAML config")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = make_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "start":
            payload = start_model(args.config)
            if args.json:
                print(_render_payload(payload))
            else:
                print(f"Started {payload['model_name']} ({payload['runtime']}) on pid {payload['pid']}")
                print(f"Log file: {payload['log_path']}")
            return 0

        if args.command == "stop":
            results = stop_model(args.model_name, args.grace_period)
            if args.json:
                print(_render_payload({"stopped": results}))
            elif results:
                for item in results:
                    print(
                        f"Stopped {item['model_name']} ({item['runtime']}) pid={item['pid']} status={item['status']}"
                    )
            else:
                print(f"No running server matched model name: {args.model_name}")
            return 0

        if args.command == "list":
            rows = list_models()
            if args.json:
                print(_render_payload({"running": rows}))
            else:
                print(_format_list(rows))
            return 0

        if args.command == "render-command":
            config = load_model_config(args.config)
            print(shlex.join(build_command(config)))
            return 0
    except LauncherError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

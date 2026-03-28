import json
import os
import shlex
import shutil
import signal
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import llm_serve.launcher as launcher
from llm_serve.launcher import RUNTIME_SGLANG, RUNTIME_VLLM, build_command, load_model_config, stop_model


class LauncherCommandTests(unittest.TestCase):
    def _write_config(self, body: str) -> Path:
        tmp_dir = Path(tempfile.mkdtemp(prefix="llm-serve-test-"))
        config_path = tmp_dir / "model.yaml"
        config_path.write_text(body, encoding="utf-8")
        self.addCleanup(lambda: config_path.unlink(missing_ok=True))
        self.addCleanup(lambda: shutil.rmtree(tmp_dir, ignore_errors=True))
        return config_path

    def test_vllm_command_includes_tool_call_flags(self) -> None:
        config = self._write_config(
            """
runtime: vllm
model_name: qwen3.5-4b
model_path: Qwen/Qwen3.5-4B
server:
  host: 0.0.0.0
  port: 8000
runtime_args:
  tensor_parallel_size: 1
  max_model_len: 262144
  reasoning_parser: qwen3
  enable_auto_tool_choice: true
  tool_call_parser: qwen3_coder
"""
        )

        model = load_model_config(config)
        command = build_command(model)
        rendered = shlex.join(command)

        self.assertEqual(model.runtime, RUNTIME_VLLM)
        self.assertIn("vllm serve Qwen/Qwen3.5-4B", rendered)
        self.assertIn("--tensor-parallel-size 1", rendered)
        self.assertIn("--enable-auto-tool-choice", rendered)
        self.assertIn("--tool-call-parser qwen3_coder", rendered)

    def test_sglang_command_includes_tool_call_flags(self) -> None:
        config = self._write_config(
            """
runtime: sglang
model_name: qwen3.5-4b
model_path: Qwen/Qwen3.5-4B
server:
  host: 0.0.0.0
  port: 8000
runtime_args:
  tp_size: 1
  mem_fraction_static: 0.8
  context_length: 262144
  reasoning_parser: qwen3
  tool_call_parser: qwen3_coder
"""
        )

        model = load_model_config(config)
        command = build_command(model)
        rendered = shlex.join(command)

        self.assertEqual(model.runtime, RUNTIME_SGLANG)
        self.assertIn("-m sglang.launch_server", rendered)
        self.assertIn("--tp-size 1", rendered)
        self.assertIn("--mem-fraction-static 0.8", rendered)
        self.assertIn("--tool-call-parser qwen3_coder", rendered)

    def test_stop_model_only_kills_requested_model(self) -> None:
        root_dir = Path(tempfile.mkdtemp(prefix="llm-serve-root-"))
        logs_dir = root_dir / "logs"
        run_dir = root_dir / "run"
        state_dir = run_dir / "instances"
        state_dir.mkdir(parents=True, exist_ok=True)
        self.addCleanup(lambda: shutil.rmtree(root_dir, ignore_errors=True))

        proc_a = subprocess.Popen(["sleep", "60"], start_new_session=True)  # noqa: S603,S607
        proc_b = subprocess.Popen(["sleep", "60"], start_new_session=True)  # noqa: S603,S607

        def cleanup_process(process: subprocess.Popen[bytes]) -> None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                pass

        self.addCleanup(lambda: cleanup_process(proc_a))
        self.addCleanup(lambda: cleanup_process(proc_b))

        (state_dir / "a.json").write_text(
            json.dumps(
                {
                    "instance_id": "a",
                    "pid": proc_a.pid,
                    "model_name": "model-a",
                    "runtime": "vllm",
                    "log_path": str(logs_dir / "a.log"),
                }
            ),
            encoding="utf-8",
        )
        (state_dir / "b.json").write_text(
            json.dumps(
                {
                    "instance_id": "b",
                    "pid": proc_b.pid,
                    "model_name": "model-b",
                    "runtime": "sglang",
                    "log_path": str(logs_dir / "b.log"),
                }
            ),
            encoding="utf-8",
        )

        with (
            mock.patch.object(launcher, "ROOT_DIR", root_dir),
            mock.patch.object(launcher, "LOGS_DIR", logs_dir),
            mock.patch.object(launcher, "RUN_DIR", run_dir),
            mock.patch.object(launcher, "STATE_DIR", state_dir),
        ):
            results = stop_model("model-a", grace_period=1.0)

        proc_a.wait(timeout=2)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["model_name"], "model-a")
        self.assertIsNotNone(proc_a.poll())
        self.assertIsNone(proc_b.poll())


if __name__ == "__main__":
    unittest.main()

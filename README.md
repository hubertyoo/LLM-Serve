# LLM Serve

A general-purpose launcher for `vLLM` and `SGLang`:

- Define each model instance in a YAML configuration file.
- Specify either `vllm` or `sglang` as the runtime in the YAML file.
- Use a single Python launcher to start services, write logs, and track process IDs and state.
- Stop a specific model by `model_name` using a Bash script without stopping other services.

## Installation

```bash
cd /home/jys3649/projects/LLM-Serve
python3 -m pip install -e .
```

If you prefer `uv`:

```bash
cd /home/jys3649/projects/LLM-Serve
uv pip install -e .
```

## Configuration Format

```yaml
runtime: vllm        # or sglang
model_name: qwen3.5-4b
model_path: Qwen/Qwen3.5-4B

server:
  host: 0.0.0.0
  port: 8000

env:
  CUDA_VISIBLE_DEVICES: "0"

runtime_args:
  tensor_parallel_size: 1
  max_model_len: 262144
  reasoning_parser: qwen3
  enable_auto_tool_choice: true
  tool_call_parser: qwen3_coder
```

Configuration rules:

- `server.port` and `server.host` are automatically mapped to the corresponding launch arguments.
- Keys in `runtime_args` are automatically converted from `snake_case` to CLI flags, for example, `max_model_len -> --max-model-len`.
- Boolean values set to `true` become standalone flags, for example, `enable_auto_tool_choice -> --enable-auto-tool-choice`.
- `model_name` is used to organize log directories, index runtime state, and stop services by model name.

## Included Examples

- `configs/models/qwen/qwen3_5_4b_vllm_tool.yaml`
- `configs/models/qwen/qwen3_5_4b_sglang_tool.yaml`

Both configurations follow the official `Qwen3.5-4B` tool-calling launch examples, with MTP disabled.

## Starting a Service

```bash
./scripts/start_model.sh configs/models/qwen/qwen3_5_4b_vllm_tool.yaml
```

Alternatively:

```bash
python3 -m llm_serve.launcher start configs/models/qwen/qwen3_5_4b_sglang_tool.yaml
```

The launcher will:

- Run the service in the background.
- Write logs to `logs/<model_name>/...log`.
- Write runtime state to `run/instances/*.json`.

## Stopping a Service

Stop a service by model name:

```bash
./scripts/stop_model.sh qwen3.5-4b
```

Alternatively:

```bash
python3 -m llm_serve.launcher stop qwen3.5-4b
```

This stops only the process group associated with `model_name: qwen3.5-4b`, leaving other model services running.

## Listing Running Services

```bash
python3 -m llm_serve.launcher list
```

## Rendering the Launch Command

```bash
python3 -m llm_serve.launcher render-command configs/models/qwen/qwen3_5_4b_vllm_tool.yaml
```

## Logs and State Files

- Log directory: `logs/`
- Runtime state directory: `run/instances/`

Each state file records:

- `pid`
- `model_name`
- `runtime`
- `config_path`
- `log_path`
- The actual launch command

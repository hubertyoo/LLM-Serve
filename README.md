# LLM Serve

一个通用的 `vLLM` / `SGLang` 启动仓库：

- 每个模型实例用一个 YAML 配置文件描述
- YAML 中声明 runtime 是 `vllm` 还是 `sglang`
- 统一通过 Python launcher 启动、记录日志、记录 pid/state
- 提供按 `model_name` 精准停止的 bash 脚本，不会把所有服务一起杀掉

## 安装

```bash
cd /home/jys3649/projects/LLM-Serve
python3 -m pip install -e .
```

如果你更习惯 `uv`：

```bash
cd /home/jys3649/projects/LLM-Serve
uv pip install -e .
```

## 配置格式

```yaml
runtime: vllm        # 或 sglang
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

规则：

- `server.port` 和 `server.host` 会自动映射成对应启动参数
- `runtime_args` 会自动从 `snake_case` 转成 CLI flag，例如 `max_model_len -> --max-model-len`
- 布尔值 `true` 会转成纯开关，例如 `enable_auto_tool_choice -> --enable-auto-tool-choice`
- `model_name` 用来做日志目录、运行态索引、以及按模型名停止

## 已提供的样例

- `configs/models/qwen/qwen3_5_4b_vllm_tool.yaml`
- `configs/models/qwen/qwen3_5_4b_sglang_tool.yaml`

这两个配置都参考了你给的官方 `Qwen3.5-4B` tool-call 启动方式，没有加 MTP。

## 启动

```bash
./scripts/start_model.sh configs/models/qwen/qwen3_5_4b_vllm_tool.yaml
```

或者：

```bash
python3 -m llm_serve.launcher start configs/models/qwen/qwen3_5_4b_sglang_tool.yaml
```

启动后会：

- 把服务放到后台运行
- 把日志写到 `logs/<model_name>/...log`
- 把运行态信息写到 `run/instances/*.json`

## 停止

按模型名停止：

```bash
./scripts/stop_model.sh qwen3.5-4b
```

或者：

```bash
python3 -m llm_serve.launcher stop qwen3.5-4b
```

这会只停止 `model_name: qwen3.5-4b` 对应的进程组，不影响其他模型服务。

## 查看运行中的服务

```bash
python3 -m llm_serve.launcher list
```

## 仅渲染启动命令

```bash
python3 -m llm_serve.launcher render-command configs/models/qwen/qwen3_5_4b_vllm_tool.yaml
```

## 日志和状态文件

- 日志目录：`logs/`
- 运行态目录：`run/instances/`

状态文件中会记录：

- `pid`
- `model_name`
- `runtime`
- `config_path`
- `log_path`
- 实际启动命令

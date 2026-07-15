# L20 无外网部署与冒烟测试

本文只准备离线部署和从 1 条虚构样本开始的验证路径，不声称复现论文指标。L20 上首先复用本地已验证的小模型；只有在显存、上下文与吞吐均实测稳定后才评估更大的模型。

## 1. 有网电脑预下载

在与服务器 Linux、Python 和 CUDA ABI 匹配的 Linux/WSL 环境准备一个传输目录。只下载最终确定的一个 Hugging Face 模型快照，建议先使用 Qwen3-4B 的非 GGUF Transformers 权重；不要把 Ollama 的 blob 直接当作 vLLM 模型目录。

需要预下载并上传：

- 模型完整快照：配置、tokenizer、safetensors 权重和 generation config；
- 与服务器 Python/CUDA 匹配的 vLLM、PyTorch 及其完整 wheel 依赖集合；
- 本仓库的 `local-small-llm` 分支代码；
- 如后续正式复现才需要的 EHR 检查点、已保存专家输出、MSD/MedCPT/GatorTron；首次冒烟不需要这些；
- MIMIC 只能按数据使用协议单独传入受控存储，绝不能放入代码包。

示例（有网 Linux/WSL；实际版本需按 L20 驱动兼容矩阵锁定）：

```bash
python3 -m venv download-env
source download-env/bin/activate
python -m pip install --upgrade pip huggingface_hub
hf download Qwen/Qwen3-4B --revision <MODEL_COMMIT> \
  --local-dir transfer/models/Qwen3-4B
python -m pip download --dest transfer/wheels \
  vllm torch transformers safetensors sentencepiece
```

如果模型仓库要求接受许可证或认证，只在有网机器使用 Hugging Face token；不要复制 token、缓存凭据或 shell history 到服务器传输包。

`hf download --local-dir` 与 `--revision` 的行为以 [Hugging Face 官方下载文档](https://huggingface.co/docs/huggingface_hub/guides/download) 为准；把 `<MODEL_COMMIT>` 替换为审核过的不可变 commit，并记录 [Qwen3-4B 模型卡](https://huggingface.co/Qwen/Qwen3-4B) 的许可证和 Transformers 版本要求。

## 2. SHA256 完整性验证

在有网电脑生成 SHA256 清单，并把清单与压缩包分开保存：

```bash
cd transfer
find models wheels -type f -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS
sha256sum -c SHA256SUMS
tar --zstd -cf colacare-offline-assets.tar.zst models wheels SHA256SUMS
sha256sum colacare-offline-assets.tar.zst > colacare-offline-assets.tar.zst.sha256
```

Windows PowerShell 可对传输包执行 `Get-FileHash -Algorithm SHA256`。上传后先验证外层压缩包，再解压并运行 `sha256sum -c SHA256SUMS`。任何一项失败都应重新传输，不能跳过校验。

## 3. 上传与服务器目录

通过组织允许的 SCP、SFTP、移动介质或堡垒机上传到服务器暂存区。建议目标布局：

```text
/srv/colacare/                 # Git 工作树，不含数据和权重
/srv/models/Qwen3-4B/         # 只读模型快照
/srv/offline-wheels/           # wheel 仓库
/srv/colacare-private/         # 检查点、EHR 输出和受限数据（仓库外）
```

先在暂存区完成 SHA256 校验，再移动到最终目录。为模型和私有数据设置最小权限；患者级输出应进入受控目录，而不是仓库下的 `response/`。

## 4. 离线安装和离线模式

在服务器创建独立环境并禁止 pip 访问索引：

```bash
python3 -m venv /srv/venvs/colacare
source /srv/venvs/colacare/bin/activate
python -m pip install --no-index --find-links /srv/offline-wheels \
  torch vllm transformers safetensors sentencepiece
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
```

`HF_HUB_OFFLINE` 与 `TRANSFORMERS_OFFLINE` 防止运行时尝试访问 Hugging Face。模型必须使用本地绝对目录。安装后记录 `pip freeze`、驱动、CUDA、GPU 与 commit hash；不要把整个虚拟环境提交到 Git。

## 5. 使用本地目录启动 vLLM

复制示例配置但不要提交实际文件：

```bash
cd /srv/colacare
cp configs/l20.env.example configs/l20.env
chmod 600 configs/l20.env
# 编辑 MODEL_DIR、端口及服务名；不要填写外部 API 密钥
bash scripts/start_vllm_server.sh configs/l20.env
```

启动脚本使用 `vllm serve /srv/models/...`，不使用 Hugging Face 仓库 ID。默认绑定 `127.0.0.1`，上下文 4096、并发序列 1、显存利用率上限 0.85。若需要跨主机访问，必须先经过防火墙、认证和组织安全审批，而不是直接改成公网监听。

这些参数对应 [vLLM 官方 `serve` CLI](https://docs.vllm.ai/en/stable/cli/serve/) 的 `--max-model-len`、`--max-num-seqs` 与 `--gpu-memory-utilization`。上传前应根据实际离线 wheel 版本再次运行 `vllm serve --help`，因为 CLI 会随版本变化。

## 6. OpenAI 兼容调用和冒烟命令

vLLM 提供 OpenAI 兼容的 `/v1/chat/completions`。在另一个终端运行：

```bash
cd /srv/colacare
source /srv/venvs/colacare/bin/activate
set -a
source configs/l20.env
set +a
bash scripts/run_l20_smoke_test.sh configs/l20.env
```

脚本依次实际运行环境检查、独立 JSON API 检查、单 DoctorAgent 虚构样本链。结果保存到被忽略的 `artifacts/smoke/single/`。也可用 `curl` 调用 `${LLM_BASE_URL}/chat/completions`，但不要在命令历史中放真实密钥。

## 7. 上下文与并发控制

首次固定 `VLLM_MAX_MODEL_LEN=4096`、`VLLM_MAX_NUM_SEQS=1`、`LLM_MAX_TOKENS=256`、batch size 1、temperature 0。多个 Agent 必须顺序执行。用 `nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv -l 1` 观察峰值；OOM 时依次降低显存利用率、上下文、输出长度，确认没有残留服务，再重试 1 条样本。

只有单样本稳定后才把上下文提高到 8192；每次只改变一个参数并记录峰值、延迟和错误。升级模型也遵循相同原则。

## 8. 从 1 条样本逐步扩量

逐步扩量顺序为：1 条虚构样本 → 10 条去标识化开发样本 → 100 条 → 一个小分片 → 全量。每一级先验证合法 JSON 比率、失败重试、显存峰值和平均延迟，再进入下一级。先接入“已保存 EHR 专家输出”，不要在多智能体运行时重复训练专家模型。

正式数据不得作为命令行日志或异常消息输出。每个分片使用稳定样本 ID、输入清单 hash、模型目录 hash、Git commit 与配置 hash，便于审计。

## 9. 断点续跑

批处理应按样本写独立临时文件，校验通过后原子重命名为 `<synthetic-or-pseudonymous-id>.json`。重启时只跳过同时满足以下条件的样本：输出存在、JSON Schema 验证通过、配置 hash 与当前运行一致。失败项写入单独的非患者级索引，记录错误类型和尝试次数；不要以一个不断追加的大 JSON 文件作为唯一状态。

本仓库当前单样本脚本已使用原子 JSON 写入。正式批处理适配器仍需在获得真实“已保存 EHR 专家输出”的格式后实现和验证。

## 10. 防止权重和受限数据进入 GitHub

仓库 `.gitignore` 已覆盖 `models/`、`data/`、`datasets/`、MIMIC、模型权重、患者输出、日志、缓存与大型中间结果。服务器上的实际 `configs/l20.env` 也必须保持未跟踪，只提交 `configs/l20.env.example`。

提交前运行：

```bash
git status --short --ignored
git diff --cached --name-only
find . -type f -size +50M -not -path './.git/*'
git check-ignore models/example.safetensors data/mimic.csv artifacts/smoke/single/result.json configs/l20.env
```

如果敏感文件已进入暂存区，立即取消暂存并调查；不要依赖“稍后删除”。不得上传模型权重、MIMIC、密钥、患者级输出或大型生成文件，也不得向官方上游创建 PR，除非另有明确授权。

## 故障记录最小模板

每次阻塞记录完整错误、已尝试方法、当前判断和最小下一步。常见检查：`nvidia-smi` 是否可见、wheel 是否与 Python/CUDA 匹配、模型 SHA256 是否通过、服务名是否与 `LLM_MODEL_NAME` 一致、端口是否仅被一个 vLLM 进程占用。首次 L20 真机启动无法在 Windows 本地代替验证，因此本地只能验证脚本语法和同一 OpenAI 兼容冒烟链；L20 的实测显存与吞吐必须在目标机器记录。

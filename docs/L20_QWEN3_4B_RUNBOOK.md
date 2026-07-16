# L20 Qwen3-4B-Instruct-2507 离线运行手册

本手册只覆盖固定模型快照、离线传输和从 1 条虚构样本开始的 L20 验证。所有路径均由变量
定义；不要把真实 `.env`、模型、日志、患者输出或受限数据放入 Git。

## 0. 固定版本和目录

官方模型：[Qwen/Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)。
本手册固定 2026-07-16 核对的不可变 revision：

```bash
export MODEL_ID='Qwen/Qwen3-4B-Instruct-2507'
export MODEL_REVISION='cdbee75f17c01a7cc42f958dc650907174af0554'
export COLACARE_COMMIT='5374852ed2a082a37ada3102c6b91d0c94eda232'
```

该快照约 8.061 GB（十进制，7.51 GiB），其中 3 个 BF16 safetensors 权重约 8.045 GB。
本地 Ollama `qwen3:4b` 是量化缓存，只用于 Agent 链路开发；它不是本快照的文件来源，
不能复制 Ollama blob 代替 Transformers 模型目录。正式 L20 基线、消融和对比实验必须固定
同一 revision、chat template、提示词、JSON Schema、context 4096、max new tokens 256 和
确定性生成参数。

## 1. 有网机器准备下载环境

在有足够空间的 Linux/WSL 有网机器克隆本分支。以下环境只用于下载，不会进入代码归档：

```bash
export WORK_ROOT="$PWD/colacare-offline-work"
export REPO_DIR="$PWD/ColaCare"
export MODEL_DIR="$WORK_ROOT/models/Qwen3-4B-Instruct-2507"
export PACKAGE_DIR="$WORK_ROOT/packages"
python3 -m venv "$WORK_ROOT/download-venv"
source "$WORK_ROOT/download-venv/bin/activate"
python -m pip install --upgrade pip huggingface_hub
mkdir -p "$MODEL_DIR" "$PACKAGE_DIR"
```

如模型仓库以后要求认证，只在有网机器使用 Hugging Face 凭据；不要复制 token、缓存凭据、
shell history 或 `.env` 到 L20/代码包。

## 2. 固定 revision 并下载完整快照

先确认代码是计划分支，再运行唯一下载命令：

```bash
cd "$REPO_DIR"
git switch l20-qwen3-4b
git rev-parse HEAD
python scripts/prepare_qwen3_snapshot.py \
  --model-id "$MODEL_ID" \
  --revision "$MODEL_REVISION" \
  --output-dir "$MODEL_DIR"
```

脚本调用 `huggingface_hub.snapshot_download`，不读取或转换 Ollama 缓存。不要用 `main` 代替
revision；若未来更新 revision，应建立新资产目录和独立实验批次。

## 3. 检查完整模型快照

脚本会拒绝缺少 `config.json`、tokenizer 配置/词表、safetensors、分片索引或 chat template
的快照，并生成 `MODEL_MANIFEST.json` 与 `SHA256SUMS`。再次手工检查：

```bash
cd "$MODEL_DIR"
sha256sum -c SHA256SUMS
test -s config.json
test -s tokenizer_config.json
test -s tokenizer.json
test -s model.safetensors.index.json
find . -maxdepth 1 -type f -name '*.safetensors' -print
python - <<'PY'
import json
from pathlib import Path
root = Path('.')
config = json.loads((root / 'config.json').read_text())
tokenizer = json.loads((root / 'tokenizer_config.json').read_text())
assert config['model_type'] == 'qwen3'
assert config.get('torch_dtype') == 'bfloat16'
assert tokenizer.get('chat_template') or (root / 'chat_template.jinja').is_file()
print('model_type=', config['model_type'])
print('chat_template=present')
PY
```

预期权重文件为 `model-00001-of-00003.safetensors` 至
`model-00003-of-00003.safetensors`，索引为 `model.safetensors.index.json`。不要只看文件名；
必须让 SHA256 全部通过。

## 4. 打包模型和指定 commit 的代码

脚本从 Git 对象中的指定 commit 生成代码包，不会把当前工作树、`.git` 或未跟踪文件打入：

```bash
cd "$REPO_DIR"
bash scripts/package_l20_assets.sh \
  --model-dir "$MODEL_DIR" \
  --repo-dir "$REPO_DIR" \
  --commit "$COLACARE_COMMIT" \
  --output-dir "$PACKAGE_DIR" \
  --split-size 2G
```

它生成代码 `.tar.gz`、模型 `.tar.gz`、2GB 分片、`OFFLINE_ASSET_SHA256SUMS` 和
`OFFLINE_ASSET_MANIFEST.txt`。模型包压缩率通常很低，准备目录至少预留约 20 GiB，若同时
保留原始模型、完整 tar.gz 和分片则建议 30 GiB。

## 5. 本地校验包和分片

```bash
cd "$PACKAGE_DIR"
sha256sum -c OFFLINE_ASSET_SHA256SUMS
sed -n '1,200p' OFFLINE_ASSET_MANIFEST.txt
find . -maxdepth 1 -type f -printf '%f %s bytes\n' | sort
```

保留完整模型包可用于本地总哈希复核；实际网页上传可只选择代码包、所有模型分片、
`OFFLINE_ASSET_SHA256SUMS` 和 `OFFLINE_ASSET_MANIFEST.txt`。

## 6. 网页 VSCode 上传

在 L20 网页 VSCode 中先建立仓库外暂存目录：

```bash
export TRANSFER_ROOT="$HOME/colacare_transfer"
export INSTALL_ROOT="$HOME/colacare_offline"
export SERVER_REPO="$INSTALL_ROOT/ColaCare"
export MODEL_PATH="$INSTALL_ROOT/models/Qwen3-4B-Instruct-2507"
mkdir -p "$TRANSFER_ROOT" "$INSTALL_ROOT/models"
df -h "$HOME" "$TRANSFER_ROOT"
```

通过 VSCode 文件浏览器把上述传输文件上传到 `$TRANSFER_ROOT`。网页上传是人工操作；脚本
不会启动实例、修改资源或访问外网。上传前先确认个人配额，因为审计中 `quota` 不可用。

## 7. L20 校验模型分片

不要在校验前合并。进入暂存目录，只验证实际上传的分片和代码包：

```bash
cd "$TRANSFER_ROOT"
grep -E '(\.part-[0-9]+|colacare-.*\.tar\.gz)$' OFFLINE_ASSET_SHA256SUMS \
  | sha256sum -c -
```

任一项出现 `FAILED` 就停止并重新上传对应文件；不要跳过、重命名后继续或用新哈希覆盖旧哈希。

## 8. 合并模型包并校验

从 manifest 读取模型归档名，按数值后缀顺序合并：

```bash
cd "$TRANSFER_ROOT"
export MODEL_ARCHIVE="$(awk -F= '$1=="MODEL_ARCHIVE" {print $2}' OFFLINE_ASSET_MANIFEST.txt)"
test -n "$MODEL_ARCHIVE"
cat "${MODEL_ARCHIVE}.part-"* > "$MODEL_ARCHIVE"
grep -F "  $MODEL_ARCHIVE" OFFLINE_ASSET_SHA256SUMS | sha256sum -c -
```

只有完整归档哈希通过后才解压代码和模型：

```bash
export CODE_ARCHIVE="$(awk -F= '$1=="CODE_ARCHIVE" {print $2}' OFFLINE_ASSET_MANIFEST.txt)"
mkdir -p "$SERVER_REPO" "$INSTALL_ROOT/models"
tar -xzf "$CODE_ARCHIVE" -C "$INSTALL_ROOT"
tar -xzf "$MODEL_ARCHIVE" -C "$INSTALL_ROOT/models"
test -d "$SERVER_REPO"
test -d "$MODEL_PATH"
```

归档均含稳定顶层目录；不要以 root 身份解压，也不要覆盖平台共享 Conda 环境。

## 9. 设置完全离线模式

每个终端都执行：

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME="$INSTALL_ROOT/hf-cache"
export MODEL_PATH="$INSTALL_ROOT/models/Qwen3-4B-Instruct-2507"
```

模型参数必须是本地目录，不得是 `Qwen/...` 仓库 ID。脚本也会将
`local_files_only=True` 传给 tokenizer 和模型加载器。

## 10. 使用现有 pytorch Conda 环境

```bash
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate /opt/miniconda3/envs/pytorch
command -v python
python -c 'import torch, transformers; print(torch.__version__); print(transformers.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0)); print(torch.cuda.is_bf16_supported())'
```

预期 Python 3.10.13、torch 2.7.1+cu126、Transformers 4.57.6、CUDA/BF16 为 True、GPU 为
NVIDIA L20。第一阶段不运行 `pip install` 或 `conda install`；OpenAI Python 客户端和 vLLM
都不是直接 Transformers 冒烟的依赖。

## 11. 解压后校验模型、磁盘、GPU 和 Python

```bash
cd "$SERVER_REPO"
bash scripts/verify_l20_assets.sh \
  --model-dir "$MODEL_PATH" \
  --python-bin /opt/miniconda3/envs/pytorch/bin/python
```

该脚本只读检查，不联网。它会验证模型内部 `SHA256SUMS`、关键文件、磁盘空间、`nvidia-smi`、
PyTorch CUDA、BF16 和 Transformers；任一关键项失败都停止。

## 12. 第一条直接 Transformers 推理

```bash
cd "$SERVER_REPO"
MODEL_PATH="$MODEL_PATH" \
  /opt/miniconda3/envs/pytorch/bin/python scripts/test_l20_transformers.py \
  --model-path "$MODEL_PATH" \
  --fixture tests/fixtures/synthetic_patient.json \
  --output-dir artifacts/l20_smoke
```

固定 BF16、`cuda:0`、`max_new_tokens=256`、`do_sample=False`、最大输入 4096 token。脚本使用
模型 chat template，只接受包含 `risk_probability`、`prediction`、`reasoning_summary` 的简短
JSON；解析失败最多重试一次，不请求或保存完整思维过程。

## 13. 运行 ColaCare 虚构样本一键链路

一键脚本包含环境激活、离线校验、直接 Transformers 冒烟、日志和最终 `nvidia-smi`：

```bash
cd "$SERVER_REPO"
MODEL_PATH="$MODEL_PATH" bash scripts/run_l20_transformers_smoke.sh
```

输入必须保留 `synthetic: true`。输出在 `artifacts/l20_smoke/`，该目录被 `.gitignore` 排除。
这是直接 Transformers 单模型结构化预测，还不是正式多 Agent、RAG 或论文复现。

## 14. 查看显存、结果和日志

另开终端观察整卡：

```bash
nvidia-smi --query-gpu=timestamp,name,memory.total,memory.used,memory.free,utilization.gpu \
  --format=csv -l 1
```

任务完成后：

```bash
cd "$SERVER_REPO"
find artifacts/l20_smoke -maxdepth 1 -type f -printf '%f %s bytes\n' | sort
tail -n 200 artifacts/l20_smoke/run.log
```

Python 结果同时打印加载耗时、推理耗时、输入/输出 token、加载前后整卡显存和 PyTorch 推理
峰值。不要把 `artifacts/` 加入 Git。

## 15. 常见失败排查

| 现象 | 检查 | 最小处理 |
|---|---|---|
| SHA256 失败 | 分片名、大小、是否漏传 | 只重传失败分片，再从旧清单校验 |
| `No space left` | `df -h`、个人配额 | 清理自己的旧传输副本或向平台申请空间，不删共享文件 |
| `KeyError: qwen3` | `python -c 'import transformers; print(...)'` | 确认激活审计过的 4.57.6 环境；不要现场联网升级 |
| 尝试访问 HF | 三个 offline 变量、模型是否本地绝对路径 | 修正变量/路径；保留 `local_files_only=True` |
| CUDA 不可用 | `nvidia-smi`、`torch.cuda.is_available()` | 确认实例确有 L20 且进入 `pytorch` 环境；不要自行改实例 |
| BF16/设备错误 | device 名、Compute Capability、torch 版本 | 保存 traceback，停止；不要退化成未记录的精度设置 |
| OOM | 是否有其他用户进程、输入 token、残留进程 | 先清理自己的残留；维持 batch 1/4096/256 后再重试 |
| JSON 仍失败 | 保存的短原始模型文本和异常 | 不扩大重试；先改提示词/解析测试并形成新 commit |

每次故障记录完整错误、已尝试方法、当前判断和最小下一步。不要用删除校验、联网下载或修改
共享环境来“绕过”问题。

## 16. 浏览器关闭后保证任务继续

首选平台已有的 `tmux`，先只检查是否存在：

```bash
command -v tmux
tmux new -s colacare-l20
```

在 tmux 内运行一键命令；按 `Ctrl-b d` 分离，之后：

```bash
tmux attach -t colacare-l20
```

若平台没有 tmux 且管理员允许后台作业，可使用 `nohup`，记录精确 PID：

```bash
cd "$SERVER_REPO"
mkdir -p artifacts/l20_smoke
nohup env MODEL_PATH="$MODEL_PATH" bash scripts/run_l20_transformers_smoke.sh \
  > artifacts/l20_smoke/nohup.log 2>&1 &
echo $! > artifacts/l20_smoke/nohup.pid
```

这只保证终端断开后 shell 任务继续；平台若回收实例，进程仍会停止。长任务应遵循学校调度规则。

## 17. 清理自己的模型进程

直接 Transformers 脚本正常结束后会释放进程。若异常残留，先查看所有权和命令：

```bash
nvidia-smi
ps -fu "$USER" | grep -E 'test_l20_transformers|run_l20_transformers_smoke' | grep -v grep
```

只终止确认属于自己的精确 PID：

```bash
export TARGET_PID='<replace-with-your-pid>'
kill "$TARGET_PID"
sleep 3
ps -p "$TARGET_PID" -o pid,user,cmd
```

仍未退出时先保存日志再决定是否 `kill -TERM`；不要使用宽泛 `pkill python`，也不要终止其他
用户或平台进程。

## 18. 从 1 条样本逐步扩大

扩量顺序：1 条虚构 fixture → 10 条获准且去标识开发样本 → 100 条 → 小分片 → 全量。
每一级固定模型 revision、代码 commit、prompt/schema hash 和生成参数，先统计 JSON 成功率、
重试、耗时和峰值，再进入下一档。多 Agent 仍顺序调用；正式患者级结果放在仓库外受控目录。

批处理必须按样本原子写临时文件，校验后重命名；重启时只跳过同时满足输出存在、Schema
有效和配置 hash 一致的样本。当前脚本只实现 1 条虚构样本，不得假称已支持全量断点续跑。

## 19. 后续升级 vLLM（本阶段不执行）

只有直接 Transformers 冒烟稳定后，才在独立环境准备与驱动/CUDA/torch 匹配的离线 vLLM
wheels，运行 `vllm serve --help` 核对 CLI，再以同一本地模型目录、4096 上下文、序列 1
启动。Agent 端只需：

```bash
export LLM_BASE_URL='http://127.0.0.1:8000/v1'
export LLM_API_KEY='EMPTY'
export LLM_MODEL_NAME='qwen3-4b-local'
```

详细风险见 `docs/L20_VLLM_NEXT_STEP.md`。不要在共享 `pytorch` 环境安装 vLLM。

## 20. 绝对不能提交到 GitHub 的文件

- 模型权重、Ollama blob、HF cache、模型/资产 tar 包及分片；
- MIMIC、MSD 或任何真实/受限数据集；
- 患者级输入、输出、日志、检索语料和中间 embedding/index；
- API key、HF token、平台 cookie、`.env`、SSH key 和机器私有路径；
- checkpoints、wheels、Conda/venv、`artifacts/`、缓存和大型生成文件。

提交前执行：

```bash
cd "$SERVER_REPO"
git status --short --ignored
git ls-files | grep -Ei '(^|/)(models?|data|datasets|artifacts|logs?|patient_outputs)/|\.safetensors$|\.gguf$|(^|/)\.env$' && exit 1 || true
find . -type f -size +50M -not -path './.git/*' -print
git check-ignore models/example.safetensors data/mimic.csv artifacts/l20_smoke/result.json .env
```

四个哨兵路径必须都显示为 ignored；任何意外跟踪或大文件都应先停止调查。不要改写 Git 历史、
不要强推、不要合并 `main`，也不要向官方仓库创建 Pull Request。


# L20 vLLM 后续步骤与风险

## 当前结论

本阶段不执行 vLLM 安装、Conda 环境创建、模型服务启动或共享 `pytorch` 环境修改。第一道门
是让 `Qwen/Qwen3-4B-Instruct-2507` 在现有环境中以直接 Transformers BF16 完成 1 条虚构
样本，并记录加载/推理时间和显存。只有该门通过后，才评估 vLLM 的吞吐收益。

现有 `utils/llm_client.py` 使用标准 OpenAI 兼容 HTTP 合同，不依赖 OpenAI Python 包；其
行为测试已覆盖 vLLM 切换。未来服务通过验收后，Agent 侧只需设置：

```bash
export LLM_BASE_URL=http://127.0.0.1:8000/v1
export LLM_API_KEY=EMPTY
export LLM_MODEL_NAME=qwen3-4b-local
```

其余第一版参数继续使用 context 4096、max tokens 256、temperature 0、序列 1 和顺序 Agent。

## 以后获准时的离线步骤

以下是后续计划，不是本分支已经执行的命令或结果。

1. 保留已通过直接 Transformers 验证的同一个模型目录、revision、SHA256 和 chat template，
   不重新下载另一份“同名”模型。
2. 在 L20 记录驱动、Python、glibc、Compute Capability 和现有 torch/CUDA 版本；根据所选
   vLLM release 的官方安装矩阵选择兼容组合。模型卡建议 vLLM 0.8.5 或更高，但不能仅凭
   这个下限假设任意当前 wheel 与 torch 2.7.1+cu126 兼容。
3. 在有网、与 L20 ABI 匹配的 Linux 环境中为一个固定 vLLM 版本下载完整 wheelhouse，连同
   间接依赖、pip 安装报告、版本清单和 SHA256 一起打包。不要在 L20 临时解析 PyPI 依赖。
4. 通过网页 VSCode 上传并逐文件校验；任何 wheel 缺失或哈希失败都回到有网机器补齐。
5. 在管理员/用户明确允许后创建独立 vLLM 环境。不得把 vLLM 安装进
   `/opt/miniconda3/envs/pytorch`，也不得让安装器升级/降级该共享环境的 torch、Transformers
   或 CUDA 组件。
6. 先运行离线环境中的 `vllm serve --help`，核对本版本是否支持仓库脚本使用的
   `--served-model-name`、`--max-model-len`、`--max-num-seqs` 和
   `--gpu-memory-utilization`。
7. 使用本地绝对模型目录、BF16、4096 上下文、序列 1、只绑定 `127.0.0.1` 启动；记录精确
   命令、PID、环境清单、模型 manifest 和服务日志。不要使用 Hugging Face 仓库 ID。
8. 依次验证 `/v1/models`、1 个 JSON 请求、单 Doctor 虚构样本，再运行双 Doctor、Meta、
   一轮讨论和简化 RAG；所有 Agent 保持顺序调用。
9. 与直接 Transformers 的结构化成功率、峰值、首 token 延迟和总耗时比较。只有收益明确且
   稳定，才把 vLLM 作为正式运行入口。

## 主要风险与控制

| 风险 | 影响 | 控制方式 |
|---|---|---|
| vLLM wheel 锁定不同 torch/CUDA | 可能破坏共享环境或导入失败 | 独立环境、离线 wheel 清单、先审依赖再安装 |
| 无外网导致间接 wheel 缺失 | 安装中途失败 | 有网 Linux 机器完整解析并 SHA256；L20 使用 `--no-index` |
| Triton/编译扩展 ABI 不兼容 | 启动时报 symbol/compile error | 选择官方预编译组合；保留完整 traceback，不现场联网编译 |
| 4096 之外的 KV Cache 或并发增长 | OOM、延迟不稳定 | 初始 `max-model-len=4096`、`max-num-seqs=1`，一次只改一个参数 |
| served model name 不一致 | Agent 请求返回 model not found | 服务端和 `LLM_MODEL_NAME=qwen3-4b-local` 使用同一固定名称 |
| JSON Schema 支持随 vLLM 版本变化 | 严格 `response_format` 可能被拒绝 | 先测 API 合同；保留提示词约束、程序校验和一次重试 |
| 浏览器/终端断开 | 服务或实验无法审计 | 使用平台允许的 tmux/作业机制，记录 PID 与日志，不依赖标签页 |
| 残留模型进程 | 占用 L20 显存 | 只终止自己记录的 PID；运行前后执行 `nvidia-smi` |
| 模型或日志进入 Git | 权重/隐私泄漏 | 模型和 artifacts 留在仓库外/ignored；提交前做大文件与路径扫描 |

## 升级验收门

在以下条件全部具备前不开始 vLLM：直接 Transformers 真机成功；模型 SHA256 全部通过；
峰值显存有余量；固定 vLLM/torch/CUDA 兼容矩阵；完整离线 wheelhouse；独立环境路径获准；
回滚方式明确。任何一项缺失都继续保留直接 Transformers 路径，不把 vLLM 当成本阶段阻塞。


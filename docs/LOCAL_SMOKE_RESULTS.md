# 本地冒烟测试实测记录

测试日期：2026-07-15（Asia/Shanghai）。本记录是软件链路验收，不是论文指标复现，也不是临床验证。

## 环境与版本

- 稳定分支：`local-small-llm`
- 上游基线：`854878741b2a620fb7d0c643983a0f802b2bf3be`
- 本地链路稳定 commit：`5374852ed2a082a37ada3102c6b91d0c94eda232`
- Windows：Windows 11 build 26200
- Windows Python：3.13.13（Anaconda）；未安装 PyTorch
- WSL2：Ubuntu 24.04.3、Python 3.12.3；未安装 PyTorch/nvcc
- GPU：NVIDIA GeForce RTX 4060 Laptop GPU，8188 MiB，驱动 581.83
- `nvidia-smi` 报告的 driver CUDA capability：13.0；这不代表本机安装了 CUDA Toolkit
- Ollama：0.32.0
- 唯一下载模型：`qwen3:4b`，Ollama manifest 大小 2.5 GB

`origin` 为 `https://github.com/lp-0105/ColaCare.git`，`upstream` 为 `https://github.com/PKU-AICare/ColaCare.git`。

## 实际运行与结果

统一运行参数为 context 4096、max output 256、temperature 0、batch/并发序列 1、`reasoning_effort=none`。主要命令：

```powershell
python scripts/check_environment.py
python scripts/test_llm_api.py
python scripts/run_synthetic_smoke.py --stage single
python scripts/run_synthetic_smoke.py --stage two-doctors
python scripts/run_synthetic_smoke.py --stage meta
python scripts/run_synthetic_smoke.py --stage discussion
python scripts/run_synthetic_smoke.py --stage rag
python -m unittest discover -s tests -v
```

真实模型调用摘要：

| 阶段 | Doctor | Meta | 讨论轮 | RAG | 最终概率/预测 | 退出码 | 实测墙钟时间 |
|---|---:|---:|---:|---:|---|---:|---:|
| 独立 API | 0 | 0 | 0 | 否 | 0.35 / 0 | 0 | 包含在后续检查中 |
| single | 1 | 0 | 0 | 否 | 0.42 / 0 | 0 | 约 5.1 s（模型已加载） |
| two-doctors | 2 | 0 | 0 | 否 | 0.42 / 0 | 0 | 7.8 s |
| meta | 2 | 1 | 0 | 否 | 0.42 / 0 | 0 | 10.2 s |
| discussion | 2 | 1 | 1 | 否 | 0.42 / 0 | 0 | 16.9 s |
| rag | 2 | 1 | 1 | 是 | 0.42 / 0 | 0 | 20.2 s |

所有结果均为合法、可再次解析的 JSON；单 Agent、双 Doctor、Meta、讨论与 RAG 产物开关组合已由独立验证脚本复查。每个真实请求均顺序执行，观察到的这些运行均一次成功。RAG 只使用 3 条明确标记为虚构、非临床指南的本地测试文档。

最终测试套件：45 个测试通过；`python -m compileall -q utils scripts tests` 通过；Git Bash 与 WSL2 中两个 L20 shell 脚本的 `bash -n` 均返回 0。`scripts/run_l20_smoke_test.sh` 也通过本机 OpenAI 兼容 Ollama 端点真实执行了环境检查、API 检查和 single 链路。

## GPU 峰值显存

测量方法：先执行 `ollama stop qwen3:4b`，等待 2 秒后取得基线；隐藏启动一次 `single`，以 200 ms 周期轮询 `nvidia-smi --query-gpu=memory.used`。该数值是整张 GPU 的总占用，包含桌面和其他基线进程。

- 整卡基线：1276 MiB
- 整卡峰值：4418 MiB
- 相对基线增量：3142 MiB
- 任务后驻留：约 4416–4418 MiB
- 物理显存：8188 MiB
- `ollama ps`：模型运行大小 3.2 GB、100% GPU、context 4096

因此本次 1 条模拟样本的冷启动单 Agent 实测未超过 8 GB。该结果不能外推为 8192 上下文、并发请求或其他模型的峰值。

## 明确未完成项

- 没有真实 EHR 专家模型的已保存输出或检查点，因此未声称完成真实专家输出接入；当前 fixture 中的 0.42 是明确标注的模拟专家概率。
- 没有下载或运行 MIMIC、MSD、MedCPT、GatorTron，也没有调用 DeepSeek API。
- 没有在 L20 真机启动 vLLM；本地只验证了离线资产、shell 语法、启动前置保护和同一 OpenAI 兼容调用链。L20 的显存、吞吐与 wheel/CUDA 兼容性仍需目标服务器实测。
- 尚未在 L20 上上传或加载 `Qwen/Qwen3-4B-Instruct-2507`，因此也没有 L20 Transformers 实测耗时和峰值显存。本分支只准备可审计的离线下载、传输、校验和首条推理脚本。
- 官方完整 `collaboration_pipeline.py` 仍依赖缺失的正式数据、检索资产、专家输出和额外 Python 包；本次采用最小侵入适配层与独立模拟冒烟入口，没有把它包装成完整论文复现。

运行日志与患者级/样本级结果只保存在被 `.gitignore` 排除的 `artifacts/`，未进入 Git 历史。

## 本地模型与 L20 模型的关系

本地 `ollama qwen3:4b` 是为 RTX 4060 链路开发使用的 Ollama 量化缓存；L20 计划使用
Hugging Face 标准 Transformers/Safetensors 格式的 `Qwen/Qwen3-4B-Instruct-2507`
BF16 快照。两者不是同一组可互换文件，也不应把 Ollama blob 复制成 L20 模型目录。
前者证明 Agent 编排和 JSON 合同，后者才是后续 L20 正式实验应固定的模型资产。

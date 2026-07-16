# ColaCare 复现状态

更新日期：2026-07-16。本表把“真实运行通过”和“只有代码/脚本”严格分开；所有虚构样本结果
仅用于软件验收，不具有临床意义，也不是论文指标。

| 状态 | 项目 | 证据或限制 |
|---|---|---|
| 已实际完成 | RTX 4060 上的 Ollama `qwen3:4b` OpenAI 兼容 API | API 连通、非空响应、JSON 解析与 Schema 校验通过 |
| 已实际完成 | 1 DoctorAgent 虚构样本 | 顺序执行，结果合法 JSON |
| 已实际完成 | 2 DoctorAgent 顺序调用 | 共用同一模型，无并发 |
| 已实际完成 | MetaAgent 汇总 | 2 Doctor 输出进入 Meta，结果合法 JSON |
| 已实际完成 | 一轮讨论 | 精确一轮，顺序请求 |
| 已实际完成 | 简化本地 RAG | 仅 3 条明确标记为虚构的测试资料 |
| 已实际完成 | 本地自动化验证 | 45/45 单元测试、compileall、既有 shell 语法检查通过 |
| 已实际完成 | RTX 4060 峰值采样 | 整卡 4418 MiB；基线 1276 MiB；增量 3142 MiB |
| 已实际完成 | L20 只读环境审计 | L20、BF16、PyTorch CUDA 与 Transformers 导入均实测通过 |
| 已实际完成 | 固定 HF snapshot 下载、清单与 SHA256 | `Qwen/Qwen3-4B-Instruct-2507` 固定 revision 完整快照及内部 SHA256 已校验 |
| 已实际完成 | 模型/代码打包、2GB 分片与 L20 校验 | 完整模型归档、分片和代码包均已在 L20 校验；完整归档优先解压成功 |
| 已实际完成 | L20 直接 Transformers BF16 首次冒烟 | 完全离线、`cuda:0`、BF16、合法 JSON；模型从本地 Safetensors 目录加载 |
| 已实际完成 | L20 direct Transformers 五阶段链路 | `single`、`two-doctors`、`meta`、`discussion`、`rag` 全部通过；18 次调用只加载一次模型，最高进程显存约 8592 MiB |
| 只有脚本，未在真机验证 | L20 vLLM/OpenAI 兼容服务 | 仅保留配置接口；vLLM 未安装、未启动 |
| 尚未完成 | 真实 EHR 专家模型已保存输出接入 | 当前 0.42 是 fixture 中的模拟概率 |
| 尚未完成 | MSD、MedCPT、GatorTron 正式 RAG | 未下载、未构建索引、未运行 |
| 尚未完成 | MIMIC 正式数据流程 | 未下载；需授权和仓库外受控存储 |
| 尚未完成 | 官方完整 collaboration pipeline | 仍缺正式数据、专家输出、检索资产和依赖 |
| 尚未完成 | 融合网络正式训练与评估 | 未接入真实报告、checkpoint 和数据 split |
| 不能声称完成 | 论文指标、基线、消融与统计显著性复现 | 尚无固定正式数据/模型/提示词全套实验 |
| 不能声称完成 | 临床有效性或泛化能力 | 虚构样本只检验软件链路 |
| 不能声称完成 | Ollama 与 HF 模型数值等价 | 文件格式、量化方式和具体模型 revision 不同 |

## L20 direct Transformers 真机验收摘要

- 模型：`Qwen/Qwen3-4B-Instruct-2507`，固定 revision `cdbee75f17c01a7cc42f958dc650907174af0554`；
- 运行方式：单进程 direct Hugging Face Transformers，BF16、`cuda:0`、完全离线；
- 阶段：`single`、`two-doctors`、`meta`、`discussion`、`rag` 均输出通过 Schema 校验的 JSON；
- 总调用数：18；模型加载次数：1；五阶段复用同一模型实例；
- 最高 `nvidia-smi` 进程显存：约 8592 MiB；
- 当前 EHR 输入仍是明确标记为虚构的 fixture，RAG 仍是 synthetic lexical-overlap 测试资料；
- 尚未完成：MIMIC、真实 EHR 专家输出、MSD、MedCPT、GatorTron、融合网络正式训练、论文指标、基线与消融实验。

完整的逐阶段 token、耗时和显存证据见 `docs/L20_DIRECT_PIPELINE_RESULTS.md`。

## 正式实验的冻结要求

L20 后续所有基线、消融和对比实验必须共同固定：

- `MODEL_ID=Qwen/Qwen3-4B-Instruct-2507`；
- revision `cdbee75f17c01a7cc42f958dc650907174af0554`（若升级，必须形成新实验批次）；
- 同一 chat template、系统/角色提示词、JSON Schema 与 prompt 版本 hash；
- context 4096、max new tokens 256、确定性生成参数和并发序列 1，除非被测变量就是其中之一；
- ColaCare commit、数据快照/预处理 hash、EHR 专家输出 hash、随机种子和环境清单。

本地 Ollama `qwen3:4b` 只用于链路开发；L20 的标准 Safetensors BF16 快照用于正式实验。
两者结果不得混入同一实验表格。

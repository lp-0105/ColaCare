# L20 环境只读审计

审计日期：2026-07-16（Asia/Shanghai）。审计通过学校平台已登录的网页 VSCode 终端完成；
没有安装软件、下载联网资源、上传文件、启动/关闭实例或修改 GPU 资源。服务器原始输出位于
`~/colacare_audit/environment_audit.txt`，共 447 行、19,508 字节，SHA256：
`6bbf5f54e01bee68cbd8b0444af91417a61d9fd102cf51c53ec003c669aed157`。
原始文件留在服务器，不提交到仓库。

## 系统、CPU、内存和磁盘

| 项目 | 只读实测结果 |
|---|---|
| 操作系统 | Ubuntu 22.04.3 LTS |
| 内核 | `5.14` 系列（平台 EL9 宿主内核） |
| CPU | 2 × Intel Xeon Platinum 8558P，终端可见 96 个逻辑 CPU |
| 内存 | 503 GiB 总计，约 475 GiB 可用；未配置 swap |
| 根文件系统 | 890 GiB，总计约 697 GiB 可用 |
| 共享 GPFS | 约 782 TiB，总计约 771 TiB 可用 |
| 用户配额 | `quota` 命令未安装，个人实际配额未知；上传前必须在平台界面或管理员处确认 |

## GPU 与 CUDA

| 项目 | 只读实测结果 |
|---|---|
| GPU | 1 × NVIDIA L20 |
| 显存 | 46,068 MiB 总计；审计时 100 MiB 使用、45,360 MiB 空闲 |
| Compute Capability | 8.9 |
| 驱动 | 590.48.01 |
| `nvidia-smi` CUDA compatibility | 13.1（驱动支持上限，不等于环境内 CUDA Toolkit） |
| 系统 `nvcc` | CUDA 11.8 |
| PyTorch wheel CUDA | CUDA 12.6，见 `torch 2.7.1+cu126` |
| BF16 | `torch.cuda.is_bf16_supported() == True` |
| 审计时负载 | GPU 利用率 0%，未见计算进程 |

系统 `nvcc`、驱动兼容上限和 PyTorch 自带 CUDA runtime 是三个不同版本。第一阶段只运行
已有 PyTorch wheel，不编译 CUDA 扩展，因此三者不必完全相同。

## Conda、Python 与包

Conda 4.14.0 可见 `base`、`pytorch` 和 `tensorflow` 环境。首个 Transformers 冒烟应只复用
`/opt/miniconda3/envs/pytorch`，不修改共享环境。

| 组件 | `pytorch` 环境实测 | 能否直接使用 |
|---|---|---|
| Python | 3.10.13 | 是 |
| PyTorch | 2.7.1+cu126 | 是；CUDA 可用、设备数 1 |
| torchvision / torchaudio | 0.22.1 / 2.7.1 | 本链路不需要 |
| Transformers | 4.57.6 | 是；高于 Qwen3 所需 4.51.0 |
| Accelerate | 1.13.0 | 是 |
| Safetensors | 0.4.5 | 是 |
| Tokenizers | 0.22.0 | 是 |
| filelock | 3.18.x | 是 |
| vLLM | 未安装 | 否；本阶段明确不需要 |
| OpenAI Python 客户端 | 未安装 | 否；直接 Transformers 不需要，现有 Agent 适配层也使用标准库 HTTP |

`base` 环境不适合本任务：其中没有 PyTorch，Transformers 元数据虽存在，但导入因缺少
`filelock` 失败。所有命令必须确认激活的是 `pytorch` 环境。

## 无外网与平台转发结论

- GitHub、Hugging Face 和 PyPI 均不能由计算环境直接访问，模型、代码包和任何备用 wheels
  必须在有网机器准备并通过平台允许的网页 VSCode 上传。
- 对平台提供的 `PROXY_URL` 进行普通 HTTPS 代理测试时，代理 `CONNECT` 请求返回 404。
  这说明该 URL 是平台应用/端口转发入口，而不是实现通用 HTTP `CONNECT` 的正向代理；
  不能把它设置为 `HTTPS_PROXY` 来下载 GitHub、Hugging Face 或 PyPI 资源。
- 当前 SSH 问题位于平台对外端口转发层，不是 GPU 容器、PyTorch 或模型本身的问题；本阶段
  通过已登录网页 VSCode 上传和终端执行，因此不把 SSH 修复作为阻塞。

## Qwen3-4B-Instruct-2507 BF16 适配判断

固定快照包含约 8.045 GB 权重（十进制，约 7.49 GiB）。按官方配置的 36 层、8 个 KV 头、
head dim 128 估算，batch 1 的 BF16 KV Cache 在 4096 token 时约 576 MiB，8192 token 时约
1.125 GiB。加上 CUDA 上下文、临时张量和 Transformers 开销，4096 上下文的进程总显存
保守预计约 10–14 GiB，远低于 L20 的约 45 GiB 可用显存，因此第一条 BF16 推理适合该 GPU。
这只是容量估算；真实加载后、推理峰值和耗时必须由 `test_l20_transformers.py` 在 L20 记录，
在真机结果产生前不能声称已验证性能。

## 最小离线部署清单

1. 模型：固定 revision 的全部 13 个仓库文件，包括 3 个 safetensors 权重分片、权重索引、
   `config.json`、`generation_config.json`、tokenizer 配置和词表文件、模型卡与许可证；同时上传
   本项目生成的 `MODEL_MANIFEST.json` 和 `SHA256SUMS`。
2. Python：基于本次审计，第一条推理无需安装额外 wheel；已有 `pytorch` 环境满足 torch、
   Transformers、Accelerate、Safetensors、Tokenizers 和 filelock。若平台环境后续变化，再从
   有网 Linux 机器准备与 Python 3.10/CUDA 兼容的完整备用 wheelhouse，不能现场联网安装。
3. 代码：由 `package_l20_assets.sh` 从指定 Git commit 生成的 ColaCare 代码归档，不包含
   `.git`、模型、数据、日志、缓存、artifacts、`.env` 或隐私内容。


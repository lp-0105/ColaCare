# Local Environment Snapshot

Recorded on 2026-07-15 (Asia/Shanghai) before local reproduction changes.

## Repository

- Fork: `https://github.com/lp-0105/ColaCare.git`
- Official upstream: `https://github.com/PKU-AICare/ColaCare.git`
- Branch: `local-small-llm`
- Baseline commit: `854878741b2a620fb7d0c643983a0f802b2bf3be`
- `origin/main` and `upstream/main` resolved to the same baseline commit when fetched.

## Windows host

- OS: Windows 11 (`Windows-11-10.0.26200-SP0`)
- Python: 3.13.13, Anaconda build
- PyTorch: not installed in the active Windows Python environment
- CUDA toolkit (`nvcc`): not installed or not on `PATH`
- NVIDIA driver: 581.83
- Driver-advertised CUDA compatibility: 13.0
- GPU: NVIDIA GeForce RTX 4060 Laptop GPU
- VRAM: 8188 MiB total
- Ollama: 0.32.0

## WSL2

- Distribution: `UbuntuBackup`, WSL version 2
- OS: Ubuntu 24.04.3 LTS
- Python: 3.12.3
- PyTorch: not installed in the active WSL Python environment
- CUDA toolkit (`nvcc`): not installed or not on `PATH`
- GPU passthrough: RTX 4060 Laptop GPU visible, 8188 MiB total

## Interpretation

`nvidia-smi`'s CUDA value is the maximum CUDA compatibility exposed by the driver, not an installed CUDA toolkit version. The minimal local smoke path therefore uses the installed Windows Ollama runtime and Python standard-library HTTP calls. The L20 path remains Linux/vLLM oriented.

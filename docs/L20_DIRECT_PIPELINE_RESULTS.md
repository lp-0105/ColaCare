# L20 Direct Transformers Pipeline Results

## Scope and code identity

- Baseline ColaCare commit: c81e5b1f99ec3abe87b54a716893403589759a20
- Model: Qwen/Qwen3-4B-Instruct-2507
- Fixed model revision: cdbee75f17c01a7cc42f958dc650907174af0554
- Model directory: /share/home/2023300013/colacare_offline/models/Qwen3-4B-Instruct-2507
- Code directory: /share/home/2023300013/colacare_offline/work/ColaCare
- Backend: direct Hugging Face Transformers, local_files_only=True, BF16, cuda:0
- Context limit: 4096 tokens
- Maximum new tokens per call: 256
- Generation: do_sample=False
- Test data: one repository fixture explicitly marked fictional
- RAG data: repository synthetic guidelines only

This is a synthetic end-to-end engineering smoke test. It is not a reproduction of the paper metrics.

## Server environment

- OS: Ubuntu 22.04.3 LTS, x86_64
- CPU: Intel Xeon Platinum 8558P, 96 logical CPUs
- Memory: approximately 503 GiB
- GPU: one NVIDIA L20
- GPU memory reported by nvidia-smi: 46068 MiB
- Compute capability: 8.9
- NVIDIA driver: 590.48.01
- Host CUDA version reported by nvidia-smi: 13.1
- Conda environment: /opt/miniconda3/envs/pytorch
- Python: 3.10.13
- PyTorch: 2.7.1+cu126
- Transformers: 4.57.6
- PyTorch CUDA available: yes
- BF16 supported: yes

The existing shared Conda environment was not modified.

## SoundFile vendor repair

The first Transformers smoke test failed while importing the audio dependency path with:

    OSError: cannot load library 'libsndfile.so'

The repair used SoundFile 0.13.1 from the Linux x86_64 manylinux_2_28 wheel. Its uploaded wheel SHA256 was:

    03267c4e493315294834a0870f31dbb3b28a95561b80b134f0bd3cf2d5f0e618

It was installed without dependencies into:

    /share/home/2023300013/colacare_offline/vendor/soundfile-0.13.1

That directory was prepended to PYTHONPATH. The shared Conda environment and existing ~/.local packages were not changed. The vendor SoundFile and embedded libsndfile import check passed before inference was retried.

## Direct pipeline implementation

The L20 code tree was compared with the original c81e5b1f archive while excluding artifacts, logs, caches, models, data, vendor files, environment files, and generated outputs. The comparison found:

- Modified baseline files: none
- Removed baseline files: none
- Added runtime script: scripts/run_l20_pipeline_direct.py

The direct adapter presents the same chat interface used by the synthetic DoctorAgent and MetaAgent helpers. It loads one local Transformers model once and reuses that instance for every requested stage in the same process. Stages execute strictly in this order:

1. single
2. two-doctors
3. meta
4. discussion
5. rag

A stage failure stops the process before the next stage. Each prediction is parsed as JSON and checked with the repository prediction validator. Full model reasoning is not requested or persisted.

## Measured results

The shared model load took 3.8589 seconds. GPU memory was 103 MiB before load and 8120 MiB after load.

| Stage | Status | LLM calls | Input tokens | Output tokens | Inference seconds | Stage wall seconds | GPU before stage MiB | GPU after inference MiB | Peak allocated MiB | Peak reserved MiB |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| single | success | 1 | 509 | 150 | 3.3753 | 3.4704 | 8120 | 8384 | 7844.64 | 7930.00 |
| two-doctors | success | 2 | 1018 | 300 | 6.0590 | 6.1063 | 8384 | 8384 | 7844.64 | 7930.00 |
| meta | success | 3 | 1584 | 445 | 8.8077 | 8.8555 | 8384 | 8444 | 7867.83 | 7990.00 |
| discussion | success | 6 | 3990 | 716 | 14.1750 | 14.2365 | 8444 | 8592 | 7963.22 | 8138.00 |
| rag | success | 6 | 4083 | 577 | 11.5707 | 11.6242 | 8592 | 8592 | 7947.39 | 8138.00 |

Totals:

- LLM calls: 18
- Model loads: 1
- Input tokens: 11184
- Output tokens: 2188
- Summed inference time: 43.9877 seconds
- Summed stage wall time: 44.2929 seconds
- Maximum nvidia-smi process memory observed in the stage metrics: 8592 MiB
- Maximum PyTorch allocated memory: 7963.22 MiB
- Maximum PyTorch reserved memory: 8138.00 MiB
- Final output schema validation: passed for all five stages
- RAG backend: synthetic-lexical-overlap

## Offline status

The run used HF_HUB_OFFLINE=1, TRANSFORMERS_OFFLINE=1, and HF_DATASETS_OFFLINE=1. The model and tokenizer were opened only from MODEL_PATH with local_files_only=True. The run logs contain no offline-resource-missing error or network-download attempt. No model, dependency, or dataset was downloaded during the staged run.

## Work not completed

The following must not be claimed as reproduced:

- Paper metrics, statistical evaluation, or full experimental scale
- MIMIC or any other real patient dataset
- Training or validation of real EHR expert models
- Integration of real EHR expert checkpoints or patient-level saved outputs
- The paper MSD corpus and full MedCPT or GatorTron retrieval stack
- Paper-scale DoctorAgent counts, longer discussions, and production RAG
- Fusion network training and evaluation
- Full baselines and ablations with fixed prompts and generation parameters
- Throughput, concurrency, and vLLM deployment validation

## Key server paths

Environment and validation logs:

- /share/home/2023300013/colacare_offline/logs/l20_pipeline/environment_and_code_identity.log
- /share/home/2023300013/colacare_offline/logs/l20_pipeline/direct_backend_validation.log
- /share/home/2023300013/colacare_offline/logs/l20_pipeline/all_stages.log
- /share/home/2023300013/colacare_offline/logs/l20_pipeline/final_acceptance_check.log
- /share/home/2023300013/colacare_offline/logs/l20_pipeline/code_change_audit.log

Stage results and per-stage logs:

- /share/home/2023300013/colacare_offline/work/ColaCare/artifacts/l20_pipeline/single/result.json
- /share/home/2023300013/colacare_offline/work/ColaCare/artifacts/l20_pipeline/single/run.log
- /share/home/2023300013/colacare_offline/work/ColaCare/artifacts/l20_pipeline/two-doctors/result.json
- /share/home/2023300013/colacare_offline/work/ColaCare/artifacts/l20_pipeline/two-doctors/run.log
- /share/home/2023300013/colacare_offline/work/ColaCare/artifacts/l20_pipeline/meta/result.json
- /share/home/2023300013/colacare_offline/work/ColaCare/artifacts/l20_pipeline/meta/run.log
- /share/home/2023300013/colacare_offline/work/ColaCare/artifacts/l20_pipeline/discussion/result.json
- /share/home/2023300013/colacare_offline/work/ColaCare/artifacts/l20_pipeline/discussion/run.log
- /share/home/2023300013/colacare_offline/work/ColaCare/artifacts/l20_pipeline/rag/result.json
- /share/home/2023300013/colacare_offline/work/ColaCare/artifacts/l20_pipeline/rag/run.log

These logs and full result JSON files stay on the server and are not included in the code export.

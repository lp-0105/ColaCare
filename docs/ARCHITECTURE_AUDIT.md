# ColaCare Architecture Audit

## Audit baseline

- Paper: `2410.02551v2`, WWW 2025, reviewed from the local 12-page PDF.
- Fork and upstream baseline: `854878741b2a620fb7d0c643983a0f802b2bf3be`.
- `origin/main` and `upstream/main` resolved to the same commit on 2026-07-15.
- Scope: README, `collaboration_pipeline.py`, `train_fusion.py`, `utils/`, `pyehr/`, `ehr_datasets/`, and `requirements.txt`.

This is a code and asset audit, not a claim that the paper's metrics have been reproduced.

## Paper data flow

1. A domain EHR model encodes a patient's structured longitudinal record into an EHR embedding.
2. An MLP produces a task probability. SHAP or model importance weights identify influential features.
3. Each DoctorAgent is paired with one EHR expert model. Its probability, feature importance, demographics, and time-series values form an initial patient record.
4. MedCPT retrieves top-3 passages from the MSD guideline. The patient record and passages are sent to each DoctorAgent.
5. A MetaAgent combines initial reviews into a preliminary mortality/readmission report.
6. DoctorAgents critique the report, retrieve more evidence, and the MetaAgent stops or revises. The paper caps this at three rounds.
7. GatorTron encodes the final report. The report embedding is concatenated with N EHR expert embeddings.
8. An MLP fusion network produces the final probability and is trained with binary cross entropy.

## Public-code data flow

### EHR expert outputs

`pyehr/train_test.py` trains one model configuration at a time. `pyehr/importance.py` derives top features from saved outputs. The collaboration path does not run them automatically; it expects processed pickles and files like:

```text
ehr_datasets/<dataset>/processed/fold_1/
  <mode>_x.pkl
  <mode>_raw_x.pkl
  <mode>_pid.pkl
  <mode>_y.pkl
  basic.pkl
  survival.pkl
  dead.pkl
  dl_data/<Model>_<task>_<mode>_output.pkl
  dl_data/<Model>_<task>_<mode>_features.pkl
```

`ContextBuilder` turns these files into prose containing demographics, longitudinal values, an expert probability, and feature importance values.

### Retrieval and agents

`collaboration_pipeline.py` constructs `RetrievalSystem` unconditionally. `utils/retrieve_utils.py` loads or builds a FAISS inner-product index using the MedCPT Article Encoder and embeds queries with the MedCPT Query Encoder.

Each `DoctorAgent.analysis()` obtains basic information, a shorter retrieval query, and a full healthcare context. It retrieves top-3 documents and sends the documents plus full context to the LLM. Initial DoctorAgents are submitted through `ThreadPoolExecutor`.

`LeaderAgent.summary(..., is_initial=True)` creates the preliminary report. `Collaboration.collaborate()` performs up to `max_round` concurrent critiques and MetaAgent revisions. `LeaderAgent.revise_logits()` asks the LLM to revise every expert probability and emit a final probability. Per-patient prompts, responses, and summaries are written beneath `response/`.

### Report encoding and fusion

`utils/process_output.py` is intended to encode collaboration text with local `lm/GatorTron`, combine it with EHR embeddings, and write fusion pickles. `train_fusion.py` loads those pickles, concatenates three EHR embeddings with a 1024-dimensional text embedding in `utils/fusion.py`, and trains a sigmoid predictor with BCE loss.

## Main modules

| Path | Responsibility | Required external state |
| --- | --- | --- |
| `collaboration_pipeline.py` | Full agent orchestration | EHR pickles, MedCPT, corpus/index, LLM API |
| `utils/framework.py` | DoctorAgent, LeaderAgent, discussion, JSON parsing | LLM client configuration |
| `utils/healthcare_context_utils.py` | Convert EHR files/model outputs into prompts | Dataset-specific pickles |
| `utils/retrieve_utils.py` | MedCPT embedding, FAISS indexing/retrieval | MedCPT checkpoints and corpus chunks |
| `pyehr/train_test.py` | Train/test EHR experts | Missing dataset loader/data/configuration |
| `pyehr/importance.py` | Extract important features | Saved pyehr outputs and feature names |
| `utils/process_output.py` | Encode final reports and assemble fusion data | Reports, GatorTron, EHR embeddings |
| `train_fusion.py` | Train/evaluate fusion network | Fusion pickles and CUDA |

## Paper/code mismatches and risks

| Finding | Evidence and impact |
| --- | --- |
| LLM identity is underspecified | The paper names DeepSeek-V2.5; code defaults to mutable API alias `deepseek-chat`. |
| Client configuration is incomplete | `utils/config.py` defines only `deepseek_config`; `utils/framework.py` imports undefined `v1_config`, `v2_config`, and `default_config`. |
| No environment-only adapter | API selection uses model-name substrings and Python config, preventing a clean Ollama-to-vLLM switch. |
| RAG cannot be disabled | The main pipeline always constructs retrieval, and DoctorAgent assumes it exists. |
| Official local behavior is concurrent | Initial agents and discussion rounds use thread pools, unsafe for the requested 8 GB single-sequence setting. |
| Retrieval dependencies are incomplete | `sentence_transformers` and `tenacity` are imported but absent from requirements; CPU and GPU FAISS are pinned together. |
| Data preparation is incomplete | No processed datasets, pyehr dataset loader, checkpoints, EHR probabilities, features, or embeddings are checked in. |
| MSD acquisition is unscripted | Code expects `corpus/msd/chunk`, but no acquisition, terms, chunking, or checksum workflow exists. |
| GatorTron is ambiguous | `process_output.py` loads `lm/GatorTron`; exact checkpoint and revision are unspecified. |
| Output path/schema mismatch | Collaboration writes `response/.../result.json`; processing reads `output/.../result.json` and treats dictionary values as tokenizer strings. |
| Fusion forward appears wrong | `Pipeline.forward()` calls `self.model(ehr, rag)` and `.to(rag.device)`, while the paper and `Fusion.forward()` require the report text embedding. `rag` may be `None`. |
| Single-GPU selection is wrong | Lightning uses `devices=[1]`; a one-GPU host normally exposes GPU 0 only. |
| Metric formatting changes the mean | Bootstrap display uses `mean * 100 + 1`, adding one percentage point. |
| Import executes the full pipeline | `collaboration_pipeline.py` lacks a main guard, complicating tests and reuse. |
| Loaded predictions are unused | `preds = load_preds(config)` is evaluated but never referenced afterward. |
| Patient artifact privacy is unmanaged | Raw prompts, full messages, IDs, and reports are saved per patient; ignore rules are essential but not enforced in code. |

## Minimal-change boundary

The local smoke adds an environment-driven OpenAI-compatible adapter and a separate synthetic-patient runner. It does not claim equivalence to EHR training, MedCPT/MSD retrieval, GatorTron encoding, or fusion training. Formal-data fixes stay isolated until required assets are available.

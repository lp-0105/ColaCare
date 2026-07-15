# Local Small-LLM Reproduction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run a validated fictional-patient ColaCare chain against one local 4B OpenAI-compatible model, then prepare an offline L20 path.

**Architecture:** A standard-library HTTP adapter isolates provider configuration from agent logic. A staged synthetic runner exercises DoctorAgent, MetaAgent, one discussion round, and lexical RAG sequentially. The formal-data path changes only at its LLM boundary.

**Tech Stack:** Python 3.9+ standard library, Ollama OpenAI API, Qwen3-4B Q4_K_M, unittest, Bash.

---

## File map

- `utils/llm_client.py`: environment-driven transport and JSON contract.
- `utils/framework.py`, `utils/llm.py`, `baselines/llm.py`: route official calls through the adapter.
- `utils/smoke_agents.py`: dependency-light staged agent roles.
- `utils/local_rag.py`: deterministic local retrieval.
- `scripts/check_environment.py`, `scripts/test_llm_api.py`, `scripts/run_synthetic_smoke.py`: acceptance entry points.
- `tests/` and `tests/fixtures/`: offline regression tests and fictional inputs.
- `.env.example`, `configs/l20.env.example`, L20 shell scripts and deployment documentation.

### Task 1: Settings and response contract

**Files:** create `tests/test_llm_client.py`, `utils/llm_client.py`.

- [ ] Write tests for defaults, invalid numeric values, `/v1` URL normalization, empty content, malformed envelopes, and plain/fenced JSON.
- [ ] Run `python -m unittest tests.test_llm_client -v`; expect missing-module failure.
- [ ] Implement immutable settings, typed errors, `urllib.request` POST, usage normalization, `response_format`, and strict JSON-object extraction.
- [ ] Re-run the test; expect all pass, then run `python -m compileall -q utils tests`.

### Task 2: Official-agent adapter routing

**Files:** modify `utils/framework.py`, `utils/llm.py`, `baselines/llm.py`; create `.env.example`.

- [ ] Add a failing injection/import test proving `Agent` no longer constructs the OpenAI SDK.
- [ ] Run it and confirm failure on the current `openai` import/config path.
- [ ] Replace provider-name branching with `LLMClient.from_env()` and preserve existing result/usage shapes.
- [ ] Run unit tests and `python -c "import utils.framework"`.
- [ ] Search tracked changes for secret-like values and machine-specific paths.

### Task 3: Environment and live API checks

**Files:** create `scripts/check_environment.py`, `scripts/test_llm_api.py`.

- [ ] Add failing tests for redaction, unreachable service, timeout mapping, empty content, invalid JSON, and range violations.
- [ ] Implement checks using schema `{risk_probability, prediction, reasoning_summary}`.
- [ ] Run unit tests, then `python scripts/check_environment.py` and `python scripts/test_llm_api.py`.
- [ ] Record summaries only; never request or save full hidden reasoning.

### Task 4: Single fictional DoctorAgent

**Files:** create fixture, `tests/test_smoke_agents.py`, `utils/smoke_agents.py`, `scripts/run_synthetic_smoke.py`.

- [ ] Commit a fictional fixture containing demographics, diagnoses, vitals, labs, expert probability, and important features.
- [ ] Write failing tests for validation, no-RAG prompt, bounded retry, schema validation, and atomic ignored output.
- [ ] Implement one sequential DoctorAgent call with `/no_think`, 256 tokens, temperature zero, and at most three attempts.
- [ ] Run unit tests and `python scripts/run_synthetic_smoke.py --stage single`.

### Task 5: Staged extensions

**Files:** modify smoke files; create `tests/test_local_rag.py`, synthetic guidelines, `utils/local_rag.py`.

- [ ] Write failing tests proving two doctors run in order, MetaAgent consumes both, discussion runs once, and RAG returns deterministic IDs.
- [ ] Implement `two-doctors`, `meta`, `discussion`, and `rag` without threads/futures.
- [ ] Run every stage independently and finally re-run `single`.

### Task 6: L20 offline package

**Files:** create `docs/L20_OFFLINE_DEPLOYMENT.md`, `configs/l20.env.example`, `scripts/start_vllm_server.sh`, `scripts/run_l20_smoke_test.sh`; modify README.

- [ ] Write scripts with strict shell mode, required-variable checks, local model paths, offline flags, `--max-model-len`, and `--max-num-seqs 1`.
- [ ] Run `bash -n` through WSL on both scripts.
- [ ] Document wheel/model manifests, SHA-256, upload, offline startup, scaling, resume, and Git exclusions.
- [ ] Add exact from-zero commands and limitations to README.

### Task 7: Fresh acceptance and commits

- [ ] Run `python -m unittest discover -s tests -v` and `python -m compileall -q scripts utils tests`.
- [ ] Run live API and all staged smokes at context 4096 and one sequence.
- [ ] Poll `nvidia-smi`, record baseline/peak, and verify `ollama ps` reports GPU execution and context 4096.
- [ ] Run ignore, secret, tracked-size, and `git diff --check` audits.
- [ ] Commit stable nodes clearly; do not push without an explicit handoff decision.

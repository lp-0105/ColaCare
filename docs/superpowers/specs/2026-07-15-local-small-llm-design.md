# Local Small-LLM Reproduction Design

## Status

The user's eight-stage specification is the approved design authority. The user explicitly requested autonomous execution, so no additional design pause is required.

## Architecture

Add a narrow OpenAI-compatible HTTP adapter in `utils/llm_client.py`. It reads service parameters from environment variables, sends deterministic chat-completion requests, requests JSON Schema output, normalizes usage, and exposes clear transport/protocol/schema errors. Existing official agents obtain the adapter from their common base class; provider-specific model-name branching is removed.

A separate synthetic pipeline exercises the same adapter and ColaCare roles without importing datasets, MedCPT, GatorTron, or PyTorch. Its default is one DoctorAgent, no MetaAgent, no discussion, and no RAG. Later modes add one feature at a time while preserving earlier commands. Calls remain sequential.

## Components

- `utils/llm_client.py`: environment settings, HTTP transport, JSON extraction, response contract.
- `utils/smoke_agents.py`: synthetic DoctorAgent and MetaAgent prompts with small schemas and bounded retries.
- `utils/local_rag.py`: deterministic lexical retrieval over fictional guideline snippets; explicitly not MedCPT/MSD.
- `scripts/check_environment.py`: non-secret machine/repository/service inventory.
- `scripts/test_llm_api.py`: independent JSON contract test with actionable errors.
- `scripts/run_synthetic_smoke.py`: selectable staged pipeline and ignored artifact writer.
- `tests/`: standard-library unit tests with no live model dependency.

## Privacy

The patient fixture is explicitly fictional and uses a synthetic identifier. Runtime results/logs go under ignored `artifacts/`. Diagnostics redact keys. Formal datasets, weights, embeddings, reports, and indexes remain ignored.

## Error handling

Transport distinguishes refused connection, timeout, HTTP error, malformed envelope, empty content, invalid JSON, and schema violation. Agent retries are bounded. Writes use a temporary file followed by replacement so interruptions do not create false completion artifacts.

## Testing

Tests precede implementation and cover environment parsing, URL normalization, response envelopes, JSON extraction, validation, bounded retries, sequential ordering, retrieval, and artifact redaction. Live acceptance calls Ollama, validates JSON, runs each stage, and measures GPU memory.

## Exclusions

The local milestone does not repair or validate full pyehr training, restricted-data preprocessing, MedCPT/MSD indexing, GatorTron encoding, or fusion training. Those audit findings remain for the L20/formal-data phase.

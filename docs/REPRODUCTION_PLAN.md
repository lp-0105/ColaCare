# ColaCare Reproduction Plan

## Scope

The local milestone proves transport, configuration, validation, sequential agent orchestration, and artifact hygiene with a fictional patient. It does not reproduce paper metrics or validate clinical predictions.

## RTX 4060 local stages

| Stage | Input | Command | Output | Expected service VRAM | Failure checks |
| --- | --- | --- | --- | ---: | --- |
| Environment | Local checkout/tools | `python scripts/check_environment.py` | JSON summary | 0 | Branch, remotes, Python, Ollama, GPU |
| API contract | LLM environment | `python scripts/test_llm_api.py` | Parsed risk JSON | 3.8-4.8 GiB | URL, model, timeout, schema |
| Single Doctor | Fictional fixture | `python scripts/run_synthetic_smoke.py --stage single` | Ignored result/log | 3.8-4.8 GiB | Retry, `/no_think`, schema |
| Two Doctors | Same fixture | `python scripts/run_synthetic_smoke.py --stage two-doctors` | Two sequential reviews | 3.8-4.8 GiB | No concurrency |
| MetaAgent | Two reviews | `python scripts/run_synthetic_smoke.py --stage meta` | Consensus JSON | 3.8-4.8 GiB | Nested input/final schema |
| One discussion | Meta result | `python scripts/run_synthetic_smoke.py --stage discussion` | Critiques and revised result | 3.8-4.8 GiB | Exactly one round |
| Simplified RAG | Synthetic snippets | `python scripts/run_synthetic_smoke.py --stage rag` | Retrieval IDs and result | 3.8-4.8 GiB | Deterministic local ranking |

All earlier modes remain runnable. Runtime data goes to ignored `artifacts/smoke/`; the committed fixture contains no real patient data.

## Formal-data gates

1. Obtain authorized data outside Git and create a hash/source/preprocessing manifest.
2. Pin/reconstruct the pyehr loader and train one expert on a tiny authorized subset.
3. Export probabilities, feature importance, and embeddings with a documented schema.
4. Feed one saved expert output into one DoctorAgent without RAG and inspect prompt structure.
5. Pin MedCPT Query/Article encoders and an approved MSD snapshot; build FAISS offline.
6. Run one sample through the official route sequentially before broad batches.
7. Pin GatorTron and repair/test report-to-fusion conversion.
8. Repair fusion forward with a failing regression test; start with one batch on GPU 0.
9. Only then scale to three DoctorAgents, three rounds, fixed splits, and repeated seeds.

## L20 ladder

| Step | Samples | Configuration | Resume/output | Memory plan |
| --- | ---: | --- | --- | ---: |
| Service validation | 0 | API only | Health/model list | Measure allocation |
| Synthetic parity | 1 | 1 doctor, 0 rounds, no RAG | Same laptop schema | Small model under 8 GiB target |
| Saved EHR output | 1 | 1 doctor, 0 rounds | Atomic patient JSON | Small model first |
| Collaboration | 10 | 3 sequential, 1 round | Per-patient atomic JSON | Measure chosen model/context |
| RAG | 10 | 3 sequential, 1 round | Retrieval IDs/reports | Separate MedCPT allocation if needed |
| Fusion check | one batch | Fusion only | Tensors/checkpoint | Measure apart from LLM |
| Validation scale | 100 | Candidate config | Sharded manifest | Keep `max-num-seqs=1` first |
| Authorized full split | all | Fixed formal config | Atomic shards/completion manifest | Upgrade model only after headroom test |

## Troubleshooting order

1. Query `GET /v1/models`, then run a minimal non-JSON completion.
2. Ensure `LLM_BASE_URL` ends in `/v1`, the model exists, and API key is non-empty even when ignored locally.
3. Reduce context to 4096, output to 256, concurrency to one, and unload other GPU applications.
4. Inspect raw responses only in ignored local logs; never paste patient content into Git/issues.
5. For schema errors, use temperature zero, `/no_think`, `response_format`, and bounded retries.
6. For OOM, record baseline and peak `nvidia-smi`, then reduce context before changing runtime/quantization.
7. For formal data, validate saved-output schemas without copying unrelated patient records.

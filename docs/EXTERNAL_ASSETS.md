# External Assets Inventory

| Asset | Expected use/path | Access or licensing concern | First smoke? |
| --- | --- | --- | --- |
| `qwen3:4b` Q4_K_M | Ollama model store | One 2.5 GB download; Apache-2.0 model entry | Yes; only model downloaded |
| DeepSeek API | Original `https://api.deepseek.com/v1` | Paid network service and secret key; mutable alias | No |
| MedCPT Query Encoder | `ncbi/MedCPT-Query-Encoder` or local `retriever/` | Download and pin revision/hash | No |
| MedCPT Article Encoder | Derived Article Encoder path | Separate download; required to build index | No |
| MSD guideline corpus | `corpus/msd/chunk/*.json` | No downloader or redistribution grant included; review terms before scraping/copying | No |
| MSD FAISS index | `corpus/msd/index/...` | Large generated embeddings/index; never commit | No |
| GatorTron | `lm/GatorTron` | Exact checkpoint/revision unspecified; may be gated and large | No |
| MIMIC-IV | `ehr_datasets/mimic-iv/processed/fold_1` | PhysioNet credentialing and DUA; patient-level controls | No |
| CDSL | `ehr_datasets/cdsl/processed/fold_1` | No downloader or immutable preprocessing manifest | No |
| ESRD | `ehr_datasets/esrd/processed/fold_1` | Paper data is not present and may not be redistributable | No |
| pyehr loader | `pyehr/ehrdatasets/loader/` | Ignored/missing; may need pinned upstream pyehr source | No |
| EHR checkpoints | pyehr log/checkpoint paths | Train or supply; large and data-derived | No |
| EHR outputs/features | `dl_data/*_{output,features}.pkl` | Missing patient-level derivatives | No |
| EHR embeddings | `*_embeddings*.pkl` | Missing patient-level derivatives used by fusion | No |
| Collaboration reports | `response/` versus `output/` | Patient-level text plus path/schema mismatch | No |
| Fusion artifacts | `fusion*/*.pkl`, Lightning checkpoints | Large, contains labels/IDs | No |
| Python dependencies | requirements plus missing `sentence-transformers`, `tenacity` | Old pins target Python 3.9; install only per phase | No for standard-library smoke |

## Missing baseline files

- No processed or raw EHR dataset.
- No MedCPT checkpoint, MSD chunks, embeddings, metadata, or FAISS index.
- No GatorTron checkpoint.
- No trained EHR expert checkpoint, probability, feature-importance, or embedding pickle.
- No collaboration output compatible with `utils/process_output.py`.
- No fusion-ready train/validation/test pickles.
- No complete checked-in dataset preprocessing pipeline or input manifest.

## First-smoke boundary

The first smoke uses the public baseline source, one fully fictional committed patient fixture, installed Ollama 0.32.0, the single downloaded `qwen3:4b`, and Python's standard library. It does not use or emulate MIMIC, CDSL, ESRD, MedCPT, MSD, GatorTron, pyehr checkpoints, or fusion training.

## Transfer policy

Small license-compatible manifests, checksums, and scripts may be versioned. Weights, corpora, datasets, patient-level outputs, secrets, logs, caches, and generated indexes remain ignored and move to the L20 through controlled file transfer rather than GitHub.

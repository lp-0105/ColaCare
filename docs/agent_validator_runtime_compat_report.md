# Agent retry4 validator runtime compatibility report

## Scope and outcome

This was a local-only compatibility and verification change. No L20 system,
browser control, file-transfer client, Qwen model, MIMIC record, fixed real
sample, Agent benchmark, Fusion run, or formal RAG run was accessed or started.

The fix uses **scheme A: guarded replacement of legacy calls**. The three
DoctorAgent/MetaAgent call sites in the SHA-pinned retry2 template are replaced
with the canonical `validate_prediction_response` callback. The generator
requires exactly three matches and fails if the template drifts. No validator
implementation was copied and no `NameError` fallback or compatibility alias
was introduced.

## Why 143 passing tests did not detect the failure

The previous tests validated the tracked generator with a shortened fixture and
validated the new response contract directly. That fixture contained the old
import but omitted all three runtime `validate_prediction` calls. `compileall`
compiled the tracked generator, not the source emitted later on L20. Even a
syntax compile of that emitted source would not resolve a global used only when
`run_one` executes. The generated program also constructed the Transformers
client before reaching the first unresolved call, so the error appeared only
after Qwen had loaded.

The missing layers were: materializing the complete SHA-pinned runtime, truly
importing it, resolving callback symbols/signatures, executing the actual
DoctorAgent and MetaAgent call graph with a stub, and blocking execution with a
pre-model preflight.

## Evidence and call sites

- Canonical validator: `utils/agent_response_contract.py`,
  `validate_prediction_response(value: Mapping[str, Any]) -> dict[str, Any]`.
- Compatibility wrapper still used by unrelated older smoke paths:
  `utils/prediction_schema.py`, `validate_prediction`; it delegates to the
  canonical function and was not weakened.
- SHA-pinned template:
  `tests/fixtures/run_real_agent_benchmark_retry2_template.py.txt`.
- Normalized template SHA256:
  `21977f9c0d993bead3f8d437e6adf0a750931433521edbd41f5b53fe54e2fda8`.
- Legacy references in that template: DoctorAgent (line 94), initial MetaAgent
  (line 99), and final MetaAgent (line 108).
- Generator entry: `scripts/run_real_agent_benchmark_retry4.py`,
  `materialize_retry4_runner`.
- Final local generated runner (ignored, not committed):
  `tmp/agent_runtime_compat_v3/run_real_agent_benchmark_retry4.generated.py`.
- Generated runner SHA256:
  `531c4f019e59f7d7c15394387f1245cc8358a4a2335a8f0dbedb639f0fde1660`.
- Generation manifest:
  `tmp/agent_runtime_compat_v3/generated_runner_manifest.json`.

## Generated runner and preflight

The complete retry2 body is retained inside an import-safe `deployed_main`.
Importing the generated module does not read sample bodies, create the Agent
output directory, construct the Transformers client, allocate CUDA model
memory, or make an LLM call. Production factories remain unchanged and the
synthetic injection switch defaults to false.

Before `runpy` invokes `deployed_main`, the launcher performs a fail-closed
preflight that checks:

- Python executable, runner existence/SHA, `py_compile`, and true import;
- canonical DoctorAgent and MetaAgent callbacks and their one-value signature;
- repair callback signature and structured validator return behavior;
- rejection of 641 Unicode characters;
- unchanged target 500, hard limit 640, and two repairs;
- absence of silent truncation and unresolved `validate_prediction` loads;
- exactly three canonical prediction-validator call sites;
- isolation from retry3 and the failed retry4 directory;
- absence of the new `agent_outputs_v1_retry4_runtimefix_v3` directory;
- exact placeholder-KB marker;
- absence of label/identifier fields from the declared Agent input contract;
- exactly 32 frozen anonymous manifest entries in fixed order;
- production dependency-injection defaults.

Any failed check produces a non-passing JSON report and exit code 4 before
Qwen, sample-body reads, output creation, or LLM calls. The verified report is
`tmp/agent_runtime_compat_v3/agent_runtime_preflight_report.json`: `passed=true`,
`qwen_loaded=false`, and `llm_calls_started=false`.

## Synthetic dynamic integration smoke

The integration test materializes and imports the real generated runner, then
injects a no-network fake client, fake CUDA surface, and fake resource monitor.
Its inputs are 32 anonymous synthetic fixtures generated in a temporary
directory and explicitly marked `SYNTHETIC STUB — NOT REAL PATIENT DATA`.

The actual runner performed its fixed 8 gate, continued to 32 synthetic
samples, reused the first 8, and made two deterministic repeat checks. Each of
the 34 executions traversed two DoctorAgents, initial MetaAgent, one two-doctor
discussion round, and final MetaAgent. Every stage first returned a 641-character
summary and then used the summary-only repair path. The test observed:

- 32/32 primary synthetic outputs passed, with two DoctorAgents and both
  MetaAgent stages present;
- 408 stub LLM calls (204 initial responses plus 204 summary repairs);
- 272 real calls to the canonical prediction validator: 34 executions × four
  prediction stages × initial/repaired validation;
- 192 unique initial raw response files and 192 matching SHA256 files;
- one stub client construction, no Qwen load, no network request, and no real
  data access;
- all privacy gates passed and no outcome label or patient identifier appeared;
- retry3 and failed retry4 outputs remained protected;
- PLACEHOLDER KB remained exactly
  `PLACEHOLDER KB — NOT PAPER RAG REPRODUCTION`.

The validator was wrapped only to count invocations; its real implementation
executed and was not replaced with a pass-through.

The dynamic smoke also exposed a self-match in the generated privacy metadata:
the key `test_metrics_provided=false` contained the prohibited token
`test_metric`. The generated negative audit key is now
`evaluation_metrics_provided=false`; the scanner and privacy rules are
unchanged.

## Contract and regression status

The response contract remains unchanged:

- prompt target: 500 Unicode characters;
- validator hard limit: 640 Unicode characters;
- maximum summary-only repairs: two;
- repairs may change only `reasoning_summary`;
- original response and SHA256 are retained;
- Markdown and silent truncation remain prohibited.

The full local suite passed `149 passed`. It covers short summaries, 639/640
acceptance, 641 rejection, Unicode character counting, protected probability,
evidence and conclusion fields, repair exhaustion, raw-response preservation,
generated-runner compile/import, preflight success/failure, and the full
synthetic DoctorAgent/MetaAgent runtime path.

No clinical reasoning semantics, prompt clinical content, model parameters,
expert probabilities, formal data, KB content, fixed real samples, retry
failure limits, privacy rules, label gates, or patient identifiers were changed.

"""Run the real 8/32 Agent gate with summary-only JSON repair.

This launcher intentionally reuses the already-validated retry2 runtime source
from ``RUNROOT``. It writes only to ``agent_outputs_v1_retry4`` and new retry4
report names. The script contains no data, identifiers, checkpoints, or
machine-specific paths; callers provide paths through environment variables.
"""

from __future__ import annotations

import os
from pathlib import Path
import re
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def retry4_resume_is_valid(previous: dict, source_input_sha256: str) -> bool:
    return (
        previous.get("status") == "PASS"
        and previous.get("source_input_sha256") == source_input_sha256
    )


def _replace_once(source: str, old: str, new: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(
            f"retry2 source contract changed for {old!r}: expected 1, found {count}"
        )
    return source.replace(old, new, 1)


def _replace_block(source: str, pattern: str, replacement: str, name: str) -> str:
    rendered, count = re.subn(pattern, replacement, source, count=1, flags=re.DOTALL)
    if count != 1:
        raise RuntimeError(f"retry2 source contract changed: missing {name} block")
    return rendered


_ASK_REPLACEMENT = '''def ask(client,messages,schema,validator,sid,stage,max_attempts=3):
 # Contract: target at most 500 Unicode characters; hard limit 640 Unicode characters; Do not use Markdown.
 before=len(client.calls); started=time.perf_counter()
 contract_validator=(validate_prediction_response if set(schema.get('required',[]))=={'risk_probability','prediction','reasoning_summary'} else validate_discussion_response)
 raw_path=RESULTS/'raw_responses'/sid/f'{stage}.initial.raw.txt'
 try:
  outcome=request_with_summary_repair(client,messages,response_schema=schema,validator=contract_validator,original_response_path=raw_path)
  new=client.calls[before:]
  records=[]
  for offset,item in enumerate(new):
   kind='INITIAL_REASONING' if offset==0 else 'SUMMARY_REPAIR'
   records.append({**item,'sample_id':sid,'stage':stage,'attempt':offset+1,'call_kind':kind,'status':'PASS'})
  return outcome.payload,{'attempts':1+outcome.repair_attempts,'repair_attempts':outcome.repair_attempts,'errors':outcome.repair_errors,'wall_seconds':round(time.perf_counter()-started,4),'calls':records,'usage':{'prompt_tokens':outcome.prompt_tokens,'completion_tokens':outcome.completion_tokens},'original_response_sha256':outcome.original_response_sha256,'original_summary_length':outcome.original_summary_length}
 except Exception as exc:
  new=client.calls[before:]
  records=[]
  for offset,item in enumerate(new):
   kind='INITIAL_REASONING' if offset==0 else 'SUMMARY_REPAIR'
   records.append({**item,'sample_id':sid,'stage':stage,'attempt':offset+1,'call_kind':kind,'status':'PARSE_OR_VALIDATION_FAIL'})
  return None,{'attempts':len(new),'repair_attempts':max(0,len(new)-1),'errors':[{'attempt':len(new),'type':type(exc).__name__,'message':str(exc)[:500]}],'wall_seconds':round(time.perf_counter()-started,4),'calls':records,'usage':{'prompt_tokens':sum(int(r.get('input_tokens',0)) for r in records),'completion_tokens':sum(int(r.get('output_tokens',0)) for r in records)}}
'''


def build_retry4_source(source: str) -> str:
    """Apply guarded, auditable transformations to the validated retry2 runner."""
    source = _replace_once(
        source,
        "from utils.prediction_schema import PREDICTION_SCHEMA, validate_prediction",
        "from utils.agent_response_contract import (DISCUSSION_RESPONSE_SCHEMA, "
        "PREDICTION_RESPONSE_SCHEMA as PREDICTION_SCHEMA, SUMMARY_MAX_CHARS, "
        "SUMMARY_PROMPT_TARGET_CHARS, request_with_summary_repair, "
        "validate_discussion_response, validate_prediction_response)",
    )
    source = _replace_once(
        source,
        "RESULTS=RUNROOT/'agent_outputs_v1_retry2'",
        "RESULTS=RUNROOT/'agent_outputs_v1_retry4'",
    )
    source = _replace_block(
        source,
        r"DISCUSSION_SCHEMA=.*?\ndef sha_bytes",
        "DISCUSSION_SCHEMA=DISCUSSION_RESPONSE_SCHEMA\ndef sha_bytes",
        "discussion schema",
    )
    source = _replace_block(
        source,
        r"def validate_discussion\(v\):.*?\ndef core_prediction",
        "def validate_discussion(v): return validate_discussion_response(v)\n"
        "def core_prediction",
        "discussion validator",
    )
    source = _replace_block(
        source,
        r"def ask\(client,messages,schema,validator,sid,stage,max_attempts=3\):.*?\nprov=",
        _ASK_REPLACEMENT + "prov=",
        "ask",
    )
    source = _replace_once(
        source,
        "client=DirectTransformersClient(MODEL,device='cuda:0',context_length=4096,max_new_tokens=128)",
        "client=DirectTransformersClient(MODEL,device='cuda:0',context_length=4096,max_new_tokens=256)",
    )
    source = _replace_once(source, "'max_new_tokens':128", "'max_new_tokens':256")
    source = _replace_once(
        source,
        "LOGDIR/'agent_smoke_8_report.json'",
        "LOGDIR/'agent_smoke_8_report_retry4.json'",
    )
    source = _replace_once(
        source,
        "LOGDIR/'agent_benchmark_32_report.json'",
        "LOGDIR/'agent_benchmark_32_report_retry4.json'",
    )
    return source


def main() -> None:
    runroot = Path(os.environ["RUNROOT"])
    source_path = Path(
        os.environ.get(
            "RETRY4_SOURCE",
            str(runroot / "run_real_agent_benchmark_retry2.py"),
        )
    )
    if not source_path.is_file():
        raise FileNotFoundError(f"validated retry2 source is missing: {source_path}")
    rendered = build_retry4_source(source_path.read_text(encoding="utf-8"))
    exec(compile(rendered, str(source_path.resolve()), "exec"), globals(), globals())


if __name__ == "__main__":
    main()

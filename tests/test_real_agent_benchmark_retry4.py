import unittest


MINIMAL_RETRY2_SOURCE = """\
from utils.prediction_schema import PREDICTION_SCHEMA, validate_prediction
INPUTS=RUNROOT/'real_agent_inputs_32'; RESULTS=RUNROOT/'agent_outputs_v1_retry2'; RESULT_FILES=RESULTS/'results'
DISCUSSION_SCHEMA={'type':'object','properties':{'reasoning_summary':{'type':'string','maxLength':600}}}
def sha_bytes(value): return 'sha'
def validate_discussion(v):
 return v
def core_prediction(value): return value
def ask(client,messages,schema,validator,sid,stage,max_attempts=3):
 errors=[]
 return None,{'attempts':max_attempts,'errors':errors}
prov=json.loads((INPUTS/'input_provenance_manifest.json').read_text(encoding='utf-8'))
client=DirectTransformersClient(MODEL,device='cuda:0',context_length=4096,max_new_tokens=128)
run_manifest={'generation':{'max_new_tokens':128}}
messages=[{'role':'system','content':'Return only the requested JSON.'}]
atomic_json(LOGDIR/'agent_smoke_8_report.json',smoke)
atomic_json(LOGDIR/'agent_benchmark_32_report.json',benchmark)
"""


class Retry4SourceTests(unittest.TestCase):
    def test_retry4_uses_isolated_output_and_leaves_retry3_untouched(self):
        from scripts.run_real_agent_benchmark_retry4 import build_retry4_source

        rendered = build_retry4_source(MINIMAL_RETRY2_SOURCE)
        self.assertIn("agent_outputs_v1_retry4", rendered)
        self.assertNotIn("agent_outputs_v1_retry3", rendered)
        self.assertNotIn("RESULTS=RUNROOT/'agent_outputs_v1_retry2'", rendered)
        self.assertIn("agent_smoke_8_report_retry4.json", rendered)
        self.assertIn("agent_benchmark_32_report_retry4.json", rendered)

    def test_retry4_uses_contract_repair_not_full_reinference_loop(self):
        from scripts.run_real_agent_benchmark_retry4 import build_retry4_source

        rendered = build_retry4_source(MINIMAL_RETRY2_SOURCE)
        self.assertIn("request_with_summary_repair", rendered)
        self.assertNotIn("for attempt in range(1,max_attempts+1)", rendered)
        self.assertIn("max_new_tokens=256", rendered)
        self.assertIn("'max_new_tokens':256", rendered)

    def test_retry4_prompt_and_schema_contract_are_aligned(self):
        from scripts.run_real_agent_benchmark_retry4 import build_retry4_source

        rendered = build_retry4_source(MINIMAL_RETRY2_SOURCE)
        self.assertIn("SUMMARY_MAX_CHARS", rendered)
        self.assertIn("SUMMARY_PROMPT_TARGET_CHARS", rendered)
        self.assertIn("500 Unicode characters", rendered)
        self.assertIn("640 Unicode characters", rendered)
        self.assertIn("Do not use Markdown", rendered)

    def test_retry4_preserves_resume_rule_for_matching_pass_results(self):
        from scripts.run_real_agent_benchmark_retry4 import retry4_resume_is_valid

        previous = {"status": "PASS", "source_input_sha256": "abc"}
        self.assertTrue(retry4_resume_is_valid(previous, "abc"))
        self.assertFalse(retry4_resume_is_valid(previous, "different"))
        self.assertFalse(retry4_resume_is_valid({"status": "FAIL", "source_input_sha256": "abc"}, "abc"))


if __name__ == "__main__":
    unittest.main()

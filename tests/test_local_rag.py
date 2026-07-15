import json
from pathlib import Path
import unittest

from utils.local_rag import load_synthetic_guidelines, retrieve_lexically


ROOT = Path(__file__).resolve().parents[1]
GUIDELINES = ROOT / "tests" / "fixtures" / "synthetic_guidelines.json"


class LocalRAGTests(unittest.TestCase):
    def test_fixture_is_fictional_and_loadable(self):
        documents = load_synthetic_guidelines(GUIDELINES)
        self.assertEqual(len(documents), 3)
        self.assertTrue(all(document["synthetic"] for document in documents))

    def test_lexical_retrieval_is_deterministic(self):
        documents = load_synthetic_guidelines(GUIDELINES)
        first = retrieve_lexically("low oxygen saturation respiratory rate", documents, top_k=2)
        second = retrieve_lexically("low oxygen saturation respiratory rate", documents, top_k=2)
        self.assertEqual(first, second)
        self.assertEqual(first[0]["id"], "SYN-GUIDE-RESP-001")
        self.assertGreater(first[0]["score"], 0)

    def test_non_fictional_document_is_rejected(self):
        documents = json.loads(GUIDELINES.read_text(encoding="utf-8"))
        documents[0]["synthetic"] = False
        with self.assertRaisesRegex(ValueError, "synthetic"):
            retrieve_lexically("oxygen", documents)


if __name__ == "__main__":
    unittest.main()

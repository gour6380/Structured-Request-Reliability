"""Reference integrity, ID pairing, and honest smoke acceptance."""

import copy
import hashlib
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import MagicMock, patch

from request_reliability.metrics import evaluate_predictions, score_prediction
from request_reliability.regression import (
    SYSTEMS,
    audit_reference,
    build_regression_report,
    compare_predictions,
    fixed_reference_cases,
)

ONTOLOGY = {"intents": ["alarm_set", "alarm_query"], "slots": ["time"]}


def example(identifier, correct=True):
    target = {"intent": "alarm_set", "slots": []}
    text = json.dumps(target if correct else {"intent": "alarm_query", "slots": []})
    return {
        "id": identifier,
        "utterance": f"request {identifier}",
        "target": target,
        "novel_input": True,
        "prediction_text": text,
        "generated_token_ids": [1, 2],
        **score_prediction(text, target, ONTOLOGY["intents"], ONTOLOGY["slots"]),
    }


def write_bundle(bundle, *, first_correct=False):
    records = [example("z-first", first_correct), example("a-second"), example("m-third")]
    files = {}

    def write(relative, value, jsonl=False):
        path = bundle / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        text = "".join(json.dumps(row) + "\n" for row in value) if jsonl else json.dumps(value)
        contents = text.encode()
        path.write_bytes(contents)
        files[relative] = {"sha256": hashlib.sha256(contents).hexdigest(), "bytes": len(contents)}

    write("ontology.json", ONTOLOGY)
    gold = [
        {key: record[key] for key in ("id", "utterance", "target", "novel_input")}
        for record in records
    ]
    write("examples/test.jsonl", gold, True)
    write("examples/tokenized_test.jsonl", [{**row, "prompt_ids": [10, 11]} for row in gold], True)
    for system in SYSTEMS:
        write(f"reference/{system}.jsonl", records, True)
        write(
            f"reference/{system}.metrics.json",
            evaluate_predictions(records, ONTOLOGY["intents"], ONTOLOGY["slots"]),
        )
    (bundle / "manifest.json").write_text(json.dumps({"files": files}))
    return records


def update_bundle_file(bundle, relative, mutate, *, jsonl=False):
    path = bundle / relative
    values = (
        [json.loads(line) for line in path.read_text().splitlines()]
        if jsonl
        else json.loads(path.read_text())
    )
    mutate(values)
    contents = (
        "".join(json.dumps(row) + "\n" for row in values) if jsonl else json.dumps(values)
    ).encode()
    path.write_bytes(contents)
    manifest = json.loads((bundle / "manifest.json").read_text())
    manifest["files"][relative] = {
        "sha256": hashlib.sha256(contents).hexdigest(),
        "bytes": len(contents),
    }
    (bundle / "manifest.json").write_text(json.dumps(manifest))


class ComparisonTests(unittest.TestCase):
    def test_pairing_uses_ids_and_recomputes_instead_of_saved_flags(self):
        reference = [example("1"), example("2", False)]
        candidate = copy.deepcopy(list(reversed(reference)))
        candidate[1]["prediction_text"] = "not JSON"
        # Preserve misleading original True flags: they must not affect scoring.
        report = compare_predictions(reference, candidate, ONTOLOGY)
        self.assertEqual(report["regressions"]["joint_exact_match"], ["1"])
        self.assertEqual(report["regressions"]["schema_valid"], ["1"])
        self.assertEqual(report["candidate_metrics"]["joint_exact_match"], 0)

    def test_missing_extra_duplicate_and_empty_ids_fail(self):
        reference = [example("1"), example("2")]
        for candidate in (
            [example("1")],
            [example("1"), example("3")],
            [example("1"), example("1")],
            [example("1"), example("")],
        ):
            with self.subTest(candidate=[row["id"] for row in candidate]):
                with self.assertRaises(ValueError):
                    compare_predictions(reference, candidate, ONTOLOGY)

    def test_conflicting_gold_rejected_but_absent_candidate_gold_allowed(self):
        reference = [example("1")]
        minimal = [{"id": "1", "prediction_text": reference[0]["prediction_text"]}]
        report = compare_predictions(reference, minimal, ONTOLOGY)
        self.assertEqual(report["token_agreement"]["unavailable"], 1)
        candidate = copy.deepcopy(reference)
        candidate[0]["target"]["intent"] = "alarm_query"
        with self.assertRaisesRegex(ValueError, "inconsistent target"):
            compare_predictions(reference, candidate, ONTOLOGY)

    def test_text_and_token_agreement_are_distinct(self):
        reference = [example("1")]
        candidate = copy.deepcopy(reference)
        candidate[0]["generated_token_ids"] = [1, 3]
        report = compare_predictions(reference, candidate, ONTOLOGY)
        self.assertEqual(report["text_agreement"]["equal"], 1)
        self.assertEqual(report["token_agreement"]["equal"], 0)
        self.assertEqual(len(report["mismatches"]), 1)


class BundleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.bundle = Path(self.temp.name)
        self.records = write_bundle(self.bundle)

    def test_full_reference_audit_recomputes_all_systems(self):
        result = audit_reference(self.bundle)
        self.assertEqual(result["status"], "passed")
        self.assertFalse(result["inference_performed"])
        self.assertEqual(set(result["systems"]), set(SYSTEMS))
        self.assertTrue(
            all(row["saved_aggregate_metrics_checked"] for row in result["systems"].values())
        )

    def test_manifest_corruption_detected(self):
        with (self.bundle / "reference/lora.jsonl").open("a") as handle:
            handle.write("\n")
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            audit_reference(self.bundle)

    def test_saved_score_lie_detected_even_with_valid_checksum(self):
        update_bundle_file(
            self.bundle,
            "reference/lora.jsonl",
            lambda rows: rows[0].update(joint_exact_match=True),
            jsonl=True,
        )
        with self.assertRaisesRegex(ValueError, "saved score mismatch"):
            audit_reference(self.bundle)

    def test_gold_disagreement_detected_even_with_valid_checksum(self):
        update_bundle_file(
            self.bundle,
            "reference/qlora.jsonl",
            lambda rows: rows[0].update(utterance="wrong input"),
            jsonl=True,
        )
        with self.assertRaisesRegex(ValueError, "inconsistent utterance"):
            audit_reference(self.bundle)

    def test_tokenized_gold_mismatch_detected(self):
        update_bundle_file(
            self.bundle,
            "examples/tokenized_test.jsonl",
            lambda rows: rows[0].update(utterance="different request"),
            jsonl=True,
        )
        with self.assertRaisesRegex(ValueError, "inconsistent utterance"):
            audit_reference(self.bundle)
        with self.assertRaisesRegex(ValueError, "inconsistent utterance"):
            fixed_reference_cases(self.bundle, count=2)

    def test_aggregate_lie_detected_even_with_valid_checksum(self):
        update_bundle_file(
            self.bundle,
            "reference/base.metrics.json",
            lambda row: row.update(joint_exact_match=1.0),
        )
        with self.assertRaisesRegex(ValueError, "saved aggregate mismatch"):
            audit_reference(self.bundle)

    def test_reference_missing_and_duplicate_ids_detected(self):
        update_bundle_file(self.bundle, "reference/base.jsonl", lambda rows: rows.pop(), jsonl=True)
        with self.assertRaisesRegex(ValueError, "missing IDs"):
            audit_reference(self.bundle)
        write_bundle(self.bundle)
        update_bundle_file(
            self.bundle, "reference/base.jsonl", lambda rows: rows.append(rows[0]), jsonl=True
        )
        with self.assertRaisesRegex(ValueError, "duplicate ID"):
            audit_reference(self.bundle)

    def test_prefix_preserves_original_order_and_does_not_select_successes(self):
        cases = fixed_reference_cases(self.bundle, 2)
        self.assertEqual([row["id"] for row in cases], ["z-first", "a-second"])
        self.assertEqual(cases[0]["prompt_ids"], [10, 11])
        self.assertNotIn("input_ids", cases[0])
        self.assertNotIn("labels", cases[0])

    def test_subset_pass_is_never_full_reference_parity(self):
        report = build_regression_report(self.bundle, self.records[:2], "lora", {"case_count": 2})
        self.assertTrue(report["acceptance"]["passed"])
        self.assertTrue(report["acceptance"]["output_parity_passed"])
        self.assertFalse(report["acceptance"]["full_reference_evaluated"])
        self.assertFalse(report["acceptance"]["full_reference_output_parity"])
        self.assertFalse(report["acceptance"]["full_equivalence_demonstrated"])

    def test_report_does_not_infer_scope_from_candidate_count(self):
        with self.assertRaisesRegex(ValueError, "missing IDs"):
            build_regression_report(self.bundle, self.records[:1], "lora", {"case_count": 2})
        with self.assertRaisesRegex(ValueError, "duplicate ID"):
            build_regression_report(self.bundle, [self.records[0]] * 2, "lora", {"case_count": 2})

    def test_quality_regression_is_not_offset_by_improvements(self):
        candidate = copy.deepcopy(self.records[:2])
        candidate[0]["prediction_text"] = json.dumps(candidate[0]["target"])
        candidate[1]["prediction_text"] = "invalid"
        report = build_regression_report(self.bundle, candidate, "lora", {"case_count": 2})
        self.assertFalse(report["acceptance"]["quality_passed"])
        self.assertFalse(report["acceptance"]["passed"])
        self.assertEqual(report["comparison"]["improvements"]["joint_exact_match"], ["z-first"])

    def test_quality_pass_can_differ_from_output_parity(self):
        candidate = copy.deepcopy(self.records[:2])
        candidate[1]["prediction_text"] = json.dumps(candidate[1]["target"], separators=(",", ":"))
        report = build_regression_report(self.bundle, candidate, "lora", {"case_count": 2})
        self.assertTrue(report["acceptance"]["quality_passed"])
        self.assertFalse(report["acceptance"]["output_parity_passed"])
        self.assertTrue(report["acceptance"]["passed"])
        strict = build_regression_report(
            self.bundle, candidate, "lora", {"case_count": 2, "require_text_parity": True}
        )
        self.assertFalse(strict["acceptance"]["passed"])

    def test_full_reference_scope_is_explicit(self):
        report = build_regression_report(self.bundle, self.records, "lora", {"case_count": None})
        self.assertTrue(report["acceptance"]["full_reference_output_parity"])
        self.assertFalse(report["acceptance"]["full_equivalence_demonstrated"])

    def test_invalid_thresholds_rejected(self):
        for settings in (
            {"case_count": 0},
            {"max_exact_match_regressions": -1},
            {"require_text_parity": "false"},
            {"typo": 1},
        ):
            with self.subTest(settings=settings):
                with self.assertRaises(ValueError):
                    build_regression_report(self.bundle, self.records[:2], "lora", settings)

    def test_cli_stops_before_generation_on_prompt_token_mismatch(self):
        # Exercise the real CLI integration without model imports or downloads.
        from request_reliability.cli import run_compare

        project = Path(self.temp.name) / "independent-project"
        bundle = project / "data/frozen"
        bundle.mkdir(parents=True)
        write_bundle(bundle)
        (project / "configs").mkdir()
        (project / "configs/regression.json").write_text(json.dumps({"case_count": 2}))
        engine = MagicMock()
        engine.__enter__.return_value = engine
        engine.identity = {"system": "lora"}
        engine.prompt_ids.return_value = [10, 99]
        args = Namespace(
            count=2,
            systems=["lora"],
            device="mps",
            timeout=1,
            output=str(project / "reports/prompt-mismatch"),
        )
        with patch("request_reliability.inference.FrozenEngine", return_value=engine):
            result, exit_code = run_compare(args, bundle)
        engine.predict.assert_not_called()
        self.assertEqual(exit_code, 2)
        self.assertFalse(result["passed"])
        self.assertIn("Prompt tokens differ", result["systems"]["lora"]["error"])
        self.assertEqual(result["systems"]["lora"]["completed_requests"], 0)


if __name__ == "__main__":
    unittest.main()

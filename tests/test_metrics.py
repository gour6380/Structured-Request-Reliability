"""Strict output validation and occurrence-aware scoring contracts."""

import json
import unittest

from request_reliability.metrics import evaluate_predictions, score_prediction, validate_prediction


class StrictMetricsTests(unittest.TestCase):
    intents = ["alarm_set", "alarm_query"]
    slots = ["time", "date"]
    target = {
        "intent": "alarm_set",
        "slots": [{"type": "time", "value": "five am"}],
    }

    def score(self, text):
        return score_prediction(text, self.target, self.intents, self.slots)

    def test_valid_json_is_not_repaired_or_extracted(self):
        valid = json.dumps(self.target)
        invalid = [
            f"```json\n{valid}\n```",
            f"Result: {valid}",
            valid + " trailing",
            valid + valid,
            "{",
            "",
            '{"intent":NaN,"slots":[]}',
            '{"intent":"alarm_set","intent":"alarm_query","slots":[]}',
            '{"intent":"alarm_set","slots":[{"type":"time","type":"date","value":"five am"}]}',
        ]
        for text in invalid:
            with self.subTest(text=text):
                score = self.score(text)
                self.assertFalse(score["json_valid"])
                self.assertFalse(score["joint_exact_match"])
                self.assertEqual(score["slot_fn"], 1)
                self.assertIn("invalid_json", score["failure_reasons"])

    def test_extra_keys_and_bad_types_are_invalid_structures(self):
        invalid = [
            [],
            None,
            {**self.target, "explanation": "ok"},
            {"intent": "alarm_set", "slots": {}},
            {"intent": 1, "slots": []},
            {"intent": "alarm_set", "slots": [{"type": "time", "value": ""}]},
            {
                "intent": "alarm_set",
                "slots": [{"type": "time", "value": "five am", "confidence": 1}],
            },
        ]
        for value in invalid:
            with self.subTest(value=value):
                score = self.score(json.dumps(value))
                self.assertTrue(score["json_valid"])
                self.assertFalse(score["structure_valid"])
                self.assertIn("invalid_structure", score["failure_reasons"])

    def test_unknown_labels_keep_field_diagnostics_but_fail_schema(self):
        score = self.score('{"intent":"unknown","slots":[{"type":"time","value":"five am"}]}')
        self.assertFalse(score["schema_valid"])
        self.assertEqual(score["slot_tp"], 1)
        self.assertIn("unknown_intent", score["failure_reasons"])
        score = self.score('{"intent":"alarm_set","slots":[{"type":"unknown","value":"five am"}]}')
        self.assertTrue(score["intent_correct"])
        self.assertFalse(score["schema_valid"])
        self.assertEqual((score["slot_tp"], score["slot_fp"], score["slot_fn"]), (0, 1, 1))
        self.assertIn("unknown_slot_type", score["failure_reasons"])

    def test_repeated_slot_occurrences_are_not_collapsed(self):
        duplicated = {**self.target, "slots": self.target["slots"] * 2}
        score = self.score(json.dumps(duplicated))
        self.assertFalse(score["joint_exact_match"])
        self.assertEqual((score["slot_tp"], score["extra_slots"]), (1, 1))
        score = score_prediction(json.dumps(self.target), duplicated, self.intents, self.slots)
        self.assertEqual((score["slot_tp"], score["missing_slots"]), (1, 1))

    def test_slot_order_ignored_but_case_is_exact(self):
        target = {
            "intent": "alarm_set",
            "slots": [
                {"type": "time", "value": "five am"},
                {"type": "date", "value": "today"},
            ],
        }
        reversed_output = {**target, "slots": list(reversed(target["slots"]))}
        self.assertTrue(
            score_prediction(json.dumps(reversed_output), target, self.intents, self.slots)[
                "joint_exact_match"
            ]
        )
        score = self.score('{"intent":"alarm_set","slots":[{"type":"time","value":"Five AM"}]}')
        self.assertEqual(score["wrong_values"], 1)
        self.assertEqual((score["missing_slots"], score["extra_slots"]), (0, 0))

    def test_wrong_values_pair_only_same_slot_types(self):
        score = self.score('{"intent":"alarm_set","slots":[{"type":"date","value":"five am"}]}')
        self.assertEqual(
            (score["wrong_values"], score["missing_slots"], score["extra_slots"]), (0, 1, 1)
        )

    def test_schema_valid_does_not_mean_correct(self):
        score = self.score('{"intent":"alarm_query","slots":[]}')
        self.assertTrue(score["schema_valid"])
        self.assertFalse(score["joint_exact_match"])
        self.assertEqual(set(score["failure_reasons"]), {"wrong_intent", "missing_slots"})

    def test_invalid_ground_truth_raises(self):
        with self.assertRaisesRegex(ValueError, "Ground truth"):
            score_prediction("{}", {"intent": "unknown", "slots": []}, self.intents, self.slots)

    def test_validation_needs_no_reference_and_reports_unknown_type(self):
        result = validate_prediction('{"intent":"alarm_set","slots":[]}', self.intents, self.slots)
        self.assertTrue(result["schema_valid"])
        self.assertNotIn("joint_exact_match", result)

    def test_aggregate_recomputes_text_and_rejects_duplicate_ids(self):
        record = {
            "id": "0",
            "target": self.target,
            "prediction_text": "bad",
            "joint_exact_match": True,
        }
        result = evaluate_predictions([record], self.intents, self.slots)
        self.assertEqual(result["joint_exact_match"], 0.0)
        self.assertEqual(result["counts"]["slot_fn"], 1)
        with self.assertRaisesRegex(ValueError, "IDs"):
            evaluate_predictions([record, record], self.intents, self.slots)


if __name__ == "__main__":
    unittest.main()

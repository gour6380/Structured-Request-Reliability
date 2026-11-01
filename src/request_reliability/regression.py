"""Audit frozen evidence and compare new outputs without trusting saved scores.

All paths resolve within the supplied bundle. The smoke prefix is chosen
in the original test-file order, independently of any model's successes.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .metrics import evaluate_predictions, score_prediction

SYSTEMS = ("base", "quantized_base", "lora", "qlora")
SCORE_FIELDS = (
    "json_valid",
    "structure_valid",
    "schema_valid",
    "joint_exact_match",
    "intent_correct",
    "predicted_intent",
    "slot_tp",
    "slot_fp",
    "slot_fn",
    "missing_slots",
    "extra_slots",
    "wrong_values",
    "failure_reasons",
    "parsed_prediction",
)
AGGREGATE_FIELDS = (
    "example_count",
    "joint_exact_match",
    "exact_match",
    "json_valid_rate",
    "structure_valid_rate",
    "schema_valid_rate",
    "intent_accuracy",
    "intent_macro_f1",
    "slot_precision",
    "slot_recall",
    "slot_f1",
    "counts",
    "failure_counts",
    "intent_per_class_f1",
    "confusion_matrix",
)
DEFAULT_THRESHOLDS = {
    "case_count": 8,
    "max_exact_match_regressions": 0,
    "max_schema_valid_regressions": 0,
    "require_text_parity": False,
    "require_token_parity": False,
}


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"Blank record in {path.name}:{number}")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Record must be an object in {path.name}:{number}")
            records.append(value)
    return records


def _indexed(records: list[dict[str, Any]], name: str) -> dict[str, dict[str, Any]]:
    result = {}
    for record in records:
        identifier = record.get("id")
        if isinstance(identifier, bool) or not isinstance(identifier, (str, int)):
            raise ValueError(f"{name}: each record must have a string or integer ID")
        identifier = str(identifier)
        if not identifier or identifier in result:
            raise ValueError(f"{name}: empty or duplicate ID {identifier!r}")
        result[identifier] = record
    return result


def _require_same_ids(left: dict, right: dict, name: str) -> None:
    missing, extra = set(left) - set(right), set(right) - set(left)
    if missing or extra:
        raise ValueError(
            f"{name}: missing IDs {sorted(missing)[:10]}, extra IDs {sorted(extra)[:10]}"
        )


def _text(record: dict[str, Any]) -> str:
    value = record.get("prediction_text", record.get("prediction"))
    if not isinstance(value, str):
        raise ValueError(f"Prediction text must be a string for ID {record.get('id')}")
    if "prediction_text" in record and "prediction" in record:
        if record["prediction_text"] != record["prediction"]:
            raise ValueError(f"Conflicting prediction text fields for ID {record.get('id')}")
    return value


def _tokens(record: dict[str, Any]) -> list[int] | None:
    tokens = record.get("generated_token_ids")
    if tokens is not None and (
        not isinstance(tokens, list)
        or not all(type(token) is int and token >= 0 for token in tokens)
    ):
        raise ValueError(f"Invalid generated_token_ids for ID {record.get('id')}")
    return tokens


def _ontology(bundle: Path) -> dict[str, list[str]]:
    ontology = _read_json(bundle / "ontology.json")
    for key in ("intents", "slots"):
        values = ontology[key]
        if (
            not isinstance(values, list)
            or not values
            or not all(isinstance(value, str) and value for value in values)
            or len(values) != len(set(values))
        ):
            raise ValueError(f"Ontology {key} must be a nonempty list of unique strings")
    return ontology


def _verified_evidence_file(bundle: Path, manifest: dict, relative: str) -> Path:
    path = bundle / relative
    record = manifest.get("files", {}).get(relative)
    if record is None:
        raise ValueError(f"Frozen manifest does not identify {relative}")
    if (
        path.is_symlink()
        or not path.is_file()
        or not path.resolve().is_relative_to(bundle.resolve())
    ):
        raise ValueError(f"Frozen evidence must be a local regular file: {relative}")
    contents = path.read_bytes()
    if len(contents) != record["bytes"] or hashlib.sha256(contents).hexdigest() != record["sha256"]:
        raise ValueError(f"Frozen evidence checksum mismatch: {relative}")
    return path


def _check_gold(expected: dict, actual: dict, context: str, *, require: bool) -> None:
    for field in ("target", "utterance", "novel_input"):
        if field not in actual:
            if require:
                raise ValueError(f"{context}: missing {field}")
        elif actual[field] != expected[field]:
            raise ValueError(f"{context}: inconsistent {field}")


def audit_reference(bundle: Path) -> dict[str, Any]:
    """Recompute all four references and validate their original recorded scores.

    Requires manifest hashes for the ontology, test examples, tokenized examples,
    and four reference JSONL files. Optional ``reference/{system}.metrics.json``
    files, when imported, are checked field by field against the recomputation.
    Raises ValueError on conflicting evidence; does not run a model.
    """
    bundle = Path(bundle)
    manifest = _read_json(bundle / "manifest.json")
    verified = []
    for relative in ("ontology.json", "examples/test.jsonl", "examples/tokenized_test.jsonl"):
        _verified_evidence_file(bundle, manifest, relative)
        verified.append(relative)
    ontology = _ontology(bundle)
    gold = _read_jsonl(bundle / "examples/test.jsonl")
    if not gold:
        raise ValueError("Frozen test set is empty")
    gold_index = _indexed(gold, "test examples")
    tokenized = _read_jsonl(bundle / "examples/tokenized_test.jsonl")
    token_index = _indexed(tokenized, "tokenized examples")
    _require_same_ids(gold_index, token_index, "tokenized examples")
    for identifier, example in gold_index.items():
        row = token_index[identifier]
        _check_gold(example, row, f"tokenized ID {identifier}", require=True)
        if (
            not isinstance(row.get("prompt_ids"), list)
            or not row["prompt_ids"]
            or not all(type(token) is int and token >= 0 for token in row["prompt_ids"])
        ):
            raise ValueError(f"Invalid frozen prompt token IDs: {identifier}")
    systems = {}
    for system in SYSTEMS:
        relative = f"reference/{system}.jsonl"
        reference = _read_jsonl(_verified_evidence_file(bundle, manifest, relative))
        verified.append(relative)
        index = _indexed(reference, f"{system} reference")
        _require_same_ids(gold_index, index, f"{system} reference")
        score_fields_checked = 0
        for identifier, example in gold_index.items():
            row = index[identifier]
            _check_gold(example, row, f"{system} ID {identifier}", require=True)
            score = score_prediction(
                _text(row), example["target"], ontology["intents"], ontology["slots"]
            )
            if _tokens(row) is None:
                raise ValueError(f"{system} ID {identifier}: missing reference token IDs")
            for field in SCORE_FIELDS:
                if (
                    field not in row
                    or type(row[field]) is not type(score[field])
                    or row[field] != score[field]
                ):
                    raise ValueError(f"{system} ID {identifier}: saved score mismatch for {field}")
                score_fields_checked += 1
        recomputed = evaluate_predictions(reference, ontology["intents"], ontology["slots"])
        saved_metric_path = f"reference/{system}.metrics.json"
        aggregate_checked = False
        if saved_metric_path in manifest["files"]:
            saved = _read_json(_verified_evidence_file(bundle, manifest, saved_metric_path))
            verified.append(saved_metric_path)
            for field in AGGREGATE_FIELDS:
                if field not in saved or saved[field] != recomputed[field]:
                    raise ValueError(f"{system}: saved aggregate mismatch for {field}")
            for field in AGGREGATE_FIELDS:
                if (
                    saved.get("novel_input_subset", {}).get(field)
                    != recomputed["novel_input_subset"][field]
                ):
                    raise ValueError(f"{system}: novel-input aggregate mismatch for {field}")
            aggregate_checked = True
        systems[system] = {
            "metrics": recomputed,
            "saved_score_fields_checked": score_fields_checked,
            "saved_aggregate_metrics_checked": aggregate_checked,
        }
    return {
        "status": "passed",
        "example_count": len(gold),
        "systems": systems,
        "verified_evidence_files": verified,
        "inference_performed": False,
        "qualification": (
            "Recomputed imported source-comparison outputs; "
            "this is evidence integrity, not new local inference."
        ),
    }


def fixed_reference_cases(bundle: Path, count: int = 8) -> list[dict[str, Any]]:
    """Return the first count test cases and frozen prompt IDs in original order.

    Targets are retained for scoring, but prompt_ids contain only the model input.
    Never feed tokenized input_ids/labels (which include answers) into generation.
    """
    if type(count) is not int or count < 1:
        raise ValueError("case count must be a positive integer")
    bundle = Path(bundle)
    examples = _read_jsonl(bundle / "examples/test.jsonl")
    example_index = _indexed(examples, "test examples")
    token_index = _indexed(
        _read_jsonl(bundle / "examples/tokenized_test.jsonl"), "tokenized examples"
    )
    _require_same_ids(example_index, token_index, "tokenized examples")
    if count > len(examples):
        raise ValueError(f"Requested {count} cases but reference contains {len(examples)}")
    cases = []
    for example in examples[:count]:
        tokenized = token_index[str(example["id"])]
        _check_gold(example, tokenized, f"tokenized ID {example['id']}", require=True)
        cases.append({**example, "prompt_ids": tokenized["prompt_ids"]})
    return cases


def compare_predictions(reference: list, candidate: list, ontology: dict) -> dict[str, Any]:
    """Pair exactly the same IDs, recompute metrics and list changed behavior.

    Candidate gold is optional; any supplied gold must match the reference. Missing,
    duplicate or extra IDs are errors. Candidate ordering does not affect pairing.
    """
    if not reference:
        raise ValueError("Cannot compare an empty reference")
    expected = _indexed(reference, "reference")
    actual = _indexed(candidate, "candidate")
    _require_same_ids(expected, actual, "candidate")
    reference_rows, candidate_rows, mismatches = [], [], []
    regressions = {"joint_exact_match": [], "schema_valid": [], "intent_correct": []}
    improvements = {key: [] for key in regressions}
    text_matches = token_matches = tokens_available = 0
    for identifier, original in expected.items():
        updated = actual[identifier]
        _check_gold(original, updated, f"candidate ID {identifier}", require=False)
        left_text, right_text = _text(original), _text(updated)
        left_tokens, right_tokens = _tokens(original), _tokens(updated)
        common = {
            "id": identifier,
            "target": original["target"],
            "novel_input": original.get("novel_input", False),
        }
        reference_rows.append({**common, "prediction_text": left_text})
        candidate_rows.append({**common, "prediction_text": right_text})
        left = score_prediction(left_text, common["target"], ontology["intents"], ontology["slots"])
        right = score_prediction(
            right_text, common["target"], ontology["intents"], ontology["slots"]
        )
        for metric in regressions:
            if left[metric] and not right[metric]:
                regressions[metric].append(identifier)
            if right[metric] and not left[metric]:
                improvements[metric].append(identifier)
        text_equal = left_text == right_text
        text_matches += int(text_equal)
        token_equal = None
        if left_tokens is not None and right_tokens is not None:
            tokens_available += 1
            token_equal = left_tokens == right_tokens
            token_matches += int(token_equal)
        if not text_equal or token_equal is False:
            mismatches.append(
                {
                    "id": identifier,
                    "utterance": original.get("utterance"),
                    "text_equal": text_equal,
                    "tokens_equal": token_equal,
                    "reference_text": left_text,
                    "candidate_text": right_text,
                    "reference_failure_reasons": left["failure_reasons"],
                    "candidate_failure_reasons": right["failure_reasons"],
                    "reference_joint_exact_match": left["joint_exact_match"],
                    "candidate_joint_exact_match": right["joint_exact_match"],
                }
            )
    count = len(reference_rows)
    return {
        "example_count": count,
        "ids": list(expected),
        "reference_metrics": evaluate_predictions(
            reference_rows, ontology["intents"], ontology["slots"]
        ),
        "candidate_metrics": evaluate_predictions(
            candidate_rows, ontology["intents"], ontology["slots"]
        ),
        "regressions": regressions,
        "improvements": improvements,
        "text_agreement": {"equal": text_matches, "compared": count, "rate": text_matches / count},
        "token_agreement": {
            "equal": token_matches,
            "compared": tokens_available,
            "unavailable": count - tokens_available,
            "rate": token_matches / tokens_available if tokens_available else None,
        },
        "mismatches": mismatches,
    }


def build_regression_report(
    bundle: Path, predictions: list, system: str, thresholds: dict
) -> dict[str, Any]:
    """Evaluate the declared prefix (or full reference with case_count=null).

    The quality gate counts per-example regressions; aggregate improvements do not
    cancel regressions. Output parity is reported separately from quality. Passing
    a smoke gate never establishes equivalence on the full benchmark or new data.
    """
    if system not in SYSTEMS:
        raise ValueError(f"Unknown frozen system: {system}")
    unknown = set(thresholds) - set(DEFAULT_THRESHOLDS)
    if unknown:
        raise ValueError(f"Unknown regression thresholds: {sorted(unknown)}")
    settings = {**DEFAULT_THRESHOLDS, **thresholds}
    for key in ("max_exact_match_regressions", "max_schema_valid_regressions"):
        if type(settings[key]) is not int or settings[key] < 0:
            raise ValueError(f"{key} must be a nonnegative integer")
    for key in ("require_text_parity", "require_token_parity"):
        if type(settings[key]) is not bool:
            raise ValueError(f"{key} must be a boolean")
    bundle = Path(bundle)
    all_reference = _read_jsonl(bundle / f"reference/{system}.jsonl")
    reference_index = _indexed(all_reference, "reference")
    count = len(all_reference) if settings["case_count"] is None else settings["case_count"]
    cases = fixed_reference_cases(bundle, count=count)
    all_gold = _indexed(_read_jsonl(bundle / "examples/test.jsonl"), "test examples")
    _require_same_ids(all_gold, reference_index, "reference")
    reference = [reference_index[str(case["id"])] for case in cases]
    for case, row in zip(cases, reference, strict=True):
        _check_gold(case, row, f"reference ID {case['id']}", require=True)
    comparison = compare_predictions(reference, predictions, _ontology(bundle))
    exact_regressions = len(comparison["regressions"]["joint_exact_match"])
    schema_regressions = len(comparison["regressions"]["schema_valid"])
    quality_passed = (
        exact_regressions <= settings["max_exact_match_regressions"]
        and schema_regressions <= settings["max_schema_valid_regressions"]
    )
    text_parity = comparison["text_agreement"]["equal"] == count
    token_parity = comparison["token_agreement"]["equal"] == count
    output_parity = text_parity and token_parity
    passed = (
        quality_passed
        and (not settings["require_text_parity"] or text_parity)
        and (not settings["require_token_parity"] or token_parity)
    )
    full_reference = count == len(all_reference)
    return {
        "schema_version": 1,
        "system": system,
        "case_selection": "Prefix of original test-file order; independent of outcome",
        "reference_example_count": len(all_reference),
        "thresholds": settings,
        "comparison": comparison,
        "acceptance": {
            "quality_passed": quality_passed,
            "text_parity_passed": text_parity,
            "token_parity_passed": token_parity,
            "output_parity_passed": output_parity,
            "passed": passed,
            "full_reference_evaluated": full_reference,
            "full_reference_output_parity": full_reference and output_parity,
            "full_equivalence_demonstrated": False,
        },
        "qualification": (
            "Engineering regression check on reused test data. A subset pass establishes only the "
            "declared subset gate; even full-reference parity does not establish statistical "
            "equivalence, correctness on new requests, or general hardware equivalence."
        ),
    }

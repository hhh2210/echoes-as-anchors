from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.two_stage_common import (  # noqa: E402
    DEFAULT_INJECTION_TEMPLATE,
    apply_until,
    extract_question,
    run_manifest,
    sanitize_until,
    validate_injection_template,
)
from echoes_as_anchors.evaluation.two_stage_eval import _write_results, build_parser  # noqa: E402


class TwoStageCommonTests(unittest.TestCase):
    def test_extracts_last_harness_question(self) -> None:
        prompt = "Question: first?\nAnswer: one\n\nQuestion: final?\nAnswer:"
        self.assertEqual(extract_question(prompt), "final?")

    def test_stop_sanitization_preserves_cot(self) -> None:
        self.assertEqual(sanitize_until(["Question:", "END"]), ["END"])
        self.assertEqual(apply_until("answer END ignored", ["END"]), "answer ")

    def test_injection_requires_exactly_one_question_slot(self) -> None:
        self.assertEqual(
            validate_injection_template(DEFAULT_INJECTION_TEMPLATE),
            DEFAULT_INJECTION_TEMPLATE,
        )
        with self.assertRaises(ValueError):
            validate_injection_template("no question slot")
        with self.assertRaises(ValueError):
            validate_injection_template("{question} and {question}")

    def test_manifest_refuses_to_claim_figure_reproduction(self) -> None:
        manifest = run_manifest(
            model="fixture/model",
            task="gsm8k",
            backend="hf",
            mode="two_stage_echo",
            protocol_id="fixture",
            first_stage_tokens=8,
            second_stage_tokens=8,
            injection_template=DEFAULT_INJECTION_TEMPLATE,
            continue_template="",
            seed=7,
            limit=11,
            dtype="float32",
            tensor_parallel_size=1,
            hf_device_map="cpu",
            model_revision="fixture-revision",
            lm_eval_version="0.4.12",
        )
        self.assertIsNone(manifest["paper_figure_claim"])
        self.assertEqual(manifest["seed"], 7)
        self.assertEqual(manifest["lm_eval_version"], "0.4.12")
        self.assertEqual(manifest["limit"], 11)
        self.assertEqual(manifest["model_revision"], "fixture-revision")

    def test_committed_config_records_exact_protocol(self) -> None:
        config = json.loads(
            (ROOT / "configs/ep_standalone_gsm8k.json").read_text(encoding="utf-8")
        )
        self.assertEqual(config["injection_template"], DEFAULT_INJECTION_TEMPLATE)
        self.assertIn("not-figure4", config["protocol_id"])

    def test_legacy_cli_option_aliases_remain_parseable(self) -> None:
        args = build_parser(
            [
                "--model_path",
                "fixture/model",
                "--tasks",
                "gsm8k",
                "--output_dir",
                "/tmp/fixture-output",
                "--first_stage_tokens",
                "13",
                "--second_stage_tokens",
                "17",
                "--injection_template",
                "Recheck {question}",
                "--continue_template",
                "Continue carefully.",
                "--tensor_parallel_size",
                "2",
                "--hf_device_map",
                "cpu",
            ]
        ).parse_args(
            [
                "--model_path",
                "fixture/model",
                "--tasks",
                "gsm8k",
                "--output_dir",
                "/tmp/fixture-output",
                "--first_stage_tokens",
                "13",
                "--second_stage_tokens",
                "17",
                "--injection_template",
                "Recheck {question}",
                "--continue_template",
                "Continue carefully.",
                "--tensor_parallel_size",
                "2",
                "--hf_device_map",
                "cpu",
            ]
        )
        self.assertEqual(args.model_path, "fixture/model")
        self.assertEqual(args.task, "gsm8k")
        self.assertEqual(args.first_stage_tokens, 13)
        self.assertEqual(args.second_stage_tokens, 17)
        self.assertEqual(args.injection_template, "Recheck {question}")
        self.assertEqual(args.continue_template, "Continue carefully.")
        self.assertEqual(args.tensor_parallel_size, 2)
        self.assertEqual(args.hf_device_map, "cpu")

    def test_lm_eval_sample_mapping_is_written_as_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            _write_results(
                output,
                {"results": {}, "samples": {"gsm8k": [{"doc_id": 1}]}},
                {"schema_version": 1},
                "gsm8k",
            )
            sample_files = list(output.glob("samples_gsm8k_*.jsonl"))
            self.assertEqual(len(sample_files), 1)
            self.assertEqual(json.loads(sample_files[0].read_text()), {"doc_id": 1})


if __name__ == "__main__":
    unittest.main()

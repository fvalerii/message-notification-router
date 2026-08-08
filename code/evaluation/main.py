"""Local evaluation harness.

Runs the exact same production pipeline as code.main (guardrails ->
media pre-processing -> evidence retrieval -> LLM routing ->
calibration), but against dataset/sample_messages.csv instead of
dataset/messages.csv, since sample_messages.csv is the only file in this
challenge that ships with known-correct `action` and `message_type`
labels. Reports Action Accuracy and Message Type Accuracy so the
pipeline's quality can be checked locally before submitting.

This is NOT the hidden hackathon grader (see problem_statement.md's
"Evaluation" section for the real, hidden rubric) — it's a fast,
free-standing sanity check.

Run from the project root:
    python -m code.evaluation.main
    python -m code.evaluation.main --verbose

A local `.env` file (repo root, git-ignored) is loaded first thing below,
before code.main / code.routing.engine / code.media.audio ever construct
an API client, so ANTHROPIC_API_KEY / GEMINI_API_KEY / GOOGLE_API_KEY can
be set there instead of the shell environment.
"""

from __future__ import annotations

import argparse
import logging
import os

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

# Loaded explicitly here too (code.main below also loads it), so this
# script never silently depends on import order for its own API key
# availability. python-dotenv's load_dotenv() is idempotent to call twice.
load_dotenv()

from code.data.loader import _DatasetTables, _build_message_context, _read_csv
from code.data.schemas import MessageContext
from code.main import prepare_context
from code.routing.engine import route_message

logger = logging.getLogger(__name__)

SAMPLE_MESSAGES_CSV = "sample_messages.csv"
MESSAGE_HISTORY_CSV = "message_history.csv"
MESSAGE_EVENTS_CSV = "message_events.csv"


def load_sample_contexts(
    dataset_dir: str = "dataset", filename: str = SAMPLE_MESSAGES_CSV
) -> list[MessageContext]:
    """Adapter for code.data.loader.load_dataset(): reuses the exact same
    user/group/business join logic (via the same private _DatasetTables /
    _build_message_context helpers, so this can never drift from how
    dataset/messages.csv is actually joined in production), but ingests
    sample_messages.csv instead of the real prediction target.

    sample_messages.csv has every input column messages.csv has, plus
    ground-truth action/message_type/reason/confidence/evidence_message_ids
    columns appended at the end; _build_message_context only ever reads
    the input columns it needs, so the extra ground-truth columns are
    simply ignored during context construction and read separately by
    _load_ground_truth() below.
    """
    tables = _DatasetTables(dataset_dir)
    tables.messages = _read_csv(dataset_dir, filename)
    records = tables.messages.to_dict(orient="records")
    return [_build_message_context(tables, row) for row in records]


def _load_ground_truth(
    dataset_dir: str, filename: str = SAMPLE_MESSAGES_CSV
) -> dict[str, dict[str, str]]:
    df = _read_csv(dataset_dir, filename)
    return {
        row["message_id"]: {"action": row["action"], "message_type": row["message_type"]}
        for row in df.to_dict(orient="records")
    }


def run_evaluation(dataset_dir: str = "dataset") -> list[dict]:
    """Run the full pipeline on every sample_messages.csv row and return a
    list of per-message result dicts comparing the prediction against the
    known ground truth.
    """
    history_df = pd.read_csv(os.path.join(dataset_dir, MESSAGE_HISTORY_CSV))
    events_df = pd.read_csv(os.path.join(dataset_dir, MESSAGE_EVENTS_CSV))

    contexts = load_sample_contexts(dataset_dir)
    ground_truth = _load_ground_truth(dataset_dir)

    results: list[dict] = []
    for context in tqdm(contexts, desc="Evaluating sample messages", unit="msg"):
        expected = ground_truth.get(context.message_id)
        if expected is None:
            logger.warning("No ground truth row found for %s; skipping.", context.message_id)
            continue

        prepare_context(context, history_df, events_df, dataset_dir)
        decision = route_message(context)

        results.append(
            {
                "message_id": context.message_id,
                "predicted_action": decision.action,
                "expected_action": expected["action"],
                "action_correct": decision.action == expected["action"],
                "predicted_message_type": decision.message_type,
                "expected_message_type": expected["message_type"],
                "message_type_correct": decision.message_type == expected["message_type"],
                "confidence": decision.confidence,
                "reason": decision.reason,
            }
        )
    return results


def print_report(results: list[dict]) -> None:
    total = len(results)
    if total == 0:
        print("No evaluated rows; nothing to report.")
        return

    action_correct = sum(r["action_correct"] for r in results)
    message_type_correct = sum(r["message_type_correct"] for r in results)
    both_correct = sum(r["action_correct"] and r["message_type_correct"] for r in results)

    print(f"\n{'=' * 64}")
    print("EVALUATION REPORT - dataset/sample_messages.csv")
    print(f"{'=' * 64}")
    print(f"Messages evaluated:     {total}")
    print(f"Action Accuracy:        {action_correct}/{total} ({100 * action_correct / total:.1f}%)")
    print(f"Message Type Accuracy:  {message_type_correct}/{total} ({100 * message_type_correct / total:.1f}%)")
    print(f"Both Correct:           {both_correct}/{total} ({100 * both_correct / total:.1f}%)")

    mismatches = [r for r in results if not (r["action_correct"] and r["message_type_correct"])]
    if mismatches:
        print(f"\n--- Mismatches ({len(mismatches)}) ---")
        for r in mismatches:
            print(
                f"  [{r['message_id']}] "
                f"action: predicted={r['predicted_action']!r} expected={r['expected_action']!r} | "
                f"message_type: predicted={r['predicted_message_type']!r} expected={r['expected_message_type']!r} "
                f"(confidence={r['confidence']})"
            )
            print(f"      reason: {r['reason']!r}")
    print(f"{'=' * 64}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the routing pipeline against dataset/sample_messages.csv."
    )
    parser.add_argument("--dataset-dir", default="dataset", help="Directory containing the input CSVs.")
    parser.add_argument("--verbose", action="store_true", help="Enable INFO-level pipeline logging.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    results = run_evaluation(args.dataset_dir)
    print_report(results)


if __name__ == "__main__":
    main()

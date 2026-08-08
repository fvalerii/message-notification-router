"""Non-classification quality checks: evidence validity, confidence
calibration, and reason consistency.

Action/message_type correctness is already covered by code.evaluation.main.
This module checks the three remaining rubric criteria from
problem_statement.md's "Evaluation" section that aren't a simple
predicted-vs-expected comparison:

1. Evidence validity — every non-"none" evidence_message_ids entry must be
   an ID that was actually retrieved as a candidate for that message (never
   invented, never a calibration-example ID from the prompt).
2. Confidence calibration — confidence must be in [0.0, 1.0], and messages
   the router itself treats as ambiguous (message_type="unknown", or no
   evidence at all for a first-contact sender) should not carry a
   maxed-out 1.0 confidence.
3. Reason consistency — a manual-read sample of mute/notify reasons to spot
   check tone, brevity, and logical fit with the decision.

Works against any CSV with messages.csv's input columns — sample_messages.csv
(default) or the real dataset/messages.csv target set — since these checks
are about internal consistency (evidence/confidence/reason), not
predicted-vs-ground-truth accuracy, so no label column is required.

Run from the project root:
    python -m code.evaluation.quality_checks
    python -m code.evaluation.quality_checks --filename messages.csv
"""

from __future__ import annotations

import argparse
import os
import re

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

from code.evaluation.main import SAMPLE_MESSAGES_CSV, load_sample_contexts
from code.main import prepare_context
from code.routing.engine import route_message

# The 13 curated sample_messages.csv rows used to build the SYSTEM_PROMPT's
# fixed calibration examples (code/routing/prompts.py). None of these are
# message_history.csv IDs, so they could never legitimately be cited in
# evidence_message_ids — this is the concrete "calibration exemplar ID" set
# to defend against.
CALIBRATION_EXEMPLAR_SAMPLE_IDS = {
    "sample_msg_001",
    "sample_msg_002",
    "sample_msg_004",
    "sample_msg_005",
    "sample_msg_009",
    "sample_msg_011",
    "sample_msg_012",
    "sample_msg_013",
    "sample_msg_014",
    "sample_msg_019",
    "sample_msg_020",
    "sample_msg_043",
    "sample_msg_049",
}

EVIDENCE_ID_PATTERN = re.compile(r"^message_\d+$")


def _parse_ids(raw: str) -> list[str]:
    if not raw or raw.strip().lower() == "none":
        return []
    return [part.strip() for part in re.split(r"[;,]", raw) if part.strip()]


def run_checks(dataset_dir: str = "dataset", filename: str = SAMPLE_MESSAGES_CSV) -> list[dict]:
    """Run the full pipeline on the given message CSV and collect everything
    needed for the three checks: the final decision, the actual retrieved
    candidate IDs, and enough context to judge ambiguity.
    """
    history_df = pd.read_csv(os.path.join(dataset_dir, "message_history.csv"))
    events_df = pd.read_csv(os.path.join(dataset_dir, "message_events.csv"))
    contexts = load_sample_contexts(dataset_dir, filename)

    rows: list[dict] = []
    for context in tqdm(contexts, desc="Running pipeline for quality checks", unit="msg"):
        prepare_context(context, history_df, events_df, dataset_dir)
        decision = route_message(context)

        rows.append(
            {
                "message_id": context.message_id,
                "action": decision.action,
                "message_type": decision.message_type,
                "reason": decision.reason,
                "confidence": decision.confidence,
                "evidence_message_ids": decision.evidence_message_ids,
                "valid_candidate_ids": {c.message_id for c in context.evidence_candidates},
                "conversation_type": context.conversation_type,
                "sender_user_id": context.sender_user_id,
                "has_evidence_candidates": bool(context.evidence_candidates),
                "hard_routed": context.guardrail_flags.hard_routed,
            }
        )
    return rows


def check_evidence_validity(rows: list[dict]) -> dict:
    invalid: list[dict] = []
    exemplar_leaks: list[dict] = []
    malformed: list[dict] = []
    total_cited = 0

    for row in rows:
        cited = _parse_ids(row["evidence_message_ids"])
        total_cited += len(cited)
        for cited_id in cited:
            if cited_id in CALIBRATION_EXEMPLAR_SAMPLE_IDS:
                exemplar_leaks.append({"message_id": row["message_id"], "cited_id": cited_id})
            if not EVIDENCE_ID_PATTERN.match(cited_id):
                malformed.append({"message_id": row["message_id"], "cited_id": cited_id})
            elif cited_id not in row["valid_candidate_ids"]:
                invalid.append(
                    {
                        "message_id": row["message_id"],
                        "cited_id": cited_id,
                        "valid_candidate_ids": sorted(row["valid_candidate_ids"]),
                    }
                )

    return {
        "total_rows": len(rows),
        "total_cited_ids": total_cited,
        "rows_with_evidence": sum(1 for r in rows if _parse_ids(r["evidence_message_ids"])),
        "invalid_ids": invalid,
        "exemplar_leaks": exemplar_leaks,
        "malformed_ids": malformed,
        "passed": not invalid and not exemplar_leaks and not malformed,
    }


def check_confidence_calibration(rows: list[dict]) -> dict:
    out_of_bounds = [
        r for r in rows if not (0.0 <= r["confidence"] <= 1.0)
    ]

    ambiguous_but_maxed: list[dict] = []
    for row in rows:
        is_unfamiliar_sender = (
            row["conversation_type"] == "personal"
            and not row["has_evidence_candidates"]
        )
        is_ambiguous = (row["message_type"] == "unknown") or (
            is_unfamiliar_sender and not row["hard_routed"]
        )
        if is_ambiguous and row["confidence"] >= 1.0:
            ambiguous_but_maxed.append(
                {
                    "message_id": row["message_id"],
                    "message_type": row["message_type"],
                    "confidence": row["confidence"],
                    "reason_signal": "unknown_type" if row["message_type"] == "unknown" else "unfamiliar_sender_no_evidence",
                }
            )

    return {
        "total_rows": len(rows),
        "out_of_bounds": out_of_bounds,
        "ambiguous_but_maxed": ambiguous_but_maxed,
        "confidence_min": min(r["confidence"] for r in rows),
        "confidence_max": max(r["confidence"] for r in rows),
        "confidence_mean": round(sum(r["confidence"] for r in rows) / len(rows), 3),
        "passed": not out_of_bounds and not ambiguous_but_maxed,
    }


def print_reason_consistency_sample(rows: list[dict], n: int = 5) -> None:
    mute_rows = [r for r in rows if r["action"] == "mute"][:n]
    notify_rows = [r for r in rows if r["action"] == "notify"][:n]

    print(f"\n--- {len(mute_rows)} sample MUTE reasons ---")
    for r in mute_rows:
        print(f"  [{r['message_id']}] type={r['message_type']} conf={r['confidence']}")
        print(f"      reason: {r['reason']!r}")

    print(f"\n--- {len(notify_rows)} sample NOTIFY reasons ---")
    for r in notify_rows:
        print(f"  [{r['message_id']}] type={r['message_type']} conf={r['confidence']}")
        print(f"      reason: {r['reason']!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Non-classification quality checks on a message CSV.")
    parser.add_argument("--dataset-dir", default="dataset")
    parser.add_argument(
        "--filename",
        default=SAMPLE_MESSAGES_CSV,
        help="CSV under dataset-dir to check (e.g. sample_messages.csv or messages.csv).",
    )
    args = parser.parse_args()

    rows = run_checks(args.dataset_dir, args.filename)

    evidence_report = check_evidence_validity(rows)
    confidence_report = check_confidence_calibration(rows)

    print(f"\n{'=' * 70}")
    print(f"QUALITY CHECKS ({args.filename}) — evidence validity, confidence calibration, reason")
    print(f"{'=' * 70}")

    print("\n[1] Evidence ID check")
    print(f"    rows: {evidence_report['total_rows']} | rows with evidence: {evidence_report['rows_with_evidence']} "
          f"| total cited IDs: {evidence_report['total_cited_ids']}")
    print(f"    invalid (not in retrieved candidates): {len(evidence_report['invalid_ids'])}")
    for item in evidence_report["invalid_ids"]:
        print(f"      [{item['message_id']}] cited {item['cited_id']!r} not in {item['valid_candidate_ids']}")
    print(f"    calibration-exemplar ID leaks: {len(evidence_report['exemplar_leaks'])}")
    for item in evidence_report["exemplar_leaks"]:
        print(f"      [{item['message_id']}] cited exemplar ID {item['cited_id']!r}")
    print(f"    malformed IDs (not message_<digits>): {len(evidence_report['malformed_ids'])}")
    for item in evidence_report["malformed_ids"]:
        print(f"      [{item['message_id']}] cited {item['cited_id']!r}")
    print(f"    RESULT: {'PASS' if evidence_report['passed'] else 'FAIL'}")

    print("\n[2] Confidence calibration check")
    print(f"    range: min={confidence_report['confidence_min']} max={confidence_report['confidence_max']} "
          f"mean={confidence_report['confidence_mean']}")
    print(f"    out-of-bounds rows: {len(confidence_report['out_of_bounds'])}")
    print(f"    ambiguous-but-confidence>=1.0 rows: {len(confidence_report['ambiguous_but_maxed'])}")
    for item in confidence_report["ambiguous_but_maxed"]:
        print(f"      [{item['message_id']}] type={item['message_type']} conf={item['confidence']} "
              f"signal={item['reason_signal']}")
    print(f"    RESULT: {'PASS' if confidence_report['passed'] else 'FAIL'}")

    print("\n[3] Reason consistency check (manual read)")
    print_reason_consistency_sample(rows, n=5)

    print(f"\n{'=' * 70}")
    all_passed = evidence_report["passed"] and confidence_report["passed"]
    print(f"OVERALL (checks 1-2, automatable): {'PASS' if all_passed else 'FAIL'}")
    print("Check 3 (reason consistency) requires manual read of the printed samples above.")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()

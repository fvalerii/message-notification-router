"""Final CSV output writer.

Converts the pipeline's list of calibrated RoutingDecision objects into
dataset/output.csv, enforcing the exact column order and evidence-ID
format required by problem_statement.md's submission contract.

Note the one deliberate format translation that happens only here: every
other stage of this pipeline (RoutingDecision, calibrate_decision) works
with evidence_message_ids as a comma-separated string, matching how the
LLM is asked to produce it. problem_statement.md's actual CSV spec
requires semicolons ("evidence_message_ids: semicolon-separated
historical message IDs"), matching dataset/sample_messages.csv (e.g.
"message_0013;message_0014"). This module is the single place that
converts between the two, so the rest of the pipeline never has to think
about the submission file format.
"""

from __future__ import annotations

import csv
import os
import re
from typing import Any, Iterable, Union

from code.data.schemas import RoutingDecision

OUTPUT_COLUMNS = [
    "message_id",
    "action",
    "message_type",
    "reason",
    "confidence",
    "evidence_message_ids",
]
DEFAULT_OUTPUT_PATH = "dataset/output.csv"


def _format_evidence_ids(value: Union[str, Iterable[str], None]) -> str:
    """Normalize evidence_message_ids to the submission format required by
    problem_statement.md: a semicolon-separated list of IDs, or the exact
    string "none".

    Accepts the internal comma-separated string convention used
    everywhere else in this pipeline, a string that already uses
    semicolons, or a plain list/tuple of IDs, so this stays correct even
    if that internal convention changes upstream. Blank/whitespace-only
    entries and a missing value both collapse to "none".
    """
    if value is None:
        return "none"

    if isinstance(value, str):
        text = value.strip()
        if not text or text.lower() == "none":
            return "none"
        ids = [part.strip() for part in re.split(r"[;,]", text) if part.strip()]
    else:
        ids = [str(item).strip() for item in value if str(item).strip()]

    return ";".join(ids) if ids else "none"


def _decision_to_row(decision: RoutingDecision) -> dict[str, Any]:
    return {
        "message_id": decision.message_id,
        "action": decision.action,
        "message_type": decision.message_type,
        "reason": decision.reason,
        "confidence": decision.confidence,
        "evidence_message_ids": _format_evidence_ids(decision.evidence_message_ids),
    }


def write_predictions(
    decisions: list[RoutingDecision], output_path: str = DEFAULT_OUTPUT_PATH
) -> None:
    """Write decisions to output_path as CSV with the exact required
    column order (message_id, action, message_type, reason, confidence,
    evidence_message_ids). Creates the parent directory if needed and
    overwrites any existing file at output_path.
    """
    parent_dir = os.path.dirname(output_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for decision in decisions:
            writer.writerow(_decision_to_row(decision))


if __name__ == "__main__":
    import tempfile

    import pandas as pd

    sample_decisions = [
        RoutingDecision(
            message_id="msg_a",
            action="notify",
            message_type="urgent",
            reason="Direct emergency request from a close contact.",
            confidence=0.93,
            evidence_message_ids="message_0001,message_0002",  # internal comma format
        ),
        RoutingDecision(
            message_id="msg_b",
            action="mute",
            message_type="scam",
            reason="Failed security guardrails: high-confidence scam pattern.",
            confidence=1.0,
            evidence_message_ids="none",
        ),
        RoutingDecision(
            message_id="msg_c",
            action="digest",
            message_type="promotion",
            reason='A promotion, with a "quoted" phrase and, a comma in it.',
            confidence=0.7,
            evidence_message_ids="message_0034;message_0052",  # already semicolon format
        ),
    ]

    with tempfile.TemporaryDirectory() as tmp_dir:
        out_path = os.path.join(tmp_dir, "nested", "output.csv")
        write_predictions(sample_decisions, out_path)
        assert os.path.exists(out_path), "write_predictions should create parent dirs as needed"

        df = pd.read_csv(out_path)
        print(df.to_string(index=False))

        assert list(df.columns) == OUTPUT_COLUMNS
        assert df.loc[df.message_id == "msg_a", "evidence_message_ids"].item() == "message_0001;message_0002"
        assert df.loc[df.message_id == "msg_b", "evidence_message_ids"].item() == "none"
        assert df.loc[df.message_id == "msg_c", "evidence_message_ids"].item() == "message_0034;message_0052"
        # Embedded quote/comma in `reason` must survive a real CSV round-trip.
        assert df.loc[df.message_id == "msg_c", "reason"].item() == sample_decisions[2].reason
        print("\nAll writer.py smoke tests passed (column order, comma->semicolon conversion, "
              "already-semicolon passthrough, none handling, and CSV quoting all verified).")

    # Formatting edge cases that don't need a real file
    assert _format_evidence_ids(None) == "none"
    assert _format_evidence_ids("") == "none"
    assert _format_evidence_ids("   ") == "none"
    assert _format_evidence_ids("none") == "none"
    assert _format_evidence_ids(["message_0001", "message_0002"]) == "message_0001;message_0002"
    assert _format_evidence_ids([]) == "none"
    print("Edge-case formatting checks passed.")

"""Pipeline orchestrator: the full Message Notification Router.

For every message in dataset/messages.csv, runs the pipeline stages built
in the previous steps, strictly in this order:

1. Deterministic security guardrails (code.guardrails.security) — fast,
   free, and evaluated first so a confirmed scam or severe prompt-
   injection attempt can hard-route straight to `mute` without spending
   any media-processing or LLM budget on it at all.
2. Media pre-processing (code.media.image / code.media.audio) — only for
   messages the guardrail layer did NOT already hard-route, since a
   hard-routed message's decision no longer depends on its media content.
   When a voice note yields a transcript, the guardrails are re-evaluated
   over it, so a spoken scam or injection attempt gets the same hard-route
   treatment as a typed one instead of bypassing the deterministic layer.
3. Historical evidence retrieval (code.evidence.retrieval) against
   message_history.csv / message_events.csv.
4. LLM routing (code.routing.engine) — Claude Sonnet 5 with structured
   output, calibrated by code.routing.calibration.

Each message is processed inside its own error boundary: an unexpected
exception in stages 1-3 degrades that single message to a conservative
low-confidence fallback decision instead of aborting the entire run
(stage 4 already degrades internally the same way).

Every decision is collected and written to dataset/output.csv (or
--output-path) via code.output.writer, in the exact column order and
format required by problem_statement.md.

Run from the project root:
    python -m code.main
    python -m code.main --limit 10              # quick smoke run, first 10 messages
    python -m code.main --output-path out.csv
    python -m code.main --verbose                # show per-stage INFO logs

Required environment variables (read from the process environment only,
never hardcoded): ANTHROPIC_API_KEY for routing, and one of
GEMINI_API_KEY / GOOGLE_API_KEY for voice-note analysis. Missing keys do
not crash the run — image-only and text-only messages are unaffected,
voice notes are routed without a transcript, and any message that needed
a Claude call but couldn't get one falls back to a low-confidence
`digest`/`unknown` decision (see code.routing.engine).

A local `.env` file (in the repo root, git-ignored, never committed) is
loaded first thing below so ANTHROPIC_API_KEY / GEMINI_API_KEY /
GOOGLE_API_KEY can be set there instead of the shell environment. This
must happen before code.routing.engine or code.media.audio ever
construct their API clients, since both read the key from os.environ at
call time.
"""

from __future__ import annotations

import argparse
import logging
import os

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

# Load .env (repo root, git-ignored) before importing any pipeline module
# that constructs an API client (code.routing.engine's anthropic.Anthropic,
# code.media.audio's genai.Client), so ANTHROPIC_API_KEY / GEMINI_API_KEY /
# GOOGLE_API_KEY are available in os.environ the first time those modules
# look them up.
load_dotenv()

from code.data.loader import load_dataset
from code.data.schemas import MessageContext, RoutingDecision
from code.evidence.retrieval import retrieve_evidence
from code.guardrails.security import evaluate_security_risk
from code.media.audio import analyze_audio
from code.media.image import process_image
from code.output.writer import DEFAULT_OUTPUT_PATH, write_predictions
from code.routing.engine import route_message

logger = logging.getLogger(__name__)

MESSAGE_HISTORY_CSV = "message_history.csv"
MESSAGE_EVENTS_CSV = "message_events.csv"
IMAGE_MEDIA_TYPE = "image/jpeg"  # process_image() always re-encodes to JPEG


def _process_media_if_needed(context: MessageContext, dataset_dir: str = "dataset") -> None:
    """Populate image_base64 or audio_analysis in place for one message
    that was NOT hard-routed by the guardrail layer. Marks
    guardrail_flags.media_integrity_ok = False whenever media was expected
    (media_type is set) but the file could not be resolved or processed,
    so the routing prompt and confidence calibration both know to
    discount it rather than silently reasoning over missing context.

    context.media_file_path is resolved relative to dataset_dir (see
    MessageContext's field docstring in code/data/schemas.py), so it must
    be joined with dataset_dir here before touching the filesystem.
    """
    if context.media_type == "image":
        if context.media_file_path:
            full_path = os.path.join(dataset_dir, context.media_file_path)
            encoded = process_image(full_path)
            if encoded is not None:
                context.image_base64 = encoded
                context.image_media_type = IMAGE_MEDIA_TYPE
            else:
                context.guardrail_flags.media_integrity_ok = False
        else:
            logger.warning("Message %s has media_type=image but no resolvable file path.", context.message_id)
            context.guardrail_flags.media_integrity_ok = False

    elif context.media_type == "voice":
        if context.media_file_path:
            full_path = os.path.join(dataset_dir, context.media_file_path)
            analysis = analyze_audio(full_path)
            if analysis is not None:
                context.audio_analysis = analysis
            else:
                context.guardrail_flags.media_integrity_ok = False
        else:
            logger.warning("Message %s has media_type=voice but no resolvable file path.", context.message_id)
            context.guardrail_flags.media_integrity_ok = False


def prepare_context(
    context: MessageContext,
    history_df: pd.DataFrame,
    events_df: pd.DataFrame,
    dataset_dir: str = "dataset",
) -> None:
    """Run pipeline stages 1-3 in place on one message: guardrails, media
    pre-processing (with a guardrail re-run once a voice transcript
    exists), and evidence retrieval. Shared by the production orchestrator
    and both evaluation harnesses so their stage wiring can never drift.
    """
    # Stage 1: deterministic guardrails, always first.
    context.guardrail_flags = evaluate_security_risk(context)

    # Stage 2: media pre-processing, skipped entirely for hard-routed
    # messages since their action is already decided.
    if not context.guardrail_flags.hard_routed:
        _process_media_if_needed(context, dataset_dir)

        # A voice transcript is new scannable text the stage-1 pass never
        # saw (guardrails run before media by design, to save budget on
        # already-decided messages). Re-evaluate so a spoken scam or
        # injection attempt can hard-route exactly like a typed one —
        # otherwise voice content would bypass the deterministic layer
        # entirely. media_integrity_ok is preserved because the fresh
        # evaluation resets it to True while stage 2 may have set it False.
        if context.audio_analysis is not None:
            media_ok = context.guardrail_flags.media_integrity_ok
            context.guardrail_flags = evaluate_security_risk(context)
            context.guardrail_flags.media_integrity_ok = media_ok

    # Stage 3: historical evidence retrieval.
    context.evidence_candidates = retrieve_evidence(context, history_df, events_df)


def _pipeline_error_fallback(context: MessageContext) -> RoutingDecision:
    """Conservative decision for a message whose pipeline stages raised an
    unexpected exception — mirrors the routing engine's own API-failure
    fallback so one bad message degrades instead of aborting the run.
    """
    return RoutingDecision(
        message_id=context.message_id,
        action="digest",
        message_type="unknown",
        reason="Pipeline error while processing this message; conservative fallback decision",
        confidence=0.1,
        evidence_message_ids="none",
    )


def run_pipeline(dataset_dir: str = "dataset", limit: int | None = None) -> list[RoutingDecision]:
    """Run the full 4-stage pipeline over every message in dataset_dir and
    return the resulting RoutingDecision list, in input order.
    """
    history_df = pd.read_csv(os.path.join(dataset_dir, MESSAGE_HISTORY_CSV))
    events_df = pd.read_csv(os.path.join(dataset_dir, MESSAGE_EVENTS_CSV))

    contexts = load_dataset(dataset_dir)
    if limit is not None:
        contexts = contexts[:limit]

    decisions: list[RoutingDecision] = []
    for context in tqdm(contexts, desc="Routing messages", unit="msg"):
        try:
            # Stages 1-3 (guardrails, media, retrieval).
            prepare_context(context, history_df, events_df, dataset_dir)
            # Stage 4: LLM routing (or the guardrail fast path inside it);
            # degrades internally on API failures, so an exception escaping
            # it here is just as unexpected as one from stages 1-3.
            decision = route_message(context)
        except Exception as exc:  # noqa: BLE001 - one bad message must not abort the run
            logger.error("Pipeline failed for %s: %s", context.message_id, exc)
            decision = _pipeline_error_fallback(context)
        decisions.append(decision)

    return decisions


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the Message Notification Router pipeline end to end."
    )
    parser.add_argument(
        "--dataset-dir", default="dataset", help="Directory containing the input CSVs (default: dataset)."
    )
    parser.add_argument(
        "--output-path",
        default=DEFAULT_OUTPUT_PATH,
        help=f"Where to write predictions (default: {DEFAULT_OUTPUT_PATH}).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N messages, for a quick end-to-end smoke run.",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable INFO-level pipeline logging.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    decisions = run_pipeline(args.dataset_dir, args.limit)
    write_predictions(decisions, args.output_path)

    print(f"Wrote {len(decisions)} predictions to {args.output_path}")


if __name__ == "__main__":
    main()

"""Post-processing and confidence calibration for LLM routing decisions.

Two independent things happen here, both defensively — the LLM's raw
output should never be trusted blindly:

1. evidence_message_ids is filtered against the actual retrieval
   candidates the model was shown, so a hallucinated or malformed ID can
   never make it into the final output.
2. confidence is adjusted down when the decision was made with degraded
   or missing information (an unclear/unavailable voice-note transcript,
   or a missing image), since the model cannot reliably know how much
   that hurt its own certainty.
"""

from __future__ import annotations

import re

from code.data.schemas import MessageContext, RoutingDecision

CONFIDENCE_FLOOR = 0.05
CONFIDENCE_CEILING = 1.0

# Applied when the message has a voice note but no transcript was ever
# produced (audio pipeline failed) — the decision was made almost blind.
MISSING_MEDIA_PENALTY = 0.25

# Applied when a transcript exists but is too short/thin to be a reliable
# signal (e.g. a couple of words from a noisy or clipped recording).
UNCERTAIN_TRANSCRIPT_PENALTY = 0.15
MIN_RELIABLE_TRANSCRIPT_CHARS = 8

# Applied when the message has an image but it could not be processed.
MISSING_IMAGE_PENALTY = 0.25


def _parse_evidence_ids(raw: str) -> list[str]:
    if not raw or raw.strip().lower() == "none":
        return []
    # The schema asks the model for a comma-separated list, but tolerate a
    # semicolon (the CSV submission format, and an easy model slip) too.
    parts = re.split(r"[;,]", raw)
    return [part.strip() for part in parts if part.strip()]


def _validate_evidence(decision: RoutingDecision, context: MessageContext) -> str:
    valid_ids = {candidate.message_id for candidate in context.evidence_candidates}
    cited_ids = _parse_evidence_ids(decision.evidence_message_ids)
    filtered_ids = [message_id for message_id in cited_ids if message_id in valid_ids]
    # De-duplicate while preserving the model's cited order.
    seen: set[str] = set()
    deduped = [i for i in filtered_ids if not (i in seen or seen.add(i))]
    return ",".join(deduped) if deduped else "none"


def _media_confidence_penalty(context: MessageContext) -> float:
    """Confidence penalty for uncertain or unavailable media analysis. The
    LLM sees the same "unavailable"/short-transcript context in its prompt,
    but cannot be relied on to discount its own stated confidence enough on
    its own, so this is enforced deterministically afterward.
    """
    penalty = 0.0

    if context.media_type == "voice":
        if context.audio_analysis is None:
            penalty += MISSING_MEDIA_PENALTY
        else:
            transcript = (context.audio_analysis.transcript or "").strip()
            if len(transcript) < MIN_RELIABLE_TRANSCRIPT_CHARS:
                penalty += UNCERTAIN_TRANSCRIPT_PENALTY

    if context.media_type == "image" and not context.image_base64:
        penalty += MISSING_IMAGE_PENALTY

    if not context.guardrail_flags.media_integrity_ok:
        penalty += UNCERTAIN_TRANSCRIPT_PENALTY

    return penalty


def calibrate_decision(decision: RoutingDecision, context: MessageContext) -> RoutingDecision:
    """Return a new, validated RoutingDecision with confidence adjusted for
    media-quality uncertainty and evidence_message_ids restricted to the
    IDs actually retrieved for this message.
    """
    penalty = _media_confidence_penalty(context)
    adjusted_confidence = max(
        CONFIDENCE_FLOOR, min(CONFIDENCE_CEILING, decision.confidence - penalty)
    )

    validated_evidence = _validate_evidence(decision, context)

    updated_fields = decision.model_dump()
    updated_fields["confidence"] = round(adjusted_confidence, 3)
    updated_fields["evidence_message_ids"] = validated_evidence
    # model constructor re-runs full validation (ge/le bounds, literals, etc.)
    return RoutingDecision(**updated_fields)

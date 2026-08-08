"""Evidence retrieval engine.

For a given incoming message, finds and ranks historical messages sent to
the same user (message_history.csv) that are useful evidence for the
routing decision, and joins in how the user actually reacted to them
(message_events.csv: opened, replied, dismissed, muted-after, reported).

The ranked candidate list returned here is later used both as context for
the LLM routing prompt and as a whitelist, so the LLM can only cite
evidence_message_ids that are real historical messages this user actually
received.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Optional

import pandas as pd
from rapidfuzz import fuzz

from code.data.schemas import EvidenceCandidate, MessageContext

DEFAULT_TOP_K = 5

# Below this combined score, a candidate isn't good enough evidence to cite;
# if nothing clears it, retrieve_evidence returns [] so the router knows to
# emit "none" for evidence_message_ids.
EVIDENCE_SCORE_THRESHOLD = 0.40

# Small, additive bonuses on top of the text-similarity score. Kept modest
# so text similarity remains the dominant signal — a shared sender/group
# alone should not manufacture "evidence" out of an unrelated message.
_RELATIONSHIP_BONUS = 0.10
_BEHAVIORAL_SIGNAL_BONUS = 0.05  # any recorded reaction at all
_STRONG_BEHAVIORAL_BONUS = 0.05  # additional, for a muted/reported outcome specifically

# match_type buckets, based on raw text_similarity (before bonuses).
_EXACT_DUPLICATE_THRESHOLD = 0.90
_NEAR_DUPLICATE_THRESHOLD = 0.65


def _normalize(text: str) -> str:
    """Normalize text for fuzzy comparison.

    Applies Unicode NFKD normalization and transliterates to ASCII before
    the punctuation cleanup, so accented non-English words (e.g. "café",
    "Votre passeport a été trouvé") fold to their base ASCII letters
    ("cafe", "votre passeport a ete trouve") instead of the accented
    characters being silently stripped out by the a-z0-9-only regex below
    — which would otherwise break token boundary matching and similarity
    scoring against otherwise-identical text.
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _query_text(context: MessageContext) -> str:
    """Text to match against history: the message text itself, or the
    voice-note transcript when there is no text (media_type == 'voice').
    """
    if context.message_text and context.message_text.strip():
        return context.message_text
    if context.audio_analysis is not None and context.audio_analysis.transcript:
        return context.audio_analysis.transcript
    return ""


def _same_relationship(context: MessageContext, candidate_row: dict) -> bool:
    """Same sender, group, or business as the incoming message — a prior
    on relevance even before looking at the text.
    """
    if context.conversation_type == "group" and context.group_context is not None:
        if candidate_row.get("group_id") == context.group_context.group_id:
            return True
    if context.conversation_type == "business" and context.business_context is not None:
        if candidate_row.get("business_id") == context.business_context.business_id:
            return True
    if context.sender_user_id and candidate_row.get("sender_user_id") == context.sender_user_id:
        return True
    return False


def _clean_text(value: object) -> Optional[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value)
    return text if text.strip() else None


def _bool_or_none(value: object) -> Optional[bool]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    return bool(int(value))


def retrieve_evidence(
    context: MessageContext,
    history_df: pd.DataFrame,
    events_df: pd.DataFrame,
    top_k: int = DEFAULT_TOP_K,
) -> list[EvidenceCandidate]:
    """Return up to top_k ranked EvidenceCandidate objects for context, or
    an empty list if no historical message clears EVIDENCE_SCORE_THRESHOLD.
    """
    normalized_query = _normalize(_query_text(context))
    if not normalized_query:
        return []

    same_user_history = history_df[history_df["user_id"] == context.user_id]
    if same_user_history.empty:
        return []

    same_user_events = events_df[events_df["user_id"] == context.user_id]
    events_by_message_id = same_user_events.set_index("message_id").to_dict(orient="index")

    scored: list[tuple[float, float, EvidenceCandidate]] = []
    for row in same_user_history.to_dict(orient="records"):
        candidate_text = _clean_text(row.get("message_text"))
        normalized_candidate = _normalize(candidate_text or "")
        if not normalized_candidate:
            # No text to compare (e.g. a historical image/voice message with
            # no caption) — this simple text-similarity engine can't score it.
            continue

        text_similarity = fuzz.token_sort_ratio(normalized_query, normalized_candidate) / 100.0
        relationship_bonus = _RELATIONSHIP_BONUS if _same_relationship(context, row) else 0.0

        event = events_by_message_id.get(row["message_id"])
        message_opened = message_replied = notification_dismissed = None
        muted_after_message = message_reported = None
        behavioral_bonus = 0.0
        if event is not None:
            message_opened = _bool_or_none(event.get("message_opened"))
            message_replied = _bool_or_none(event.get("message_replied"))
            notification_dismissed = _bool_or_none(event.get("notification_dismissed"))
            muted_after_message = _bool_or_none(event.get("muted_after_message"))
            message_reported = _bool_or_none(event.get("message_reported"))
            # Any recorded reaction makes this a slightly more useful
            # citation than an untested message; a muted/reported outcome
            # is a strong negative signal worth surfacing more prominently.
            behavioral_bonus += _BEHAVIORAL_SIGNAL_BONUS
            if muted_after_message or message_reported:
                behavioral_bonus += _STRONG_BEHAVIORAL_BONUS

        final_score = min(1.0, text_similarity + relationship_bonus + behavioral_bonus)
        if final_score < EVIDENCE_SCORE_THRESHOLD:
            continue

        if text_similarity >= _EXACT_DUPLICATE_THRESHOLD:
            match_type = "exact_duplicate"
        elif text_similarity >= _NEAR_DUPLICATE_THRESHOLD:
            match_type = "near_duplicate"
        elif text_similarity >= EVIDENCE_SCORE_THRESHOLD:
            match_type = "semantic"
        else:
            # Cleared the combined threshold on relationship/behavioral
            # bonuses alone, with weak raw text similarity.
            match_type = "same_relationship"

        candidate = EvidenceCandidate(
            message_id=row["message_id"],
            message_text=candidate_text,
            similarity_score=round(final_score, 3),
            match_type=match_type,
            message_opened=message_opened,
            message_replied=message_replied,
            notification_dismissed=notification_dismissed,
            muted_after_message=muted_after_message,
            message_reported=message_reported,
        )
        scored.append((final_score, text_similarity, candidate))

    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [candidate for _, _, candidate in scored[:top_k]]


if __name__ == "__main__":
    import os

    from code.data.loader import load_dataset

    history_df = pd.read_csv(os.path.join("dataset", "message_history.csv"))
    events_df = pd.read_csv(os.path.join("dataset", "message_events.csv"))

    contexts = load_dataset("dataset")
    context_by_id = {c.message_id: c for c in contexts}

    # sample_msg_013/014 (the classic repeated-forward mute example) live in
    # sample_messages.csv, which load_dataset() does not ingest (only
    # messages.csv is the prediction target). These three messages.csv rows
    # are the real equivalents found by inspection: a reworded forward-chain
    # "good morning" blessing (semantic/near match), an exact-duplicate scam
    # with negative behavioral history, and an exact-duplicate promo with
    # multiple historical hits to rank.
    sample_ids = ["msg_069", "msg_091", "msg_066"]

    for message_id in sample_ids:
        context = context_by_id[message_id]
        candidates = retrieve_evidence(context, history_df, events_df, top_k=5)

        print(f"--- {message_id} (user={context.user_id}, forwarded_count={context.forwarded_count}) ---")
        print(f"  query text: {(_query_text(context) or '<empty>')[:100]!r}")
        if not candidates:
            print("  No evidence candidates cleared the threshold -> router should emit 'none'.")
        for candidate in candidates:
            assert re.match(r"^message_\d+$", candidate.message_id), (
                f"Unexpected evidence ID format: {candidate.message_id}"
            )
            print(
                f"  -> {candidate.message_id} score={candidate.similarity_score} "
                f"type={candidate.match_type} | opened={candidate.message_opened} "
                f"dismissed={candidate.notification_dismissed} "
                f"muted_after={candidate.muted_after_message} "
                f"reported={candidate.message_reported}"
            )
        print()

    # Confirm the threshold actually excludes weak/no matches for at least
    # some real messages, i.e. the engine doesn't just return top_k=5 for
    # everyone regardless of relevance.
    empty_count = sum(
        1 for c in context_by_id.values() if not retrieve_evidence(c, history_df, events_df)
    )
    nonempty_count = len(context_by_id) - empty_count
    print(f"Messages with >=1 evidence candidate: {nonempty_count} / {len(context_by_id)}")
    print(f"Messages with zero evidence candidates (-> 'none'): {empty_count} / {len(context_by_id)}")
    assert empty_count > 0, "Expected at least one message with no useful historical evidence"
    assert nonempty_count > 0, "Expected at least one message with useful historical evidence"

    print("\nAll retrieval.py smoke tests passed.")

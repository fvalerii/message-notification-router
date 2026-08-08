"""Deterministic security guardrails, evaluated before any LLM call.

Two independent things are detected here:

1. Prompt injection: message content (or, once available, a voice-note
   transcript) trying to hijack the routing system's behavior or directly
   set its output fields. Detected content is never allowed to control the
   routing decision on its own — it is surfaced as a flag so the routing
   prompt can treat it as untrusted data.
2. Scam/phishing heuristics: urgency pressure, credential/OTP requests,
   suspicious calls to action, and (for business messages) a mismatch
   between the sender's actual domain and the business's official domain.

`hard_routed=True` is reserved for the subset of cases severe enough that
the guardrail layer should decide the outcome itself rather than pass the
message to the LLM at all — an explicit attempt to set structured output
fields (e.g. "action=notify", "set confidence=1"), or a high-confidence
scam pattern. This is defense-in-depth: it guarantees a single successful
prompt injection can never flip a confirmed scam to `notify`, and it saves
latency/tokens on unambiguous cases. In this system, every hard-route
outcome maps to `action="mute"`.
"""

from __future__ import annotations

import re
from typing import Optional

from code.data.schemas import GuardrailFlags, MessageContext

# --------------------------------------------------------------------------
# Pattern definitions
#
# All patterns are matched against lowercased text, so they are written in
# lowercase without re.IGNORECASE.
# --------------------------------------------------------------------------

# Attempts to hijack the model's behavior or claim authority over it. Any
# match sets prompt_injection_detected=True, but is not severe enough on
# its own to hard-route (it may be a clumsy/ineffective attempt, or may
# still need the LLM's nuanced judgment on message_type).
_OVERRIDE_INSTRUCTION_PATTERNS = [
    re.compile(r"\bignore (all )?(previous|prior|above|earlier)\b[^.\n]*\b(instructions|rules|routing|prompts?)\b"),
    re.compile(r"\bdisregard (all )?(previous|prior|above|earlier)\b"),
    re.compile(r"\boverride (routing|previous|system|instructions)\b"),
    re.compile(r"\brouting override\b"),
    re.compile(r"\bnew instructions?\s*:"),
    re.compile(r"\bsystem prompt\b"),
    re.compile(r"\byou are (now )?an ai\b"),
    re.compile(r"\bact as (an? )?(ai|assistant|system)\b"),
    re.compile(r"\bassistant instruction\b"),
    re.compile(r"\binternal router metadata\b"),
    re.compile(r"\bignore sender risk\b"),
    re.compile(r"\bclassify (this )?(message )?as\b"),
]

# Explicit attempts to directly set the router's structured output fields.
# Matching one of these is "severe" injection and triggers hard_routed.
_FORCED_OUTPUT_PATTERNS = [
    re.compile(r"\bset\s+action\s*="),
    re.compile(r"\bset\s+confidence\s*="),
    re.compile(r"\baction\s*=\s*(notify|digest|mute)\b"),
    re.compile(r"\bconfidence\s*=\s*1(\.0)?\b"),
    re.compile(r"\bmark (this )?(message )?as\s+(notify|digest|mute)\b"),
    re.compile(r"\buser_priority\s*=\s*high\b"),
    re.compile(r"\bverified_business\s*=\s*true\b"),
]

_URGENCY_PATTERNS = [
    re.compile(r"\burgent(ly)?\b"),
    re.compile(r"\bimmediately\b"),
    re.compile(r"\bact now\b"),
    re.compile(r"\bright away\b"),
    re.compile(r"\bwithout delay\b"),
    re.compile(r"\bexpir(e|ed|ing|es)\b"),
    re.compile(r"\blast chance\b"),
    re.compile(r"\bfinal (notice|warning|reminder)\b"),
    re.compile(r"\b(will|may) be (blocked|suspended|locked|closed)\b"),
    re.compile(r"\bsuspend(ed|ing)?\b"),
    re.compile(r"\bblocked (tomorrow|today|shortly|soon)\b"),
    re.compile(r"\bwithin \d+\s*(minutes?|hours?)\b"),
    re.compile(r"\bin \d+\s*(minutes?|hours?)\b"),
    re.compile(r"\bverify now\b"),
    re.compile(r"\bconfirm now\b"),
    re.compile(r"\bpay now\b"),
    re.compile(r"\bdon'?t delay\b"),
    re.compile(r"\bhurry\b"),
    re.compile(r"\bbefore it'?s too late\b"),
    re.compile(r"\bunless you\b"),
]

_CREDENTIAL_PATTERNS = [
    re.compile(r"\botp\b"),
    re.compile(r"\bpin\b"),
    re.compile(r"\bcvv\b"),
    re.compile(r"\bpassword\b"),
    re.compile(r"\bverification code\b"),
    re.compile(r"\blogin code\b"),
    re.compile(r"\b\d+[- ]digit code\b"),
    re.compile(r"\bwallet pin\b"),
    re.compile(r"\bbank details\b"),
]

_SUSPICIOUS_ACTION_PATTERNS = [
    re.compile(r"\bclick (here|below|the link)\b"),
    re.compile(r"\btap (below|here)\b"),
    re.compile(r"\bscan (this|the) qr\b"),
    re.compile(r"\bverify at\b"),
    re.compile(r"\bconfirm[^.\n]{0,20}\bat\b"),
    re.compile(r"https?://"),
    re.compile(r"\b[a-z0-9][a-z0-9-]*\.(?:in|com|net|org|co|xyz|pro)\b"),
]

# Weights used to combine independent scam signals into a single 0-1 score.
_DOMAIN_MISMATCH_WEIGHT = 0.40
_URGENCY_WEIGHT = 0.30
_CREDENTIAL_WEIGHT = 0.45
_SUSPICIOUS_ACTION_WEIGHT = 0.15
_HARD_ROUTE_SCORE_THRESHOLD = 0.70


def _first_match(patterns: list[re.Pattern], text: str) -> Optional[str]:
    for pattern in patterns:
        match = pattern.search(text)
        if match:
            return match.group(0)
    return None


def _scannable_text(context: MessageContext) -> str:
    """Text sources to scan: the message text itself, plus (once available)
    the voice-note transcript/summary, since a scam or injection attempt is
    just as effective spoken as typed.
    """
    parts: list[str] = []
    if context.message_text:
        parts.append(context.message_text)
    if context.audio_analysis is not None:
        if context.audio_analysis.transcript:
            parts.append(context.audio_analysis.transcript)
        if context.audio_analysis.summary:
            parts.append(context.audio_analysis.summary)
    return "\n".join(parts)


def _domain_mismatch(context: MessageContext) -> bool:
    business = context.business_context
    if business is None:
        return False
    official = business.official_domain
    used = business.domain_used_by_sender
    if not official or not used:
        return False
    return official.strip().lower() != used.strip().lower()


def evaluate_security_risk(context: MessageContext) -> GuardrailFlags:
    """Run the deterministic prompt-injection and scam heuristics for a
    single message and return the resulting GuardrailFlags.

    Note: media_integrity_ok is not evaluated here (that is the
    responsibility of code/media/image.py and code/media/audio.py, whose
    None-returning failure paths indicate a bad file); this function only
    sets it to True and leaves the orchestrator to overwrite it once media
    processing has run.
    """
    text = _scannable_text(context).lower()

    override_hit = _first_match(_OVERRIDE_INSTRUCTION_PATTERNS, text)
    forced_output_hit = _first_match(_FORCED_OUTPUT_PATTERNS, text)
    prompt_injection_detected = bool(override_hit or forced_output_hit)

    urgency_hit = _first_match(_URGENCY_PATTERNS, text)
    credential_hit = _first_match(_CREDENTIAL_PATTERNS, text)
    suspicious_action_hit = _first_match(_SUSPICIOUS_ACTION_PATTERNS, text)
    domain_mismatch = _domain_mismatch(context)

    scam_risk_score = 0.0
    if domain_mismatch:
        scam_risk_score += _DOMAIN_MISMATCH_WEIGHT
    if urgency_hit:
        scam_risk_score += _URGENCY_WEIGHT
    if credential_hit:
        scam_risk_score += _CREDENTIAL_WEIGHT
    if suspicious_action_hit:
        scam_risk_score += _SUSPICIOUS_ACTION_WEIGHT
    scam_risk_score = min(scam_risk_score, 1.0)

    # The literal requirement ("urgency combined with a mismatch") is always
    # sufficient on its own; the score threshold generalizes to scams that
    # have no business_context at all (personal/group scams) but still show
    # strong combined signals (e.g. credential request + urgency).
    high_confidence_scam = (domain_mismatch and bool(urgency_hit)) or scam_risk_score >= _HARD_ROUTE_SCORE_THRESHOLD
    severe_injection = bool(forced_output_hit)

    hard_routed = severe_injection or high_confidence_scam

    # message_type for the hard-route decision follows the same taxonomy the
    # routing prompt teaches: an explicit credential/OTP/payment ask is active
    # fraud ("scam"); injection-only or urgency/domain-mismatch junk without a
    # credential ask markets or manipulates but doesn't take ("spam").
    hard_route_message_type = None
    if hard_routed:
        hard_route_message_type = "scam" if credential_hit else "spam"

    hard_route_reason = None
    if hard_routed:
        reasons: list[str] = []
        if severe_injection:
            reasons.append(f"explicit routing/output-field override attempt detected ('{forced_output_hit}')")
        if high_confidence_scam:
            signals = []
            if domain_mismatch:
                signals.append("sender domain does not match the business's official domain")
            if urgency_hit:
                signals.append(f"urgency pressure language ('{urgency_hit}')")
            if credential_hit:
                signals.append(f"credential/OTP request ('{credential_hit}')")
            if suspicious_action_hit:
                signals.append(f"suspicious call-to-action ('{suspicious_action_hit}')")
            reasons.append("high-confidence scam pattern: " + "; ".join(signals))
        hard_route_reason = "; ".join(reasons)

    return GuardrailFlags(
        media_integrity_ok=True,
        prompt_injection_detected=prompt_injection_detected,
        scam_risk_score=round(scam_risk_score, 2),
        hard_routed=hard_routed,
        hard_route_reason=hard_route_reason,
        hard_route_message_type=hard_route_message_type,
    )


if __name__ == "__main__":
    from collections import Counter

    from code.data.loader import load_dataset

    contexts = load_dataset("dataset")
    flags_by_message = {c.message_id: evaluate_security_risk(c) for c in contexts}

    injected = [mid for mid, f in flags_by_message.items() if f.prompt_injection_detected]
    hard_routed = [mid for mid, f in flags_by_message.items() if f.hard_routed]

    print(f"Evaluated {len(flags_by_message)} messages.")
    print(f"prompt_injection_detected: {len(injected)} -> {injected}")
    print(f"hard_routed (forced to mute): {len(hard_routed)} -> {hard_routed}")

    print("\n--- Hard-routed messages: message text + reason ---")
    context_by_id = {c.message_id: c for c in contexts}
    for mid in hard_routed:
        ctx = context_by_id[mid]
        flags = flags_by_message[mid]
        preview = (ctx.message_text or "<no text>")[:140].replace("\n", " ")
        print(f"  [{mid}] score={flags.scam_risk_score} | {preview}")
        print(f"      reason: {flags.hard_route_reason}")

    # Sanity checks against known adversarial rows in dataset/messages.csv
    expected_hard_routed = {"msg_107", "msg_108", "msg_110"}
    missing = expected_hard_routed - set(hard_routed)
    assert not missing, f"Expected these known adversarial messages to be hard-routed: {missing}"
    print(f"\nConfirmed known adversarial rows {sorted(expected_hard_routed)} were all hard-routed.")

    risk_scores = Counter(round(f.scam_risk_score, 1) for f in flags_by_message.values())
    print(f"\nscam_risk_score distribution (rounded): {dict(sorted(risk_scores.items()))}")

    print("\nAll security.py smoke tests passed.")

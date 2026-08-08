"""LLM routing engine: turns a fully joined MessageContext into a final,
calibrated RoutingDecision using Anthropic Claude Sonnet 5.

Two paths:
1. Fast path — if the deterministic guardrail layer already decided this
   message must be muted (context.guardrail_flags.hard_routed), no API
   call is made at all.
2. LLM path — build the routing prompt, call Claude Sonnet 5 with
   structured output enforcing the RoutingDecision schema, retry on
   transient API errors, then run the result through calibrate_decision().
"""

from __future__ import annotations

import logging
import os

from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from code.data.schemas import MessageContext, RoutingDecision
from code.routing.calibration import calibrate_decision
from code.routing.prompts import SYSTEM_PROMPT, build_user_content

logger = logging.getLogger(__name__)

CLAUDE_MODEL = "claude-sonnet-5"
# Sonnet 5 uses adaptive thinking by default, and thinking tokens are drawn
# from the same max_tokens budget as the final JSON answer. 1024 was
# occasionally too tight (a same-request retry at 1024 succeeded cleanly
# every time in testing, confirming this is a budget/variance issue rather
# than a schema/prompt problem), so this is sized with headroom; the
# retryable-error handling above is the actual safety net either way.
MAX_OUTPUT_TOKENS = 2048
ANTHROPIC_API_KEY_ENV_VAR = "ANTHROPIC_API_KEY"

HARD_ROUTE_REASON_FALLBACK = "Failed security guardrails"

# Hard routes are regex/heuristic decisions, not model judgments. They are
# high-precision but not infallible, so confidence is capped below 1.0 —
# an absolute 1.0 on a pattern match would be miscalibrated the moment a
# single false positive exists.
HARD_ROUTE_CONFIDENCE = 0.95


def _hard_route_decision(context: MessageContext) -> RoutingDecision:
    """Deterministic decision for messages the guardrail layer already
    decided must be muted. No LLM call is made — a single successful
    prompt injection must never be able to override this.

    message_type comes from the guardrail layer's own signal analysis
    (credential ask -> scam, otherwise spam); "scam" is only a fallback
    for a hard-routed context that somehow carries no type.
    """
    detail = context.guardrail_flags.hard_route_reason
    reason = f"{HARD_ROUTE_REASON_FALLBACK}: {detail}" if detail else HARD_ROUTE_REASON_FALLBACK
    return RoutingDecision(
        message_id=context.message_id,
        action="mute",
        message_type=context.guardrail_flags.hard_route_message_type or "scam",
        reason=reason,
        confidence=HARD_ROUTE_CONFIDENCE,
        evidence_message_ids="none",
    )


def _fallback_decision(context: MessageContext, reason: str) -> RoutingDecision:
    """Conservative decision used when the LLM call fails even after
    retries, so a transient API/infra problem degrades the pipeline
    instead of crashing it. Low confidence signals this was not a real
    model judgment.
    """
    return RoutingDecision(
        message_id=context.message_id,
        action="digest",
        message_type="unknown",
        reason=reason,
        confidence=0.1,
        evidence_message_ids="none",
    )


class _UnparsableRoutingResponse(RuntimeError):
    """Raised when Claude's response has no usable structured output at
    all (no text content block, typically because adaptive thinking used
    up the token budget before any final answer was emitted). Treated as
    retryable — see _is_retryable_error.
    """


def _is_retryable_error(exc: BaseException) -> bool:
    import anthropic
    import pydantic

    if isinstance(exc, (anthropic.RateLimitError, anthropic.APIConnectionError, anthropic.APITimeoutError)):
        return True
    if isinstance(exc, anthropic.APIStatusError):
        return exc.status_code == 429 or exc.status_code >= 500
    # client.messages.parse() raises pydantic.ValidationError directly when
    # the model's JSON text is malformed/truncated (observed in practice:
    # Sonnet 5's adaptive thinking occasionally leaves too little of the
    # max_tokens budget for the JSON answer to finish cleanly), and Claude
    # can also return a response with no text block at all for the same
    # underlying reason (-> _UnparsableRoutingResponse). Both have been
    # confirmed to succeed on an immediate retry with identical settings,
    # so both are treated as retryable rather than failing straight to the
    # low-confidence fallback decision.
    if isinstance(exc, (pydantic.ValidationError, _UnparsableRoutingResponse)):
        return True
    return False


@retry(
    retry=retry_if_exception(_is_retryable_error),
    wait=wait_exponential(multiplier=2, min=2, max=60),
    stop=stop_after_attempt(5),
    reraise=True,
)
def _call_claude(client, context: MessageContext) -> RoutingDecision:
    response = client.messages.parse(
        model=CLAUDE_MODEL,
        max_tokens=MAX_OUTPUT_TOKENS,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": build_user_content(context)}],
        output_format=RoutingDecision,
    )
    if response.parsed_output is None:
        raise _UnparsableRoutingResponse(
            f"Claude response for {context.message_id} had no parsed structured output "
            f"(stop_reason={response.stop_reason})"
        )
    return response.parsed_output


def _get_client():
    import anthropic

    api_key = os.environ.get(ANTHROPIC_API_KEY_ENV_VAR)
    if not api_key:
        raise RuntimeError(
            f"No Anthropic API key found in {ANTHROPIC_API_KEY_ENV_VAR}; cannot call Claude Sonnet 5."
        )
    return anthropic.Anthropic(api_key=api_key)


def route_message(context: MessageContext) -> RoutingDecision:
    """Return the final, calibrated RoutingDecision for one message."""
    if context.guardrail_flags.hard_routed:
        decision = _hard_route_decision(context)
        return calibrate_decision(decision, context)

    try:
        client = _get_client()
        parsed = _call_claude(client, context)
    except Exception as exc:  # noqa: BLE001 - degrade to a safe fallback, never crash the pipeline
        logger.error("Claude routing call failed for %s: %s", context.message_id, exc)
        decision = _fallback_decision(
            context, reason="Routing model did not return a usable structured decision after retries"
        )
        return calibrate_decision(decision, context)

    # Defensive: the model is asked for this message_id, but never trust it
    # blindly over the ground-truth ID we already know.
    if parsed.message_id != context.message_id:
        parsed = RoutingDecision(**{**parsed.model_dump(), "message_id": context.message_id})

    return calibrate_decision(parsed, context)


if __name__ == "__main__":
    import pandas as pd

    from code.data.loader import load_dataset
    from code.evidence.retrieval import retrieve_evidence
    from code.guardrails.security import evaluate_security_risk

    logging.basicConfig(level=logging.INFO)

    contexts = load_dataset("dataset")
    history_df = pd.read_csv(os.path.join("dataset", "message_history.csv"))
    events_df = pd.read_csv(os.path.join("dataset", "message_events.csv"))

    # --- Fast-path test: no API key or network needed ---
    hard_routed_contexts = []
    for context in contexts:
        flags = evaluate_security_risk(context)
        context = context.model_copy(update={"guardrail_flags": flags})
        if flags.hard_routed:
            hard_routed_contexts.append(context)

    print(f"Found {len(hard_routed_contexts)} hard-routed messages to test the fast path on.")
    for context in hard_routed_contexts[:3]:
        decision = route_message(context)
        print(f"  [{decision.message_id}] action={decision.action} message_type={decision.message_type} "
              f"confidence={decision.confidence} reason={decision.reason!r}")
        assert decision.action == "mute"
        assert decision.evidence_message_ids == "none"
    print("Fast-path (hard-routed) smoke tests passed: no API call was made for these.\n")

    # --- LLM path test: requires ANTHROPIC_API_KEY ---
    if not os.environ.get(ANTHROPIC_API_KEY_ENV_VAR):
        print(
            f"No {ANTHROPIC_API_KEY_ENV_VAR} set in the environment; skipping a live Claude call. "
            "The fast path above was still fully verified without any network access."
        )
    else:
        normal_context = next(c for c in contexts if not evaluate_security_risk(c).hard_routed)
        flags = evaluate_security_risk(normal_context)
        normal_context = normal_context.model_copy(update={"guardrail_flags": flags})
        candidates = retrieve_evidence(normal_context, history_df, events_df)
        normal_context = normal_context.model_copy(update={"evidence_candidates": candidates})

        print(f"Running a live Claude Sonnet 5 call for {normal_context.message_id}...")
        decision = route_message(normal_context)
        print(decision.model_dump_json(indent=2))
        assert decision.message_id == normal_context.message_id
        assert 0.0 <= decision.confidence <= 1.0
        print("\nLive LLM-path smoke test passed.")

    print("\nAll engine.py smoke tests passed.")

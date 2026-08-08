"""Data ingestion and context-join stage.

Loads every CSV in dataset/ and, for each row in messages.csv, assembles a
fully populated MessageContext by joining user, group, business, and media
catalog data. This stage does not touch media content (no transcription or
base64 encoding), does not run guardrails, and does not retrieve evidence —
it only loads and joins structured context so later stages have a single,
validated object to work with.
"""

from __future__ import annotations

import math
import os
from typing import Any, Optional

import pandas as pd

from code.data.schemas import (
    BusinessContext,
    GroupContext,
    MessageContext,
    UserProfile,
)

# Filenames as documented in problem_statement.md / AGENTS.md section 6.1.
MESSAGES_CSV = "messages.csv"
USERS_CSV = "users.csv"
GROUPS_CSV = "groups.csv"
GROUP_MEMBERS_CSV = "group_members.csv"
BUSINESS_ACCOUNTS_CSV = "business_accounts.csv"
USER_BUSINESS_HISTORY_CSV = "user_business_history.csv"
DAILY_NOTIFICATION_SUMMARY_CSV = "daily_notification_summary.csv"
IMAGES_CSV = "images.csv"
VOICE_NOTES_CSV = "voice_notes.csv"


# --------------------------------------------------------------------------
# Value-cleaning helpers
#
# pandas represents missing CSV cells as NaN (a float), including for
# otherwise-string or otherwise-int columns. These helpers normalize NaN to
# None and coerce numpy scalar types to plain Python types before they reach
# Pydantic.
# --------------------------------------------------------------------------


def _is_missing(value: Any) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def _opt_str(value: Any) -> Optional[str]:
    if _is_missing(value):
        return None
    text = str(value).strip()
    return text if text else None


def _opt_int(value: Any) -> Optional[int]:
    if _is_missing(value):
        return None
    return int(value)


def _int(value: Any, default: int = 0) -> int:
    if _is_missing(value):
        return default
    return int(value)


def _opt_bool(value: Any) -> Optional[bool]:
    if _is_missing(value):
        return None
    return bool(int(value))


# --------------------------------------------------------------------------
# CSV loading
# --------------------------------------------------------------------------


def _read_csv(dataset_dir: str, filename: str) -> pd.DataFrame:
    path = os.path.join(dataset_dir, filename)
    return pd.read_csv(path)


class _DatasetTables:
    """Holds every loaded DataFrame plus O(1) lookup indices used for the
    per-message joins. Kept private to this module; callers only interact
    with load_dataset().
    """

    def __init__(self, dataset_dir: str) -> None:
        self.messages = _read_csv(dataset_dir, MESSAGES_CSV)
        self.users = _read_csv(dataset_dir, USERS_CSV)
        self.groups = _read_csv(dataset_dir, GROUPS_CSV)
        self.group_members = _read_csv(dataset_dir, GROUP_MEMBERS_CSV)
        self.business_accounts = _read_csv(dataset_dir, BUSINESS_ACCOUNTS_CSV)
        self.user_business_history = _read_csv(dataset_dir, USER_BUSINESS_HISTORY_CSV)
        self.daily_notification_summary = _read_csv(dataset_dir, DAILY_NOTIFICATION_SUMMARY_CSV)
        self.images = _read_csv(dataset_dir, IMAGES_CSV)
        self.voice_notes = _read_csv(dataset_dir, VOICE_NOTES_CSV)

        self.users_by_id: dict[str, dict[str, Any]] = self.users.set_index(
            "user_id"
        ).to_dict(orient="index")
        self.groups_by_id: dict[str, dict[str, Any]] = self.groups.set_index(
            "group_id"
        ).to_dict(orient="index")
        self.group_member_by_key: dict[tuple[str, str], dict[str, Any]] = (
            self.group_members.set_index(["group_id", "user_id"]).to_dict(orient="index")
        )
        self.business_by_id: dict[str, dict[str, Any]] = self.business_accounts.set_index(
            "business_id"
        ).to_dict(orient="index")
        self.user_business_by_key: dict[tuple[str, str], dict[str, Any]] = (
            self.user_business_history.set_index(["user_id", "business_id"]).to_dict(
                orient="index"
            )
        )
        # Kept for later guardrail/retrieval stages (notification-fatigue signal);
        # not yet attached to MessageContext, which has no field for it.
        self.daily_summary_by_user: dict[str, list[dict[str, Any]]] = {
            user_id: group.to_dict(orient="records")
            for user_id, group in self.daily_notification_summary.groupby("user_id")
        }
        self.image_path_by_id: dict[str, str] = self.images.set_index("image_id")[
            "file_path"
        ].to_dict()
        self.voice_path_by_id: dict[str, str] = self.voice_notes.set_index("voice_note_id")[
            "file_path"
        ].to_dict()


# --------------------------------------------------------------------------
# Per-message join logic
# --------------------------------------------------------------------------


def _build_user_profile(tables: _DatasetTables, user_id: str) -> UserProfile:
    row = tables.users_by_id.get(user_id)
    if row is None:
        raise ValueError(f"users.csv has no row for user_id={user_id!r}")
    return UserProfile(
        user_id=user_id,
        do_not_disturb_window=_opt_str(row.get("do_not_disturb_window")),
        messages_opened_30d=_int(row.get("messages_opened_30d")),
        messages_replied_30d=_int(row.get("messages_replied_30d")),
        notifications_dismissed_30d=_int(row.get("notifications_dismissed_30d")),
        messages_reported_30d=_int(row.get("messages_reported_30d")),
    )


def _build_group_context(
    tables: _DatasetTables, group_id: str, user_id: str
) -> GroupContext:
    group_row = tables.groups_by_id.get(group_id) or {}
    member_row = tables.group_member_by_key.get((group_id, user_id)) or {}
    return GroupContext(
        group_id=group_id,
        group_name=_opt_str(group_row.get("group_name")),
        group_type=_opt_str(group_row.get("group_type")),
        member_count=_opt_int(group_row.get("member_count")),
        admin_count=_opt_int(group_row.get("admin_count")),
        messages_30d=_opt_int(group_row.get("messages_30d")),
        member_role=_opt_str(member_row.get("role")),
        group_muted_by_user=_opt_bool(member_row.get("group_muted_by_user")),
        member_messages_read_30d=_opt_int(member_row.get("messages_read_30d")),
        member_replies_sent_30d=_opt_int(member_row.get("replies_sent_30d")),
        member_notifications_dismissed_30d=_opt_int(
            member_row.get("notifications_dismissed_30d")
        ),
    )


def _build_business_context(
    tables: _DatasetTables, business_id: str, user_id: str
) -> BusinessContext:
    biz_row = tables.business_by_id.get(business_id) or {}
    hist_row = tables.user_business_by_key.get((user_id, business_id)) or {}
    return BusinessContext(
        business_id=business_id,
        display_name=_opt_str(biz_row.get("display_name")),
        brand_name=_opt_str(biz_row.get("brand_name")),
        category=_opt_str(biz_row.get("category")),
        verified=_opt_bool(biz_row.get("verified")),
        official_domain=_opt_str(biz_row.get("official_domain")),
        domain_used_by_sender=_opt_str(biz_row.get("domain_used_by_sender")),
        account_age_days=_opt_int(biz_row.get("account_age_days")),
        domain_used_by_sender_age_days=_opt_int(
            biz_row.get("domain_used_by_sender_age_days")
        ),
        user_reports_30d=_opt_int(biz_row.get("user_reports_30d")),
        # The remaining fields describe this specific user's relationship with
        # the business. No row in user_business_history.csv means the user has
        # no known prior relationship — leave as None rather than 0, since
        # "no relationship" and "zero activity with a known relationship" are
        # different signals for the routing engine.
        allows_promotions=_opt_bool(hist_row.get("allows_promotions")),
        promotions_opted_out_at=_opt_str(hist_row.get("promotions_opted_out_at")),
        activity_count_180d=_opt_int(hist_row.get("activity_count_180d")),
        user_messages_opened_30d=_opt_int(hist_row.get("messages_opened_30d")),
        user_messages_dismissed_30d=_opt_int(hist_row.get("messages_dismissed_30d")),
        user_messages_replied_30d=_opt_int(hist_row.get("messages_replied_30d")),
    )


def _resolve_media_file_path(
    tables: _DatasetTables, media_type: Optional[str], media_id: Optional[str]
) -> Optional[str]:
    if not media_type or not media_id:
        return None
    if media_type == "image":
        return tables.image_path_by_id.get(media_id)
    if media_type == "voice":
        return tables.voice_path_by_id.get(media_id)
    return None


def _build_message_context(tables: _DatasetTables, row: dict[str, Any]) -> MessageContext:
    conversation_type = row["conversation_type"]
    user_id = row["user_id"]
    group_id = _opt_str(row.get("group_id"))
    business_id = _opt_str(row.get("business_id"))
    media_type = _opt_str(row.get("media_type"))
    media_id = _opt_str(row.get("media_id"))

    group_context = (
        _build_group_context(tables, group_id, user_id)
        if conversation_type == "group" and group_id
        else None
    )
    business_context = (
        _build_business_context(tables, business_id, user_id)
        if conversation_type == "business" and business_id
        else None
    )

    return MessageContext(
        message_id=row["message_id"],
        user_id=user_id,
        conversation_type=conversation_type,
        sender_user_id=_opt_str(row.get("sender_user_id")),
        created_at=row["created_at"],
        message_text=_opt_str(row.get("message_text")),
        media_type=media_type,
        media_id=media_id,
        media_file_path=_resolve_media_file_path(tables, media_type, media_id),
        forwarded_count=_int(row.get("forwarded_count")),
        user_profile=_build_user_profile(tables, user_id),
        group_context=group_context,
        business_context=business_context,
        # guardrail_flags, audio_analysis, image_base64, evidence_candidates
        # are left at their schema defaults; populated by later stages.
    )


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------


def load_dataset(dataset_dir: str = "dataset") -> list[MessageContext]:
    """Load every CSV under dataset_dir and return one MessageContext per
    row in messages.csv, in the same order as the CSV.
    """
    tables = _DatasetTables(dataset_dir)
    records = tables.messages.to_dict(orient="records")
    return [_build_message_context(tables, row) for row in records]


# --------------------------------------------------------------------------
# Smoke test
# --------------------------------------------------------------------------

if __name__ == "__main__":
    from collections import Counter

    contexts = load_dataset("dataset")

    # Row count via pandas (not raw line count) since message_text can
    # contain embedded newlines inside quoted CSV fields.
    expected_count = len(_read_csv("dataset", MESSAGES_CSV))
    print(f"Loaded {len(contexts)} MessageContext instances (expected {expected_count}).")
    assert len(contexts) == expected_count, "Loaded count does not match messages.csv row count"

    conversation_counts = Counter(c.conversation_type for c in contexts)
    media_counts = Counter((c.media_type or "text") for c in contexts)
    print("conversation_type breakdown:", dict(conversation_counts))
    print("media_type breakdown:", dict(media_counts))

    group_with_image = next(
        (c for c in contexts if c.conversation_type == "group" and c.media_type == "image"),
        None,
    )
    business_with_voice = next(
        (c for c in contexts if c.conversation_type == "business" and c.media_type == "voice"),
        None,
    )

    print("\n--- Sample: group message with image ---")
    if group_with_image is not None:
        print(group_with_image.model_dump_json(indent=2))
    else:
        print("No group message with an image was found in this dataset.")

    print("\n--- Sample: business message with voice note ---")
    if business_with_voice is not None:
        print(business_with_voice.model_dump_json(indent=2))
    else:
        print("No business message with a voice note was found in this dataset.")

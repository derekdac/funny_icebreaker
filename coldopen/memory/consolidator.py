"""Async background memory consolidation to avoid blocking host interaction."""

from __future__ import annotations

import asyncio
from typing import Any

from coldopen.memory.store import ColdOpenStore
from coldopen.observability.logging import get_logger

_LOGGER = get_logger("coldopen.memory")


async def _consolidate_team_preferences(
    store: ColdOpenStore,
    team_id: str,
    theme: str,
    item_count: int,
) -> dict[str, Any]:
    """Merge newly played deck metadata into the team's long-term preference profile."""
    await asyncio.sleep(0)  # yield control to event loop
    existing = store.get_team_memory(team_id)
    themes_played = list(existing.get("themes_played", []))
    if theme not in themes_played:
        themes_played.append(theme)

    updated = {
        "team_id": team_id,
        "themes_played": themes_played,
        "total_decks_generated": int(existing.get("total_decks_generated", 0)) + 1,
        "total_items_played": int(existing.get("total_items_played", 0)) + item_count,
    }
    store.save_team_memory(team_id, updated)
    _LOGGER.info(
        "async_memory_consolidated",
        extra={"structured_fields": {"team_id": team_id, "theme": theme, "item_count": item_count}},
    )
    return updated


def schedule_memory_consolidation(
    store: ColdOpenStore,
    team_id: str,
    theme: str,
    item_count: int,
) -> asyncio.Task[dict[str, Any]]:
    """Launch memory consolidation as a non-blocking background asyncio task."""
    return asyncio.create_task(
        _consolidate_team_preferences(store, team_id, theme, item_count)
    )

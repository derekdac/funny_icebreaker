"""CLI entry point for ColdOpen."""

from __future__ import annotations

import argparse
import asyncio
import os
import uuid

from coldopen.config.secrets import get_runtime_config, load_secret
from coldopen.guardrails.policy import evaluate_policy_guardrail
from coldopen.memory.compaction import compact_history
from coldopen.memory.consolidator import schedule_memory_consolidation
from coldopen.memory.store import ColdOpenStore
from coldopen.observability.logging import get_logger
from coldopen.observability.tracing import init_tracing, traced_span

_LOGGER = get_logger("coldopen.main")


async def run_cli(team: str, region: str, theme: str, output: str) -> None:
    """Initialize config, guardrails, tracing, and run the ColdOpen ADK pipeline."""
    config = get_runtime_config()
    init_tracing(export_to_console=False)

    api_key = load_secret("coldopen-google-api-key", config.project_id, env_fallback="GOOGLE_API_KEY")
    if api_key and not os.getenv("GOOGLE_API_KEY"):
        os.environ["GOOGLE_API_KEY"] = api_key

    brief = f"Create an 18-item ColdOpen deck for team '{team}' (region={region}) on theme '{theme}'."
    allowed, reason = evaluate_policy_guardrail(brief)
    if not allowed:
        raise SystemExit(reason)

    store = ColdOpenStore(db_path=config.db_path)
    session_id = f"{team}-{uuid.uuid4().hex[:8]}"
    store.append_session_message(session_id, "user", brief)
    compacted = compact_history(store.get_session_messages(session_id))

    with traced_span("coldopen.run", {"team": team, "region": region}):
        from google.adk.runners import InMemoryRunner
        from google.genai import types

        from coldopen.agents.pipeline import build_coldopen_pipeline

        pipeline = build_coldopen_pipeline()
        runner = InMemoryRunner(agent=pipeline, app_name="coldopen")
        session = await runner.session_service.create_session(app_name="coldopen", user_id=team)

        prompt_text = f"{compacted.summary_prefix}\n{brief}".strip()
        content = types.Content(role="user", parts=[types.Part.from_text(text=prompt_text)])

        async for event in runner.run_async(
            user_id=team,
            session_id=session.id,
            new_message=content,
        ):
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if part.text:
                        print(part.text)

        consolidation_task = schedule_memory_consolidation(store, team_id=team, theme=theme, item_count=18)
        await consolidation_task
        _LOGGER.info("coldopen_run_complete", extra={"structured_fields": {"output": output, "team": team}})


def main() -> None:
    parser = argparse.ArgumentParser(description="ColdOpen: AI icebreaker deck producer")
    parser.add_argument("--team", default="default-team", help="Team identifier for deduplicating past games")
    parser.add_argument("--region", default="global", help="Cultural/geographic reach: global, us, or india")
    parser.add_argument("--theme", default="IKEA or Death Metal", help="Binary guessing theme")
    parser.add_argument("--output", default="coldopen_deck.pptx", help="Output .pptx file path")
    args = parser.parse_args()
    asyncio.run(run_cli(args.team, args.region, args.theme, args.output))


if __name__ == "__main__":
    main()

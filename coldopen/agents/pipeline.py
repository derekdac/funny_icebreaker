"""ADK multi-agent orchestration for ColdOpen with strategic Pro/Flash model routing."""

from __future__ import annotations

import os

from a2ui.adk import SendA2uiToClientToolset
from a2ui.core.basic_catalog import BasicCatalog
from a2ui.processor.processor import A2uiRequestProcessor
from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.tools import ToolContext
from google.adk.tools.agent_tool import AgentTool
from google.genai import types

from coldopen.guardrails.policy import adk_policy_and_compaction_callback
from coldopen.prompts.constitution import (
    ASSET_INSTRUCTION,
    COMPILER_INSTRUCTION,
    HOST_INSTRUCTION,
    IDEATOR_INSTRUCTION,
    SYSTEM_CONSTITUTION,
)
from coldopen.tools.compiler import (
    compile_pptx_from_draft,
    get_team_played_history,
    record_deck_feedback,
    save_and_audit_draft,
)
from coldopen.tools.media import fetch_wikimedia_specimens_batch

MODEL_PRO = os.getenv("COLDOPEN_MODEL_PRO", "gemini-3.1-pro-preview")
MODEL_FLASH = os.getenv("COLDOPEN_MODEL_FLASH", "gemini-3.8-flash")


def exit_loop(tool_context: ToolContext) -> dict[str, str]:
    """Exit the draft-and-audit loop once headline leak and entropy checks pass."""
    tool_context.actions.escalate = True
    return {"status": "loop_exited"}


def build_coldopen_pipeline() -> LlmAgent:
    """Construct the multi-agent ColdOpen ADK pipeline.

    Architecture:
        0. HostAgent (coldopen_host): greets the user, proposes 5 funny themes,
           and delegates to the studio pipeline once a theme is chosen.
        1. DraftAuditLoop (LoopAgent):
           - IdeatorAgent: generates creative binary pairs, checks SQLite history.
           - CriticAgent: audits headline leaks and sequence entropy.
        2. AssetAgent: fetches disambiguated Wikimedia thumbnails.
        3. CompilerAgent: compiles 16:9 PPTX deck and records history.
    """
    sfw_safety_settings = [
        types.SafetySetting(
            category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
            threshold=types.HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
        ),
        types.SafetySetting(
            category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
            threshold=types.HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
        ),
        types.SafetySetting(
            category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
            threshold=types.HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
        ),
        types.SafetySetting(
            category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
            threshold=types.HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
        ),
    ]

    low_thinking = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_level=types.ThinkingLevel.LOW),
        safety_settings=sfw_safety_settings,
    )
    pro_sfw_config = types.GenerateContentConfig(safety_settings=sfw_safety_settings)

    ideator_agent = LlmAgent(
        name="ideator_agent",
        model=MODEL_PRO,
        description="Generates ambiguous binary icebreaker items, runs deterministic SFW/leak/entropy audits, and saves draft_id.",
        instruction=IDEATOR_INSTRUCTION,
        generate_content_config=pro_sfw_config,
        tools=[get_team_played_history, save_and_audit_draft],
        output_key="draft_summary",
    )

    asset_agent = LlmAgent(
        name="asset_likeness_agent",
        model=MODEL_FLASH,
        description="Concurrently sources Wikimedia specimen images for a saved draft_id.",
        instruction=ASSET_INSTRUCTION,
        generate_content_config=low_thinking,
        tools=[fetch_wikimedia_specimens_batch],
        output_key="asset_summary",
    )

    compiler_agent = LlmAgent(
        name="deck_compiler_agent",
        model=MODEL_FLASH,
        description="Compiles a saved draft_id into a minimalist 16:9 PowerPoint deck.",
        instruction=COMPILER_INSTRUCTION,
        generate_content_config=low_thinking,
        tools=[compile_pptx_from_draft],
        output_key="compile_result",
    )

    production_pipeline = SequentialAgent(
        name="coldopen_production_pipeline",
        description="Concurrently fetches images and compiles the final 16:9 PPTX deck from draft_id.",
        sub_agents=[asset_agent, compiler_agent],
    )

    a2ui_enabled = os.getenv("COLDOPEN_ENABLE_A2UI", "true").lower() == "true"
    a2ui_processor = A2uiRequestProcessor(catalogs=[BasicCatalog("0.9")])
    a2ui_toolset = SendA2uiToClientToolset(
        a2ui_enabled=a2ui_enabled,
        a2ui_processor=a2ui_processor,
    )

    return LlmAgent(
        name="funny_icebreaker_game_agent",
        model=MODEL_FLASH,
        description=f"{SYSTEM_CONSTITUTION}\nFunny Icebreaker Game Agent: brainstorms 10 fresh themes and orchestrates HITL deck generation.",
        instruction=HOST_INSTRUCTION,
        generate_content_config=low_thinking,
        before_model_callback=adk_policy_and_compaction_callback,
        tools=[
            AgentTool(agent=ideator_agent),
            AgentTool(agent=production_pipeline),
            record_deck_feedback,
            a2ui_toolset,
        ],
    )

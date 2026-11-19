# Funny Icebreaker Game Agent

**Funny Icebreaker Game Agent** is a multi-agent meeting icebreaker and game-show deck producer built with the Google Agent Development Kit (ADK) and A2UI. Given a short host prompt, it brainstorms 10 fresh, absurd binary guessing themes, audits every item for headline leaks and predictable patterns, pauses for host approval, sources authentic reveal imagery, and compiles a minimalist, presenter-ready `.pptx` deck with speaker notes.

## Architecture

```
Host Brief
   │
   ▼
[PII Redaction & Policy Guardrail]
   │
   ▼
[Coordinator Agent (Pro)]
   ├─► [LoopAgent: Draft & Audit]
   │      ├─► Ideator Agent (Pro): proposes 18 binary items + trivia notes
   │      └─► Leak & Entropy Critic (Flash): checks giveaways & sequence randomness
   │
   ├─► [HITL Approval Gate]: host reviews & approves the 18-item manifest
   │
   ├─► [Asset & Likeness Agent (Flash)]: fetches Wikimedia specimens & verifies likeness
   │
   └─► [Deck Compiler]: builds 38-slide 16:9 .pptx + saves team history to SQLite
```

## Quick Start

1. **Create a virtual environment and install dependencies:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -e ".[dev]"
   ```

2. **Configure environment variables (never commit `.env`):**
   ```bash
   cp .env.example .env
   # Set GOOGLE_API_KEY or GOOGLE_CLOUD_PROJECT + Secret Manager
   ```

3. **Run the offline test & golden evaluation suite:**
   ```bash
   pytest
   ```

4. **Run ColdOpen interactively (or via ADK Web):**
   ```bash
   python -m coldopen.main --team "platform-eng" --region "global" --theme "IKEA or Death Metal"
   ```

## Project Layout

- `coldopen/agents/` — ADK multi-agent orchestration (`SequentialAgent`, `LoopAgent`, Pro/Flash routing)
- `coldopen/tools/` — Pydantic-validated tools with guided error recovery (`leak_audit`, `entropy_check`, `wikimedia_fetcher`, `pptx_compiler`)
- `coldopen/memory/` — SQLite persistent session & team history store, sliding-window compaction, and async background consolidation
- `coldopen/guardrails/` — Safety, policy, and self-evaluation guardrail plugins + HITL approval hook
- `coldopen/observability/` — OpenTelemetry distributed tracing, structured JSON logging, intent-vs-outcome tracking, and PII scrubbing
- `coldopen/config/` — Secret Manager integration (with local env fallback)
- `coldopen/prompts/` — System constitution and agent instructions
- `infra/` — Terraform IaC for Vertex AI, Secret Manager, Cloud Trace, and Cloud DLP
- `tests/` — Unit tests and golden-dataset regression evals

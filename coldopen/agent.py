"""ADK Web UI entry point exposing root_agent."""

from coldopen.agents.pipeline import build_coldopen_pipeline

root_agent = build_coldopen_pipeline()

from google.adk.apps import App

app = App(root_agent=root_agent, name="coldopen")

# ADR-0001: Record decisions as ADRs in the repo

Status: accepted (2026-10-26)

## Context
Several agents (Claude Code cloud and local, Codex, OpenCode) and the owner work on
this repo across sessions with no shared memory. Decisions living only in chat are lost.

## Decision
Durable decisions are Markdown ADRs in `docs/decisions/`, numbered, append-only.
Task state lives in `docs/roadmap/BACKLOG.md`; Mac-only work in
`docs/handoffs/MAC_LOCAL_HANDOFF.md`. `AGENTS.md` is the single tool-neutral entry point.

## Consequences
Any agent can resume from the repo alone. Contradictions are resolved by the newest ADR.

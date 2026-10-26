# ADR-0008: Every result is labelled with the environment that produced it

Status: accepted (2026-10-26)

## Decision
Labels: `cloud`, `gha-linux`, `gha-macos`, `mac-m5`, `self-hosted`, `hw`
(defined in `docs/testing/TESTING_STRATEGY.md`). A task's acceptance criteria name
the label required; evidence from another label does not close it. In particular:
GitHub's macOS arm64 runner is not the owner's M5; simulation never closes hardware tasks.

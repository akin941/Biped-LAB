# ADR-0004: Apache-2.0 for this repo; no third-party meshes; original robot design

Status: accepted (2026-10-26). Owner may revisit the license before any commercial plan.

## Context
- Microduck code and MJCF: Apache-2.0. Microduck STL meshes: **CC BY-NC-SA 4.0**
  (non-commercial, share-alike). Pretrained policies: apache-2.0 (HF card).
- This repo is public on GitHub.
- The long-term goal is an original robot; the owner's commercial intent is unknown.

## Decision
- This repo is Apache-2.0 (compatible with upstream code we learn from).
- No Microduck meshes or derived meshes are committed; they are only used from the
  pinned `third_party/` checkout at runtime.
- Any code adapted from upstream keeps its attribution in a header comment and NOTICE.
- The custom biped (Phase 4+) uses only original geometry, so it is free of NC/SA terms.

## Consequences
We cannot publish screenshots/derived meshes of Microduck without CC BY-NC-SA attribution
terms; the custom robot has no such constraint.

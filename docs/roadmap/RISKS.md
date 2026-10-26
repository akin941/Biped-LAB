# Risk register

Likelihood/Impact: L/M/H. Review at each phase gate.

| ID | Risk | L | I | Mitigation | Owner/status |
|---|---|---|---|---|---|
| R-01 | Official training needs CUDA; cannot reproduce on Mac | H | M | ADR-0005: CPU MuJoCo + PPO; scale experiments; compare with official weights | open, accepted |
| R-02 | Sim-to-sim gap: MJCF position actuators ≠ BAM training actuator → official policies misbehave in our runner | H | M | WP1-06 BAM integration; label reports with actuator model | open |
| R-03 | Runtime post-processing (scale 0.9, voltage comp, low-pass) not in sim | M | M | WP1-09; document contract | open |
| R-04 | Upstream churn breaks assumptions (esp. microduck-lab single maintainer) | H | L | pinned shas; characterization tests; bump PRs | mitigated |
| R-05 | Malicious/unsafe upstream code or instructions | L | H | ADR-0003: read as data, never execute; review on bump | mitigated |
| R-06 | Pretrained weights tampered/changed on HF | L | M | pinned revision + sha256 verify | mitigated |
| R-07 | License: CC BY-NC-SA meshes leak into our repo/robot | M | M | ADR-0004; .gitignore third_party; original geometry | mitigated |
| R-08 | Cloud container cannot reach Hugging Face | H (now) | L | downloads in GitHub CI and on Mac; owner can allow `huggingface.co` in environment network settings | open |
| R-09 | Mac-only issues (viewer needs mjpython, macOS ≥ 14 for onnxruntime) | M | L | handoff MAC-01..04; gha-macos CI as early warning | open |
| R-10 | CPU training too slow for useful gaits | M | M | curricula, symmetry augmentation (proven by microduck-lab), fewer envs, owner can approve compute later | open |
| R-11 | Custom robot actuators under-sized (torque/thermal) | M | H | Phase 4 sizing from sim torque traces with 2× margin; actuator identification | later |
| R-12 | Physical injury / damage at first power-on | L | H | ADR-0006, hardware e-stop, suspended tests, torque limits, written procedure | later |
| R-13 | Battery fire (LiPo/Li-ion) | L | H | protected packs, fuse, charging rules (SECURITY_AND_SAFETY S-4) | later |
| R-14 | LLM/agent issues unsafe motion | L | H | bounded command schema only; supervisor on MCU | later |
| R-15 | CI cost/time grows (macOS minutes, long sims) | L | L | public repo (free minutes); tiers; no training in CI | mitigated |
| R-16 | Agents claim unrun results | M | H | ADR-0008 labels; AGENTS.md hard rule; evidence records | mitigated |
| R-17 | Self-hosted runner (8 GB) overloaded | L | M | not used until needed; job memory caps | later |

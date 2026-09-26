# Release Cleanup Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans or subagent-driven-development for implementation and review.

**Goal:** Remove disposable model and match artifacts while preserving baselines and deliver a verified source repository.

**Architecture:** Keep existing training entrypoints and native implementations. Separate reproducible source, external inputs, and disposable outputs; do not invent a second pipeline.

**Tech Stack:** Python 3.12, PowerShell, Rust, PyTorch GPU.

**Spec:** User request: retain models inside baselines only, remove match records, update documentation and Git project.

## Constraints

- Never delete baselines, downloaded training data, environments, toolchains, native libraries, or source fixtures.
- Hash baseline inputs before and after cleanup. Reject reparse points and paths outside this repository.
- Preserve and review existing uncommitted native/online source work.
- No model promotion, training restart, force-push, or history rewriting.

## Tasks

- [ ] Inventory exact removable files, bytes and baseline hashes; save a dry-run manifest outside Git.
- [ ] Delete only reviewed generated outputs/models/match records; verify baseline hashes and remaining model inventory.
- [ ] Redirect default sanma input paths to baselines; align dependency and usage documentation; keep tests publishable.
- [ ] Run profile, one-click dry-run, source checks and native/online regression checks; distinguish missing-input runtime checks.
- [ ] Independently review release diff, tracked file sizes and confidential data risks; address findings.
- [ ] Commit and push to the existing project branch without rewriting history; report exact verification and any remaining restrictions.

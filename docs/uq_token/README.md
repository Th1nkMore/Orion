# UQ Token Research Track

> Status: active
> Started: 2026-06-20
> Branch: `mid-report`
> Baseline commit: `45a4c4b`

Current gate:

> Full training is paused until UQ-token grounding and correct/zero/shuffled
> controls are validated on a small route subset.

This directory is the source of truth for the Density UQ + LLM uncertainty
token stage of UQ-ORION. New design decisions, implementation notes, experiment
plans, run records, and result interpretations for this stage must be recorded
here.

Older documents elsewhere in the repository remain historical references. If
they conflict with this directory, use the documents here unless a later entry
explicitly reverses a decision.

## Current Objective

Provide EVAViT representation uncertainty to the language model as continuous
uncertainty tokens, so the language model can incorporate perception
reliability when producing the waypoint representation.

The main claim is:

> A normal-feature density model extracts visual reliability information from
> the frozen EVAViT backbone, and projected uncertainty tokens expose that
> information to the LLM before it forms the planning representation.

FiLM L1 and L2 are retained as comparison methods. L2 is not the primary method
because it changes the planning feature after LLM reasoning.

## Document Map

| Document | Purpose |
| --- | --- |
| [context.md](context.md) | Project background, completed work, constraints, and current assets |
| [method.md](method.md) | Proposed UQ token architecture and training design |
| [experiments.md](experiments.md) | Experimental protocol, ablations, metrics, and acceptance criteria |
| [implementation.md](implementation.md) | Code plan, interfaces, checkpoints, and implementation status |
| [decisions.md](decisions.md) | Dated architectural decisions and rejected alternatives |
| [log.md](log.md) | Chronological code changes, server runs, failures, and results |

## Update Rules

1. Record a design change in `decisions.md` before or alongside implementation.
2. Update `implementation.md` whenever code ownership, interfaces, or status changes.
3. Record every substantial server run in `log.md`, including the exact command,
   commit, configuration, output path, and result.
4. Update `experiments.md` when an experiment is added, removed, or redefined.
5. Do not report a result as final until the split, checkpoint, and metric source
   are recorded in `log.md`.

## Current Status

| Work item | Status |
| --- | --- |
| EVAViT feature extraction, 12,806 val frames | Complete |
| Density descriptor cache | Complete |
| Normal-only density model | Complete |
| Density UQ integrated with ORION interface | Complete |
| UQ token architecture | Implemented |
| UQ token implementation | Complete |
| UQ token projector training | Multi-step smoke test passed |
| LLM LoRA training with UQ tokens | Multi-step smoke test passed |
| UQ grounding head and loss | Designed, not implemented |
| Baseline and ablation evaluation | Not started |

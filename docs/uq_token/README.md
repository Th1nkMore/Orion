# UQ Token Research Track

> Status: active
> Started: 2026-06-20
> Branch: `mid-report`
> Baseline commit: `45a4c4b`

Current gate:

> Counterfactual score-only grounding passed both causal-use and held-out
> calibration gates, but the grounding-only checkpoint is not a valid planning
> initializer. Joint waypoint-grounding also failed the planning gate because
> it forces the trajectory representation itself to encode score. Further
> scaling is stopped pending a separate LLM UQ readout token.

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
| [next_stage.md](next_stage.md) | Next architecture, supervision plan, gates, fallbacks, and stop rules |
| [risk_qa_plan.md](risk_qa_plan.md) | Explicit language-risk interface and pre-experiment protocol |

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
| UQ grounding head and loss | Complete |
| Correct/zero/shuffled/none intervention evaluation | Complete |
| Counterfactual score-only grounding | Complete; causal and MAE gates passed |
| Grounding-only checkpoint as planning initializer | Rejected |
| Joint grounding + planning pilot | Failed planning gate; stopped |
| Separate LLM UQ readout token | Next architecture revision |
| Baseline and ablation evaluation | In progress |

# Qwen visibility-belief research track

> Status: implementation in progress; O3 U tokenizer implemented, real-artifact validation pending
> Decision date: 2026-09-06 (Asia/Shanghai)
> Branch: `codex/qwen-drive-transition`

This directory is the current design authority for the transition from the
EVAViT/Orion corruption-UQ line to explicit occlusion-aware visibility belief
for Qwen-Drive. The older `docs/uq_token/` directory remains experiment
history and must not be rewritten as if its EVAViT artifacts were compatible
with Qwen.

## Documents

| Document | Purpose |
| --- | --- |
| [context.md](context.md) | Verified starting state, evidence, constraints, and claim boundary |
| [adr.md](adr.md) | Accepted architecture, training, evaluation, fallback, and rejected alternatives |
| [implementation.md](implementation.md) | Ordered implementation ladder, current status, and execution trace |

## Immediate execution target

Build the smallest oracle-U vertical slice on Route 151:

1. generate oracle visibility belief from CARLA depth and calibration;
2. form the accepted 2.5D BEV and frontier/global tokens;
3. inject those tokens into the 4B VLM, not directly into the Planning Expert;
4. run structured U-grounding warm-up with staged LoRA;
5. train the longitudinal trajectory response while preserving the released
   Planning Expert's 50-waypoint action representation;
6. compare the fixed Qwen baseline with the otherwise identical oracle-U arm.

The Route 151 overfit checkpoint, if used to verify plumbing, is disposable
and may not be reported as generalization evidence.

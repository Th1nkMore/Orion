# Qwen visibility-belief research track

> Status: implementation in progress; V1k valid negative, field alignment next
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
| [vlm_insertion_contract.md](vlm_insertion_contract.md) | Verified Qwen3.5 embedding, mRoPE, cache, anchor, and control contract |
| [grounding_contract.md](grounding_contract.md) | V1 labels, anti-shortcut permutation, freezing boundary, and acceptance gates |
| [field_alignment_contract.md](field_alignment_contract.md) | A0 textual upper bound and A1/A2 field/language/visual alignment contract |

## Immediate execution target

Establish route-disjoint U alignment before resuming the driving objective:

1. measure the literal text/numeric upper bound with and without native RGB;
2. train field/address/numeric U-language alignment with dense supervision;
3. align frontier records to calibrated camera regions;
4. verify structured continuous-U reasoning on unseen routes;
5. only then resume longitudinal trajectory response and closed-loop work.

The Route 151 overfit checkpoint, if used to verify plumbing, is disposable
and may not be reported as generalization evidence.

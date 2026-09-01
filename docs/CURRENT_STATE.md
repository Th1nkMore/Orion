# UQ-ORION current state

Last audited: 2026-09-02 (Asia/Shanghai)

This is the canonical human-readable status document. It separates completed
evidence from planned claims and points to the machine-readable protocols that
own experiment authority. When an older report calls Density UQ, FiLM, a scalar
UQ token, a pre-LLM scalar adapter, or a hard governor the "current mainline",
that statement is historical and superseded here.

## 1. Active research claim and responsibility boundary

The active architecture is:

```text
multi-view observations + temporal context
                    |
                    v
Stage 1: frozen task-agnostic spatial observation-UQ adapter
                    |
                    v
frozen U-tokenizer preserving view, position, component and time
                    |
                    v
ORION visual evidence + navigation + ego state
                    |
                    v
Stage 2L: VLM task relevance R + structured QA
                    |
                    v
Stage 2P: trajectory/planning response
```

Stage 1 answers where visual evidence is unreliable. It must not consume route,
actor, TTC, collision, corruption-family, or action labels. Stage 2L decides
whether that uncertainty matters to the current driving task. Stage 2P decides
how the trajectory should respond. Planning gradients stop at the Stage-1
boundary.

The machine-readable responsibility contract is
`configs/scenario_factory/protocol_v1.json`; the detailed architecture contract
is `docs/spatial_uq_two_stage_v2.md`.

Retired from the mainline:

- legacy Density UQ as the closed-loop decision signal;
- global scalar UQ tokens as the planning mechanism;
- FiLM or the historical pre-LLM scalar vision adapter as the main method;
- BEV costs and scalar speed governors;
- fixed-duration stop-and-hold control.

These implementations and results remain in the repository as baselines and
negative evidence.

## 2. Authoritative checkpoint lineage

Two different hashes refer to two different Stage-1 artifacts and must not be
conflated:

1. Spatial observation-UQ adapter:
   `/public/home/lidachuan/orion_work/observation_uq_v3/runs/counterfactual_pairwise_native_repair_seed20260828_r1/counterfactual_evidence_pairwise_native_repair.pt`
   (`0555f0f341c80a88e18c5864573f0be0641fb828931bea7809e2f5544665f2c8`).
   It produces the spatial U maps. It is diagnostic/auxiliary because it missed
   the frozen heavy-native-fog patch-AUROC gate (`0.590313 < 0.600000`).

2. Task-agnostic U-tokenizer:
   `/public/share/lidachuan/orion_assets/scenario_factory/stage1_u_tokenizer_pretraining/stage1_u_tokenizer_task_agnostic_v1_200_retry1/stage1_u_tokenizer_task_agnostic_v1.pt`
   (`c727aa89d21b9a3362b240509e9ea75011bbe3d20d58e996c69816b97afa38a2`).
   It compresses maps from the first artifact into tokens. It does not replace
   or retrain the spatial adapter.

The Stage-1 tokenizer completed 200 optimizer steps on 60 train and 20 dev map
presentations. Dev summary MAE improved from `0.169409` to `0.031972`; the
zero-U anchor MAE improved from `0.147658` to `0.010475`. This passes the
representation-only gate. It does not establish U correctness, task relevance,
VLM understanding, planning benefit, or safety.

Authoritative result:
`configs/scenario_factory/amendments/20260831_stage1_u_tokenizer_pretraining_result_v1.json`.

## 3. Stage-2L evidence and current largest gap

No released checkpoint has yet combined the frozen U-tokenizer, explicit
same-view ORION visual evidence, route/ego context, and a held-out-passing
task-relevance head.

Stage-2L v10 loaded the frozen U-tokenizer and optimized stably, but failed the
Phase-A foreground-recall gates after 40 steps:

- train recall `0.628571` (gate `0.9`);
- dev recall `0.341455` (gate `0.8`).

Stage-2L v10.1 isolated the view-binding problem by adding six same-view ORION
feature grids and intentionally not loading Stage-1 U or the U-tokenizer. After
120 steps it improved train/dev AP to `0.910158 / 0.415242`, with train/dev
recall `0.866435 / 0.443067`, but still failed both recall gates. The large
train/dev gap indicates an unresolved interface/target/generalization problem;
it is not evidence that adding epochs or parameters alone will solve it.

The v10.1 run used 13 independent train events and 4 dev events. Its per-event
dev AP ranged from `0.104170` to `0.662231`, and held-out predictions remained
over-concentrated on `CAM_FRONT`. These results do not isolate architecture
from data coverage: independent-event diversity, sparse-support supervision,
coordinate/view binding, and scene-template shortcuts all remain plausible
causes. In particular, the available evidence does not justify claiming that
route/event count has been ruled out as a root cause.

Authoritative results:

- `configs/scenario_factory/amendments/20260831_stage2l_v10_phase_a_result_v1.json`;
- `configs/scenario_factory/amendments/20260831_stage2l_v101_phase_a_terminal_v1.json`.

The v11.1 consumer-grid-controlled result has now made the largest immediate
gap more precise: **held-out contextual task-relevance localization is not yet
reliable enough to expose U to the language bridge**. The dense R target is
owned by route-corridor and visible conflict-actor geometry, so a model can
learn substantial R structure from visual/route context without using U. The
v10.1 no-U result demonstrates that this bypass is real, while the valid v11.1
controlled comparison shows train on/off ordering at `52/60` but held-out
ordering at only `11/20`. All ten held-out controls whose resulting risk view
was `CAM_FRONT` passed; the five left-side controls and four controls with no
effective R risk region all failed. Thus the current first bottleneck is
R-target/view/coordinate/event generalization, not U-tokenizer reconstruction
or a demonstrated language-bridge failure. The frozen Stage-1 adapter remains
a separate diagnostic evidence-loss proxy and has not passed its independent
native gate.

A CPU-only frozen-artifact audit has now ruled out the most direct coordinate
implementation explanation. All 80 targets were rebuilt from their original
frame metadata; route corridor, actor support, combined relevance, QA sidecar,
and pooled 10x10 trainer target matched with maximum absolute error exactly
`0.0`. Camera order and every referenced hash also matched. This establishes
reproducibility, not semantic completeness of the weak geometry target.

The remaining gap separates into supervision and coverage:

- `CAM_FRONT` receives `65.76%` of the train foreground-Brier weight, while
  front-left, front-right and back-left each receive only `4.13–5.06%`;
  back-right receives no positive train support.
- held-out foreground recall is `0.5854` on front, `0.3833` on back,
  `0.0089` on back-left, and exactly `0` on both front-side views.
- Route195's pure front-right failures have same-view train-neighbour cosine
  only `0.3841 / 0.3952` and only two train events contain any front-right
  support: independent-event/view coverage is a primary limitation.
- Route152 front-left neighbours are much closer (`0.75–0.86`), yet R remains
  front-dominant. Thus feature OOD is not a sufficient explanation; the loss
  and route-plus-actor binding also fail to use the limited side supervision.
- combined route-plus-conflict-actor controls pass only `2/8` on dev, versus
  `8/11` for route-only controls. These support modes must no longer be hidden
  inside one aggregate metric.

The immediate engineering gap is therefore a held-out-generalizing,
**factorized** bridge:

```text
R_context = f(same-view visual evidence, route, ego state)   # no U input
U         = frozen Stage-1 adapter and frozen U-tokenizer
K         = U * sigmoid(R_context)
```

A combined run is useful only if matched counterfactuals hold visual, route,
ego state and R fixed while changing U, then show the correct change in K and
language-facing task semantics. Injecting U directly into the R query would
erase this identifiability and is not the next repair. Tokenizer
reconstruction, decoder choice, and GPU budget are not immediate blockers;
independent-event coverage and Stage-1 external validity remain separate open
problems.

### 3.1 Evidence-based gap reassessment

An independent priority audit at `2026-09-01T15:52:23+08:00` deliberately
separated the distance to the final paper claim from the next scientifically
identifiable experiment. Its conclusion is that the largest **scientific** gap
is not a particular corruption, decoder or benchmark size. It is the absence
of a held-out causal bridge in which visual evidence, route, ego state and one
shared contextual R are fixed, changing only spatial U produces the correct
localized `K = U * sigmoid(R)`, and the structured/language answer changes
consistently without universal caution.

The immediate engineering blocker inside that larger gap is held-out
conflict-actor R localization. The root identifiability limitation is sparse
independent event/component-view support, while learned Stage-1 U external
validity is a parallel foundational gap. Stage2-P and closed-loop safety are
the largest remaining distance to the final claim, but optimizing them before
the semantic bridge would not establish why a behavior changed. Likewise, a
new positive corruption-induced failure case would be useful downstream but
would not by itself distinguish task-relevant U use from a route, corruption
or universal-slowing shortcut.

At the audit snapshot, Job `1121553` remained `PENDING (Priority)` with no
optimizer step or outcome. The audit therefore does not use its result, does
not alter its frozen gates, and authorizes no retry or downstream job. Its
machine-readable evidence hierarchy and terminal branches are frozen in
`configs/scenario_factory/amendments/20260901_research_gap_reassessment_v1.json`.

### 3.2 Soft-gate vertical-slice progression

At `2026-09-01T16:06:49+08:00`, while Job `1121553` was running but before its
training log or metrics were reviewed, the engineering progression policy was
changed prospectively. Model-quality thresholds are now **soft diagnostics**
during the first end-to-end pass. A weak per-view recall, background FPR,
train/dev gap, controlled-U ordering, QA score or first closed-loop outcome is
recorded but does not by itself prevent the next interface from being run.

Only integrity and causal-contract failures remain hard stops: hash or split
drift, locked-test leakage, forbidden labels or planning gradients crossing
the Stage-1 boundary, non-finite optimization, incompatible tensors or an
invalid runtime/artifact. Job `1121553` retains its original seven-gate verdict
and all metrics must still be reported, but an integrity-valid terminal
checkpoint is carried into one bounded controlled-U structured-QA smoke even
if one or more model-quality gates fail. No extra R epochs are inserted first.

The first vertical slice then continues through one Stage2-L semantic run, one
Stage2-P interface run and a very small CARLA closed-loop smoke. These runs are
engineering diagnostics, not formal releases or safety claims. Once the whole
chain exposes its actual failures, the earliest broken causal stage is repaired
and its downstream consumers are rerun. Front and central rear views remain the
primary longitudinal analysis; side and oblique views remain reported
auxiliary generalization rather than stage blockers. In particular, current
rear actor localization does not yet prove braking-conditioned relevance; that
distinction is probed in QA instead of being required from the present scene
bank alone.

This policy authorizes no new job by itself and does not unlock locked test or
formal benchmark claims. The machine-readable record is
`configs/scenario_factory/amendments/20260901_vertical_slice_soft_gate_progression_v1.json`.

The first Stage2-L semantic slice is now terminal. Job `1122494` completed on
gpu4 in `00:28:40` with all 40 steps finite and only
`TaskRiskLanguageBridge` trainable. Independent validation reproduces all 17
event artifacts, separate route/actor R components, derived union R and K, with
exact zero-U/zero-K and exact `K=U*R`; the only cross-device probability
difference is one float32 ULP (`5.96e-08`, below the fixed `1e-07` audit
tolerance). Controlled-U on-over-off ordering is `0.9667` on train but only
`0.6` on dev, so that diagnostic remains a soft failure.

The language result exposes the earlier break more sharply. Mean target NLL
drops from `13.9605` to `10.3044` on train and from `15.1263` to `10.7124` on
dev, but dev target preference falls from `0.375` to `0.25`, and the full-U and
no-U variants are both exactly `0.25` after training. The bridge therefore
learned generic answer likelihood rather than counterfactual U semantics. This
checkpoint is retained as `semantic-insensitive` and carried into one bounded
Stage2-P interface smoke under the soft-gate policy; it is not a Stage2-L pass.
The downstream smoke must consume controlled K, preserve exact zero-K native
behavior, keep privileged route/actor/TTC/outcome labels out of its forward
path, and train only the trajectory-response interface. The terminal record is
`configs/scenario_factory/amendments/20260901_stage2l_v122_vertical_slice_semantic_terminal_v1.json`.

### 3.3 First engineering vertical slice is complete

The downstream interfaces have now been exercised without repairing the
upstream soft failures. Controlled-K Stage2-P Job `1123187` completed all 80
finite optimizer steps after Job `1123186` was invalidated as a zero-step
launch-field error. Its forward path contains only frozen ORION planning
context and already task-relevant K; raw U and privileged route, actor, TTC and
outcome context do not enter. The trained checkpoint preserves the native
planning context and exact zero-K trajectory identity. All four positive
targets produce nonzero response and their MAE is `0.101049 m`.

This is not a Stage2-P quality pass. The mean hard-negative response is only
`0.031567 m`, but the maximum irrelevant-K and view-shuffled-K responses are
`0.371179 / 0.397581 m`, above the frozen `0.2 m` soft specificity bound. The
checkpoint is explicitly `engineering_smoke_only`, `formal_stage2p_ready=false`
and `closed_loop_eligible=false`. Its terminal record is
`configs/scenario_factory/amendments/20260901_stage2p_v1_controlled_k_interface_terminal_v1.json`.

One explicit engineering override then loaded that exact checkpoint in live
ORION/CARLA. Job `1123244` ran Route147 (`Town02`,
`DynamicObjectCrossing_1`) to terminal completion. Across 614 contiguous
control frames, the clean visual input, disabled legacy Density UQ, disabled
Stage-1 sidecar, disabled scalar governor and disabled privileged planning
response all matched the preregistration. An external oracle K occupied the
frozen front-view on-path region for exactly 60 frames. It produced gate
`0.80000001` and finite residuals with maxima `0.401899 m` lateral and
`18.771713 m` longitudinal. Every frame before and after that window had an
exact zero residual; after slowing to near zero during the window, the car
recovered above `2 m/s` by step 269 and ended at `4.9247 m/s`.

Official soft outcomes were `100%` route completion, zero collisions, zero
red-light/route-deviation/blocked infractions, composed score `100`, and
minimum recorded OBB TTC `1.351309 s`. The leaderboard route banner still
reported `FAILURE` because MinSpeedTest recorded 20 entries (`78.79%` display),
which must not be hidden or interpreted as a safety improvement. The external K
is an engineering oracle, not learned U. Therefore this run establishes only
that K can reach the Stage2-P checkpoint, modify live trajectory control for a
bounded window, and restore native behavior afterward.

The first vertical slice is thus an engineering pass with two upstream soft
failures: Stage2-L counterfactual U semantics and Stage2-P low-K specificity.
The next repair order is (1) make held-out QA preference depend on U while R is
fixed, (2) add irrelevant/view-shuffled hard negatives to Stage2-P, and (3)
rerun this same bounded slice with learned U-to-K. Formal Stage2-L, formal
Stage2-P, learned-U closed loop, locked test and benchmark expansion remain
locked. The combined terminal record is
`configs/scenario_factory/amendments/20260901_vertical_slice_u_r_k_qa_trajectory_carla_terminal_v1.json`.

Route expansion is proceeding independently without widening the first model
smoke. At `2026-09-01T16:23:59+08:00`, the scheduler had no immediately usable
CARLA A800: the only physically unallocated card was on gpu5, which remains
excluded after the recorded Vulkan initialization failures. One development-
only `clean_off` replay, Route211 (`T_Junction`, Town04), was therefore queued
as Job `1121900` with 2 CPU, 192 GB and `gpu2,gpu5` excluded. It is a route-
geometry hard negative selected using the published ORION outcome and is not a
clean-valid, formal-split or locked-test route. No U, tokenizer, Density UQ,
Stage2 conditioning, risk governor or planning response is active. No second
route or automatic retry is authorized. The frozen batch and submission record
are under
`results/scenario_factory/batches/vertical_slice_route211_20260901_v1/`.

### 3.4 Direct-U/R ORION capacity comparison is terminal

The historical `TaskRiskLanguageBridge` is no longer part of the active
Stage2-L path. In v13.1, ORION receives the 600 frozen Stage-1 U tokens and 150
pooled, U-independent R hidden tokens directly alongside its 529 native visual
tokens. `K=U*R` is computed only for post-hoc audit and never conditions the
model. A detached leaf at the concatenated conditioning tensor exists solely
because ORION uses re-entrant gradient checkpointing; it restores LoRA/base
decoder gradients while terminating gradients before Stage 1 and the U/R
producers.

The first v13 Job `1125288` is an invalid zero-step runtime result. Its direct
forward completed, but checkpointing saw no gradient-requiring input and
dropped the LoRA graph. The corrected v13.1 runtime probe moved before the
expensive baseline evaluation and independently verified the actual gradient
groups: 256 LoRA tensors, 14 R-query tensors and 8 R-head tensors; the partial
arm additionally reached 36 base-decoder parameter tensors. Both probes saw
zero U-tokenizer gradients and took no optimizer step.

Jobs `1125300` (LoRA) and `1125510` (LoRA plus decoder layers 28–31) then each
completed 200 finite steps with zero exit code. Independent terminal validation
loads both checkpoints, verifies every tensor is finite, checks complete logs,
and reproduces protocol/preflight/launch/attestation hashes. This is a hard
runtime/integrity pass and a soft semantic failure:

| Metric | LoRA | LoRA + last 4 layers |
|---|---:|---:|
| Train target NLL, before → after | 13.8595 → 3.6266 | 13.8595 → 0.3394 |
| Dev target NLL, before → after | 14.3769 → 3.9626 | 14.3769 → 0.3850 |
| Dev full-minus-no-U preference | 0.0625 | 0.0625 |
| Dev on-path target preference | 0.0 | 0.0 |
| Dev zero-U target preference | 0.25 | 0.25 |
| Trainable parameters | 20.24 M | 829.78 M |

The partial arm lowers held-out NLL to `9.72%` of the LoRA value and improves
all four process-step NLLs, yet its entire dev preference profile is bitwise
identical to LoRA. Added capacity therefore learns the answer surface more
strongly without learning the intended counterfactual dependence on U. The
current bottleneck is not demonstrated LoRA under-capacity; it is that the
process-QA objective/data still permit route/template likelihood fitting while
ignoring direct U tokens. No post-training spatial R metric was emitted, so
finite factorized-R training loss cannot be promoted to held-out R evidence.

The next repair is to make U use identifiable in supervision and evaluation:
matched samples must prevent the same route/template prior from explaining
zero/on/off/shuffled answers, and the held-out diagnostic must directly test
whether changing U changes the correct process fields. Extra epochs, another
capacity arm, formal Stage2-L, Stage2-P, learned-U closed loop and locked test
remain unauthorized. The terminal record is
`configs/scenario_factory/amendments/20260902_stage2l_v13_1_direct_u_r_capacity_terminal_v1.json`.

## 4. Formal Stage-2L data readiness

The frozen formal target is 24 independent events with a 16 train / 4 dev / 4
locked-test split, covering 6 towns and 15 scenario families. The currently
accepted bank contains 18 events: 14 train, 4 dev, and 0 locked-test. All 18
present events are human accepted and QA-input ready, but the formal bank and
formal QA dataset are incomplete.

Two locked-test events (Routes 206 and 212) have machine-assisted technical
reviews but still require immutable human decisions. Completing the 24-event
bank is a formal-training prerequisite. The current 14/4/0 bank and the 13/4
v10.1 training split are too small to support a formal generalization claim or
to rule out event coverage as one contributor to the failed engineering
bridge. They are nevertheless sufficient for one bounded combined-interface
diagnostic before deciding whether the next repair belongs primarily in the
architecture, supervision, or event bank.

The 14 accepted train event identities do not imply 14 geometry-eligible
training events. Route177 is human accepted but retained only two frozen
fixed-offset keyframes with visible route support, below the minimum of three,
so it cannot enter the formal R/QA package. The two missing train identities
also cannot repair coverage from existing assets: Route201 exhausted its
technical retry with an invalid sensor/runtime trace, and Route208 failed the
frozen clean-liveness screen. A legacy Route208 liveness amendment labelled the
split `dev`; the frozen formal plan owns the authoritative `train` split, and
the mismatch is now recorded as clerical lineage rather than silently edited.

Authoritative records:

- `results/scenario_factory/formal_route_plans/stage2l_formal24_16_4_4_20260829_v1/formal_route_plan.json`;
- `results/scenario_factory/event_banks/stage2l_formal24_partial18_reviewed_v1.json`;
- `results/scenario_factory/review_queues/stage2l_formal24_test_technical_review_20260831/technical_review.json`.

Separately, the bounded v11 engineering dataset has completed its
identifiability preflight. A versioned route-context upgrade added only the
exact current ORION speedometer reading, bound through the existing hashed
geometry manifest and source metadata. The signed reading is preserved exactly
(including six near-zero negative sensor readings); desired speed, route
progress, TTC and outcomes were not added. The source 17-event records remain
unchanged.

The upgraded dataset contains 17 events, 80 matched groups (60 train / 20 dev)
and 1,600 QA records. All metadata checks and all 400 referenced U-tensor
loads/hashes passed; every zero/on-path/off-path/shuffled group preserved the
matched observation, route/ego context, R supervision and Stage-1 lineage. This
passes the **dataset-input gate only**. It does not prove that a model reuses a
shared R, responds to U in K or language, generalizes, or drives safely.

Authoritative result:
`configs/scenario_factory/amendments/20260901_stage2l_v11_identifiability_dataset_preflight_result_v1.json`.

## 5. Closed-loop failure-induction evidence

The approved naturalistic corruption families are front stale 200 ms,
paired-template waterdrop medium, and native motion blur medium. They are
development interventions, not uncertainty ground truth.

The completed prospective screens have not found a defensible positive case:

- Wave0: three valid clean/corrupt pairs on Route151, all valid negatives;
  Route180 was invalid because clean collided; Route194 was invalid by the
  clean-liveness gate.
- Wave1 Route158: stale 200 ms and motion blur medium were valid negatives;
  waterdrop medium was runtime inconclusive after its bounded retry.

Across those screens there are five valid negative pairs and zero positive
failure-induction pairs. This blocks a corruption-induced safety case study,
but even a new positive route would not by itself close the missing U -> R
bridge.

Authoritative results:

- `configs/scenario_factory/amendments/20260901_corruption_hardcase_online_wave0_result_v1.json`;
- `configs/scenario_factory/amendments/20260901_corruption_hardcase_wave1_route158_result_v1.json`.

The most recent Q1 clean-runtime qualification produced:

- Route160: clean qualified, 100% completion, no hard infraction, longest
  post-warmup low-speed interval 0.2 s;
- Route161: same result;
- Route165: runtime invalid after a 301 s control-trace stall near 93.5% route
  progress; not a model or safety result.

Routes160 and 161 are static-obstacle events and are retained as runtime/clean
evidence, not promoted to primary perception-dependence cases.

## 6. Stage-2P and closed-loop method status

Formal Stage-2P has not started. The completed controlled-K Stage2-P and
Route147 CARLA runs in Section 3.3 are engineering-interface diagnostics only.
Current Stage-2L runs use zero trajectory and control loss. No current-mainline
checkpoint has closed-loop authority, and no learned UQ-aware ORION result
establishes collision, TTC, traffic-rule, or route-completion improvement.
Conditional diffusion remains a later decoder ablation; changing the decoder
cannot repair the missing U-identifiability failure.

## 7. Next executable vertical slice

**Current decision (2026-09-02):** the direct-U/R v13.1 capacity comparison in
Section 3.4 supersedes the older bridge-oriented execution notes below. Do not
run more epochs or another capacity arm. The next bounded repair must make U
counterfactuals identifiable to ORION: matched prompts and answer candidates
must prevent route/event/template priors from producing the same likelihood
ranking when U is removed or moved off path. The repair must retain direct
frozen U tokens, U-independent R, zero Stage-1 gradients, all four structured
process steps, no bridge and no K model input. It should first be tested on the
existing train/dev bank with exact no-U and matched spatial counterfactuals;
only an integrity-valid result that materially improves held-out on-path and
zero-U preference justifies a later formal-data run. Formal Stage2-L, Stage2-P,
closed loop, locked test, extra epochs and a new GPU job are currently locked.

The remainder of this section records the earlier vertical-slice contract and
is retained as historical lineage; it is not current launch authority.

The next bounded engineering experiment must combine the two previously
separated interfaces without collapsing U and R into one predictor:

```text
explicit same-view ORION evidence + route/ego
  -> shared contextual relevance R

frozen spatial U maps -> frozen c727... U-tokenizer
shared R + matched U variant -> K = U * sigmoid(R)
  -> structured task fields/QA
```

Before GPU submission it must prove on CPU/preflight that:

- both Stage-1 artifacts match their distinct expected hashes;
- all Stage-1 parameters are frozen and planning gradients stop at U;
- same-view feature and U-token coordinates share one canonical camera/grid
  order;
- every matched U variant reuses bitwise-identical R logits;
- on-path/off-path U have matched peak, total mass and support count while
  changing spatial support;
- zero U preserves absence/non-conservative semantics;
- no Density score, corruption family, TTC, collision, answer-derived R, or
  ground-truth stance enters the forward pass;
- train/dev route identities remain disjoint and locked test is unread.

Held-out R foreground recall/background FPR, on-path-over-off-path K ordering,
per-variant answer preference, zero/irrelevant-U false-conservatism and
map/structured-field/text consistency remain mandatory outputs. They are soft
diagnostics during the first vertical slice rather than permission switches.
The run must also include a no-U ablation and report per-event and per-view
results rather than only aggregates.

If R remains high on train and low on dev, do not insert extra R epochs before
the first semantic run; carry the attested checkpoint forward with that failure
label. If controlled U does not change K/QA correctly, record the failed bridge
and still exercise the bounded Stage2-P interface unless the failure is an
integrity, leakage, numerical or tensor-contract defect. If learned U fails an
independent native/sensor diagnostic, record Stage 1 as a bottleneck while
retaining controlled U for interface testing. After the complete slice, repair
the earliest broken causal stage and rerun it plus downstream consumers.
Formal release gates are frozen again only before locked-test or benchmark
claims.

The machine-readable preflight contract for this distinction is
`configs/scenario_factory/stage2l_v11_identifiable_factorized_bridge_v1.json`.

The route/ego and matched-U dataset checks above are complete. The model-side
v11 runtime and bounded trainer are now also implemented. The runtime accepts
one U-independent contextual-R callable, invokes it exactly once per matched
group, reuses that tensor for zero/on-path/off-path/shuffled U, derives K, and
exposes baseline ORION visual tokens as an explicit no-U ablation. The bounded
trainer freezes the failed v10.1 step-120 same-view R checkpoint, ORION LoRA,
Stage 1 and the U-tokenizer; only `TaskRiskLanguageBridge` may update. Thus this
run cannot silently repair R by adding more R epochs.

The remote CPU preflight passed on Python 3.8.20 with 17 events, 80 groups and a
fresh full 400-U-tensor audit. It did not use a GPU, load ORION weights or start
training. A first otherwise identical preflight is retained as superseded
because its protocol contained a clerical future `created_at` timestamp; after
correcting only that timestamp, the terminal rerun was identical after removing
`protocol_sha256`. The terminal report and correction trail are bound by
`configs/scenario_factory/amendments/20260901_stage2l_v11_identifiability_model_preflight_result_v1.json`.

Exactly one 40-step, bridge-only engineering run was authorized by
`configs/scenario_factory/amendments/20260901_stage2l_v11_identifiability_smoke_launch_v1.json`.
That exact run was submitted as Slurm Job `1120666` at
`2026-09-01T11:58:20+08:00` and started on `gpu4` at
`2026-09-01T11:59:08+08:00`. The hash-bound attestation and scheduler snapshot
are recorded by
`configs/scenario_factory/amendments/20260901_stage2l_v11_identifiability_smoke_submission_v1.json`.
The job completed normally with exit code 0 after 9 minutes 54 seconds, but
stopped before language optimization at step 0 because the controlled-U gate
failed. A post-run deterministic audit found that this is first an input
contract defect: all 80 on/off pairs were magnitude-matched on the stored
40x40 grid, but only 7/80 remained matched after the U-tokenizer's actual
10x10 area pooling. Median peak mismatch was 21.5%, while mass remained matched
to numerical tolerance. Therefore the observed train/dev on-over-off fractions
(`0.867 / 0.550`) are not a valid controlled comparison and do not add a new R
failure claim. The exact terminal evidence and repair rule are frozen in
`configs/scenario_factory/amendments/20260901_stage2l_v11_identifiability_smoke_terminal_v1.json`.

That required replacement is now complete. The v11.1 dataset was constructed
and independently audited on the exact 10x10 consumer grid: all 80 on/off
pairs match peak, mass and support count exactly after pooling, while retaining
spatially distinct support and bitwise-shared R. Source v11 records were not
overwritten. The fresh CPU/model preflight passed, and exactly one separately
attested replacement was submitted as Slurm Job `1120954`.

Job `1120954` completed normally on `gpu4` with exit code 0 after 9 minutes 15
seconds and peak batch RSS `65,430,240 KiB`. It failed closed before language
optimization (`optimizer_steps=0`): train controlled-U ordering passed at
`0.866667`, but dev reached only `0.55` against the frozen `0.8` gate. All
shared-R, zero-U, spatial-distinction, matched-magnitude and off-path-low checks
passed. Held-out results localize the gap: Route147 and Route162 each passed
4/5 controls, while Route152 passed 1/5 and Route195 2/5; `CAM_FRONT` controls
passed 10/10, while `CAM_FRONT_LEFT`, `CAM_BACK_LEFT`, and no-effective-risk-
region cases passed 0/9 combined. This is valid evidence against the frozen
v10.1 R interface on held-out event/view patterns. It is not a language result,
because the language bridge was never optimized or evaluated.

The CPU-first objective preflight is now complete. The v12 primitive gives each
active view equal foreground/background region mass within a group, accepts
only R logits and soft R targets, preserves the calibrated target optimum, and
passed its tensor/unit-mass tests. It is not a complete view-balance repair.
Across the frozen 60 train groups, proposed mean foreground share changes only
from `65.76%` to `64.86%` on front; front-left and front-right actually fall to
`3.75% / 3.89%`, while back-right remains `0%`. The limiting variable is how
often independent events contain a view, which an objective cannot create.

A separate train-only candidate audit read no dev or locked-test result files
and used no model/U/QA outcomes for selection. It found no currently frozen
candidate able to add eligible geometry: Route177 has `2 < 3` keyframes,
Route201 is runtime-invalid, and Route208 is liveness-invalid. Therefore no GPU
R-only smoke is currently authorized. The next executable milestone is to
freeze a separate train-only coverage-repair candidate pool using static
route/event geometry before model outcomes, require accepted 3–5-keyframe
support with nonzero side/rear-view contribution, and rerun the CPU coverage
gate. Route152/195 stay held out and locked-test outcomes remain unread.

The subsequent coverage-repair protocol first exhausted existing raw assets.
Route196's previously frozen engineering-train identity was replayed only
offline through its original metadata at the unchanged `[-2,-1,0,+1,+2] s`
offsets. All five frames were geometry-eligible: front-right appears in `4/5`,
front-left in `3/5`, back-left in `2/5`, and back in `1/5`. It can therefore
add one independent front-right train event after separate Stage1/QA and human
review, but back-right remains absent. All aligned raw frames across the 14
accepted train event identities were then scanned using only ORION plan and
privileged actor geometry. Every event had exactly zero back-right-positive R
frames, so fixed-keyframe reselection cannot solve that view.

The preregistered outcome-blind fallback is Route167 in Town03,
`YieldToEmergencyVehicle`, selected from static XML because its emergency actor
approaches from behind. A new `train_coverage_repair` batch split requires no
published/model outcome, maps its event package to `qa_train_candidate`, and is
permanently ineligible as held-out evidence. Exactly one Route167 `clean_off`
collection was submitted as Job `1121242` and started on gpu4 at
`2026-09-01T13:45:09+08:00` with 2 CPU, 192 GB host memory and one A800; all UQ,
Stage2, governor, planning-response and corruption paths were disabled. The job
terminated `FAILED (1:0)` after 3 minutes 45 seconds before RouteIndexer loaded
any route: the submitter exported a relative `PILOT_ROUTE_DIR`, then the
evaluator changed working directory and could not open `route_167_hazard.xml`.
No control trace or event package exists. Job `1121242` is therefore an invalid
technical launch, not a model, safety or coverage result.

The submitter now canonicalizes the batch directory before export and has a
remote-passing relative-manifest regression test. The failed result directory
is preserved. A separate `retry1` batch reuses byte-identical hazard/no-hazard
XML and the same runtime contract under a new output path; it also corrects the
batch metadata lineage to the dedicated train-coverage-repair protocol. One
technical replacement was separately authorized and submitted as Job `1121244`
at `2026-09-01T13:57:52+08:00`. The absolute route path worked: Town03 and
RouteScenario 25378 loaded, ORION reached real control, and 144 trace records
were written. CARLA then blocked inside synchronous `world.tick`; the 301-second
progress watchdog terminated the job `FAILED (124:0)` after 15 minutes 41
seconds, with peak batch RSS `70,469,056 KiB`. This is an invalid runtime, not a
complete route or model/safety result, and the one-job replacement authorization
is exhausted.

Fifteen fully aligned six-view/meta frames were retained only for a CPU geometry
diagnostic. All 15 had valid geometry; front was positive in 15, back in two,
and back-right in zero. The two actor-grounded frames placed the emergency
vehicle in `CAM_BACK`, not `CAM_BACK_RIGHT`. Route167 is therefore retired as
the missing-view fallback independently of its later CARLA stall. No further
Route167 retry is authorized, the partial frames are not accepted into the
formal/training bank, and the aggregate coverage gate remains failed.

The next CPU inventory deliberately reconsidered whether a rear-right route was
the correct repair. It scanned all 150 manifests under the unchanged strict
filter, retained 28 clean-off non-held-out runs, and decomposed 24 geometry runs
covering 21 route identities. Independent verification passed every count,
frame-list, component-partition, aggregate and held-out-exclusion check. The
union label has route support on 1,103 front frame-view positives but only 15
front-left and 11 front-right, with none behind. Conflict-actor support is
distributed differently: 179 front, 90 front-left, 29 front-right, 263 back,
128 back-left and zero back-right frame-view positives. At the route level,
front-right actor support appears only in Routes157/177/192/194; back-right
appears in none. Route202's nine apparent front-right union positives are all
route-only and provide no conflict-actor binding evidence.

This changes the diagnosis. The front-dominant union objective is partly a
route-corridor volume effect: 951/1,130 front union positives are route-only,
whereas every back and back-left positive is actor-grounded. Only one
front-left and one front-right frame-view positive contain route and actor
support together. `CAM_BACK_RIGHT` is therefore an unsupported region of the
current frozen inventory, not a universal release gate that justifies a custom
scenario by itself. The immediate custom rear-right collection is retired.
The next executable work is a CPU-only factorized-R target/interface preflight
that exposes route-corridor and conflict-actor components separately, retains
their union only as a derived diagnostic, and declares unsupported views. It
must not read dev/test outcomes or authorize training.

That v12.1 CPU preflight is now complete. A shared contextual trunk with
separate route and actor logits passed shape, serialization, soft-target
stationarity, component-independence, empty-component negative-anchor and
finite-gradient tests. Across the exact 80 frozen groups, stored union equals
`max(route, actor)`, both components rebuild from original metadata, and their
10x10 consumer targets match with maximum error `0.0`. There are 114 active
sample-components and 46 empty sample-components; the latter retain explicit
negative supervision instead of disappearing from the loss.

The prospectively frozen identifiability rule requires at least two positive
train events and one positive dev event per component/view. It supports only
route/front (`12/4` train/dev events), plus actor/front (`6/3`), actor/front-left
(`4/1`), actor/front-right (`2/1`), actor/back (`11/2`) and actor/back-left
(`5/2`). All other route views and actor/back-right remain outside the release
average. The frozen single-union R's dev foreground recall is `0.6744` for
route/front, but only `0.4538` for actor/front, `0` for actor/front-left and
front-right, `0.3833` for actor/back, and `0.0089` for actor/back-left. Thus the
existing interface learned route structure much more reliably than held-out
conflict-actor structure.

This result allows preparation—not submission—of one bounded factorized-R-only
engineering smoke on the same 60 train/20 dev groups. It must exclude U,
tokenizer, language, trajectory and control paths, report both components per
event/view, exclude unsupported cells from release averages, and label the
front-right result fragile because it has only two train events and one dev
event. A separate frozen implementation preflight and launch record are still
required.

That separate implementation preflight is now complete. It loaded all 80
factorized targets, verified the exact input hashes, and showed bitwise identity
between the v10.1 single-R probability and both duplicated component branches
plus their derived maximum before training. Fourteen groups confirm that
pooling and component maximum do not commute (maximum absolute difference
`0.273959`); both union conventions are therefore report-only and the loss is
strictly component-factorized. The related remote suite passed 14 tests, and
the fail-closed submitter/attester contract passed shell and lineage checks.

Exactly one separately authorized 40-step R-only run was submitted as Slurm
Job `1121553` at `2026-09-01T15:27:29+08:00`. It requests one A800, two CPU
cores and 192 GB host memory, excludes gpu5, allows no automatic retry, and is
bound to the frozen preflight, trainer and protocol hashes. It started on gpu1
at `2026-09-01T16:00:10+08:00` and completed normally after 44 minutes 51
seconds with exit code `0:0`. All 40 optimizer steps were present and finite.
The frozen independent validator verified both checkpoints, all 80 component
maps at both milestones, exact lineage and disabled-path locks; component
maximum and derived-union errors are exactly `0.0`.

The original seven-gate verdict is `held_out_factorized_r_transfer_failed`.
Train supported-view macro recall (`0.8329`), dev route/front (`0.7358`), dev
actor/front (`0.5074`) and background FPR pass. Dev actor non-front macro
recall is only `0.2531 < 0.35`; front-right remains `0`, front-left is
`0.0494 < 0.05`, and the absolute improvement over the frozen baseline is
`0.1551 < 0.25`. Thus factorization is numerically healthy and train-learnable
but does not close held-out non-front conflict-actor transfer. No extra epoch,
threshold change, retry, language run, Stage2-P, closed loop or locked-test
access is unlocked by the terminal audit. The terminal record is
`configs/scenario_factory/amendments/20260901_stage2l_v121_factorized_r_smoke_terminal_v1.json`.

While the same job still had no log or training output, the terminal audit was
frozen prospectively in
`configs/scenario_factory/stage2l_v12_1_factorized_r_terminal_audit_v1.json`.
Its independent validator passed 18 related remote tests. It recomputes all
seven engineering checks without calling the trainer gate function, verifies
the complete finite step history, both milestone checkpoint contracts, every
one of the 80 component-map artifacts, the exact frozen hashes and all disabled
paths, and rejects unregistered output files. Threshold changes, extra epochs,
automatic retry and downstream unlocks are explicitly prohibited after the
result is observed.

Only after the factorized-R CPU preflight defines component targets, loss
normalization and supported-view gates may one separately authorize a bounded
R-only engineering smoke. It must preserve the held-out event identities,
v11.1 controls and `0.8` gate and report route and actor components per event
and view. No U-token, language, trajectory or control optimization is
authorized in that run; additional v10/v10.1 epochs alone are not a repair.

Authoritative v11.1 results:

- `configs/scenario_factory/amendments/20260901_stage2l_v111_consumer_grid_preflight_result_v1.json`;
- `configs/scenario_factory/amendments/20260901_stage2l_v111_identifiability_model_preflight_result_v1.json`;
- `configs/scenario_factory/amendments/20260901_stage2l_v111_identifiability_smoke_terminal_v1.json`;
- `configs/scenario_factory/amendments/20260901_stage2l_v111_r_binding_cpu_audit_v1.json`.
- `configs/scenario_factory/amendments/20260901_stage2l_v12_objective_and_train_coverage_preflight_v1.json`.
- `configs/scenario_factory/stage2l_v12_train_coverage_repair_protocol_v1.json`;
- `configs/scenario_factory/amendments/20260901_stage2l_v12_route167_coverage_collection_launch_v1.json`.
- `configs/scenario_factory/amendments/20260901_stage2l_v12_route167_coverage_collection_submission_v1.json`.
- `configs/scenario_factory/amendments/20260901_stage2l_v12_route167_relative_path_failure_retry1_authorization_v1.json`.
- `configs/scenario_factory/amendments/20260901_stage2l_v12_route167_retry1_submission_v1.json`.
- `configs/scenario_factory/amendments/20260901_stage2l_v12_route167_retry1_terminal_and_partial_coverage_v1.json`.
- `configs/scenario_factory/stage2l_v12_existing_raw_coverage_inventory_protocol_v1.json`.
- `configs/scenario_factory/stage2l_v12_existing_raw_actor_support_inventory_amendment_v1.json`.
- `configs/scenario_factory/amendments/20260901_stage2l_v12_existing_raw_actor_support_inventory_result_v1.json`.
- `configs/scenario_factory/stage2l_v12_1_factorized_r_cpu_preflight_v1.json`.
- `configs/scenario_factory/amendments/20260901_stage2l_v121_factorized_r_cpu_preflight_result_v1.json`.
- `configs/scenario_factory/amendments/20260901_research_gap_reassessment_v1.json`.
- `configs/scenario_factory/amendments/20260901_vertical_slice_soft_gate_progression_v1.json`.

Closed-loop failure-induction discovery may continue independently, but it
must not change Stage-2L labels after model outcomes are seen.

## 8. Claim boundary

Current evidence supports a tokenizer that preserves the frozen pilot-U
representation, exact consumer-grid counterfactual controls, exact
reproducibility and component decomposition of the current geometry
supervision, and partial/front-dominant learnability of a spatial relevance
head. It does not establish semantic correctness of that weak target,
a held-out-generalizing task-relevance model, learned language use of U,
semantic UQ correctness, risk-aware planning, closed-loop safety improvement,
or a 200-route benchmark claim.

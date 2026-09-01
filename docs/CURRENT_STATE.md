# UQ-ORION current state

Last audited: 2026-09-01 (Asia/Shanghai)

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

The most important scientific gap is now more precise than a missing tensor
connection: **the marginal contribution of U is not identifiable in the
current results**. The dense R target is owned by route-corridor and visible
conflict-actor geometry, so a model can learn substantial R structure from
visual/route context without using U. The v10.1 no-U result demonstrates that
this bypass is real. Conversely, the frozen Stage-1 adapter is only a
diagnostic evidence-loss proxy and has not passed its independent native gate.

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

Stage-2P has not started. Current Stage-2L runs use zero trajectory and control
loss. No current-mainline checkpoint has closed-loop authority, and no learned
UQ-aware ORION result establishes collision, TTC, traffic-rule, or route-
completion improvement. Conditional diffusion remains a later decoder
ablation; changing the decoder cannot repair the missing semantic bridge.

## 7. Next executable milestone

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

The bounded release gates remain held-out R foreground recall/background FPR,
on-path-over-off-path K ordering, per-variant answer preference,
zero/irrelevant-U false-conservatism, and map/structured-field/text
consistency. It must also include a no-U ablation and report per-event and
per-view results rather than only aggregate metrics. If R remains high on train
and low on dev, do not add epochs or model capacity: audit support labels and
coordinate binding, then expand independent-event coverage. If controlled U
does not change K/QA correctly under a shared R, repair the semantic bridge
before Stage-2P. If controlled U passes but learned U fails an independent
native/sensor gate, Stage 1—not Stage-2L—is the bottleneck.

The machine-readable preflight contract for this distinction is
`configs/scenario_factory/stage2l_v11_identifiable_factorized_bridge_v1.json`.

The route/ego and matched-U dataset checks above are now complete. The remaining
pre-GPU blocker is model-output-side implementation and testing: the v11
forward path must compute R once, reuse the exact logits across all U variants,
derive K, score the matched QA answers, and execute the no-U ablation. A new
hash-bound launch amendment is still required after that implementation passes
CPU tests; the dataset result does not authorize a training job by itself.

Closed-loop failure-induction discovery may continue independently, but it
must not change Stage-2L labels after model outcomes are seen.

## 8. Claim boundary

Current evidence supports a task-agnostic U representation and partial
learnability of a spatial relevance head. It does not yet support semantic UQ
correctness, a held-out-generalizing task-relevance model, risk-aware planning,
closed-loop safety improvement, or a 200-route benchmark claim.

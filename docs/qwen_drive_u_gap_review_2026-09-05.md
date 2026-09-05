# Qwen-Drive transition: U, bridge, fine-tuning and control gap review

Date: 2026-09-05 (Asia/Shanghai)

## Decision

Keep the Bench2Drive/CARLA shell and the new Qwen sidecar, but do not reuse the
old EVAViT-dependent U checkpoint as if it were backbone-independent.

The migration has two different bridges which must not be conflated:

1. The **runtime bridge** between CARLA Python 3.8 and Qwen Python 3.10 remains
   necessary until those environments can be unified. It transports images,
   ego state, commands and trajectories; it has no U semantics.
2. The old **Orion feature/language bridge** must not be transplanted. If U is
   retained as an explicit non-RGB modality, Qwen still needs a small
   Qwen-native U interface or an explicit textual serialization. Native
   multimodality means that RGB and text already share one model; it does not
   mean an unseen `[view, y, x, time, component]` tensor has a defined meaning.

The recommended scientific path is therefore:

```text
Bench2Drive RGB/history
        |
        v
frozen Qwen visual pathway
        |
        +--> Qwen-native Stage-1 U probe/adapter -- U map/schema
        |                                      (retrained)
        v
Qwen VLM cache + explicit U condition
        |
        v
Qwen Planning Expert -- 50 poses -- trajectory/control adapter -- CARLA
```

## 1. What in the old U is backbone-dependent?

The old Stage-1 adapter is materially EVAViT-dependent, not merely associated
with EVAViT by configuration:

- Orion's image backbone emits six-view `1024`-wide EVAViT grids. The active
  extractor reshapes them to `[B,V,H,W,D]`; the real pipeline uses
  `[6,40,40,1024]` per frame.
- The paired Stage-1 target is itself computed from clean/degraded EVAViT
  features: persistent cosine-direction change, RMS-magnitude change and
  temporal-cosine change.
- The adapter's `LayerNorm`, current/previous projections, calibration
  quantiles and learned weights are fitted in that coordinate system.
- Downstream Orion alignment was trained for Orion's six-view tokens and a
  `4096`-wide language stream. The released Qwen config has a `1024`-wide
  vision encoder projected to a `2560`-wide VLM and a separate `1024`-wide
  Planning Expert, with a different cache/planner interface.

The identical nominal encoder width (`1024`) does not establish compatibility: a
channel in one frozen backbone has no agreed coordinate correspondence to a
channel in another, and Qwen also changes image tokenization, spatial merging,
temporal serialization and camera coverage.

### Reuse decision

| Artifact or contract | Qwen reuse decision | Reason |
| --- | --- | --- |
| U meaning and three-component schema | Reuse | The scientific question remains observation evidence loss. |
| Route/event splits, corruption schedules, raw RGB and audit metrics | Reuse | They do not depend on EVAViT coordinates. |
| Privileged localization masks | Auxiliary reuse only | They were never uncertainty ground truth. |
| EVAViT feature cache | Do not reuse | Wrong backbone and feature distribution; the large generated tensor has been deleted. |
| Paired feature targets | Recompute | They explicitly measure differences in the frozen backbone's representation. |
| Stage-1 observation adapter checkpoint | Retrain | Its inputs, normalization and weights are EVAViT-specific. |
| Task-agnostic U tokenizer | Warm-start at most | It consumes a common U-map schema, but it was fitted to the old map distribution. Recalibration is required. |
| Tokenizer language projection / Orion LoRA / FiLM / scalar adapters | Do not reuse | They target Orion's `4096`-wide stream and Orion-specific injection points. |
| Stage-2P trajectory-response checkpoint | Do not reuse directly | It modifies Orion's six-point trajectory context, not Qwen's 50-point Planning Expert. |

Therefore the answer to “does U need retraining?” is **yes for the Stage-1 U
estimator and every consumer-facing adapter**. Only schema-level and
data/protocol-level assets transfer safely. A tokenizer warm-start is an
experiment, not a compatibility claim.

## 2. Does native multimodality remove the bridge?

Not by itself. There are three distinct options:

### A. Text-U baseline: no continuous semantic bridge

Serialize the six U fields into ordinary prompt text and use Qwen's native
multimodal prompt/cache. This is the fastest capacity diagnostic and matches
the concern that Orion's language backbone could not reliably read U.

The completed text-only diagnostic supports this direction but does not pass
it: Qwen achieved `69.4%` nonzero field accuracy excluding presence and
`42.19%` changed-field exact response, versus `6.4%` and `0%` for the v15
Orion LoRA. It still missed the frozen `80% / 70%` gates and saw no images,
continuous U tokens, planner, or CARLA. It proves stronger textual U capacity,
not multimodal U grounding.

### B. Qwen predicts U directly from RGB: no external U modality

Fine-tune the VLM with matched multi-view/multi-frame U questions and answers.
This is architecturally simple, but it merges observation-U extraction, task
relevance and language behavior inside one model. It is useful as a baseline,
but is weaker for the causal claim because it is harder to prove that a plan
changed because of the intended U rather than an image or scenario shortcut.

### C. Explicit Qwen-native U condition: recommended mainline

Train a U adapter on a fixed Qwen visual feature tap, keep its output as the
versioned spatial U schema, and expose that U explicitly to the Qwen consumer.
Start with the least invasive consumer interface:

1. use textual six-field U as the first end-to-end baseline;
2. add a learned U condition to the Planning Expert only after the matched
   counterfactual test shows Qwen consumes the text correctly;
3. preserve spatial information either through compact U tokens projected
   into the expert's scene key/value memory or through trajectory-query-to-U
   cross-attention; a single global scalar is not sufficient.

Qwen's released Planning Expert already reads cached keys and values from the
VLM's eight grouped-query attention layers. A Qwen-native U module can add a
small trainable U memory to that interface without transplanting Orion's
4096-wide tokens. This is still a bridge in the useful sense: it assigns a
meaning and geometry to a new modality. It is not an EVAViT-to-Qwen feature
converter.

## 3. Training gap and proposed curriculum

The official release documents four training stages: perception-head-only;
joint vision encoder/VLM perception and VQA; frozen-VLM Planning Expert flow
matching; and frozen-VLM planner reward optimization. The public checkout at
commit `28091c1532e869bc7aee91fc0aef6b3e6fd0b2e0` describes itself as inference
code and contains inference/evaluation entry points but no official training
entry point. Reproducing or adapting the training recipe therefore requires a
local trainer rather than a hidden command from the release.

Use the following bounded curriculum:

### Q0. Freeze the interface and attribution controls

- Keep camera order, timestamps, coordinates, command mapping and trajectory
  resampling fixed.
- Add an oracle/recorded-future trajectory through the same 50-to-6-point
  adapter and PID. If that cannot complete the route, do not blame Qwen.
- Evaluate Qwen open-loop on recorded Bench2Drive sequences before training.

### Q1. Retrain Qwen-native Stage-1 U

- Select one fixed Qwen visual feature tap and record exact view/frame/token
  lineage.
- Re-extract clean/degraded pairs from raw RGB and recompute Qwen-space paired
  targets; do not regenerate an EVAViT cache.
- Train only the small U probe/adapter first. Preserve route-disjoint and
  held-out-family/native-weather gates.
- Refit calibration and verify zero-U, localization, onset/recovery and
  counterfactual equivariance.

### Q2. Prove multimodal U consumption before planning

- Hold RGB, route and ego state fixed while changing only U.
- Train a small U interface plus Qwen LoRA, or use text-U first; keep the
  Planning Expert frozen.
- Require per-field balanced accuracy, changed-field response, unchanged-field
  invariance, zero-U identity and spatial/view grounding.
- Compare `(RGB only)`, `(RGB + true U)`, `(RGB + zero U)`, `(RGB + shuffled
  U)` and `(text oracle U)` on the same examples.

### Q3. Train trajectory response

- Freeze the validated U extractor initially.
- Fine-tune the Planning Expert or a small identity-initialized U-conditioning
  module using the released flow-matching endpoint objective plus temporal
  smoothness, with matched clean/off-path/shuffled-U negatives.
- Add collision, drivable-area, progress and prolonged-stop terms only at the
  planner stage; stop their gradients before Stage 1.
- Begin with SFT/DAgger-style expert trajectories. Closed-loop reward tuning is
  later and must report the safety/progress trade-off.

### Q4. Matched closed loop

- First rerun task 203 with the same controller and each model arm.
- Then use a small preregistered route set; one failed route is not a model
  comparison.
- Report route completion, collisions, lane departure, blockage, comfort,
  progress, inference latency, U calibration and U ablations together.

## 4. Attribution of the current control failure

The `52.84%` completion, pedestrian collision, vehicle collision, `30.22%`
outside-route-lane distance and final `Failed - Agent got blocked` status are
**our run**, not an official Qwen-Drive Bench2Drive result.

More precisely, Job `1155519` is a valid run of the official Bench2Drive
evaluator on task 203, but the evaluated agent is our integration:

- released Qwen-Drive-1.0-4B VLM plus released `planner-sft`;
- direct planning, one deterministic sample;
- three Bench2Drive/CARLA cameras with our resolution/history compression;
- our command, coordinate and 50-point-to-six-point adapters;
- Orion's existing PID controller and our speed/staleness limits.

So “official” applies to the Bench2Drive scoring machinery, not to the score as
an official Qwen result. The Qwen paper does not report Bench2Drive. Its
published closed-loop result is on 916 AlpaSim scenarios, and the reported SFT
arm uses reasoning. Our run uses a different simulator/domain, SFT direct mode,
one sample and a foreign controller.

This run establishes that the interface executes end to end: 151 Qwen plans,
1503 control frames and zero inference exceptions. It does **not** isolate the
cause of the driving failure. The unresolved causes include:

1. Qwen-to-CARLA camera/domain and temporal-history shift;
2. direct SFT one-sample policy quality versus reasoning/RL variants;
3. route-command serialization and coordinate/time resampling;
4. PID tuning for a trajectory distribution it was not designed on;
5. true Qwen perception/planning errors and recovery weakness.

The official Qwen paper itself shows that the model is not a solved
closed-loop policy: on AlpaSim it reports nonzero close-encounter and off-road
rates and a safety/progress trade-off. That context does not turn our task-203
failure into their result.

## 5. Next executable gates

Run these in order; each isolates a different missing fact:

1. **Controller oracle gate:** recorded/expert future through our trajectory
   adapter and PID on task 203.
2. **Qwen open-loop gate:** Qwen prediction versus recorded future before PID,
   split by straight/turn and hazard phase.
3. **Mode gate:** SFT direct versus SFT reasoning; add RL reasoning only after
   its planner weight is installed and the comparison is frozen.
4. **Native U gate:** Qwen-feature Stage-1 probe versus the old U protocol.
5. **Causal U consumer gate:** matched RGB with true/zero/shuffled/text U.
6. **Small closed-loop matrix:** only after gates 1--5 identify whether the
   next repair belongs to control, domain adaptation, U extraction or U use.

## Sources and evidence

- Qwen-Drive official repository: <https://github.com/QwenLM/Qwen-Drive-1.0>
- Official model architecture: <https://github.com/QwenLM/Qwen-Drive-1.0/blob/main/docs/model.md>
- Official technical report: <https://arxiv.org/html/2609.00111v1>
- Official public evaluation protocol: <https://github.com/QwenLM/Qwen-Drive-1.0/blob/main/docs/evaluation.md>
- Local architecture status: `docs/CURRENT_STATE.md`
- Local Qwen/Orion text diagnostic:
  `docs/qwen_drive_orion_backbone_diagnostic_2026-09-05.md`
- Local Bench2Drive integration and result lineage:
  `docs/qwen_drive_b2d_integration_v1.md`
- Server evaluator record:
  `/public/share/lidachuan/orion_assets/qwen_drive_b2d_smokes/closedloop_route203_flash_gpu4_v4/eval_qwen_drive_traj_203.json`
- Server control trace:
  `/public/share/lidachuan/orion_assets/qwen_drive_b2d_smokes/closedloop_route203_flash_gpu4_v4/records_qwen_drive_traj_203/qwen_drive_control_trace.jsonl`

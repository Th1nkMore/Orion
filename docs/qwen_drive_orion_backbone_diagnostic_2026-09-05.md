# Qwen-Drive / ORION backbone diagnostic

## Question

Does the current ORION language backbone materially limit its ability to read an exact textual description of the six uncertainty fields, before considering vision, U-token projection, planning, or closed-loop CARLA behavior?

## Frozen diagnostic

- Input: the same 120 states from the completed ORION v15.2 text-oracle report (20 groups, six counterfactual variants per group).
- Prompt: the exact authoritative natural-language U description used by v15.2.
- Decode: length-normalized NLL over every legal canonical answer.
- Qwen model: `Qwen/Qwen-Drive-1.0-4B`, revision `9d2eb187c2fe03d0e30fd58c0638058980ee6267`.
- Training: none.
- Images, continuous U tokens, Planning Expert weights, trajectory and CARLA: none.
- Predeclared pass gates: nonzero accuracy excluding presence at least 0.80 and changed-field exact response at least 0.70.

Authoritative remote report:

`/public/share/lidachuan/orion_assets/scenario_factory/stage2l_smokes/v15_3_qwen_drive_text_oracle_120dev_v1/report.json`

Report SHA-256: `1a5a6aa3f1336f1b45ef30c6922c38e854ae2676f4364cb56ce45e77622c875b`

Valid Slurm job: `1155328` (`COMPLETED`, exit code 0, elapsed 00:09:29).

Job `1155299` is explicitly invalidated. It omitted the official flash-linear-attention backend, stopped at 20/120 under the pathological Transformers fallback, was cancelled, and wrote no report.

## Results

| Model | Nonzero U accuracy, excluding presence | Changed-field exact response |
| --- | ---: | ---: |
| Original ORION | 6.2% | 0.0% |
| v15 LoRA | 6.4% | 0.0% |
| Qwen-Drive 1.0 4B | **69.4%** | **42.19%** |

Qwen's overall field accuracy was 70.42% (507/720). It did not pass the frozen 80% / 70% sufficiency gates.

| Field | All 120 states | Nonzero 100 states | Balanced accuracy |
| --- | ---: | ---: | ---: |
| `U_PRESENT` | 100.0% | 100.0% | 100.0% |
| `U_VIEW` | 52.5% | 63.0% | 61.90% |
| `U_REGION` | 79.17% | **95.0%** | 86.15% |
| `U_LEVEL` | **100.0%** | **100.0%** | **100.0%** |
| `U_TREND` | 69.17% | 63.0% | 66.67% |
| `U_COMPONENT` | 21.67% | 26.0% | 40.0% |

The main residual failures are structured, not random:

- Every `rising` trend was scored as `falling` (37 cases).
- The component answer collapsed mostly to `transient_inconsistency` (110/120 predictions).
- All zero-U `none` answers for view, region and component were missed, although `U_PRESENT=no` and `U_LEVEL=none` were read correctly.
- Nonzero spatial region is nearly solved (95/100), while camera view remains only partial (63/100).

## Interpretation

The user's backbone concern is supported as a causal contributor: under the same literal text-oracle contract, Qwen-Drive improves nonzero field accuracy by 63 percentage points over the v15 LoRA and changes its output when counterfactual facts change, whereas both ORION controls are nearly constant.

The stronger claim, “replace the backbone and the U problem is solved,” is not supported. Qwen remains badly calibrated for some canonical labels, especially component, temporal direction, camera view and zero-U `none`. This diagnostic also says nothing yet about whether Qwen can read visual U evidence or convert it into safer trajectories.

## Integration decision

### Recommended: keep the Bench2Drive shell, replace the model behind it

Create a new `QwenDriveBench2DriveAgent` while reusing the existing CARLA launch, route handling, corruption injection, telemetry, artifact writing, safety metrics and PID/control plumbing. This is cleaner than transplanting the Qwen backbone into ORION.

The adapter should:

1. Map Bench2Drive front/front-left/front-right cameras into Qwen's three-view, four-timestamp scene format (1.5 s history at 2 Hz plus current frame).
2. Maintain the required 16 ego history poses, velocities and accelerations at 10 Hz.
3. Map Bench2Drive route commands to Qwen's `nav_command` and four-way driving-command vector.
4. Run the SFT Planning Expert first in direct mode with one deterministic sample. Official best-of-six selection uses ground truth and is therefore not valid online.
5. Convert Qwen's 50 `(x_forward, y_left, heading)` poses at 10 Hz to ORION PID coordinates and initially sample 0.5–3.0 s at indices 4, 9, 14, 19, 24 and 29. The expected ORION PID mapping is `(x_orion, y_orion) = (-y_left, x_forward)` and must be verified with straight/left/right synthetic tests before CARLA.
6. Run offline replay and a single Bench2Drive route before any 220-route claim.

This preserves the expensive closed-loop engineering without retaining ORION's weak language path.

### Optional low-cost bridge: Qwen as a structured U sidecar

Use Qwen only to turn U evidence into calibrated field distributions or a compact embedding, then project that output into the existing Stage-2 planning context. This is useful for attribution experiments but is not the preferred final architecture because the ORION language model remains the downstream consumer.

### Not recommended now: literal backbone transplant inside ORION

ORION injects detector/map/U queries as 4096-dimensional custom LLaVA visual embeddings and extracts one 4096-dimensional waypoint-token state for its six-point planner. Qwen uses a 2560-dimensional native multimodal stream with multimodal RoPE, and its Planning Expert reads VLM key/value caches to generate 50 poses. A transplant therefore needs new input projection, token/position handling, output projection and planner retraining; it cannot reuse both checkpoints unchanged.

## Next gate

Implement only the scene/trajectory adapter and validate it on recorded Bench2Drive frames. Required gates before closed loop:

- camera order and timestamps verified;
- ego-frame sign and 10 Hz to 2 Hz resampling unit-tested;
- straight/left/right trajectory-to-PID smoke tests pass;
- no ground-truth trajectory is used for online sample selection;
- inference latency and GPU memory are measured on A800;
- clean replay does not regress gross route-command compliance.

Only after those gates should a one-route closed-loop smoke be run. If the native Qwen planner fails because of CARLA domain shift, fine-tune its Planning Expert or a small output adapter on Bench2Drive; do not immediately reopen ORION's full three-stage training stack.

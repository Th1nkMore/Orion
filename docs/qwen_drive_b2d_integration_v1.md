# Qwen-Drive 1.0 → Bench2Drive closed-loop integration v1

Date: 2026-09-05 (Asia/Shanghai)

## Outcome

The integration uses a new Bench2Drive agent and does not transplant Qwen into
Orion's checkpoint graph.  Qwen-Drive's Planning Expert consumes the Qwen VLM
attention cache, while Orion's planning path consumes Orion-specific 4096-wide
features and heads; treating those as checkpoint-compatible would be an
unvalidated architectural change.

The v1 runtime is therefore:

```text
Bench2Drive/CARLA (Python 3.8)
  3 RGB cameras + bounded compressed history + ego/route state
                         |
                         | local Unix-socket RPC
                         v
Qwen sidecar (Python 3.10, Torch 2.8)
  Qwen-Drive-1.0-4B VLM + planner-sft, direct planning, one sample
                         |
                         v
  50 x (x_forward, y_left, heading), 10 Hz, 5 seconds
                         |
                         v
coordinate/time adapter -> existing PID -> CARLA VehicleControl
```

This isolates the incompatible Python/PyTorch stacks and means the CARLA
process never imports Torch, MMCV, Orion, EVA, the six-view detection/map
heads, the old LLM, BEV input, or the legacy UQ feature branches.
The loaded sidecar is reused across routes inside one evaluator process and is
closed at process exit, avoiding a multi-minute model reload for every route.

## Frozen input/output contract

- Cameras, in model order: `CAM_FRONT`, `CAM_FRONT_LEFT`,
  `CAM_FRONT_RIGHT`.
- Images: four frames per camera at 2 Hz, oldest to current. Only three
  losslessly PNG-encoded history frames per camera are retained.
- Images remain at the CARLA sensor resolution (`1600x900`) during transport.
  The released Qwen processor owns the history/current resize and applies the
  checkpoint's official pixel budgets.
- Ego history: 16 poses/velocities/accelerations at 10 Hz.  CARLA's
  left-handed `(forward,right)` convention is converted to Qwen's
  `(forward,left)` convention; current pose is forced to `(0,0,0)`.
- Commands: Bench2Drive left/change-left, straight/lane-follow, and
  right/change-right map to Qwen's four-way driving command and three-way
  navigation command.
- Planning: SFT expert, `direct_planning`, `num_samples=1`, seed `42`.  No
  ground-truth best-of-N selection is possible in the agent.
- Control: Qwen's 10 Hz trajectory is converted to world coordinates at the
  inference pose, age-adjusted, resampled to the six 2 Hz points expected by
  the existing PID, and expressed as `(right,forward)`.
- Failure behavior: no fresh plan means full brake.  A previous plan is reused
  for at most one configured second.

## Runtime/storage boundary

"Remove old features" is implemented in two places.  At runtime, the new agent
has only three RGB sensors and two tiny bounded buffers; it does not load the
original six camera pipeline or any Orion intermediate feature tensor.  On
disk, the user-authorized generated counterfactual tensor below was deleted
after confirming no active job or open handle used it.  Source/hash manifests
and a deletion receipt were retained.

The server has one separate, generated 70,837,587,338-byte counterfactual
feature tensor at:

`/public/share/lidachuan/orion_assets/observation_uq_v3/runs/counterfactual_evidence_features_windowcycle_seed20260827_r1/counterfactual_evidence_features.pt`

It was deleted on 2026-09-05 and freed 70,837,587,338 bytes, moving `/public`
from 98% used (9.6 GiB free) to 79% used (76 GiB free).  It is not recoverable
from trash; rerun `scripts/submit_counterfactual_evidence_extraction_v2.sh`
with a new `RUN_ID` to regenerate it.  The receipt is
`results/observation_uq_v3/counterfactual_evidence_features_windowcycle_seed20260827_r1.deleted_artifact.json`.

## Files and launch

- Agent: `team_code/qwen_drive_b2d_agent.py`
- Bridge/sidecar: `uq_estimator/qwen_drive_bridge.py`
- Config: `configs/qwen_drive_b2d_agent_v1.json`
- Real-model bridge smoke: `scripts/smoke_qwen_drive_b2d_bridge.py`
- Reproducible smoke submission: `scripts/submit_qwen_drive_b2d_bridge_smoke.sh`
- Reproducible CUDA extension build:
  `scripts/build_qwen_drive_cuda_extensions.sh` and
  `scripts/submit_qwen_drive_cuda_extensions.sh`
- One-route closed-loop wrapper: `scripts/run_qwen_drive_b2d_smoke.sh`
- One-route Slurm submission:
  `scripts/submit_qwen_drive_b2d_closedloop_smoke.sh`
- Tests: `tests/test_qwen_drive_bridge.py`

The server-side SFT planner was downloaded at the same frozen model revision as
the VLM and verified as:

```text
planner-sft/config.json
  sha256 4dbfc812ac2f4b5e6995220be6330f97f8f1c5827a8d5dbf061c50cf5e66d798
planner-sft/model.safetensors
  bytes  2079739550
  sha256 dcf5989ed292799e77f539b21e5d8b701566676c1c5d13038135e97f96e2d7b4
```

Local verification currently covers config locks, import isolation, coordinate
conventions, history padding, command mapping, request shapes, time resampling,
left-turn sign, and launcher syntax.  The same bridge and agent import were
also checked in the server's Python 3.8 CARLA environment with `torch_loaded
False`.

The cluster's prebuilt FlashAttention wheel required GLIBC 2.32, while the
Qwen runtime is deliberately held to GLIBC 2.28 so it remains compatible with
the CARLA host.  Job `1155362` therefore rebuilt `causal-conv1d 1.6.2.post1`
and `flash-attn 2.8.3` from their official source distributions against the
runtime's Torch 2.8/CUDA 12 ABI.  The installed, reusable wheels are:

```text
causal_conv1d-1.6.2.post1-cp310-cp310-linux_x86_64.whl
  sha256 c0c5ac58cd47d1aa8fb6184e509075dd047dbd76062e87d48b602646cd162620
flash_attn-2.8.3-cp310-cp310-linux_x86_64.whl
  sha256 a69da685a7d6f6ebf7aa0622ce207564bfccf5ffdb617ab6dc0b11c3fe0ea6c9
```

Real-model bridge smoke Job `1155364` then completed two consecutive forwards
through the same loaded sidecar.  It produced a finite, continuous `50x3`
trajectory with forward displacement from 0 to 14.99 m and a small left
offset from 0.0015 to 0.185 m.  The cold forward, including first-use kernel
compilation, took 86.67 s; the warm forward took 3.80 s.  CUDA peak allocation
was 11,877 MiB and peak reservation was 12,764 MiB on an A800 80 GB.  The
machine-readable evidence is
`/public/share/lidachuan/orion_assets/qwen_drive_b2d_smokes/bridge_flash_source_v1/report.json`
(SHA-256 `b94ccf2f8ad25c89095810d6d830eedd5f0940b6bdd71bb30f6a4ade554bc3cc`).

These measurements establish that the bridge performs real Qwen inference,
but also expose a throughput limit: 3.80 s per warm plan is slower than the
requested 2 Hz planning cadence.  Synchronous CARLA can still exercise the
protocol because simulation time pauses during inference, but wall-clock route
evaluation will be substantially slower than real time.

The first closed-loop submission, Job `1155376`, was rejected by the Vulkan
preflight on `gpu5`; no CARLA route or Qwen inference ran.  Job `1155377` was
therefore pinned to the previously validated `gpu4`, passed the Vulkan check,
and connected to CARLA, but Bench2Drive task 0 requests `Town12`, which is not
present in the installed CARLA map bundle.  Rather than consume storage by
downloading all additional maps for a smoke test, the launcher now defaults to
the short task 203 (`Town04`, approximately 66.5 m) and keeps both failed output
roots as infrastructure evidence.

Job `1155378` reached live Qwen inference and exposed a NumPy wire-compatibility
bug: the Qwen sidecar uses NumPy 2 while CARLA uses NumPy 1.24, so an ndarray
pickled by the former referenced `numpy._core`, which the latter cannot import.
The RPC now sends only Python lists, bytes, strings, and scalars across the
environment boundary and recreates float32 arrays locally.  The CARLA Python
3.8 environment verifies this wire payload without importing Torch.

## Our one-route run scored by the official Bench2Drive evaluator

After that fix, Job `1155519` completed normally with evaluator exit code 0 on
Bench2Drive task 203, Town04, scenario `PedestrianCrossing_1`, weather 23.
This is **our integration result, not an official Qwen-Drive benchmark
result**.  The official component here is the Bench2Drive evaluator.  The
evaluated system combines released Qwen weights with our camera/history,
command, coordinate, resampling and Orion-PID adapters.  It is a successful
end-to-end integration test but a failed driving-quality result:

```text
Qwen planning calls       151
control trace frames     1503
inference errors            0
maximum speed          5.244 m/s
route completion       52.84 %
infraction penalty     0.209334
driving score          11.061207
route status           Failed - Agent got blocked
route system/game time 630.788 s / 75.15 s (0.119x)
```

The route registered one pedestrian collision, one vehicle collision, 30.22%
outside-route-lane distance, and eventual blockage.  It did not run a red
light or stop sign, and it did not fail from an agent exception, route timeout,
or scenario timeout.  The trace contains a valid Qwen trajectory every ten
simulation steps and no inference error.  Its first in-route forward was the
cold outlier; the final 20 warm planning calls averaged about 1.73 s.  The
official evaluator record and full control trace are:

```text
/public/share/lidachuan/orion_assets/qwen_drive_b2d_smokes/closedloop_route203_flash_gpu4_v4/eval_qwen_drive_traj_203.json
  sha256 46fa7d5e865aafed4013e1917438d4035aa9c383b8c7252860aa29aab9dfe2eb
/public/share/lidachuan/orion_assets/qwen_drive_b2d_smokes/closedloop_route203_flash_gpu4_v4/records_qwen_drive_traj_203/qwen_drive_control_trace.jsonl
  sha256 787c2b548c9d825e414fda2510005a64eee4cd083eac7dff4acc48bb7f65b024
```

This result established technical connectivity, but it used the now-retired
low-resolution transport profile. It cannot establish the formal Qwen
zero-shot baseline on this route.

## Official-input profile and dropout screen

The formal profile now transports the original 1600x900 CARLA frames as
lossless PNG, sets no `CameraFrame.target_size`, and delegates the distinct
history/current resize to the released Qwen processor. The retired compressed
profile is not retained as an experimental arm.

GPU preflight Job `1160181` completed two real forwards on the official-input
profile:

```text
trajectory shape             50x3
cold inference             54.60 s
warm inference              3.87 s
CUDA peak allocated       13114 MiB
CUDA peak reserved        14330 MiB
device              NVIDIA A800 80 GB
```

It completed without OOM. The report is:

```text
/public/share/lidachuan/orion_assets/qwen_drive_b2d_smokes/bridge_official_input_png_v1/report.json
```

The first clean/corruption screen was submitted as six paired jobs. All arms
use the same released Qwen checkpoint, official-input profile, controller and
route. The only paired difference is front-view `camera_dropout` before the
four-frame history buffer:

| Route | Scenario role | Clean job | Dropout job | Dropout window |
| --- | --- | ---: | ---: | --- |
| 146 | Established Orion clean-safe/dropout-pedestrian-collision case | 1160227 | 1160228 | route progress 0.30--0.55 |
| 151 | Parking-crossing-pedestrian event | 1160229 | 1160230 | route progress 0.321623--0.475794 |
| 203 | Pedestrian crossing and prior Qwen integration route | 1160231 | 1160232 | full route |

Per-frame traces record route progress, corruption family, active state,
affected sensors and schedule. This screen measures Qwen's native corruption
sensitivity only; it contains no U injection or U-aware response.

## Claim boundary

This is an engineering bridge, not evidence that Qwen is a better closed-loop
policy.  Qwen's camera geometry and training domains differ from Bench2Drive;
the measured partial completion, collisions, lane departure, blockage, and
sub-real-time throughput are concrete reasons not to claim superiority.  It
also does not solve the task-relevant uncertainty problem by itself.  The next
scientific step should adapt/calibrate on the target domain and compare matched
routes, rather than attributing Orion's previous limitation solely to backbone
intelligence.

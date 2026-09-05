# Qwen-Drive agent fidelity and evaluation review

Date: 2026-09-05 (Asia/Shanghai)

## Bottom line

The proposed result is scientifically defensible even if the unadapted Qwen
baseline is weak in Bench2Drive: the primary question is whether adding the
intended uncertainty signal produces a route-level, paired improvement in
closed-loop safety without obtaining that improvement by stopping or giving up
progress.

The main comparison should be:

```text
same Qwen weights + same RGB/history + same command + same controller
    baseline / zero-U / shuffled-U / true-U
```

Bench2Drive is the primary closed-loop test. NAVSIM is a useful secondary
pseudo-closed-loop corroboration, but it cannot replace an interactive test.
"Qwen-Drive" is the model/release, not a separate benchmark.

The integration is close to the released implementation on the model side.
This review identified reduced image resolution as an avoidable fidelity gap,
so the formal profile now transports lossless sensor-resolution images and
lets the official Qwen processor apply its own budgets. The largest remaining
new component without an official reference is the CARLA trajectory
controller, which must be isolated before describing a low score as Qwen's
domain-transfer failure.

## 1. What is official and what is ours?

There is no released official Qwen-Drive Bench2Drive/CARLA agent to compare
against. The public repository provides the Qwen model, scene contract,
planning inference and benchmark evaluation scripts. It does not provide CARLA
sensors, route-command plumbing or a low-level controller.

The correct comparison therefore has two boundaries:

1. **Model boundary:** are we giving the released model the same kind of scene
   and calling the same inference path?
2. **Simulator boundary:** does our necessary Qwen-trajectory-to-CARLA adapter
   preserve a good trajectory and control it correctly?

### Model-side comparison

| Item | Released Qwen-Drive | Our implementation | Assessment |
| --- | --- | --- | --- |
| Source/weights | `QwenDriveForPlanning.from_pretrained` | The same public class and released 4B SFT planner | Equivalent; remote checkout is clean apart from `__pycache__`, at commit `28091c1`. |
| Planning call | `run()`/`plan_from_inputs()` | Direct call to `model.run(InferenceMode.DIRECT_PLANNING, scene, num_samples=1, seed=42)` | Official inference code, not a reimplementation. |
| Camera views | front, front-left, front-right | Same three semantic views in the same order | Equivalent contract. |
| Temporal images | four frames at `t-1.5,-1.0,-0.5,0s`, grouped view-major | Same four-frame cadence at 2 Hz, grouped view-major | Equivalent after the first 1.5 s. |
| Ego history | 16 samples at 10 Hz in current ego frame | Same length/rate and frame conversion | Equivalent contract; route start is padded. |
| Coordinates | `x` forward, `y` left, heading positive left | Same before model; explicit conversion from CARLA | Equivalent contract and unit-tested. |
| Navigation | straight/left/right plus driving-command one-hot | Bench2Drive `RoadOption` mapped to the same values | Semantically aligned. |
| Model output | 50 `(x,y,heading)` samples at 10 Hz | Same output shape validated before use | Equivalent. |
| Planning mode | direct and reasoning are both supported | SFT direct, one deterministic sample | Legitimate released mode, but not identical to the paper's strongest closed-loop SFT setting. |
| Candidate selection | online inference does not have ground truth | One sample, no oracle selection | Scientifically correct. Official best-of-six is a ground-truth upper bound and must not be used online. |
| Transport | in-process in official scripts | Unix socket between CARLA Python 3.8 and Qwen Python 3.10 | Serialization only; it does not replace the processor, VLM, prompt builder or planner. |

This is therefore not a substantially different model agent. The sidecar
decodes the images into official `CameraFrame`/`DrivingScene` objects and calls
the released planning method. No custom VLM, prompt encoder, patchifier,
diffusion/flow sampler or planner weights have been introduced.

### Real differences that can affect the baseline

| Difference | Why it exists | Risk to attribution | Required treatment |
| --- | --- | --- | --- |
| Reduced image resolution | It bounded memory and latency during initial integration | Resolved for formal runs. | The profile now sends raw-size images and lets the official processor resize them. The old profile is retired, not an experimental arm. |
| JPEG quality 90 | It kept the cross-environment payload and retained history small | Resolved for formal runs. | Transport is now lossless PNG. |
| CARLA camera intrinsics/extrinsics | Qwen has no single published CARLA rig | Medium; genuine unseen-rig/domain shift, not an RPC bug | Report the rig and, if practical, test a source-like rig. Do not claim official-domain parity. |
| Initial history padding | A route begins without 1.5 s of past frames/state | Localized to route start | Ignore a warm-up prefix in open-loop analysis or prime the history before scoring. |
| SFT direct mode | Fastest valid online configuration | Medium; official closed-loop SFT reporting uses reasoning | Compare SFT direct and SFT reasoning with all other inputs fixed. Never use ground-truth best-of-N. |
| CARLA trajectory adapter and PID | The release outputs trajectories but no Bench2Drive controls | **High.** A good plan can be lost in coordinates, temporal resampling or PID tuning. | Pass expert/recorded future trajectories through the exact same adapter/PID as a controller oracle. |
| 5 m/s cap and 1 s staleness guard | Integration safety controls | Low for the observed run, but they are ours | Keep identical across arms and report intervention frequency. |

The retired profile pre-resized history images to 384x224 and current images to
768x416. The formal profile now sends all frames losslessly at the configured
CARLA sensor resolution of 1600x900 with `CameraFrame.target_size=None`. The
official processor consequently applies the checkpoint's distinct history and
current pixel budgets itself, exactly as it does for an unforced official
scene.

## 2. What the current Bench2Drive run does and does not show

Job `1155519`, task 203 (`Town04`, `PedestrianCrossing_1`, weather 23) is **our
Qwen integration scored by the official Bench2Drive evaluator**. It is not a
published or official Qwen-Drive Bench2Drive result.

The evaluator recorded:

- route completion: `52.84%`;
- infraction penalty: `0.209334`;
- driving score: `11.061207`;
- one pedestrian collision and one vehicle collision;
- `30.22%` outside-route-lane distance;
- final status: `Failed - Agent got blocked`.

The integration itself stayed alive: it produced 151 plans and 1,503 control
frames with zero inference exceptions. Replanning happened every 0.5 s as
configured; plan age was 0--0.45 s, so the result was not caused by a dead RPC
or a stale-plan emergency-stop loop.

The trace also gives limited evidence against the 5 m/s cap being the cause of
the final blockage. Speed exceeded 5 m/s for only 27 frames. In the last 20
plans, predicted forward displacement at 3 s averaged 0.467 m and ranged from
-0.765 to 4.633 m. Thus the controller often received a near-stationary or
backwards future and correctly asked for braking. That makes model/input
behavior a plausible contributor, but it does not yet distinguish Qwen's true
domain failure from reduced visual fidelity, history startup, planning mode or
the trajectory-control interface.

## 3. Minimum proof that our agent did not manufacture the weak baseline

These gates should be completed before the U closed-loop matrix.

### Gate A: official-scene numerical parity

Run one released scene through both:

1. the untouched official planning script/API; and
2. our sidecar scene path,

using identical images, target sizes, weights, mode, seed and sample count.
The two 50x3 trajectories should agree to numerical tolerance. This directly
tests that RPC serialization and scene construction do not change the model.
It has not yet been run, so parity must not be claimed as measured.

### Gate B: untouched NAVSIM reproduction

Run the official checkpoint and official NAVSIM prediction/evaluation path on
the provided data before adding U. Reproducing the published neighborhood
validates the checkpoint, environment, official scene files and official
processor independently of CARLA.

### Gate C: official input profile lock

Use only the lossless 1600x900 transport profile with no bridge-side resize.
Record the effective Qwen image grids, latency and peak GPU memory in the first
run, then keep this profile fixed across clean, corruption and future U arms.
The retired low-resolution profile is not a required ablation.

### Gate D: controller oracle

Feed an expert or recorded valid 50-point future through the exact coordinate
conversion, temporal resampling and PID on task 203 and a small fixed route
set. If it cannot follow lanes and complete routes, repair/control-calibrate
this layer before drawing a model conclusion. Also replay a Qwen trajectory in
an isolated controller harness to distinguish prediction geometry from CARLA
actuation.

### Gate E: matched relative comparison

Once A--D pass, freeze the agent implementation. In every uncertainty arm keep
the sensor rig, image profile, checkpoint, reasoning mode, random seed,
trajectory adapter, PID and route set identical. Only the U condition changes.
This is what makes the relative gain attributable even if the absolute Qwen
baseline remains weak.

## 4. Bench2Drive primary evaluation

Use a paired route/seed matrix with four arms:

1. `Qwen baseline`: no learned U effect;
2. `Qwen + zero-U`: exercises the interface with a neutral value;
3. `Qwen + shuffled-U`: same marginal U distribution but mismatched to the
   scene/route;
4. `Qwen + true-U`: matched uncertainty estimate.

If a text oracle is useful during development, report it separately as a
capacity upper bound, not as the learned-system result.

Primary safety outcomes should include collision rate/type, off-road or
outside-lane distance, route timeout/blockage and infraction penalty. Always
co-report route completion/progress, minimum useful speed, prolonged-stop time
and comfort. A policy that brakes everywhere may reduce collisions but has not
demonstrated useful safety.

Report paired route-level differences and confidence intervals rather than one
aggregate run. The claim is strongest if true-U improves collision/lane safety
and driving score over baseline and zero-U, beats shuffled-U, and preserves a
pre-registered minimum progress level across multiple fixed routes and seeds.

## 5. NAVSIM secondary evaluation

NAVSIM is appropriate as a secondary check because the official Qwen release
already supports its input/evaluation protocol and reports NC (no collision),
DAC (drivable-area compliance), EP (ego progress), TTC, comfort and PDMS.
Run the same baseline/zero-U/shuffled-U/true-U comparison with identical scenes,
seeds, mode and sample count.

Its interpretation must stay narrow:

- it queries the planner once per scene;
- surrounding agents replay recorded behavior and do not react;
- it cannot measure compounding errors, recovery or interactive closed-loop
  behavior;
- Qwen was trained/evaluated on NAVSIM and already reports high PDMS, so the
  available headroom may be small.

A NAVSIM improvement would corroborate better trajectory quality and
off-road/collision proxies. A null improvement would not by itself refute a
Bench2Drive closed-loop safety gain, because the domains and interaction model
are different. Ground-truth best-of-six candidate selection must be excluded;
the official evaluation documentation describes it as an oracle upper bound,
not an online policy.

## 6. Claim boundary after these checks

If the fidelity and controller gates pass and the paired experiment succeeds,
the defensible conclusion is:

> On an unseen target driving domain where the released Qwen-Drive baseline is
> weak, explicitly conditioning the same policy on matched observation
> uncertainty improves closed-loop safety and overall driving performance,
> with zero-U and shuffled-U controls ruling out an interface-only or generic
> conservatism explanation.

It would not establish that Qwen has become state of the art on Bench2Drive, or
that the same gain automatically transfers to its training domain. Those are
not required for the intended uncertainty claim.

## Sources and local evidence

- Official Qwen-Drive repository: <https://github.com/QwenLM/Qwen-Drive-1.0>
- Official model interface: <https://github.com/QwenLM/Qwen-Drive-1.0/blob/main/docs/model.md>
- Official scene/data contract: <https://github.com/QwenLM/Qwen-Drive-1.0/blob/main/src/qwen_drive/scene.py>
- Official evaluation protocol: <https://github.com/QwenLM/Qwen-Drive-1.0/blob/main/docs/evaluation.md>
- Official technical report: <https://arxiv.org/html/2609.00111v1>
- Our Qwen sidecar: `uq_estimator/qwen_drive_bridge.py`
- Our Bench2Drive agent: `team_code/qwen_drive_b2d_agent.py`
- Our active bridge configuration: `configs/qwen_drive_b2d_agent_v1.json`
- U migration gap review: `docs/qwen_drive_u_gap_review_2026-09-05.md`
- Server evaluator record:
  `/public/share/lidachuan/orion_assets/qwen_drive_b2d_smokes/closedloop_route203_flash_gpu4_v4/eval_qwen_drive_traj_203.json`
- Server control trace:
  `/public/share/lidachuan/orion_assets/qwen_drive_b2d_smokes/closedloop_route203_flash_gpu4_v4/records_qwen_drive_traj_203/qwen_drive_control_trace.jsonl`

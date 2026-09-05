# Qwen-Drive Route 151 failure diagnosis and NAVSIM pair plan

Date: 2026-09-05

## Executive conclusion

The available evidence does **not** support the simple explanation that the
Qwen-Drive 4B vision-language backbone cannot see the pedestrian.  The stronger
diagnosis is a failure at the perception/reasoning-to-trajectory interface under
Bench2Drive/CARLA domain transfer: even the released SFT planner's explicit
reasoning path recognizes the pedestrian but converts that observation into an
unsafe maintain-speed trajectory.  Cross-domain shift remains a likely
contributor, but has not yet been isolated from run-to-run effects.

The low-level PID controller is unlikely to be the primary cause.  In the clean
run it was given a fast, straight trajectory through the conflict point and did
what that trajectory requested.  In the front-camera-dropout run, the same
controller executed a much slower model trajectory and the pedestrian cleared
before the ego arrived.

NAVSIM is therefore useful as a **secondary evaluation track**: it removes the
CARLA camera/controller transfer from the first experiment and evaluates Qwen in
one of its native planning domains.  It should not replace Bench2Drive, because
NAVSIM is non-reactive and cannot establish closed-loop collision avoidance.

## Route 151 evidence

### Clean versus front dropout

The collision-aligned trace gives the following comparison:

| Run | State near the clean collision location | Model/controller behavior | Outcome |
| --- | --- | --- | --- |
| Qwen SFT direct, clean | step 258, progress 0.527952, pose `(102.44, 302.78)`, speed 4.731 m/s | desired speed 6.422 m/s, throttle 0.75, brake 0; successive plans remain straight and put the 3 s point about 18.9-20.1 m ahead | pedestrian collision; route completion 100, score 50 |
| Qwen SFT direct, clean repeat | step 292, progress 0.539878, pose `(101.21, 303.15)`, speed 4.349 m/s | desired speed 4.691 m/s, throttle 0.75, brake 0; nearby 3 s plans extend 14.5-20.6 m with at most about 0.5 m lateral displacement | pedestrian collision; route completion 100 |
| Qwen SFT direct, front dropout | same location reached later, step 386, progress 0.527462, pose `(102.44, 303.23)`, speed 4.642 m/s | while dropout is active, desired speed falls to 0.99-1.94 m/s and ego slows; it accelerates again after corruption ends | no collision; route completion 100, score 100 |

This is a timing intervention, not evidence that black images improve semantic
understanding: dropout delayed arrival until the crossing pedestrian had cleared.
It does, however, show that the existing controller can execute a conservative
Qwen plan.  The clean failure originates upstream of the actuator command.

The evaluator and control traces are stored remotely under:

- clean: `/public/share/lidachuan/orion_assets/qwen_drive_b2d_smokes/qwen_official_input_dropout_screen_v1_route151_clean`
- dropout: `/public/share/lidachuan/orion_assets/qwen_drive_b2d_smokes/qwen_official_input_dropout_screen_v1_route151_front_dropout_event`

### Is the visual backbone blind?

An official Qwen VQA inference was run on two 1600 x 900 Route 151 source frames
around the pedestrian's lane entry, independently of the planner and controller.
The model reported that the road was clear in the first frame, that a pedestrian
entered from the right in the second frame, and recommended hard braking.  The
artifact is:

`/public/share/lidachuan/orion_assets/qwen_drive_b2d_smokes/route151_vqa_layer_diagnostic_v1/vqa-1160361.out`

This demonstrates that the shared 4B multimodal model has the capacity to
recognize and reason about this CARLA pedestrian.  It does **not** prove that the
exact online frames at the collision generated the same internal representation;
the VQA input used historical frames from the same route/event.

The follow-up closed-loop reasoning-planning run removes that remaining ambiguity.
Job `1160367` completed normally with no inference errors and produced the same
pedestrian collision (route completion 100, score 50).  At progress 0.486, roughly
3.7 m before the eventual contact location, its rationale was “Maintain speed and
nudge left to increase clearance from the pedestrian on the right.”  It repeated
that policy at progress 0.511.  At contact, the ego was moving at 4.529 m/s, the
trajectory implied a desired speed of 5.773 m/s, and the controller brake was 0.
The planned lateral displacement was only about 0.1 m over the next 3 s.  The
model therefore saw the relevant object online but underestimated the conflict
and failed to express a braking trajectory.

The [released model's own limitations](https://arxiv.org/html/2609.00111v1) fit the observed behavior: the authors say
reasoning can miss causal/temporal scope and a generated trajectory need not
follow its textual rationale.  They also present unseen-camera-rig transfer only
qualitatively and state that reliable quantitative accuracy and target-domain
adaptation remain open.  Thus “domain shift” is plausible, but the more precise
hypothesis is **domain-sensitive grounding/action alignment in the SFT planning
head**, not generic visual blindness.

### What is and is not established

- Eleven historical Orion Route 151 clean/corruption evaluator records complete
  the route without pedestrian, vehicle, or layout collision.  The event is
  avoidable and is not an unavoidable scenario-script collision.
- The original and one new Qwen SFT direct Route 151 clean run both collide with
  the scenario pedestrian (2/2 direct clean runs).  The second run's per-route
  evaluator record and trace were complete, but its wrapper hung after printing
  the final route criteria and was canceled during global-statistics cleanup;
  its SLURM terminal state must not be mistaken for an unfinished scenario.
- The reasoning-planning diagnostic is a third independent clean run and also
  collides (1/1 reasoning clean run), but it is not a repeat of direct mode.
  These small counts show a repeatable failure across modest actor-timing/input
  changes, not a statistically estimated collision probability.
- The current Bench2Drive bridge had been hard-coded to direct-planning mode.
  Reasoning mode is now supported without changing sensors, resolution,
  preprocessing, checkpoint, controller, sample count, or seed.

## NAVSIM setup assessment

### What is already set up

- The official NAVSIM v1.1 source is pinned remotely at commit
  `0811876c274e8b058ab2be9b3dcd4d37bd23f177` in
  `/public/share/lidachuan/orion_assets/third_party/navsim-v1.1`.
- No NAVSIM dataset was downloaded and no existing environment was modified.
- A deterministic Qwen image resolver was added in
  `uq_estimator/qwen_drive_navsim_images.py`.  In clean mode it returns the
  original image path; in dropout mode it changes only selected camera/frame
  pixels and preserves the official target size and preprocessing path.
- `scripts/run_qwen_navsim_pair.sh` invokes the unmodified upstream runner with
  fixed SFT/reasoning/single-sample settings for both halves, and
  `scripts/audit_qwen_navsim_pair.py` rejects token/sample mismatches and records
  per-token trajectory changes and input/output hashes.
- `scripts/build_qwen_navsim_dropout_manifest.py` derives either current-front
  or all-history-front frame selections from the exact scene JSONL, so corrupted
  frame membership is explicit rather than inferred from file naming.

The [NAVSIM v1.1 installation](https://github.com/autonomousvision/navsim/blob/v1.1/docs/install.md)
pins Python 3.9 and PyTorch 2.0.1, whereas the working Qwen runtime is
Python 3.10 and PyTorch 2.8.  Scoring/data construction must therefore use a
separate environment instead of changing the known-good Qwen environment.

Storage is a hard constraint: the [documented NAVSIM v1.1 `test` sensor
archive](https://github.com/autonomousvision/navsim/blob/v1.1/docs/splits.md) is
about 217 GB, while the shared filesystem currently has about 75 GB free.  The pilot
must use selected scenes and sequential shard extraction, retaining only the
three cameras and four timestamps consumed by Qwen, rather than unpacking the
whole sensor split.

The remote machine also timed out when probing both the Hugging Face OpenScene
metadata URL and the nuPlan S3 map URL.  Its current Qwen environment lacks
`nuplan`, GeoPandas, Shapely, Hydra, Pandas, PyArrow, and SQLAlchemy, so merely
putting NAVSIM on `PYTHONPATH` cannot run PDM scoring.  The next data/environment
step therefore requires either an available outbound mirror or transferring a
prebuilt Linux environment plus the selected metadata, maps, metric cache, and
camera shards from another machine.

### Reproducibility gap in the Qwen release

Qwen's [official evaluator](https://github.com/QwenLM/Qwen-Drive-1.0/blob/main/docs/evaluation.md)
consumes a Qwen-specific scene JSONL containing 10 Hz ego
history/future, metadata, prompts, and image references.  The public repository
([data documentation](https://github.com/QwenLM/Qwen-Drive-1.0/blob/main/docs/data.md))
documents this schema but does not ship the benchmark scene JSONL and instructs
users to build it from the official dataset split.  A simple conversion from
NAVSIM's standard 2 Hz agent input would be wrong: Qwen states that its 10 Hz
trajectories are read directly from the original nuPlan database.

Until an author-provided scene file or exact builder is obtained, any converter
we implement must be clearly labeled as an Orion reproduction and validated
against the documented schema, frame timestamps, coordinate convention, and
official clean score neighborhood before drawing corruption conclusions.

### Clean-corruption pair contract

Each pair must keep all of the following identical:

1. scene JSON record and token, model/planner checkpoint, planning mode, prompt,
   Qwen preprocessing, number of samples, and seed;
2. ordered camera/timestamp references and trajectory/ego/navigation inputs;
3. evaluator metric cache and scoring configuration.

Only the resolver's selected pixels may differ.  Initial experiments should use
the installed `planner-sft`, reasoning-planning mode, `num_samples=1`, and a fixed
seed.  Qwen's best-of-N result must not be used for a paired robustness claim.

For each token, retain clean and corrupted predictions and report paired changes
in PDMS and its no-at-fault-collision, drivable-area, progress, TTC, and comfort
components.  Also report trajectory displacement/speed change.  Aggregate with
paired bootstrap confidence intervals, and separately count harmful, neutral,
and apparently beneficial corruptions.  A corruption that merely stops the car
must not be described as a general driving improvement.

Recommended pilot conditions:

- clean;
- current-frame front-camera dropout;
- all-four-history-frames front-camera dropout.

Start with a small deterministic token manifest containing interactions and
non-interaction controls.  First reproduce a plausible clean score; then run the
paired corruptions.  Expand only after the converter and clean baseline pass.

## Acceptance status and remaining gates

Completed:

- Route 151 SFT reasoning-planning clean was compared with SFT direct at the
  rationale, trajectory, braking, collision, and completion layers.
- One additional direct-clean repetition reproduced the pedestrian collision,
  giving two direct clean collisions in two available direct runs.  A larger
  sample is still required before estimating a collision probability.
- The clean/dropout resolver, exact frame-manifest builder, paired runner, and
  pair-integrity/trajectory audit are implemented and unit-tested.

Remaining:

1. Provision an isolated NAVSIM v1.1 scoring environment plus maps/metadata
   without touching the Qwen runtime.
2. Obtain or produce and validate a small Qwen-format scene JSONL and its exact
   camera-frame set.
3. Reproduce a plausible clean NAVSIM score on the pilot tokens.
4. Run deterministic clean/dropout pairs and pass the pair-integrity audit before
   interpreting metric deltas.

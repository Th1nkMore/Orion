# ORION Stage-1 actual-target chronological replay smoke

> Date: 2026-08-26
> Status: CPU plan/audit entry and shared-dataset file preflight complete; no
> Slurm job, GPU, CARLA, model, or training has been started.  A generated plan
> is G0 metadata evidence only and can never by itself pass G1.

## Frozen candidate

The persisted exploratory pilot resolves `Town04/Route214` to exactly one
folder:

```text
v1/OppositeVehicleTakingPriority_Town04_Route214_Weather6
```

The pilot's route and temporal sections agree on the exact source range:

- complete route: frame 0 through 135, contiguous, 136 frames;
- complete clean + observed replay: `136 * 2 = 272` model forwards;
- full-route measurement states: 90 paired target records;
- smoke prefix: frame 0 through 63, 64 frames;
- smoke clean + observed replay: `64 * 2 = 128` model forwards;
- smoke measurement states: 43 paired target records;
- smoke stratification: 20 dynamic-object annotation candidates and 23
  background annotations;
- non-measurement warm-up states: 21 per branch.  They must still be forwarded
  to construct ORION temporal memory, but must not be persisted.

The prefix ends after the first object candidates appear (the first candidate
measurement is frame 39).  A shorter frame-0 prefix would reduce cost but
would not exercise the intended visible-object projection path.

## Entry point

Local metadata-only plan:

```bash
.venv/bin/python scripts/plan_orion_actual_target_smoke.py --summary-only
```

This command imports no ORION/MMCV/CARLA/CUDA code, writes nothing unless
`--output` is given, and reports `g1_passed=false`, `job_submitted=false`.

On the compute server, perform the exact source and disk preflight before
writing a runnable plan:

```bash
python scripts/plan_orion_actual_target_smoke.py \
  --infos /public/share/lidachuan/orion_assets/data/infos/b2d_infos_val.pkl \
  --dataset-root /public/share/lidachuan/orion_assets/data/bench2drive \
  --output /public/share/lidachuan/orion_assets/spatial_uq_v1/manifests/route214_prefix63_replay_plan.json \
  --summary-only
```

The preflight checks the info pickle SHA-256 against the pilot lineage, an
exact frame-0-through-63 sequence with no gap or duplicate, all six canonical
camera metadata records, every image file, and every raw annotation file.  A
missing item raises an error instead of shortening the replay.

The preflight was run on `login1` with Python 3.8 and passed on the real shared
dataset:

- info SHA-256:
  `d31e151fbc1854ccc8b7288445f3585d1f6dcf660f08bbfb90f71a5660943798`;
- 64/64 consecutive info states verified;
- canonical ORION camera insertion order verified on every state;
- 384/384 camera images present;
- 64/64 raw annotation files present;
- missing file count: zero.

Persistent server artifacts:

```text
/public/share/lidachuan/orion_assets/spatial_uq_v1/manifests/route214_prefix63_replay_plan.json
  sha256 99879f0b47bb4af186bbd00ad82970fbd79f49be85020fc6f42a57df24d9e9ef

/public/share/lidachuan/orion_assets/spatial_uq_v1/manifests/route214_prefix63_replay_preflight_summary.json
  sha256 1a820f1f42126fad6e71cc2959f682daf31c248a081433979ce86515e0b02576
```

After the traffic-state formatter fix, a non-destructive amended plan was
written instead of overwriting the original artifact:

```text
/public/share/lidachuan/orion_assets/spatial_uq_v1/manifests/route214_prefix63_replay_plan_v2.json
  sha256 904c9d3187373bc44937d33c98f84ff406c7bec9b7d6467bbf9fa5e0d0de0538
```

The amended plan moves the formatter overwrite from an unresolved blocker to
a locally fixed finding that still requires runtime attestation. It remains
`execution_ready=false`, `g1_passed=false`.

The server `squeue` remained empty after the check.  The plan still reports
`execution_ready=false`, `g1_passed=false` because runtime model/adapter and
per-frame geometry/traffic attestations have not been executed.

## Temporal execution contract

The future real runner must execute two independent passes, never alternating
branches through one mutable model:

1. reset `pts_bbox_head` memory and participating `map_head` memory;
2. assert all tracked memory fields are empty;
3. replay clean frames exactly `0,1,...,63`, batch size 1, no shuffle;
4. save decoded output only for the 43 declared measurement frames;
5. reset and re-assert empty memory;
6. replay observed frames exactly `0,1,...,63` with fixed
   `local_occlusion`, severity 2, seed 20260826;
7. again save only the 43 measurement frames;
8. pair exact `(folder, frame_idx)` records using one shared replay protocol ID
   and distinct clean/observed branch-history IDs.

The corruption spans the whole bounded prefix.  It is a diagnostic exporter
smoke and is explicitly not time-alignment or closed-loop causal evidence.

## G1 is fail-closed

`uq_estimator/orion_replay_smoke.py` defines a runtime-attestation checker.
G1 can pass only when a future real runner records all 128 frames and verifies:

- exact frame order in both branches and resets before both frame-zero calls;
- no clean/observed interleaving;
- six images loaded per frame and canonical camera order;
- traffic state is `[N,2]`, with an object-aligned validity mask;
- the `lidar2img` matrices and image shape are the final post-augmentation
  values for all six views;
- decoded-output and actual-target adapters are connected on every frame;
- repository decoder parity, ORION-selected motion mode, and projection
  overlay QA pass;
- persisted frame identities equal the 43 declared measurement frames, with
  no warm-up frame accidentally stored.

Process exit code, a serialized mock bundle, or a completed replay without
this attestation cannot set `g1_passed=true`.

## Static blockers found before a real run

The current agent test pipeline still cannot provide honest traffic-state
targets because `adzoo/orion/configs/orion_stage3_agent.py` uses
`LoadAnnotations3D` without `with_light_state=True` in its test pipeline.  The
actual-target exporter therefore needs a dedicated config; changing target
generation must not silently change the deployed closed-loop agent.

The formatter bug found during this audit has been fixed locally:
`traffic_state [N,2]` and `traffic_state_mask [N]` are now validated and
filtered together from the same original `gt_bboxes_3d_mask`.  A dependency-
light regression test covers shape, dtype, and retained values.  This removes
the known code overwrite, but per-frame runtime attestation is still mandatory;
a unit test is not evidence that the full exporter batch is aligned.

The real frozen-ORION decoded-output hook and actual-target adapter also remain
unconnected.  These are explicit blockers, not zero-valued labels.

## Resource envelope

Exact computational count for the prefix is 128 perception forwards.  Based
on the prior full ORION load failure at 64 GB and successful known envelope,
the first real smoke should reserve one A800 80 GB, at least 220 GB host memory
against the approximately 192 GB known load envelope, and 8 CPU cores.  A
two-hour limit is a conservative reservation, not a runtime estimate; the
first smoke must report measured wall time, peak host RSS, and peak GPU memory.

No CARLA instance is required.  Do not persist raw images or non-measurement
features.  Do not expand to all ten pilot routes or start Stage-1 training
until this single-prefix G1 audit passes.

# Route214 six-view projection overlay preflight

Date: 2026-08-26

## Current outcome

The independent CPU implementation is complete:

- `uq_estimator/projection_overlay_preflight.py`;
- `scripts/preflight_route214_projection_overlays.py`;
- `tests/test_projection_overlay_preflight.py`.

It does not modify the chronological runner, decoded exporter, or visible
support projector. It does not construct ORION, call CUDA, submit Slurm, start
CARLA, or train.

The local mock plus relevant runner/exporter/target regression result is:

```text
56 passed, 1 existing torch TypedStorage deprecation warning
```

No real Route214 artifact is claimed yet. A login-node import probe entered
uninterruptible shared-filesystem I/O (`D` state), after which further remote
login-node probes were explicitly stopped. There is no Python import traceback
to report and therefore no evidence of a package-level import failure. The
real dataset mode must be run once inside the scheduled CPU/A800 preflight job,
where shared storage is accessed under the scheduler.

## Frozen real inputs

- route: `Town04/Route214`;
- concrete folder:
  `v1/OppositeVehicleTakingPriority_Town04_Route214_Weather6`;
- frames: `0` and `39`;
- frame 39 is the first preregistered annotation candidate in the pilot;
- GT box z-origin: `bottom`;
- camera order: the canonical six-view ORION order;
- patches: exact `40 x 40` row-major grid.

The script loads frames through the in-memory dedicated actual-target pipeline
mutation (`with_light_state=True`, `with_actor_ids=True`). For overlay-only CPU
execution it removes the non-geometric VQA tokenizer transform and its two
unused collect keys. That removal is recorded in the audit; image transforms,
post-augmentation matrices, GT object filtering and object-axis alignment are
unchanged.

## Scheduled-job command

After the new files are synchronized to the server project, run this inside
the already planned CPU/A800 preflight allocation, from the project root:

```bash
export ASSET_ROOT=/public/share/lidachuan/orion_assets
export COMPAT_PYTHON_BIN=/public/share/lidachuan/orion_assets/envs/orion-cl-centos7/bin/python
export COMPAT_GLIBC_SYSROOT=/public/share/lidachuan/orion_assets/envs/glibc-2.28/x86_64-conda-linux-gnu/sysroot
export COMPAT_LIBRARY_PATH=/public/share/lidachuan/orion_assets/envs/orion-cl/lib

./scripts/run_compat_python.sh \
  scripts/preflight_route214_projection_overlays.py \
  --dataset \
  --frames 0,39 \
  --candidate-frame 39 \
  --infos /public/share/lidachuan/orion_assets/data/infos/b2d_infos_val.pkl \
  --dataset-root /public/share/lidachuan/orion_assets/data/bench2drive \
  --map-file /public/share/lidachuan/orion_assets/data/infos/b2d_map_infos.pkl \
  --output-dir /public/share/lidachuan/orion_assets/spatial_uq_v1/overlays/route214_projection_preflight_20260826
```

The output directory must be absent or empty. Existing non-empty output is
never overwritten.

## Expected artifacts

For each selected frame:

- six `<CAMERA>.overlay.png` files;
- six matching `<CAMERA>.overlay.json` files;
- `six_view_contact_sheet.png`;
- `frame_audit.json`.

The root contains `manifest.json`. Audits include:

- the full post-augmentation six-view matrices;
- processed image shape and transform ID;
- exact camera order;
- bottom-origin GT boxes, class IDs and actor IDs;
- support tensor and valid-mask shapes;
- visible-object/nonzero-support counts per camera;
- SHA-256 for each PNG/JSON/contact sheet;
- explicit CPU/no-model/no-Slurm/no-CARLA claim boundaries.

Frame 39 must have nonempty projected support for the automated preflight to
pass. Mock outputs are explicitly labeled `input_mode=mock_fixture` and
`real_route214_data_used=false`; they cannot be mistaken for real evidence.

## G1 boundary

Artifact generation is not visual validation. Both frame and root audits keep

```text
human_visual_alignment_review_performed = false
g1_projection_overlay_gate_passed = false
```

even when every automated check succeeds. A reviewer must inspect all six
views, especially frame 39, and verify that the projected GT polygons and
40 x 40 heat agree with the processed RGB before G1 can be changed outside
this generator. Projected support remains an attribution proxy, not a causal
pixel explanation.


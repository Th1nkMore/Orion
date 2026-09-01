# Closed-loop held-out route audit (2026-08-26)

## Outcome

Eight routes are suitable for a **screening-only held-out pool** for localized
front-camera corruption.  Six are primary candidates and two are reserves.  No
CARLA process was started and no Slurm job was submitted during this audit.

The machine-readable manifest is
`configs/closedloop_uq_pilot/heldout_route_candidates_v1.json`.  The deterministic
resource parser is `scripts/select_closedloop_heldout_routes.py`.

## Meaning of “held out”

Here, held out means that the route is outside the current route-146/148/203
mechanism-development pilot.  It does **not** prove that the route, town, or
scenario family was absent from ORION pretraining or any future adapter training
set.  A separate data-lineage audit is required before calling these routes
training-held-out.

## Evidence and gates

The route geometry and scenario parameters were read from the official per-route
Bench2Drive XML files under:

```text
/public/share/lidachuan/orion_assets/Bench2Drive/leaderboard/data/
bench2drive220_<index>_orion_traj.xml
```

The clean-completion prior was cross-referenced by XML route id against the
repository's `assets/results/ORION.json`.  Every selected route has an existing
record with `Completed`, route score 100, penalty 1, and no collision or
completion-failure entry.  This is evidence that clean ORION *can* complete the
route, not a substitute for replaying `clean_off` in the current CARLA build,
checkpoint, seed, and weather.  Current-environment clean replay remains a hard
gate before running a corrupt condition.

Town03 and Town12 were excluded.  Route 148 was excluded because its current
clean baseline is invalid, route 203 because it is the headline/development
route, and route 146 because it is the mechanism-development route rather than a
held-out test.

## Ranked candidates

| Rank | Route index | Town | Scenario | Trigger `(x, y, z, yaw)` | Why it is useful |
|---:|---:|---|---|---|---|
| 1 | 147 | Town02 | DynamicObjectCrossing | `(144.5, 105.4, 0.0, 180.0)` | Same crossing mechanism as development route 146 but on a new town; a blocker and compact crossing actor make path-aligned vs equal-area off-path corruption interpretable. |
| 2 | 151 | Town02 | ParkingCrossingPedestrian | `(143.8, 302.6, 0.0, 179.9)` | A pedestrian enters the future corridor from a parking-side context; the safety-relevant image region is localized. |
| 3 | 194 | Town04 | OppositeVehicleRunningRedLight | `(169.3, -169.7, 0.2, 0.3)` | A moving vehicle crosses the conflict zone; tests whether uncertainty over the approaching actor matters more than equal corruption elsewhere. |
| 4 | 157 | Town05 | ParkingCutIn | `(79.8, 145.3, 0.0, 355.0)` | The vehicle transitions from roadside to ego lane, exposing both spatial localization and response-timing effects. |
| 5 | 164 | Town04 | HazardAtSideLane | `(-457.8, 12.7, 0.0, -202.3)` | Three bicycles encroach from the side lane; their changing relation to the path is useful for a localized rather than global anomaly test. |
| 6 | 168 | Town05 | HazardAtSideLane | `(-27.9, -207.5, 2.9, -179.9)` | A higher-speed Town05 bicycle variant, likely to create a measurable margin change under short local corruption. |
| 7 | 180 | Town05 | HazardAtSideLaneTwoWays | `(155.0, -47.0, 0.0, 269.9)` | Clean-valid bicycle encroachment with opposite traffic; kept as reserve because that traffic adds a control confound. |
| 8 | 208 | Town04 | NonSignalizedJunctionLeftTurnEnterFlow | `(222.8, -246.0, 0.0, 359.6)` | Ego crosses an actor flow at an unsignalized left turn; the conflict zone is localized, but continuous flow complicates attribution. |

The XML route lengths, trigger progress, scenario parameters, and clean-reference
scores are retained in the JSON manifest.

## Spatial corruption protocol for these routes

The XML trigger point activates the scenario.  It is **not necessarily** the
first frame in which the hazard becomes visible or enters the ego path.  For each
route:

1. Run only `clean_off` and save the front trace.
2. Mark the first hazard-visible frame, first path-conflict frame, and conflict
   resolution frame using route progress, not wall-clock time.
3. Freeze one short event window from those marks before looking at corrupt
   outcomes.
4. Define the on-path patch as the relevant actor/conflict region intersecting
   the projected ego path corridor.
5. Define a same-camera off-path patch with equal pixel area and matched clean
   luminance, but no overlap with the path corridor or relevant actor.
6. Hold corruption family, severity, area, event window, and stochastic seed
   fixed across the on-path/off-path pair.

Use local blur, darkening, glare, or occlusion.  Full-route and full-camera
dropout are excluded from the spatial causal comparison because they collapse
back into a global anomaly/slowdown test.

## Why several dynamic routes were rejected

The clean-reference gate removed tempting but confounded routes:

- route 179: two vehicle collisions;
- routes 196 and 197: one vehicle collision each;
- route 201: one vehicle collision;
- routes 209 and 210: one vehicle collision each.

These routes may be useful later for robustness analysis, but they cannot first
establish that corruption caused a clean-safe baseline to fail.  Route 174 was
also unsuitable because the existing clean record ended with `Failed -
TickRuntime` and only 33.83% route completion.

## Recommended minimal execution order (not executed here)

Use routes 147, 151, and 194 first because they cover pedestrian/cyclist and
vehicle conflicts with minimal background-flow confounds.  For each route, stop
after `clean_off` if current ORION does not finish cleanly.  If clean succeeds,
run only the frozen-window `corrupt_on_path_off` and
`corrupt_off_path_off` failure-induction pair.  Do not add a governor until a
route shows a reproducible on-path-specific degradation relative to both clean
and off-path corruption.

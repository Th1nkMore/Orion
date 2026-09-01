# Route197 dynamics-aware yield oracle v4 — job 1087488

Status: **resource-stopped structural deadlock; non-terminal; Stage2-ineligible**.

This run is not an official zero-collision result. The evaluator remained at
`Started`, emitted no terminal record, and cannot determine either collision or
route-completion outcome. The job was cancelled after the mechanism remained
stationary for 23.2 simulated seconds and the continuous ActorFlow exposed a
structural incompatibility in the release rule.

The trace contains neither legacy Density UQ nor the new learned observation
adapter. The scalar risk governor is an exact passthrough. The only
intervention is the privileged dynamics-aware planning oracle.

The recorded manifest field `orion_closedloop_conditioning=vision_adapter` is
therefore a metadata defect: its adapter checkpoint is empty and all 746 trace
frames have `observation_uq=null`. Future manifests must derive this field from
the actually loaded checkpoint and planning-response mode.

Key evidence:

- 746 frames through simulation time 37.25 s.
- Longest `<0.25 m/s` interval: steps 281–745, 23.2 s, 465 frames.
- Route-progress change during that interval: `4.7420027278533006e-05`.
- Maximum observed conflict-free clearance: 0.7 s; required: 1.0 s.
- ActorFlow source spacing: 25–50 m at 20 m/s, or 1.25–2.5 s.
- Oracle conflict horizon: 3.0 s. In steady flow the release premise cannot
  remain clear long enough and is unsuitable for a merge task.

Decision: `do_not_train_stage2_redesign_as_gap_acceptance_merge`.

The follow-up gap audit also fails its conservative utility gate. Across 19
observed flow actors (median headway 2.08 s), the native stationary ORION plan
has only one dynamics-aware accepted insertion candidate. It would release at
35.75 s, merge at 38.26 s beyond the saved trace, and wait 21.7 s after the
persistent stop began. A counterfactual merge-speed floor of 8 m/s produces
utility-window candidates, but that is an assertive acceleration redesign, not
the conservative response currently under test. Route197 is therefore retained
as a hard merge-planning negative case rather than the primary Stage2 oracle.

## SHA-256

```text
f72b27890a7d72766f257d3fa7051707960eb0d64520b2e68c572597dc24eb95  control_trace.jsonl
62339586af997351e8a6051667ee679e9d4dc76dee0af04a301f2425ab97555e  eval_orion_traj_0.json
4e2a55a54ab20fdd16dfe0df8d2cd610501c77f01152aec5f594b50b034b8af0  junction_geometry_v4_partial.json
028409b2a6ed654d2cd45f77e77b63e415d9c9993ecff7a794b70ae58c632948  manifest.json
41c2e80cb2aaf9566f336b606cdc3b482086be452145ae5cd697dde071176366  preregistration.json
bbe1474888a6fea2d7cb9a3bcbb152eef945435fe123b33ea62a5c05e2a9d4bd  route197_v4_structural_deadlock_report.json
d8e0d39f69350da7c9ed6798995389045d57448cfe7ed472601cc90301d12f6d  route197_gap_acceptance_audit.json
6cdaf15520a063f872f3522ef2bdbd072fd49a891b1c9a4939e0445855b068c6  signalized_junction_right_turn.py
```

# Qwen-Drive official-input dropout screen acceptance

Date: 2026-09-05 (Asia/Shanghai)

## Verdict

The **engineering screen passes**. The **scientific failure-induction gate does
not yet pass**.

All six clean/dropout jobs completed with evaluator exit code 0, wrote complete
Bench2Drive records and control traces, and had zero Qwen inference errors.
The official-input profile is operational in CARLA. Each dropout was applied
only to `CAM_FRONT` in its declared route-progress window.

The outcomes establish that Qwen reacts strongly to front-camera dropout, but
the direction is scenario-dependent. Route 146 degrades substantially; Route
151 becomes safer by predicting much shorter trajectories and slowing down;
Route 203 is invalid in both conditions. No route in this screen provides a
Qwen clean-safe-to-corrupt-collision pair, and there is only one repetition per
condition. These runs are therefore diagnostics, not evidence for a formal
uncertainty benefit claim.

## Engineering acceptance

### Official input

The bridge sends lossless 1600x900 RGB frames without bridge-side resizing and
constructs official Qwen `CameraFrame` objects with `target_size=None`. The
released processor applies the checkpoint's history/current pixel budgets.

Preflight Job `1160181` passed:

| Item | Result |
| --- | ---: |
| Output shape | 50x3 |
| Cold inference | 54.60 s |
| Warm inference | 3.87 s |
| CUDA peak allocated | 13,114 MiB |
| CUDA peak reserved | 14,330 MiB |
| OOM/runtime error | none |

### Closed-loop integrity

| Route/condition | Job | Slurm | Evaluator artifact | Trace | Plans | Inference errors |
| --- | ---: | --- | --- | --- | ---: | ---: |
| 146 clean | 1160227 | completed, exit 0 | complete | complete | 63 | 0 |
| 146 dropout | 1160228 | completed, exit 0 | complete | complete | 133 | 0 |
| 151 clean | 1160229 | completed, exit 0 | complete | complete | 50 | 0 |
| 151 dropout | 1160230 | completed, exit 0 | complete | complete | 64 | 0 |
| 203 clean | 1160231 | completed, exit 0 | complete | complete | 229 | 0 |
| 203 dropout | 1160232 | completed, exit 0 | complete | complete | 213 | 0 |

Dropout readback from the traces:

| Route | Requested window | Observed active progress | Active plans | Views |
| --- | --- | --- | ---: | --- |
| 146 | 0.30--0.55 | 0.30115--0.54988 | 17 | `CAM_FRONT` |
| 151 | 0.321623--0.475794 | 0.32329--0.47563 | 16 | `CAM_FRONT` |
| 203 | full route | 0.00000--0.52137 | 213 | `CAM_FRONT` |

Route 203 stops at 52.14% projected progress, so its observed upper bound is
expected: dropout remained active for every frame actually executed.

The retained lossless history used about 5--16 MiB depending on image content,
and every closed-loop run reported a maximum Qwen CUDA allocation of about
13,114 MiB (12.8 GiB). The earlier feature-cache storage problem has not
reappeared.

## Closed-loop outcomes

| Route | Condition | Completion | Penalty | Driving score | Pedestrian | Vehicle | Layout | Final status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 146 | clean | 100% | 0.401561 | 40.156 | 1 | 0 | 0 | completed |
| 146 | dropout | 100% | 0.130932 | 13.093 | 1 | 2 | 0 | completed |
| 151 | clean | 100% | 0.500000 | 50.000 | 1 | 0 | 0 | completed |
| 151 | dropout | 100% | 1.000000 | 100.000 | 0 | 0 | 0 | completed |
| 203 | clean | 78.56% | 0.055027 | 4.323 | 2 | 0 | 2 | blocked |
| 203 | dropout | 52.84% | 0.290487 | 15.349 | 1 | 1 | 0 | blocked |

The Route 203 score increase under dropout is not an overall improvement: it
comes with a 25.72-point loss in route completion, and both arms are blocked.
Driving score alone is misleading for this pair.

## Qwen's native response to dropout

The event-window traces show a consistent reduction in motion commitment, but
not a consistent safety outcome:

| Route | Condition | Event mean speed | Predicted forward at 3 s | Full-brake frames in event |
| --- | --- | ---: | ---: | ---: |
| 146 | clean | 4.48 m/s | 16.16 m | 7/147 |
| 146 | dropout | 3.75 m/s | 12.52 m | 11/176 |
| 151 | clean | 4.95 m/s | 19.23 m | 0/64 |
| 151 | dropout | 1.99 m/s | 5.65 m | 17/160 |
| 203 | clean, full run | 0.52 m/s | 2.34 m | 656/2286 |
| 203 | dropout, full run | 0.38 m/s | 1.97 m | 567/2126 |

Route 151 is the clearest behavioral finding: blanking the front camera makes
Qwen sharply shorten its trajectory, slow through the event and avoid the
pedestrian while still completing the route. This is native conservative
behavior, not injected U. Route 146 also slows, but still retains long plans in
part of the window and accumulates two additional vehicle collisions. Qwen's
response to evidence loss is therefore neither simply "ignore the dropout"
nor reliably safe.

## Scientific acceptance boundary

This screen supports three statements:

1. The official-input Qwen-to-Bench2Drive path is technically valid.
2. Front-camera dropout materially changes Qwen trajectories and control.
3. The response is context-dependent: sometimes protective, sometimes more
   dangerous, and sometimes dominated by an already failed baseline.

It does **not** yet support these statements:

- front dropout reliably converts a clean-safe Qwen route into a collision;
- Qwen has no native conservative response to missing evidence;
- an uncertainty-conditioned Qwen policy improves closed-loop safety;
- the observed pairwise differences are repeatable rather than one-run CARLA
  variation.

For the next scientific screen, keep this official-input agent fixed and first
find routes on which Qwen clean is safe. Repeat each clean candidate before
spending its paired corruption run. Route 146 remains useful as a relative
degradation case, and Route 151 should be retained as a negative/control case
where naive dropout already induces conservatism. Route 203 should not be used
as a primary causal case until its clean blockage is resolved or a different
route from the same hazard family qualifies.

## Artifact lineage

All artifacts live under:

```text
/public/share/lidachuan/orion_assets/qwen_drive_b2d_smokes/
  qwen_official_input_dropout_screen_v1_route{146,151,203}_*
```

Evaluator/trace SHA-256 pairs:

```text
route146 clean
  eval  9ef3cf65256c192cbe7d37ca49d564c5ba295edc7d92a5846b4db39b457ba853
  trace 8d64fd9699ffab9715dd9b11b720c254af254f9687b1c99d4c90c5978df2d112
route146 dropout
  eval  63d24bd5191ecce6ac8321df7851417b3a3d4a95bd319d8dc7c4dac3b341cc2b
  trace 551cacf51f8f126e62732274f7b41b4c1eed29aa860f70e12f93d9cd67fd09c8
route151 clean
  eval  230d60d4354d620e673b8d2390d3efaa8386b6d55424df37e73eaa41c8a6308e
  trace d7ba981d32110a1da261562bd4a91e2dedc887a7e68a35098a7e501f8809e1b6
route151 dropout
  eval  47810656a36d63a498d3e4dc192c1b14f53f714bac07d2ff35f148b07bf5d988
  trace 3af48fcb8791a0d441a93ea7defc5bf85510b9847c791470b66a26f7b0584e63
route203 clean
  eval  2d31cd471961c1742825e60e89b282768069510e676c0594c409e9dfce81c2ff
  trace 0e2b7644e5998841683d8297589e0e0045a06b0e9a9ebd0108b0a5a8ed5cf45b
route203 dropout
  eval  f0e69a158f932f7b7af916342bd091859d97a2a8cbc43d96ce58511957855fdd
  trace a99167a07726b7364e8882af5103bf7a6841a3ba1614320549b11547295d6e4a
```

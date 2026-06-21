# Reliability-to-Planning Pilot

## Purpose

Test whether the frozen R2d model naturally converts explicit reliability
language into a meaningful planning change.

For the same image, visual tokens, and VAE random seed, compare:

1. the original ORION planning prompt;
2. a multi-round prompt containing the correct reliability sentence;
3. the same multi-round prompt containing a shuffled reliability sentence.

The correct-versus-shuffled comparison is the valid semantic intervention.
The original-versus-multi-round comparison is confounded by prompt structure.

## Free Risk-Synthesis Attempt

The initial 200-frame synthesis run supervised all assistant turns and produced
0/20 parseable risk summaries. After masking supervision to the final
assistant turn, another 200-frame run reduced loss from 8.28 to 5.31 but still
produced 0/10 parseable summaries.

This route is stopped for the midterm. It asks a small LoRA update to learn
free-form object reconstruction and reliability-conditioned synthesis at the
same time, while R2d already provides a reliable explicit UQ statement.

## Planning Sensitivity Result

Ten valid calibration frames were evaluated with fixed visual features and
fixed decoder noise.

| Metric | Result |
| --- | ---: |
| Baseline ADE | 0.7373 m |
| Correct-text ADE | 0.2362 m |
| Shuffled-text ADE | 0.2367 m |
| Correct vs shuffled waypoint hidden L2 | 0.4301 |
| Correct vs shuffled trajectory displacement | 0.00283 m |

The large baseline-to-multi-round difference cannot be attributed to
reliability because the prompt structure also changed. The controlled
correct-versus-shuffled intervention changes the hidden state but leaves the
trajectory and ADE effectively unchanged.

## Conclusion

R2d supports the claim that ORION reads the continuous UQ token and maps it to
an explicit calibrated reliability concept. This pilot does not support the
stronger claim that a frozen planner naturally uses that concept.

The next planning experiment must fine-tune the planning path with explicit
behavioral supervision. Prompt injection alone is not a defensible planning
method.

Raw server artifacts:

```text
/root/autodl-tmp/orion_assets/reports/risk_qa/risk_synthesis_eval20.json
/root/autodl-tmp/orion_assets/reports/risk_qa/risk_synthesis_finalonly_eval10.json
/root/autodl-tmp/orion_assets/reports/risk_qa/risk_planning_text_valid10.json
```

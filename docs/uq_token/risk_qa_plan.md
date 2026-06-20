# Risk QA Pre-Experiment

## Goal

Verify that the ORION language model can:

1. read the injected continuous Density UQ token;
2. express the supplied reliability as explicit language;
3. preserve image-derived scene facts when the UQ token is changed;
4. carry the generated risk analysis into the later planning conversation.

This stage does not claim that the resulting trajectory is safer. Closed-loop
simulation remains the later behavioral validation.

## Existing ORION Interface

ORION already supports multi-round visual question answering. Non-final rounds
use `lm_head.generate()`, their generated tokens are appended to conversation
history, and the final planning round extracts the `<waypoint_ego>` hidden
state.

The intended conversation is:

```text
User: Describe the scene and identify critical objects.
Assistant: <scene facts>

User: Assess visual reliability and explain which observations require caution.
Assistant: <structured Risk QA answer>

User: Based on the analysis, provide the planning trajectory.
Assistant: Here is the planning trajectory <waypoint_ego>
```

## Supervision

### Reliability Field

Use the calibrated Density UQ percentile:

```text
reliability_percentile = round(100 * (1 - uq_score))
```

The answer contains both a number and a deterministic descriptor:

```text
visual_reliability_percentile: 23
reliability_level: low
```

The level is used only as a textual rendering of the continuous percentile:

```text
0-24: very low
25-49: low
50-74: moderate
75-89: high
90-100: very high
```

Scientific evaluation uses the numeric percentile, not the bucket.

### Scene-Fact Field

Until Chat-B2D descriptions are installed, derive objective critical-object
facts from GT 3D boxes:

- keep dynamic road users and vehicles;
- rank by distance to the ego vehicle;
- retain at most three objects within 30 meters;
- report category and coarse relative position;
- do not let UQ score alter this field.

Example:

```text
critical_objects:
- pedestrian, front-right, 8.4 m
- car, front, 13.2 m
```

### Risk-Interpretation Field

The first pilot uses a conservative statement:

```text
risk_interpretation:
Visual evidence has low reliability. Treat the listed critical-object
observations as uncertain and verify them before committing to the maneuver.
```

It does not invent object-level uncertainty. The current Density UQ estimator
is global over six camera views.

## Counterfactual Pair

For each image, construct:

```text
correct branch:
  image + correct UQ token
  target reliability = correct score
  target critical objects = scene GT

shuffled branch:
  same image + another sample's UQ token
  target reliability = shuffled score
  target critical objects = identical scene GT
```

Loss:

```text
L_risk_qa =
    0.5 * CE(answer_correct)
  + 0.5 * CE(answer_shuffled)
```

Train only the UQ projector and LLM LoRA parameters. EVAViT, QT-Former,
detectors, density estimator, base LLM, and trajectory decoder remain frozen.

## Intervention Groups

Evaluate the same prompt and image with:

| Mode | UQ input |
| --- | --- |
| none | no continuous UQ token |
| zero | null token with score zero |
| shuffled | another sample's score |
| correct | the sample's density score |

## Metrics

Reliability semantics:

- parse success rate;
- percentile MAE;
- Pearson and Spearman correlation;
- correct-versus-shuffled intervention sensitivity.

Scene-fact stability:

- critical-object category precision/recall against generated GT facts;
- exact stability of object categories between correct and shuffled outputs;
- hallucinated-object count.

Language quality diagnostics:

- output length;
- malformed structured output rate;
- repeated-text rate.

Planning linkage, after Risk QA passes:

- waypoint hidden-state displacement with and without Risk QA history;
- trajectory displacement;
- ADE/FDE under correct, zero, and shuffled Risk QA history.

## Pre-Experiment Schedule

### R0: Data and Prompt Audit

```text
Samples: 50 calibration frames
Training: none
Outputs: target answers and GT object facts only
```

Acceptance:

- every target parses;
- at least 80% of frames contain one or more reportable critical objects;
- prompt plus answer remains below the 2,048-token limit.

### R1: Generation Plumbing Smoke

```text
Samples: 5
Checkpoint: base ORION and existing UQ adaptation checkpoints
Training: none
Modes: none, zero, shuffled, correct
```

Purpose:

- verify generation works with appended UQ tokens;
- measure whether previous grounding LoRA already changes language;
- inspect formatting and hallucination failure modes.

No scientific claim is made from R1.

### R2: Counterfactual Risk QA Alignment

```text
Train routes: density split training routes
Training samples: 300 effective frames
Counterfactual forwards: correct + shuffled per frame
Gradient accumulation: 4
Projector LR: 5e-5
LoRA LR: 5e-6
Calibration evaluation: 100 frames
Seed: 42
```

Proceed only if:

1. parse success is at least 95%;
2. correct percentile MAE is at most 15 points;
3. correct Pearson is at least 0.7;
4. shuffled output follows the injected shuffled score;
5. critical-object F1 drops by no more than 5 points from none to correct;
6. correct-versus-shuffled object-category stability is at least 90%.

Initial structured-output R2 result:

- 300 effective samples completed;
- teacher-forced loss decreased, but five-frame generation parse rate was 0%;
- strict uppercase fields and exact numeric percentiles produced broken text;
- correct and shuffled inputs changed the text, but not in a usable form.

R2b therefore switches the language target to ORION-style natural language:

```text
Visual reliability is moderate.
Critical objects: traffic sign in the front-right; car in the front.
Risk assessment: ...
```

R2b evaluates five calibrated reliability levels rather than exact generated
numbers. Continuous-score evaluation remains available through the separate
numeric readout; the language experiment tests semantic risk understanding.

R2b also failed free-generation parsing after 300 samples. Token-level audit
confirmed that teacher-forcing and generation prompts were identical. The
remaining failure was task interference: one new QA round simultaneously tried
to teach reliability semantics and retrain detailed critical-object language.

R2c decomposes the interface into native multi-round responsibilities:

```text
Round 1: existing ORION critical-object QA
Round 2: "Visual reliability is {very low|low|moderate|high|very high}."
Round 3: risk synthesis and planning
```

Only Round 2 is aligned in the next pilot. Round 1 remains frozen and provides
scene facts. This directly tests whether the continuous UQ token acquires an
explicit language meaning without rewriting ORION's existing scene QA ability.

### R2c Result

Training:

```text
Effective frames: 200
Counterfactual forwards: correct + shuffled
Optimizer updates: 50
Trainable: UQ projector + LLM LoRA
Target: "Visual reliability is {five-level label}."
```

Five-frame smoke generation succeeded:

- correct and shuffled branches were both parseable on all five samples;
- score near 0.36 produced `moderate`;
- shuffled score near 0.13 produced `high`;
- shuffled score 1.0 produced `low`, one level above the `very low` target;
- none and zero controls did not fabricate a trained reliability statement.

Fifty-frame calibration result:

| Branch | Parse rate | Level accuracy | Ordinal MAE | Spearman |
| --- | ---: | ---: | ---: | ---: |
| correct | 1.00 | 0.88 | 0.12 | 0.354 |
| shuffled | 0.86 | 0.68 | 0.21 | 0.953 |

The correct calibration subset is concentrated in the `moderate` level, so its
rank correlation is not informative. The shuffled intervention spans a much
wider score range.

Among 33 frames where correct and shuffled targets belonged to different
levels, the generated reliability statement changed on 31 frames:

```text
intervention response rate: 93.9%
```

This is the first successful explicit-language evidence that the LLM reads the
continuous UQ token and assigns it a reliability meaning.

### R2d: Balanced Reliability Alignment

The seven shuffled parse failures all occurred at very high reliability
(`uq_score` approximately 0.04-0.07). The model fell back to its original scene
QA responses instead of emitting `very high`.

The next run should balance the five reliability levels:

```text
60 frames per reliability level
300 effective frames total
correct + shuffled counterfactual forwards
same learning rates and frozen components as R2c
```

R2d acceptance:

- at least 95% parse rate on both branches;
- at least 80% five-level accuracy;
- ordinal MAE at most 0.25;
- intervention response at least 90%;
- no degradation of the separately evaluated original critical-object QA.

Only after R2d passes should Risk QA text be inserted into the planning
conversation history.

### R3: Risk QA to Planning Interface

Run only after R2 passes.

Compare:

```text
image -> planning
image + Risk QA history -> planning
image + shuffled Risk QA history -> planning
```

This stage measures whether explicit analysis reaches the waypoint token. It
does not require immediate ADE improvement. The required result is:

- correct and shuffled risk histories cause measurable, systematic waypoint
  changes;
- scene facts remain fixed;
- clean-view ADE remains within 3% of base ORION.

## Stop Rule

Stop Risk QA alignment if two controlled R2 runs fail either reliability
semantics or scene-fact stability. In that case, use a direct textual
reliability prompt as the interpretability baseline and move the primary
conditioning experiment to the pre-LLM adapter route.

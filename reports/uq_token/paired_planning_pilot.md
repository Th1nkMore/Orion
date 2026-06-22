# Paired Corruption Planning Pilot

## Corruption Audit

Ten calibration frames were evaluated with three deterministic corruption
families and three severity levels.

| Corruption | Severity | Mean UQ delta | Increase rate |
| --- | ---: | ---: | ---: |
| camera dropout | 1 | +0.0895 | 1.00 |
| camera dropout | 2 | +0.6052 | 1.00 |
| camera dropout | 3 | +0.6369 | 1.00 |
| blur | 2 | +0.0017 | 0.50 |
| dark | 3 | -0.0167 | 0.00 |

Only camera dropout is consistently recognized by the current Density UQ
estimator. The first planning pilot therefore uses one dropped camera.

## First Paired Pilot

Configuration:

```text
initialization: R2d reliability-language checkpoint
effective training frames: 50
corruption: one dropped camera
trainable: UQ projector and LLM LoRA
loss: corrupted planning + 0.1 * clean/corrupted trajectory consistency
```

Corrupted-view calibration result on 35 valid planning frames:

| UQ mode | ADE | FDE |
| --- | ---: | ---: |
| none | 0.1910 | 0.3059 |
| zero | 0.2887 | 0.4396 |
| shuffled | 0.2147 | 0.3889 |
| correct | 0.2447 | 0.4359 |

Clean-view result:

| UQ mode | ADE | FDE |
| --- | ---: | ---: |
| none | 0.0808 | 0.0771 |
| zero | 0.1169 | 0.1090 |
| shuffled | 0.0954 | 0.1108 |
| correct | 0.1031 | 0.1018 |

The pilot fails both gates:

- correct UQ does not outperform none or shuffled on corrupted views;
- correct-UQ clean ADE degrades by about 27.5% relative to no-token.

## Diagnosis and Next Test

The consistency objective never requires correct UQ to be more useful than
shuffled UQ. The model can minimize the objective while ignoring token
semantics.

The next minimal test adds a counterfactual ranking objective:

```text
distance(corrupted + correct UQ, clean reference) + margin
  < distance(corrupted + shuffled UQ, clean reference)
```

This does not prescribe a hand-designed trajectory. It only requires correct
reliability information to recover the clean expert behavior better than an
incorrect reliability token.

## Counterfactual Ranking Result

The ranking loss was active on 14 of 20 training frames, so the experiment did
receive a correct-versus-shuffled planning signal.

Corrupted-view calibration result on the same 35-frame protocol:

| UQ mode | ADE | FDE |
| --- | ---: | ---: |
| none | 0.1964 | 0.3120 |
| zero | 0.3024 | 0.4611 |
| shuffled | 0.3054 | 0.5472 |
| correct | 0.3159 | 0.5698 |

Correct UQ remains worse than shuffled UQ and substantially worse than the
no-token path. The explicit-token planning route therefore fails the
predefined stop rule after two controlled architecture pilots.

Training-time pair losses in these early token pilots did not reset every
random generator between clean, correct, and shuffled forwards. Their
intervention evaluations remain valid because evaluation reseeded each mode,
but the training comparison was noisier than intended. The later vision
adapter experiments reset Python, NumPy, Torch, and CUDA randomness for every
paired branch.

## Decision

- Keep R2d as evidence that the LLM reads and verbalizes continuous UQ.
- Do not claim that explicit UQ tokens improve planning.
- Stop scaling the current token-plus-LoRA planning route.
- Move the next behavioral baseline to a pre-LLM uncertainty adapter, where
  UQ changes the evidence presented to the LLM without directly modifying the
  trajectory decoder.
- Retain Density UQ as a monitoring contribution if the adapter also fails.

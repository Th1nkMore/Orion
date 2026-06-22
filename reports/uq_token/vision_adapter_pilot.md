# Pre-LLM UQ Vision Adapter Pilot

## Method

The adapter acts on the 513 object/map visual queries before they enter the
LLM:

```text
adapted_query = query + uq_score * low_rank_residual(query)
```

The output projection is zero-initialized, so the adapter starts as an exact
identity. It does not append tokens and does not access the waypoint or
trajectory decoder.

Training uses one-camera dropout, corrupted-view planning loss,
clean/corrupted waypoint-state consistency, and correct-versus-shuffled
ranking. The 100-step pilot updates only the adapter; the LoRA learning rate
is zero. All paired branches reset Python, NumPy, Torch, and CUDA randomness.

## Results

### Route1115, First 50 Frames

35 frames contain valid planning targets.

| View | UQ mode | ADE | FDE |
| --- | --- | ---: | ---: |
| corrupted | none | 0.1991 | 0.3145 |
| corrupted | shuffled | 0.1583 | 0.2490 |
| corrupted | correct | **0.1520** | **0.2398** |
| clean | none | 0.0821 | 0.0752 |
| clean | correct | **0.0789** | 0.0795 |

Correct UQ improves corrupted ADE by 23.7% over none and 4.0% over shuffled.

### Route1115, First 100 Frames

85 frames contain valid planning targets.

| UQ mode | ADE | FDE |
| --- | ---: | ---: |
| none | **2.7355** | **5.6492** |
| shuffled | 2.7740 | 5.7373 |
| correct | 2.7726 | 5.7350 |

The improvement does not hold over the harder later section of this route.

### Independent Route504, First 50 Frames

35 frames contain valid planning targets.

| View | UQ mode | ADE | FDE |
| --- | --- | ---: | ---: |
| corrupted | none | 0.7284 | 0.9167 |
| corrupted | shuffled | 0.6144 | 0.7593 |
| corrupted | correct | **0.5547** | **0.6749** |
| clean | none | 0.5039 | 0.5546 |
| clean | correct | **0.4650** | **0.5408** |

Correct UQ improves corrupted ADE by 23.9% over none and 9.7% over shuffled.
Clean ADE and FDE also improve.

## Interpretation

This is the first planning pilot where correct UQ outperforms shuffled UQ
without clean-view degradation. The pre-LLM adapter should be retained.

This is not yet a final claim: only two calibration routes were inspected,
Route1115 contains a hard section where the adapter fails, and confidence
intervals have not been computed. The next experiment must use route-balanced
sampling and route-bootstrap metrics.

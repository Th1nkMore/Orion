# Qwen visibility-token VLM insertion contract

Last updated: 2026-09-06 (Asia/Shanghai)

This document refines ADR-001 section 5 without changing its responsibility
boundary. It describes how continuous physical U tokens enter the released
Qwen-Drive 4B VLM before the Planning Expert.

## Verified upstream interface

The provisioned checkpoint declares `Qwen3_5ForConditionalGeneration` with a
2,560-wide text hidden state and eight full-attention layers among 32 VLM
layers. Qwen-Drive retains the post-rotary key/value cache of those eight
layers and gives it directly to the released Planning Expert. The expert places
its 50 waypoint queries immediately after the final three-axis mRoPE anchor.

The server environment currently imports Transformers 5.14.1, while the
checkpoint config records 5.15.0. Therefore source inspection alone is not an
acceptance test; the exact installed runtime must pass a full-model smoke.

Transformers' Qwen3.5 forward path accepts exactly one of `input_ids` or
`inputs_embeds`. Image features are produced separately and scattered into
positions identified by the original image placeholder ids. Multimodal
position ids are also derived from the original ids, modality map, and image
grids. Passing only custom embeddings to the top-level model would lose that
contract unless image scatter and position construction were reproduced
explicitly.

## Accepted V0 insertion

The block is inserted immediately after the final camera
`vision_end_token_id`, before the driving-history and navigation instruction:

```text
official camera prefix
  -> learned U_START vector
  -> valid global/frontier physical tokens projected to width 2560
  -> learned U_END vector
  -> unchanged official history/navigation/assistant suffix
```

The implementation:

1. builds the official text embeddings and reuses the installed Qwen3.5 image
   encoder and placeholder scatter;
2. inserts continuous boundary and projected U vectors without resizing the
   tokenizer vocabulary;
3. creates a shadow id sequence used only to recompute official three-axis
   mRoPE positions, treating the U block as text-like positions;
4. calls the unchanged Qwen3.5 language model with the explicit embeddings and
   positions;
5. extracts the same released full-attention layer caches and moves the
   Planning Expert anchor to the last augmented prefix position.

Invalid padded frontier rows are omitted from the sequence. This is necessary
because the Planning Expert consumes raw scene K/V without a scene padding
mask. True, zero, and shuffled O3 controls retain the same valid mask, so their
sequence lengths remain paired.

## Identity and causal-control boundary

`visibility_enabled=false` calls the released `_prefill` unchanged. Its
positions, cache, anchor, and fixed-seed trajectory must be exactly identical
to the official path.

This disabled identity is not the same experiment as zero-U. Zero-U preserves
the augmented architecture and block length but zeros its 23 physical features;
boundary vectors, extra attention slots, and shifted suffix positions still
exist. Consequently, neither zero-U nor a newly zero-initialized projector may
be described as bit-identical to the official no-U baseline. Reported causal
comparisons require all four arms when relevant:

- released official baseline, with no inserted block;
- true U in the augmented architecture;
- zero U with the same augmented architecture;
- spatially shuffled U with the same augmented architecture.

## V0 gates

The first direct-prefill full-model smoke must establish:

- disabled cache, anchor, and fixed-seed trajectory equal the official path;
- positions before insertion are unchanged;
- every suffix position advances by exactly the inserted block length;
- U positions are contiguous in all three mRoPE axes;
- every Planning Expert scene-cache layer has the augmented sequence length;
- the final anchor is the final augmented prefix position;
- the untrained projector is reported only as an interface probe.

Direct prefill is V0a only. V0 is not complete until reasoning-planning also
uses the augmented prompt before reasoning generation, retains the same turn
closure semantics as upstream, and passes its own identity/position/cache
tests. No V0 smoke is a grounding, trajectory-quality, or safety result.

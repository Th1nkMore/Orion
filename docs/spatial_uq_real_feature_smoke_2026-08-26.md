# Real EVAViT paired-feature smoke (2026-08-26)

## Outcome

The real frozen ORION EVAViT path successfully extracted two same-frame
clean/corrupt records on one A800.  The result validates the model-loading,
Bench2Drive identity, corruption, patch-token, schema, and persistence path.
It is not a training result and does not validate semantic uncertainty.

Slurm job `1061171` is recorded as `FAILED` because a shell-quoting error in
the final, read-only one-line audit changed `map_location="cpu"` into an
undefined Python name.  The extractor itself had already completed, printed a
valid summary, and atomically written the artifact.  The artifact was therefore
reloaded and checked on the login node instead of spending another A800 job.

## Fixed smoke condition

- partition: `Nvidia_A800`
- node: `gpu2`
- GPU: one NVIDIA A800 80 GB PCIe
- CPU: 4 cores
- requested host memory: 128 GB
- maximum observed RSS: 21,689,752 KiB (about 20.7 GiB)
- elapsed allocation time: 4 minutes 40 seconds
- source: first two frames from `b2d_infos_val.pkl`
- corruption: `local_dark`, severity 1, seed 20260826
- corrupted view index: 0, resolved and recorded as `CAM_FRONT`
- no CARLA process and no Stage-B condition

## Verified artifact

Path:

```text
/public/share/lidachuan/orion_assets/spatial_uq_v1/smoke/
paired-real-1061171.pt
```

Read-back facts:

- file size: 157,368,832 bytes;
- dataset schema: `spatial-uq-paired-dataset/v1`;
- record schema: `spatial-uq-paired-feature/v1`;
- two records from one route;
- each observed and clean tensor has shape `[6, 1600, 1024]`;
- target provenance is only
  `paired_cosine_representation_error_proxy`;
- metadata explicitly sets actual perception failure, semantic uncertainty,
  closed-loop safety, and LLM-understanding claims to false;
- camera order is
  `CAM_FRONT, CAM_FRONT_LEFT, CAM_FRONT_RIGHT, CAM_BACK, CAM_BACK_LEFT,
  CAM_BACK_RIGHT`;
- the exact affected pixel count and normalized region were retained.

## Resource implication

The two float32 records occupy about 150 MiB, or roughly 75 MiB per
frame/corruption/severity record because both clean and observed unpooled tokens
are stored.  Naively caching four corruption families and three severities for
all 12,806 validation frames would be on the order of 11 TiB before filesystem
overhead.  That is incompatible with the 350 GB shared quota.

Therefore this extractor is approved only for bounded smoke/debug shards.
Full Stage-1 training must generate paired corruptions online or adopt a
sharded format that deduplicates clean tokens and uses an audited lower-precision
representation.  No bulk feature-cache job should be submitted in the current
record format.

## Next gate

Before another A800 extraction:

1. freeze route-disjoint train/validation/calibration/held-out source folders;
2. implement or validate online paired generation/deduplicated shards;
3. choose the task-grounded actual-failure target, or retain the proxy label
   without semantic claims;
4. estimate bytes and runtime from the smoke for the exact proposed sample
   count;
5. keep the held-out closed-loop route pool outside mechanism tuning.

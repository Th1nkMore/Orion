# Round-2 Dashboard Summary

## Current Baseline

- Open-loop AUROC: `0.954` over `12806` samples.
- Open-loop normal: `UQ=0.023`, `L2@3s=2.379`, `Col@3s=0.0001`.
- Open-loop adverse: `UQ=0.893`, `L2@3s=1.779`, `Col@3s=0.0161`.
- Closed-loop all-scenario baseline ADE@3s: `2.461`.
- Closed-loop all-scenario FiLM ADE@3s: `4.440`.
- Closed-loop collision delta (FiLM - baseline): `-0.0013`.

## Conservative Shortcut Evidence

- Normal closed-loop ADE worsens from `2.776` to `6.003`.
- Adverse closed-loop collision changes from `0.0083` to `0.0064`.
- All-scenario brake MAE changes from `0.252` to `0.433`.
- All-scenario average speed changes from `4.885` to `4.885`.

## BEV / Feasibility

- IPM BEV uncertainty delta: `+0.1543` adverse-minus-normal.
- Normal BEV mean: `0.5826`.
- Adverse BEV mean: `0.7369`.

## Feasibility Snapshot

- `Q1` Flash Attention 是否阻止 attn_weights 提取？: 是，已确认阻断。有明确缓解方案。
- `Q2` BEV query 是否排列在规则 30×30 grid 上？: 不是。query 数量为 600（非 900），且 reference_points 是 learned embedding，不是固定网格。
- `Q3` poses_cls 的值分布如何？: 本地无 per-frame poses_cls 数据，需在服务器上提取。但可从侧面分析。
- `Q4` 20 个 plan_anchor 的 BEV 空间覆盖是否有足够差异？: 覆盖差异充分，BEV cost 有足够的 per-mode 区分力。
- `Q5` 能否在 16GB 机器上只加载 EVAViT + QT-Former？: 无法在本地验证（无 ORION checkpoint），需在服务器上测试。

## Next Use

- Treat this folder as the local source of truth for round-2 planning.
- Compare every future server experiment against the metrics and figures generated here.

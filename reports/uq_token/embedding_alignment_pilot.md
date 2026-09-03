# Embedding-aware Reliability QA Alignment 预实验

## 实验目的

该实验用于验证一个关键问题：连续的 Density-UQ active embedding 不能假设被 frozen LLM 天然理解，但经过很轻量的对齐训练后，LLM 是否能够把该 embedding 与视觉可靠性语义对应起来。

因此本实验不直接优化规划轨迹，而是构造一个更可控的中间任务：给定 ORION 的 frozen 视觉 token，并额外注入 Density-UQ 的 score 与 active embedding，要求 LLM 输出当前视觉可靠性等级。该任务用于验证“UQ embedding 可以被 LLM 对齐并读取”，为后续风险描述和规划监督提供依据。

## 实验设置

- 视觉与规划主干：冻结 ORION 主体参数。
- 训练参数：仅训练 UQ token projector 与 LLM LoRA。
- UQ 输入：Density-UQ score + active embedding。
- 任务形式：Reliability QA，输出五级视觉可靠性描述。
- 训练集：Bench2Drive train split，按 very low / low / moderate / high / very high 五个等级均衡采样，每级 60 个样本。
- 训练步数：200 steps。
- 评估集：Bench2Drive calibration split，五个等级均衡采样，每级 10 个样本，共 50 个样本。
- 干预方式：对同一图像分别注入 correct UQ 与 shuffled UQ，观察输出可靠性等级是否随 UQ 改变。

## 主要结果

| 方法 | Parse rate | Accuracy | Ordinal MAE | Spearman | Intervention response |
|---|---:|---:|---:|---:|---:|
| 未对齐 | 0.00 | 0.00 | - | - | 0.00 |
| 200-step embedding-aware alignment | 0.86 / 0.88 | 0.50 / 0.50 | 0.44 / 0.45 | 0.85 / 0.88 | 0.74 |

表中斜线前后分别表示 correct UQ 与 shuffled UQ 的结果。Intervention response 表示当 correct 与 shuffled 的目标可靠性等级不同时，模型输出等级发生变化的比例。

![Embedding-aware Reliability QA Alignment](../../results/risk_qa/embedding_active_alignment_summary.png)

## 代表性样例

| 样本 | Correct UQ | Correct 输出 | Shuffled UQ | Shuffled 输出 |
|---|---:|---|---:|---|
| BlockedIntersection_Town03_Route135_Weather5__00066 | 0.247 | moderate | 0.105 | high |
| NonSignalizedJunctionLeftTurnEnterFlow_Town12_Route949_Weather13__00056 | 0.631 | low | 0.247 | moderate |
| OppositeVehicleRunningRedLight_Town04_Route180_Weather23__00003 | 0.620 | low | 0.199 | high |

## 结论

该实验支持以下判断：

1. 未经过对齐训练时，LLM 基本不能稳定解析连续 UQ embedding 对应的可靠性语义。
2. 只需训练 UQ token projector 和 LLM LoRA，经过 200 step 的轻量对齐后，模型即可稳定输出可解析的视觉可靠性等级。
3. correct UQ 与 shuffled UQ 的干预对输出有明显影响，说明模型不是只依赖图像或固定模板，而是确实在利用注入的 UQ 信息。

该实验仍然是语义对齐实验，不直接证明规划轨迹会变得更安全。它的作用是补上方法链条中的关键一环：Density-UQ active embedding 可以通过轻量训练被 LLM 读取和语义化。后续若要证明规划有效性，还需要在风险描述或 waypoint 输出上加入行为监督。

## 产物路径

- 训练 checkpoint: `/root/autodl-tmp/Orion/checkpoints/risk_qa/embedding_active_level_200.pt`
- 训练后评估: `results/risk_qa/embedding_active_level_200_eval50.json`
- 未对齐 baseline: `results/risk_qa/embedding_active_untrained_eval50.json`
- 对比图: `results/risk_qa/embedding_active_alignment_summary.png`

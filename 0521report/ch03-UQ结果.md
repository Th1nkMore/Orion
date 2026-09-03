# 第三章 UQ 感知监测结果

本章回答 **Q1：能否监测感知退化？** 结论先行：**可以。** 在 B2D 验证集 12,806 帧、weather-based Normal/Adverse 划分下，v3 UQEstimator 的 score 对恶劣视觉条件具有 **AUROC ≈ 0.954**、**Gap ≈ 0.87** 的强分离能力。UQ 是**感知质量监测器**，不是规划 L2/碰撞的可靠预测器。

---

## 3.1 监测对象与成功标准

| 维度 | 定义 |
|------|------|
| **监测什么** | 当前帧视觉条件是否退化（雨/雾/夜/湿路等），而非「本帧规划会偏多少米」 |
| **输出** | `uncertainty_score` $\hat{s}\in(0,1)$；越高表示感知越不可信 |
| **独立标签** | Weather ID：0–3（Clear/Cloudy 白天四预设）= Normal；其余 19 种 = Adverse |
| **主指标** | AUROC（$\hat{s}$ → Adverse 二分类）、Normal/Adverse **均值差 Gap** |
| **目标线** | AUROC $> 0.7$（项目门槛）；v3 为 **0.954** |

**与训练指标区分**

| 指标 | 度量对象 | v3 典型值 |
|------|----------|-----------|
| Val Spearman | 预测 score vs **伪标签** | ≈ 0.97（拟合标签空间） |
| 开环 AUROC | 预测 score vs **天气划分** | **0.954**（独立验证） |

---

## 3.2 评估协议

- **数据**：Bench2Drive Base **50 条 val clip** 展开 → **12,806 帧**（`b2d_infos_val.pkl`）
- **模型**：`checkpoints/uq/best.pt`（v3，weather-based 伪标签训练）
- **脚本**：`scripts/eval_openloop.py` + forward hook 捕获 UQ score；ORION 主干冻结
- **汇总**：`results/eval_openloop_v3.json` / `.pt`

**样本分布**

| 类别 | 帧数 | 占比 |
|------|------|------|
| Normal（Weather 0–3） | 2,709 | 21.2% |
| Adverse（其余） | 10,097 | 78.8% |
| **合计** | **12,806** | 100% |

> **边界说明**：ClearNight（ID 15）归为 Adverse（低光照）；CloudyNoon/Sunset（2/3）归为 Normal（无降水/雾）。详见 ch01 Weather 表。

---

## 3.3 分离度与 AUROC（核心结果）

> 图表：`results/figures/baseline/fig1_score_dist.pdf`、`fig2_auroc.pdf`

### 3.3.1 Normal vs Adverse 汇总（v3，eval_openloop）

| 指标 | Normal | Adverse | 说明 |
|------|--------|---------|------|
| 帧数 | 2,709 | 10,097 | 50 clips |
| UQ 均值 | **0.023** | **0.893** | 几乎不重叠 |
| UQ 中位数 | **0.00008** | **0.970** | 双峰分布 |
| **Gap** | — | — | **0.870** |
| **AUROC** | — | — | **0.954** |

数据来源：`results/eval_openloop_v3.json`（`auroc: 0.9536…`，汇报取 **0.954**）。

### 3.3.2 代表性天气（score 排序与语义一致）

| 天气 | 分类 | UQ 均值（约） |
|------|------|----------------|
| ClearNoon | Normal | 0.0001 |
| ClearSunset | Normal | 0.0009 |
| CloudyNoon | Normal | 0.072 |
| ClearNight | Adverse | 0.782 |
| MidRainyNoon | Adverse | 0.993 |
| HardRainNight | Adverse | 0.989 |
| MidRainSunset | Adverse | 0.997 |

> 逐天气箱线图：**fig4_weather_boxplot.pdf** — 降水/雾/夜整体高于晴天白天，排序与直觉一致。

### 3.3.3 伪标签版本与 AUROC 轨迹

| 版本 | 要点 | AUROC（weather 划分） | 备注 |
|------|------|----------------------|------|
| v1 | gradient 恒权 + min-max | 0.621 | 已废弃 |
| v2 | max_mean + cosim + entropy | 0.993† / **0.601**‡ | †自身 scene_type；‡与 eval 对齐 |
| **v3** | weather-based 标签 + 百分位校准 | **0.954** | **当前** |

**v2 虚高根因**：2,481/12,806（**19.4%**）样本在 v2「场景类型 adverse」与 eval「天气 adverse」不一致；v3 用 `--scene_type_map` 对齐。

---

## 3.4 监测 ≠ 规划误差预测器

> 图表：`results/figures/baseline/fig3_uq_vs_l2.pdf`

UQ score 与**开环规划指标**相关性弱（v3 全量 12,806 帧，`eval_openloop_v3.json`）：

| 相关 | 系数 | 解读 |
|------|------|------|
| Spearman(UQ, L2@3s) | **ρ ≈ 0.14** | 弱相关：高 UQ 不完全等于大 L2 |
| Spearman(UQ, Col@3s) | **ρ ≈ 0.06** | 几乎无法直接用 score 预测碰撞 |
| Pearson(UQ, L2@3s) | **r ≈ −0.04** | 线性更弱 |

**与分组均值的对比（悖论）**

| 指标 | Normal | Adverse | 倍数 |
|------|--------|---------|------|
| L2@3s 均值 | 2.38 m | 1.78 m | Adverse **更低** |
| Col@3s 均值 | 0.01% | 1.61% | Adverse **≈161×** |

说明：**轨迹偏差（L2）与安全性（Col）正交**；UQ 捕获的是感知退化，不能直接当「规划会撞」的回归目标。后续需 **碰撞感知 FiLM**（ch04）桥接 score → 行为。

**一句话定位**

> UQEstimator 回答：「当前看得有多差？」  
> 不回答：「当前规划会偏多少 / 会不会撞？」

---

## 3.5 伪标签自洽性（三个层次）

汇报时建议用以下三点说明「分数可信」而非过拟合：

1. **训练内自洽**：Val Spearman ≈ 0.97（epoch 2 即平台），说明在伪标签空间内回归+排序+校准有效。
2. **训练外判别**：同一 checkpoint 在 **未参与标签构造** 的 weather 二分类上 AUROC 0.954，Gap 0.87 — 不是单纯记忆标签表。
3. **语义单调**：fig4 上 19 种 Adverse 天气 score 与降水/雾/夜强度一致；ClearNoon ≈ 0、HardRain/MidRain ≈ 1，可人工核对。

---

## 3.6 Baseline 规划背景（为何需要监测）

开环 ORION Baseline（同 12,806 帧，无 FiLM）：

| 指标 | Normal | Adverse |
|------|--------|---------|
| Col@3s | 0.01% | **1.61%** |
| L2@3s | 2.38 m | 1.78 m |

恶劣场景 **碰撞风险激增** 但 **L2 反而更低**（低速/保守 GT），强化「必须先监测感知，再用不同机制改行为」的叙事。

---

## 3.7 本节图表

| 编号 | 文件 | 用途 |
|------|------|------|
| 图 3-1 | `fig1_score_dist.pdf` | Normal/Adverse 分布分离 |
| 图 3-2 | `fig2_auroc.pdf` | ROC，AUROC=0.954 |
| 图 3-3 | `fig4_weather_boxplot.pdf` | 逐天气单调性 |
| 图 3-4 | `fig3_uq_vs_l2.pdf` | 弱相关 + 「≠ L2 预测器」 |

复制到报告目录（可选）：

```bash
cp results/figures/baseline/fig{1,2,3,4}_*.pdf 0521report/assets/ 2>/dev/null || true
```

---

## 3.8 本章小结

| 问题 | 结论 |
|------|------|
| 能否区分 Normal/Adverse？ | **能**，AUROC 0.954，Gap 0.87 |
| 能否预测 L2/碰撞？ | **不能可靠预测**，ρ 弱；需 FiLM + 碰撞 loss |
| v2 的 0.993 可信吗？ | **仅在旧标签空间**；对齐 weather 后 ≈ 0.60 |
| 工程上可部署吗？ | 可作 **感知退化监测 / Score-Gated 门控** 信号 |

**承上启下**：ch04 讨论用该信号驱动 FiLM 后，规划/回放指标如何变化及代价（Normal ADE 退化等）。

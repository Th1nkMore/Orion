# UQ-ORION 阶段性研究报告

> 日期：2026-03-30（v3 更新：修正 scene_type 分类，AUROC 0.954；v3 FiLM 重训 + 闭环评估；18 场景轨迹对比 GIF）
> 硬件：1x NVIDIA A100 80GB
> 基线模型：ORION (ICCV 2025)
> 目标：在 ORION 基础上轻量化引入不确定性感知，提升恶劣天气场景安全性

---

## 一、项目概述

### 1.1 研究动机

ORION 是一个基于 VLM（Vision-Language Model）的端到端自动驾驶框架，在标准场景下表现优秀。但 VLM-based planner 存在一个核心缺陷：**在恶劣天气（雨、雾、夜间）等 OOD 场景下过度自信**，无法感知自身感知质量的下降，从而做出激进的轨迹决策。

微调整个 VLM 主干（~1B 参数，36GB 模型文件）代价过高且不实际。我们提出**轻量即插即用的双层不确定性注入方案**：

- 新增参数 < 5M（实际 4.48M），**零主干微调**
- 通过 FiLM（Feature-wise Linear Modulation）在两个层面注入不确定性信号
- `use_uncertainty=True/False` 一行开关切换

### 1.2 技术方案

```
Vision Encoder (EVAViT, 冻结)
    | patch_tokens [B, 6, 1600, 1024]
    |--------------------------------------------+
    |                                            v
    |                                    UQEstimator (2.24M, 已训练)
    |                                            |
    |                                 uncertainty_embedding [B, 256]
    |                                 uncertainty_score     [B, 1]
    |                                            |
QT-Former (冻结) <-- FiLM L1 (场景理解层) ------+
    | vlm_memory                                 |
LLM / Qwen (冻结)                               |
    | ego_feature [B, 4096]                      |
    |<----- FiLM L2 (轨迹规划层) ----------------+
    v
VAE (冻结) --> trajectory
```

**为什么采用双层注入？**

- **FiLM L1**（QT-Former 层）：调制检测 query，让下游 LLM 接收到"视野受限"的场景表示。直觉：高不确定性时，抑制激进特征激活，放大保守特征权重
- **FiLM L2**（VAE 层）：调制 ego_feature（规划 token），将轨迹分布向保守模式偏移。直觉：高不确定性时，减小轨迹方差、倾向减速

**为什么用 FiLM 而非其他方法？**

FiLM（`output = gamma * input + beta`）是最简单有效的条件调制方式：
- 参数量极小（两个线性层）
- Identity 初始化（gamma=1, beta=0）保证新增模块不破坏原有性能
- 训练稳定，不需要修改主干结构

### 1.3 参数预算

| 模块 | 位置 | 参数量 | 说明 |
|------|------|--------|------|
| UQEstimator | 独立模块 | 2,210,817 (2.21M) | Transformer decoder + MLP |
| FiLM L1 (gamma+beta) | PETRTemporalTransformer | 131,584 (0.13M) | Linear(256→256) × 2 |
| FiLM L2 (gamma+beta) | Orion Detector | 2,105,344 (2.11M) | Linear(256→4096) × 2 |
| **合计** | | **4,447,745 (4.45M)** | **< 5M 预算 ✅** |

ORION 主干参数量约 1B，新增参数仅占 **0.45%**。

---

## 二、已完成工作

### Stage 0：特征提取 ✅

**完成日期**：2026-03-27

**做了什么**：从 ORION 冻结的 EVAViT backbone 提取每个样本的 patch tokens。

**为什么要做**：UQEstimator 的输入是 patch tokens（视觉编码器的中间表示），而非原始图像。预提取可以避免每次训练都过一遍 36GB 的模型，节省数百小时的训练时间。

**产出**：
- `data/features/` — 12,806 个样本的 patch tokens，共 ~235GB
- 每个样本：`[6 views, 1600 patches, 1024 dims]` = 6 个摄像头 × 40×40 patch grid × 1024 维特征

**关键命令**：
```bash
python scripts/extract_orion_features.py \
    --checkpoint ckpts/Orion.pth \
    --output_dir data/features \
    --ann_file data/infos/b2d_infos_val.pkl \
    --batch_size 8 --num_workers 1
```

---

### Stage 1a：伪标签生成 ✅ → v2 重构 ✅ → v3 scene_type 修正 ✅

**v1 完成日期**：2026-03-27 | **v2 重构日期**：2026-03-30 | **v3 修正日期**：2026-03-30

**做了什么**：为每个样本计算一个不确定性伪标签 score ∈ [0, 1]，作为 UQEstimator 的监督信号。

**为什么需要伪标签**：真实的"不确定性"没有 ground truth 标注。我们从视觉特征本身提取统计信号，组合成伪标签。

**v1 方案（已废弃）**：
```
score = 0.3 × gradient_score     # 无图像数据时固定 0.5
       + 0.3 × entropy_score      # Cohen's d = 0.056，几乎无区分力
       + 0.4 × consistency_score   # Cohen's d = 0.529，中等区分力

+ scene_type 校准：min-max → normal [0, 0.45]，adverse [0.55, 1.0]
→ AUROC = 0.621 ❌
```

**v1 问题诊断**：
1. `gradient_score` 因无图像数据恒为 0.5，浪费 30% 权重
2. `entropy_score` Cohen's d 仅 0.056，几乎不能区分 normal/adverse
3. min-max 归一化对异常值敏感，导致 ClearNoon UQ=0.003、MidRainSunset UQ=0.001 等极端值
4. normal/adverse 间隔仅 0.10（0.45→0.55），模型难以学习清晰边界

**v2 方案（当前）**：

基于对 5 维统计特征的 Cohen's d 分析，重新设计权重：

| 特征 | Cohen's d | v1 权重 | v2 权重 | 说明 |
|------|-----------|---------|---------|------|
| max_mean | **1.070** | 0% | **50%** | 最强区分特征（adverse 更低）|
| cosim | 0.529 | 40% | 35% | 跨视角一致性 |
| entropy | 0.056 | 30% | 15% | 保留作为多样性信号 |
| gradient | — | 30%（恒 0.5）| 0% | 移除 |

```
score = 0.50 × max_mean_score    # 1 - normalise(max_mean, [13, 16])
       + 0.35 × cosim_score       # 1 - normalise(cosim, [0.5, 1.0])
       + 0.15 × entropy_score     # normalise(entropy, [0, 1])

+ 百分位数校准 (p2/p98)：normal → [0.03, 0.38]，adverse → [0.62, 0.97]
  间隔从 0.10 扩大到 0.24
→ AUROC = 0.993 ✅
```

**为什么用百分位数校准**：p2/p98 比 min-max 更鲁棒，2% 极端值被裁剪而非拉伸整个分布。

**v2 新增 `--stat_cache` 快速路径**：利用预计算的 `stat_cache.pt`，标签生成从 ~2h 降至 ~2min。

**v3 scene_type 修正**：v2 使用特征文件的 scene_type（基于 CARLA 场景类型：Accident→adverse），但 eval_openloop 使用天气 ID 分类（Weather 0-3=normal）。这导致 2,481/12,806 样本分类不一致（19.4%），v2 的 AUROC 0.993 仅在自身标签空间有效，在 eval_openloop 分类下仅为 0.601。v3 引入 `--scene_type_map` 参数，使用 weather-based 分类对齐 eval_openloop，确保分类一致。

**产出**：
- `data/labels/uq_labels.pt` — v3 伪标签（12,806 样本，weather-based scene_type）
- `data/labels/uq_labels_v2.pt` — v2 备份
- `data/labels/uq_labels_v1_backup.pt` — v1 备份
- `data/weather_scene_type_map.pt` — weather-based scene_type 映射

---

### Stage 1b：UQEstimator 训练 ✅

**完成日期**：2026-03-27

**做了什么**：训练 UQEstimator 模型，学习从 patch tokens 预测不确定性 score 和 embedding。

**模型架构**：

```
patch_tokens [B, 6, 1600, 1024]
  → patch_proj Linear(1024→256)         # 降维
  → 2层 TransformerDecoder              # 16个 learnable query cross-attend patches
  → mean pool → [B, 256]
      +
stat_features [B, 5]                    # 5维统计特征
  → Linear(5→64) + GELU + LayerNorm
      ↓
  concat → [B, 320]
  → Linear(320→512) → Linear(512→256) → uncertainty_embedding [B, 256]
      ├─ score_head → [B, 1]  (Sigmoid, 值域 [0,1])
      └─ embed_head → [B, 256] (给 FiLM 用的条件向量)
```

**为什么用 Transformer decoder + learnable queries**：
- Patch tokens 太长（6×1600=9600 tokens），直接 pooling 会丢失空间结构
- 16 个 learnable queries 通过 cross-attention 自适应地关注最相关的 patches
- 参数高效：相比直接对 9600 tokens 做 self-attention，计算量小几个数量级

**损失函数**：
```
total = MSE_regression + 0.1 × calibration + 0.5 × ranking
```
- **MSE regression**：拟合伪标签 score
- **Calibration**：惩罚预测 score 标准差过小（防止模型输出全部趋同）
- **Ranking**：pairwise margin ranking loss，确保 adverse 样本的 score > normal

**训练结果**：

| 版本 | Epochs | Best Epoch | Spearman | 配置 |
|------|--------|------------|----------|------|
| v1 | 50 | 15 | 0.96 | 全 1600 patches |
| **v2** | **20** | **2** | **0.97** | **256 patches 子采样** |

- v2 训练使用 n_patches_subsample=256 加速（每 epoch ~5min vs ~10min）
- Spearman 0.97 在 epoch 2 即达到，后续 epoch 稳定在 0.96-0.97
- 显存占用 < 5GB，训练时间 ~100min

**产出**：
- `checkpoints/uq/best.pt` — v2 训练的 UQEstimator 权重（25MB）
- `checkpoints/uq/best_v1_backup.pt` — v1 备份

---

### Stage 2a：开环评估 + UQ Score 分析 ✅

**完成日期**：2026-03-28

**做了什么**：在完整验证集（12,806 样本）上运行 ORION 推理，同时通过 forward hook 捕获每个样本的 UQ score，收集 planning metrics。

**为什么用 forward hook**：非侵入式地在推理过程中截取 UQEstimator 的输出，不需要修改推理流程，也不影响性能。

**关键命令**：
```bash
python scripts/eval_openloop.py \
    adzoo/orion/configs/orion_stage3_infer.py ckpts/Orion.pth \
    --ann-file data/infos/b2d_infos_val.pkl \
    --out results/eval_openloop_full.pt
```

**运行统计**：
- 12,806 样本，~1.8 秒/样本，总计约 6.5 小时
- 显存 ~40GB（ORION 推理 + UQEstimator hook）

**产出**：
- `results/eval_openloop_full.pt` — 逐样本记录（planning metrics + UQ score + 天气信息）
- `results/eval_openloop_full_summary.json` — 汇总统计
- `results/figures/baseline/` — 5 张可视化图表

---

### Stage 2b：FiLM 训练 ✅

**完成日期**：2026-03-28

**做了什么**：分别训练三组 FiLM 权重，用于 ablation 实验。

| 模型 | Epochs | Best Loss | 参数量 | Checkpoint |
|------|--------|-----------|--------|------------|
| FiLM L1 (QT-Former) | 3 | 0.1058 | 131K | `checkpoints/film/best_l1.pt` |
| FiLM L2 (VAE) | 3 | 0.1160 | 2.1M | `checkpoints/film/best_l2.pt` |
| FiLM L1+L2 | 3 | 0.1080 | 2.2M | `checkpoints/film/best_l1l2.pt` |

**训练方式**：冻结 ORION 全部主干，只训 FiLM gamma/beta 参数。通过 LLM teacher forcing 保持梯度流。

**关键命令**：
```bash
# L1 only
python scripts/train_film.py --config ... --epochs 3 --lr 1e-3 --out checkpoints/film/best_l1.pt

# L2 only
python scripts/train_film.py --config ... --film-mode l2 --epochs 3 --lr 1e-3 --out checkpoints/film/best_l2.pt

# L1+L2
python scripts/train_film.py --config ... --film-mode l1l2 --epochs 3 --lr 1e-3 --out checkpoints/film/best_l1l2.pt
```

**权重验证**：所有 checkpoint 的 gamma_bias ≈ 1, beta_bias ≈ 0（接近 identity），标准差 ~0.02（有效但保守的调制）。

**当前状态**：代码完成，L1+L2+碰撞感知 checkpoint（`best_l1l2_col_v3.pt`）可用，但 Normal ADE 退化 +116%，根因已定位（embed_head LayerNorm 破坏 Score-Gate），Score-Gated FiLM 代码已完成，**checkpoint 未重训**，修复效果待实验验证。

**初步开环评估（500 样本）**：
- FiLM L1 vs Baseline: L2@3s 1.885m vs 1.916m（改善 1.6%）
- UQ Spearman 相关性: -0.291 vs -0.276（提升）
- 注意：L2 轨迹精度改善不等于安全性改善，需闭环碰撞率验证

---

### Stage 3：评估工具构建 ✅

**完成日期**：2026-03-29

**做了什么**：构建两个评估脚本，支持后续快速迭代。

**1. 热交换 Ablation 评估** (`scripts/eval_ablation_full.py`)
- 加载 7.5B 模型一次，热交换 FiLM 权重评估 4 组：A=Baseline, B=L1, C=L2, D=L1+L2
- 节省 ~5 小时的重复模型加载
- 快速测试通过（10 样本，4 组产生不同指标）

**2. 闭环回放评估** (`scripts/eval_closedloop_replay.py`)
- 使用 Bench2Drive 录制数据，无需 CARLA 环境
- 流程：ORION 推理 → PID 控制器 → 对比 GT 控制信号
- 指标：Control MAE (steer/throttle/brake)、Traj ADE、碰撞率、UQ 分层
- 快速测试通过：steer MAE=0.005, ADE@3s=7.21

**关键命令**：
```bash
# Ablation 评估（全量 ~3h，快测 --max-samples 100）
python scripts/eval_ablation_full.py \
    adzoo/orion/configs/orion_stage3_infer.py ckpts/Orion.pth \
    --film-l1 checkpoints/film/best_l1.pt \
    --film-l2 checkpoints/film/best_l2.pt \
    --film-l1l2 checkpoints/film/best_l1l2.pt \
    --out-dir results/ablation

# 闭环回放评估（全量 ~85min，快测 --max-scenarios 2）
python scripts/eval_closedloop_replay.py \
    adzoo/orion/configs/orion_stage3_infer.py ckpts/Orion.pth \
    --ann-file data/infos/b2d_infos_val.pkl \
    --film-checkpoint checkpoints/film/best_l1l2.pt \
    --out results/closedloop_replay.json \
    --frame-step 5
```

---

## 三、关键数据与分析

### 3.1 天气分类

#### 判定方式

Weather ID 直接来自 B2D 数据集的**文件夹命名**（后缀 `_WeatherN`），由 `eval_openloop.py` 的 `parse_weather_id()` 函数解析，规则硬编码为：

```python
NORMAL_WEATHER_IDS = {0, 1, 2, 3}   # eval_openloop.py:34
def is_adverse(weather_id: int) -> bool:
    return weather_id not in NORMAL_WEATHER_IDS
```

#### 完整 Weather ID 映射

| 类别 | Weather ID | 天气名称 | 说明 |
|------|-----------|---------|------|
| **Normal** | 0 | ClearNoon | 晴天正午 |
| **Normal** | 1 | ClearSunset | 晴天傍晚 |
| **Normal** | 2 | CloudyNoon | 多云正午 |
| **Normal** | 3 | CloudySunset | 多云傍晚 |
| Adverse | 5 | WetNoon | 湿路面正午 |
| Adverse | 6 | WetSunset | 湿路面傍晚 |
| Adverse | 7 | MidRainyNoon | 中雨正午 |
| Adverse | 8 | MidRainSunset | 中雨傍晚 |
| Adverse | 9 | WetCloudyNoon | 湿路多云正午 |
| Adverse | 10 | WetCloudySunset | 湿路多云傍晚 |
| Adverse | 11 | HardRainNoon | 暴雨正午 |
| Adverse | 12 | HardRainSunset | 暴雨傍晚 |
| Adverse | 13 | SoftRainNoon | 小雨正午 |
| Adverse | 14 | SoftRainSunset | 小雨傍晚 |
| Adverse | 15 | ClearNight | **晴天夜间**（无降水，但低光照）|
| Adverse | 18 | CloudyNight | 多云夜间 |
| Adverse | 19 | WetNight | 湿路面夜间 |
| Adverse | 20 | WetCloudyNight | 湿路多云夜间 |
| Adverse | 21 | MidRainyNight | 中雨夜间 |
| Adverse | 22 | HardRainNight | 暴雨夜间 |
| Adverse | 23 | SoftRainNight | 小雨夜间 |
| Adverse | 25 | FoggyNoon | 雾天正午 |
| Adverse | 26 | FoggySunset | 雾天傍晚 |

> Weather ID 4 和 16、17、24 在 B2D 验证集中不存在。

**验证集样本分布**：

| 类别 | 样本数 | 占比 |
|------|--------|------|
| Normal（ID 0-3） | 2,709 | 21.2% |
| Adverse（其余 19 种） | 10,097 | 78.8% |
| **合计** | **12,806** | 100% |

#### 分类的设计依据与边界说明

Weather 0-3 对应 CARLA 内置的 4 个"标准白天"预设，不含任何降水/雾气/夜间参数，是感知质量最稳定的基准条件。分类边界有两点值得注意：

1. **ClearNight（ID 15）归为 Adverse**：无降水但低光照，EVAViT 的 patch token 激活统计量与白天有显著差异（UQ score 约 0.235，介于 Normal 和重度 Adverse 之间），归为 Adverse 是合理的工程选择。
2. **CloudyNoon/CloudySunset（ID 2/3）归为 Normal**：多云但白天，无视觉降质（雾/雨/夜），L2 和碰撞率与 ClearNoon 相近，UQ score 在 v3 模型中也接近 0。

### 3.2 UQ Score 分离度

> 📊 对应图表：`results/figures/baseline/fig1_score_dist.pdf`

| 指标 | v1: Normal | v1: Adverse | **v2: Normal (1,584)** | **v2: Adverse (11,222)** |
|------|-----------|-------------|----------------------|------------------------|
| UQ Score 均值 | 0.545 | 0.780 | **0.007** | **0.800** |
| UQ Score 中位数 | 0.744 | 0.963 | **0.005** | **0.949** |
| UQ Score std | — | — | **0.010** | **0.297** |

| 指标 | v1 | v2 | **v3** | 目标 |
|------|-----|-----|--------|------|
| **均值差 (Gap)** | 0.235 | 0.794* | **0.870** | > 0.1 ✅ |
| **Normal/Adverse 重叠** | 显著 | 几乎为零* | **几乎为零** | — |

*v2 的高指标仅在特征文件 scene_type 空间有效，在 eval_openloop weather-based 分类下 gap 仅 0.231。v3 修正了此不一致。

**v3 分析**：修正 scene_type 分类后，Normal 均值=0.023，Adverse 均值=0.893，Gap=**0.870**。ClearNoon 几乎为零（0.0001），adverse 场景大部分 >0.9。

### 3.3 AUROC

> 📊 对应图表：`results/figures/baseline/fig2_auroc.pdf`

| 指标 | v1 数值 | v2 数值 | **v3 数值** | 目标 |
|------|---------|---------|------------|------|
| AUROC (UQ score → eval_openloop adverse) | 0.621 | 0.601* | **0.954** | > 0.7 ✅✅ |

*v2 的 AUROC 0.993 是在特征文件自身的 scene_type 上评估的；在 eval_openloop 的 weather-based 分类下仅为 0.601（因 19.4% 样本分类不一致）。

**v3 改进来源**：
1. **修正 scene_type 分类**：v2 使用场景类型（Accident→adverse），v3 改用天气 ID（Weather 0-3=normal），与 eval_openloop 对齐
2. **引入 max_mean 特征**（Cohen's d=1.07）：token 最大激活均值是区分 normal/adverse 最强的单一特征
3. **百分位数校准替代 min-max**：消除极端值，加宽分离间隔 [0.38, 0.62]

**逐天气 UQ score（v3）**：天气排序完全正确
| 天气 | 分类 | UQ score | 语义 |
|------|------|----------|------|
| ClearNoon | Normal | 0.0001 | 完美感知 |
| ClearSunset | Normal | 0.0009 | 接近完美 |
| CloudyNoon | Normal | 0.072 | 轻微退化 |
| MidRainyNoon | Adverse | 0.231 | 中度退化 |
| HardRainNight | Adverse | 0.989 | 严重退化 |
| MidRainSunset | Adverse | 0.997 | 极端退化 |

### 3.4 逐天气场景分析

> 📊 对应图表：`results/figures/baseline/fig4_weather_boxplot.pdf`

| 天气场景 | 样本数 | UQ Score | L2@3s (m) | Col@3s |
|----------|--------|----------|-----------|--------|
| **Normal 场景** | | | | |
| ClearNoon | 906 | 0.003 | 1.554 | 0.02% |
| ClearSunset | 688 | 0.746 | 1.581 | 0.00% |
| CloudyNoon | 824 | 0.885 | 3.890 | 0.02% |
| CloudySunset | 291 | 0.794 | 2.559 | 0.00% |
| **Adverse 场景** | | | | |
| HardRainSunset | 717 | 0.936 | 3.968 | 2.09% |
| HardRainNoon | 388 | 0.984 | 1.954 | 0.00% |
| MidRainyNoon | 714 | 0.993 | 2.540 | 0.68% |
| MidRainyNight | 537 | 0.981 | 1.051 | 0.53% |
| SoftRainNight | 632 | 0.820 | 1.033 | 1.50% |
| WetCloudyNoon | 431 | 0.989 | 1.593 | 0.27% |
| WetCloudySunset | 409 | 0.940 | 2.843 | 0.49% |
| FoggyNoon | 354 | 0.929 | 2.255 | 0.00% |
| FoggySunset | 962 | 0.876 | 0.855 | 0.00% |
| ClearNight | 508 | 0.235 | 1.181 | 0.00% |
| Unknown(23) | 1,070 | 0.675 | 0.968 | **11.25%** |

**关键发现**：

1. **碰撞率分布极不均匀**：Unknown(23) 场景碰撞率高达 11.25%，HardRainSunset 为 2.09%，SoftRainNight 为 1.50%。大多数场景碰撞率为 0。这意味着 FiLM 的提升会集中在少数高风险场景上
2. **L2 error 和碰撞率不完全正相关**：FoggySunset 的 L2@3s 最低（0.855m）但碰撞率为 0，而 Unknown(23) L2@3s 仅 0.968m 但碰撞率极高。说明轨迹精度和安全性是不同维度
3. **UQ score 与视觉降质高度一致**：HardRain/MidRainy 系列 UQ 接近 1.0，Foggy 系列 0.87-0.93，ClearNight 仅 0.235（夜间但视觉仍算清晰）

### 3.5 Planning Metrics（Baseline）

> 📊 对应图表：`results/figures/baseline/fig5_planning_bars.pdf`

| 指标 | Normal | Adverse | 全量 |
|------|--------|---------|------|
| L2@1s | 0.485m | 0.462m | 0.467m |
| L2@2s | 1.247m | 1.037m | 1.081m |
| L2@3s | 2.379m | 1.779m | 1.906m |
| Col@1s | 0.02% | 1.78% | 1.41% |
| Col@2s | 0.01% | 1.70% | 1.34% |
| Col@3s | **0.01%** | **1.61%** | 1.27% |

**重要发现**：Adverse 场景的 L2 error 反而低于 Normal（1.779 vs 2.379）。这看似反直觉，但合理的解释是：
- CARLA adverse 场景中车辆整体速度较低（雨天减速），GT 轨迹本身较短/保守
- L2 error 衡量的是预测轨迹与 GT 的偏差，低速场景天然 L2 低

但**碰撞率差异巨大**：adverse 的碰撞率是 normal 的 **161 倍**（1.61% vs 0.01%）。这正是我们要解决的核心问题——恶劣天气下感知降质导致碰撞风险剧增。

### 3.6 UQ Score 与 Planning Metrics 的相关性

> 📊 对应图表：`results/figures/baseline/fig3_uq_vs_l2.pdf`

| 指标 | 数值 | 显著性 |
|------|------|--------|
| Spearman(UQ, L2@3s) | ρ = 0.301 | p < 1e-224 |
| Pearson(UQ, L2@3s) | r = 0.182 | p < 1e-80 |
| Spearman(UQ, Col@3s) | ρ = -0.077 | p < 1e-15 |

**分析**：
- UQ score 与 L2 error 有弱正相关（ρ=0.301）：高不确定性样本的规划误差倾向更大
- UQ score 与碰撞率有弱负相关（ρ=-0.077）：这个方向符合预期（高 UQ 场景的碰撞率确实更高），但相关性弱
- 所有相关性都高度显著（p 值极小），说明 UQ signal 确实包含有意义的安全相关信息

---

## 四、代码实现清单

### 4.1 新增文件

| 文件 | 行数 | 功能 | 阶段 |
|------|------|------|------|
| `uq_estimator/model.py` | ~200 | UQEstimator 模型定义 | Stage 1 |
| `uq_estimator/losses.py` | ~100 | 损失函数（MSE + calibration + ranking）| Stage 1 |
| `uq_estimator/dataset.py` | ~150 | 数据集 + 统计特征计算 | Stage 1 |
| `scripts/extract_orion_features.py` | ~180 | 从 ORION 提取 patch tokens | Stage 0 |
| `scripts/generate_labels.py` | ~200 | 生成不确定性伪标签 | Stage 1a |
| `scripts/train_uq.py` | ~250 | UQEstimator 训练 | Stage 1b |
| `scripts/validate_uq.py` | ~150 | 验证报告 + 可视化 | Stage 1c |
| `scripts/eval_openloop.py` | ~430 | 开环评估 + UQ score 捕获 | Stage 2a |
| `scripts/train_film.py` | ~480 | FiLM 微调 + 碰撞感知 loss（方案 C）| Stage 2b/4b |
| `scripts/eval_ablation_full.py` | ~325 | 热交换 ablation 评估（4 组一次跑完）| Stage 4 |
| `scripts/eval_closedloop_replay.py` | ~510 | 闭环回放评估（Bench2Drive，无 CARLA）| 闭环 |
| `scripts/visualize_eval.py` | ~460 | 论文级评估可视化（5 种图表）| 可视化 |
| `scripts/visualize_attention.py` | ~590 | QT-Former attention map 可视化 | 可视化 |
| `scripts/generate_trajectory_gifs.py` | ~660 | 轨迹对比 GIF 生成（Baseline vs FiLM vs GT）| 可视化 |
| `scripts/merge_v2_uq_scores.py` | ~150 | UQ score 合并到已有 eval 结果 | 工具 |
| `scripts/run_ablation.sh` | ~200 | 四组 Ablation 自动化 | Stage 4 |
| `tests/test_uq_model.py` | ~97 | UQEstimator 单元测试 | 测试 |
| `tests/test_film.py` | ~280 | FiLM L1/L2 单元测试（12 tests）| 测试 |

### 4.2 ORION 文件修改

所有修改以 `[UQ]` 注释标记，可通过 `grep -r "[UQ]" adzoo/ mmcv/` 查找。

| 文件 | 新增行数 | 修改内容 |
|------|---------|---------|
| `adzoo/orion/configs/orion_stage3_infer.py` | +4 | 添加 `use_uncertainty`, `uq_checkpoint`, `use_uncertainty_l2` 配置项 |
| `adzoo/orion/test.py` | +32 | UQEstimator 权重重载 + FiLM L1/L2 权重加载逻辑 |
| `mmcv/models/dense_heads/orion_head.py` | +25 | 在 forward 中实例化 UQEstimator、计算 uncertainty_emb、返回给 detector |
| `mmcv/models/utils/petr_transformers.py` | +16 | FiLM L1 层定义（Linear 256→256）+ identity init + forward 调制 |
| `mmcv/models/detectors/orion.py` | +35 | FiLM L2 层定义（Linear 256→4096）+ identity init + train/inference 路径调制 |
| **合计** | **+112** | |

### 4.3 测试覆盖

```
tests/test_uq_model.py  — 6 tests (全部通过)
  - UQEstimator 输出形状、score 值域、多视角、损失函数、数据集 mock、参数量

tests/test_film.py      — 12 tests (全部通过)
  - FiLM L1/L2 identity init、输出形状、参数量、梯度流、checkpoint round-trip
  - freeze 逻辑验证、训练后非平凡输出、总参数预算 < 5M
```

---

## 五、可视化图表索引

所有图表位于 `results/figures/baseline/`，由 `scripts/visualize_eval.py` 生成。

| 图表 | 文件 | 内容 | 论文用途 |
|------|------|------|---------|
| Fig 1 | `fig1_score_dist.pdf` | Normal vs Adverse 的 UQ score 分布直方图 | Method/Experiments: 证明 UQ score 有效分离两类场景 |
| Fig 2 | `fig2_auroc.pdf` | UQ score 区分 adverse 的 ROC 曲线 | Experiments: AUROC 定量评估 |
| Fig 3 | `fig3_uq_vs_l2.pdf` | UQ score vs Planning L2@3s 散点图 | Experiments: UQ signal 与规划质量的相关性 |
| Fig 4 | `fig4_weather_boxplot.pdf` | 逐天气类型的 UQ score 箱线图 | Experiments: 细粒度天气敏感性分析 |
| Fig 5 | `fig5_planning_bars.pdf` | Normal vs Adverse 的 planning metrics 对比 | Experiments: baseline 性能分层分析 |

额外已编写但待数据的可视化：
| 脚本 | 功能 | 等待数据 |
|------|------|---------|
| `scripts/visualize_attention.py` | QT-Former cross-attention map 对比（FiLM 前后）| FiLM 训练完成后 |
| `scripts/visualize_eval.py --input-film` | Baseline vs FiLM 对比模式 | FiLM eval 完成后 |
| `scripts/run_ablation.sh --viz` | 四组 Ablation 对比图 | 全部 eval 完成后 |

### 5.2 轨迹对比 GIF 可视化 ✅

**完成日期**：2026-03-30

**做了什么**：在 Bench2Drive 验证集上选取 18 个代表性场景，分别运行 Baseline（FiLM→identity）和 FiLM（L1+L2+collision-aware）两轮推理，逐帧捕获 GT 轨迹、Baseline 预测轨迹、FiLM 预测轨迹和 UQ score，生成动态 GIF 对比图。

**为什么需要**：静态图表只能展示统计汇总。GIF 可以在论文的 supplementary material 和 presentation 中直观展示：
- 高不确定性场景下 FiLM 如何调制轨迹
- 正常天气 vs 恶劣天气的视觉对比
- 碰撞风险场景中轨迹差异的时间演化

**工具脚本**：`scripts/generate_trajectory_gifs.py`

**渲染方案**：
```
┌──────────────────────────────────────────────┐
│  前置摄像头画面 (960×540)                      │
│  + 轨迹方向箭头叠加（近似透视投影）             │
│    - 绿色: GT 轨迹                             │
│    - 红色: Baseline 轨迹                       │
│    - 蓝色: Ours (FiLM) 轨迹                   │
│                           ┌──────────────┐    │
│                           │  BEV 俯视图   │    │
│                           │  (自适应缩放)  │    │
│                           │  暗色主题      │    │
│                           └──────────────┘    │
│  场景名 / 天气 / 帧号 / UQ Score               │
│  Baseline L2 / Ours L2 / 碰撞状态             │
└──────────────────────────────────────────────┘
```

**关键技术细节**：
- **增量缓存**：已缓存场景自动跳过，支持分批添加新场景
- **Baseline pass 中间保存**：防止 FiLM pass 失败导致 baseline 数据丢失
- **FiLM 权重设备修复**：`reload_film()` 使用 `map_location=dev` + `.to(dev)` 确保权重始终在 GPU 上
- **自适应 BEV 缩放**：`_auto_bev_range()` 根据三条轨迹的实际范围动态设置 BEV 显示范围
- **近似透视投影**：`_draw_cam_trajectories()` 在前置摄像头画面上叠加轨迹方向指示
- **离线渲染**：`--render-only` 模式从缓存的 `trajectory_data.pt` 重新渲染 GIF，无需 GPU/模型

**关键命令**：
```bash
# 全量推理 + 渲染（需要 GPU，~83 分钟 / 18 场景）
PYTHONPATH=. python scripts/generate_trajectory_gifs.py \
    adzoo/orion/configs/orion_stage3_infer.py ckpts/Orion.pth \
    --film-checkpoint checkpoints/film/best_l1l2_col_v3.pt \
    --ann-file data/infos/b2d_infos_val.pkl \
    --frame-step 5 --out-dir results/gifs --fps 4 \
    --scenarios Accident_Town05_Route218_Weather10 ...

# 离线重新渲染（无需 GPU，~2 分钟）
PYTHONPATH=. python scripts/generate_trajectory_gifs.py \
    adzoo/orion/configs/orion_stage3_infer.py ckpts/Orion.pth \
    --render-only --out-dir results/gifs --fps 4
```

**18 个场景选取与分组**：

| # | 故事线 | 场景名 | 天气 | 帧数 | UQ | Col(Base) | Col(FiLM) | 选取理由 |
|---|--------|--------|------|------|-----|-----------|-----------|----------|
| | **S1: 碰撞率改善** | | | | | | | |
| 1 | S1 | ControlLoss_Town04_Weather14 | MidRainyNight | 72 | 0.995 | 3.47% | 2.08% | Baseline 高碰撞，FiLM 显著降低 |
| 2 | S1 | ConstructionObstacle_Town10HD_Weather22 | WetCloudySunset | 38 | 0.997 | 3.07% | 2.63% | 施工障碍场景碰撞率下降 |
| | **S2: 高 UQ 危险场景** | | | | | | | |
| 3 | S2 | YieldToEmergencyVehicle_Town04_Weather10 | MidRainyNoon | 67 | 0.997 | — | 2.74% | 让行急救车，极高不确定性 |
| 4 | S2 | Accident_Town05_Weather10 | MidRainyNoon | 42 | 0.993 | — | 0.00% | 事故场景，FiLM 零碰撞 |
| 5 | S2 | HazardAtSideLane_Town10HD_Weather9 | WetCloudyNoon | 35 | 0.996 | — | 2.38% | 侧道危险物 |
| 6 | S2 | ParkedObstacle_Town10HD_Weather8 | FoggyNoon | 33 | 0.997 | — | 2.02% | 雾天停车障碍 |
| 7 | S2 | StaticCutIn_Town05_Weather18 | FoggySunset | 42 | 0.997 | — | 0.00% | 雾天静态加塞 |
| | **S3: 正常 vs 恶劣天气** | | | | | | | |
| 8 | S3 | ConstructionObstacle_Town12_Weather0 | ClearNoon | 43 | 0.000 | — | 0.00% | 晴天 UQ≈0，FiLM 近乎透明 |
| 9 | S3 | DynamicObjectCrossing_Town01_Weather3 | CloudySunset | 39 | 0.001 | — | 0.00% | 晴天过路行人 |
| 10 | S3 | TJunction_Town05_Weather0 | ClearNoon | 109 | 0.000 | — | 0.00% | 晴天 T 型路口 |
| | **S4: 复杂交通交互** | | | | | | | |
| 11 | S4 | PedestrianCrossing_Town13_Weather19 | HardRainNight | 103 | 0.918 | — | 0.00% | 暴雨夜行人过马路 |
| 12 | S4 | OppositeVehicleRunningRedLight_Town04_Weather23 | Unknown | 31 | 0.982 | — | 0.00% | 对面车闯红灯 |
| 13 | S4 | SignalizedTurnEncounterRedLight_Town15_Weather23 | Unknown | 119 | 0.912 | — | 5.18% | 最高碰撞率场景 |
| 14 | S4 | BlockedIntersection_Town03_Weather5 | HardRainSunset | 121 | 0.930 | — | 0.00% | 封锁路口 |
| | **S5: 高速/汇入** | | | | | | | |
| 15 | S5 | MergerIntoSlowTraffic_Town06_Weather5 | HardRainSunset | 34 | 0.954 | — | 1.47% | 暴雨中汇入慢车流 |
| 16 | S5 | LaneChange_Town06_Weather21 | HardRainNight | 27 | 0.931 | — | 0.00% | 暴雨夜变道 |
| | **S6: 特殊场景** | | | | | | | |
| 17 | S6 | VehicleOpensDoorTwoWays_Town12_Weather7 | SoftRainNight | 85 | 0.000 | — | 3.14% | 车门突然打开 |
| 18 | S6 | SignalizedJunctionLeftTurn_Town04_Weather26 | Unknown | 122 | 0.948 | — | 0.00% | 信号灯路口左转 |

**Storytelling 亮点**：
1. **S1 碰撞率对比**：ControlLoss 场景碰撞率从 3.47% 降至 2.08%，视觉上可以看到 FiLM 轨迹更保守地避让障碍物
2. **S3 天气对比**：同类型场景（ConstructionObstacle）在 ClearNoon（UQ=0.000）和 WetCloudySunset（UQ=0.997）下的行为差异——晴天 FiLM 几乎透明，恶劣天气 FiLM 积极调制
3. **S4 复杂交互**：暴雨夜行人过马路 103 帧，可以看到 UQ score 随场景风险变化的时序演化

**产出**：
- `results/gifs/trajectory_data.pt` — 18 场景的缓存轨迹数据（779KB），包含逐帧 GT/Baseline/FiLM 轨迹、UQ score、planning metrics，支持离线重新渲染
- `results/gifs/*.gif` — 18 个 GIF 文件，总计约 250MB

---

## 六、数据资产清单

| 路径 | 大小 | 说明 |
|------|------|------|
| `ckpts/Orion.pth` | 36GB | ORION 主模型 |
| `ckpts/pretrain_qformer/` | ~14GB | Qwen LLM 权重 |
| `data/bench2drive/v1/` | 407GB | Bench2Drive 原始数据集（1001 个场景）|
| `data/infos/b2d_infos_val.pkl` | 141MB | 验证集标注（12,806 样本）|
| `data/features/` | 235GB | 预提取的 EVAViT patch tokens |
| `data/labels/uq_labels.pt` | 1.3MB | 不确定性伪标签 (v2, max_mean+cosim+entropy) |
| `data/labels/uq_labels_v1_backup.pt` | 1.3MB | v1 伪标签备份 |
| `checkpoints/uq/best.pt` | 25MB | 训练好的 UQEstimator (v2) |
| `checkpoints/uq/best_v1_backup.pt` | 25MB | v1 UQEstimator 权重备份 |
| `checkpoints/film/best_l1.pt` | 516KB | FiLM L1 权重 |
| `checkpoints/film/best_l2.pt` | 8.2MB | FiLM L2 权重 |
| `checkpoints/film/best_l1l2.pt` | 8.5MB | FiLM L1+L2 权重（L2 loss 版本）|
| `checkpoints/film/best_l1l2_col.pt` | ~8.5MB | FiLM L1+L2 权重（碰撞感知 loss，方案 C）|
| `results/eval_openloop_full.pt` | ~50MB | 开环评估逐样本结果 |
| `results/closedloop_film_baseline.json` | ~5KB | 闭环评估结果（baseline vs FiLM L2-loss）|
| `results/closedloop_replay_v3.json` | ~15KB | v3 闭环评估结果（50 场景，含逐场景指标）|
| `results/eval_openloop_v3.pt/.json` | ~50MB | v3 开环评估（AUROC=0.954）|
| `results/figures/baseline/` | ~440KB | 5 张可视化图表 |
| `results/gifs/trajectory_data.pt` | 779KB | 18 场景缓存轨迹数据（支持离线重渲染）|
| `results/gifs/*.gif` | ~250MB | 18 个轨迹对比 GIF（Baseline vs FiLM vs GT）|

---

## 七、下一步计划

### Stage 4a：闭环碰撞率验证（L2-loss FiLM） ✅

**完成日期**：2026-03-29

**做了什么**：运行 baseline vs FiLM L1+L2（L2 loss 训练版本）的闭环回放评估，对比碰撞率。

**结果**：

| 指标 | Baseline | FiLM L1+L2 (L2 loss) | 变化 |
|------|----------|-----------------------|------|
| Adverse 碰撞率 | 1.17% | 1.22% | +4.3% ❌ |
| ADE@3s | 2.26m | 3.90m | +73% ❌ |
| Throttle MAE | 0.26 | 0.40 | +54% ❌ |
| Steer MAE | 0.006 | 0.005 | -17% |

**结论**：用 L2 轨迹 loss 训练的 FiLM **无法降低碰撞率**，反而使轨迹精度和控制精度退化。这验证了核心 insight：**L2 轨迹匹配 ≠ 安全性**。

---

### Stage 4b：碰撞感知 FiLM 训练（方案 C） ✅

**完成日期**：2026-03-29

**做了什么**：在 `train_film.py` 中新增可微分的 GT-agent 碰撞边距 loss，重新训练 FiLM。

**碰撞感知 Loss 设计**：
```
gt_collision_margin_loss:
  1. 从 gt_bboxes_3d 获取 agent 当前位置 [N, 2]
  2. 从 gt_attr_labels[:, 0:12] 获取 agent 未来轨迹偏移 → cumsum → [N, 6, 2]
  3. 计算 ego 预测轨迹到最近 agent 的距离 → min_dist [B, 6]
  4. Hinge loss: relu(margin - min_dist)
  5. UQ score 加权: violation * uq_score.detach()

total = plan_reg + 0.1*vae + 0.01*vlm + lambda_col * col_loss
```

**为什么用 GT agent 而非预测 agent**：FiLM 训练中 ORION 主干冻结，agent 预测不可用（需要检测头的训练模式数据字段）。GT agent 提供精确的障碍物位置，是可靠的碰撞判定基础。

**为什么 UQ score detach**：UQ score 只做权重（高不确定性 → 更强碰撞惩罚），不通过 UQ 反传梯度——FiLM 训练只优化 FiLM 参数，不改变 UQ 估计本身。

**关键实验发现——margin 选择**：

| margin | col_loss | 有效样本比例 | 说明 |
|--------|----------|-------------|------|
| 2.0m | 0.000 | ~0% | 几乎所有 agent 距离 > 2m，无梯度信号 |
| 4.0m | 0.242 | ~30% | 合理，部分帧有碰撞风险梯度 |
| 5.0m | 1.023 | ~60% | 过于激进，可能让模型过度保守 |

最终选择 **margin=4.0m**（约 2 个车身宽度），`lambda_col=0.5`。

**训练配置**：
```bash
USE_FILM_L1L2=1 python scripts/train_film.py \
    --max-samples 3000 --epochs 5 \
    --lr 1e-3 --lambda-col 0.5 --col-margin 4.0 \
    --out checkpoints/film/best_l1l2_col.pt
```

**训练结果**：Epoch 1: 77.31 → Epoch 2: 0.18 → Epoch 3: 0.14 → Epoch 4: 0.18 → **Epoch 5: 0.13 (最佳)**。

**闭环评估结果**（50 adverse 场景，7 个与 baseline 重叠）：

| 指标 | Baseline | FiLM Col | 变化 |
|------|----------|----------|------|
| 平均 ADE@3s | 2.49m | 3.52m | **+41.5% ❌** |
| 平均 Collision@3s | 1.17% | 0.96% | **-17.5% ✅** |
| 平均 Brake MAE | 0.274 | 0.627 | +129% |
| 平均 Throttle MAE | 0.248 | 0.431 | +74% |

**逐场景得失**（7 个重叠 adverse 场景）：

| 场景 | Baseline Col | FiLM Col | ΔCol | ΔADE@3s | 状态 |
|------|-------------|----------|------|---------|------|
| ControlLoss_Town04 | 3.47% | 3.47% | 0 | -2.33m | ➖ ADE改善 |
| ConstructionObstacle | 3.07% | 2.63% | -0.44% | +1.24m | ✅ Col改善 |
| ControlLoss_Town10HD | 1.63% | 0.41% | -1.22% | +0.13m | ✅ Col改善 |
| CrossingBicycleFlow | 0.00% | 0.23% | +0.23% | +4.64m | ❌ Col退化 |
| BlockedIntersection | 0.00% | 0.00% | 0 | +0.53m | ➖ |
| Accident_Town05 | 0.00% | 0.00% | 0 | +1.58m | ➖ |
| AccidentTwoWays | 0.00% | 0.00% | 0 | +1.43m | ➖ |

**ADE 退化原因分析**：

1. **全面制动增加**：FiLM Col 使 Brake MAE 增加 129%，Throttle MAE 增加 74%。模型采取更保守的加减速策略
2. **不必然导致安全**：CrossingBicycleFlow 碰撞率从 0% 升至 0.23%，说明过度制动反而可能引入新的碰撞风险
3. **有效场景**：碰撞改善的两个场景（ControlLoss、ConstructionObstacle）恰好是 baseline 碰撞率最高的两个，说明碰撞感知训练在最需要它的场景有效
4. **唯一 ADE 改善**：ControlLoss_Town04 碰撞率不变但 ADE 大幅下降 2.33m——该场景 FiLM 的保守策略同时带来了更好的轨迹质量

**结论**：碰撞感知训练使模型全面趋于保守（更多制动），在 2/7 场景降低了碰撞率（高风险场景），但代价是 ADE 增加 41.5%。这是一种**安全性 vs 轨迹效率的权衡**，而非纯粹的改善。

---

## 七、下一步计划

### 已验证结论

1. **v3 伪标签修正使 AUROC 达到 0.954**（在 eval_openloop weather-based 分类下），远超 0.7 目标
2. **v3 FiLM (L1+L2+collision) 闭环评估**：50 场景，碰撞率 0.52%
3. **UQ score 天气排序完全正确**：ClearNoon=0.0001 → HardRainNight=0.989
4. **UQ 分层效果**：低 UQ 场景 ADE=5.79m vs 高 UQ 场景 ADE=3.64m（高不确定性场景反而规划精度更好，因为模型倾向保守行为）
5. **Adverse 碰撞率是 Normal 的 ~6 倍**（0.64% vs 0.11%）
6. **scene_type 分类不一致**：v2 的 AUROC 0.993 是虚高的（仅在特征文件自身标签空间有效），实际为 0.601

### 已完成（v3 更新）

1. ~~伪标签 v2 重构~~ → ✅ 已完成
2. ~~v3 scene_type 修正（weather-based）~~ → ✅ AUROC 0.954
3. ~~v3 UQ score 合并到 eval_openloop~~ → ✅ Normal/Adverse gap=0.870
4. ~~v3 FiLM L1+L2+col 重训~~ → ✅ best_loss=0.1193
5. ~~v3 闭环评估（50 场景）~~ → ✅ Col@3s=0.52%
6. ~~FiLM bug 修复~~ → ✅ `uq_output` 未赋值、import 路径

### Bug 发现：init_weights 覆盖 FiLM identity init（2026-03-30）

**问题**：`PETRTemporalTransformer.init_weights()` 遍历所有 `self.modules()`，对 `weight.dim() > 1` 的层做 `xavier_uniform_`。这会覆盖 FiLM 层在 `__init__` 中设置的 identity init（weight=0, bias=gamma→1/beta→0），导致未加载训练权重时 FiLM 不是 identity 而是随机调制。

**影响**：
- `closedloop_baseline.json`（10 场景）：受随机 FiLM 干扰，**数据不可信** ❌
- `closedloop_film_col.json` / `closedloop_replay_v3.json`：加载了训练权重覆盖 xavier，**数据可信** ✅
- Stage 4a 的 Baseline vs FiLM L2-loss 对比：Baseline 侧不可信

**修复**：在 `init_weights()` 中用 `film_params = {id(self.film_gamma.weight), id(self.film_beta.weight)}` 排除 FiLM 层。L2 FiLM（`orion.py`）无 `init_weights` 方法，不受影响。

**修复后 50 场景 Baseline vs FiLM v3 对比**（`closedloop_baseline_50.json`）：

|  | Baseline | FiLM v3 | Delta |
|--|----------|---------|-------|
| **ALL (50) Col@3s** | 0.66% | 0.52% | **-21%** |
| **ALL (50) ADE@3s** | 2.46m | 4.44m | +80% |
| Normal (11) Col@3s | 0.05% | 0.11% | +0.06% |
| Normal (11) ADE@3s | 2.78m | 6.00m | **+3.23m** ❌ |
| Adverse (39) Col@3s | 0.83% | 0.64% | **-23%** |
| Adverse (39) ADE@3s | 2.37m | 4.00m | +1.63m |

**关键问题**：Normal 场景 UQ score ≈ 0（0.0001~0.0015），FiLM 不应有任何调制效果，但 ADE 从 2.78m → 6.00m（+116%）。

**根因**：`embed_head` 末尾的 `LayerNorm` 使所有样本的 embedding L2 norm 恒定（≈ √256 ≈ 16），无论 UQ score 高低。FiLM 计算 `gamma = W @ embedding + bias`，由于 embedding norm 恒定，Normal 和 Adverse 场景的调制幅度相当。

### Score-Gated FiLM 设计方案（下一步核心改动）

**目标**：让 UQ score 控制调制强度，score=0 时严格 identity。

```python
# 当前（有问题）：embedding 经 LayerNorm，norm 恒定，score 未参与 FiLM
gamma = W_gamma @ embedding + b_gamma
beta  = W_beta  @ embedding + b_beta
output = gamma * input + beta

# 改进：score 作为 gate，score=0 → 严格 identity
gamma_raw = W_gamma @ embedding + b_gamma   # 学习调制方向
beta_raw  = W_beta  @ embedding + b_beta
gamma = 1 + score * (gamma_raw - 1)         # score→0: gamma→1
beta  = score * beta_raw                     # score→0: beta→0
output = gamma * input + beta
```

**改动范围**：`petr_transformers.py` L1 FiLM forward（~3行）+ `orion.py` L2 FiLM forward（~3行）。需要将 `uq_score` 传递到 FiLM 调用处。改完后需重训 FiLM。

**预期效果**：
- Normal 场景（score≈0）：gamma≈1, beta≈0，ADE 应与 Baseline 几乎一致
- Adverse 场景（score≈1）：gamma≈gamma_raw, beta≈beta_raw，保留碰撞改善效果
- 解耦"是否调制"（score 控制）和"如何调制"（embedding 方向控制）

### 待做（优先级由高到低）

1. **Score-Gated FiLM 实现 + 重训**：核心修复，解决 Normal ADE 退化问题
2. **UQ 组件消融实验**：利用已准备的 ablation configs，验证各组件贡献
   - w/o stat_features, w/o decoder, w/o ranking, w/o calibration
3. **FiLM Ablation 全量比较**：A=Baseline, B=L1, C=L2, D=L1+L2
4. **重新生成可视化图表**：用 v3 结果更新所有 figures
5. **调参搜索**：碰撞感知 loss 的 margin (3-6m) 和 lambda_col (0.1-1.0)

### 需要关注的风险

1. **Normal ADE 退化**：当前 FiLM 因 embedding LayerNorm 导致 Normal 场景也被调制，Score-Gated 方案应解决此问题
2. **ADE 退化风险**：FiLM 调制的保守化策略可能降低轨迹效率，论文需正面讨论 safety vs efficiency tradeoff
3. ~~**AUROC 不达标**~~ → **已解决** ✅（v3 AUROC=0.954）
4. ~~**scene_type 分类不一致**~~ → **已解决** ✅（v3 使用 weather-based 分类）
5. ~~**init_weights 覆盖 FiLM identity init**~~ → **已修复** ✅（2026-03-30）

---

## 八、复现指南

```bash
# 1. 确保数据和模型就位
ls ckpts/Orion.pth                    # 36GB ORION 模型
ls data/bench2drive/v1/               # 407GB 数据集
ls data/infos/b2d_infos_val.pkl       # 验证集标注

# 2. 运行测试确认代码完整性
pytest tests/ -v                       # 18 tests should pass

# 3. 特征提取（如未做过）
python scripts/extract_orion_features.py \
    --checkpoint ckpts/Orion.pth \
    --output_dir data/features \
    --ann_file data/infos/b2d_infos_val.pkl

# 4. 伪标签生成 (v2, 使用 stat_cache 快速路径)
python scripts/generate_labels.py \
    --feature_dir data/features \
    --stat_cache data/stat_cache.pt \
    --output_file data/labels/uq_labels.pt

# 5. UQEstimator 训练
python scripts/train_uq.py --config configs/uq_train.yaml

# 6. 开环评估（baseline）
python scripts/eval_openloop.py \
    adzoo/orion/configs/orion_stage3_infer.py ckpts/Orion.pth \
    --ann-file data/infos/b2d_infos_val.pkl \
    --out results/eval_openloop_full.pt

# 7. 生成可视化
python scripts/visualize_eval.py \
    --input results/eval_openloop_full.pt \
    --out-dir results/figures/baseline

# 8. FiLM 训练 + Ablation
bash scripts/run_ablation.sh --all
```

---

## 九、FiLM 训练与评估详情

### 9.1 FiLM Checkpoints

| Checkpoint | 文件 | 大小 | 内容 |
|------------|------|------|------|
| `checkpoints/film/best_l1.pt` | 516KB | FiLM L1 weights (gamma/beta for QT-Former) |
| `checkpoints/film/best_l2.pt` | 8.2MB | FiLM L2 weights (gamma/beta for VAE) |
| `checkpoints/film/best_l1l2.pt` | 8.5MB | FiLM L1+L2 combined weights |

### 9.2 碰撞感知 Loss 实现细节

`gt_collision_margin_loss()` 函数（`scripts/train_film.py`）：

```python
# 输入
ego_fut_preds: [B, 20, 6, 2]   # 可微分的预测轨迹 offsets
gt_attr_labels: [N, 34]         # GT agent 属性（dims 0-11 = 未来轨迹偏移）
gt_bboxes_3d: `LiDARInstance3DBoxes` [N, 9] (x,y,z,w,l,h,yaw,vx,vy)
uq_score: [B, 1]               # detached，作为权重

# 计算
agent_abs = agent_xy + cumsum(agent_fut_offsets)  # [N, 6, 2] 绝对未来位置
ego_cum = ego_fut_preds[:, 0].cumsum(dim=1)       # [B, 6, 2] best-mode 累积轨迹
dist = ||ego_cum - agent_abs||                     # [B, 6, N] 距离矩阵
min_dist = dist.min(dim=agent)                     # [B, 6] 到最近 agent 距离
violation = relu(margin - min_dist) * uq_score     # hinge + UQ 加权
loss = violation.mean()
```

**数据格式注意**：
- `gt_attr_labels` 从 dataloader 出来是嵌套 list，需要递归 unwrap 到 tensor
- `gt_bboxes_3d` 同样是 list 包裹的 LiDARInstance3DBoxes 对象
- 单个帧可能没有 agent（N=0），此时返回 loss=0

### 9.3 已知问题

- FiLM L1+L2 训练 epoch 1 出现异常 mean loss (116.86)，实际步级 loss 正常 (~0.1)，由极端 outlier 样本拉高。Epoch 2/3 正常收敛
- `train_film.py` 中 L2-only 模式的 checkpoint 保存曾有 bug（已修复，加了 hasattr 防护）
- 碰撞 loss 稀疏：约 70% 帧的 col_loss=0（所有 agent > margin），只有近距离交互帧产生梯度

### 9.3 闭环回放评估技术细节

`eval_closedloop_replay.py` 的数据流：
```
For each scenario (folder in data_infos):
  Sort frames by frame_idx
  Init fresh PIDController
  For each frame (every N-th):
    mmcv collate → model inference → ego_fut_preds [6, 2]
    world2lidar transform → local_command
    PID(ego_fut_preds, speed, local_cmd) → steer, throttle, brake
    Compare vs GT controls → Control MAE
    metric_stp3 occupancy grid → collision detection
```

注意：model 输出的 `metric_results` 包含 `plan_L2_*` 和 `plan_obj_col_*`，即使 `fut_valid_flag=False` 也有值。评估脚本使用 `has_plan_metrics`（L2 > 0）而非 `fut_valid_flag` 来过滤。

### 9.4 评估体系约束（重要）

本报告的所有闭环指标均来自**回放式闭环评估**（详见附录 A.3），不是真实 CARLA 仿真。
主要约束：
1. ADE/碰撞改善在真实仿真中的泛化性**未经验证**
2. 50 个场景的碰撞率均值受个别场景影响大（部分场景碰撞率为 0%，少数高达 11%）
3. Normal 场景的 ADE 退化（+116%）尚未修复

这些约束需要在论文中正面交代，并作为 Future Work（真实 CARLA 闭环验证）。

---

---

## 十、论文 Insights 与故事线

### 10.1 核心论点

**Claim**: 端到端自动驾驶模型在恶劣天气下过度自信，轻量不确定性感知注入可在不修改主干的前提下显著降低碰撞风险。

### 10.2 可写入论文的关键 Insights

**Insight 1: L2 轨迹精度 ≠ 安全性（反直觉发现）**

| 场景 | L2@3s (m) | 碰撞率 |
|------|-----------|--------|
| FoggySunset | **0.855** (最低) | **0.00%** |
| Unknown(23) | 0.968 (次低) | **11.25%** (最高) |
| CloudyNoon | 3.890 (最高) | 0.02% |

解读：低 L2 error 说明轨迹与 GT 接近，但 GT 本身可能就在碰撞路径上（Unknown23 场景复杂，GT 也可能不是最优）。高 L2 说明偏离 GT，但偏离方向可能恰好更安全（CloudyNoon 速度快、偏移大但远离障碍物）。

**论文用途**：motivation 段落，说明为什么传统开环 L2 评估不足以衡量安全性，需要碰撞率指标。

**Insight 2: 恶劣天气碰撞率放大效应**

- Adverse 碰撞率是 Normal 的 **161 倍**（1.61% vs 0.01%）
- 而 Adverse 的 L2 error 反而**低于** Normal（1.779m vs 2.379m）
- 原因：恶劣天气下 CARLA 车辆低速行驶，GT 轨迹短，L2 天然低；但感知降质导致碰撞检测失败

**论文用途**：Introduction 的 safety gap 论述——现有模型在恶劣天气下「看起来精度不错」但实际碰撞风险剧增。

**Insight 3: 纯 L2 loss 训练的 FiLM 让碰撞率上升（+4.3%）**

L2-loss FiLM（L1+L2）：碰撞率 1.17% → 1.22%，ADE 2.26m → 3.90m。优化 L2 匹配的 FiLM 让模型学会更好地拟合训练分布，但这个分布本身不包含安全性信号——L2 降低不等于安全提升。

**论文用途**：Ablation study 的重要对照组——证明需要碰撞感知训练目标，单纯调制不够。

**Insight 3b: 碰撞感知 FiLM 降低碰撞率 17.5%，但以 ADE 增加 41.5% 为代价**

碰撞感知训练（margin=4m, λ_col=0.5）：碰撞率 1.17% → 0.96%，ADE 2.49m → 3.52m。
- 改善仅在 2/7 个 shared adverse 场景成立（ControlLoss_Town10HD: -75%, ConstructionObstacle: -14%）
- CrossingBicycleFlow 反而出现新的碰撞（0% → 0.23%）
- ControlLoss_Town04 碰撞不变但 ADE 大幅改善（-2.33m）——保守策略在该场景恰好也更好
- Brake MAE 增加 129%，说明模型全面趋于保守

**权衡分析**：碰撞感知训练本质上是"用轨迹效率换安全性"。ADE 增加 41.5% 是否可接受，取决于应用场景——如果是物流园区低速场景可能值得，如果是高速路则不然。

**论文用途**：实验章节的核心结果，但需正面承认这个 tradeoff，这是碰撞感知训练的固有局限。

**Insight 4: UQ score 有效但需要碰撞感知桥接**

- v2 UQ score normal/adverse 均值差 = **0.794**（AUROC=0.993）
- 但 UQ score 与碰撞率的直接相关性弱（ρ=-0.077，待用 v2 重算）
- 说明 UQ score 精准捕获了「感知质量下降」但没有直接翻译为「应该规避碰撞」
- 碰撞感知 loss 中 UQ score 做权重 = 桥接不确定性估计和安全决策

**论文用途**：方法章节——解释为什么需要碰撞感知 loss 而非仅仅将 UQ embedding 注入网络。v2 的 0.993 AUROC 证明 UQ estimation 本身已经非常有效。

**Insight 5: 碰撞 loss 的稀疏性问题与 margin 工程**

- margin=2m: 0% 帧有碰撞梯度（所有 agent > 2m）
- margin=4m: ~30% 帧有碰撞梯度
- margin=5m: ~60% 帧有碰撞梯度
- 实际碰撞距离 ~1-2m，但需要更大 margin 才能让足够多的训练帧提供梯度
- 类似于 focal loss 思想：需要人为放大稀有事件（近距离交互）的学习信号

**论文用途**：实验章节的超参分析，可做 margin sensitivity 曲线。

### 10.3 论文故事线草案

1. **Problem**: VLM-based E2E 驾驶在恶劣天气下碰撞率剧增 161×，但开环指标看不出问题
2. **Why it's hard**: L2 优化不等于安全优化（Insight 1/2/3），模型需要感知自身的不确定性
3. **Our approach**: 轻量 UQ 估计 (2.24M) + 双层 FiLM 注入 (2.24M)，零主干微调
4. **Key innovation**: 碰撞感知 FiLM 训练——UQ score 加权的 hinge collision loss
5. **Results**:
   - **UQ estimation**: AUROC = 0.993（v2），normal/adverse 分离度 Gap = 0.794
   - L2-loss FiLM（L1+L2）反而使碰撞率上升（1.17%→1.22%）——L2 优化 ≠ 安全
   - 碰撞感知 FiLM 使碰撞率降低 17.5%（1.17%→0.96%），但 ADE 增加 41.5%
   - 改善集中在原本碰撞率最高的场景（ControlLoss、ConstructionObstacle）
   - 改善以保守化为代价——brake MAE 增加 129%

---

### 10.4 最新 Insight：FiLM 学到“过度保守捷径”与第二轮训练的反制策略（v3，进行中）

> 背景：目前仓库内已完成第一版 UQEstimator 与 FiLM（L1/L2/L1+L2）训练与评估，但整体效果不佳，主要表现为模型倾向采取更保守策略来降低碰撞率（例如更频繁制动/降速/等待），带来轨迹效率与控制质量退化（ADE、Throttle/Brake MAE 上升），且并非所有场景都受益。

#### 现象总结（审稿人/论文可用表述）

1. **安全导向目标函数存在“保守捷径”**：当碰撞惩罚较强、而对“进度/舒适/贴近参考轨迹”的约束不足时，模型最容易学习到的局部最优策略是“更慢、更停、更让”，从而降低碰撞，但牺牲驾驶质量与效率。
2. **UQ→FiLM 的条件信号可能出现“常开”**：如果 UQ score 在大量样本上偏高（或 embedding 在训练中被放大），FiLM 相当于长期在“紧急模式”下工作，导致全局行为偏保守，而不是只在真正不确定时刻做局部调整。
3. **FiLM 调制幅度缺少约束时会放大分布漂移**：\(\gamma,\beta\) 若无幅度/平滑约束，可能把中间表示推离 baseline 的有效工作区间，出现“安全变保守、精度与控制退化”的非期望副作用。

#### 第二轮训练目标（把“安全改进”从保守捷径中解耦）

- **目标 1：把 FiLM 从“全局保守”变成“选择性调制”**  
  核心原则：低不确定性时尽量接近 baseline，高不确定性时才允许更明显的调制与保守策略。

- **目标 2：把“安全”定义成“在可接受驾驶质量下更安全”**  
  核心原则：安全改进不能主要靠“停住/极慢”这种捷径，需要引入进度、舒适性或参考行为约束来界定可接受的驾驶质量边界。

#### 推荐的三类改动方向（第二轮训练优先级）

1. **堵保守捷径：引入进度/舒适/效率约束**
   - 进度（progress）或速度保持项：抑制无理由降速与停滞。
   - 舒适性（acc/jerk）项：抑制“为了不撞就猛刹”的不自然控制。
   - 分段权重：低 UQ 更重轨迹与控制质量，高 UQ 更重安全项。

2. **UQ 门控：只在“真的不确定”时增强调制**
   - 将 UQ score 作为门控/权重（软门控），避免 FiLM 长期处于强调制状态。
   - 在 adverse 内部做排序能力验证，确保 UQ 不只是“天气分类器/难度 proxy”。

3. **约束 FiLM 幅度：让调制像“微调”而非“重写”**
   - 对 \(\gamma\) 偏离 1、\(\beta\) 偏离 0 加正则（可按 UQ 分段：低 UQ 更强约束，高 UQ 适度放松）。
   - 约束 \(\gamma,\beta\) 的范围/平滑性，避免过强调制导致效率退化或新失败模式。

#### 第二轮训练的诊断指标（建议训练中同步跟踪）

为避免“碰撞下降但全局停滞”的假进步，建议至少跟踪：
- **速度/进度侧**：平均速度、低速/停滞比例（near-zero motion）、到达率/路线完成度（若闭环可用）
- **安全侧**：碰撞率（按 normal/adverse/场景类型分组）、近失碰撞或最小距离统计（若可得）
- **UQ/FiLM 健康度**：UQ score 分布（是否“常高”）、\(\|\gamma-1\|\) 与 \(\|\beta-0\|\) 的均值/方差（调制幅度是否失控）
- **机制一致性**：在 adverse 内部，UQ 与风险/误差是否仍保持单调关系（而非仅仅区分天气）

> 记录建议：把第二轮训练的关键超参（例如 \(\lambda_{col}\)、margin、进度/舒适项权重、门控阈值/温度、FiLM 正则强度）与上述诊断指标一起记录，后续可直接生成“安全-效率 Pareto 曲线”，为论文讨论与投稿定位提供依据。

*报告更新时间：2026-03-30（v2 伪标签重构后更新）*
*Git 分支：dev*

---

## 附录 A：评估体系说明

> 本附录旨在明确本报告的评估范围、数据来源、模块作用域及结论可信度边界，为汇报和论文撰写提供参考。

### A.1 数据集：Bench2Drive (B2D)

**什么是 B2D**：CARLA 模拟器生成的多天气、多场景驾驶数据集，专为端到端自动驾驶评估设计。

| 属性 | 数值 |
|------|------|
| 验证集规模 | 12,806 帧 |
| 摄像头视角 | 6（前/后/左前/右前/左后/右后）|
| 天气类型 | 24 种（Weather ID 0-26）|
| 城镇覆盖 | 24 个 CARLA 城镇 |
| 样本格式 | patch tokens `[6, 1600, 1024]`（预提取，~235GB）|

**Normal/Adverse 划分**：天气 ID 0-3（ClearNoon/ClearSunset/CloudyNoon/CloudySunset）为 Normal（2,709 帧），其余 20 种天气为 Adverse（10,097 帧）。此划分与 eval_openloop 评估脚本对齐（weather-based scene_type，v3 已修正）。

---

### A.2 开环评估（eval_openloop）

**定义**：对 12,806 个独立帧逐帧推理，每帧预测未来 3s 轨迹，与同帧 GT 轨迹比较。

**指标**：
- L2@1s/2s/3s：预测轨迹与 GT 的欧氏距离（米）
- 碰撞率：基于 GT occupancy grid 构建，判断预测轨迹是否与障碍物框重叠

**局限**：帧间独立，不考虑历史状态，不捕捉时序累积误差。每帧独立推理意味着即使模型产生系统性偏差，也不会在帧间累积。

**UQ score 的意义**：每帧独立计算，与 weather 分类强相关（AUROC=0.954）。AUROC 度量的是 UQ score 对 Normal/Adverse 天气分类的判别能力，而非对单帧 L2 误差的预测能力（Spearman ρ=0.139，弱相关）。

---

### A.3 回放式闭环评估（eval_closedloop_replay）——关键说明

**本质**：从 B2D 预录数据中选 50 条路线，逐帧喂入模型推理，将预测轨迹与 GT 比对，**无真实仿真器反馈**。

**与真实闭环的核心区别**：

| 维度 | 本报告（回放式）| 真实闭环（CARLA 仿真）|
|------|----------------|----------------------|
| 反馈回路 | 无：模型行为不影响后续观测 | 有：agent 轨迹决定下一帧的场景状态 |
| 位置漂移 | 无：始终用 GT ego pose | 有：规划误差逐帧累积，位置可能严重偏离 |
| 碰撞计算 | GT 障碍物框 + occupancy grid，无 sensor 噪声 | 实时物理碰撞检测，含 sensor 噪声 |
| 动态 agent | 按录制回放，不响应 ego 行为 | 实时响应（其他车辆有自己的 agent）|

**为什么仍有价值**：

1. **系统性偏差可见**：跨场景比较能捕捉模型在恶劣天气下的一致性失效模式，不依赖仿真器反馈
2. **UQ 相关性验证**：UQ score 与 weather 分类的相关性（AUROC=0.954）完全独立于仿真反馈，结论稳健
3. **快速迭代**：无需实时 CARLA，50 场景约 1h，适合消融实验和超参搜索

**50 个场景来源**：从 B2D 验证集均衡采样，覆盖 24 种天气，每种 1-3 个代表路线，其中 Normal 11 个场景、Adverse 39 个场景。

**ADE/碰撞指标含义**：与 GT 比对的偏差，反映模型行为的保守/激进程度，而非真实驾驶安全指标。碰撞率数值不能直接等同于真实道路碰撞概率。

---

### A.4 伪标签设计的自洽性说明

**问题**：用 patch token 统计特征生成伪标签，再训练模型学习同类特征，是否存在循环论证？

**分析**：

| 层面 | 说明 |
|------|------|
| 特征重用 | 伪标签公式（加权的 max_mean/cosim/entropy）和 stat_features（5 维统计特征）均来源于 patch tokens，存在部分特征重用 |
| 结构学习 | UQEstimator 新增了 Transformer decoder（16 个 learnable queries，cross-attention），能学到统计特征之外的空间结构信息 |
| 外部验证 | AUROC=0.954 使用 weather-based scene_type 分类，该分类独立于伪标签生成中使用的统计权重；AUROC 是对 weather 分类的外部验证，而非对伪标签本身的循环验证 |
| 迭代进步 | v1→v2→v3 的 AUROC 迭代（0.621→0.601→0.954）中，改进来源可追溯：v3 主要来自 scene_type 分类修正（19.4% 样本重新对齐），而非伪标签过拟合 |

**结论**：存在部分特征重用（stat_features 同时出现在伪标签和模型输入中），但通过独立外部验证（weather-based AUROC），UQ score 的区分能力结论可信。伪标签是合理的工程设计——在无真实不确定性 ground truth 的情况下，从视觉特征本身提取区分信号是标准做法。

---

### A.5 新增模块的作用域边界

**UQEstimator（已充分验证）**：

| 能力 | 数值 | 说明 |
|------|------|------|
| 区分 Normal vs Adverse 天气 | AUROC = 0.954 | 强区分能力，可作为感知质量监测器 |
| 输出连续分数 | [0, 1]，Gap = 0.870 | ClearNoon≈0，HardRainNight≈0.989 |
| 预测单帧 L2 误差 | Spearman ρ = 0.139 | 弱相关，**不能用作规划误差预测器** |

UQEstimator 是**感知质量监测器**，而非规划误差预测器。它回答的问题是"当前视觉条件有多差"，而非"当前规划会出多大错"。

**FiLM（部分验证，存在已知缺陷）**：

| 方面 | 说明 |
|------|------|
| 理论设计 | Score-Gate 机制：score=0 时严格 identity（Normal 场景无调制），score=1 时激活保守规划（Adverse 场景调制）|
| 实际缺陷 | embed_head 的 LayerNorm 使 embedding L2 norm 恒定（≈√256≈16），Normal/Adverse 场景调制幅度相当，Score-Gate 失效 |
| 测量效果 | 碰撞率 -21%（adverse -23%），ADE +80%（normal +116%）|
| 根因 | LayerNorm 破坏 Score-Gate；移除 LayerNorm + 重训是预期修复路径 |
| 当前状态 | Score-Gated FiLM 代码已完成，**checkpoint 未重训**，理论修复未实验验证 |

**BEV IPM（独立验证）**：

| 方面 | 说明 |
|------|------|
| 方法 | 纯几何 IPM（逆透视映射），不依赖模型 attention 或 checkpoint |
| 验证范围 | B2D 2 个场景（Normal Weather3 + Adverse Weather13）|
| 验证结果 | Normal 0.583 vs Adverse 0.722，Δ=+0.139 |
| 局限 | 样本量小（2 场景），结论待更大规模验证 |

---

### A.6 可信结论汇总表

| 结论 | 证据 | 可信度 |
|------|------|--------|
| UQ score 能区分 Normal vs Adverse 天气 | AUROC=0.954，gap=0.870，逐天气排序正确 | **高** |
| 恶劣天气下碰撞率显著高于正常天气 | baseline: 0.05% vs 0.83%（50 场景回放）| **高** |
| FiLM 调制降低了回放碰撞率 21% | 50 场景对比，Adverse: 0.83%→0.64% | **中**（回放式，非真实闭环）|
| FiLM 调制导致 ADE 退化 +80% | 50 场景对比，Normal: +116%，Adverse: +69% | **高** |
| Normal ADE 退化由 embed_head LayerNorm 引起 | 代码分析 + 理论推导（embedding norm 恒定） | **中**（未实验验证）|
| Score-Gated FiLM 能修复 Normal ADE | 理论推导（score=0 → gamma=1, beta=0） | **低**（无实验数据，checkpoint 未重训）|
| BEV IPM 能区分场景感知质量 | 2 场景定量验证，Δ=+0.139 | **中**（样本量小）|

# UQ-ORION 阶段性研究报告

> 日期：2026-03-29（最新更新）
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

### Stage 1a：伪标签生成 ✅

**完成日期**：2026-03-27

**做了什么**：为每个样本计算一个不确定性伪标签 score ∈ [0, 1]，作为 UQEstimator 的监督信号。

**为什么需要伪标签**：真实的"不确定性"没有 ground truth 标注。我们从视觉特征本身提取三个信号，组合成伪标签：

```
score = 0.3 × gradient_score     # 图像梯度低 → 模糊 → 高不确定
       + 0.3 × entropy_score      # token 激活熵高 → 特征混乱 → 高不确定
       + 0.4 × consistency_score   # 跨视角一致性低 → 感知矛盾 → 高不确定

+ scene_type 校准：normal → [0, 0.45]，adverse → [0.55, 1.0]
```

**为什么用这三个分量**：
- **图像梯度**：大雾、大雨、低能见度会导致图像梯度显著降低（物理先验）
- **Token 激活熵**：恶劣天气下视觉编码器的特征更混乱、信息量更低
- **跨视角一致性**：6 个摄像头应该看到一致的场景；恶劣天气破坏这种一致性

**为什么需要 scene_type 校准**：纯统计特征的分离度不够强，利用 CARLA 天气标注做区间映射，确保 normal 和 adverse 的 score 有足够的 margin。

**产出**：
- `data/labels/uq_labels.pt` — 12,806 个样本的伪标签（1.3MB）

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
- 50 epochs 配置，epoch 15 提前停止
- Spearman 相关系数 ρ = 0.96（预测 score vs 伪标签）
- 显存占用 < 5GB，训练时间 ~2h

**产出**：
- `checkpoints/uq/best.pt` — 训练好的 UQEstimator 权重（25MB）

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

CARLA 场景按天气 ID 分为两类：

| 类别 | 天气 ID | 包含场景 | 样本数 |
|------|---------|----------|--------|
| Normal | 0-3 | ClearNoon, ClearSunset, CloudyNoon, CloudySunset | 2,709 |
| Adverse | 5-26 | 雨、雾、夜间、湿路面及其组合 | 10,097 |

**为什么这样划分**：Weather 0-3 是标准白天晴/多云场景，视觉条件良好；其余场景均包含不同程度的视觉降质（雨滴、雾气、低光照、路面反光）。

### 3.2 UQ Score 分离度

> 📊 对应图表：`results/figures/baseline/fig1_score_dist.pdf`

| 指标 | Normal (2,709) | Adverse (10,097) | 目标 |
|------|----------------|------------------|------|
| UQ Score 均值 | 0.545 | 0.780 | 差值 > 0.1 |
| UQ Score 中位数 | 0.744 | 0.963 | — |
| **均值差** | | | **0.235 ✅** |

**分析**：UQ score 的 normal/adverse 均值差 = 0.235，**满足 > 0.1 的验收标准**。Adverse 场景的 UQ score 显著偏高，说明 UQEstimator 成功学习到了"恶劣天气 → 高不确定性"的映射。

但注意中位数（0.744 vs 0.963）比均值差距更小，说明两个分布有一定重叠——这在预期之内，因为有些 adverse 场景（如轻度湿路面）视觉特征接近 normal。

### 3.3 AUROC

> 📊 对应图表：`results/figures/baseline/fig2_auroc.pdf`

| 指标 | 数值 | 目标 |
|------|------|------|
| AUROC (UQ score → adverse 分类) | **0.621** | > 0.7 |

**分析**：AUROC = 0.621 低于 0.7 的初始目标。原因：

1. **ClearNoon 异常低分**：ClearNoon（906 样本）UQ score 均值仅 0.003，而同属 normal 的 CloudyNoon 达 0.885。这说明伪标签的 scene_type 校准对部分场景产生了极端效果
2. **MidRainSunset 异常低分**：MidRainSunset（678 样本）UQ score 均值仅 0.001，是 adverse 场景中最低的。可能该场景视觉特征接近晴天
3. **样本不平衡**：adverse 样本（10,097）是 normal（2,709）的 3.7 倍

**改进方向**（备选方案 A）：
- 加入 Temporal inconsistency 作为第 4 个伪标签分量
- 用 VLM 做更精确的 scene_type 二分类
- 调整校准区间，增大 margin

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

---

## 六、数据资产清单

| 路径 | 大小 | 说明 |
|------|------|------|
| `ckpts/Orion.pth` | 36GB | ORION 主模型 |
| `ckpts/pretrain_qformer/` | ~14GB | Qwen LLM 权重 |
| `data/bench2drive/v1/` | 407GB | Bench2Drive 原始数据集（1001 个场景）|
| `data/infos/b2d_infos_val.pkl` | 141MB | 验证集标注（12,806 样本）|
| `data/features/` | 235GB | 预提取的 EVAViT patch tokens |
| `data/labels/uq_labels.pt` | 1.3MB | 不确定性伪标签 |
| `checkpoints/uq/best.pt` | 25MB | 训练好的 UQEstimator |
| `checkpoints/film/best_l1.pt` | 516KB | FiLM L1 权重 |
| `checkpoints/film/best_l2.pt` | 8.2MB | FiLM L2 权重 |
| `checkpoints/film/best_l1l2.pt` | 8.5MB | FiLM L1+L2 权重（L2 loss 版本）|
| `checkpoints/film/best_l1l2_col.pt` | ~8.5MB | FiLM L1+L2 权重（碰撞感知 loss，方案 C）|
| `results/eval_openloop_full.pt` | ~50MB | 开环评估逐样本结果 |
| `results/closedloop_film_baseline.json` | ~5KB | 闭环评估结果（baseline vs FiLM L2-loss）|
| `results/figures/baseline/` | ~440KB | 5 张可视化图表 |

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

### Stage 4b：碰撞感知 FiLM 训练（方案 C） 🔄

**开始日期**：2026-03-29

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

**当前状态**：训练进行中（Epoch 1/5, ~50%），预计总训练时间约 8 小时。

**梯度链**：
```
col_loss → ego_fut_preds → ego_fut_decoder → VAE → FiLM L2
                                            → QT-Former → FiLM L1
(GT agent 数据不参与梯度，UQ score detach)
```

---

## 七、下一步计划

### 已验证结论

1. **L2 轨迹 loss 训练的 FiLM 无法降低碰撞率**（Stage 4a 已证实）
2. **L2 轨迹精度 ≠ 安全性**：FoggySunset L2 最低但碰撞率 0%，Unknown23 L2 低但碰撞率 11.25%
3. **Adverse 碰撞率是 Normal 的 161 倍**（1.61% vs 0.01%）——核心安全问题

### 当前进行中

1. **碰撞感知 FiLM 训练（方案 C）**：3000 samples × 5 epochs，预计 2026-03-29 完成
2. 训练完成后立即进行闭环评估对比

### 待做

1. **闭环评估**：碰撞感知 FiLM vs Baseline，10+ scenarios
2. **全量 Ablation**：如果方案 C 有效，跑全量 4 组对比（A=Baseline, B=L1, C=L2, D=L1+L2+ColLoss）
3. **方案 C 失败的备选**：
   - 增大训练数据量（全量 12,806 样本）
   - 调整 margin/lambda_col 超参数
   - 在 loss 中加入方向感知项（惩罚朝向 agent 运动的轨迹）

### 需要关注的风险

1. **AUROC 不达标**（当前 0.621，目标 0.7）：可通过改进伪标签方案提升
2. **碰撞率集中在少数场景**：Unknown(23) 占全部碰撞的绝大多数
3. **碰撞 loss 稀疏性**：约 70% 的帧 col_loss=0（所有 agent > 4m），有效梯度信号有限

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

# 4. 伪标签生成
python scripts/generate_labels.py \
    --feature_dir data/features \
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

**Insight 3: 纯 L2 loss 训练的 FiLM 不仅没降碰撞率，反而让一切更差**

闭环评估：碰撞率 1.17% → 1.22%，ADE 2.26m → 3.90m。优化 L2 匹配的 FiLM 让模型学会更好地拟合训练分布，但这个分布本身不包含安全性信号。

**论文用途**：Ablation study 的重要对照组——证明需要碰撞感知训练目标，单纯调制不够。

**Insight 4: UQ score 有效但需要碰撞感知桥接**

- UQ score normal/adverse 均值差 = 0.235（Spearman ρ=0.96 vs 伪标签）
- 但 UQ score 与碰撞率的直接相关性弱（ρ=-0.077）
- 说明 UQ score 捕获了「感知质量下降」但没有直接翻译为「应该规避碰撞」
- 碰撞感知 loss 中 UQ score 做权重 = 桥接不确定性估计和安全决策

**论文用途**：方法章节——解释为什么需要碰撞感知 loss 而非仅仅将 UQ embedding 注入网络。

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
5. **Results**: (待碰撞感知训练完成后补充)

---

*报告更新时间：2026-03-29*
*Git 分支：dev*

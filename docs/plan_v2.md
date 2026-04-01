# UQ-ORION 研究计划 v2

> 版本：2026-04-01
> 状态：待 Gemini 审核
> 目标：申请计算资源前的完整规划

---

## 一、研究目标

在极端感知条件（雨、雾、夜晚）下，端到端自动驾驶模型（ORION）因图像质量退化而产生不安全的规划行为。本项目目标：

1. **量化感知不确定性**：设计轻量 UQEstimator，从视觉 token 中提取每帧的不确定性分数与嵌入
2. **空间化不确定性**：将不确定性从全局标量升维为 BEV 空间热力图，使规划器感知"哪个区域的感知不可靠"
3. **安全规划**：用不确定性约束轨迹模式选择，主动规避感知可靠性低的区域

**核心创新点**：首次在高速端到端自动驾驶中，将 aleatoric 感知不确定性（传感器退化型）提升为 BEV 代价图，直接约束多模态轨迹决策，参数增量 < 0.5M，不侵入冻结的 backbone。

---

## 二、当前状态（已完成工作）

### 实验结果基线

| 指标 | Baseline | FiLM-v3 | Δ |
|------|----------|----------|---|
| ADE@3s (normal) | 2.78 m | **6.00 m** | **+116% ❌** |
| ADE@3s (adverse) | 2.37 m | 3.99 m | +68% |
| Col@3s (normal) | 0.05% | 0.11% | +120% ❌ |
| Col@3s (adverse) | 0.83% | **0.64%** | **−23% ✓** |
| AUROC (UQ→weather) | — | 0.954 | ✓ |

**结论**：FiLM 在恶劣场景下减少碰撞，但严重损害正常场景性能。

### 已识别的根本问题

1. **LayerNorm 范数崩塌**：`embed_head` 末尾 LayerNorm 使所有样本 embedding 模恒为 √256，正常场景（score≈0）也受到等幅 FiLM 调制 → Normal ADE +116%
2. **空间信息丢失**：UQEstimator 把 6×1600 个 patch 的注意力权重 mean pool 为 256 维全局向量，失去"哪个 BEV 区域不确定"的空间信息
3. **Score 未参与调制强度**：FiLM 强度不随 UQ score 变化

### 已完成的代码与数据

| 文件 | 状态 | 说明 |
|------|------|------|
| `uq_estimator/model.py` | ✅ 完成 | UQEstimator (2.24M params) |
| `uq_estimator/losses.py` | ✅ 完成 | MSE + calibration + ranking loss |
| `checkpoints/uq/best.pt` | ✅ 完成 | v3 权重，AUROC=0.954 |
| `checkpoints/film/best_l1l2_col_v3.pt` | ✅ 完成 | 当前最佳 FiLM 权重 |
| `results/closedloop_baseline_50.json` | ✅ 完成 | 50场景正确 baseline |
| `results/closedloop_replay_v3.json` | ✅ 完成 | 50场景 FiLM v3 结果 |
| `results/gifs/bev_only/*.gif` | ✅ 完成 | 18场景 BEV 轨迹对比 GIF |

---

## 三、新架构设计

### 3.1 总览

```
ORION（完全冻结）:
  6×Camera → EVAViT → patch_tokens [B, 6, 1600, 1024]
                    → QT-Former → BEV queries [B, 900, 256]
                               → attn_weights [B, 900, 9600]  ← 读取
                    → LLM → ego_feature [B, 4096]
                    → VAE + plan_anchor → 20个轨迹模式 [B, 20, 6, 2]
                    → cls_logits [B, 20]

UQ 扩展模块（新训练）:
  ┌─ [已有] UQEstimator:
  │    patch_tokens → uncertainty_score [B,1] + uncertainty_emb [B,256]
  │
  ├─ [修复] Score-Gated FiLM (L1+L2):
  │    gamma = 1 + score*(gamma_raw - 1)   ← 正常场景 score≈0 → 恒等
  │    beta  = score * beta_raw
  │
  └─ [新增] BEV Uncertainty Cost:
       ① patch_quality[j] = f(Laplacian_var, gradient_mag, contrast)  # 无参数，可微
       ② bev_uncertainty[i] = Σ_j attn_weight[i,j] * patch_quality[j]  # 线性传播
       ③ uncertainty_cost[m] = mean_t bilinear_sample(bev_unc, waypoint[m,t])
       ④ adjusted_logit = cls_logit - λ * score * uncertainty_cost   # λ 唯一可学参数
```

### 3.2 关键设计决策

**为什么不学习 per-patch uncertainty head？**
- 无 per-patch GT，伪标签质量不可靠
- Image quality metrics（Laplacian variance、梯度幅值）与恶劣天气直接对应，物理含义明确，无需学习
- 保持方案可解释性，每步都是可验证的确定性计算

**为什么通过 attention 传播而不做 camera-to-BEV lifting？**
- 不需要深度估计，无深度歧义问题
- QT-Former attention 已经编码了"哪个 BEV 位置依赖哪些 patch"的关系
- 完全在 ORION 的特征空间内操作，无需改动 backbone

**λ 为什么只有 1 个参数？**
- 避免过拟合小数据集
- 物理意义清晰：λ 控制不确定性代价的全局权重
- 可扩展为 λ(score)：按 UQ score 大小动态调整权重

---

## 四、分阶段计划

---

### Phase 0：信号验证
**目标**：验证 image quality metrics 能有效区分恶劣/正常天气 patch，确认信号可用性
**前置条件**：现有 18 个场景 GIF 对应的原始图像
**计算资源**：本地 CPU，无 GPU 需求

#### 要做的事

1. 对现有 18 个场景的前置摄像头图像，按帧计算 per-patch Laplacian variance 和梯度幅值
2. 按天气类型（normal / adverse）分组，绘制统计对比图
3. 验证假设：恶劣天气帧的 patch quality 均值显著低于正常天气帧

#### 需要准备的文件

| 文件 | 来源 | 说明 |
|------|------|------|
| 原始 RGB 图像 | B2D 数据集 `data/bench2drive/*/camera/rgb_front/` | 需要数据集可访问 |
| `results/gifs/trajectory_data.pt` | 已有 | 场景列表和 weather_id |

#### 产出

- `results/signal_validation/patch_quality_stats.json`：各场景各帧的质量统计
- `results/signal_validation/fig_quality_vs_weather.png`：质量分布对比图（正常 vs 恶劣）
- **决策点**：如果 p-value < 0.01 且 effect size > 0.5，继续 Phase 2；否则考虑替换 quality 指标

#### 实现脚本

`scripts/validate_patch_quality.py`（待创建，纯 numpy + scipy + PIL）

---

### Phase 1：Score-Gated FiLM 修复与重训
**目标**：修复 LayerNorm 范数崩塌问题，使 FiLM 在正常场景下不影响原始规划
**前置条件**：需要 ORION 完整推理（LLM INT8 约 12 GB）
**计算资源**：本地 16 GB（INT8 量化），或服务器

#### 要做的事

1. 修改 `mmcv/models/utils/petr_transformers.py`（~3 行）：实现 Score-Gated FiLM L1
2. 修改 `mmcv/models/detectors/orion.py`（~3 行）：实现 Score-Gated FiLM L2
3. 重训 FiLM（使用现有 `scripts/train_film.py`，collision-aware loss）
4. 开环验证：Normal ADE 应恢复到接近 baseline，Adverse Col 保持或改善

#### 代码改动（明确）

```python
# petr_transformers.py，FiLM forward 中（约3行替换）
# Before:
gamma = self.film_gamma(uncertainty_emb).unsqueeze(1)
beta  = self.film_beta(uncertainty_emb).unsqueeze(1)
query = gamma * query + beta

# After (Score-Gated):
gamma_raw = self.film_gamma(uncertainty_emb).unsqueeze(1)   # [B,1,D]
beta_raw  = self.film_beta(uncertainty_emb).unsqueeze(1)    # [B,1,D]
score     = uq_score.unsqueeze(-1)                           # [B,1,1]
gamma     = 1.0 + score * (gamma_raw - 1.0)
beta      = score * beta_raw
query     = gamma * query + beta
```

#### 需要准备的文件

| 文件 | 来源 |
|------|------|
| ORION 主 checkpoint | `/workspace/uq-orion/ckpts/Orion.pth` |
| LLM pretrain checkpoint | `/workspace/uq-orion/ckpts/pretrain_qformer/` |
| B2D 训练集 | `data/bench2drive/` + `data/infos/b2d_infos_train.pkl` |
| UQEstimator checkpoint | `checkpoints/uq/best.pt` |

#### 计算资源估算

| 任务 | 显存 | 时间估算 |
|------|------|---------|
| FiLM 训练（INT8 LLM, bs=1） | ~13 GB | 6–12 h / epoch |
| 开环验证（500 帧）| ~12 GB | ~2 h |

#### 产出

- `checkpoints/film/best_score_gated_v1.pt`：Score-Gated FiLM 权重
- 开环对比表：baseline / FiLM-v3 / Score-Gated-FiLM 三组结果

---

### Phase 2：BEV 不确定性热力图（无 LLM，16GB 本地可运行）
**目标**：实现 patch quality → BEV uncertainty map 的计算链路，验证热力图质量
**前置条件**：ORION checkpoint（仅需 EVAViT + QT-Former 部分），B2D 数据集
**计算资源**：本地 16 GB（只加载 EVAViT + QT-Former，约 0.73 GB）

#### 要做的事

1. 编写 `scripts/extract_bev_uncertainty.py`：
   - 加载 ORION（仅 EVAViT + QT-Former，不实例化 LLM）
   - 对验证集每帧计算：
     - `patch_quality [B, 6, 1600]`：per-patch Laplacian var + gradient mag 归一化
     - `attn_weights [B, 900, 9600]`：从 QT-Former cross-attention 读取（`register_forward_hook`）
     - `bev_uncertainty [B, 900]`：attention-weighted patch quality
     - `cls_logits [B, 20]`：mode 得分（**注意：此步仍需 LLM，需另行缓存**）
   - 保存缓存到 `data/bev_cache/`

2. 可视化验证：
   - 热力图可视化：reshape `[B, 900]` → `[B, 30, 30]`，叠加到 BEV 图上
   - 对比正常帧 vs 恶劣帧的热力图分布

3. 验证指标：
   - Spearman 相关：全局 UQ score vs BEV uncertainty mean（应该高度相关）
   - 空间差异性：恶劣帧中不同 BEV 区域的 uncertainty 方差应显著大于正常帧

#### 需要准备的文件

| 文件 | 来源 |
|------|------|
| ORION checkpoint | `ckpts/Orion.pth` |
| B2D 验证集 | `data/bench2drive/` |
| `cls_logits` 缓存 | 需要从 Phase 1 的完整推理中提取并保存 |

#### 关键技术细节：如何不加载 LLM

```python
# 修改 build_orion_model，传入 lm_head=None
cfg.model.lm_head = None
cfg.model.tokenizer = None
# ORION 在 lm_head=None 时会跳过 VLM 前向
# 但 cls_logits 依赖 LLM 输出的 ego_feature，需要分离缓存
```

`cls_logits` 需要额外一步：在有 LLM 的机器（或 INT8）上一次性提取并保存。

#### 产出

- `data/bev_cache/{scene_id}.pt`：每帧的 `bev_uncertainty [900]` + `patch_quality [6,1600]` + `frame_meta`
- `results/figures/bev_uncertainty_heatmaps/`：可视化热力图（正常 vs 恶劣天气对比，共 ~20 图）
- `results/signal_validation/bev_uncertainty_stats.json`：统计验证结果
- **论文图**：BEV uncertainty heat map 可视化（强可解释性）

---

### Phase 3：λ 训练与轨迹代价集成
**目标**：训练 λ 参数，验证 BEV uncertainty cost 对轨迹 mode 选择的影响
**前置条件**：Phase 2 的 BEV 缓存 + `cls_logits` 缓存 + GT 轨迹
**计算资源**：CPU 或极小 GPU（< 0.5 GB），无 LLM 需求

#### 要做的事

1. 编写 `scripts/train_lambda.py`：
   - 加载缓存数据（不需要 ORION）
   - 计算 `uncertainty_cost [B, 20]`：对每个 plan_anchor 轨迹做 BEV uncertainty 双线性采样 + 平均
   - 优化目标：`minimize E[Col(argmax(cls_logit - λ*score*uncertainty_cost), GT)]`
   - 使用验证集 held-out 分割，防止过拟合

2. 消融实验（小规模，CPU 可跑）：
   - λ=0（纯 FiLM，无 BEV cost）
   - λ=最优（Score-Gated FiLM + BEV cost）
   - 固定 λ=1（不学习，看 BEV cost 方向是否正确）

3. 可视化：展示高不确定性场景下 mode 选择的变化

#### 需要准备的文件

| 文件 | 来源 |
|------|------|
| `data/bev_cache/*.pt` | Phase 2 产出 |
| `cls_logits` 缓存 | Phase 1 完整推理时同步提取 |
| `plan_anchor.pkl` | ORION 训练数据 |
| GT 轨迹（open-loop） | B2D 验证集标注 |

#### 产出

- `checkpoints/lambda/best_lambda.pt`：训练好的 λ（1个标量）
- `results/lambda_ablation.json`：λ 消融结果
- **验证标准**：`adjusted_logit` 的 mode 选择在 adverse 场景的碰撞率应低于 `cls_logit`

---

### Phase 4：完整集成与闭环评估
**目标**：将 Score-Gated FiLM + BEV Uncertainty Cost 完整集成，跑大规模闭环评估
**前置条件**：Phase 1–3 全部完成
**计算资源**：**需要服务器**（A100/V100，CARLA 仿真环境，至少 200 场景）

#### 要做的事

1. 完整集成代码，更新 `adzoo/orion/test.py` 加载新模块
2. 开环评估（500+ 帧）：baseline / Score-Gated FiLM / +BEV Cost / Full Model
3. 闭环评估（100+ 场景）：重点对比 normal ADE 和 adverse collision rate
4. 消融实验矩阵（4组 × 2指标）

#### 消融实验矩阵

| 组别 | Score-Gated FiLM | BEV Cost (λ) | 预期效果 |
|------|:----------------:|:------------:|---------|
| A: Baseline | ✗ | ✗ | 参考基线 |
| B: FiLM-v3 (现有) | ✗（buggy）| ✗ | Normal ADE 退化 |
| C: Score-Gated FiLM | ✓ | ✗ | Normal ADE 恢复，Adverse Col 改善 |
| D: BEV Cost only | ✗ | ✓ | Adverse Col 改善，空间感知 |
| E: Full (C+D) | ✓ | ✓ | 双重改善 |

#### 计算资源估算

| 任务 | 硬件 | 时间 |
|------|------|------|
| 开环评估 500帧 × 5组 | 1×A100 40GB | ~10 h |
| 闭环 100场景 × 3组 | 1×A100 + CARLA | ~48 h |
| FiLM 重训（Score-Gated）| 1×A100, bs=4 | ~24 h |

#### 产出

- `results/eval_openloop_full_v4.json`：5组开环结果
- `results/closedloop_full_v4.json`：100场景闭环结果
- 论文主表格（Table 1）：开环+闭环 all/normal/adverse 三组指标

---

### Phase 5：可视化与论文写作
**目标**：生成论文所需所有图表，完成初稿
**前置条件**：Phase 4 完成
**计算资源**：本地即可

#### 核心图表

| 图 | 内容 | 脚本 |
|----|------|------|
| Fig 1 | BEV uncertainty heatmap（正常 vs 恶劣天气对比） | Phase 2 产出 |
| Fig 2 | 系统架构图 | 手绘/tikz |
| Fig 3 | UQ score 分布 + AUROC 曲线 | `scripts/visualize_eval.py` |
| Fig 4 | 消融实验柱状图（5组 × normal/adverse） | Phase 4 产出 |
| Fig 5 | 轨迹对比 GIF（GT / Baseline / Ours，恶劣场景选例） | `scripts/generate_trajectory_gifs.py` |
| Fig 6 | BEV uncertainty map + 轨迹 mode 选择可视化 | 新脚本 |
| Fig 7 | 校准曲线（UQ score vs 实际 L2/碰撞率）| `scripts/visualize_eval.py` |

---

## 五、计算资源需求汇总

### 本地 16GB 可完成（无需申请）

| Phase | 任务 | 显存 | 备注 |
|-------|------|------|------|
| 0 | 信号验证 | CPU | 纯 numpy |
| 2 | BEV uncertainty map 提取 | ~1 GB | 只加载 EVAViT+QT-Former |
| 3 | λ 训练 | CPU | 只读缓存 |
| 1（可试） | Score-Gated FiLM 代码修改 | 0 | 只改代码 |
| 1（可试） | FiLM 重训 | ~13 GB | INT8 量化，紧张但可能可行 |

**本地 INT8 量化方法**（需要在 `mmcv/utils/misc.py` 中加一行）：
```python
model = LlavaLlamaForCausalLM.from_pretrained(
    base_model, torch_dtype=torch.float16,
    load_in_8bit=True,   # 新增，需要 bitsandbytes
    device_map='auto',
)
```

### 需要申请服务器资源

| Phase | 任务 | 最低硬件 | 时间 | 优先级 |
|-------|------|---------|------|--------|
| 1 | Score-Gated FiLM 重训 | 1×A100 40GB | 24 h | 高 |
| 4 | 开环评估（5组） | 1×A100 40GB | 10 h | 高 |
| 4 | 闭环评估（100场景 × 3组） | 1×A100 + CARLA | 48 h | 高 |
| 4 | λ 对 cls_logits 缓存提取 | 1×A100 40GB | 4 h | 中 |

**申请资源时的 justification**：
- FP16 LLM 需要 ~18 GB，超出消费级 GPU 上限，必须用 A100/V100
- 闭环评估需要实时 CARLA 仿真，无法在推理机上跑

---

## 六、风险分析

| 风险 | 可能性 | 影响 | 缓解方案 |
|------|--------|------|---------|
| QT-Former attention hook 读取失败 | 中 | 高 | 改 QT-Former forward 接口，显式返回 attn_weights |
| BEV uncertainty map 与 UQ score 相关性弱 | 低 | 高 | Phase 0 验证，失败则换 patch quality 指标 |
| INT8 LLM 量化导致 FiLM 训练不稳定 | 中 | 中 | 用 bfloat16 代替，或 CPU offload |
| λ 过拟合小数据集 | 低 | 低 | k-fold 交叉验证，或固定 λ=1 |
| Normal ADE 仍不达 baseline | 低 | 高 | Score-Gated 设计理论保证 score=0 时恒等 |

---

## 七、与相关工作的差异化定位

### 对比 VLN 论文（LPv59noPAy，ICLR 2026 under review）

**"Uncertainty-Aware Gaussian Map for Vision-Language Navigation"**

| 维度 | 彼（VLN 论文） | 我们 |
|------|--------------|------|
| 任务 | 室内离散导航 | 高速连续轨迹规划 |
| 不确定性来源 | 3D 重建质量（epistemic）| 传感器退化/天气（aleatoric）|
| 空间表示 | 在线增量 3D Gaussian 地图 | 单帧多摄像头 BEV |
| 不确定性计算 | 变分推断 + Fisher Information | Image quality metrics + attention 传播 |
| 注入方式 | 3D Value Map → action 约束 | BEV uncertainty cost → mode selection |
| Backbone | 端到端训练 | Frozen ORION backbone |
| 参数增量 | 大（新 3D 模块）| < 0.5M |

**reviewer 差异化话术**：
> *While [LPv59noPAy] similarly leverages spatial uncertainty for navigation, their work targets epistemic uncertainty arising from incomplete 3D scene reconstruction in indoor VLN. In contrast, our method addresses aleatoric sensor uncertainty in high-speed autonomous driving, where uncertainty stems from weather-induced image degradation rather than scene incompleteness. We do not build an explicit 3D map; instead, we propagate patch-level image quality through the frozen QT-Former's cross-attention to construct a BEV uncertainty field, achieving spatial uncertainty awareness with near-zero additional parameters, compatible with single-frame frozen E2E backbones.*

---

## 八、本地可立即开始的工作

按优先级排列，无需服务器即可推进：

1. **今天**：实现 `scripts/validate_patch_quality.py`，验证 image quality signal
2. **本周**：实现 `scripts/extract_bev_uncertainty.py`（仅 EVAViT+QT-Former 部分），验证 BEV 热力图
3. **本周**：修改 3 行代码实现 Score-Gated FiLM，尝试 INT8 量化本地训练
4. **下周**：实现 `scripts/train_lambda.py`，用离线缓存数据训练 λ

---

## 附录：文件依赖关系图

```
B2D 数据集
    ├── Phase 0 → patch_quality_stats.json（决策点）
    ├── Phase 1 → checkpoints/film/best_score_gated_v1.pt
    │              + cls_logits 缓存（服务器上提取）
    ├── Phase 2 → data/bev_cache/*.pt
    │              + BEV heatmap 可视化图
    └── Phase 3 → checkpoints/lambda/best_lambda.pt（依赖 Phase 1+2）
                   ↓
              Phase 4（完整评估，服务器）
                   ↓
              Phase 5（论文图表）
```

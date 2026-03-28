# UQ-ORION 项目计划

> 最后更新：2026-03-27（v2，incorporated Gemini review）
> 硬件约束：1x NVIDIA A100 80GB
> 目标：投稿论文（CVPR / CoRL / ICRA 级别）

---

## 一、核心命题

> 在 ORION 基础上，以**轻量、即插即用**的方式引入不确定性感知，
> **不损失普通场景下的性能**，同时在**恶劣天气 / OOD 场景下开环+闭环指标显著提升**。

### 论文标题候选
- *Dual-Layer Uncertainty Injection for Robust Vision-Language Planners*
- *Plug-and-Play Uncertainty Modulation for End-to-End Autonomous Driving*

### 三项核心贡献（审稿人视角）
1. **零主干微调**：新增 < 5M 参数，完全冻结 VLM 主干，单卡可复现
2. **双层调制机制**：从场景理解（QT-Former）和轨迹规划（VAE）两个层面联合建模不确定性
3. **泛化性**：该范式不绑定 ORION，可扩展到任意 VLM-based Planner

---

## 二、技术方案

### 整体架构

```
Vision Encoder (EVAViT, 冻结)
    ↓ patch_tokens [B, N_views, N_patches, 1024]
    ├──────────────────────────────────────────────────────┐
    │                                                      ↓
    │                                              UQEstimator (2.24M, 可训练)
    │                                                      ↓
    │                                         uncertainty_embedding [B, 256]
    │                                         uncertainty_score     [B, 1]
    │                                                      │
QT-Former (冻结)  ←── FiLM L1 (场景理解层) ───────────────┤
    ↓ vlm_memory                                           │
   LLM / Qwen (冻结)                                       │
    ↓ ego_feature (planning token)                         │
   VAE / Diffusion ←── FiLM L2 (轨迹规划层, Phase 2) ─────┘
    ↓
   trajectory
```

### 双层注入的直觉解释（论文用语）

不确定性信号本质是**全局环境上下文**（感知质量的摘要）。
- **L1 (QT-Former)**：高不确定性时，FiLM 通过 scale/shift 抑制激进特征激活、放大保守特征权重，让 LLM 接收到"视野受限"的场景表示
- **L2 (VAE)**：在轨迹分布采样阶段注入不确定性，将分布向保守模式偏移（相当于减小轨迹方差、降速）

### 设计原则
- ORION 主干全程冻结
- 新增参数 < 5M（UQEstimator 2.24M + FiLM L2 ~1M）
- `use_uncertainty=True/False` 一行切换

---

## 三、伪标签方案（需在论文中自圆其说）

### 当前方案（3 分量加权）
```
score = 0.3 × gradient_score   # 图像梯度低 → 模糊 → 高不确定
       + 0.3 × entropy_score    # token 激活熵高 → 特征混乱 → 高不确定
       + 0.4 × consistency_score # 跨视角一致性低 → 感知矛盾 → 高不确定

+ scene_type 校准：normal → [0, 0.45]，adverse → [0.55, 1.0]
```

### 论文中如何辩护（必须做）
1. **可视化相关性**：随机抽 50-100 个样本，并排展示原始图像 + UQ score，展示大雨/大雾场景确实得到高分
2. **分布分析**：normal vs adverse 两组的 score 分布直方图，必须有明显分离
3. **与 scene_type 的一致性**：计算伪标签与 scene_type 标注的 AUROC（应 > 0.7）

### 可选增强（若第一版效果不佳再做）
- **Temporal inconsistency**：相邻帧特征余弦距离作为第 4 个分量（需修改提取 pipeline，约 1 天工作量）
- **VLM 场景分类**：用 GPT-4V / LLaVA 对 scene_type 做二分类，替换规则校准（提高标签精度）
- 第一版**不做** MC-Dropout（需要 5-10x 推理时间，代价过高）

---

## 四、训练阶段

### Stage 0：特征提取（进行中）

```bash
python scripts/extract_orion_features.py \
    --checkpoint /workspace/uq-orion/ckpts/Orion.pth \
    --output_dir /workspace/uq-orion/data/features \
    --ann_file /workspace/uq-orion/data/infos/b2d_infos_val.pkl \
    --batch_size 8 --num_workers 1
```

- ✅ 完成（2026-03-27），12806 样本，~235GB
- 显存：~40GB

---

### Stage 1：UQEstimator 独立训练 ✅

```bash
# 1a. 生成伪标签
python scripts/generate_labels.py \
    --feature_dir data/features \
    --output_file data/labels/uq_labels.pt \
    --n_workers 4

# 1b. 训练
python scripts/train_uq.py \
    --config configs/uq_train.yaml \
    2>&1 | tee /workspace/train_uq_log.txt

# 1c. 验证 + 可视化
python scripts/validate_uq.py \
    --checkpoint checkpoints/uq/best.pt \
    --feature_dir data/features \
    --label_file data/labels/uq_labels.pt \
    --visualize  # 生成分布图 + 样本可视化
```

**验收标准**（分离度优先，非绝对数值）：
- normal 均值 < adverse 均值，且差值 > 0.1
- Spearman 相关系数 > 0.4
- 伪标签与 scene_type 的 AUROC > 0.7

**显存**：< 5GB
**估计时间**：2-4 小时
**实际结果**：Epoch 15/50 停止，Spearman ρ=0.96，checkpoint: `checkpoints/uq/best.pt`

---

### Stage 2：Phase 1 开环评估 🔄

**目标**：验证 UQ 信号有效性 + FiLM L1 注入效果

**重要发现（2026-03-28）**：
- FiLM L1 权重（film_gamma, film_beta）在 PETRTemporalTransformer 中，**未包含在 UQ checkpoint 中**
- 需要额外的 FiLM fine-tune 步骤（~131K 可训参数，用 trajectory loss）
- 分两步：(2a) 无 FiLM 的 UQ score 分析；(2b) FiLM 训练 + L1 eval

**评估指标策略**（重要）：
- **不要**只看全量 L2 error（不确定性注入可能让 normal 场景微降）
- **重点看**：adverse 场景切片的 L2 error、碰撞率
- **关键数字**：adverse 场景提升 > normal 场景下降

```bash
# Stage 2a: UQ Score Analysis（无 FiLM，收集 baseline + UQ score）
python scripts/eval_openloop.py \
    adzoo/orion/configs/orion_stage3_infer.py ckpts/Orion.pth \
    --ann-file data/infos/b2d_infos_val.pkl \
    --out results/eval_openloop_full.pt

# Stage 2b: FiLM fine-tune（冻结全部，只训 gamma/beta）
python scripts/train_film.py \
    --config adzoo/orion/configs/orion_stage3_infer.py \
    --checkpoint ckpts/Orion.pth \
    --epochs 3 --lr 1e-3 \
    --out checkpoints/film/best.pt

# Stage 2b: L1 eval（带训练后的 FiLM）
python scripts/eval_openloop.py \
    adzoo/orion/configs/orion_stage3_infer.py ckpts/Orion.pth \
    --ann-file data/infos/b2d_infos_val.pkl \
    --film-checkpoint checkpoints/film/best.pt \
    --out results/eval_openloop_film_l1.pt
```

**显存**：推理 ~35-40GB，FiLM 训练 ~45GB（加激活值缓存）
**时间**：eval ~6h（12806 samples），FiLM 训练 ~2h

---

### Stage 3：FiLM L2 实现 + 联合训练

**代码改动**：
- `mmcv/models/detectors/orion.py`：ego_feature 进 VAE 前加 FiLM（约 50 行）
- `scripts/train_uq_finetune.py`：联合训练脚本

**Loss 策略**（防冲突）：
```python
# 总 loss = trajectory_loss（主导）+ lambda_uq × uq_loss（正则化）
# lambda_uq 初始 0.1，若 trajectory_loss 不收敛则降至 0.01
total_loss = trajectory_loss + lambda_uq * uq_loss
```
- trajectory loss 量纲主导，uq loss 作为正则化项
- 若训练不稳定：先只训练 FiLM L2（冻结 UQEstimator），收敛后再联合

**同步完成**：Attention map 可视化代码
- 比较注入 UQ 前后 QT-Former 的 attention 分布
- 高不确定性时，attention 是否更集中于自车周围安全区域
- 这张图放论文效果极佳

**显存**：~40GB（ORION 推理无梯度 + FiLM 层梯度）

---

### Stage 4：完整 Ablation + 闭环评估（论文核心数据）

**四组对比**：

| 组别 | 配置 | 用途 |
|------|------|------|
| **A** | 原版 ORION | Baseline |
| **B** | ORION + UQ (L1 only) | Ablation |
| **C** | ORION + UQ (L2 only) | Ablation |
| **D** | ORION + UQ (L1 + L2) | Full model (Ours) |

**开环评估**（按场景分层）：
- 指标：L2 error、碰撞率、规划成功率
- 分层：全量 / normal / adverse（雨雪雾低能见度）
- 期望：D 在 adverse 上显著优于 A，在 normal 上接近 A

**Bench2Drive 完整评估（离线，无需 CARLA）**：
- 数据：1001 个预录制场景已在本机（407GB，`data/bench2drive/v1/`）✅
- 执行方式：`B2DOrionDataset.evaluate()` 读预录数据，无需启动 CARLA 服务器
- 感知指标：mAP、NDS（已验证可跑 ✅）
- 规划指标：轨迹 L2、碰撞率（对比预测轨迹 vs GT 障碍物位置）
- 天气效果：已烘焙在图像渲染里（雨、雾、雪场景均有）
- ⚠️ 当前推理配置的 planning head 未启用（plan_results 为空），需在 Stage 2 前修复

---

## 五、如果效果不佳的备选方案

### 情形 A：UQ 分数分离度不够（normal/adverse 分不开）
1. 加入 Temporal inconsistency 作为第 4 个伪标签分量
2. 引入 VLM 对 scene_type 做精确二分类（替换规则校准）
3. 调整 scene_type 校准区间（扩大 normal/adverse 的 margin）

### 情形 B：开环 adverse 提升不明显
1. 调整 FiLM gamma/beta 的初始化 scale
2. 增大 ranking loss 权重，强化分数两端的拉开
3. 检查 adverse 样本量是否足够（若太少，分层评估噪声大）

### 情形 C：Phase 2 联合训练不收敛
1. 先只训练 FiLM L2（固定 UQEstimator 权重）
2. 降低 lambda_uq（0.1 → 0.01）
3. 使用更小学习率（3e-5 而非 3e-4）

---

## 六、论文故事线结构

```
1. Introduction
   痛点：VLM E2E planner 在 OOD（恶劣天气）下过度自信，做出激进决策
   代价：微调几十 GB 大模型不现实
   我们的解：轻量即插即用，< 5M 参数，零主干微调

2. Method
   2.1 UQ Estimation：从视觉特征提取不确定性信号（科学依据 + 可视化佐证）
   2.2 Dual-level FiLM Injection：
       L1 调制场景理解（QT-Former）→ 直觉：让 LLM 感知"视野受限"
       L2 调制轨迹规划（VAE）→ 直觉：将分布向保守模式偏移

3. Experiments
   3.1 UQ Score Analysis：分布可视化 + AUROC + Attention Map 对比
   3.2 Bench2Drive 开环（分 normal/adverse 场景切片）
   3.3 Bench2Drive 闭环（碰撞率、路线完成率 ← 杀手锏）
   3.4 Ablation：L1-only / L2-only / L1+L2

4. Discussion
   泛化性：可扩展到任意 VLM-based Planner
   局限性：伪标签依赖启发式规则

5. Future Work
   - LLM 层 native uncertainty token（VLM 直接输出不确定性）
   - 可解释安全驾驶：LLM 用自然语言解释不确定性（"因前方大雾，决定减速"）
   - VLM 辅助伪标签标注，提升质量上限
```

---

## 七、里程碑

| 里程碑 | 目标 | 预计日期 |
|--------|------|----------|
| M1 | 特征提取完成，伪标签生成，UQ 分布验证 | 2026-03-28 |
| M2 | Phase 1 开环结果（baseline vs L1），adverse 提升验证 | 2026-03-30 |
| M3 | 修复 planning head 启用问题，验证 plan_results 非空 | 2026-03-29 |
| M4 | FiLM L2 实现 + Attention map 可视化 | 2026-04-02 |
| M5 | 完整 Ablation 开环结果（A/B/C/D 四组） | 2026-04-05 |
| M6 | 闭环结果（碰撞率 + 路线完成率） | 2026-04-10 |
| M7 | 论文初稿 | 2026-04-20 |

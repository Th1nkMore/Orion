# 不确定性空间化利用：设计方案探讨

> 日期：2026-04-01
> 背景：当前 FiLM 方案的根本缺陷分析 + 对标 VLN 论文 + 新方案设计

---

## 一、当前架构的根本问题

### 信息流失路径

```
patch_tokens [B, 6, 1600, 1024]
      │
      ▼  Cross-Attention (16 learnable queries)
attn_weights [B, 16, 9600]   ← 空间信息在此，但被丢弃
      │  mean pool
      ▼
uncertainty_embedding [B, 256]   ← 全局聚合，空间信息消失
uncertainty_score     [B, 1]
      │
      ▼  FiLM
gamma * query + beta             ← 对所有 BEV query 施加相同调制
```

**两个根本缺陷：**

1. **空间盲目**：256 维 embedding 无法区分"前方不确定"还是"侧方不确定"，FiLM 对 900 个 BEV query 一视同仁
2. **Score 未参与调制强度**：embed_head 末尾 LayerNorm 使 embedding 模恒为 √256，正常场景也受到等幅调制 → Normal ADE +116%

### Score-Gated FiLM 的局限

已提出的修复方案（`gamma = 1 + score*(gamma_raw-1)`）解决了问题 2，但问题 1 依然存在：
- 高 score 时 FiLM 全力调制，但调制方向仍是全局的、非空间感知的
- 本质上是"感知到不确定，但不知道哪里不确定，于是全局调整驾驶风格"

---

## 二、对标文献：Uncertainty-Aware Gaussian Map (ICLR 2026 under review)

**论文**：*Uncertainty-Aware Gaussian Map for Vision-Language Navigation*
**链接**：https://openreview.net/forum?id=LPv59noPAy

### 他们的核心思路

针对 VLN（室内具身导航），构建 **3D Semantic Gaussian Map (SGM)**，对每个 Gaussian primitive 建模三类不确定性：

| 不确定性类型 | 建模方法 | 含义 |
|------|---------|------|
| Geometric | 变分推断（位置/尺度扰动后验） | 3D 重建位置是否可信 |
| Semantic | 语言 grounding 歧义 | 空间线索语义是否清晰 |
| Appearance | Fisher Information（渲染损失曲率） | 观测是否对 Gaussian 扰动敏感 |

三类不确定性集成进 **3D Value Map**，直接约束 action 选择空间。

### 与我们的关键差异

| 维度 | VLN 论文 | UQ-ORION |
|------|----------|----------|
| 任务 | 室内 VLN，离散导航步 | 高速自动驾驶，连续轨迹 |
| 不确定性来源 | 3D 重建质量（epistemic） | 传感器退化/恶劣天气（aleatoric） |
| 场景表示 | 在线增量 3D Gaussian 地图 | 单帧多摄像头 BEV query |
| 不确定性注入 | 3D Value Map → action 约束 | FiLM → QT-Former query 调制 |
| 空间保持 | ✓ 显式 3D 空间结构 | ✗ 压缩为全局向量 |

> **结论**：题目相似，但不确定性来源、场景表示、注入方式三者均不同，且室内导航与高速驾驶的动力学约束差异巨大，不构成直接竞争关系。在 related work 中主动对比即为加分项。

---

## 三、新方案设计：空间化不确定性利用

核心问题：**如何把不确定性信息保留空间结构，并用它来指导轨迹决策？**

---

### 方案 A：BEV 不确定性热力图 + 轨迹代价惩罚

**最接近占据网格的思路**

#### 架构

```
patch_tokens [B, 6, 1600, 1024]
      │
      ▼  Per-patch uncertainty head（轻量 MLP，0.3M 参数）
per_patch_uncertainty [B, 6, 1600, 1]
      │
      ▼  Camera-to-BEV lifting（利用已有 lidar2img 矩阵，ground plane z=0 投影）
BEV uncertainty map [B, H_bev, W_bev]   e.g. [B, 50, 50]
      │                │
      │        ▼  可视化/辅助监督
      │
      ▼  对 plan_anchor [20, 6, 2] 每条轨迹做路径积分
per_mode_uncertainty_cost [B, 20]
      │
      ▼  轨迹 mode 选择时加权（score 做门控强度）
      cls_score - λ * score * uncertainty_cost
```

#### 监督信号设计

- **弱监督**：全局 UQ score 做软标签，高 score 帧的 patch uncertainty 平均值应高于低 score 帧
- **强监督（可选）**：用图像梯度/亮度/模糊度等统计量作为 per-patch uncertainty 的伪 GT
- **对比损失**：同一场景不同天气下，恶劣天气帧的 BEV uncertainty map 平均值应显著高于晴天帧

#### 可行性

| 维度 | 评估 |
|------|------|
| 参数量 | Per-patch MLP: ~0.3M；总增量 < 0.5M |
| Camera lifting | ground-plane 假设（z=0）在近处（<30m）精度合理，中远处有偏差 |
| 监督信号 | 弱监督可行，强监督需要额外标注 |
| 对 ORION 侵入性 | 仅在推理时加代价项，不改动 backbone |
| 推理延迟增量 | ~3ms（MLP + bilinear sample） |

#### 创新性

★★★★☆

- 自动驾驶中首次（据我们了解）把 **sensor aleatoric uncertainty** 显式提升为 **BEV 代价图** 用于轨迹 mode 选择
- 与 VLN 论文的 3D Value Map 相似但针对不同任务；与占据网格不同（占据描述障碍物，我们描述感知可靠性）
- 可解释性强：可以直接可视化"模型认为哪个区域感知不可靠"

#### 实用价值

★★★★★

- 直接解决"不知道哪里不确定导致全局激进"的问题
- 轨迹会主动绕开高不确定性区域（如大雾中模糊的前方区域）
- 和碰撞感知 loss 正交，可叠加

#### 主要挑战

- Camera-to-BEV 投影的深度歧义：patch 对应图像中某一列，但深度未知，z=0 假设仅对地面目标成立
- 需要设计合理的路径积分（轨迹 waypoint 与 BEV map 坐标系对齐）

---

### 方案 B：Per-BEV-Query 不确定性分类 + 空间 FiLM

**在 BEV 特征空间直接计算 per-query uncertainty，不需要 camera lifting**

#### 架构

```
QT-Former 输出: BEV queries [B, 900, 256]（frozen，不改动）
      │
      ▼  UQ Spatial Head（轻量 MLP + cross-attn from patch uncertainty）
per_query_uncertainty [B, 900]  →  reshape [B, 30, 30]
      │
      ├──▶  BEV uncertainty heatmap（可视化 / 辅助损失）
      │
      ▼  Spatial FiLM：每个 BEV query 独立调制
gamma_i = 1 + u_i * (gamma_raw_i - 1)   # u_i ∈ [0,1] 是第 i 个 query 的不确定性
beta_i  = u_i * beta_raw_i
query_i = gamma_i * query_i + beta_i
```

这里 gamma_raw/beta_raw 来自全局 uncertainty_embedding（原 FiLM）或 per-query 学习的 linear。

#### 监督信号设计

- QT-Former 的 cross-attention 已经给出了每个 BEV query 对哪些 patch 关注最多
- 可以用 patch-level 统计特征（梯度幅值、模糊度）加权 attention 得到 per-query 不确定性软标签
- 公式：`u_i = Σ_j attn(i,j) * patch_uncertainty(j)`  —— 无需额外标注

#### 可行性

| 维度 | 评估 |
|------|------|
| 参数量 | Spatial Head: ~0.5M；Spatial FiLM: 与原 FiLM 相同量级 |
| Camera lifting | 不需要！直接在 BEV 特征空间操作 |
| 监督信号 | 可从 QT-Former attention + patch 统计量直接构造，无需额外标注 |
| 对 ORION 侵入性 | 需要读取 QT-Former 中间 attention 权重（轻微侵入） |
| 推理延迟增量 | ~5ms |

#### 创新性

★★★★☆

- 空间化 FiLM：不同 BEV 位置接受不同强度的调制，比全局 FiLM 更精细
- 从 QT-Former attention 中"反推"per-query 不确定性的方法较新颖
- 与 Score-Gated FiLM 正交，可以叠加

#### 实用价值

★★★★☆

- 解决了原 FiLM 的空间盲目问题
- 不需要 camera-BEV 投影，工程复杂度低
- 可以与现有 UQEstimator 无缝衔接

#### 主要挑战

- Per-query 不确定性软标签的质量取决于 QT-Former attention 的可解释性
- Spatial FiLM 参数量显著增大（原来 256→256，现在 900 个独立调制）
- 需要在 frozen QT-Former 上挂 hook 读取 attention 权重

---

### 方案 C：不确定性感知的轨迹 Mode 重排序（最轻量）

**不改动任何中间特征表示，只在输出端加一个不确定性感知的 reranking**

#### 架构

```
ORION 原始输出：
  - 20 个 plan_anchor 模式得分 cls_logits [B, 20]
  - 轨迹 waypoints [B, 20, 6, 2]

新增 UQ Reranker：
  uncertainty_emb [B, 256]  +  cls_logits [B, 20]  +  traj_features [B, 20, 12]
        │
        ▼  MLP reranker（0.1M 参数）
  adjusted_logits [B, 20]
        │
        ▼  softmax → mode selection
```

- `traj_features`：把每个模式轨迹的几何特征（最大横向位移、总行驶距离、曲率）拼成 12 维向量
- 训练目标：在高 UQ score 时，reranker 学会偏好"保守"模式（小横向位移、低速）

#### 可行性

| 维度 | 评估 |
|------|------|
| 参数量 | ~0.1M，极轻量 |
| 对 ORION 侵入性 | 几乎为零，只在输出端加一个 MLP |
| 监督信号 | 用碰撞结果做 RL-style 训练，或用现有 closed-loop 数据 |
| 推理延迟增量 | <1ms |

#### 创新性

★★☆☆☆

- 思路相对保守，类似 post-hoc 校准
- 没有利用空间信息，只是把全局 uncertainty 和轨迹几何特征结合

#### 实用价值

★★★☆☆

- 工程上最容易实现，风险最低
- 可作为基线或与其他方案组合使用

---

### 方案 D：Uncertainty-Aware Attention Mask（前向过滤）

**在 QT-Former cross-attention 阶段，让不确定的 patch 降低对 BEV feature 形成的贡献**

#### 架构

```
原始 QT-Former cross-attention：
  attn_weight[i,j] = softmax(Q_i · K_j / √d)

修改后（adapter 形式，不改 frozen 权重）：
  patch_reliability[j] = 1 - patch_uncertainty(j)   ∈ [0,1]
  attn_weight'[i,j] = softmax((Q_i · K_j / √d) + log(patch_reliability[j]))
  # 等价于在 logit 上减去不可靠性分数
```

`patch_uncertainty(j)` 来自 UQEstimator 的 attn_weights 均值（无需额外计算）。

#### 可行性

| 维度 | 评估 |
|------|------|
| 参数量 | 基本无新增参数，复用 UQEstimator attn_weights |
| 对 ORION 侵入性 | 中等：需要在 QT-Former 中注入 attention bias |
| Backbone frozen | 需要用 register_forward_hook 方式注入，不修改权重 |
| 推理延迟增量 | ~2ms |

#### 创新性

★★★☆☆

- "让模型忽略不可靠的视觉输入"这个想法直观，但在 BEV 感知中较新
- 和 CBAM/SE-Net 的通道注意力不同，我们的 mask 基于外部不确定性信号而非自监督

#### 实用价值

★★★★☆

- 从源头提升 BEV feature 质量，可能比下游调制更有效
- 对所有下游任务（轨迹、速度等）都有收益

---

## 四、方案综合对比

| 维度 | A：BEV热力图+代价 | B：Spatial FiLM | C：Mode重排序 | D：Attention Mask |
|------|:-----------:|:-----------:|:---------:|:-----------:|
| **创新性** | ★★★★☆ | ★★★★☆ | ★★☆☆☆ | ★★★☆☆ |
| **实用价值** | ★★★★★ | ★★★★☆ | ★★★☆☆ | ★★★★☆ |
| **可行性（实现难度）** | 中 | 中 | 易 | 中 |
| **参数增量** | ~0.5M | ~0.8M | ~0.1M | ~0M |
| **需要 camera lifting** | ✓（z=0近似） | ✗ | ✗ | ✗ |
| **空间感知** | ✓（显式BEV map） | ✓（per-query） | ✗ | ✓（per-patch） |
| **与现有 FiLM 兼容** | ✓ | 替代现有FiLM | ✓ | ✓ |
| **可视化/可解释性** | ★★★★★ | ★★★☆☆ | ★★☆☆☆ | ★★★★☆ |
| **对 ORION frozen backbone 友好** | ✓ | 需小改 | ✓ | 需hook注入 |
| **与 VLN 论文的差异度** | 高（不同投影方式+任务） | 高 | 很高 | 高 |

---

## 五、推荐路线

### 短期（论文初版）：A + C 组合

1. **方案 C**（Mode 重排序）：1周内可实现，作为 baseline 消融对照组
2. **方案 A**（BEV 热力图 + 轨迹代价）：2周内可实现核心模块，是主要贡献

**论文叙事逻辑**：
> *"现有方法（包括我们的 FiLM）使用全局不确定性信号，忽视了不确定性的空间分布。受占据网格启发，我们提出把感知不确定性显式映射到 BEV 代价图，直接约束轨迹 mode 选择，使规划器主动规避感知可靠性低的区域。"*

### 中期（完整方案）：A + B 组合

- 方案 A 提供"哪里不确定"的空间信息，作用于轨迹选择
- 方案 B 提供细粒度的 BEV feature 调制，作用于特征表示
- 两者互补，可做消融实验展示各自贡献

### 与 VLN 论文的差异化定位

在 related work 中写：

> *While [LPv59noPAy] builds explicit 3D Gaussian uncertainty maps for indoor navigation, our work addresses a fundamentally different problem: sensor-degradation-induced aleatoric uncertainty in high-speed autonomous driving. Rather than reconstructing 3D scene geometry, we directly lift per-patch vision uncertainty to a BEV cost field and integrate it into trajectory mode selection, requiring no online map building and operating within a single-frame frozen E2E backbone.*

---

## 六、下一步行动

- [ ] 实现 Score-Gated FiLM（修复 LayerNorm 问题，3行代码 + 重训）
- [ ] 实现方案 A：per-patch uncertainty MLP + ground-plane BEV lifting + mode cost
- [ ] 设计 per-patch 监督信号（图像统计量作为伪 GT）
- [ ] 实现方案 C 作为消融基线
- [ ] 扩展实验到 100 场景闭环评估

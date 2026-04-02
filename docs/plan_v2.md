# UQ-ORION 研究计划 v2

> 版本：2026-04-02 rev.3（IPM 验证后更新）
> 状态：Phase -1 部分完成，Phase 0 完成，Phase 1 代码完成待重训
> 目标：申请计算资源前的完整规划
>
> **rev.3 主要变更**（IPM 验证）：
> - Phase -1 Q1 重新分类：Flash Attn 对 BEV 主链路不再是阻塞项（IPM 不需要 attention）
> - Phase 0 已完成：B2D 两场景验证，Δ=+0.139，两组分布无重叠
> - BEV 不确定性主方案升级为 IPM（纯几何），Attention-based 降为消融 F 组
> - 关键实现发现：log-scale 归一化是必需的（线性归一化因重尾分布使 Δ→+0.011）
> - Phase 2 架构更新：BEV 提取不再依赖 QT-Former hook
>
> **rev.2 主要变更**（二次审查）：
> - 新增 Phase -1：阻塞性可行性验证（Flash Attention、BEV query 布局、poses_cls 分布、LLM-free 加载）
> - Phase 1：修正代码改动范围——需额外传递 uq_score 到 FiLM 层（涉及 3 个文件而非 2 个）
> - Phase 2：新增 Flash Attention 阻断风险及缓解；新增 BEV query 空间布局验证步骤
> - Phase 3：修正 poses_cls 为 sigmoid 非 softmax 的问题；用 pairwise ranking loss 替代不可微的 Spearman
> - Phase 4：新增 G 组消融（C+F，验证 FiLM+BEV 组合 vs FiLM+Global 的增量价值）
> - 风险表新增 5 项：Flash Attention、score 分布、plan_anchor 覆盖、poses_cls 空间、LLM-free 加载
> - 修正定位措辞："lightweight side-channel injection" 替代 "plug-and-play"
>
> **rev.1 变更**（Gemini 反馈）：
> - Phase 0：新增 Local Contrast 质量指标
> - Phase 2：补充注意力非局域性风险及 Attention Rollout 备用方案；修正 attn_weights 显存估算
> - Phase 3：λ 训练目标加入排序相关性优化
> - Phase 4：新增 F 组消融（全局惩罚 vs 空间惩罚，证明 BEV Map 的空间信息有独立贡献）

---

## 一、研究目标

在极端感知条件（雨、雾、夜晚）下，端到端自动驾驶模型（ORION）因图像质量退化而产生不安全的规划行为。本项目目标：

1. **量化感知不确定性**：设计轻量 UQEstimator，从视觉 token 中提取每帧的不确定性分数与嵌入
2. **空间化不确定性**：将不确定性从全局标量升维为 BEV 空间热力图，使规划器感知"哪个区域的感知不可靠"
3. **安全规划**：用不确定性约束轨迹模式选择，主动规避感知可靠性低的区域

**核心创新点**：首次在高速端到端自动驾驶中，将 aleatoric 感知不确定性（传感器退化型）提升为 BEV 代价图，直接约束多模态轨迹决策，参数增量 < 0.5M，不侵入冻结的 backbone。

**定位**：*Lightweight Side-Channel Safety Injection for Frozen E2E Models*——在冻结的端到端自动驾驶模型的 forward path 中注入轻量不确定性调制（< 2.5M 参数），无需重训视觉骨干网络或语言模型。注意：本方法修改了冻结层的中间激活值（FiLM 乘加），而非纯粹的 adapter/prompt tuning，因此更准确的描述是"side-channel injection"而非"plug-and-play"。

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
                    → QT-Former → BEV queries [B, 600, 256]  （600 detection + 256 VLM queries）
                               → attn_weights [B, N_q, 9600]  （N_q=600 或 900，待 hook 验证）  ← 读取（600 detection queries）
                    → LLM → ego_feature [B, 4096]
                    → VAE + plan_anchor → 20个轨迹模式 [B, 20, 6, 2]
                    → poses_cls [B, 20]  ← **sigmoid scores**（非 softmax logits）
                    → 最终轨迹：argmax(poses_cls) 选 mode

UQ 扩展模块（新训练）:
  ┌─ [已有] UQEstimator:
  │    patch_tokens → uncertainty_score [B,1] + uncertainty_emb [B,256]
  │
  ├─ [修复] Score-Gated FiLM (L1+L2):
  │    gamma = 1 + score*(gamma_raw - 1)   ← 正常场景 score≈0 → 恒等
  │    beta  = score * beta_raw
  │    **注意**：需要同时传递 uq_score 和 uncertainty_emb（当前代码只传了 emb）
  │
  └─ [新增] BEV Uncertainty Cost:
       ① patch_quality[j] = f(Laplacian_var, gradient_mag, contrast)  # 无参数
       ② bev_uncertainty[i] = Σ_j attn_weight[i,j] * patch_quality[j]  # 线性传播，i∈[0,600)
       ③ uncertainty_cost[m] = mean_t knn_interp(bev_unc, ref_xy, waypoint[m,t], k=5)  # k-NN 插值
       ④ adjusted_score = poses_cls * (1 - λ * score * uncertainty_cost)  # 乘法门控
       # 或 adjusted_logit = logit(poses_cls) - λ * score * uncertainty_cost  # logit 空间减法
       # 具体方案取决于 Phase -1 对 poses_cls 分布的分析
```

### 3.2 关键设计决策

**为什么不学习 per-patch uncertainty head？**
- 无 per-patch GT，伪标签质量不可靠
- Image quality metrics（Laplacian variance、梯度幅值）与恶劣天气直接对应，物理含义明确，无需学习
- 保持方案可解释性，每步都是可验证的确定性计算

**为什么用 IPM（camera-to-BEV lifting）而非 QT-Former attention 传播？**
- **Flash Attention 阻断**：QT-Former 使用 Flash Attn，无法在 inference 时获取 attn_weights
- **IPM 已在 B2D 上验证有效**（Δ=+0.139，两组无重叠）
- IPM 物理含义更直接："哪些地面区域被相机清晰覆盖"，无需依赖模型内部表示
- Attention-based 路径仍可作为消融 F 组（需关闭 Flash Attn，推理速度 2× 下降）

**λ 为什么只有 1 个参数？**
- 避免过拟合小数据集
- 物理意义清晰：λ 控制不确定性代价的全局权重
- 可扩展为 λ(score)：按 UQ score 大小动态调整权重

**poses_cls 是 sigmoid 而非 softmax——注入方式需要注意**
- ORION 用 `py_sigmoid_focal_loss` 训练 `poses_cls`，各 mode 的分数是独立 sigmoid，非互斥
- 直接在 sigmoid 空间做减法（`poses_cls - cost`）语义不清晰，且在分数极端（>0.9 或 <0.1）时无效
- 两种备选方案：
  1. **乘法门控**：`adjusted = poses_cls * (1 - λ·score·cost)`，保证值域仍在 [0,1]
  2. **logit 空间操作**：`adjusted = σ(logit(poses_cls) - λ·score·cost)`，等价于在 pre-sigmoid logit 上操作
- 具体选择取决于 Phase -1 对 poses_cls 值分布的分析（若集中在 0.5 附近，两种方案差别不大）

---

## 四、分阶段计划

---

### Phase -1：阻塞性可行性验证（最高优先级）
**目标**：在投入任何实现工作之前，验证计划的核心技术假设是否成立
**前置条件**：现有 eval 结果 + ORION 代码 + ORION checkpoint
**计算资源**：本地 CPU/GPU 均可，每项 <30 分钟

#### 必须验证的 5 个问题

**Q1：Flash Attention 是否阻止 attn_weights 提取？**
- **✅ 已确认阻断，但主链路已绕开。**
- QT-Former decoder 层 `flash_attn=True`，确认无法提取 attn_weights
- **BEV 主方案已切换为 IPM（无需 attention）**：patch quality → 相机标定 → 地面平面投影 → BEV 网格
- Flash Attn 阻断仅影响消融 F 组（Attention-based BEV），该组需要临时设 `flash_attn=False`
- **当前状态**：非阻塞。IPM 方案已在 B2D 数据上验证有效（Δ=+0.139）

**Q2：BEV query 是否排列在规则网格上？**
- **已确认：不是。** 实际组成：600 main queries + 300 temporal propagated queries = 900 总 query，均为 `nn.Embedding` learned 参数，`uniform(0,1)` 初始化
- BEV 空间范围：±51.2m（`pc_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]`）
- 另有 256 个 VLM query（`num_extra=256`）无空间 reference_points，不参与 BEV 空间计算
- 位置编码路径：`reference_points [600/300, 3]` → NERF encoding (6 bands) → MLP → `query_pos [*, 256]`
- **attn_weights shape 可能仍是 `[B, 900, 9600]`**，需 hook 验证 propagated queries 是否参与 cross-attention
- **决策**：reshape 为 30×30 grid 不可用，改用 k-NN 插值（k=5，基于 reference_points 的 xy 坐标）
- **待服务器补充**：加载 trained reference_points 可视化 900 个 query 的实际空间分布

**Q3：poses_cls 的值分布如何？**
- `poses_cls` 用 sigmoid focal loss 训练，是独立 per-mode sigmoid score（非 softmax 互斥概率）
- 若分布极端（多数 >0.9 或 <0.1），加减一个小的 cost 对 argmax 无影响
- **验证方法**：从已有 `results/eval_openloop_v3.pt` 提取 poses_cls，统计 mean/std/histogram
- **关联决策**：确定 BEV cost 应在 sigmoid 空间还是 logit 空间操作

**Q4：20 个 plan_anchor 的 BEV 空间覆盖是否有足够差异？**
- BEV cost 的区分力取决于不同 mode 轨迹是否经过不同的 BEV 区域
- 若 20 个 anchor 都集中在正前方 ±3m 的狭窄走廊，所有 mode 的 `uncertainty_cost` 接近相等 → BEV 空间信息无法区分 mode → D≈F 几乎必然
- **验证方法**：加载 `plan_anchor.pkl`，可视化 20 个 anchor 的 BEV 空间分布；计算 pairwise BEV 区域重叠率
- **若重叠率 >80%**：BEV cost per-mode 区分度不足，可能需要方案 B（per-query Spatial FiLM）替代

**Q5：能否在 16GB 机器上只加载 EVAViT + QT-Former？**
- 计划假设 `cfg.model.lm_head = None` 可跳过 LLM 初始化
- 但 ORION 的 `build_model` + `load_state_dict` 可能仍会实例化完整模型再加载权重
- **验证方法**：写 10 行脚本，尝试 `lm_head=None` 加载，监控峰值显存
- **若失败**：需手动拆解 state_dict，只加载 vision encoder + transformer 的 key

#### 产出

- `results/feasibility/feasibility_report.md`：5 个问题的验证结果和决策
- **决策点**：任何一个 Q 的结果为"阻断且无缓解方案"，需要重新设计对应 Phase

---

### Phase 0：信号验证 ✅ 已完成（2026-04-02）
**目标**：验证 image quality metrics 能有效区分恶劣/正常天气 patch，确认信号可用性
**实际方法**：下载 B2D 两个场景（各 ~150MB），用 IPM 方案端到端验证

#### 完成情况

1. 下载 B2D 样本数据（见 `scripts/download_b2d_sample.py`）：
   - Normal: `AccidentTwoWays_Town12_Route1121_Weather3`（CloudySunset，181MB）
   - Adverse: `AccidentTwoWays_Town12_Route1105_Weather13`（HardRainNight，146MB）
2. 实现 `compute_bev_uncertainty_ipm` + `make_b2d_calibration`（见 `uq_estimator/bev_uncertainty.py`）
3. 端到端评估（见 `scripts/eval_bev_noattn.py`）：
   - 每场景 10 帧 × 6 相机 → per-patch 质量 → IPM 投影 → 256×256 BEV 网格
   - **log-scale 全局归一化**是必需的（线性归一化使 Δ 退化为 +0.011）

#### 定量结果

| 条件 | BEV 不确定性均值（覆盖像素）| Std |
|------|--------------------------|-----|
| Normal (CloudySunset) | 0.583 | 0.019 |
| Adverse (HardRainNight) | **0.722** | 0.027 |
| **Δ** | **+0.139** | — |

原始 patch 质量：Normal 68.97 vs Adverse 31.61（**2.2× 差异**，两组 BEV 分布无重叠）。

#### 产出

- `data/b2d_sample/` ：两个 B2D 场景（各 15 帧图像，gitignored）
- `results/bev_noattn/comparison.png`：定量对比图（Δ=+0.139 标注）
- `results/bev_noattn/mean_bev_maps.png`：平均 BEV 热力图（normal vs adverse）
- `results/bev_noattn/panel_*.png`：逐帧相机图像 + BEV 热力图

#### 决策

✅ **信号有效（p 值显著，效应量大）。继续 Phase 2，使用 IPM 作为主方案。**

---

### Phase 1：Score-Gated FiLM 修复与重训
**目标**：修复 LayerNorm 范数崩塌问题，使 FiLM 在正常场景下不影响原始规划
**前置条件**：需要 ORION 完整推理（LLM INT8 约 12 GB）
**计算资源**：本地 16 GB（INT8 量化），或服务器

#### 要做的事

1. 修改 `mmcv/models/dense_heads/orion_head.py`（~2 行）：将 `uq_score` 与 `uncertainty_emb` 一同传递给 transformer 和返回给 orion.py
2. 修改 `mmcv/models/utils/petr_transformers.py`（~5 行）：forward 签名增加 `uncertainty_score`，实现 Score-Gated FiLM L1
3. 修改 `mmcv/models/detectors/orion.py`（~5 行）：接收 `uq_score`，实现 Score-Gated FiLM L2
4. 重训 FiLM（使用现有 `scripts/train_film.py`，collision-aware loss）
5. 开环验证：Normal ADE 应恢复到接近 baseline，Adverse Col 保持或改善

**实际涉及 3 个文件，约 12 行改动**（原计划低估为 2 个文件 6 行）。

#### 代码改动（明确）

```python
# === orion_head.py（~2 行）===
# 当前只传 uncertainty_emb，需要同时传 uq_score
uncertainty_score = uq_out.score  # [B, 1]  ← 新增
# 传给 transformer:
outs_dec = self.transformer(..., uncertainty_emb=uncertainty_emb, uncertainty_score=uncertainty_score)
# 返回给 orion.py:
return outs, vlm_memory, uncertainty_emb, uncertainty_score  # ← 增加一个返回值

# === petr_transformers.py（~5 行）===
# forward 签名增加 uncertainty_score=None
def forward(self, query, key, ..., uncertainty_emb=None, uncertainty_score=None):
    if uncertainty_emb is not None and self.use_uncertainty:
        gamma_raw = self.film_gamma(uncertainty_emb).unsqueeze(1)   # [B,1,D]
        beta_raw  = self.film_beta(uncertainty_emb).unsqueeze(1)    # [B,1,D]
        score     = uncertainty_score.unsqueeze(-1)                  # [B,1,1]
        gamma     = 1.0 + score * (gamma_raw - 1.0)
        beta      = score * beta_raw
        query     = gamma * query + beta

# === orion.py（~5 行）===
# 接收新返回值:
outs, det_query, _uncertainty_emb, _uncertainty_score = self.pts_bbox_head(...)
# L2 FiLM 同理:
if self.use_uncertainty_l2 and _uncertainty_emb is not None:
    gamma_raw_l2 = self.film_gamma_l2(_uncertainty_emb)
    beta_raw_l2  = self.film_beta_l2(_uncertainty_emb)
    s = _uncertainty_score.unsqueeze(-1)  # [B,1,1]
    gamma_l2 = 1.0 + s * (gamma_raw_l2.unsqueeze(1) - 1.0)
    beta_l2  = s * beta_raw_l2.unsqueeze(1)
    current_states = gamma_l2 * current_states + beta_l2
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
     - `attn_weights [B, N_q, 9600]  （N_q=600 或 900，待 hook 验证）`：从 QT-Former cross-attention 读取（`register_forward_hook`）
     - `bev_uncertainty [B, N_q]`：attention-weighted patch quality
     - `poses_cls [B, 20]`：per-mode sigmoid 得分（**注意：此步仍需 LLM + diffusion decoder，需另行缓存**）
   - 保存缓存到 `data/bev_cache/`

2. 可视化验证：
   - 热力图可视化：用 600 个 query 的 learned reference_points (x,y) 做散点图/Voronoi 渲染（非规则网格，不可 reshape 为矩阵）
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

#### Phase 2 核心假设的风险与缓解

**假设**：BEV query 对某 patch 的注意力权重 ≈ 该 patch 对该 BEV 区域规划质量的影响程度。

**合理性**：有 Attention Rollout、Grad-CAM 等可解释性研究先例，Cross-Attention 权重反映信息流向，物理意义基本成立。

**已知风险 1：Flash Attention 阻断（rev.2 新增，高优先级）**
- `OrionTransformerDecoderLayer` 有 `flash_attn=True` 选项
- Flash Attention 内核不输出 attention weights，hook 无法拦截
- **必须在 Phase -1 确认**；若启用需在提取时临时关闭 `flash_attn=False`

**已知风险 2：注意力的非局域性**
- QT-Former 使用多层 Transformer，最终 attention 经过多次 softmax renormalization，导致权重"弥散"——即使某个 patch 与某 BEV query 几何上不相关，也可能因为全局归一化获得非零权重
- 结果：BEV uncertainty map 可能过于平滑，丧失空间分辨率

**已知风险 3：BEV query 布局可能非规则网格（rev.2 新增）**
- **已确认**：600 个 learned query，非规则网格，需用 k-NN 插值替代双线性采样
- 需 Phase -1 验证，若非均匀需用 k-NN 或 Voronoi 插值替代双线性采样

**缓解方案（分优先级）**：
1. **首选**：直接用最后一层 cross-attention（而非累积 rollout），空间局域性更强
2. **备用**：Gradient-weighted attention（类 Grad-CAM）：`attn_weight * |∂output/∂attn|`，保留空间敏感度
3. **验证**：Phase 2 验证时计算热力图的空间熵（高熵=弥散，低熵=集中），若大多数帧熵过高则切换到方案 2

**显存估算修正（Gemini 提示）**：
- `attn_weights [B, N_q, 9600]  （N_q=600 或 900，待 hook 验证）`：N_q × 9600 × 4 bytes = **23~34 MB/sample (fp32)**，按序处理存磁盘即可，不需同时在显存中保存全部验证集

#### 关键技术细节：如何不加载 LLM

```python
# 方案 A：修改 build_orion_model 配置
cfg.model.lm_head = None
cfg.model.tokenizer = None
# 风险：ORION 的 build_model 可能在 __init__ 中仍实例化完整模型

# 方案 B（更稳妥）：手动拆解 state_dict
full_ckpt = torch.load('Orion.pth', map_location='cpu')
vision_keys = {k: v for k, v in full_ckpt.items()
               if k.startswith(('img_backbone.', 'pts_bbox_head.transformer.'))}
# 只构建 EVAViT + QT-Former 子模块，手动加载对应权重
```

**Phase -1 的 Q5 必须先验证方案 A 是否可行**；若不可行则用方案 B。

`cls_logits`（即 `poses_cls`）需要额外一步：在有 LLM 的机器（或 INT8）上一次性提取并保存。注意 `poses_cls` 是 diffusion decoder 最后一步的输出（经过 50 步去噪），依赖完整的 LLM → ego_feature → VAE 链路。

#### 产出

- `data/bev_cache/{scene_id}.pt`：每帧的 `bev_uncertainty [N_q]` + `patch_quality [6,1600]` + `ref_points_xy [N_q, 2]` + `frame_meta`
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
   - 计算 `uncertainty_cost [B, 20]`：对每个 plan_anchor 轨迹做 BEV uncertainty 采样 + 平均
     - 采样方式取决于 Phase -1 Q2 结果：均匀网格用双线性，非均匀用 k-NN
   - **优化目标（rev.2 修正——Spearman 不可微，改用 pairwise ranking）**：
     ```
     Loss = α * Collision_Loss
          + (1-α) * Σ_{i,j: safety[i]>safety[j]} max(0, margin - (adjusted[j] - adjusted[i]))
     ```
     即 pairwise margin ranking loss：对每对 mode，若 mode j 比 mode i 更安全（L2 更小），则 adjusted_score[j] 应高于 adjusted_score[i]。
     - 理由：Spearman Rho 基于 rank 排序操作（`torch.argsort`），不可微分，无法反传梯度。Pairwise ranking loss 在连续空间中直接优化排序关系，效果等价且可微。
     - `α` 建议初始设为 0.5，做超参搜索
     - **备选方案（无需梯度）**：λ 只有 1 个标量参数，可直接 grid search（λ ∈ {0.01, 0.05, 0.1, 0.5, 1.0, 2.0}），用 validation set 上的碰撞率/ADE 选最优。比训练更简单可靠。
   - 使用验证集 held-out 分割，防止过拟合（k=5 fold）

**关于 poses_cls 分布问题（rev.2 新增，依赖 Phase -1 Q3）**：
- `poses_cls` 是 per-mode 独立 sigmoid score（非 softmax 互斥概率），最终用 argmax 选 mode
- 风险 1：若分数集中在极端值（>0.9），减法/乘法微调对 argmax 结果无影响
- 风险 2：若分数非常平坦（std<0.01），所有 mode 几乎等价，任何微扰都可能随机翻转结果
- 缓解：Phase -1 Q3 检查分布后决定操作空间（sigmoid vs logit）

**关于 BEV cost 注入时机（rev.2 新增）**：
- ORION 使用 50 步 diffusion denoising 生成轨迹，`poses_cls` 是最后一步输出
- BEV cost 应只在最终 argmax 前注入（不在每步去噪时注入），否则计算量 ×50
- 这意味着 BEV cost 只影响 mode selection，不影响 mode 内的轨迹生成

**关于 VAE 随机性问题（Gemini 提示）**：
- `poses_cls` 是 mode 选择分数，最终轨迹还经过 VAE sampling
- 风险：`poses_cls` 分布平坦时（std 极小），λ 扰动可能无效
- 缓解：记录各帧 poses_cls 的分布统计，若 std < 0.01 说明 mode selection 本身不稳定

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
- **验证标准**：adjusted score 的 mode 选择在 adverse 场景的碰撞率应低于原始 `poses_cls`

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

#### 消融实验矩阵（rev.2 新增 G 组）

| 组别 | Score-Gated FiLM | BEV Cost | 说明 |
|------|:----------------:|:--------:|------|
| A: Baseline | ✗ | ✗ | 纯 ORION，参考基线 |
| B: FiLM-v3 (现有) | ✗（buggy）| ✗ | Normal ADE 退化，展示问题 |
| C: Score-Gated FiLM | ✓ | ✗ | 修复 normal 退化 |
| D: BEV Cost only | ✗ | ✓（spatial） | 验证空间不确定性的独立贡献 |
| F: Global Score Penalty | ✗ | λ·score（global） | 对照组：全局 vs 空间 |
| **G: FiLM + Global（rev.2 新增）** | **✓** | **λ·score（global）** | **对照组：C+D vs C+F** |
| E: Full (C+D) | ✓ | ✓（spatial） | 最终完整方案 |

**关键对比关系**：
- **D vs F**（Gemini 提出）：BEV 空间信息是否比全局标量有额外收益
- **E vs G**（rev.2 新增）：在已有 FiLM 的基础上，BEV 空间 cost 是否比全局 cost 更好。这是更严格的测试——如果 E≈G，说明空间信息在 FiLM 已调制的情况下冗余
- 如果 D 显著优于 F **且** E 显著优于 G，说明 BEV 空间信息在任何配置下都有独立贡献
- 如果 D ≈ F，说明空间信息贡献有限，需要重新审视方案 B（per-BEV-query Spatial FiLM）

**plan_anchor 覆盖率风险（rev.2 新增）**：
- 若 Phase -1 Q4 发现 20 个 anchor 空间覆盖高度重叠（>80%），D≈F 几乎必然成立
- 此时 BEV cost 的价值不在于区分现有 mode，而在于提供可视化和可解释性（论文 Fig 6）
- 考虑退而求其次：BEV cost 作为辅助可解释性工具，而非核心规划组件

#### 计算资源估算

| 任务 | 硬件 | 时间 |
|------|------|------|
| 开环评估 500帧 × 7组 | 1×A100 40GB | ~14 h |
| 闭环 100场景 × 4组 (A/C/E/best-of-D,F,G) | 1×A100 + CARLA | ~60 h |
| FiLM 重训（Score-Gated）| 1×A100, bs=4 | ~24 h |

#### 产出

- `results/eval_openloop_full_v4.json`：7组开环结果（A/B/C/D/F/G/E）
- `results/closedloop_full_v4.json`：100场景闭环结果（4组）
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
| Fig 4 | 消融实验柱状图（7组 × normal/adverse） | Phase 4 产出 |
| Fig 5 | 轨迹对比 GIF（GT / Baseline / Ours，恶劣场景选例） | `scripts/generate_trajectory_gifs.py` |
| Fig 6 | BEV uncertainty map + 轨迹 mode 选择可视化 | 新脚本 |
| Fig 7 | 校准曲线（UQ score vs 实际 L2/碰撞率）| `scripts/visualize_eval.py` |

---

## 五、计算资源需求汇总

### 本地 16GB 可完成（无需申请）

| Phase | 任务 | 显存 | 备注 |
|-------|------|------|------|
| -1 | 可行性验证（5 项） | <1 GB | 只需加载 config 和少量权重 |
| 0 | 信号验证 | CPU | 纯 numpy（**前提：B2D 数据集在本地**） |
| 2 | BEV uncertainty map 提取 | ~1 GB | 只加载 EVAViT+QT-Former（**需 Phase -1 Q5 验证**） |
| 3 | λ 训练/grid search | CPU | 只读缓存 |
| 1（可试） | Score-Gated FiLM 代码修改 | 0 | 涉及 3 个文件约 12 行 |
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
| 4 | 开环评估（**7组**，含 F/G 组） | 1×A100 40GB | 14 h | 高 |
| 4 | 闭环评估（100场景 × **4组** A/C/E/best） | 1×A100 + CARLA | 60 h | 高 |
| 4 | cls_logits + attn_weights 缓存提取（全验证集）| 1×A100 40GB | 4 h | 中 |

**申请资源时的 justification**：
- FP16 LLM 需要 ~18 GB，超出消费级 GPU 上限，必须用 A100/V100
- 闭环评估需要实时 CARLA 仿真，无法在推理机上跑

---

## 六、风险分析

| 风险 | 可能性 | 影响 | 缓解方案 | Phase |
|------|--------|------|---------|-------|
| **Flash Attention 阻止 attn_weights 提取** | ✅已确认 | 高 | 提取时临时 `flash_attn=False`（速度 ~2x 降低） | -1→2 |
| **LLM-free 加载失败（ORION build 仍实例化 LLM）** | 中 | 高 | 手动拆解 state_dict 只加载 vision 部分的 key | -1 |
| ~~plan_anchor 空间覆盖高度重叠~~ | ✅已排除 | — | IoU mean=0.261，88.9% pair <0.5，覆盖充分 | -1 |
| **poses_cls 分布极端（sigmoid>0.9），cost 调整无效** | 中 | 高 | 改在 logit 空间操作（`logit(poses_cls) - cost`） | -1 |
| ~~UQ score 分布不够双峰~~ | ✅已排除 | — | Normal mean=0.023, median≈0，高度双峰 | -1 |
| QT-Former attention hook 读取失败 | 中 | 高 | 改 QT-Former forward 接口，显式返回 attn_weights | 2 |
| **BEV map 过于平滑（注意力非局域性）** | 中 | 中 | Phase 2 计算热力图空间熵；过高则切换 Gradient-weighted attention | 2 |
| **BEV query 600 个 learned 非网格** | ✅已确认 | 中 | 已决定：k-NN 插值（k=5）替代双线性 | -1→2 |
| BEV uncertainty map 与 UQ score 相关性弱 | 低 | 高 | Phase 0 验证，失败则换 patch quality 指标组合 | 0 |
| **D 组 ≈ F 组（空间信息无额外贡献）** | 中 | 高 | 若成立，转向方案 B（Per-BEV-query Spatial FiLM）| 4 |
| **poses_cls 分布平坦（std<0.01），mode 选择不稳定** | 中 | 中 | 记录分布统计；若 std 极小说明 ORION 本身 mode collapse | 3 |
| INT8 LLM 量化导致 FiLM 训练不稳定 | 中 | 中 | 用 bfloat16 代替，或 CPU offload | 1 |
| λ 过拟合小数据集 | 低 | 低 | grid search 替代梯度训练；备选：固定 λ=1 零训练验证 | 3 |
| Normal ADE 仍不达 baseline | 低 | 高 | Score-Gated 设计理论保证 score=0 时恒等；但需验证 score 分布 | 1 |

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

**补充定位（修正措辞）**：
> *Our approach is a lightweight side-channel injection for frozen end-to-end autonomous driving models. By adding fewer than 2.5M trainable parameters (UQEstimator + FiLM + λ) and requiring no retraining of the vision backbone or language model, it enables safety-aware planning under adverse perception conditions. Note that our FiLM layers modify intermediate activations within the frozen forward path — this is distinct from pure adapter or prompt-tuning approaches, but still avoids the prohibitive cost of full backbone retraining.*

---

## 八、本地可立即开始的工作

按优先级排列，无需服务器即可推进：

1. **最高优先（今天）**：Phase -1 可行性验证
   - Q1：检查 `orion_stage3_infer.py` 中 `flash_attn` 配置
   - Q2：~~加载 checkpoint 可视化~~ **已确认：600 learned queries，非网格**（待服务器看训练后分布）
   - Q3：从 `eval_openloop_v3.pt` 提取 `poses_cls` 分布统计（mean/std/histogram）
   - Q4：加载 `plan_anchor.pkl`，可视化 20 个 anchor 的 BEV 覆盖并计算 pairwise 重叠率
   - Q5：尝试 `lm_head=None` 加载 ORION，检查峰值显存
   - 额外：检查现有 UQ score 分布是否足够双峰
2. **第二优先**：Phase 0 信号验证（`scripts/validate_patch_quality.py`）
   - **前提**：确认 B2D 原始 RGB 图像在本地可用
3. **第三优先**：Score-Gated FiLM 代码修改（3 个文件 ~12 行，不需要 GPU）
4. **第四优先**：Phase 2 BEV 热力图提取（依赖 Phase -1 Q1/Q2/Q5 的结果）
5. **第五优先**：`scripts/train_lambda.py`（依赖 Phase 2 缓存数据）

---

## 附录：文件依赖关系图

```
Phase -1（可行性验证，本地，无数据集依赖）
    │   Q1: flash_attn? → 决定 Phase 2 能否提取 attn_weights
    │   Q2: BEV grid?   → 决定 Phase 2/3 的采样方式
    │   Q3: poses_cls?  → 决定 Phase 3 的 cost 注入方式
    │   Q4: anchors?    → 决定 BEV cost 区分力是否足够
    │   Q5: LLM-free?   → 决定 Phase 2 本地可行性
    ▼
B2D 数据集（需确认本地可用性）
    ├── Phase 0 → patch_quality_stats.json（决策点）
    ├── Phase 1 → checkpoints/film/best_score_gated_v1.pt
    │              + poses_cls + attn_weights 缓存（服务器上提取）
    ├── Phase 2 → data/bev_cache/*.pt（依赖 Phase -1 Q1/Q2/Q5）
    │              + BEV heatmap 可视化图
    └── Phase 3 → checkpoints/lambda/best_lambda.pt（依赖 Phase -1 Q3/Q4 + Phase 1+2）
                   ↓
              Phase 4（完整评估，服务器，7组消融含 G 组）
                   ↓
              Phase 5（论文图表）
```

# UQ-ORION 组内进展报告（0521）

> **状态**：大纲 v0.2 · 已确认 · 正文撰写中  
> **受众**：组内汇报（导师 / 同组）  
> **原则**：无附录、无外链；图表与数字在正文中直接展示；负结果写成诊断产出  
> **素材**：正文写作时从仓库复制图表到 `0521report/assets/`，报告内只引用本地路径

---

## 审核批注（请在此填写）

| 项 | 你的意见 |
|----|----------|
| 章节数（当前 8 章）是否合适？ | |
| BEV 单独成章还是并入「结果」？ | |
| 是否需要压缩「Phase 2 计划」篇幅？ | |
| 篇幅目标（页数 / 演讲分钟）？ | |
| 其他： | |

---

## 一句话定位（摘要草稿）

在冻结 ORION 上，用不足 5M 参数的侧信道完成恶劣天气感知监测（开环 AUROC 0.954），并揭示开环 L2 与碰撞安全正交（恶劣场景 L2 更低、Col 高约 161 倍）。全局 FiLM 在 50 场景回放中降低恶劣碰撞约 23%，但正常场景 ADE 退化约 116%；并行完成的 IPM BEV 不确定性可视化表明恶劣场景空间不确定性显著更高（Δ≈+0.14）。下一步：Score-Gated FiLM 重训 + BEV 代价注入规划。

---

## 叙事主线（五幕）

```text
第1幕  悖论      L2 好、碰撞 161× → 需要新的安全视角
第2幕  方案      UQ 监测 + 双层 FiLM（零主干微调）
第3幕  分裂结果  UQ 强 / FiLM 有代价
第4幕  BEV 进展  IPM 热力图 + 轨迹 GIF → 空间化方向已验证信号
第5幕  诊断升级  根因 + Phase 2（门控、λ、消融）
```

---

## 章节结构（共 8 章，无附录）

| ID | 文件 | 标题 | 正文必须展示的内容 | 状态 |
|----|------|------|-------------------|------|
| 0 | `ch00-摘要.md` | 摘要 | 背景、方法、三条结论、一条局限、一句下一步 | 待写 |
| 1 | `ch01-背景与问题.md` | 背景与问题 | 正文 + **11 页 PPT 脚本**（叙事线、逐页元素、承上启下） | ✅ v2（含 PPT） |
| 2 | `ch02-方法与工程.md` | 方法与工程 | 架构图；UQEstimator；伪标签 v1→v2→v3（含 19.4% 重标）；FiLM L1/L2；损失；**工作量一页**；Stage 0–4 时间线 | ✅ 初稿 |
| 3 | `ch03-UQ结果.md` | UQ 感知监测结果 | AUROC、分离度、fig1/2/4；明确「≠ L2 预测器」；伪标签自洽性 3 点 | ✅ 初稿 |
| 4 | `ch04-FiLM与回放.md` | FiLM 与回放评估 | A/B/C/D；L2-FiLM vs 碰撞感知表；50 场景回放表；保守捷径（Brake/Throttle）；**18 场景 GIF 故事线表** | 待写 |
| 5 | `ch05-BEV可视化.md` | BEV 可视化与空间不确定性 | **本章重点**：见下节「BEV 素材清单」 | ✅ 初稿 |
| 6 | `ch06-诊断与Phase2.md` | 诊断与后续方案 | 六张决策卡；Normal ADE 根因；Score-Gate；BEV cost / λ；7 组消融设计；Round-2 / 资源 | 待写 |
| 7 | `ch07-总结.md` | 总结与可信度 | 已完成 / 进行中 / 待验证；可信度表（高/中/低）；组内可讨论问题 | 待写 |

**原附录内容并入位置（不再单独成章）**

| 原附录主题 | 并入章节 |
|------------|----------|
| Weather ID 全表 | ch01（正文简表 + 边界说明） |
| 伪标签 v1→v3 公式与 AUROC 轨迹 | ch02 + ch03 |
| 评估协议（开环 / 回放） | ch01 末 + ch04 开篇 |
| Phase −1 可行性 Q1–Q5 | ch05 + ch06 |
| 图表索引 | 各章内「本节图表」小节 |

---

## BEV 可视化：仓库素材清单（写入 ch05）

组内汇报建议 **先讲清三类 BEV 产物分别回答什么问题**，再贴图。

### 类型 A — IPM 感知不确定性热力图（不依赖 ORION 权重）

**做什么**：从 B2D 样例场景的原始相机图，用 patch 图像质量 + 逆透视映射（IPM）生成 256×256 BEV 不确定性网格；无需 attention、无需 36GB checkpoint。

**代码**

| 模块 | 路径 |
|------|------|
| 核心实现 | `uq_estimator/bev_uncertainty.py`（`compute_patch_quality`、`compute_bev_uncertainty_ipm`、`render_bev_heatmap`、`make_b2d_calibration`） |
| 评估脚本 | `scripts/eval_bev_noattn.py` |
| 样例数据 | `scripts/download_b2d_sample.py` → `data/b2d_sample/`（Normal W3 / Adverse W13） |
| 单元测试 | `tests/test_bev_uncertainty.py` |

**产出目录** `results/bev_noattn/`（跑完 `eval_bev_noattn.py` 后生成）

| 文件 | 内容 | 汇报用途 |
|------|------|----------|
| `panel_normal_w3.png` | 6 相机 + BEV 热力图（CloudySunset） | 正常天气空间分布 |
| `panel_adverse_w13.png` | 6 相机 + BEV 热力图（HardRainNight） | 恶劣夜晚对比 |
| `mean_bev_maps.png` | 两条件平均 BEV 图 | 一眼看全局差异 |
| `comparison.png` | 定量对比（含 Δ 标注） | **主图**：Normal vs Adverse |
| `score_boxplot.png` | 覆盖区域不确定性分布 | 分布无重叠证据 |
| `report.txt` | 数值摘要 | 填 ch05 表格 |

**关键数字（写入 ch05，填稿时核对 `report.txt`）**

| 指标 | Normal (W3) | Adverse (W13) | 备注 |
|------|-------------|---------------|------|
| 覆盖像素均值 BEV 不确定性 | ≈ 0.583 | ≈ 0.722 | 文档记载 Δ≈+0.139 |
| 原始 patch 质量均值 | ≈ 68.97 | ≈ 31.61 | 约 2.2× 差异 |
| 归一化 | log-scale 全局 | — | 线性归一化会使 Δ 退化到 ≈+0.011 |

**设计文档（写 ch05/ch06 时摘录要点，不链外链）**

- `docs/design_uncertainty_spatial.md` §六「IPM 方案验证结果」
- `docs/plan_v2.md` rev.3：IPM 升为主方案，Attention-based 降为消融

---

### 类型 B — 轨迹对比 GIF（相机 + BEV 小窗）

**做什么**：18 个 B2D 场景逐帧对比 GT / Baseline / FiLM 轨迹；画面含前置相机 + 右下角 BEV 俯视图（自适应范围）。

**代码**

| 脚本 | 路径 |
|------|------|
| 推理 + 渲染 | `scripts/generate_trajectory_gifs.py` |
| 缓存 | `results/gifs/trajectory_data.pt` |
| 离线重渲染 | `--render-only` 模式 |

**产出**

| 路径 | 数量 | 说明 |
|------|------|------|
| `results/gifs/*.gif` | 18 | 全画幅（相机 + BEV inset + UQ/L2 字幕） |
| 场景分组 | 4 条故事线 | S1 碰撞改善 / S2 高 UQ 危险 / S3 晴 vs 恶劣 / S4 复杂交互（见 REPORT §5.2 表） |

**汇报建议**：ch04 放 2–3 个代表 GIF 路径 + 1 张故事线汇总表；ch05 说明「BEV 小窗是轨迹行为的空间上下文，与类型 A 的 IPM 热力图互补」。

---

### 类型 C — 纯 BEV 轨迹 GIF（轻量）

**做什么**：仅从 `trajectory_data.pt` 离线渲染 BEV 俯视动画，无需 GPU。

| 脚本 | `scripts/render_bev_gifs.py` |
| 产出 | `results/gifs/bev_only/*.gif`（18 个，约 20MB） |
| 注意 | 缓存中 baseline 曾受 init_weights bug 影响，图例标为 Baseline*（虚线） |

**汇报建议**：若全画幅 GIF 太大，组内播放 **bev_only** 版本；在 ch05 用一句话说明 baseline 对照已用修复后 50 场景 JSON 重做。

---

### 类型 D — 仪表盘与开环图表（含 BEV 快照）

| 文件 | 路径 | 用途 |
|------|------|------|
| BEV 可行性快照 | `results/round2_dashboard_test/fig_bev_feasibility_snapshot.png` | 一页汇总 IPM Δ + Q1–Q5 |
| 安全-效率权衡 | `fig_safety_efficiency_tradeoff.png` | ch04/ ch07 |
| 保守捷径证据 | `fig_conservative_shortcut_evidence.png` | ch04 |
| 总览 | `fig_current_best_overview.png` | ch00 或 ch07 |

**标准开环图**（UQ 章节也可引用）：`results/figures/baseline/fig1`–`fig5`（pdf/png）

---

### ch05 建议小节结构

```text
5.1 为什么需要空间化（全局 FiLM 的局限，1 段）
5.2 IPM BEV 不确定性（方法 1 段 + comparison/mean_bev_maps 图 + 数字表）
5.3 样例面板解读（panel_normal vs panel_adverse，各 1 图）
5.4 轨迹 GIF 中的 BEV（类型 B/C，故事线表 + 2 个代表场景）
5.5 与 Phase 2 的衔接（BEV cost、λ、k-NN；Flash Attn 对 IPM 主链路非阻塞）
5.6 局限（2 场景、未接入 forward；待扩样与 7 组消融）
```

---

## 工作量数字（写入 ch02，审核后核实）

| 类别 | 数字 |
|------|------|
| 开环 | 12,806 帧；23 天气；Normal 2,709 / Adverse 10,097 |
| 回放 | 50 场景（11 Normal + 39 Adverse） |
| 特征 | ~235 GB patch tokens |
| 迭代 | 伪标签 v1/v2/v3；UQ 训练 v1/v2；FiLM checkpoint ×5 |
| 消融设计 | FiLM A–D；UQ yaml ×5；Phase2 规划 A–G |
| 可视化 | 开环 fig1–9（9 类）；baseline 已生成 5 张；dashboard 4 张；GIF 18；**BEV IPM 图 6 张**；bev_only GIF 18 |
| 工程 | 脚本 19；测试 115 passed；新增参数 4.45M；ORION `[UQ]` 补丁约 25 处 |
| 文档 | 项目内 REPORT / plan / design / progress_report 等（本报告独立存放在 `0521report/`） |

---

## 关键决策卡（写入 ch06，每张 4 行）

| 卡 | 决策 | 理由摘要 |
|----|------|----------|
| D1 | 伪标签 v1→v3 | entropy 无效 → Cohen's d 重加权 → weather 对齐（19.4% 重标） |
| D2 | 标签空间一致 | v2 AUROC 0.993 在 eval 空间仅 0.601 |
| D3 | FiLM identity init | 冻结主干；init_weights 须跳过 FiLM |
| D4 | 不用纯 L2 训 FiLM | L2-FiLM 恶劣 Col 升；需碰撞 loss |
| D5 | UQ ≠ 规划误差 | AUROC 高、Spearman(UQ,L2) 弱 |
| D6 | 全局 FiLM → 空间 BEV | Col↓ 但 ADE↑；IPM Δ≈+0.14 支持空间化 |

---

## 负结果 → 诊断（写入 ch06/ch07）

| 现象 | 诊断产出 | Phase 2 |
|------|----------|---------|
| Normal ADE +116% | 全局调制误伤正常场景 | Score-Gate 重训 |
| v2 AUROC 虚高 | 标签空间不一致 | v3 已修正 |
| L2-FiLM Col↑ | L2≠安全 | 碰撞感知训练 |
| 回放 Col↓ | 快速迭代证据，非部署级 | 官方 CARLA 对齐 |
| BEV 仅 2 场景 | 空间信号存在，待扩样 | 扩场景 + D/F 消融 |

---

## 图表放入本目录的约定

写作时执行：

```bash
# 示例：复制到 0521report/assets/ 后在正文中写
# ![](../assets/bev_comparison.png)  或 LaTeX \includegraphics
cp results/bev_noattn/comparison.png 0521report/assets/
cp results/round2_dashboard_test/fig_bev_feasibility_snapshot.png 0521report/assets/
```

| 章节 | 建议复制的资产 |
|------|----------------|
| ch01 | （可选）架构简图手绘 |
| ch03 | fig1, fig2, fig4 |
| ch04 | fig5；dashboard 保守捷径图；1–2 个 gif 缩略或静态首帧 |
| ch05 | comparison, mean_bev_maps, panel_normal, panel_adverse, fig_bev_feasibility_snapshot |
| ch07 | fig_current_best_overview；可信度表 |

---

## 写作进度

| 章节 | 状态 |
|------|------|
| index.md | v0.2 已确认 |
| ch01, ch02, ch05 | ✅ 初稿 |
| ch03 | ✅ 初稿 |
| ch04, ch06, ch07, ch00 | 未建 |
| assets/ | 空目录（待复制图表） |

---

## 建议完善顺序（审核通过后）

1. ch01 — 定调（悖论 + 评估边界）  
2. ch02 — 工作量 + 方法骨架  
3. ch05 — **BEV 可视化**（组内差异化亮点）  
4. ch03 + ch04 — 数字结果  
5. ch06 — 决策与 Phase 2  
6. ch07 + ch00 — 总结与摘要  

---

*大纲 v0.2 · 2026-05-18 · 分支 `report/framework` · 变更：去掉附录与外链；新增 ch05 BEV 专章及仓库 BEV 素材清单*

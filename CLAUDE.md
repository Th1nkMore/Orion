# UQ-ORION 项目上下文

## 项目背景
基于小米 ORION 框架的 uncertainty-aware 自动驾驶安全扩展。
目标：在极端场景（低能见度、雨雪雾）下通过不确定性感知提升安全性。

## 核心架构
ORION 原有流程：Vision Encoder → QT-Former → VLM → planning token → VAE → 轨迹
本项目扩展：新增 UQ Estimator 分支，输出 uncertainty embedding 和 score，
            注入 QT-Former 并通过 uncertainty token 调制 VAE 输出

### UQEstimator 模型结构（已实现）
```
patch_tokens [B, N_views, N_patches, 1024]
  → patch_proj Linear(1024→256)       # 降维，控制参数量
  → 2层 TransformerDecoder            # 16个 learnable query cross-attend patches
  → mean pool queries → mean pool views → [B, 256]
                                            ↓
stat_features [B, 5]                        │
  → Linear(5→64) + GELU + LayerNorm        │
  → [B, 64]                                │
                                            ↓
                              concat → [B, 320]
                              → Linear(320→512) + GELU + Dropout(0.1)
                              → Linear(512→256) → uncertainty_embedding
                                            ↓
                              ├─ score_head: Linear(256→1) + Sigmoid → [B, 1]
                              └─ embed_head: Linear(256→256) + LayerNorm → [B, 256]
```
- 总参数量：2.24M（上限 5M）
- 输出封装为 `UQOutput` dataclass（embedding, score, attn_weights）

### 损失函数（已实现）
- `UQRegressionLoss`：MSE + calibration 正则（惩罚 score 标准差过小，lambda_cal=0.1）
- `UQRankingLoss`：pairwise margin ranking loss（margin=0.1）
- `CombinedUQLoss`：total = regression + calibration + 0.5 * ranking
  - forward 返回 dict：{'total', 'regression', 'ranking', 'calibration'}

### 统计特征（dataset 中计算，5 维原始特征）
- 各视角 patch token 激活值方差均值（1 维）
- 跨 patch softmax 熵均值（归一化到 [0,1]）（1 维）
- 跨视角 token 余弦相似度矩阵均值（1 维）
- token 激活值绝对均值（1 维）
- token 激活值最大值均值（1 维）
- 模型内部 Linear(5→64) 投影到 d_stat

## Tensor 维度约定（严格遵守）
- patch_tokens: [B, N_views, N_patches, D]，D=1024（EVAViT embed_dim=1024）
- N_patches=1600（640/16 × 640/16 = 40×40），N_views=6
- uncertainty_embedding: [B, 256]
- uncertainty_score: [B, 1]，值域 [0, 1]
- stat_features（原始）: [B, 5]
- stat_features（投影后）: [B, 64]
- FiLM gamma/beta: [B, 1, 256]（broadcast over num_query dim）
- QT-Former query（FiLM 输入）: [num_query, B, 256]

## 项目结构
```
uq-orion/
├── adzoo/                  # ORION 原始代码（2 个文件有 [UQ] 标记的修改）
├── team_code/              # ORION 原始代码
├── mmcv/                   # ORION 依赖（3 个文件有 [UQ] 标记的修改，另有 1 个 NumPy 兼容修复）
├── uq_estimator/           # UQ 扩展模块（所有新增代码在此）
│   ├── __init__.py         # 导出：UQEstimator, UQOutput, CombinedUQLoss, UQFeatureDataset
│   ├── model.py            # UQEstimator 模型 + smoke test（支持消融开关）
│   ├── losses.py           # 损失函数
│   └── dataset.py          # UQFeatureDataset + compute_stat_features
├── scripts/
│   ├── extract_orion_features.py  # Stage 0: 从 ORION 提取 patch tokens
│   ├── generate_labels.py         # Stage 1a: 生成不确定性伪标签
│   ├── train_uq.py                # Stage 1b: 训练 UQEstimator（支持消融配置）
│   ├── validate_uq.py            # Stage 1c: 验证报告 + 可视化
│   ├── eval_openloop.py           # Stage 2a: 开环评估 + UQ score 分析
│   ├── train_film.py             # Stage 2b/4b: FiLM 微调 + 碰撞感知 loss（方案 C）
│   ├── eval_ablation_full.py     # Stage 4: 热交换 ablation 评估（A/B/C/D 四组）
│   ├── eval_closedloop_replay.py # 闭环回放评估 + 场景类型汇总
│   ├── generate_trajectory_gifs.py # 轨迹对比 GIF（Baseline vs FiLM vs GT，18 场景）
│   ├── merge_v2_uq_scores.py     # UQ score 合并到已有 eval 结果
│   ├── visualize_eval.py         # 论文图表生成（9 种图 + 文本摘要）
│   ├── visualize_attention.py    # QT-Former 注意力可视化 + FiLM 对比
│   ├── visualize_trajectory.py   # BEV 轨迹对比可视化
│   ├── render_bev_gifs.py        # BEV-only 离线 GIF 渲染（从 trajectory_data.pt 缓存）
│   ├── benchmark_overhead.py     # 计算开销测量（UQEstimator + FiLM 延迟）
│   ├── run_ablation.sh           # 编排脚本：训练 + 评估 + 可视化
│   └── e2e_mock_test.py          # 端到端 mock 测试
├── configs/
│   ├── uq_train.yaml              # 模型/训练/数据配置（含消融开关）
│   ├── uq_ablation_no_stat.yaml   # 消融：去掉统计特征
│   ├── uq_ablation_no_decoder.yaml # 消融：去掉 Transformer decoder
│   ├── uq_ablation_no_ranking.yaml # 消融：去掉排序损失
│   └── uq_ablation_no_cal.yaml    # 消融：去掉校准正则
├── tests/
│   ├── __init__.py
│   ├── fixtures.py          # mock 数据生成器（normal/adverse/random 样本）
│   ├── test_uq_model.py    # UQEstimator 模型/损失/数据集测试
│   ├── test_film.py         # FiLM 初始化/梯度/checkpoint/freeze 测试
│   ├── test_training.py     # 训练脚本 smoke test
│   └── test_generate_labels.py  # 标签生成验证
├── checkpoints/uq/best.pt  # v3 UQEstimator 权重（weather-based scene_type，已备份，gitignored）
├── checkpoints/uq/best_v2.pt  # v2 备份（scenario-based scene_type，已备份，gitignored）
├── checkpoints/film/        # FiLM 训练权重（已提交到 git）
│   ├── best_l1l2_col_v3.pt  # v3 FiLM L1+L2+collision（当前最佳）
│   ├── best_l1l2_col.pt     # FiLM L1+L2+collision（旧版）
│   ├── best_l1.pt, best_l2.pt, best_l1l2.pt  # 旧版 FiLM
├── results/
│   ├── eval_openloop_v3.json      # v3 开环评估（AUROC=0.954）
│   ├── eval_openloop_v3.pt        # v3 开环评估 PyTorch 格式（已备份，gitignored）
│   ├── closedloop_replay_v3.json  # v3 闭环评估（50场景，Col=0.52%）
│   ├── eval_openloop_full.pt      # 原始开环评估（v1 UQ score，已备份，gitignored）
│   ├── eval_openloop_full_summary.json  # 原始开环评估摘要
│   └── gifs/                      # 轨迹对比 GIF（18 场景 ~250MB）
│       ├── trajectory_data.pt     # 缓存轨迹数据（离线重渲染用）
│       ├── *.gif                  # 18 个场景 GIF 文件
│       └── bev_only/              # 18 个 BEV-only GIF（轻量，~20MB）
├── requirements.txt         # ORION 原始依赖（勿动）
├── requirements_uq.txt      # UQ 项目依赖（uv 管理）
└── .gitignore
```

## 实验矩阵

### FiLM 消融实验（eval_ablation_full.py / run_ablation.sh）
| 组 | 设置 | FiLM L1 | FiLM L2 | Checkpoint |
|----|------|---------|---------|------------|
| A | Baseline | ✗ | ✗ | identity |
| B | L1 only | ✓ | ✗ | best_l1.pt |
| C | L2 only | ✗ | ✓ | best_l2.pt |
| D | L1+L2 | ✓ | ✓ | best_l1l2.pt |

指标：L2@1s/2s/3s, Col@1s/2s/3s, UQ score，分 all/normal/adverse 三组

### UQ 组件消融实验（train_uq.py + 消融 config）
| 消融 | 配置文件 | 说明 |
|------|---------|------|
| Full model | uq_train.yaml | 完整 UQEstimator |
| w/o stat_features | uq_ablation_no_stat.yaml | 去掉 5 维统计特征 |
| w/o decoder | uq_ablation_no_decoder.yaml | mean pool 替代 Transformer decoder |
| w/o ranking loss | uq_ablation_no_ranking.yaml | 仅 MSE + calibration |
| w/o calibration | uq_ablation_no_cal.yaml | 仅 MSE + ranking |

指标：val_loss, Spearman, separation（validate_uq.py 验证）

### 可视化产出目录（visualize_eval.py）
| 编号 | 文件名 | 内容 |
|------|--------|------|
| fig1 | fig1_score_dist | UQ 分数分布直方图（Normal vs Adverse） |
| fig2 | fig2_auroc | ROC 曲线（UQ → 劣天气检测 AUROC） |
| fig3 | fig3_uq_vs_l2 | UQ score vs L2 error 散点图 + Spearman |
| fig4 | fig4_weather_boxplot | 按天气类型分组的 UQ 箱线图 |
| fig5 | fig5_planning_bars | Normal/Adverse 规划指标对比柱状图 |
| fig6 | fig6_comparison | Baseline vs FiLM 对比（需 --input-film） |
| fig7 | fig7_reliability | 校准曲线（UQ bin → 实际 L2/碰撞率） |
| fig8 | fig8_scenario_breakdown | 按场景类型拆解（需 --closedloop-json） |
| fig9 | fig9_uq_temporal | UQ score 时序演化（选取高方差场景） |

### 其他可视化
- visualize_attention.py: 注意力热力图、空间分布、熵分析、FiLM 对比
- visualize_trajectory.py: BEV 轨迹对比（GT/Baseline/FiLM）+ UQ 分层分析
- generate_trajectory_gifs.py: 18 场景逐帧 GIF（前置摄像头+BEV inset，GT/Baseline/FiLM 三线对比）
- benchmark_overhead.py: 计算开销报告（参数量、延迟、吞吐）

## ORION 文件修改清单
所有修改均以 `[UQ]` 注释标记（共 25 处），可通过 `grep -r "\[UQ\]" adzoo/ mmcv/` 查找。
- `adzoo/orion/configs/orion_stage3_infer.py` (+3行): use_uncertainty, uq_checkpoint 配置
- `adzoo/orion/test.py` (+26行): UQ checkpoint 和 FiLM 权重重新加载
- `mmcv/models/dense_heads/orion_head.py` (+23行): UQEstimator 初始化 + forward 中计算 uncertainty_emb
- `mmcv/models/utils/petr_transformers.py` (+16行): FiLM 调制层 + identity 初始化 + init_weights 保护
- `mmcv/models/detectors/orion.py` (+35行): FiLM L2 层定义 + identity init + train/inference 调制
- `mmcv/datasets/pipelines/loading.py` (+3行): NumPy 2.0+ 兼容性修复（np.int64/np.bool_）

## 已知问题与待修复
- **FiLM embed_head LayerNorm 问题**：embed_head 末尾的 LayerNorm 使所有样本 embedding norm 恒定（≈√256），
  导致 FiLM 对 Normal 场景（UQ score≈0）也产生非平凡调制。Normal ADE 从 2.78m→6.00m（+116%）。
  **解决方案**：Score-Gated FiLM（`gamma = 1 + score*(gamma_raw-1)`, `beta = score*beta_raw`），
  需改 petr_transformers.py + orion.py 各 ~3 行 + 重训 FiLM。详见 REPORT.md。
- **闭环 baseline 数据**：修复 init_weights bug 前的 `closedloop_baseline.json`（10 场景）不可信。
  修复后的 `closedloop_baseline_50.json`（50 场景）是正确的 baseline。

## 环境管理
- **禁止在 base conda 环境中安装任何依赖**
- 使用 uv 管理的 venv，位于项目根目录 `.venv/`
- 激活方式：`source .venv/bin/activate`
- Python 3.11.5，torch 2.1.0
- 安装依赖：`uv pip install -r requirements_uq.txt`
- 运行测试：`pytest tests/test_uq_model.py -v`
- 运行 smoke test：`python uq_estimator/model.py`

## 代码规范
- 所有新增代码放在 uq_estimator/ 目录下
- 对 ORION 原文件的修改，commit message 必须以 [UQ] 开头
- 每个函数必须有 shape 注释，格式：# [B, N, D]
- 不允许在模型代码里出现硬编码数字，全部从 config 读取
- mock 数据统一用：B=2, N_views=6, N_patches=256, D=1152（注：实际 EVAViT D=1024，mock 用 1152 测试兼容性）

## 不要做的事
- 不要修改 adzoo/ 目录下的任何文件（ORION 原始代码），除非明确要求
- 不要引入除 requirements_uq.txt 之外的新依赖，除非得到同意
- 不要删除或重命名任何 ORION 原有文件
- 不要往 base conda 环境装任何东西

## 测试规范
- 每个新模块都需要对应 pytest 测试
- smoke test 放在文件末尾的 __main__ 块里
- 所有测试用 mock 数据，不依赖真实数据集

## Git 规范
- 开发分支：dev
- 主分支：main
- UQ 新增文件的 commit 用常规前缀（feat/fix/refactor 等）
- 修改 ORION 原文件的 commit 必须以 [UQ] 开头

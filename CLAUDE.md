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
patch_tokens [B, N_views, N_patches, 1152]
  → patch_proj Linear(1152→256)       # 降维，控制参数量
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
- 图像梯度幅值的均值和方差（2 维）
- patch token 激活值的均值和方差（2 维）
- 跨视角 token 余弦相似度矩阵均值（1 维）
- 模型内部 Linear(5→64) 投影到 d_stat

## Tensor 维度约定（严格遵守）
- patch_tokens: [B, N_views, N_patches, D]，D=1024（EVAViT output，注意：非 1152）
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
├── adzoo/                  # ORION 原始代码（4 个文件有 [UQ] 标记的修改）
├── team_code/              # ORION 原始代码
├── mmcv/                   # ORION 依赖（2 个文件有 [UQ] 标记的修改）
├── uq_estimator/           # UQ 扩展模块（所有新增代码在此）
│   ├── __init__.py         # 导出：UQEstimator, UQOutput, CombinedUQLoss, UQFeatureDataset
│   ├── model.py            # UQEstimator 模型 + smoke test
│   ├── losses.py           # 损失函数
│   └── dataset.py          # UQFeatureDataset + compute_stat_features
├── scripts/
│   ├── extract_orion_features.py  # Stage 0: 从 ORION 提取 patch tokens
│   ├── generate_labels.py         # Stage 1a: 生成不确定性伪标签
│   ├── train_uq.py                # Stage 1b: 训练 UQEstimator
│   ├── validate_uq.py            # Stage 1c: 验证报告 + 可视化
│   ├── eval_openloop.py           # Stage 2a: 开环评估 + UQ score 分析
│   ├── train_film.py             # Stage 2b/4b: FiLM 微调 + 碰撞感知 loss（方案 C）
│   ├── eval_ablation_full.py     # Stage 4: 热交换 ablation 评估（A/B/C/D 四组）
│   ├── eval_closedloop_replay.py # 闭环回放评估（Bench2Drive 数据，无需 CARLA）
│   └── e2e_mock_test.py          # 端到端 mock 测试
├── configs/
│   └── uq_train.yaml       # 模型/训练/数据配置
├── tests/
│   ├── __init__.py
│   └── test_uq_model.py    # pytest 测试
├── checkpoints/uq/best.pt  # 已训练的 UQEstimator 权重
├── checkpoints/film/        # FiLM 训练权重（best_l1.pt, best_l2.pt, best_l1l2.pt）
├── results/                 # 评估结果输出目录
├── requirements.txt         # ORION 原始依赖（勿动）
├── requirements_uq.txt      # UQ 项目依赖（uv 管理）
└── .gitignore
```

## ORION 文件修改清单
所有修改均以 `[UQ]` 注释标记，可通过 `grep -r "\[UQ\]" adzoo/ mmcv/` 查找。
- `adzoo/orion/configs/orion_stage3_infer.py` (+3行): use_uncertainty, uq_checkpoint 配置
- `adzoo/orion/test.py` (+26行): UQ checkpoint 和 FiLM 权重重新加载
- `mmcv/models/dense_heads/orion_head.py` (+23行): UQEstimator 初始化 + forward 中计算 uncertainty_emb
- `mmcv/models/utils/petr_transformers.py` (+16行): FiLM 调制层 + identity 初始化

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
- mock 数据统一用：B=2, N_views=6, N_patches=256, D=1152

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

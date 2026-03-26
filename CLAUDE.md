# UQ-ORION 项目上下文

## 项目背景
基于小米 ORION 框架的 uncertainty-aware 自动驾驶安全扩展。
目标：在极端场景（低能见度、雨雪雾）下通过不确定性感知提升安全性。

## 核心架构
ORION 原有流程：Vision Encoder → QT-Former → VLM → planning token → VAE → 轨迹
本项目扩展：新增 UQ Estimator 分支，输出 uncertainty embedding 和 score，
            注入 QT-Former 并通过 uncertainty token 调制 VAE 输出

## Tensor 维度约定（严格遵守）
- patch_tokens: [B, N_views, N_patches, D]，D=1152
- uncertainty_embedding: [B, 256]
- uncertainty_score: [B, 1]，值域 [0, 1]
- stat_features: [B, 64]

## 代码规范
- 所有新增代码放在 uq_estimator/ 目录下
- 对 ORION 原文件的修改，commit message 必须以 [UQ] 开头
- 每个函数必须有 shape 注释，格式：# [B, N, D]
- 不允许在模型代码里出现硬编码数字，全部从 config 读取
- mock 数据统一用：B=2, N_views=6, N_patches=256, D=1152

## 不要做的事
- 不要修改 adzoo/ 目录下的任何文件（ORION 原始代码），除非我明确要求
- 不要引入除 requirements.txt 之外的新依赖，除非我同意
- 不要删除或重命名任何 ORION 原有文件

## 测试规范
- 每个新模块都需要对应 pytest 测试
- smoke test 放在文件末尾的 __main__ 块里
- 所有测试用 mock 数据，不依赖真实数据集

# ORION Spatial UQ / Adapter 代码审计（2026-08-26）

## 审计边界

本报告只审计当前工作树，不修改现有模型、训练或闭环代码。当前工作树已有大量用户改动；尤其是下列待改造热点本身处于 dirty/untracked 状态，后续实现必须逐文件保留并重放这些改动，不能用上游版本覆盖：

- `/Users/th1nkmore/th1nkmore_ws/Orion/mmcv/models/detectors/orion.py`
- `/Users/th1nkmore/th1nkmore_ws/Orion/scripts/train_uq_token.py`
- `/Users/th1nkmore/th1nkmore_ws/Orion/team_code/orion_b2d_agent.py`
- `/Users/th1nkmore/th1nkmore_ws/Orion/uq_estimator/corruptions.py`
- `/Users/th1nkmore/th1nkmore_ws/Orion/uq_estimator/__init__.py`
- 未跟踪的 `/Users/th1nkmore/th1nkmore_ws/Orion/adzoo/orion/configs/orion_stage3_agent_uq.py`

本机只有 Density checkpoint，缺少用于重新训练的 descriptor cache 和 B2D annotation：

- 存在：`/Users/th1nkmore/th1nkmore_ws/Orion/checkpoints/density_uq/best.pt`（936 KB）
- 缺少：`/Users/th1nkmore/th1nkmore_ws/Orion/data/density_uq/descriptors.pt`
- 缺少：`/Users/th1nkmore/th1nkmore_ws/Orion/data/infos/b2d_infos_val.pkl`

因此本地可以完成代码/接口审计，但 spatial UQ 数据生成和训练仍应在服务器共享数据目录执行。

## 1. 当前主 UQ 信号的真实定义

### 1.1 Density UQ 是 sample-level、全局、多视角 OOD 分数

当前主信号位于 `/Users/th1nkmore/th1nkmore_ws/Orion/uq_estimator/density.py`：

1. `compute_view_moments` 对每个相机的所有 patch 分别求 mean/std，然后把相机、统计量、通道全部 flatten（14-31 行）。输入为 `[B, V, P, D]`，输出 `[B, V*2*D]`。
2. 按当前 ORION 配置，`V=6, P=40*40=1600, D=1024`，所以 descriptor 为 `[B, 12288]`。图像骨干形状来自 `/Users/th1nkmore/th1nkmore_ws/Orion/adzoo/orion/configs/orion_stage3_infer.py` 195-212 行及 `/Users/th1nkmore/th1nkmore_ws/Orion/mmcv/models/dense_heads/orion_head.py` 752-775 行。
3. descriptor 经 StandardScaler、16-D PCA、Ledoit-Wolf precision whitening，然后计算 whitened residual 的 L2 norm（`density.py` 98-133 行）。
4. score 是该距离在 **normal calibration distances** 中的经验分位数，形状 `[B,1]`（121-131 行）；active embedding 是 16-D 单位方向，输出 embedding 用零填充/投影成 256-D（118-132 行）。
5. `UQOutput` 的现有约定是 `embedding [B,256]`、`score [B,1]`、可选 `active_embedding [B,16]`，定义在 `/Users/th1nkmore/th1nkmore_ws/Orion/uq_estimator/model.py` 16-23 行。

Density 模型并不使用 adverse 标签训练。`/Users/th1nkmore/th1nkmore_ws/Orion/scripts/fit_density_uq.py` 104-150 行按完整 route 分 train/calibration/test，只用 normal train 拟合 scaler/PCA/covariance，只用 normal calibration 建 score CDF；adverse 只用于测试 AUROC/AUPRC。其物理含义因此是“冻结 EVAViT 全局特征偏离 normal distribution 的程度”，不是 object-level、location-level 或 path-relevant uncertainty。

现有离线结果 `/Users/th1nkmore/th1nkmore_ws/Orion/reports/density_uq/metrics.json` 为 12,806 帧、50 routes、AUROC 0.799（route-bootstrap 95% CI 0.675-0.915），说明它能作为中等强度的天气/OOD monitor，但不能证明空间或风险语义。

### 1.2 Density 数据来源

- `/Users/th1nkmore/th1nkmore_ws/Orion/scripts/extract_orion_features.py` 120-148 行从冻结 ORION EVAViT 保存每帧 `[6,1600,1024]` fp16 token；场景 normal/adverse 只按 Weather ID 划分（24-46、87-89 行）。
- `/Users/th1nkmore/th1nkmore_ws/Orion/scripts/cache_density_descriptors.py` 47-89 行把 token 压为 per-view mean/std descriptor，并从文件名记录 route、weather、scene type。
- `/Users/th1nkmore/th1nkmore_ws/Orion/configs/density_uq.yaml` 1-10 行固定 PCA=16、输出 embedding=256、route split=0.6/0.2/0.2。

关键后果：patch 空间轴在 descriptor 生成的第一步就被消掉，后续不可能从 `score` 或 `active_embedding` 唯一恢复“哪里不确定”。

## 2. 旧 learned UQEstimator（当前不是主信号）

`/Users/th1nkmore/th1nkmore_ws/Orion/uq_estimator/model.py` 26-141 行定义了旧神经 UQEstimator：

- 输入 `[B,6,1600,1024]` 和 5-D 手工统计量；
- 每个相机独立 cross-attention pooling，再跨相机平均（107-124 行）；
- 输出仍只有全局 `embedding [B,256]` 与 `score [B,1]`（134-141 行）。

训练数据 `/Users/th1nkmore/th1nkmore_ws/Orion/uq_estimator/dataset.py` 193-293 行来自预提取 token 和 `uq_labels.pt` 的每帧 scalar pseudo-label。损失 `/Users/th1nkmore/th1nkmore_ws/Orion/uq_estimator/losses.py` 是：

- MSE scalar regression；
- `1/std(pred)` 防塌缩项；
- batch 内 pairwise margin ranking。

这条旧路径同样会先跨 patch、再跨 view 聚合，不能直接扩展成可信 spatial UQ；并且 pseudo-label 本身来自启发式统计量。建议保留为历史 baseline，不在它上面继续叠空间语义。

注意配置注释漂移：`model.py` 35 行注释仍写 `d_patch=1152`，实际 `/Users/th1nkmore/th1nkmore_ws/Orion/configs/uq_train.yaml` 2 行及当前 EVAViT 均为 1024。

## 3. 当前 adapter / token 的接口和训练信号

### 3.1 Pre-LLM vision adapter

`/Users/th1nkmore/th1nkmore_ws/Orion/uq_estimator/vision_adapter.py` 9-48 行实现：

```text
vision_tokens [B,N,4096]
score         [B,1]
residual      = up(GELU(down(LN(vision_tokens))))
output        = vision_tokens + clamp(score,0,1) * residual
```

当前 N 通常为 513（object query 与 map query 拼接；`/Users/th1nkmore/th1nkmore_ws/Orion/mmcv/models/detectors/orion.py` 680-689 行）。adapter 的 `up` 零初始化，所以初始为 identity。虽然每个 visual token 的 residual 因输入 token 不同而不同，但所有 token 共享同一个 global scalar gate；它无法表达“只降低某个相机/区域证据的可信度”。

### 3.2 Explicit UQ token

`/Users/th1nkmore/th1nkmore_ws/Orion/uq_estimator/token_projector.py` 9-73 行将 `[B,16] active_embedding + [B,1] score` 投成默认一个 `[B,1,4096]` 连续 token，并 append 到 LLM visual sequence（`orion.py` 554-594 行）。它仍是全局 token，不带 camera、patch、BEV 或 path position。

`/Users/th1nkmore/th1nkmore_ws/Orion/uq_estimator/grounding.py` 10-31 行从 waypoint hidden state `[B,4096]` 回归 global density score；这只能验证 LLM hidden state 是否可线性读出 scalar，不能生成空间 UQ 可视化。

### 3.3 adapter 训练入口和实际 loss

主入口是 `/Users/th1nkmore/th1nkmore_ws/Orion/scripts/train_uq_token.py`：

- 使用 B2D **test-format val annotation**，默认 `b2d_infos_val.pkl`，batch=1（32-65、527-600 行）；
- 冻结 ORION，训练 projector/vision adapter、LLM LoRA、grounding head（642-697 行以及 `/Users/th1nkmore/th1nkmore_ws/Orion/uq_estimator/training.py` 13-52 行）；
- corruption 只有 blur、dark、camera_dropout 三类（`train_uq_token.py` 58-65 行；实现位于 `/Users/th1nkmore/th1nkmore_ws/Orion/uq_estimator/corruptions.py` 16-65 行）；
- 训练复用 `/Users/th1nkmore/th1nkmore_ws/Orion/scripts/train_film.py` 的 `forward_film_training`。

`forward_film_training` 的默认组合是（`train_film.py` 220-228、528-553 行）：

```text
L = lambda_plan * open-loop trajectory regression
  + lambda_vae  * VAE probabilistic loss
  + lambda_vlm  * language teacher-forcing loss
  + lambda_consistency * low-UQ hidden-feature consistency
  + lambda_ground * global score SmoothL1
  + lambda_col * GT-agent future collision-margin loss
```

其中 collision margin 用 GT 物体未来 6 步与预测 ego 轨迹的距离 hinge，并乘 global UQ score（`train_film.py` 110-139、539-553 行）。它不是 CARLA rollout collision loss，也没有直接惩罚 no-hazard 中不必要停车。

paired corruption 分支尤其需要重新解释：`/Users/th1nkmore/th1nkmore_ws/Orion/scripts/train_uq_token.py` 775-940 行先保存 clean planning feature，再让 corrupted+correct-UQ feature靠近 clean feature；counterfactual rank 进一步要求 correct 比 shuffled 更接近 clean（900-920 行）。这套目标本质上在训练“受损输入下恢复 clean/GT open-loop planning 表示”，因此 ADE 变好是损失直接优化的结果，而不是额外 UQ 信号自然提供了缺失视觉信息。它不适合作为未来“选择性保守”主监督。

## 4. 当前 B2D/ORION 数据结构中的可用监督

`/Users/th1nkmore/th1nkmore_ws/Orion/mmcv/datasets/b2d_orion_dataset.py`：

- 6 相机 RGB path、`lidar2img`、camera intrinsics：215-246 行；
- 当前 ego future 6 步 offsets、mask、command：275-282 行；
- 3D objects、ID、当前 boxes、future agent offsets：194-213、460-517 行；
- route folder/frame/timestamp，可用于 paired event 对齐：194-213 行。

inference pipeline 收集相同的 `lidar2img`、ego future、GT attr 等键，见 `/Users/th1nkmore/th1nkmore_ws/Orion/adzoo/orion/configs/orion_stage3_infer.py` 325-353 行。训练 pipeline 原本还含 photometric distortion 和 Chat-B2D QA，见 `/Users/th1nkmore/th1nkmore_ws/Orion/adzoo/orion/configs/orion_stage3_train.py` 370-390 行。

这些信息足以构造第一版：

- exact synthetic corruption mask / severity / onset / offset；
- clean-vs-corrupt patch feature discrepancy target；
- 3D object box 投影到各 camera 的 object-region target；
- GT ego future corridor 与 object future 的 path overlap/TTC target。

但当前 loader 并未加载 CARLA depth/semantic/instance image；如果第一阶段要用这些更强监督，需要新增 paired dataset/pipeline keys，不能假定现有 `B2DOrionDataset` 已经提供。

另一个必要修复是 camera order：闭环 agent 在 `/Users/th1nkmore/th1nkmore_ws/Orion/team_code/orion_b2d_agent.py` 39-47、647-653 行显式固定为 FRONT、FRONT_LEFT、FRONT_RIGHT、BACK、BACK_LEFT、BACK_RIGHT；离线 dataset 在 `b2d_orion_dataset.py` 222-235 行依赖 `info['sensors'].items()` 插入顺序。spatial label 与 heatmap 接入前必须显式统一并断言 camera name/order，否则空间监督可能静默错位。

## 5. 已有 BEV uncertainty 原型能复用什么

`/Users/th1nkmore/th1nkmore_ws/Orion/uq_estimator/bev_uncertainty.py` 有两条手工路径：

1. RGB Laplacian/Sobel/contrast patch quality `[B,V,P]`（33-114 行）；
2. attention-weighted query uncertainty `[B,Nq]`（120-143 行）或通过固定相机几何 IPM 得到 `[B,Hbev,Wbev]`（384-524 行）。

可复用的部分：camera geometry、trajectory sampling/cost 的函数边界、heatmap renderer。不能直接把它叫 learned spatial UQ，原因是 patch quality 是手工图像质量，且默认逐帧 min-max normalization 会抹去跨帧绝对严重度（108-113、507-520 行）。

现有集成还没有达到可运行主路径：

- `register_attn_hook` 要求 `flash_attn=False`（606-634 行），当前 ORION config 是 `flash_attn=True`；
- 全仓没有初始化 `self._attn_hook` 或 `self._patch_quality`，但 `orion.py` 1032-1045 行直接读取二者；
- 这段 mode-score adjustment 只在 diffusion decoder 分支中，当前用户明确暂缓 diffusion，现有 VAE 路径不会使用它；
- `compute_patch_quality`/IPM 更适合作为必须比较的 simple quality baseline，而不是主 learned UQ。

`/Users/th1nkmore/th1nkmore_ws/Orion/scripts/build_uq_localization_labels.py` 1-5、79-101 行已有“遮挡哪个 block 最改变 Density active embedding”的 coarse label。它衡量 density embedding sensitivity，不是感知错误或风险真值；可以保留为 probe，但不要作为第一阶段唯一 spatial target。

## 6. 闭环接口现状

闭环 agent `/Users/th1nkmore/th1nkmore_ws/Orion/team_code/orion_b2d_agent.py` 当前：

- 在 normalized multi-view input 上注入 corruption（728-737 行）；
- 从 `pts_bbox_head.uq_output.score` 读取单个 raw scalar（747-754 行）；
- risk governor 只消费 scalar 或已知 `corruption_active` oracle（755-763 行）；
- trace 只记录 raw score、corruption state、route progress 和 scalar governor decision（781-818、842-856 行）。

因此当前闭环已经能验证“oracle event -> 保守响应”，但没有 spatial map、path overlap、per-step path risk 的接口或日志。

## 7. 实现 spatial UQ head + path-risk aggregator 的最小代码改造清单

建议不改 diffusion、不删 Density UQ、不直接改现有 `UQVisionAdapter` 的签名；新增并行接口，先保证旧实验可复现。

### A. 新增输出契约（小改、向后兼容）

修改 `/Users/th1nkmore/th1nkmore_ws/Orion/uq_estimator/model.py` 的 `UQOutput`，只追加可选字段：

```text
spatial_logits: [B,V,Hf,Wf]       # 第一版为 [B,6,40,40]
spatial_score:  [B,V,Hf,Wf]       # sigmoid 后概率
bev_score:      [B,Hb,Wb] | None
path_risk:      [B,T] | None      # 建议 T=6，与 ORION future steps 一致
path_risk_max:  [B,1] | None
```

旧构造器全用 keyword 参数，追加默认 None 不会破坏 Density/legacy UQ consumer。不要把 `path_risk_max` 覆盖到原 `score`；global density score 应继续作为 OOD monitor，便于消融。

### B. 新增 spatial head（新文件，避免污染旧 density）

新增 `/Users/th1nkmore/th1nkmore_ws/Orion/uq_estimator/spatial_head.py`：

- 输入直接使用 `img_feats [B,6,1024,40,40]`，不要使用已经 flatten/mean 的 density descriptor；
- 最小 head：共享 `1x1 Conv 1024->256 + GELU + 3x3 Conv 256->64 + GELU + 1x1 Conv 64->1`；
- 输出 `[B,6,40,40]` logits；head 零/低 bias 初始化以控制 clean false positives；
- 可选输入 clean teacher feature discrepancy，但推理只需要 corrupt/current feature；
- 第一阶段冻结 EVAViT，只训练该 head。

新增 `/Users/th1nkmore/th1nkmore_ws/Orion/uq_estimator/spatial_losses.py`，至少拆开记录：

- `L_region`：mask/soft-error target 的 BCE/focal；
- `L_error`：预测 clean-vs-corrupt patch feature discrepancy（soft target）；
- `L_clean_fp`：clean map 稀疏/假阳性约束；
- `L_severity_rank`：同 patch、同场景的 severity 单调 ranking；
- `L_temporal`：event onset/offset 后的响应/恢复一致性；
- `L_cal`：held-out route 上的 calibration，不再用 `1/std` 伪校准项。

### C. 新增 path-risk aggregator（新文件）

新增 `/Users/th1nkmore/th1nkmore_ws/Orion/uq_estimator/path_risk.py`：

- 输入 `spatial_score [B,V,40,40]`、`path_xy [B,T,2]`、增强后的 `lidar2img [B,V,4,4]`；
- 将未来 route corridor/waypoints 投影到每个相机 feature grid；按 corridor 半径采样/softmax-pool spatial UQ；
- 输出 `path_risk [B,T]` 和 `path_risk_max [B,1]`；另输出 coverage mask，防止“投影不可见”被当成低风险；
- 训练时可用 GT ego future `[B,6,2]` 做 teacher path；闭环时不能用未知 GT，应由 RoutePlanner 提供固定长度 local route polyline，或由 base planner 的未调制候选轨迹 `stop_gradient` 后做第二次 risk refinement。

第一版更推荐 local route polyline，因为 adapter 需要在生成最终轨迹前获得 path relevance。`/Users/th1nkmore/th1nkmore_ws/Orion/team_code/planner.py` 41-130 行已经维护剩余 route deque，可以增加一个只读 `sample_local_route(K, spacing)`，在 agent 中转成 lidar frame后放入 batch。仅用当前 `command` scalar/one-hot 不足以定义空间路径。

### D. 模型接入点

修改 `/Users/th1nkmore/th1nkmore_ws/Orion/mmcv/models/dense_heads/orion_head.py`：

- `__init__` 增加 `use_spatial_uq`, `spatial_uq_checkpoint`, `path_risk_cfg`；
- 在 752-784 行、即 `x=[B,V,C,H,W]` 尚未 flatten/投影之前运行 spatial head；
- 保持当前 DensityUQ 计算不变，把两者都写入扩展后的 `self.uq_output`；
- forward 返回值先不要再增加 tuple 元素，避免破坏 `orion.py` 多处四元解包；通过 `self.uq_output` 读取新增字段最安全。

修改 `/Users/th1nkmore/th1nkmore_ws/Orion/mmcv/models/detectors/orion.py`：

- 新增 `SpatialUQVisionAdapter` 或 `PathRiskConditioner`，不要修改旧 `UQVisionAdapter(score)`；
- 最小行为接口用 `path_risk [B,6] + path coverage` 生成 1-2 个 `[B,4096]` condition tokens，或对与 path/object query 对齐的 visual tokens做 cross-attention；
- clean/no-path-risk 时严格 identity；
- VAE 保持原样，先只改变 LLM 前视觉证据或 waypoint hidden state；
- 不复用 1032-1045 行的 diffusion-only BEV 分支。

### E. 数据与 corruption API

新增 `/Users/th1nkmore/th1nkmore_ws/Orion/uq_estimator/paired_spatial_dataset.py` 或 B2D pipeline transform，返回：

```text
clean_img, corrupt_img
corruption_type, severity, start/end, camera_ids
corruption_mask [B,V,Hf,Wf]
path_xy [B,T,2], path_mask [B,T]
optional object_mask / perception_error_target
hazard_present, on_path
```

保留 `/Users/th1nkmore/th1nkmore_ws/Orion/uq_estimator/corruptions.py` 现有返回 tensor 的 API；新增 `corrupt_multiview_with_metadata`，不要直接改返回类型。第一批支持 blur、dark、glare、fog、local occlusion，并明确 on-path/off-path；full-camera black dropout只保留 sensor-failure/simple-detector baseline。

### F. 两阶段训练入口（新脚本，避免继续膨胀旧脚本）

1. 新增 `/Users/th1nkmore/th1nkmore_ws/Orion/scripts/train_spatial_uq.py`：冻结 ORION backbone，训练 spatial head + path-risk calibration；只做第一阶段 localization/calibration/monotonicity/latency，绝不使用 ADE 作为成功门槛。
2. 新增 `/Users/th1nkmore/th1nkmore_ws/Orion/scripts/train_spatial_uq_adapter.py`：冻结或小学习率更新 spatial head，先用 oracle spatial map 训练 adapter，再切 learned map；训练目标加入 hazard safety、no-hazard unnecessary intervention、progress、recovery，避免旧 `pair_loss=corrupt feature -> clean feature` 成为主目标。
3. 标准 ORION 全量训练入口 `/Users/th1nkmore/th1nkmore_ws/Orion/adzoo/orion/train.py` 196-233 行暂时不需要改；先通过独立脚本降低对原训练栈的侵入。

### G. 配置、测试、可视化

新增而非覆盖：

- `/Users/th1nkmore/th1nkmore_ws/Orion/configs/spatial_uq_stage1.yaml`
- `/Users/th1nkmore/th1nkmore_ws/Orion/configs/spatial_uq_adapter_stage2.yaml`
- `/Users/th1nkmore/th1nkmore_ws/Orion/tests/test_spatial_uq_head.py`
- `/Users/th1nkmore/th1nkmore_ws/Orion/tests/test_path_risk.py`
- `/Users/th1nkmore/th1nkmore_ws/Orion/tests/test_paired_spatial_dataset.py`
- `/Users/th1nkmore/th1nkmore_ws/Orion/scripts/render_spatial_uq.py`

必须测试：shape、全 clean identity、on-path 高 UQ > off-path 高 UQ、projection coverage、camera-order invariance/error、severity monotonic pair、event recovery、checkpoint 向后兼容。闭环 trace 在现有 scalar 字段之外追加 heatmap artifact path、`path_risk[6]`、`path_risk_max`、coverage、intervention reason。

## 8. 预计冲突与实现顺序

高冲突文件：

1. `team_code/orion_b2d_agent.py`：当前已有约 380 行未提交闭环改动；只做小型追加，不重排 setup/run_step。
2. `mmcv/models/detectors/orion.py`：已有 inference mode/conditioning 未提交改动；新增 spatial conditioner 应为独立 method/flag。
3. `scripts/train_uq_token.py`：已有 route-balanced 与 clean-preservation 未提交改动；建议新建 stage1/stage2 脚本，避免冲突。
4. `uq_estimator/corruptions.py`：已有 view selection 未提交改动；新增 metadata API，保留原函数契约和测试。

低冲突接入顺序：

1. 先写新 spatial head、path-risk、loss、dataset wrapper 与单元测试；
2. 扩展 `UQOutput` 可选字段并跑现有 UQ tests；
3. 只在 `orion_head.py` 加 feature-level hook，确认旧 Density 输出逐 bit/容差不变；
4. 新训练脚本完成 stage1 离线 gate；
5. 再在 `orion.py` 接 spatial adapter；
6. 最后小范围修改 agent，传 local route corridor并记录 heatmap/risk。

## 9. 最小结论

现有系统已经具备三个可复用部件：冻结 EVAViT feature、可靠的 paired closed-loop corruption schedule、以及 oracle risk-response 机制。但当前 Density UQ、UQ token、vision adapter、grounding head全是 global-scalar contract；已有 BEV 模块是手工质量原型且只挂在 diffusion 分支。要让“两阶段训练 + 可视化叙事”成立，最小必要变化不是继续调 global score，而是保留 Density 作为 baseline，新增 `[B,6,40,40]` learned spatial head、基于 local route corridor 的 `[B,6]` path-risk aggregator，以及独立的 stage1/stage2 训练入口。

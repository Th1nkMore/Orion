# Observation UQ v3：clean-conditional surprise 与独立退化验证

日期：2026-08-26
状态：`active bounded MVP`
替代的正式训练配置：`configs/spatial_uq_stage1.yaml` v2
当前配置：`configs/observation_uq_v3.yaml`

## 1. 时间戳 amendment

此前 v2 Stage-1 把 actual ORION failure target 作为优先监督，并把 paired
clean/corrupt representation error 作为可执行 fallback。该管线已经完成真实
ORION forward 和 target 对齐验证，但不再作为 observation uncertainty adapter
的正式主要监督。

原因如下：

1. actual target 混合了观测退化、ORION 模型能力和驾驶任务相关性；
2. paired representation error 仍依赖人为 corruption generator；
3. corruption mask、类型和 severity 不是 uncertainty ground truth；
4. 在相同 generator 上训练和测试，只能证明模型识别了 augmentation；
5. adapter 的新职责是通用空间观测不足，任务相关性由 ORION/VLM 学习。

因此 v2 保留为历史诊断和可选辅助监督，不删除结果，也不静默改写其原始设计。
`configs/spatial_uq_stage1.yaml` 已标记为
`superseded_diagnostic_history_only_2026-08-26`。在另行批准前，不运行 v2
actual-target 正式训练，也不扩展对应实验矩阵。

## 2. v3 的可主张对象

v3 学习的是：

> 在只用正常驾驶序列学到的条件特征分布下，当前空间观测相对其上下文有多大
> 条件 surprise；随后将该 surprise 蒸馏成单次前向的 observation-insufficiency
> score。

它暂时不能被称为：

- 唯一或真实 uncertainty ground truth；
- 语义正确的危险概率；
- ORION 的失败概率；
- 路径风险；
- LLM 已理解不确定性的证据；
- 闭环安全已经改善的证据。

## 3. 架构与梯度边界

### 3.1 CleanConditionalTeacher

输入为 frozen EVAViT 的六视角 patch grid，形状为 `[B,V,H,W,D]`。训练数据仅含
route-manifest 中 `train` split 的 clean 序列。

本段最初实现使用四相 2x2 single-patch lattice mask；该 v3.0 设计已被第 10 节的
v3.1 amendment 替代，保留在这里用于解释 job `1062956`。v3.0 中被预测 patch 的
当前特征会被替换为 learned masked token，但后来确认上一帧同位置与相邻退化 patch
仍会泄漏。上下文包括：

- 同一视角当前帧的空间邻域；
- 存在时的一帧历史特征；
- view embedding。

第一版尚未加入基于相机几何的 cross-view correspondence，这是明确的 MVP
限制，不会在结果中声称已实现。

Teacher target 是 withheld-patch cosine prediction error。两个独立初始化的
Teacher 还提供 prediction disagreement。最终原始 target 为：

`conditional residual + 0.25 * ensemble disagreement`

并以 clean train patch 的 P95 归一化。该数值是 operational score，不是概率。

### 3.2 ObservationUQAdapter

Adapter 接收当前帧、上一帧和 previous-valid flag，输出 `[B,V,H,W]` 非负分数。
其 forward API 不接收：

- corruption family；
- severity；
- corruption mask；
- actual ORION failure target。

Adapter 使用 Teacher score 蒸馏。高 surprise patch 通过只依赖 Teacher target 的
连续权重提高占比，避免少量异常 patch 被大面积正常背景淹没。该权重不读取
corruption mask。

Teacher 拟合完成后，compact `[V,H,W]` target 只预计算一次；后续 adapter epoch
不会重复跑 Teacher。目标缓存不保存或输入 synthetic label。

未来接入 ORION 时采用：

`L_driving(image, stop_gradient(UQ))`

驾驶损失不能把 adapter 反向训练成 task-risk detector。

## 4. Corruption protocol

Synthetic corruption 仅有三个用途：

1. 管线干预；
2. 给 student 提供有信息损失的观测输入；
3. 划分独立 family 并做后验诊断。

第一版 MVP：

| 用途 | family |
|---|---|
| adapter train | `local_blur`, `local_dark` |
| entire-family held-out | `local_glare` |
| diagnostic only | `local_occlusion`, `camera_dropout`, full-route blackout |

正式扩展前必须加入与训练图像函数不同的来源：CARLA 原生雾、雨、夜间/低太阳角度、
时间掉帧或 freeze。后续还应留出 generator、Town、route 和 severity range。

当前 `local_glare` 整 family 留出只能证明最弱的跨算子趋势；它仍来自同一套本地
corruption library，不能代替 native-engine 或真实数据外部验证。

## 5. 防泄漏与测试

新增测试覆盖：

- 四相 mask 恰好覆盖每个 patch 一次；
- Teacher 训练中出现任何非 `clean` example 会 fail closed；
- train/held-out family 重叠会 fail closed；
- 修改同一张量的 family、severity 和 mask，不改变 Teacher/adapter 输出；
- v3 转换旧 paired record 时不读取 actual target 字段；
- held-out family 不进入 student optimizer batch；
- bounded mock 的 Teacher 和 adapter loss 出现下降。

2026-08-26 本地证据：

`40 passed`：

- `tests/test_observation_uq_v3.py`
- `tests/test_paired_feature_extraction.py`
- `tests/test_spatial_training.py`
- `tests/test_spatial_uq.py`

远端 Python 3.8 / Torch 2.4：v3 与 paired extractor 的 `16 passed`。

## 6. 本地 mock 结果

固定 seed：`20260826`
train family：blur + dark
held-out family：glare
route split：8 train / 2 validation / 2 held-out
Teacher epochs：12
Adapter epochs：24

产物：

- `results/observation_uq_v3/mock_seed20260826.pt`
- `results/observation_uq_v3/mock_seed20260826.report.json`

主要数值：

| 指标 | 初始 | 最终 |
|---|---:|---:|
| Teacher train loss | 0.5081 | 0.00628 |
| Adapter train loss | 0.0991 | 0.00253 |

完全留出的 route + glare family：

| 指标 | 数值 |
|---|---:|
| clean adapter mean | 1.240 |
| glare adapter mean | 1.803 |
| glare uplift over clean | +0.563 |
| glare severity Spearman | 0.602 |
| Teacher glare severity Spearman | 0.865 |
| adapter/Teacher patch Spearman | 0.538 |
| corruption-mask patch AUROC（仅诊断） | 0.637 |

解释：

- Teacher 与 adapter 均出现明确优化趋势；
- adapter 对未见 family 的平均强度和 severity 方向正确；
- 但 held-out route 上空间定位只有弱到中等趋势，且 adapter 明显低估 Teacher
  在 glare patch 上的幅度；
- 因此结果只允许进入真实 EVAViT bounded smoke，不允许扩展闭环矩阵。

## 7. A800 8-clean-frame infrastructure smoke

Slurm job：`1062951`
节点：`gpu2`
资源：1×A800、8 CPU、96 GB、2 小时上限
输出：

`/public/share/lidachuan/orion_assets/observation_uq_v3/runs/observation_uq_v3_real_seed20260826_r1/`

数据边界：

- 16 个连续帧；
- 8 条 route；
- 4 train / 2 validation / 2 held-out route；
- 每条 route 连续 2 帧，允许验证最小 temporal input；
- train family：blur、dark；
- held-out family：glare；
- severity：1、3；
- 只损坏 annotation 中解析出的 camera index 0；不口头假设 camera 名称；
- 旧 record 中的 representation/actual target 均不被 v3 trainer 读取。

该作业不运行 CARLA、LLM、VAE/diffusion、actual-target 训练、governor 或 Stage B。

作业成功完成，耗时 7 分 09 秒，退出码 0，峰值 RSS 约 30.5 GiB。产物已同步到：

`results/observation_uq_v3/real_seed20260826_r1/`

主要结果：

| 指标 | 数值 |
|---|---:|
| Teacher loss | 0.9526 → 0.6058 |
| Adapter loss | 0.00511 → 0.000150 |
| validation adapter/Teacher patch Spearman | 0.957 |
| held-out route adapter/Teacher patch Spearman | 0.944 |
| validation glare Teacher uplift over clean | +0.000532 |
| held-out glare Teacher uplift over clean | +0.001144 |
| validation Teacher mask AUROC | 0.460 |
| held-out Teacher mask AUROC | 0.450 |

Adapter 对 Teacher 的蒸馏是成功的，但 Teacher 几乎输出常数。该结果不能判断
clean-conditional 架构是否可行，因为 Teacher optimizer 实际只看到 4 条 train
route × 每条 2 帧，即 8 个 clean frame。8 帧只够验证接口、显存、存储和训练闭环，
不足以学习正常驾驶特征分布。

这是 2026-08-26 的正式纠正 amendment：job 1062951 降级为
`infrastructure smoke only`，不再用作 architecture stop/continue gate。

### 7.1 输入变化诊断

为区分“corruption 没改变 backbone”与“Teacher 没学会”，job `1062954` 只读取
paired feature 做非训练诊断，54 秒完成。held-out glare：

| 指标 | 数值 |
|---|---:|
| mask 内 paired cosine error | 0.3701 |
| mask 外 paired cosine error | 0.00563 |
| inside - outside | 0.3644 |
| paired feature mask AUROC | 0.9907 |
| severity Spearman | 0.8095 |

因此真实 EVAViT 对干预有强且局部的响应。job 1062951 的阴性 UQ 结果来自
Teacher 没学到正常条件分布，而不是 corruption 或 backbone 没产生信号。

## 8. 继续/停止 gate

有意义的 Teacher gate 至少需要：

1. Teacher train loss 下降；
2. validation 和 held-out route 上，未见 glare family 的 Teacher 均值高于 clean；
3. severity 1→3 的 Teacher 分数总体呈正趋势；
4. Teacher 的空间 mask AUROC 明显高于随机；
5. clean route shift 不使 false trigger 完全淹没退化响应。

若 Teacher 本身对真实 EVAViT 的 held-out glare 没有响应，暂停并修改 Teacher，
不归咎于 adapter。若 Teacher 有响应而 adapter 没有，保留职责划分，只修改蒸馏
结构或采样。只有真实 smoke 有趋势后，才生成 native-engine degradation 数据；
在 native-engine gate 通过前仍不扩完整闭环矩阵。

## 9. 更正后的 560-clean Teacher viability run

job：`1062955`（首次运行）/ `1062956`（复用同一 shard 的修复重试）
状态：Teacher gate 已完成，未通过
目标：先判断 Teacher，不训练 adapter

数据采用去重 FP16 shard：

| split | route | clean frame | glare observation |
|---|---:|---:|---:|
| train | 35 | 560 | 0 |
| validation | 5 | 80 | 160（severity 1/3） |
| held-out | 5 | 80 | 160（severity 1/3） |
| 合计 | 45 | 720 | 320 |

每条 route 使用 16 个连续帧。预计 feature payload 约 19.5 GiB；clean token 只保存
一次，glare 不进入 train split。Teacher 使用两个独立成员、12 epoch、batch 8。

该规模仍是第一版，而不是最终训练规模，但已经足够作为“是否存在收敛和未见 route
响应趋势”的最低有意义实验。只有该 Teacher gate 通过后才重新启动 adapter 蒸馏。

首次 job `1062955` 已成功生成不可变 shard，但旧版 AUROC 实现构造
`N_positive × N_negative` 矩阵，Teacher 训练后在评估阶段被 Slurm OOM kill。这是
评估实现错误，不是模型结果。已将 AUROC 改为排序/Mann–Whitney 的
`O(N log N)` 精确实现，并以 job `1062956` 复用同一 shard 重跑；shard SHA256：

`ab8d16ce9ffe67aba192ae331b102bcda8ccf917b4bbe491e86e452367b5beac`

job `1062956` 运行 12 epoch，训练 loss 为 `0.6860 → 0.4356`，但独立响应没有
通过 gate：

| split | glare uplift | severity Spearman | mask AUROC |
|---|---:|---:|---:|
| validation route | +0.00236 | -0.0103 | 0.5174 |
| held-out route | +0.00295 | +0.0398 | 0.4444 |

因此不能把 loss 下降解释成 observation uncertainty 已学会，也不能开始 adapter。

## 10. v3.1 masking leakage amendment

对 job `1062956` 的代码审计发现两个结构性泄漏，而不是单纯 epoch 不足：

1. 当前目标 patch 虽被 masked，上一帧同一位置仍原样进入
   `previous_projection`；连续退化可从时间支路直接复制。
2. 单 patch 棋盘遮挡仍暴露同一连续 glare 区域的相邻 patch；Teacher 可以预测
   “退化后的局部一致模式”，而不是相对 clean 条件分布产生 surprise。

这解释了训练 loss 持续下降但 glare 空间 AUROC 近似随机。继续训练原 v3 只会强化
该捷径，因此 job `1062956` 的 12 epoch checkpoint 被保存为诊断产物，但不续训。

v3.1 做最小且仍然 generator-independent 的修复：

- 用 `4×4` target block、`2` patch halo、`3×3` phase grid；9 个 phase 对所有
  patch 恰好覆盖一次；
- context mask 同时用于当前帧和上一帧，目标区域不能通过 temporal branch 泄漏；
- spatial context 改为 dilation `1/2/4` 的卷积，感受野半径 7；
- 加入归一化二维坐标投影，帮助大块遮挡时恢复视角内空间先验；
- 优化仍只读取 560 个 clean train frame；glare family、severity 和 mask 均不进入
  loss；
- 训练改为 24 epoch，每 4 epoch 用 clean validation prediction loss 选一次
  checkpoint，最终恢复 validation 最优状态。

由于 glare 已用于定位和修改架构，它从此降级为 `architecture-development
diagnostic`，不能再作为最终论文中的严格未见 family。正式泛化证据必须换用冻结
设计后才生成的 CARLA native weather/temporal sensor fault 等新来源。

本地 v3.1 相关测试：`45 passed`。在 v3.1 Teacher 出现明确的独立空间响应以前，
adapter、ORION 微调和 Stage B 闭环矩阵继续冻结。

### 10.1 v3.1 A800 submission

job：`1062977`
run id：`observation_uq_teacher560_v31_seed20260826_r1`
资源：1×A800、2 CPU、64 GiB、1 小时（提交后由 8 CPU 原地缩减，保留排队年龄）
训练：24 epoch；每 4 epoch 仅以 clean validation prediction loss 选择 checkpoint
状态（2026-08-26 16:39 CST）：`PENDING (Priority)`

提交前远端 Python 3.8 / Torch 环境测试 `10 passed`，dry-run 确认不重新抽取
feature、不训练 adapter、不运行 Stage B。Slurm 显示 gpu1/gpu2/gpu4 各有 1 个
未分配 GPU，但分区启用 `PrivateData=jobs`，本作业前存在不可见的更高优先级队列；
缩小 CPU/内存/时限以及指定候选节点的 test-only 调度均不能更早启动。实机
`nvidia-smi` 还显示多张 Slurm 已分配 GPU 处于近零利用率；这些卡不能绕过调度器
直接使用。当前预测开始时间为 2026-08-28 17:21:50，作业保留以累计排队年龄。

16:50 CST 调度器已用 2 CPU 的新资源形状重新评估，仍为 `PENDING (Priority)`，
预测时间未改变，证明 CPU 数不是当前 blocker。不能取消后重新提交；新 Job ID 会
丢失排队年龄。为避免得到 GPU 后因时限中断而丢失训练，v3.1 已加入每次 clean
validation 后原子保存的 `teacher_v31.progress.pt`，并加入 `--resume` 续训路径；
本地相关测试仍为 `45 passed`。

### 10.2 Portal resource counter cross-check

平台页面显示 `CPU 总核数 448 / 可用 448 / 已用 0` 与
`加速卡总数 56 / 可用 56 / 已用 0`。这两个总数恰好等于 `Nvidia_A800` 分区的
7 个节点 × 每节点 64 CPU / 8 A800。相同时间的 Slurm 权威状态为：

- CPU：`398 allocated / 50 idle / 448 total`；
- GPU GRES：节点级 `53 allocated / 3 unallocated / 56 total`；
- 本账号：0 个 RUNNING job，只有 job `1062977` PENDING。

因此 portal 的“已用 0”与“可用 56”更可能是账号自身用量/名义额度视图，而不是
集群全局可立即调度量。portal 的“加速卡”口径即使还聚合 DCU，当前账号可见的
Slurm GRES 也只有 `gpu:NVIDIAA80080GBPCIe`；没有可提交的 DCU partition/GRES。
此外当前 ORION 环境依赖 CUDA/PyTorch/CARLA，DCU 不能作为无需迁移的等价替代。
调度判断继续以 `squeue/scontrol/sinfo` 为准。

### 10.3 Single-allocation conditional continuation

为避免 Teacher 通过后再次排队，同时避免用空壳进程长期占卡，job `1062977` 已在
不取消、不改变 Job ID 的情况下采用显式 continuation config：

`configs/observation_uq_v31_continuation.json`

作业仍只申请 1×A800、2 CPU、64 GiB，不独占节点；时限由 1 小时原地调整为
2 小时，预测开始时间没有因此变晚。集群 `PreemptMode=OFF`，一旦 allocation 开始，
不会因更高优先级普通作业被抢占。

Teacher 完成后使用预先冻结的 development gate：

- validation 与 held-out route 的 mask AUROC 均至少 0.55；
- 两个 split 的 glare score uplift 均至少 0.005；
- 两个 split 的 severity Spearman 均至少 0.10。

所有条件同时通过，才在同一 allocation 内继续 24 epoch clean-only adapter
distillation；任一条件失败即写出 `adapter_continuation_decision.json` 并释放 GPU。
Adapter optimizer 只读取 560 个 clean train frame，target 只来自 frozen v3.1
Teacher；glare、mask、severity、actual target、driving gradient 均不进入 optimizer。
该 clean-only adapter 是 generator-independent 的第一版外推 gate，不代表正式
adapter 方案已经通过。

Teacher 和 adapter 都保存原子 progress checkpoint。新增独立恢复入口：

`scripts/train_observation_uq_adapter_v31.py`

本地完整相关回归仍为 `45 passed`，其中 Teacher/adapter 均覆盖断点续训。远端
Python 3.8 语法检查通过。continuation config SHA256：

`609189a3b3a486785426f5420604ea90e96dc885719528edd2ef63c01dbe6695`

该 continuation 明确不授权 actual-target 训练、ORION fine-tuning、Stage B 或
空闲 GPU workspace。

### 10.4 Resource-limit amendment and local compute plan

2026-08-26 17:20 CST 再次读取 Slurm controller 时，job `1062977` 当前显示
`TimeLimit=06:00:00`、2 CPU、64 GiB，仍为 `PENDING (Priority)`，预计开始时间仍是
2026-08-28 17:21:50。10.3 中的 2 小时是更早时刻的历史状态；这里按时间戳追加，
不静默改写。当前不主动把时限改回较短值。

为避免 A800 排队阻塞 Teacher/adapter 迭代，新增三级算力执行计划：

`docs/compute_tier_execution_plan_2026-08-26.md`

核心划分是：Mac 负责测试、指标与可视化；4060 Ti 负责预提取 feature 上的完整
Teacher/adapter 训练；A800 只负责 ORION/视觉特征抽取、native-engine 数据、CARLA
闭环和后续 ORION/VLM 微调。现有 20 GiB 单文件 shard 对 32 GiB 台式机主存过紧，
因此正式迁移前先实现 lazy、带 hash 的分片 feature store；job `1062977` 继续使用
现有不可变 shard，不因基础设施修改改变输入。

## 11. v3.1 Teacher result and stop decision

job `1062977` 比预测时间提前调度，在 `gpu5` 于 2026-08-26 18:00:49--18:16:38
CST 完成，`ExitCode=0:0`，wall time 15:49。Slurm accounting 最终记录 8 CPU、
64 GiB request、batch `MaxRSS=22569244K`（约 21.5 GiB）；这与 pending 阶段曾显示
的 2 CPU 不一致，后续资源规划以完成后的 accounting 为准并保留该差异。

优化过程本身正常：

- clean train loss：`0.705599 -> 0.501646`；
- clean validation loss：epoch 4 的 `0.582345 ->` epoch 24 的 `0.533258`；
- validation-best checkpoint 为 epoch 24；
- optimizer 只读 560 clean frame / 35 route，没有读取 corruption target 或 actual
  target。

但是预注册的 6 个独立诊断检查全部失败：

| split | mask AUROC（gate 0.55） | glare uplift（gate 0.005） | severity rho（gate 0.10） |
|---|---:|---:|---:|
| validation held-out family | 0.431484 | 0.000524 | 0.062068 |
| held-out route + family | 0.370239 | 0.000819 | 0.046593 |

两项空间 AUROC 不仅未过 gate，还低于随机 0.5。相比 v3.0，v3.1 的 glare uplift
从 `0.00236/0.00295` 下降到 `0.00052/0.00082`，AUROC 从
`0.517/0.444` 下降到 `0.431/0.370`；severity 正相关略有改善，但仍弱且远低于
阈值。因此不能用“再增加 epoch”解释或修复这一结果。

更关键的是 clean score 的 route shift：train、validation、held-out route 均值分别为
`0.648689 / 0.688272 / 0.707755`。train 到 validation/held-out 的 clean shift
约为 `0.0396/0.0591`，是相应 glare uplift 的约 75/72 倍。当前 Teacher 主要编码
scene/route conditional prediction difficulty，而不是局部 observation insufficiency。

按照预先冻结的 continuation policy：

- `adapter_continuation_decision.json` 为 `passed=false`；
- clean-only adapter 未训练，也没有 `adapter_v31.pt`；
- actual-target training 和 Stage B 仍未授权；
- 当前 masked-convolution Teacher 路线停止，不继续加 epoch，也不调整 gate 为结果
  让路。

远端产物已经同步到：

`results/observation_uq_v3/teacher560_v31_seed20260826_r1/`

Teacher checkpoint 与 report 的本地 SHA256 分别为：

- `27b12408e2aaa20fccb59c32e17a1dd8a8386d9b670f0bfb85c4886463c2f46d`
- `4e8c6b185f7bcb1bd41be535e5e5bfc07fab97987150bf6ff51f52f80806b217`

下一步不应直接设计更复杂 adapter，而应先在现有 feature shard 上比较不需要
corruption label 训练的候选基础信号：标准化时序残差、跨视角一致性、clean
conditional residual 的 view/position calibration。只有其中至少一个在 validation 和
held-out route 同时产生稳定的空间定位与 severity 趋势，才将其作为 adapter 的
clean-only 蒸馏 target；否则 observation-UQ 的监督定义需要重新设计。

## 12. Candidate signal audit v1 preregistration

用户授权继续并优先使用算力平台。当前不恢复 adapter 或 Stage B，而是在同一不可变
shard 上做一次短的、无 optimizer 的 candidate signal audit。所有 deployable 候选
都满足：score 和 calibrator 只能读取 observation feature；view/position calibrator
只能用 560 个 clean train frame；family、severity 和 mask 只在全部 score map 计算
完成后用于评估。

预先冻结的候选为：

1. `teacher_raw`：失败的 v3.1 Teacher 原始 conditional surprise，用于复现基线；
2. `teacher_viewpos_z`：按 clean train 的 view/position median/MAD 标准化；
3. `temporal_raw` / `temporal_viewpos_z`：相邻帧同 patch cosine residual；
4. `spatial_raw` / `spatial_viewpos_z`：当前 patch 与 8-neighbour mean 的 cosine
   residual；
5. `feature_rms_raw` / `feature_rms_viewpos_z`：feature magnitude 及其双侧 clean
   标准化偏差；
6. `paired_clean_delta_oracle`：同一帧 observed 与 clean feature 的 cosine delta，
   只作为非部署 diagnostic upper bound，禁止作为 adapter target。

现有 shard 不含相机投影或跨视角对应关系，因此 v1 不把同 grid coordinate 的不同
camera patch 冒充 cross-view consistency。真正的 geometry-aware cross-view signal
留到带投影元数据的新 shard。

任一 deployable candidate 的 development gate 预先固定为 validation 与 held-out
route 两个 split 同时满足：

- mask AUROC >= 0.55；
- family mean uplift > 0；
- severity Spearman >= 0.10；
- uplift / absolute clean route shift >= 0.25。

所有检查同时通过也只表示“值得作为下一版 adapter target 候选”，不会自动授权
adapter、actual-target training 或 Stage B。若 paired oracle 高而所有 deployable
candidate 低，说明 feature 中存在可观测差异但当前自监督定义失败；若 paired oracle
也低，则优先审计 feature/mask 对齐和 degradation 是否真正改变 backbone 表征。

### 12.1 Tie-aware severity metric amendment

首次 audit v1（job `1064261`）完成后，唯一接近 gate 的
`temporal_viewpos_z` 在 validation severity Spearman 上失败。代码复核发现共用的
`_rankdata` 对重复值使用 ordinal rank，而 severity 只有 1/3 且每一级大量重复；
正确 Spearman 必须给 ties 分配 average rank。该错误会使 rho 依赖同 severity 样本的
内部顺序。

修复只改变 Spearman 评估，不改变任何 score map、AUROC、uplift、route-shift、模型
或数据。旧 job `1064261` 结果保留为 v1，不覆盖；以新 run id 提交 v1.1 tie-aware
复算。原 v3.1 Teacher 即使重算 rho 也不会通过，因为两个 split 的 AUROC 和 uplift
已独立失败，因此 adapter stop decision 不被追溯性推翻。

### 12.2 Temporal candidate route-robustness follow-up

tie-aware v1.1 显示 `temporal_viewpos_z` 通过 aggregate development gate。由于
`local_glare` 已是 architecture-development family，且 aggregate 可能被少数路线
驱动，提交 v1.2 前追加以下预固定稳健性条件；它们仍只决定是否把时序自一致性提升
为“下一版 target 候选”，不授权 adapter：

- 每个 split 至少 80% route 的 mask AUROC >= 0.55；
- 每个 split 的 median route AUROC >= 0.60；
- 排除 `previous_valid=false` 的首帧后，aggregate AUROC >= 0.60；
- severity 1 和 3 的 mask-inside mean 均高于 mask-outside mean；
- 高 severity 的 example mean 高于低 severity。

同一份 detailed report 还输出逐 view AUROC 和 paired oracle 的对应分解。计算仍复用
同一 shard、同一 clean calibrator、同一 score 定义，不训练参数、不读取 actual
target、不启动 Stage B。

## 13. First post-freeze cross-family screen

v1.2 route-robustness gate 通过后，下一步不在 glare 上训练 adapter。冻结
`temporal_viewpos_z` 的定义和全部 gate，生成一个此前没有进入真实 feature audit 的
`local_blur` family，作为 post-freeze cross-family screen：

- 同一 35/5/5 route split、每 route 16 连续帧；
- train split 仍只有 clean feature；
- validation/held-out route 生成 severity 1/3 的 front-view local blur；
- score、clean calibration 和 gate 完全复用 v1.2；
- 不因 blur 结果调整 target 或阈值；
- paired clean delta 仍只是 diagnostic oracle；
- 不训练 adapter、actual target 或 ORION，不启动 Stage B。

该实验可以检验从 glare-driven architecture development 到另一个 operator family 的
泛化，但仍属于 synthetic cross-family evidence，不能替代 CARLA native weather/
sensor fault 或真实恶劣数据。它通过后才值得准备 native-engine feature extraction；
失败则不把 temporal target 推进到 adapter。

### 13.1 Post-freeze `local_blur` result

job `1064728` 于 `gpu2` 完成，UTC 时间为
`2026-08-26T11:49:01Z--12:04:59Z`，Slurm wall time `15:59`，退出码 `0:0`。
作业申请 1×A800、8 CPU、96 GiB，batch `MaxRSS=53160780K`（约 50.7 GiB）。它生成
720 个 clean feature 和 320 个仅用于诊断的 `local_blur` observation，feature shape
为 `[6,40,40,1024]`，FP16 shard 大小约 19.04 GiB。shard 和 report SHA256 分别为：

- `3f01ff65a6f39426bf7fb1518ef80fd2cc540bc7ea5ee02eca5032961928eb12`
- `d51054d8cb19b40304705bcc0377e760f8850ade804f4b381a4c92e169f5a9b3`

`temporal_viewpos_z` 是唯一保留明显跨 family 信号的 deployable candidate：

| metric | validation | held-out route |
|---|---:|---:|
| aggregate mask AUROC | 0.615532 | 0.595533 |
| positive score uplift | +0.027959 | +0.024529 |
| uplift / clean-route-shift | 1.3799 | 0.4361 |
| severity Spearman | **0.095953** | **0.066720** |
| previous-valid-only AUROC | 0.625798 | 0.602839 |
| median per-route AUROC | 0.618495 | 0.600694 |
| front-view AUROC | 0.585674 | 0.575376 |

预注册 aggregate gate **失败**，失败项仅为两个 split 的 severity Spearman 没有达到
`0.10`。不得因数值接近阈值而事后放宽 gate。与此同时，预注册的 detailed
route-robustness follow-up 全部通过：validation 5/5、held-out 4/5 route 的 AUROC
至少 0.55；两个 split 的 median route 和 previous-valid-only AUROC 均至少 0.60；
severity 1/3 的 mask 内分数均高于 mask 外，且 severity 3 的 example mean 高于
severity 1。mask-inside minus outside 从 severity 1 到 3 为：

- validation：`0.5051 -> 1.1440`；
- held-out route：`0.4052 -> 1.1164`。

因此结果应解释为：冻结后的时序自一致性定义能跨到未见 `local_blur`，并保留较稳定
的局部空间响应和组均值 severity 方向；但样本级 severity 排序不够稳定，尤其整个
example 同时平均六个 view 会稀释只发生在 front 局部区域的信号。这个解释不追溯性
改变 aggregate gate 的失败状态，也不授权 adapter 训练。

paired-clean diagnostic oracle 的 AUROC 为 `0.990269/0.989923`，说明 backbone 对
blur 干预的局部变化仍然清楚；瓶颈是 deployable temporal statistic 对变化幅度和
个体排序的表达，而不是 corruption 没改变 feature。

小型 report、hash 和 Slurm log 已同步到：

`results/observation_uq_v3/unseen_local_blur_seed20260827_r1/`

下一步不继续枚举 synthetic family，也不训练 adapter。保留 temporal map 为
`native-engine candidate`，先构造不复用 pixel corruption generator 的 CARLA 原生
天气或传感器事件观测，检验其是否能预测独立的 ORION 表征/任务退化。native gate
通过以前，actual-target training、ORION fine-tuning 和 Stage B 继续冻结。

## 14. Native CARLA weather gate preregistration

2026-08-26 20:53:59 CST，在生成任何 native observation 以前冻结：

`configs/observation_uq_native_weather_v1.json`

config SHA256：
`ada4335e9438c68c8b394e1ceb15750fd2e4cc4dfbef748adba1a16ac9fa53a8`。

该实验不使用 `uq_estimator.corruptions` 或任何 pixel-space corruption generator。
CARLA 0.9.15 在完全相同的 ego/camera/world pose 上直接渲染 `clear`、
`fog_light(fog_density=25)`、`fog_heavy(fog_density=75)`。固定两条基础地图兼容路线：

- `Town01/Route146`，16 个连续 route pose；
- `Town04/Route203`，16 个连续 route pose。

每个 condition 保存六视图与 BEV；随后只加载冻结 ORION checkpoint 的 EVAViT
backbone，按正式 inference resize/crop/normalize 提取 `[6,40,40,1024]` FP16 feature。
不会加载 LLM、planner、actual target、adapter 或 governor。`temporal_viewpos_z` 的
view/position calibrator 仍只用既有 560 clean frame / 35 route 拟合。

paired clean feature delta 只在全部 deployable score 计算完成后作为诊断参考，不进入
score 或 calibrator。全部预注册检查必须同时通过：

1. light fog score uplift over clear > 0；
2. heavy fog score mean > light fog；
3. fog severity 的 sample-level Spearman >= 0.10；
4. light/heavy 各自的 score 与 paired-clean patch delta Spearman >= 0.10；
5. light/heavy 各自识别 paired-delta top 20% patch 的 AUROC >= 0.60；
6. paired delta 为正且 heavy > light；
7. 两条 route 分别满足 heavy > light > clear。

这是有意严格的 persistent-appearance gate。雾是持续且稳定的观测退化；如果 temporal
score 失败，不能继续把它当作通用 adapter 唯一 target，而应把它降级为多源 target
中的 temporal component，并新增能表达“稳定但信息不足”的 appearance/evidence
component。通过也只允许另行预注册 adapter 训练，不会自动启动训练或 Stage B。

### 14.1 gpu2 Vulkan infrastructure failure amendment

首次 job `1065401` 在 `gpu2` 启动 30 秒后，于任何 CARLA capture、feature extraction
和 score evaluation 之前被 Vulkan preflight 拒绝：NVIDIA ICD 的
`vkCreateInstance` 返回 `ERROR_INCOMPATIBLE_DRIVER`。Slurm batch MaxRSS 仅约 1.7 MiB，
输出目录不含 capture 或实验 report。历史 job `1050547` 已在同一 `gpu2` 复现完全
相同的 Vulkan 错误，而已验证的 CARLA 闭环作业均在 `gpu4` 通过 Vulkan。

因此 `1065401` 只记为节点运行时失败，不属于 native-weather 阴性结果。保留其 log
和不完整 output root，不覆盖。实现/预注册/gate 均不修改；唯一运行 amendment 是用
新 run id `observation_uq_native_weather_seed20260826_r2_gpu4` 指定已验证的 `gpu4`
重试。

### 14.2 Multi-map CARLA lifecycle amendment

job `1065402` 在 `gpu4` 通过 Vulkan，并完整渲染了第一条 Town01 route 的
`3 conditions × 16 poses × 7 sensors = 336` 张图；随后同一 UE4 进程执行
`client.load_world(Town04)` 时 segmentation fault。capture client 等待 120 秒后以
timeout abort，fail-closed cleanup 删除了未形成双 route manifest 的临时 capture；
只保留 Slurm/CARLA log 和 source hash。没有运行 feature extraction 或 UQ audit，
因此 `1065402` 也不是实验结果。

该现象与 signal/gate 无关，且不需要改变预注册设计。r3 唯一实现修复是：Town01 和
Town04 各自使用全新 CARLA server 进程和独立 capture manifest，服务器退出后由同一
冻结 EVAViT extractor 合并两条 route。天气、位姿、路线、图像预处理、calibrator、
score 和 gate 均保持不变。资源按实测单进程路径从 8 CPU 原地修正为提交时直接申请
2 CPU，内存仍为 96 GiB。

### 14.3 Per-route server port-lifecycle amendment

job `1065416` 在 `gpu4` 使用每条 route 一个全新 CARLA server。Town01/Route146 已
完整成功，保留了独立 `capture_manifest.json` 和
`3 conditions × 16 poses × 7 sensors = 336` 张图。第一台 server 停止后，第二台
server 尚未开始 Town04 capture 就因 `bind: Address already in use` 退出；没有产生
Town04 capture、feature 或 UQ report，因此该 job 仍只属于实现层失败，不是实验结果。

根因是旧 runner 将第二个 RPC port 设为第一个 port 加 1，但 CARLA 除 RPC 外还会
占用相邻 streaming port；同时只终止 `CarlaUE4.sh` PID 不能显式保证 UE4 descendant
一并退出。r4 只做以下运行时修复，不改变任何数据、score 或 gate：

- 每个 CARLA server 用 `setsid` 建立独立 process group，结束时对该组发送 TERM；
- Route203 RPC port 改为 Route146 port 加 1000，确保两组端口范围不重叠；
- 只读复用 r3 已有完整 manifest 的 Route146 capture，r4 仅重新采集 Route203；
- extractor 仍在一次冻结模型运行中读取两个 capture root，输出同一双 route report。

复用路径和 manifest hash 必须进入 r4 日志及 artifact hash；r3 capture 不移动、不覆盖。

### 14.4 Low-quality fog visual-QA rejection and Epic amendment

job `1065700` 修复了端口生命周期并完成 Town04/Route203 capture；r3 Route146 和 r4
Route203 的前视图/BEV 被同步到本地并生成 GIF 后，人工 visual QA 在 feature report
形成以前发现：`fog_light` 与 clear 几乎同样清晰，`fog_heavy` 主要是全局变暗，远处
建筑、车道线和物体仍保持锐利，没有形成研究问题要求的能见度/视觉证据损失。CARLA
server log 同时显示 `-quality-level=Low` 将 `sg.EffectsQuality` 设为 `0`。

因此于 `2026-08-26T21:36+08:00` 主动取消 job `1065700`。它没有写出
`native_weather_features.pt` 或 audit report；两条 Low capture 只保留为 pipeline/visual
QA failure evidence，不能进入 native signal gate，也不能与后续 Epic capture 混用。

在任何新的正式 capture 之前冻结 v2：

`configs/observation_uq_native_weather_v2.json`

config SHA256：
`667b853676c25451b0d4348741496b92961e5fe970188e72279ed54b05a3a123`。

v2 保持路线、pose 数、天气参数、score、clean calibrator 和全部数值 gate 不变；唯一
研究必要 amendment 是强制 `quality_level=Epic`，将渲染质量写入 capture/feature
attestation，并让 extractor/audit fail-closed 拒绝非 Epic 输入。正式双 route capture
前先跑 Town04/Route203 三个位姿、不加载 ORION 的视觉预检；只有 heavy 明显损失远景
细节而非仅全局变暗、light 位于 clear 与 heavy 之间、几何仍对齐，才允许正式采集。

Epic visual preflight job `1065701` 在 `gpu4` 于 2 分 52 秒完成，退出码 `0:0`；仅
渲染 Town04/Route203 的 3 个 exact-pose 样本，不加载 ORION。前视图与 BEV GIF 均
显示：light 相比 clear 已降低远景对比度；heavy 明显遮蔽远处山坡、树木、路灯和道路
细节，同时近处道路结构仍相对可辨；三条件几何对齐。它通过 v2 的人工视觉前置门槛。

正式 v2 job `1065702` 使用新 run id
`observation_uq_native_weather_epic_seed20260826_r5_gpu4` 从零重采两条 Epic route；不
复用任何 Low-quality capture。其范围仍仅限 frozen feature/signal audit。

### 14.5 Epic native-weather signal-gate result

正式 job `1065702` 在 `gpu4` 完成，UTC 时间
`2026-08-26T13:45:36Z--14:07:32Z`，Slurm wall time `21:57`，退出码 `0:0`，batch
MaxRSS `22500444K`（约 21.5 GiB）。两条 route 各有 16 个 exact-pose 样本，manifest
均证明 `renderer_quality=Epic`、`paired_world_pose=true`、
`pixel_corruption_generator_used=false`。冻结 EVAViT feature shape 为
`[32,6,40,40,1024]`，shard 大小约 1.76 GiB。

artifact SHA256：

- native feature：`debdeac1cbb1bb72227604a79ef41f38c1be13dde6562256599055d6ea20ddda`；
- audit report：`313ed4c8266ec9ab51578151a49a6f5d30b5b88506092eae7f5e87c09c369245`；
- Town01/Route146 manifest：`2df5ad3db9704577fd244b3e47a49e27996dc397ea25ff2cb39e4acdf766f90b`；
- Town04/Route203 manifest：`de4611d5e7d20e03262b80a2d808f48884aa1a19e98d6b9f0c77e4a277241234`。

原生干预和 `temporal_viewpos_z` 的结论清楚分离。paired-clean diagnostic oracle 表明
backbone 确实观察到随雾增强的表征变化：mean cosine delta 从 light 的 `0.326430`
上升到 heavy 的 `0.447784`。但 deployable temporal score 为：

| condition | score mean | uplift over clear | paired-delta localization Spearman | top-20% delta AUROC |
|---|---:|---:|---:|---:|
| clear | 1.633820 | -- | -- | -- |
| fog light | 1.675412 | +0.041592 | 0.116773 | 0.545402 |
| fog heavy | 1.669846 | +0.036026 | 0.062294 | 0.518480 |

冻结 gate **失败**，10 项仅 4 项通过。关键失败包括：heavy score 低于 light、
sample-level severity Spearman 为 `0.0`、两个 severity 的 localization AUROC 均低于
`0.60`，并且 Town04 的 score 顺序是 `light > heavy > clear`。Town01 虽形式上满足
`heavy > light > clear`，heavy-light 差仅约 `0.0000285`，不能弥补 aggregate 和另一
route 的失败。

因此，Epic fog 已证明“持续外观证据损失存在且 backbone 可见”，同时否定了
`temporal_viewpos_z` 作为通用空间 observation-uncertainty 唯一 target 的方案。它最多
保留为瞬态/时序异常 component；不得基于本结果训练 adapter，也不得启动 actual-target
training、ORION fine-tuning 或 Stage B。下一版监督必须增加对持续退化敏感、推理时不
依赖 paired clean 的 appearance/evidence component，并在同一冻结 Epic shard 上先做
无训练 candidate audit。

小型 report、hash、Slurm log、正式前视图/BEV GIF 和摘要图已同步到：

`results/observation_uq_v3/observation_uq_native_weather_epic_seed20260826_r5_gpu4/`

## 15. Clean-only native appearance candidate preregistration

2026-08-26 22:31:23 CST，在对 Epic native feature 计算任何新 candidate 以前冻结：

`configs/observation_uq_native_appearance_candidates_v1.json`

config SHA256：
`5d38dd2ef936aa4a1cb97884dbc89ae4071065d59c3b2d807a86ee39a4acf456`。

该 follow-up 不生成新图、不加载 GPU、不训练参数，只复用 immutable r5 native feature
和既有 560 clean frame / 35 route。四个 candidate 均只读取 current feature 与 clean
statistics：feature RMS absolute deviation、spatial-neighbour residual absolute
deviation、到 clean view/position prototype 的 cosine distance、对角标准化 feature
distance。corruption condition、severity 和 paired-clean delta 只在分数全部完成以后用于
评价。

每个 candidate 独立沿用第 14 节同一组 native gate；不做 candidate 间事后挑阈值或
学习融合。任一通过也只表示可以成为另行预注册的 multi-source adapter target 中的
appearance component，不授权训练。全部失败则说明手工 clean statistics 不够，应先
设计 clean self-supervised appearance predictor；temporal 继续仅作为 transient
component。

调度 amendment：账号 association 为 `lidachuan/user_lidachuan`，`comput`、
`cigit_cpu_70`、`zhanghuili_cpu`、`sunxianhu_cpu` 均未在 `AllowAccounts` 中包含
`lidachuan`，所以首次 CPU partition submit 被调度器以 invalid account/partition
拒绝，未产生 job。改用账号可访问的 `Nvidia_A800` partition，但提交参数明确不含
`--gres`，因此分配 GPU 数为 0；指定 8 张 GPU 已全部被其他作业占用、但仍有 32 CPU
和充足内存的 `gpu3`，只填充其闲置 CPU/64 GiB memory，不占用或阻塞 A800。

上述 CPU-only 提交随后被 QoS 以 `QOSMinGRES` 保持 pending：该账号在 A800 partition
也被强制要求最少一个 GRES。job `1066323` 在运行前取消，elapsed `0:00`。为不中断
诊断，第二个纯运行时 amendment 是申请 1×A800，并把 clean moment accumulation、
prototype/diagonal distance 和 spatial-neighbour 计算实际移到 CUDA；candidate 定义、
输入、数值 gate 和无训练边界均不改变。指定无需 Vulkan/CARLA、仍有 2 张空闲 A800
和 10 个 CPU 的 `gpu2`，资源改为 1 GPU、8 CPU、64 GiB、30 分钟上限。

首次 CUDA run `1066509` 在 report 形成前退出：native raw maps 位于 CUDA，而冻结的
robust view/position calibrator 位于 CPU，transform 前遗漏 device transfer。它没有
candidate 结果。r3 唯一实现修复是在 calibration 前调用 `item.cpu()`；不改变 raw
score、clean statistics、calibrator、输入或 gate，使用新 output root 重试。

### 15.1 Native appearance candidate result

修复后的 job `1066568` 在 `gpu2` 完成，wall time `2:13`，退出码 `0:0`。它使用 CUDA
计算 clean moments/raw maps，但没有训练任何参数。report SHA256 为
`903f15c2429c71927c605d0d05ca5c8dd5bbf3e6a9ce713f3e663f7e6f855d6b`。

四个预注册 candidate **全部失败**：

| candidate | light uplift | heavy uplift | severity rho | light rho/AUROC | heavy rho/AUROC |
|---|---:|---:|---:|---:|---:|
| feature RMS abs-z | -0.00350 | +0.00443 | +0.16747 | -0.00712 / 0.59672 | +0.05551 / 0.61056 |
| spatial-neighbour abs-z | +0.00294 | +0.00394 | -0.04906 | +0.01538 / 0.56146 | +0.03755 / 0.57616 |
| clean prototype distance | -0.00078 | -0.00131 | -0.00507 | +0.01248 / 0.45812 | +0.01026 / 0.45763 |
| diagonal feature distance | -0.01814 | -0.02519 | -0.08458 | -0.10638 / 0.38219 | -0.10474 / 0.38147 |

feature RMS 的 heavy AUROC 和 severity rho 单项通过，但 light uplift 为负、light
spatial correlation 为负，且两条 route 都不满足 `heavy > light > clear`；不得将临界
单项挑出冒充 candidate 通过。spatial-neighbour 只有很小的 aggregate uplift，severity
和定位均失败。prototype/diagonal 更出现系统性反向响应。

因此 stop decision 是：不以简单 feature statistics 拼接 adapter target；不训练当前
adapter。下一步需要 clean self-supervised appearance/evidence predictor，例如利用被
mask 的当前 patch、邻域、多视角/时序上下文预测 clean feature，并将预测残差/ensemble
disagreement 作为持续退化 component。其训练只使用 clean observation；synthetic
corruption mask 只可作为辅助诊断，Epic fog 和后续独立 sensor/native family 保持为
外部验证。

小型结果已同步到：

`results/observation_uq_v3/native_appearance_candidates_seed20260826_r3_gpu2/`

## 16. Route-balanced clean-manifold audit preregistration

第 15 节排除了简单 clean feature statistics，但不能据此直接重跑 masked
conditional Teacher。对 v3.1 代码的复核表明，它在每个 token 上先做 LayerNorm、训练
目标只比较 cosine 方向，而且推理时用 query 当前帧的空间邻域预测被 mask patch。对于
持续且空间一致的雾，target 与可见 context 会一起进入同一种退化分布；Teacher 可以
重建“雾中自洽的特征”，不需要恢复 clean evidence。这与其 clean prediction loss
持续下降、而 glare/native fog 的空间响应失败并不矛盾，也不是增加 epoch 能修复的
问题。

因此在任何 learned appearance predictor 或 adapter 训练以前，先做最后一个无训练的
多模态 clean prior 诊断。2026-08-26 22:52:51 CST，在计算任何新 score 以前冻结：

`configs/observation_uq_native_manifold_candidates_v1.json`

config SHA256：
`9ded13edae72a6275e6a0fc7d3dfb0fbd6b4251c8f1b7d489c3f3ae55cfa42bd`。

两个 candidate 都只使用 frozen 560 clean train frame / 35 route，并在相同 camera view
与 `40×40` grid position 上建立参考：

1. `appearance_route_knn_cosine_z`：完整 1024-D patch 方向的 cosine distance；
2. `appearance_route_knn_standardized_l2_z`：用 clean-only view/position/channel
   mean/std 标准化后的完整 1024-D Euclidean distance，保留联合特征模式与幅值。

两者都先在每条 clean route 的 16 帧中取最近 reference，再平均最近 5 条不同 route，
避免一个近重复轨迹支配最近邻。560 个 clean calibration query 必须排除自身整条 route，
其 raw distance 再按 view/position 做 median/MAD positive-tail calibration；native query
使用全部 35 条 clean route。与失败的 prototype/diagonal score 的实质差异是：不把
复杂 clean 分布压缩成单一均值，也不假设 1024 个 channel 独立；与失败的 masked
Teacher 的实质差异是：query 当前邻域和上一帧都不作为“预测答案”。

candidate 不读取 condition、severity、corruption mask、paired clean 或 actual target。
paired clean 仍只在所有 score map 完成后评价。沿用第 14 节全部 gate，任一通过也只
能把定义冻结后送往一个新的、未参与架构设计的 native sensor/weather family；Epic fog
从此是 architecture-development evidence，不再是 untouched confirmation set。若两个
candidate 都失败，则停止在 Epic fog 上继续枚举 clean-feature heuristic，转而设计可
学习的 clean density/evidence model；adapter、actual-target training、ORION fine-tuning
和 Stage B 均不自动授权。

### 16.1 Clean-manifold result and supervision correction

job `1066763` 在 `gpu2` 完成，wall time `2:25`，退出码 `0:0`。两个 candidate 均沿
错误方向响应，且两条 route 一致：

| candidate | clear | light | heavy | severity rho | light rho/AUROC | heavy rho/AUROC |
|---|---:|---:|---:|---:|---:|---:|
| route-kNN cosine | 0.54664 | 0.53949 | 0.52996 | -0.22161 | -0.07029 / 0.38373 | -0.09201 / 0.37445 |
| route-kNN standardized L2 | 0.65976 | 0.64381 | 0.63499 | -0.14887 | -0.09779 / 0.38166 | -0.10131 / 0.38045 |

report SHA256：
`9bdda90f69a07c920a6faee9a0e0029f2d07665342b16d97940bf97ac8048829`。

这不是 nearest-neighbour 数量的边缘失败：fog 越重，query 越接近当前 reference bank；
所有全局顺序、空间相关和两个 route 的顺序都反向。数据 lineage 复核同时发现，“560
clean frame”这一历史命名不等于“560 个清晰/高证据 frame”。它们来自 B2D val 的
35 条未施加本地 pixel corruption 的原生记录，但 route 文件夹覆盖 Weather 0--26；
按项目既有 `Weather 0--3 = normal` 口径，35 条 train route 中只有 8 条 normal，27 条
本身是 adverse weather。因此该 bank 应追溯性称为
`unintervened mixed-weather reference`，而不是 certainty/clear manifold。

这使阴性结果具有明确方法含义：observation uncertainty 不能再定义成“偏离训练分布
有多远”。低能见度可能在训练分布内，甚至比新的 Epic-clear render 更接近 B2D 的混合
天气分布；density/OOD、masked local predictability 和信息缺失是三个不同量。基于同一
bank 调 k、换 flow 或训练更复杂 density model 都不会闭合研究问题，因此停止这条
路线。

下一版 Stage-1 主监督改为 `counterfactual evidence loss`，而非 clean density：对相同
世界/相机 pose 的 reference observation 与 degraded observation，计算冻结视觉表征的
空间损失（方向与幅值）和可选的图像结构/可见度损失；adapter 推理时只看 degraded
observation，学习预测证据损失的位置与强度。corruption mask 只能是低权重边界辅助，
actual ORION target 只做独立任务退化诊断/辅助，不定义通用 observation uncertainty。
训练与验证必须按 intervention family 整类隔离；Epic fog 已用于架构开发，正式确认需
使用冻结方案后生成的另一类 native sensor/weather event。Stage-2 才由 ORION/VLM 学习
哪些空间 uncertainty 与路径风险相关，以及应采取何种保守行为。

本结果只停止 density/manifold target，不授权立即训练 adapter。下一步先冻结
counterfactual target 的组成、训练 family、整类留出、clean false-positive 约束与验证
gate，再做 bounded learnability pilot。

## 17. Counterfactual evidence Stage-1 protocol

2026-08-26 23:31:10 CST，在生成新的 optimizer observation 以前冻结：

`configs/observation_uq_counterfactual_evidence_v1.json`

config SHA256：
`4d40c1f81455336e763b8396d02fa39ecf82b2212f010029b264b745f70b1686`。

新定义不再把 reference distribution density 当 uncertainty。每个相同 pose 的
unintervened/observed pair 产生三个空间 target：

1. `persistent_direction = 1 - cosine(F_obs, F_ref)`；
2. `persistent_magnitude = |log RMS(F_obs) - log RMS(F_ref)|`；
3. `transient = |temporal_cosine_obs - temporal_cosine_ref|`，首帧无效。

三分量各自只用 train target 的 q95 缩放。reference observation 显式用零 target 加入
false-positive 训练；severity ranking 仅在“实测 target 的高 severity 确实高于低
severity”patch 上激活，不能用 corruption 名称强迫排序。mask optimizer weight 固定为
0；actual-target pilot weight 也固定为 0。两者只在全部 prediction 完成后诊断。

新的 `ObservationEvidenceAdapter` 只接收 current/previous frozen EVAViT patch、显式
feature RMS、temporal cosine change、camera embedding 和二维坐标，输出上述三个
non-negative map。route、planned path、hazard、family、severity、mask 和 paired
reference 均不进入 forward。这样 adapter 只回答“哪里/多强/哪一视角/是否发生时间
变化的观测证据损失”；哪些位置影响行驶以及如何保守响应仍留给 Stage-2 ORION/VLM。

optimizer 使用 35 route × 16 frame = 560 个 reference frame，在
`local_blur/local_dark × severity 1/3` 上构造增量证据损失；每个 route-condition 用
固定 hash 均衡选择六视图之一，每帧 region 独立确定，从而同时覆盖 view 与时间变化。
五条 route-disjoint validation route 复用 train family；`local_glare` 整 family 只在
validation/held-out route 做 development；Epic fog 只做 native development。模型与
gate 冻结后，必须新生成 native rain/low-sun 和有限时间 sensor event 才能充当
confirmation。

为避免此前 8-frame smoke 过度谨慎，正式 extraction 计划直接覆盖：

- 720 reference frames（train/validation/held-out 为 35/5/5 route）；
- 2240 train observed frames；
- 480 validation observed frames；
- 160 held-out-family observed frames；
- 合计 3600 个 `[6,40,40,1024]` FP16 feature grid，预计约 65.92 GiB。

提取作业只生成 route-disjoint paired feature shard，不训练 adapter；请求
1×A800、8 CPU、160 GiB、3 小时上限。训练必须等 shard count/hash/family isolation
验证通过后另行提交。当前仍不授权 ORION fine-tuning、Stage B、governor matrix 或
diffusion decoder。

本地相关测试在提交前为 `35 passed`。

### 17.1 Counterfactual feature extraction and train-target audit

feature-only job `1067228` 在 `gpu2` 完成，wall time `37:55`、退出码 `0:0`。输出严格
满足冻结计数：720 个 reference、2880 个 observed、45 条 route，FP16 shard 约 66 GiB；
artifact SHA256 为
`6381c09ae3818e35f93e5c44c29c2230a7e299f7c621316bfc21e04342bc07e5`。该 job 没有
训练 adapter、读取 actual target、加载 ORION LLM/planner 或运行 Stage B。

首次只读 target audit job `1067818` 在任何 report 形成前被 PyTorch 的超大张量
`quantile()` 元素数限制终止；它不是 target 阴性结果。实现修复用排序后的精确线性插值
order statistic 替代 `torch.quantile`，同时覆盖后续 scale fit/evaluation 中相同风险，
不改变 target、数据或阈值。r2 job `1068444` 完成，wall time `6:32`、退出码 `0:0`，
report SHA256 为
`14d72e4dbbbe12aedbb7e1d63a5924f590b0c511289e07c264a0c111a53d0076`。

三分量 train-only responsive q95 分别为：direction `0.469111`、magnitude
`0.297814`、transient `0.486558`。四个 optimizer condition 的全图平均 target 均满足
severity 3 > severity 1；blur/dark 的 560 个 frame-level paired comparison 在三个分量和
combined 上均为 `100%` 单调。mask 未参与这些统计或 scale fit。

约 `1/6` patch 超过数值响应下限，恰好对应单个干预视图。这既证明跨相机定位没有泄漏，
也提出 ViT 全局注意力是否把局部 target 扩散到整幅相机的问题。因此在训练前新增一个
只读 spatial-support gate：mask 仅作为训练集 target 的事后定位度量，optimizer weight
继续为 0；度量必须限制在已受影响视图内部，不能靠识别哪台相机变化来通过。

spatial-support job `1068507` 在 `gpu2` 完成，wall time `6:23`、退出码 `0:0`，report
SHA256 为
`d6907ca9c78e802b6638adf7c4de1c10d713ece61ece51243fcb4579caeec5c9`。全部冻结门槛通过：

| metric | result |
|---|---:|
| combined within-view mask AUROC median / p10 | 0.8928 / 0.7972 |
| combined inside/outside ratio median | 5.7368 |
| equal-area top-support IoU median | 0.3312 |
| direction / magnitude / transient AUROC median | 0.9400 / 0.8602 / 0.7822 |

四个 family/severity condition 的 combined median AUROC 均在 `0.8346--0.9286`。因此
paired frozen-feature target 虽有整视图低幅扩散，高幅证据仍稳定集中在真实局部干预
区域，满足启动一次 bounded learnability pilot 的前置条件。这只证明 controlled target
具有空间结构，不把 synthetic mask 或 paired delta 提升为真实 uncertainty truth。

### 17.2 Bounded counterfactual-evidence adapter pilot

2026-08-27 01:04 CST，在任何 adapter optimizer step 以前冻结：

`configs/observation_uq_counterfactual_evidence_training_run_v1.json`

config SHA256：
`9494bc67754b0dbd15f14669df374bb266e21ae7dbcf4502ef9d463726b7fe92`。

唯一 optimizer 数据为 35 条 train route 的 `local_blur/local_dark`；5 条 disjoint route
只用于 validation loss 选 checkpoint。整类 `local_glare` 和 Epic fog 都不参与选模。
新增 route gate 还要求在 target 识别的受影响视图内部完成 top-20% 定位，避免 all-view
零 patch 抬高指标。训练固定 24 epoch，不自动续训；任一 route/glare/native gate 失败
即停止，不进入 adapter 集成、ORION/VLM fine-tuning 或 Stage B。

job `1068530` 在 `gpu2` 完成，wall time `1:18:22`、batch MaxRSS 约 73.8 GiB、退出码
`0:0`。该 job 明确记录 corruption-mask optimizer weight `0`、actual-target weight `0`、
ORION fine-tuning `0`、Stage B `0`。report SHA256 为
`c38712428cec2c6ec2e5b5d9f82e99f50c07a45f902b56be5f542bf65791d48f`。

正式 gate **全部失败**，而且首先在同 family 的 route validation 失败：direction、
magnitude、transient patch Spearman 分别为 `0.0035/-0.0101/-0.0129`；combined AUROC
`0.4090`，route median AUROC `0.3919`，受影响视图内部 median AUROC `0.3881`。最佳
checkpoint 被 validation loss 选在 epoch 1；validation loss 从 epoch 1 到 24 固定为
`0.0398933`，ranking loss 始终等于未学习的 margin baseline `0.1`。预测均值约
`2.5e-10`，reference p95 约 `5.1e-15`，证明 adapter 选择了几乎精确的全零解。

因此 glare/native fog 的失败不用于声称 target 不可泛化：route learnability 已先失败。
本结果也不是“epoch 不足”；增加 epoch 不会离开 softplus 饱和的零解。根因是一个可由
train/route 数据单独诊断的 loss imbalance：单相机干预使 5/6 patch 为精确零背景，原
regression 在所有 cell 上归一化；再叠加显式 reference zero loss，零项压倒 responsive
target。原有 target weight 最大仅 4 倍，无法抵消背景数量，输出在首轮即被推入
softplus 低梯度区。

### 17.3 Route-only loss-repair smoke

2026-08-27 07:36 CST，在任何修复后 optimizer step 以前冻结：

`configs/observation_uq_counterfactual_loss_repair_smoke_v1.json`

config SHA256：
`98d4f0d5570e6b07ddc3d22fc272b83a0436b62f6d86146836ff537c8392bb75`。

该 smoke 不改变 target、feature shard、route split、adapter 输入或正式 gate，只做三项
由 route collapse 直接导出的训练修复：

1. 对 measured-responsive cell 与 exact-zero background 分别归一化，再按 `0.75/0.25`
   合并；responsive 由 paired target `>1e-6` 定义，不读取 corruption mask；
2. 重复的显式 reference loss weight 从 `0.5` 降为 `0.1`，背景回归仍保留；
3. output bias 初始化为 `-3`，使初始 softplus 输出低但不饱和。

只训练 4 epoch，只读取 train 和相同 family 的 route validation；不读取 glare、Epic
fog、actual target，不自动触发 24-epoch full run。smoke gate 要求 prediction 明显离开
零解、ranking loss 下降、combined/within-view route 指标出现最低可学习性。job
`1069932` 已在 `gpu2` 运行，资源为 1×A800、4 CPU、140 GiB、1 小时上限；full adapter、
ORION fine-tuning 和 Stage B 继续冻结。

job `1069932` 随后在 `gpu2` 完成，wall time `18:47`、batch MaxRSS 约 73.9 GiB、退出码
`0:0`。report SHA256 为
`e369f8ccb9ee9e2e224e423e4f448426704f3abee5d466a865caa21715912d30`。
修复明确消除了全零塌缩：最佳 checkpoint 为 epoch 3，combined top-20 AUROC
`0.73836`、受影响视图内 record median AUROC `0.68943`，三个 component 中两个达到
Spearman `0.1`，blur/dark 相对 reference 的最小 uplift 为 `0.01027`，ranking loss
降至 `0.03605`。

不过预注册总 gate 仍应正式记为 **失败**，不能按四舍五入改写为通过：combined patch
Spearman 为 `0.0997109 < 0.1`；更关键的是 clean/reference prediction p95 为
`0.39231 > 0.2`。这说明单头回归现在能找出 intervention-responsive 区域，却以显著
抬高 clean 背景为代价。继续增加 epoch 或在同一单头上调 loss weight 不能回答这个
结构冲突，因此不启动原计划的 24-epoch full run，也不读取 glare/native fog 来挑修复。

### 17.4 Sparse hurdle-head architecture smoke

为分离“该 patch 是否发生可测 evidence loss”和“发生以后强度多大”，新增两部分
task-agnostic head：presence 使用 paired feature target `>1e-6` 的支持集监督，
magnitude 只在该 measured-responsive 支持集上回归，最终 score 为
`sigmoid(presence_logit) × softplus(conditional_magnitude)`。support 标签仍完全由冻结
feature pair 计算，不读取 corruption mask；推理输入也仍只有 current/previous frozen
feature、view 和坐标。clean/reference 对 presence 与最终 score 提供显式零监督。

2026-08-27 08:06:15 CST，在任何 hurdle-head optimizer step 以前冻结：

`configs/observation_uq_counterfactual_hurdle_smoke_v1.json`

config SHA256：
`6ec1a09b2ca8eb35325f7ec9b698a93d1a0b50ffcabd680d6c031139b47e0b60`。

该 architecture smoke 沿用 loss-repair smoke 的同一 4-epoch train/route-validation
数据范围和全部 gate，尤其不放宽 reference p95 `<=0.2`；不读取 held-out glare、Epic
fog、actual target 或 mask，不自动触发 full training。平台相关单测为 `11 passed`。
job `1070480` 已在 `gpu2` 运行，请求 1×A800、4 CPU、140 GiB、1 小时上限。只有
route gate 全部通过，才允许冻结独立的正式训练配置；ORION fine-tuning 与 Stage B
仍未授权。

该 job 在 epoch 3 后按运行中明确记录的统一判断提前停止：若 train loss 继续下降，而
route-disjoint validation、尤其 presence loss 连续明显恶化，则无需消耗 epoch 4。
实际曲线为：

| epoch | train total | route-val total | val presence | val magnitude |
|---:|---:|---:|---:|---:|
| 1 | 0.949811 | 1.475510 | 1.177389 | 0.056151 |
| 2 | 0.716362 | 1.767545 | 1.514645 | 0.053467 |
| 3 | 0.646045 | 2.563275 | 2.269363 | 0.060031 |

train 持续改善而 validation 加速恶化，且差异几乎完全由 presence 判别造成；conditional
magnitude 并非主瓶颈。job `1070480` 于 2026-08-27 08:27:56 CST 取消，elapsed
`16:04`，第 4 轮未运行。由于原脚本只在四轮完成后写 checkpoint/report，本次没有
checkpoint，也不伪造完整 gate；独立 early-stop 记录保存在
`results/observation_uq_v3/counterfactual_evidence_hurdle_smoke_seed20260827_r1.early_stop.json`。

结论是：当前 frozen feature + region construction 可以提供空间自洽的 paired target，
单头可以学到部分同-family 定位；但把非零 support 当作 presence 分类，会迅速学习
train-route 的 appearance/corruption shortcut，不能跨 route 泛化。因此停止当前
hurdle-head 和 24-epoch adapter 扩展。下一步不应继续枚举 loss 权重，而应先做按 route
的可分性/shortcut 诊断，并重新审视输入表征、region 生成与 supervision 是否允许模型
在不依赖场景外观的情况下识别 evidence loss。

### 17.5 Presence shortcut diagnosis and v2 feature schedule

只读 shortcut audit 没有重新加载 66 GiB feature tensor，而是联合冻结 target audit、
spatial-support audit、route manifest 和确定性 view-schedule 代码。report：

`results/observation_uq_v3/counterfactual_presence_shortcut_audit_seed20260827_r1.json`

SHA256：
`fbfa8697a3b38e596e99819aeaf348eabf55e9619bcdc58e26cd07e3757f052c`。

诊断确认 hurdle presence 的定义和数据调度共同构成了 shortcut：

- direction/magnitude/transient 的 `target > 1e-6` 比例分别为
  `0.165959/0.166634/0.166605`，几乎精确等于单台相机占六视图的 `1/6`；
- 每个 `route × family × severity` 的受干预相机在全部 16 帧固定不变，因此给定该 key
  的 positive-view 条件熵为 `0`；
- train 的全局 view-key 分布本身接近均匀（entropy `2.5557/2.5850` bit），所以此前
  “覆盖六视图”的检查没有发现 route-level 条件泄漏；
- 与此同时，target 的高幅部分仍具有 within-view mask AUROC `0.8928`、inside/outside
  ratio `5.7368`，说明应保留 continuous paired target，而不是丢弃整个监督定义。

因此禁止继续使用数值 `>1e-6` footprint 作为 presence label，也禁止复用
`route_condition_hash_single/v1` 调度。2026-08-27 08:52:17 CST，在任何 v2 extraction
以前冻结：

`configs/observation_uq_counterfactual_evidence_v2.json`

config SHA256：
`ac5c4ce3d97691e5e1274823256c6ca50c8a4fa67cd2f23752544584313a964d`。

v2 对同一 route-condition 每 4 帧保持一个相机，随后按确定性 cycle 切换；每 16 帧
覆盖 4 台相机，view transition 比例为 `0.2`。预计 train 六视图计数为
`376/400/356/336/388/384`，entropy `2.58257/2.58496` bit。提取器在保存 shard 前还会
用实际 metadata 强制要求每个 route-condition 至少覆盖 4 台相机。family、severity、
route split、region generator、reference lineage 和 feature 数量均不改变。

共享目录在提交前只剩约 66 GiB，不能安全容纳预计 65.92 GiB 的新 shard。因此删除
已被本节确认失效、且不被 v2 使用的旧 v1 tensor：

`counterfactual_evidence_features_seed20260826_r1/counterfactual_evidence_features.pt`

删除大小 `70,837,568,586` bytes，原 SHA256
`6381c09ae3818e35f93e5c44c29c2230a7e299f7c621316bfc21e04342bc07e5`。source/artifact
sidecar、audit、report 和日志全部保留；删除记录为
`results/observation_uq_v3/counterfactual_evidence_features_seed20260826_r1.deleted_artifact.json`。
该文件不在回收站，只能按冻结 v1 extraction 重建。清理后共享空间使用率由 `82%`
降至 `63%`，剩余约 132 GiB；未删除仍被 v2 依赖的 20.45 GB reference shard。

平台兼容测试 `16 passed`。feature-only job `1071011` 已在 `gpu2` 启动，请求 1×A800、
8 CPU、160 GiB、3 小时；运行明确记录 exact-nonzero presence `0`、adapter training `0`、
actual-target training `0`、Stage B `0`。完成后必须先做 lineage/count/schedule 和新的
continuous-target audit，不自动开始 adapter optimizer。

job `1071011` 随后在 `gpu2` 完成，wall time `37:17`、batch MaxRSS 约 82.1 GiB、
退出码 `0:0`。输出计数为 720 reference、2880 observed、45 route，feature shape
`[6,40,40,1024]`，projected/actual payload 规模与 `65.92 GiB` 设计一致；全部 family/split
计数符合冻结协议。protocol SHA256 为
`ac5c4ce3d97691e5e1274823256c6ca50c8a4fa67cd2f23752544584313a964d`，新 feature shard
SHA256 为 `d53ab9b7acc91b89cb061ce1ec880d05b390675f29c515337bcffceda3143499`。提取器只有在
实际每个 route-condition 至少覆盖 4 台相机时才会保存，因此成功标记同时证明 runtime
schedule gate 通过。

平台回归测试仍为 `16 passed`。只读 train target/schedule audit job `1071625` 已在
`gpu2` 启动，请求 1×A800、2 CPU、120 GiB、45 分钟上限。该 audit 明确记录 continuous
target audit `1`、exact-nonzero presence `0`、validation read `0`、adapter training `0`
和 Stage B `0`。

target/schedule audit job `1071625` 在 `gpu2` 完成，wall time `6:42`、batch MaxRSS 约
73.2 GiB、退出码 `0:0`。report SHA256 为
`83715249c5ab5de24e8e83e8fd7166a13c5f7ef8cf8d0cd3cf0645a3b56a2fae`。全部 train-only
诊断通过：180 个实际 route-condition 最少覆盖 4、最多 5 台相机；exact-nonzero
presence 明确禁用。direction/magnitude/transient 的 train responsive q95 scale 分别为
`0.474248/0.295654/0.431052`。前两个分量仍约 `1/6` patch 有非零响应；transient 为
`0.20020`，符合四帧 window 边界会同时产生前后视角变化。blur/dark 两档 severity 在
三个分量和 combined 上的 frame-level paired 单调率均为 `100%`。

随后只运行 train-only spatial-support audit job `1071773`，wall time `6:59`、batch
MaxRSS 约 73.1 GiB、退出码 `0:0`。report SHA256 为
`c3bf88254b2dc61b6ab08b6622acb6a6e3b1fd0e324be45d25f27eed0afdff81`，沿用 v1 原阈值
且全部通过：

| metric | v2 result |
|---|---:|
| combined within-view mask AUROC median / p10 | 0.8987 / 0.8124 |
| combined inside/outside ratio median | 6.1168 |
| equal-area top-support IoU median | 0.3417 |
| direction / magnitude / transient AUROC median | 0.9399 / 0.8588 / 0.8057 |

因此 window-cycle 修复消除了固定相机 schedule，却没有损伤 continuous paired target 的
幅值、severity 顺序或局部空间结构。mask 在该 audit 中仍只用于事后度量，optimizer
weight 为 0。该结果本身不授权 adapter training；下一步需另行冻结一个 bounded smoke，
其中 support 只能来自 train-only target 的高响应/soft-amplitude 定义，不能复活数值
`>1e-6` footprint。

### 17.6 High-support hurdle smoke and route-overfit result

2026-08-27 11:00:43 CST，在 optimizer step 前冻结 high-support hurdle smoke：

`configs/observation_uq_counterfactual_high_support_hurdle_smoke_v1.json`

config SHA256：
`93d82c7a49d04c0446426d6df8d26fa4cf4bd66aa2f4e0c957925252651038d1`。

presence 不再使用数值非零 footprint，而定义为每个 component 的 scaled paired target
高于冻结的 train-responsive q80；三个阈值分别为 `0.0846563/0.171186/0.183789`。
conditional magnitude 继续回归连续 paired target，不读取 corruption mask。最多训练 4
epoch；若连续两次出现 train total 下降、route-validation total 上升，则在第 3 轮自动停。

首次 job `1071779` 的训练曲线在第 3 轮按规则停止，但旧 evaluator 假定每条记录只有一个
响应视角；window-cycle 边界的 transient target 会在前后两台相机同时响应，因此在结果
评估阶段报错。该失败不改变训练，但没有伪造 report。评估器随后改为仅用当前 paired
target 的 persistent direction+magnitude 总响应质量选择视角，不读取 mask 或 intervention
metadata，并加入双 transient-view 回归测试；平台测试 `13 passed`。

修复重跑 job `1071804` 在 `gpu2` 完成，wall time `16:20`、batch MaxRSS 约 73.9 GiB、
退出码 `0:0`。report SHA256 为
`020aa4545bbc3f359173ca1475025228eae101758697ac7f159276d2f7511566`。曲线为：

| epoch | train total | route-val total | val presence | val magnitude |
|---:|---:|---:|---:|---:|
| 1 | 0.941827 | 0.900034 | 0.589368 | 0.053915 |
| 2 | 0.711368 | 1.074832 | 0.637228 | 0.053606 |
| 3 | 0.630625 | 1.124876 | 0.676723 | 0.051974 |

因此最佳 checkpoint 为 epoch 1，且 route overfit 明确。正式 gate 为 **失败**：combined
patch Spearman `0.098661 < 0.1`；更关键的是 clean/reference score p95
`0.403447 > 0.2`。但最低可学习性证据成立：combined top-20 AUROC `0.753188`，当前
干预视角内 record median AUROC `0.701826`，三个 high-support presence AUROC 均超过
`0.6`，blur/dark 相对 reference 的 uplift 均超过 `0.01`。这表明模型能定位强 evidence
loss，但在 35 条独立训练路线、每条 16 个连续帧的数据上不能可靠校准未干预观测，不能
启动 full training、held-out family 或闭环扩展。

### 17.7 Single hidden-64 capacity diagnostic

当前 hidden=128 hurdle adapter 约 71 万参数；train 持续改善而 route validation 恶化，
不支持“容量不足”作为首要解释。为区分过大容量与数据覆盖问题，只冻结一个候选，而不跑
容量矩阵：hidden=64（约 24.5 万参数）、同一 seed/loss/split、只训练 1 epoch。

2026-08-27 11:51:13 CST，在 optimizer step 前冻结：

`configs/observation_uq_counterfactual_high_support_hurdle_hidden64_probe_v1.json`

config SHA256：
`f9e884f9365d8ccd9c72a6c05aa9b9eb1e7ce7bb49819a378a0355dd43def448`。

预先规定：只有当 clean/reference p95 相比 hidden=128 的 `0.403447` 至少下降 `0.05`，
同时 combined 与 within-view AUROC 各自下降不超过 `0.03`，hidden=64 才算值得保留；
否则停止容量枚举，优先增加独立 route 和时间跨度，并设计紧凑 feature shard。该 probe
不读取 glare/native/actual target，不自动触发任何后续训练或 Stage B。

job `1072081` 在 `gpu2` 完成，wall time `10:23`、batch MaxRSS 约 73.8 GiB、退出码
`0:0`。report SHA256 为
`b2fa74bfa367315578ae9327ea1a3525b6b58b84ba4d78c137e1bc47a9521e7e`。hidden=64 的
one-epoch train/route-validation total 为 `0.967803/1.047466`；正式 gate 仍失败。与
hidden=128 epoch 1 对比：

| metric | hidden 128 | hidden 64 | change |
|---|---:|---:|---:|
| clean/reference score p95 | 0.403447 | 0.507741 | +0.104294（恶化） |
| combined top-20 AUROC | 0.753188 | 0.723731 | -0.029457 |
| within-view median AUROC | 0.701826 | 0.685535 | -0.016292 |
| combined patch Spearman | 0.098661 | 0.077569 | -0.021092 |

因此 hidden=64 明确不满足预先规定的 p95 至少下降 `0.05`；容量缩小没有解决 clean
假阳性，并损伤相关性/定位。停止 32/256 或更多容量枚举。现有结果更符合：单观测输入
试图预测不可直接观测的 clean-paired counterfactual target，在仅 35 条独立 route 上会
把自然场景差异误判为 evidence loss。下一步优先扩大独立 route/时间覆盖；由于当前
40×40×1024 FP16 shard 已占约 65.9 GiB，必须先验证一个保留 40×40 空间分辨率的冻结
低维投影/紧凑 shard，不能直接线性扩展原格式。

### 17.8 Compact-input diagnostics: random projection rejected, int8 retained

为避免把 feature 压缩与 target 定义混在一起，先做不训练的 frozen projection audit。
Rademacher JL 投影固定 seed，原始 1024D paired target 始终在投影前计算。D=128 的
projection matrix SHA256 为
`e1bd22b6bb327dfc10628f10bfd203337d9c196f9083197b3a2320c06fb755b0`；audit job
`1072185` 完成，wall time `5:03`、batch MaxRSS 约 66.2 GiB。report SHA256 为
`c92f6e42b140fbd64e557d17de7dc24ee8780139caaff7809d0c8139e90785ad`。

投影对强响应区域的局部结构保留良好：within-view Spearman `0.96869`、top-20 AUROC
`0.99772`，magnitude/transient Spearman 分别为 `0.9988/0.99855`；但包含大量近零背景的
combined Spearman 仅 `0.6996`，最低 component Spearman 仅 `0.5537`。因此不允许在投影
后重新计算并替代原始 target。

随后只比较“原始 target + 投影后 adapter 输入”。D=128 one-epoch job `1072747` 的
report SHA256 为
`d311bd94f28d910346c803c829353b17d8af34704d831f4922a2fc1ef781c9af`；clean/reference
p95 降至 `0.225512`，但 combined/within-view AUROC 降至 `0.666629/0.661746`，Spearman
降至 `0.03136`。D=256 matrix SHA256 为
`a984ea77b5059ed0dee0787995dc6a6bc2856c14f5af8170ffc16d58c8cd6c23`；job `1073036`
report SHA256 为
`7466ad7c2955826ccb6ab2555524e4601aa2d45a9a54c920e59cabfa62feaf48`，combined AUROC
`0.692300`、within-view AUROC `0.684763`、Spearman `0.04806`，仍明显弱于未投影输入。
因此停止 D=384/512 和更多随机投影枚举；投影不作为当前数据扩展格式。

相同代码路径的 per-grid-per-channel dynamic symmetric int8 roundtrip job `1073038` 完成，
wall time `10:26`、batch MaxRSS 约 74.0 GiB、退出码 `0:0`。report SHA256 为
`51b57d2e5fed1089d6c51fb8b509a241316cc0d7ca858576df7e3d0bb7c4332a`。它相对 FP16
epoch-1 基线的结果近乎数值等价：

| metric | FP16 baseline | int8 roundtrip |
|---|---:|---:|
| train total | 0.941827 | 0.941832 |
| route-val total | 0.900034 | 0.900113 |
| combined Spearman | 0.098661 | 0.098649 |
| combined top-20 AUROC | 0.753188 | 0.753177 |
| within-view median AUROC | 0.701826 | 0.702410 |
| clean/reference score p95 | 0.403447 | 0.403409 |

这不改变 adapter 总 gate 失败的结论，但通过了压缩等价性比较；因此扩大独立路线时只将
int8 用作输入 feature 的存储格式，target 继续由原始 FP16 feature pair 计算后保存。平台
回归测试为 `18 passed`。

### 17.9 Audited 50-route expansion plan and storage cleanup

官方 Bench2Drive 数据目录快照包含 1000 个 archive 条目、总标称大小约 335 GB；其中
999 个文件名符合 `<Scenario>_<Town>_<Route>_WeatherN.tar.gz`，官方目录中的
`VehicleTurningRoutePedestrian_Town15_Route523_Weathe.tar.gz` 因 weather token 截断被明确
排除。目录快照保存在
`results/observation_uq_v3/b2d_official_api_catalog_20260827.tsv`，SHA256 为
`2e912c1393bfe9ab350a15ff471f6cb53fa6e3bb7ea404ade2b5e94aec760377`。

在任何下载以前，冻结 seed `20260827` 的 50-route expansion plan：保留原 50 条和原
35/5/5/5 split；新增 35/5/5/5，最终为 70 train、10 validation、10 calibration、10
held-out。官方目录恰有 5 条 Town11 路线，因此全部加入 held-out；现有 held-out Town02、
Town05 与新增 Town11 均禁止进入其他新增 split。其余路线按“当前场景族计数、非 held-out
城镇计数、天气计数、seeded SHA256”的顺序确定性选择；archive 大小只做预算，不参与
选择，避免偏向短路线。最终 43 个场景族中 30 个各 2 条、12 个各 3 条，ControlLoss 因
Town11 有两条官方路线而为 4 条；held-out town overlap 为 0。

plan 位于
`results/observation_uq_v3/b2d_expansion100_plan_seed20260827.json`，SHA256 为
`e8e4fb2573713d8c0daf4139d551bd0ae4388678ae20ff1dc70032890be0d6b5`；下载清单 SHA256
为 `b63482bc4ee105feb187f28c4e3a0b1763e28166f6e3595995e24f763a1a8b49`。新增 archive
预算为 `14,117,659,607` bytes（13.15 GiB）。若从当前 720 reference frame 扩至预定
1600 frame、输入 payload 使用已验证 int8，估算 feature payload 为 73.30 GiB；该估算
不含 target、scale、metadata 和文件系统开销。该 JSON 的 status 明确为
`pre_download_plan_not_a_training_manifest`；数据下载、解压、infos 重建和校验完成后，
必须从新 infos 重新生成正式 manifest，不能把 plan 直接喂给训练。

同日核对共享盘：`data/bench2drive_archives` 的 50 个官方 archive 与
`data/bench2drive/v1` 的 50 个解压目录一一对应，无缺项；archive 为可从官方源恢复的
冗余。使用 exact-scope cleanup 脚本删除这 50 个 archive，共 `17,573,960,485` bytes
（16.37 GiB），不删除任何解压路线。tombstone 位于
`results/observation_uq_v3/b2d_archives_redundant_cleanup_20260827.json`，SHA256 为
`bd47c32ef8e63dd5837e0a0cb1e7f24a7b38bd1e33ffbe03d5ca9980c474c4d4`。共享盘使用率从
82% 降至 77%，可用空间约 82 GiB。

Route214 actual-target 的 43 个原始 `.pt` 记录共 27.43 GiB，虽然已完成 run manifest 与
少量可视化，但尚未导出为保留全部任务退化信息的紧凑辅助监督。它们的重建需要完整
ORION 前向和约 192 GB 主机内存，因此本次不删除；先完成可验证的 compact export，才
能重新判断是否回收。

### 17.10 Lossless FP16 expansion amendment and formal extraction launch

用户随后明确允许把正式训练输入放入 500 GB 个人目录。按该新存储条件，2026-08-27
对 §17.9 的 int8 主路径作显式 amendment：int8 roundtrip 仍是通过等价性检查的备用格式，
但不再作为正式 adapter 输入。主路径改为完整 FP16 feature；dataset/CARLA/infos 留在共享
目录，按整条 route 分片的 feature 和预计算 FP32 continuous target 写入
`/public/home/lidachuan/orion_work`。这不是静默修改原计划，且不改变 target 定义、split、
corruption family 或 mask 的 audit-only 角色。

先用旧 65.92 GiB v2 单体 shard 完成一个真实 route 的 lossless conversion probe。
job `1073094` 在 `gpu2` 完成，wall time `3:25`、MaxRSS 约 65.2 GiB、退出码 `0:0`；输出
1 条 train route、16 clean、64 observed，route shard 为 `1,583,352,674` bytes。独立重载
验证 FP16 feature bitwise hash 和由源 FP16 计算的 FP32 target hash 均一致。probe manifest
SHA256 为 `adf39f056fc419b11fc8dfa8a6d49aab6974001ff47edace64a95ec8cab5ee0e`，状态明确为
`partial_probe_not_for_training`。

50 个新增官方 archive 随后通过 Hugging Face mirror 下载；下载回执 SHA256 为
`d1562f032aa3effdad3dec5dd7f5c3b8f4e030a09e45ab0162aeaea037edb1b3`。VLESS 仅获授权作为
速度异常时的备用通道，本次镜像速度正常，未启用代理。job `1073097` 对每个 archive
逐一校验下载 SHA/size，拒绝 path traversal、link/device 和错误顶层目录，解压到隐藏临时
目录并核对 tar 与落盘 file count/bytes 后原子改名。50/50 完成，wall time `20:35`、
MaxRSS `38156K`；新增落盘文件 `17,779,895,268` bytes。全部成功后才删除 50 个已验证
archive，共 `14,117,659,607` bytes。extraction receipt SHA256 为
`1334c6c996d6e7b0c216e398160b54ee32be273b7514d746e55eff2e8cb7d106`。

job `1073111` 用 8 个 CPU worker 从磁盘上的完整 100-route 集合重建新 infos，不覆盖旧
50-route infos，也不重复生成 4.43 GB map infos。作业 wall time `33:30`、MaxRSS
`728452K`、退出码 `0:0`；新 `b2d_infos_val.pkl` 为 `246,779,472` bytes，SHA256 为
`e50584ee0eb39df11068a726418117958a0b2f3e5cd82ba76fe20ca63814d1d3`。正式 manifest 只由
该 infos、原 manifest 和 frozen expansion plan 生成，不重新随机分配；结果严格为
70 train / 10 validation / 10 calibration / 10 held-out。held-out towns 为 Town02、
Town05、Town11，canonical route、folder 和 held-out-town overlap 均为空，全部 leakage
checks 通过。正式 manifest：

`configs/spatial_uq_route_manifests/b2d_expansion100_seed20260827.json`

SHA256：`9f2acaaf8b9ec291ac803bb3a014e880265f0399bcc22e3d1fdb66dd5a628fd3`。

从该 infos/manifest 冻结 expanded v3 protocol：

`configs/observation_uq_counterfactual_evidence_expanded_v3.json`

SHA256：`0c8c0ed1cc6fceccec67585fcd337f10982ff6904ef6a928bee8dc4736897bc0`。正式提取选择
70 train、10 validation、10 held-out route，每条首个 metadata-verified 连续 16 帧；
10 calibration route 完全保留，不参与本阶段。预期 1440 reference、5760 observed、
7200 个 `[6,40,40,1024]` FP16 grid；raw feature 为 `131.83594 GiB`，FP32 target 约
`0.618 GiB`，结合实测分片开销预计总计约 132--133 GiB。按个人目录面板当时约
174.5 GiB 可用空间，预计仍留约 41--42 GiB，不需要降低维度或量化。

正式 extractor 不再生成第二个 132 GiB 单体中间文件。它每完成一条 route 就：在 GPU
上由内存中的源 FP16 pair 计算 FP32 target；写入临时 shard；独立重载并校验 feature/target
hash；原子改名；更新可恢复的 partial manifest；只有 90 条全部完成才写 status=complete
的正式 manifest。中断续跑会重新校验所有已完成 shard SHA，且 extractor、storage module、
checkpoint、infos、manifest、protocol 任一 hash 改变都会拒绝 resume。平台相关回归测试
为 `20 passed`。

首次 Slurm job `1073123` 在 4 秒内因计算节点旧 Bash 对 empty array 与 `set -u` 的兼容
差异退出；MaxRSS 约 2 MiB，尚未创建 output root、未加载模型、未产生数据。修复并验证
空 resume flag 后，正式 feature-only job `1073124` 已在 `gpu2` 启动，请求 1×A800、
8 CPU、160 GiB RAM、4 小时。此前同路径 image-backbone extraction 实测 MaxRSS 约
82.1 GiB，route buffer/验证峰值预留后 160 GiB 足够；无需按完整闭环 ORION 的 192 GB
需求申请 240 GB。该作业仍明确记录 adapter training `0`、ORION finetuning `0`、Stage B
`0`。完成后只运行 train-only target/spatial/schedule audit；在审核结果以前不读取
held-out/native，不自动启动 adapter optimizer。

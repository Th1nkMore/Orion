# ORION 空间不确定性主线（2026-08-29）

## 当前唯一主线

当前闭环主线为：

`ORION 多视角特征 → 冻结的 Stage-1 空间 adapter → 因果空间校准 → Stage-2 与 ORION 任务上下文融合 → VLM/轨迹行为`

它不使用旧 Density UQ、全局 scalar UQ token、旧 vision adapter、FiLM、BEV cost 或 scalar speed governor。

新配置：

- `adzoo/orion/configs/orion_stage3_agent_spatial_stage2.py`
- `ORION_ENABLE_LEGACY_DENSITY_UQ=0`
- `ORION_CLOSEDLOOP_CONDITIONING=none`
- `ORION_CLOSEDLOOP_RISK_MODE=off`
- `ORION_STAGE2_SPATIAL_UQ_SOURCE=learned_adapter`

结果中出现 `Density=0` 或 `legacy_density_uq=False` 只是负向审计字段，表示旧方案没有运行，不是 Density 被重新启用。

## Stage 1：新训练的空间 adapter

冻结 checkpoint：

- 路径：`/public/home/lidachuan/orion_work/observation_uq_v3/runs/counterfactual_pairwise_native_repair_seed20260828_r1/counterfactual_evidence_pairwise_native_repair.pt`
- SHA256：`0555f0f341c80a88e18c5864573f0be0641fb828931bea7809e2f5544665f2c8`
- 输出：`[B,V,H,W,3]`，保留视角、空间位置、强度与时间变化证据。

Stage 1 只表达通用 observation-evidence uncertainty，不接收路线、actor、碰撞、corruption 类型或控制标签。在线运行采用每条路线前 60 帧的因果中位数/MAD 基线；基线未就绪前校准图严格为零。Stage-2 loss 在输入边界再次 `detach`，不能反向修改 Stage 1。

该 checkpoint 已通过 glare 留出测试，但 native heavy fog AUROC 约为 `0.5903`，低于预设 `0.6` 门槛。因此当前只能作为空间诊断/辅助输入，不能单独取得控制权，也不能声称已经得到语义正确的风险图。

## Stage 2：任务相关性与行为

Stage 2 同时接收：

- ORION 的 256 维 pre-LLM planning memory；
- ORION 的 89 维路线指令、当前/历史车辆状态和 ego-pose 任务上下文；
- 冻结 Stage 1 的空间 UQ 图。

它学习 on-path/off-path 相关性、未来冲突、`go/prepare_yield/hold/release` 状态和安全轨迹响应。监督来自 privileged actor occupancy、未修改的 ORION 候选轨迹和已验证的安全响应；不得使用 Density score、corruption family 标签或 Stage-1 UQ target。

结构保证：空间 UQ 全零时，conditioned context 与 ORION 原上下文逐位相等、轨迹残差严格为零、行为状态为 `go`。这条约束在训练后仍必须通过审计。

## Route147 的正确定位

Route147 braking-aware oracle v2（job `1088701`）只证明 privileged 准确信号下，响应机制能提高 walker TTC 并减少低 TTC 暴露；它没有运行 Density，也没有运行新 adapter。

当前采集使用 privileged expert 控制车辆，同时旁路记录新 Stage-1 map 和 ORION 上下文，用于构建 Stage-2 训练样本。未训练的 Stage 2 以严格 identity 接入，不应被表述成 learned-UQ 闭环结果。

- v1 job `1089220`：启动期主动取消，因为当时缺少 89 维任务上下文；无科学结果。
- v2 job `1089248`：CARLA 在第 38 帧同步停滞，watchdog 标记 runtime environment invalid；38 个基线前帧不得训练。
- v3 job `1089270`：2 Hz 采集、90 秒 CARLA RPC 上限、75 秒线程栈诊断、4 CPU、192 GiB；当前有效重试。

## 下一判据

v3 完成后先审计：

- Stage-1 checkpoint SHA 完全一致；
- 旧 Density、scalar conditioning、governor 全部关闭；
- 基线前记录严格为零；
- 基线后 map 有限、非负、非恒定；
- planning context、89 维 task context 和空间 map 同帧对齐。

随后只做标记为 non-claim 的 on-path/off-path/zero-UQ Stage-2A 辅助头优化 smoke，确认空间 projector 和 relevance/response head 至少能区分位置，并保持 zero/off-path identity。这个离线入口不训练 ORION VLM，也不训练 VAE/diffusion 轨迹解码器，产出的 checkpoint 被代码硬标记为 `closed_loop_eligible=false`，不得直接加载进闭环。

真正的 Stage-2B 仍需在 ORION 训练 forward 中，将 frozen Stage-1 map 注入 VLM memory，并以 safe trajectory、collision/TTC/traffic-rule loss 联合微调 Stage-2 projector、VLM（优先 LoRA）和 VAE decoder。只有 Stage-2B 的 checkpoint 才能被标记为 closed-loop eligible。Route147 glare 已经没有通过 failure-induction gate，因此它不能成为 learned-UQ 安全主结果；正式闭环仍需 adapter 未见过、能实际恶化安全或 TTC 的独立观测退化与多路线验证。

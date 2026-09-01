# Route214 真实 ORION actual-target 启动层（2026-08-26）

## 结论

已完成一个严格限界、默认不执行的生产启动层。它只覆盖 G1：在
`Town04/Route214` 帧 `0..63` 上，用冻结 ORION 对 clean/observed 两个分支各做 64 次前向，
并只持久化预注册的 43 个 measurement frame。它不启动 CARLA、不训练、不进入 Stage B，
Python 入口也不会提交 Slurm 作业。

当前没有由本启动层提交 GPU 作业；正式 replay 必须先让四项持久化 QA 证据通过。

## 文件与责任边界

- `scripts/run_orion_actual_target_route214.py`：CPU dry-run、真实模型初始化和限界 replay。
- `uq_estimator/orion_route214_production_integration.py`：唯一默认 factory，绑定精确
  `ProductionActualTargetBranchBuilderV1` 与四个具名 QA callback。
- `scripts/submit_orion_actual_target_route214.sh`：默认只打印 `sbatch`；仅 `--submit` 提交。
- `tests/test_orion_actual_target_route214_launch.py`：production markers、证据 lineage、固定
  corruption、设备准备、限界 sink、CLI 和 shell wrapper 回归测试。

启动层没有修改 runner、exporter 或 actual-target builder。

## 固定协议

- route/frame：`Town04/Route214`，`0..63`，按时间顺序。
- branch：先 clean，后 observed；同一 dataloader 重放一次。
- corruption：仅 `CAM_FRONT`，`local_occlusion`，severity 2，seed 20260826，窗口 `0..63`。
- forward：`64 × 2 = 128`；persisted paired records：严格 43。
- model：冻结、eval、关闭 diffusion；无 backward、optimizer、CARLA。

完整窗口只用于 failure-induction/actual-target smoke，不能作为 learned UQ 时间对齐证据。
runner 调用 unwrap 后的 ORION core、绕过 `MMDataParallel.scatter`，因此 branch provider 明确把
模型/GT tensor 和 box wrapper 移到 CUDA，仅 `img_metas` 留在 CPU。sink 对超额、重复和覆盖
全部 fail closed，并原子保存每条记录和 SHA-256 manifest。

## Production integration 与证据门

默认 factory：

```text
uq_estimator.orion_route214_production_integration:build_route214_production_integration_v1
```

factory 必须返回精确的 production builder 与 decoder parity、selected motion mode、projection
overlay、GT axis alignment 四个 callable。builder、corruption、sink、四个 callback 均须提供
非空 `production_hook_id`；QA callback 还须绑定实际 `evidence_path`。

QA JSON schema 为 `orion-route214-qa-evidence/v1`，绑定 plan ID、checkpoint/config SHA-256、
route folder、hook ID、非空且全 true 的 checks、generated_by，以及至少一个带 SHA-256 的
持久化 artifact。缺文件、陈旧 lineage、bool-only、缺 artifact 或 hash 不匹配都会返回 false，
runner readiness 保持关闭。launcher 不会自行伪造证据。

## 命令

本地 CPU 预检（不加载模型、不创建 output root）：

```bash
python scripts/run_orion_actual_target_route214.py \
  --dry-run --checkpoint-sha256 <Orion.pth 的小写 SHA-256>
```

只查看服务器提交命令：

```bash
bash scripts/submit_orion_actual_target_route214.sh --dry-run
```

真实模型/数据初始化、零帧：

```bash
bash scripts/submit_orion_actual_target_route214.sh --model-init-only --submit
```

四项证据和 source/file 校验通过后，正式单作业入口才是：

```bash
bash scripts/submit_orion_actual_target_route214.sh --submit
```

资源合同固定为 Nvidia_A800、1 GPU、8 CPU、220 GB RAM、2 小时。wrapper 会拒绝缺失路径、
正式执行时缺失的 QA JSON，以及已存在的 output root。

## 完整 LLM/map 加载理由

smoke 前向只取 perception/decoder/target 所需输出，不调用 VAE 解码或路线规划。然而 checkpoint
按完整 Stage-3 ORION 架构保存；生产初始化保留完整模型、LLM/Q-Former 和 map 依赖，避免用删改
架构得到不可比或宽松加载结果。历史完整 ORION 约占 192 GB host memory，所以申请 220 GB；
64 GB 已被 OOM 证明不足。

CPU dry-run 只能证明 factory 可导入、builder 类型和 markers 正确、计划为 128 forward/43 record，
以及没有训练/CARLA/自动提交；它不能证明数值 QA 已通过。真实 replay 仅在 source/file verification
与四项持久化证据同时通过时执行，否则 `assert_real_execution_ready` fail closed。

# Density-UQ Embedding 局部损伤定位预实验

## 实验目的

本实验用于回答中期报告中一个关键质疑：如果只是把一个 UQ score 或 embedding 注入给 LLM，模型是否真的知道这些信息对应视觉损伤或不确定区域，而不是只学会复述一个标量？

前一阶段的 Reliability QA 已经说明，LLM 可以读取 UQ 并输出整体视觉可靠性等级。但该结果主要验证的是“全局可靠性语义可读”，还不能说明 active embedding 中包含可定位的视觉损伤信息。因此本实验进一步构造一个更具体的局部任务：给定同一个场景的视觉 token 和 Density-UQ active embedding，要求 LLM 判断前视相机中最不可靠的区域位于左侧、中间还是右侧。

该实验不直接声称已经提升规划性能。它的定位是一个中间证据：验证 Density-UQ embedding 不只是一个全局风险标量，而是包含可被 LLM 对齐和读取的局部视觉不确定性方向信息。

## 方法思路

### 局部伪标签构造

由于目前没有人工标注的“图像损伤位置”标签，本实验使用 Density-UQ 对 EVAViT token 的敏感性构造伪标签。具体做法如下：

1. 对每个样本读取已提取的 EVAViT feature。
2. 在前视相机 token 网格上按局部区域进行遮挡或置零扰动。
3. 将扰动后的 feature 送入 Density-UQ estimator，计算密度不确定性变化。
4. 选择导致 UQ 变化最大的区域作为该样本的局部不确定性伪标签。
5. 为降低任务难度，将区域合并为三列：`left`、`center`、`right`。

这个标签不是人工真值，因此不能作为最终闭环安全结论。但它与 Density-UQ 的定义是一致的：如果遮挡某一区域会显著改变密度不确定性，则该区域可以被视为当前特征中与视觉可靠性最相关的位置。

### LLM 对齐任务

训练时保持 ORION 的主体结构不变，只训练轻量参数：

- UQ token projector；
- LLM LoRA。

输入包括：

- ORION 原始视觉 token；
- Density-UQ score；
- Density-UQ active embedding。

输出是一个受控文本答案：

```text
Most unreliable front-camera region is left.
Most unreliable front-camera region is center.
Most unreliable front-camera region is right.
```

评估时对同一个图像做 correct UQ 和 shuffled UQ 对照：

- `correct UQ`：注入该样本自己的 score 和 embedding；
- `shuffled UQ`：注入另一个样本的 score 和 embedding；
- 如果输出随 shuffled UQ 明显改变，说明模型确实受到注入 UQ 表征影响；
- 如果 correct UQ 在未见样本上超过多数类基线，说明 embedding 与局部伪标签之间存在可学习对齐关系。

## 实验配置

### 数据与模型

| 项目 | 配置 |
| --- | --- |
| 数据集 | Bench2Drive |
| 视觉特征 | 已缓存的 EVAViT tokens |
| UQ 模型 | Density-UQ estimator |
| 主模型 | ORION |
| 输出任务 | 前视相机三列局部不确定性定位 |
| 可训练参数 | UQ token projector + LLM LoRA |

### 关键文件

| 类型 | 路径 |
| --- | --- |
| 训练脚本 | `scripts/train_risk_qa.py` |
| 评估脚本 | `scripts/eval_risk_qa.py` |
| 伪标签脚本 | `scripts/build_uq_localization_labels.py` |
| 主要结果 | `results/risk_qa/localization_x_active_b100_s400_600_hold100.json` |
| calibration 结果 | `results/risk_qa/localization_x_active_b100_800_eval100.json` |
| train replay 结果 | `results/risk_qa/localization_x_active_b100_800_train_eval500.json` |

## 主要实验结果

### 受控同分布 held-out 实验

该实验从 500 个三列定位伪标签样本中按类别分层随机划分：

- 400 个样本用于训练；
- 100 个样本作为 held-out 测试；
- held-out 样本不参与训练，但与训练集来自同一批伪标签构造流程。

训练配置：

- `max_steps=600`
- `answer_style=localization_x`
- `use_active_embedding=True`
- 训练参数为 UQ projector 和 LLM LoRA。

结果如下：

| 方法 | 测试样本 | Parse Rate | Accuracy | 多数类基线 | Shuffled 回答改变率 | Shuffled 目标命中率 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| active embedding, held-out | 100 | 100.0% | 61.0% | 45.0% | 69.4% | 29.0% |

该结果说明，在同分布未见样本上，模型可以把 active embedding 与局部不确定性伪标签进行对齐，准确率超过多数类基线 16 个百分点。同时，shuffled UQ 会使 69.4% 的可比较样本改变输出，说明输出不是固定模板，也不只是由原始图像决定。

### 对比：train replay 与 score-only

为了判断 active embedding 是否比 score-only 更有价值，对已有 score-only 和 active embedding 结果进行对照：

| 设置 | Split | Parse Rate | Accuracy | 多数类基线 | Shuffled 回答改变率 |
| --- | --- | ---: | ---: | ---: | ---: |
| score-only | train replay 100 | 81.0% | 42.0% | 51.0% | 60.0% |
| active embedding | train replay 500 | 100.0% | 62.0% | 44.8% | 63.8% |
| active embedding | held-out 100 | 100.0% | 61.0% | 45.0% | 69.4% |

score-only 在训练回放中仍低于多数类基线，而 active embedding 在 train replay 和 held-out 上都超过多数类基线。这支持一个较谨慎但有价值的判断：局部定位任务不能仅靠全局 UQ score 解释，active embedding 提供了额外可学习信息。

### calibration split 结果

跨到 calibration split 后结果仍不稳定：

| 方法 | Split | Parse Rate | Accuracy | 多数类基线 | Shuffled 回答改变率 |
| --- | --- | ---: | ---: | ---: | ---: |
| active embedding | calibration 100 | 100.0% | 36.0% | 51.0% | 46.7% |

该结果说明，当前方法还没有形成稳定的跨场景泛化能力。可能原因包括：

- 局部伪标签本身噪声较大；
- calibration split 的场景分布与训练样本存在差异；
- 当前只用轻量 LoRA 和较少样本训练，模型容易学习到局部统计偏置；
- 三列标签过于粗糙，无法充分表达多相机和 BEV 空间中的真实风险位置。

因此，报告中不能把该实验表述为“已经可靠定位所有视觉损伤区域”。更合适的结论是：在受控同分布预实验中，active embedding 已经显示出可被 LLM 对齐的局部不确定性信息；跨场景泛化仍需要后续加强。

## 代表性样例

以下样例来自 held-out 结果。它们展示了同一图像在 correct UQ 与 shuffled UQ 下输出发生变化，说明注入的 UQ 表征会影响 LLM 对局部不可靠区域的判断。

| 样本 | Correct 目标 | Correct 输出 | Shuffled 目标 | Shuffled 输出 |
| --- | --- | --- | --- | --- |
| `ParkingExit_Town12_Route922_Weather12__00075.pt` | right | right | center | left |
| `ParkingExit_Town12_Route922_Weather12__00020.pt` | right | left | right | center |
| `ParkingExit_Town12_Route922_Weather12__00074.pt` | left | right | right | left |

第一个样例中，correct UQ 下模型正确输出 `right`；当注入 shuffled UQ 后，输出变为 `left`。这类案例可以用于解释：LLM 的回答确实被 UQ embedding 调制，而不是只依赖原始图像或固定回答模板。

需要注意的是，部分 shuffled 输出虽然发生变化，但不一定命中 shuffled 目标。这说明当前模型对 embedding 的读取具有可控性雏形，但还没有达到稳定精确定位。

## 对中期报告的写法建议

可以写入正文的结论：

1. Density-UQ 不仅可以给出样本级风险分数，还可以通过 active embedding 提供更细粒度的视觉不确定性表征。
2. 在受控同分布 held-out 实验中，LLM 经过轻量对齐后能够从 active embedding 中读取局部不确定性方向，三列定位准确率达到 61.0%，高于 45.0% 的多数类基线。
3. correct UQ 与 shuffled UQ 的对照显示，注入的 UQ 表征会显著改变 LLM 的局部不确定性判断，回答改变率达到 69.4%。
4. 与 score-only 对比，active embedding 的结果更好，说明局部定位信息不能充分由一个全局标量解释。

不应过度表述的内容：

1. 不能说该实验已经证明规划轨迹变得更安全。
2. 不能说模型已经能可靠识别任意真实图像损伤区域。
3. 不能把 Density-UQ 伪标签等同于人工标注真值。
4. calibration split 上仍未超过多数类基线，因此跨场景泛化需要作为后续工作。

建议在中期报告中将本实验放在“阶段性实验结果”或“方法有效性验证”部分，作为 Reliability QA 之后的进一步证据。它比单纯的全局可靠性问答更接近核心问题，因为它验证了 active embedding 的空间或局部信息价值。

## 后续改进方向

1. 扩大局部伪标签训练样本，并采用 route-disjoint 划分。
2. 从三列标签扩展到多相机或 BEV 网格标签。
3. 将伪标签构造从简单遮挡敏感性改为多扰动一致性评分，降低标签噪声。
4. 增加人工可视化样例，展示高 UQ 区域与真实图像损伤的对应关系。
5. 在风险描述任务中要求模型输出自然语言解释，例如“前方右侧区域视觉不稳定”，再连接到规划行为监督。

## 当前结论

该预实验提供了一个正向但仍需谨慎的阶段性结果：active embedding 中确实存在可被 LLM 学习和读取的局部视觉不确定性信息。在同分布 held-out 样本上，该信息能够支持超过多数类基线的局部定位，并且 correct/shuffled UQ 干预会显著改变输出。

这使得中期报告可以更有底气地说明：当前方法并非只是在注入一个人为设定的风险标量，而是在尝试把视觉骨干的分布不确定性表征提供给 LLM，并已经观察到该表征在语义层面的可读性和局部可控性。下一阶段需要解决的是跨场景泛化和规划行为监督，而不是重新证明 UQ token 是否可被 LLM 读取。

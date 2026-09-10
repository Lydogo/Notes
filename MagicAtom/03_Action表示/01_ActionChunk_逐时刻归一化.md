# Action Chunk 的逐时刻归一化

> 完整记录：[ActionChunk 逐时刻归一化完整记录](../99_详细记录/05_ActionChunk_逐时刻归一化_完整记录.md)

## 1. 结论

在 `chunk_delta` 表示下，chunk 内相对位置越远，累计位移通常越大、分布越宽。若所有位置共用一组 `[D]` 统计量，统计范围会被 chunk 尾部主导，前部的小信号被压缩。

因此将 normalizer 从 `[D]` 扩展为 `[H,D]`：每个 chunk 位置 `h`、每个动作维度 `d` 使用独立统计量。这里的 timestep 是 chunk 内位置，不是轨迹绝对时间，也不是 flow 的噪声时间。

## 2. 数学形式

```text
delta[l,h,d] = absolute_action[l+h,d] - current_state[l,d]
normalized[h,d] = normalize(delta[l,h,d], statistics[h,d])
```

当前 MagicVLA 实现使用每个 `(h,d)` 的 `q01/q99`，以兼容已有 bounded quantile normalization；参考论文使用 mean/std。两者不能混称。

## 3. 为什么有效

共享统计量会让近端位置集中在归一化零点附近，远端位置更容易触碰裁剪边界。逐位置统计主要重新分配 clipping，而不保证总 clipping 下降。

本地一次测量（RoboDojo、chunk size 50）显示：共享 quantile band 的 clipping 从前部约 0.46% 增至尾部约 4.06%；逐位置 band 约为 2.09% 到 1.89%；总 clipping 约 1.83% 与 1.88%，基本不变。这是分布改善证据，不是成功率提升证据。

## 4. 端到端契约

训练和推理必须使用同一组变换：

```text
absolute action/state
  → chunk delta
  → per-(h,d) normalize
  → policy target
  → inverse normalize
  → delta / absolute action
```

统计文件必须记录 `H`、`D`、归一化模式、q01/q99 或 mean/std、裁剪规则、来源数据集和版本。统计量不能跨不同 chunk 长度直接复用。

## 5. 当前实现状态

- 代码已支持逐位置统计，但开关默认关闭，现有训练 YAML 尚未启用。
- 训练统计链路已打通；部分 open-loop 可视化脚本仍按 `[D]` inverse normalization，需要同步修改。
- `mean_std` 模式的退化保护、VQA processor 的 shape guard、样本量随 `H×D` 分片等问题仍需检查。

## 6. 适用边界

适用于 chunk delta 在不同相对位置尺度明显变化的场景。若动作是 velocity、每步独立预测，或 chunk 内各位置分布近似稳定，逐时刻统计的收益可能很小。夹爪和 velocity 是否采用该方案，应按其物理含义和分布单独决定。

## 7. 验证清单

- 离线比较每个 `(h,d)` 的分布、均值、分位数和 clipping。
- 检查归一化与 inverse normalization 的数值可逆性。
- 固定数据、seed 和训练预算做 per-dimension / per-timestep A/B。
- 同时报告动作误差、边界触发率、闭环成功率和失败类型。
- 记录 checkpoint 中的统计量版本，防止训练和推理加载不同 artifact。

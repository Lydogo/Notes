# π0.5 Action-to-Vision Attention：诊断报告

> 完整报告：[Action-to-Vision Attention 完整报告](../99_详细记录/03_Action-to-Vision_Attention_完整报告.md)
> 代码附录：[Attention 诊断代码](02_Attention_诊断代码.md)

## 1. 目的

检查 action query 读取哪些视觉信息：相机路由、空间热点、action horizon 变化，以及视觉读取是否会改变动作。Attention 结果用于提出后续实验假设，不能单独作为信息因果归因。

## 2. 实验口径

| 项目 | 设置 |
|---|---|
| 模型 | `pi05_fold_box_normal_0731_e1_base_mix_disturbance` |
| episode | 466；全 episode 均匀采样 20 帧 |
| 相机 | `base_0_rgb`、`left_wrist_0_rgb`、`right_wrist_0_rgb` |
| 图像 | 224×224；14×14 patch；每相机 16×16 网格 |
| action | horizon 50；输出维度 32；query 0–49 |
| flow | 10 步去噪；默认 step 9（`t=0.1`）；seed 42 |
| attention | 8 query heads、1 KV head；汇总 layer 0/1/2/15/16/17 |

除特别说明，统计为默认层、8 heads 和 step 9 的平均。当前结论来自单个 episode，base 对照和不同指标使用独立口径。

## 3. 主要发现

1. **相机路由随阶段变化。** Query 25 在折外侧翻盖阶段偏左腕（图像 attention 内 51.8%），压盒贴胶带阶段偏右腕（51.9%）。
2. **深层偏向腕部局部区域。** Layer 15 的头部相机只占图像 attention 的 1.87%；部分热点位于 padding，不能直接解释为夹爪或接触点。
3. **远期 query 读取更多视觉。** image mass 从 query 0 的 5.58% 增至 query 25 的 8.40%、query 49 的 9.72%。
4. **去噪后期偏向 noisy-action suffix。** Step 9 的 raw attention 为视觉 / language-control / state / action = 8.33 / 21.55 / 23.01 / 47.12%；经 `W_O` 后为 5.41 / 17.40 / 15.42 / 61.77%。两组数字不能混用。
5. **干预能改变动作，但尚未证明提升任务成功率。** 视觉读取与动作变化相关，仍需闭环 A/B 实验。

## 4. 如何解释

按 key 分组的 attention 只说明当前层的读取位置。language、state 或 action suffix 的 hidden state 可能已编码视觉信息，因此不能把低 image mass 等同于“模型没有使用视觉”。同理，padding 热点首先是 mask 或 token 布局问题，不能直接当成感知失败。

## 5. 后续实验

- 在多个 episode、多个随机种子上重复相机路由和空间热点统计。
- 对 padding mask、相机 token、state token 做严格交换或遮挡干预。
- 对比 raw attention、`Attention×Value` 和 post-`W_O`，固定一个指标后再做跨实验比较。
- 将干预结果连接到动作误差、闭环成功率和接触阶段，而不是只比较 attention mass。

## 6. 图像与原始代码

常用图：

- [相机路由](../../Picture/attention_camera_routing_by_stage_query25.png)
- [逐层相机路由](../../Picture/attention_camera_routing_by_layer_group.png)
- [十帧 attention 总览](../../Picture/attention_ep466_10frames_overview.png)

完整可视化图集、指标定义和真实前向代码见原始报告及[代码附录](02_Attention_诊断代码.md)。

# Attention 诊断代码附录

> 完整代码记录：[Attention 诊断代码完整记录](../99_详细记录/04_Attention_诊断代码_完整记录.md)
> 主报告：[Action-to-Vision Attention](01_Action-to-Vision_Attention.md)

## 1. 代码作用

这部分代码在原有 Attention 前向中加入两类可选能力：

- 按静态配置提取 raw attention、`Attention×Value` 和 `post-W_O` 指标；
- 对指定视觉 token 做 attention 干预，用于比较动作变化。

普通路径不传 `diagnostics_spec` 或 `attention_intervention` 时，Q/K/V 和 KV cache 逻辑保持不变。代码不是独立脚本，必须结合 π0.5 的真实前向、token 布局和 checkpoint 使用。

## 2. 使用顺序

1. 先确认 token 区间、相机顺序、padding mask 和 action query 位置。
2. 再选择统计口径：raw、value 方向或 `W_O` 后输出。
3. 固定 layer、head、denoising step 和 query，再比较 episode 或阶段。
4. 干预后至少检查动作差异和闭环结果，不能只保存热力图。

## 3. 证据边界

代码记录保留完整实现片段和取数位置；不要把某个 episode 的诊断结果写成模型普遍规律。修改 Attention 接口后，应先验证未启用诊断时输出与原实现一致，再运行可视化实验。

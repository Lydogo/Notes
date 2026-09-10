# MagicVLA 短时视觉记忆：v0 与 v1

> 完整记录：[MagicVLA 视觉记忆完整实现记录](../99_详细记录/01_MagicVLA_视觉记忆_完整记录.md)

## 1. 结论

v0 和 `magicvla-mem-v1` 都是无状态的短时视觉记忆：每个样本显式携带固定历史窗口，模型内部不维护跨 episode 的 memory buffer。当前代码保留 v1，v0 作为对照。

## 2. 共同契约

| 记号 | 含义 | 当前设置 |
|---|---|---:|
| `B` | 每 rank batch size | v0=60，v1=48 |
| `V` | 相机数 | 3 |
| `K` | 每条序列的帧数 | 6 |
| `P` | 每帧 patch 数 | 256（16×16） |
| `H_v` | vision hidden | 1024 |
| `H_l` | merger / language hidden | 2048 |

采样配置为 `obs_steps=6`、`obs_interval_seconds=1.0`。若数据 FPS 为 `fps`：

```text
stride = max(1, round(fps × 1.0))
offsets = [-5×stride, -4×stride, -3×stride, -2×stride, -stride, 0]
```

三路相机分别处理；无效相机通过 mask 排除。语言侧 token 预算和视觉 patch 数必须在 batch 变换前后保持一致。

## 3. v0

v0 用 zero-gated temporal delta branch 注入历史变化信息。新增分支初始为零，使 base checkpoint 的初始行为近似不变，再通过训练学习历史帧的增量。

优点是改动小、易于从 base checkpoint 启动；缺点是时间建模能力受限，历史信息主要以差分形式进入，难以表达更复杂的时空关系。

## 4. v1

v1 用 time-then-space attention 处理历史视觉序列，并保持参数增量为零初始化。核心顺序是：

1. 在同一空间位置上融合不同时间帧；
2. 再在每个时间步内融合空间 patch；
3. 用随机 history mask 训练，使模型不会固定依赖完整历史。

v1 的目标不是把历史 token 直接拼到语言上下文，而是在视觉侧压缩时间信息后再进入 VLA。训练和 checkpoint 加载必须同时记录 memory 版本，否则同一组参数可能对应不同输入契约。

## 5. 训练与验证重点

- 检查 6 帧是否按真实数据 FPS 采样，而不是假定固定帧率。
- 检查三相机的 padding、无效相机 mask 和 batch reshape。
- 分别比较无历史、v0、v1，并报告单帧与历史输入的闭环成功率。
- 记录新增参数、zero-init 分支是否真正更新、checkpoint 是否包含 memory 配置。
- 将“attention 读到了历史”与“历史带来任务收益”分开验证。

## 6. 当前边界

这是短时、无状态 memory，不覆盖长期 memory bank、事件关键帧、文本摘要或跨 episode 记忆。原稿中的完整张量追踪、代码位置和验证记录见上方链接。

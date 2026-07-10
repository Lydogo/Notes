# Spatial Forcing vs LingBot-VLA 2.0：空间表征监督与应用型 VLA 系统的两条路线

> 对比对象：  
> 1. Spatial Forcing: Implicit Spatial Representation Alignment for Vision-language-action Model  
> 2. From Foundation to Application: Improving VLA Models in Practice / LingBot-VLA 2.0  
> 生成日期：2026-07-10  
> 目的：比较两篇论文对 VLA 真实能力瓶颈的不同解法，尤其是空间理解、数据效率、动作空间和部署复杂度。

---

## 一句话对比

Spatial Forcing 像是一把“空间表征手术刀”：它不改推理结构，只在训练时用 3D foundation model 逼 VLA 的视觉 token 学几何。

LingBot-VLA 2.0 像是一整套“应用型机器人系统工程”：它从数据、动作空间、MoE action expert、未来预测蒸馏和 benchmark 评估一起推进 VLA 落地。

## 核心差异

| 维度 | Spatial Forcing | LingBot-VLA 2.0 |
|---|---|---|
| 主要问题 | 2D VLA 缺少 3D 空间表征 | VLA 离真实部署还有数据、动作、动态预测差距 |
| 方法粒度 | 训练策略 / 表征监督 | 系统级 recipe |
| 核心机制 | 对齐 VLA 中间视觉 token 与 VGGT 空间 latent | 60k 小时数据 + 55 维统一动作 + MoE action expert + dual-query distillation |
| 3D/空间信息来源 | VGGT 作为训练期 3D teacher | LingBot-Depth 几何 teacher + action/data 覆盖 |
| 时序/未来建模 | 不是重点 | DINO-Video + future query 是重点 |
| 推理开销 | 基本无额外开销 | 模型系统更复杂，依赖 LingBot-VLA 2.0 架构 |
| 数据效率 | 明确强调少数据提升，真实任务 20/40 demos | 强调大规模数据质量和跨 embodiment 泛化 |
| 实验重点 | LIBERO、RoboTwin、少量真实机器人空间任务 | GM-100 generalist、长程移动操作、多平台 |
| 最强证据 | LIBERO 98.5 平均 SR + 训练/数据效率消融 | 长程移动操作 ID/OOD 均超过 π0.5 |
| 主要风险 | 依赖 3D teacher 质量，长程规划不是重点 | 成本高、模块耦合、success rate 仍偏低 |

## 共同趋势

两篇论文都在反驳一个朴素假设：只要 VLM 语义够强，接一个 action head 就能做好机器人控制。

它们共同指出，VLA 还需要更贴近物理世界的 inductive bias：

- Spatial Forcing 补的是空间几何 bias；
- LingBot-VLA 2.0 补的是 embodiment/action/data/future dynamics bias。

换句话说，下一代 VLA 不只是“看懂图 + 听懂话 + 输出动作”，而是要在内部表示里包含：

1. 物体和机器人之间的 3D 相对关系；
2. 不同机器人身体结构下的可控动作接口；
3. 当前动作对未来状态的影响；
4. 可跨任务复用的操作 primitive。

## 方法互补性

这两篇其实很适合组合，而不是互相替代。

LingBot-VLA 2.0 已经有 LingBot-Depth 和 DINO-Video 作为 teacher，偏 query-level 的 current/future perception distillation；Spatial Forcing 则更像 layer-wise visual token alignment。一个自然的扩展是：

- 在 LingBot-VLA 2.0 的视觉中间层加入 SF-style VGGT alignment；
- 保留 dual-query distillation 负责当前/未来语义和几何；
- 让 action expert 的 MoE 在更几何化的 visual token 上做路由。

潜在收益：

1. 对低数据机器人或新 embodiment 的空间泛化更强；
2. 对透明物体、高度变化、相对位姿任务更稳；
3. dual-query 学未来，SF 强化当前空间 token，两者监督位置不同，可能互补。

潜在风险：

1. 多 teacher 训练会增加工程复杂度；
2. VGGT、LingBot-Depth、DINO-Video 的表征目标可能冲突；
3. 需要仔细选择对齐层和 loss weight，避免视觉 token 被过度约束。

## 对后续研究的启发

1. **空间监督要进入中间表示，而不是只作为输入。**  
   Spatial Forcing 的关键价值是把“3D 能力”从输入模态问题变成 representation learning 问题。

2. **动作空间不是附属工程，它决定模型能学什么。**  
   LingBot-VLA 2.0 的 action target、action space、normalization 消融说明，控制表示本身会显著影响最终成功率。

3. **数据规模和数据质量同等重要。**  
   LingBot-VLA 2.0 花大量篇幅讲筛选、对齐、标注，而不是只报“收了多少小时”。这很实际：坏轨迹会直接污染 action distribution。

4. **未来预测可能成为应用型 VLA 的标配。**  
   如果机器人任务有移动、长程、动态场景，只对当前帧做反应式控制大概率不够。future query / video teacher / world model 会越来越重要。

5. **progress score 与 success rate 的 gap 值得重视。**  
   LingBot-VLA 2.0 多个任务 progress 高但 success 不高，说明模型已经能完成许多子步骤，但缺最后闭环纠错。未来评价应同时看 partial progress 和 full completion。

## 个人判断

如果目标是快速增强现有 VLA 的空间能力，Spatial Forcing 更值得先试：改动小、推理成本低、实验闭环干净。

如果目标是建设真实机器人基础模型平台，LingBot-VLA 2.0 的路线更接近工业落地：它把数据、动作、模型容量和长程评估都纳入同一个系统。

我的直觉是：这两条线最后会合流。应用型 VLA 系统会越来越复杂，但其中的每个关键表征都需要类似 SF 的“局部强监督”来降低学习难度。大系统负责覆盖世界，小机制负责让模型别在关键物理细节上瞎猜。

# RoboTTT：用测试时训练把机器人策略上下文扩展到 8K 时间步

> 原标题：RoboTTT: Context Scaling for Robot Policies  
> 作者：Yunfan Jiang, Yevgen Chebotar, Ruijie Zheng, Fengyuan Hu, Yunhao Ge, Jimmy Wu, Tianyuan Dai, Scott Reed, Li Fei-Fei, Yuke Zhu, Linxi “Jim” Fan  
> 机构：NVIDIA；Stanford University；The University of Texas at Austin  
> 发表：arXiv:2607.15275v1，2026-07-16  
> 链接：https://arxiv.org/abs/2607.15275  
> 项目页：https://research.nvidia.com/labs/gear/robottt/  
> 开源：论文/项目页提供模型说明与演示；代码链接未明确给出

---

## 一、研究背景与动机

当前机器人基础模型大多只看当前观测，或只使用 2–8 帧短历史。这对单步反应尚可，但会在多阶段装配中遇到三个问题：无法保留很久以前的任务进度，无法利用人类演示或自身失败历史，也无法在物体暂时遮挡后恢复完整状态。

直接把长历史拼接进 Transformer 会带来固定上下文上限和随历史增长的计算/显存开销；普通 RNN 或线性记忆又可能缺少从密集、重复的机器人流中提取结构的能力。RoboTTT 的目标是让策略在保持固定推理状态和近似固定延迟的同时，学习并利用最多 8K 个时间步的视觉-运动上下文。

## 二、核心贡献

1. **把 Test-Time Training（TTT）引入 VLA 策略**：用测试时通过梯度更新的 fast weights 作为循环状态，把历史压缩到参数空间，并在每一步读取。
2. **提出可训练长上下文的配方**：sequence action forcing 为每个 action chunk 独立采样 flow-matching 噪声，TBPTT 截断梯度但跨段传递 fast weights。
3. **提出两种从上下文学习的训练方式**：对人类视频遮蔽动作损失以实现 one-shot imitation；在 DAgger 中把机器人失败动作当 context、把人类修正当 target，蒸馏出在线纠错过程。
4. **给出真实双臂长程任务证据**：在 YAM 双臂平台的三种装配任务上，平均完成分数 79%，相对单步 GR00T N1.7 的 42% 提升 87%；8K 预训练上下文达到 71.5%，比 1K 的 43.9% 高 63%。

## 三、方法原理

### 3.1 整体框架

RoboTTT 以预训练 GR00T N1.7 为基础。输入是语言指令、每一步的多视角图像、proprioception 和带噪 action chunk；输出是未来 H 步动作。VLM 先产生视觉-语言 token，DiT action head 在每个时间步内做 attention，再沿时间维度通过 TTT 层更新和读取 fast weights。

与 Transformer 缓存全部历史 token 不同，RoboTTT 的历史状态是固定大小的 fast model 参数 `W`。因此上下文越长，状态大小不变；代价转化为每一步一次小模型梯度更新。

### 3.2 关键技术细节

**TTT 的 update-then-apply。** 对当前 token 投影出 key/value/query，先用 key-value binding 损失更新 fast model：

```text
W_t = W_{t-1} - η ∇_W ||f_W(K_t) - V_t||²
O_t = f_{W_t}(Q_t)
```

直观上，第一步把当前上下文写入参数，第二步从刚更新的参数中取出对动作预测有用的信息。`W0`、Q/K/V 投影和学习率由外层动作损失共同学习，因此 fast-weight 更新规则会适应机器人轨迹，而不是通用记忆模块的固定规则。

**插入位置与 token 流。** TTT 层放在 DiT 的 self-attention 和 cross-attention 之后：attention 处理单个时间步，TTT 处理跨时间信息。每个时间步包含 VL token、proprioception token、带噪动作 token 和 16 个 register token。为节省计算，完整 VL token 不进入 TTT，只用 16 个 register token 携带视觉语言信息跨时间传播。

**门控初始化。** 为保留 GR00T 的已有能力，在每个 DiT 层加入近零初始化的 `tanh(α)` 门控：

```text
O = tanh(α) ⊙ O_TTT + O_attn
```

训练初始时 TTT 分支几乎关闭，随后只在有助于任务时逐步打开。

**Sequence action forcing。** 每个时间步的 action chunk 独立采样 flow-matching 噪声：

```text
A_t^τ = τ A_t + (1-τ) ε,   ε ~ N(0,I)
```

如果整条序列共享同一个噪声水平，整段数据会同时“容易”或“困难”，训练不稳定。独立采样让一条序列同时包含不同难度的动作去噪目标。论文给出的采样为 `τ_t=s(1-u)`，`u~Beta(1.5,1)`，`s=0.999`。

**TBPTT。** 长序列被切成固定长度 segment，梯度只在 segment 内传播，但 fast weights 跨 segment 继续传递；边界处只 detach 梯度，不重置状态。因此显存主要由 segment 长度决定，而不是总上下文长度。

**从上下文中学习。** 通过 mask flow-matching loss，可以把部分时间步变成“只更新 fast weights、不提供动作监督”的纯上下文：

- 人类视频：视频段只更新 fast weights，后接的机器人轨迹承担动作损失；测试时同一语言指令下，视频提供未见配置的信息。
- DAgger Distillation：整条交互轨迹都更新 fast weights，但只在人类 correction 上计算动作损失。机器人错误是条件，人类修正是目标，因而学习的是“出现这种失败后如何纠正”，而不是孤立模仿修正动作。

### 3.3 训练与优化

| 阶段 | 数据/设置 | 可训练部分 | 训练量 |
|---|---|---|---:|
| 长上下文预训练 | 桌面双臂机器人数据 + 第一视角人类视频，强调长轨迹 | 只训练 TTT/GDN 等新增序列模块，冻结 GR00T 其余部分 | 30K steps，16×GB200 |
| 任务后训练 | 各下游任务机器人数据，1K context | 全参数 fine-tuning | 20K steps，8×GB200 |

优化器为 AdamW，weight decay `1e-5`；预训练使用 WSD，峰值学习率 `2e-5`；后训练使用 cosine，峰值 `5e-5`。预训练上下文从 128 逐步增加到目标长度（最高 8K）。4K 及以下 global batch size 为 64，超过 4K 为 16；后训练 per-device batch size 为 1。

### 3.4 数据使用与维度追踪

| 数据源 | 规模 | 样本单位 | 模态/字段 | 关键维度 | 用途阶段 | 处理方式 |
|---|---:|---|---|---|---|---|
| 桌面双臂机器人数据 | 论文未报告总量；下游任务各 8/6/5 小时 | 完整 trajectory 或 contiguous sub-trajectory | 语言、四路 RGB、proprioception、action chunk | 4 路 480p RGB；30 Hz；state/action 具体维度未说明 | 预训练/后训练/评估 | 强调长轨迹；按 context 截取 |
| 第一视角人类视频 | 预训练混合数据，规模未报告 | 视频 trajectory | 人类视觉流；在 one-shot 实验中与同配置机器人轨迹配对 | 分辨率、帧率、相机数未说明 | 预训练；视频上下文训练 | 视频段 mask 动作损失，只更新 fast weights |
| DAgger 交互数据 | 100 条（RoboTTT 50 + GR00T 50） | 含机器人动作与人类 correction 的交互 trajectory | 语言、观测、机器人动作、人类修正动作 | 具体动作维度未说明 | Pup Go Car 后训练/纠错 | 全历史更新 fast weights，仅 correction 计算 imitation loss |
| 扰动数据 | 30 分钟 | 受扰动 rollout | 观测、动作、被移除部件后的恢复轨迹 | 具体维度未说明 | 与任务数据共同训练 | 训练机器人在 episode 内利用历史恢复 |

**维度快照**

- Observation：四个 RealSense D405，相机位置为 top、bottom、left wrist、right wrist；480p RGB；30 Hz。
- Language：每条 trajectory 共享一个语言指令；Circuit one-shot 实验统一使用 “assemble circuit”。tokenizer 和最大 token 长度未说明。
- State：proprioception token；字段和维度未说明。
- Action：每步预测 H-step action chunk；动作空间、H、绝对/相对坐标系、旋转表示、夹爪编码和控制后处理均未说明。
- Context：最高 8K timestep，约 30 Hz 下五分钟；后训练使用 1K timestep。

数据流的关键不是数据规模本身，而是“哪些片段产生监督”：普通机器人轨迹每一步有 flow-matching 动作目标；人类视频只有上下文作用；DAgger 中失败机器人动作也只有上下文作用。论文报告预训练数据由双臂机器人和人类视频混合组成，并强调长轨迹，但没有给出两者的数量比例、过滤规则、同步/归一化细节或采样温度，复现时仍有明显信息缺口。

## 四、实验与结果

### 4.1 实验设置

平台为 YAM 双臂桌面机器人，四路 RGB 相机。任务为 Pup Go Car（平均 2 分钟）、Circuit（平均 1 分钟）和 Gear Bot（平均 5 分钟、10 个阶段）。Circuit 约 80 种配置，训练 20 种、测试剩余 60 种。普通任务每个方法 20 次 rollout，Gear Bot 10 次；指标为完全成功次数和按任务 rubric 归一化的完成分数。

对比包括：GR00T N1.7 单步上下文、GR00T N1.7 Hist.（增加一帧历史）和 GDN（用 Gated DeltaNet 替换 TTT，层位置、门控和参数量匹配）。

### 4.2 主要结果

| 方法 | Pup Go Car 完全成功 | Circuit 完全成功 | Gear Bot 完全成功 |
|---|---:|---:|---:|
| RoboTTT | 9/20 | 13/20 | 2/10 |
| GR00T N1.7 | 3/20 | 3/20 | 0/10 |
| GR00T N1.7 Hist. | 0/20 | 8/20 | 0/10 |
| GDN | 3/20 | 8/20 | 0/10 |

主要结果是 RoboTTT 平均 completion score 79%，而单步 GR00T 为 42%、最佳 GDN 为 56%。Gear Bot 是最强证据：RoboTTT 完成 2/10，所有基线均为 0/10。作者观察到 fast weights 能保留装配阶段、在钻孔失败后重新对齐，也能利用遮挡前的历史信息完成精细插入。

**上下文 scaling：** RoboTTT-8K 达到 71.5%，同架构 1K 仅 43.9%，相对提升 63%；相比最佳短上下文基线 45.6% 高 57%。从 128 到 8K 持续上升，尚未出现饱和；GDN 没有同样趋势。这说明收益来自“经过长上下文训练的 TTT 更新动力学”，而不是把更长历史简单喂给模型。

**One-shot human video：** Circuit 未见配置、同一语言 prompt 下，RoboTTT completion score 65%，完全成功 6/10；GDN 为 33%，0/10。

**扰动恢复：** Pup Go Car 中人为移除车顶/轮胎后，RoboTTT 分别恢复 15/20 和 18/20；单步 GR00T 为 10/20 和 11/20。

**DAgger Distillation：** 相同 100 条 DAgger 数据上，标准 DAgger 平均提升 9%，而 DAgger Distillation 平均提升 33%；RoboTTT 提升 36%，GDN 提升 29%。把失败机器人动作加入训练目标而不是仅作为 context 并不能带来额外收益，支持“失败是条件、修正是监督”的解释。

### 4.3 消融实验

- 去掉 sequence action forcing 后闭环性能明显下降，动作不准确到无法推进任务，说明独立噪声采样是稳定长序列 flow matching 的关键工程细节。
- 把 fast model 从两层 MLP 换成线性层后性能下降；线性 TTT 仍优于 GR00T，但比 MLP 低约 27%，说明非线性 fast weights 对密集历史压缩更有表达力。
- 在逐步构建中，加入 action tokens 带来 23% 相对提升，加入 register tokens 再带来 18% 相对提升。单独给 GR00T 增加 register tokens 没有帮助，说明 register 的收益依赖 TTT 的时间建模。
- 直接拼接历史并不可靠：GR00T Hist. 在 Pup Go Car 上为 39.5%，低于无历史版本的 57%，体现出时间分布偏移和伪相关问题。

## 五、局限性与展望

作者明确指出：长上下文预训练成本更高；当前 TTT loss 仍是通用 key-value binding，尚未针对机器人设计专门目标；RoboTTT 仍不能处理部署中的所有失败模式，未来可结合直接优化任务成功率的强化学习。

从部署角度看，TTT 还引入了每步梯度更新、学习率和数值稳定性的额外风险。论文证明了 8K context 的效果，但没有完整报告端到端延迟、每步更新耗时、显存、失败后状态重置策略，也未公开关键动作/状态维度和数据混合比例。实验集中在单一 YAM 双臂平台和三类装配任务，跨本体泛化尚未验证。

## 六、灵魂三问

1. **它解决了什么问题？**  它解决的是机器人策略无法有效使用长视觉运动历史的问题：普通 Transformer 历史成本随长度增长，短上下文策略又无法理解阶段进度、演示配置和自身失败。

2. **为什么这么做？**  TTT 用可梯度更新的 fast weights 取代 KV cache，把历史压缩成固定大小的非线性模型状态；配合 sequence action forcing 和 TBPTT，才能在训练中真正看到 8K 序列，并学习长程更新动力学。

3. **什么证据最有说服力？**  最干净的证据是预训练 context scaling：同一 RoboTTT 从 1K 到 8K 的闭环分数由 43.9% 升到 71.5%，且 GDN 没有同样趋势；Gear Bot 2/10 对基线 0/10 则证明收益能落到五分钟、十阶段实机器人任务。

## 七、个人总结

1. RoboTTT 的核心不是“把历史帧塞进 VLA”，而是让一个小型 fast model 在部署中持续学习，将历史变成可检索的参数状态。
2. 最大优势是把长上下文、one-shot 条件、在线纠错统一到 loss masking 框架；最大弱点是训练成本和复现信息不足，且 TTT 的在线梯度更新仍需严格评估实时性与安全性。
3. 对 VLA 数据工程的启发是：长轨迹不只是更多样本，也可以训练“如何更新记忆”的动力学；失败动作如果被当作 context、修正动作当作 target，通常比把所有动作混成 imitation target 更有信息量。

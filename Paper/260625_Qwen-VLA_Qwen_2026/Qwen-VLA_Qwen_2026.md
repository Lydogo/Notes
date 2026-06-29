# Qwen-VLA：把操作、导航与轨迹预测统一成一个 VLA 通用策略

> 原标题：Qwen-VLA: Unifying Vision-Language-Action Modeling across Tasks, Environments, and Robot Embodiments  
> 作者：Qwen Team  
> 机构：Qwen / Alibaba 相关团队  
> 发表：arXiv:2605.30280v2，2026-06-01；PDF 首页日期 2026-06-02  
> 链接：https://arxiv.org/abs/2605.30280  
> 项目：https://qwen.ai/blog?id=qwenvla  
> 开源：https://github.com/QwenLM/Qwen-VLA  

---

## 一、研究背景与动机

具身智能里的 VLA 模型正在从单任务策略走向通用策略，但现有系统通常仍被任务族、机器人形态和评测环境切开：manipulation 模型面向桌面或双臂控制，navigation 模型面向 VLN waypoint 或离散移动，egocentric human data 又提供手腕/手部轨迹而不是机器人控制量。表面上这些任务的观测、动作维度、控制频率、预测 horizon、评价协议都不同。

Qwen-VLA 的核心判断是：这些任务共享一个计算结构，即给定视觉观测、语言指令和 embodiment 约束，预测未来动作或轨迹。因此论文试图把 manipulation、vision-language navigation、egocentric trajectory、autonomous-driving-style trajectory prediction 和 VL understanding 放进一个统一的 action-and-trajectory prediction 框架。

这篇论文的目标不是只做一个更强的机械臂 policy，而是验证一个更大的命题：能否用一个 VLM backbone 加连续动作生成头，在多任务、多环境、多机器人形态之间共享视觉 grounding、空间推理和动作先验。

## 二、核心贡献

1. **提出 Qwen-VLA 通用 VLA 架构**：基于 Qwen3.5-4B multimodal backbone，接入约 1.15B 参数的 DiT-based flow-matching action decoder，用同一模型处理 manipulation、navigation 和 trajectory prediction。

2. **用 embodiment-aware prompt 统一多机器人控制语义**：每个样本前置文本 prompt，描述机器人平台、单双臂、控制频率、action chunk length 和控制 convention；模型不为不同 embodiment 设计独立 head，而是用统一张量、zero padding 和 mask 处理不同动作维度。

3. **构建大规模异构预训练混合数据**：预训练数据包含机器人 manipulation 轨迹、egocentric human trajectories、navigation trajectories、合成仿真轨迹、spatial grounding、自动驾驶 VQA 和通用 VL 数据。其中 robot manipulation 占 74.2%，human egocentric 占 6.0%，navigation 占 7.5%。

4. **提出分阶段训练 recipe**：先做 text-to-action DiT pretraining（T2A）建立语言条件动作先验，再 CPT 做视觉 grounding，之后 SFT 做任务对齐，最后用 SimplerEnv 中的 sparse success reward 做 RL，得到 Qwen-VLA-Instruct。

## 三、方法原理

### 3.1 整体框架

Qwen-VLA 输入视觉上下文、语言指令、embodiment 描述和可选任务标识，输出一个未来动作/轨迹序列。论文把目标写作：

```text
p_theta(y_{t:t+H-1} | o_t, x, e, z)
```

其中 `o_t` 是图像/视频/历史窗口，`x` 是任务指令，`e` 是 embodiment prompt，`z` 是任务类型。对 manipulation，它预测 EEF、joint、gripper 或 dexterous hand action；对 navigation，它预测地面平面上的相对位移和航向变化；对 human egocentric data，它预测手腕 SE(3) 运动和 hand articulation。

模型由两部分组成：

| 模块 | 作用 | 关键细节 |
|---|---|---|
| Qwen3.5 VLM backbone | 视觉-语言理解、referential grounding、空间推理 | early vision-language fusion，支持图像/视频/文本 |
| DiT action expert | 连续动作生成 | flow matching，16 个 DiT block，约 1.15B 参数，少量 Euler steps 推理 |

### 3.2 关键技术细节

**Embodiment-aware prompt conditioning** 是最重要的工程选择。prompt 模板会写明机器人类型、单双臂、是否有腰/底盘、控制频率、需要预测多少个 action step。它把不同机器人平台的控制语义从模型结构转移到文本条件里。

**统一 action tensor 但不强行统一物理语义**。每个样本都有 `H x K` 的目标张量，实际动作维度 `c <= K` 放在前 `c` 个 channel，剩余位置 zero padding，并用 mask 排除 padding loss。这样模型共享 DiT 参数，但不同数据集仍保留原始 action convention 和 dataset-specific normalization。

**T2A 把 action decoder 当成语言到动作的 decompressor**。作者认为一句任务指令和 embodiment prompt 是高压缩语义，而机器人动作是长序列、高维、连续控制信号。T2A 阶段冻结 VLM，只用文本训练 DiT 重建动作，使 decoder 先学到语言索引的动作先验，再进入视觉 grounding。

**VL co-training 防止 VLM 能力退化**。训练目标同时包含 flow-matching action loss 和 next-token VL loss。作者特别强调这能维持 spatial grounding、referential grounding 和 instruction following，对复杂 object recognition 与 compositional instruction 有帮助。

### 3.3 训练与优化

训练分四个阶段：

| 阶段 | 训练内容 | 目的 |
|---|---|---|
| T2A | 冻结 VLM，只训练 DiT；输入文本和 embodiment prompt，不输入图像 | 建立 language-conditioned action prior |
| CPT | 解冻 VLM 与 DiT，在异构 embodied + VL 数据上继续预训练 | 把动作先验 grounded 到视觉观测 |
| SFT | 多任务 SFT 与 real-robot ALOHA 分支 | 对齐下游任务与真实机器人部署 |
| RL | 在 SimplerEnv 用 sparse binary success reward 微调 | 优化 closed-loop task success |

T2A 消融显示：约 20% synthetic + 80% real 的 full-sequence text-action 数据最好，Simpler-WidowX SFT success rate 达 71.1%，比无 T2A 的 60.9% 高 10.2 个百分点。T2A 中加入图像反而降低效果，说明这个阶段确实应逼迫 decoder 学语言-动作映射，而不是依赖视觉 shortcut。

## 四、实验与结果

### 4.1 实验设置

评测覆盖四类能力：

| 类别 | Benchmark / 平台 | 关注点 |
|---|---|---|
| 仿真 manipulation | LIBERO、RoboCasa-GR1、Simpler-WidowX、RoboTwin 2.0 | 多 embodiment 通用 manipulation |
| 真实 manipulation | ALOHA 双臂平台 | in-domain 和 OOD 泛化 |
| navigation | VLN-CE R2R / RxR Val-Unseen | 连续视觉语言导航 |
| OOD manipulation | SimplerEnv-OOD、DOMINO | 静态空间泛化和动态物体 manipulation |

### 4.2 主要结果

**仿真 manipulation：一个 generalist 接近或超过多个 specialist。**

| 方法 | 类型 | LIBERO | RoboCasa-GR1 | Simpler-WidowX | RoboTwin-Easy | RoboTwin-Hard |
|---|---|---:|---:|---:|---:|---:|
| π0 | Specialist | 94.4 | - | - | 65.9 | 58.4 |
| StarVLA-OFT | Specialist | 96.6 | 48.8 | 64.6 | 50.4 | - |
| GR00T N1.6 | Specialist | 97.2 | 49.9 | 63.2 | 47.6 | - |
| π0.5 | Specialist | 97.6 | 37.0 | 46.9 | 82.7 | 76.8 |
| ABot-M0 | Specialist | 98.6 | 58.3 | - | 86.0 | 85.0 |
| Qwen-VLA-Base | Generalist | 90.8 | 40.4 | 64.3 | 64.3 | 66.4 |
| Qwen-VLA-Instruct | Generalist | 97.9 | 56.7 | 73.7 | 86.1 | 87.2 |

要点：Qwen-VLA-Instruct 虽然是单一 generalist，在 Simpler-WidowX、RoboTwin-Easy/Hard 上超过主要 specialist；在 LIBERO 基本达到顶尖 specialist 水平。

**真实 ALOHA：预训练贡献非常明显。**

| 模型 | Pick and Place | Table Cleaning | Bowl Stacking | Bowl Pick & Place | Towel Folding | Fine-grained | Avg. |
|---|---:|---:|---:|---:|---:|---:|---:|
| GR00T N1.6 | 30.8 | 38.5 | 53.8 | 19.2 | 19.2 | 10.3 | 28.6 |
| π0.5 | 73.1 | 84.6 | 88.5 | 69.2 | 80.8 | 33.3 | 71.6 |
| Qwen-VLA w/o pretrain | 30.8 | 53.8 | 61.5 | 64.1 | 50.0 | 30.8 | 48.5 |
| Qwen-VLA w/ pretrain | 96.2 | 92.3 | 98.7 | 87.2 | 65.4 | 61.5 | 83.6 |

OOD 真实任务平均成功率从 scratch 的 36.2% 提升到 pretrained 的 76.9%。这说明主要增益不是 architecture 本身，而是大规模预训练带来的视觉语言和动作先验。

**导航：统一 VLA 仍能保持 VLN 能力。**

| 方法 | R2R OS | R2R SR | R2R SPL | RxR SR | RxR SPL |
|---|---:|---:|---:|---:|---:|
| NaVILA | 62.5 | 54.0 | 49.0 | 49.3 | 44.0 |
| StreamVLN | 64.2 | 56.9 | 51.9 | 52.9 | 46.0 |
| Qwen-VLA-Base | 61.7 | 53.8 | 49.4 | 55.1 | 45.8 |
| Qwen-VLA-Instruct | 69.0 | 57.5 | 51.2 | 59.6 | 47.8 |

Qwen-VLA-Instruct 在 R2R/RxR 的 SR 上最好，说明 manipulation 和 navigation 的 joint training 没有把导航能力完全牺牲掉。

**OOD manipulation：动态任务上最能体现预训练广度。**

| Benchmark | 指标 | Qwen-VLA-Base | Qwen-VLA-Instruct |
|---|---|---:|---:|
| SimplerEnv-OOD | Avg. SR | 25.3 | 32.0 |
| DOMINO dynamic manipulation | SR | 21.1 | 26.6 |
| DOMINO dynamic manipulation | MS | 37.4 | 39.5 |

DOMINO 上 Qwen-VLA-Instruct 是 zero-shot 到动态 manipulation，仍达到 26.6% SR / 39.5 MS，并超过若干在 dynamic manipulation data 上 fine-tune 的方法。这个结果是论文里最强的泛化证据之一。

### 4.3 消融实验

| 消融点 | 结果 | 解释 |
|---|---|---|
| T2A 数据配比 | 20% synthetic + 80% real 最好，71.1% | synthetic 扩大语言-动作覆盖，real 锚定物理动态 |
| T2A 预测方式 | full sequence 优于 chunk | 长序列能学完整任务结构和终止模式 |
| T2A 是否输入图像 | 输入图像降低表现 | 图像 shortcut 会削弱语言-动作先验 |
| VL+VLA co-training | RoboCasa +4.9 pp，RoboTwin +4.6 pp | VL 数据帮助复杂物体识别和组合指令解析 |
| projection 设计 | Multi-MLP、Concat、Zero-Pad 差异很小 | zero padding 参数最少，因此默认采用 |
| RL post-training | Simpler 70.8 -> 73.7，DOMINO SR 25.7 -> 26.6 | RL 主要提升执行果断性，并没有明显遗忘其他任务 |
| state conditioning | RoboTwin 增益最多约 1.3 pp | 多视角视觉和相对动作已足够，显式 proprioception 性价比不高 |

## 五、局限性与展望

作者明确指出，embodied action data 的规模和多样性仍远小于 VL data，长尾物体、复杂接触、真实环境长时程部署仍是瓶颈。另一个问题是多目标 joint training 会带来优化 trade-off：action-oriented training 可能让纯 VL 或 navigation 指标有轻微回退，需要更好的 objective balancing、curriculum 或 modular specialization。

我的补充判断是：Qwen-VLA 的统一性更像是“接口统一”和“参数共享”，而不是物理语义真正统一。不同数据集仍保留 native action convention，靠 prompt 和 normalization 区分，这在多平台扩展时很实用，但也意味着模型学到的跨 embodiment 对齐是否稳健，仍依赖数据覆盖和 prompt 可辨识度。另一个部署缺口是触觉、力控、失败恢复与长时程 memory，论文自己也把 force/tactile/proprioception 和 world modeling 放到未来方向。

## 六、灵魂三问

1. **它解决了什么问题？**

它解决的是 VLA/embodied policy 被任务族和机器人形态割裂的问题。相比为 manipulation、navigation、human trajectory 分别训练模型，Qwen-VLA 试图用一个 VLM backbone + DiT action decoder，在统一 action-and-trajectory prediction 接口下吸收异构监督。

2. **为什么这么做？**

因为这些任务虽然控制量不同，但都需要把视觉、语言、embodiment 约束映射成未来动作或轨迹。作者选择 embodiment-aware prompt 加统一张量/mask，而不是为每个机器人单独建 head，是为了最大化参数共享，并让模型通过文本条件理解当前控制 convention。

3. **什么证据最有说服力？**

最有说服力的是跨 benchmark 的 generalist 结果和真实 ALOHA 预训练对比：同一 Qwen-VLA-Instruct 在 LIBERO、Simpler、RoboTwin、VLN-CE 和 DOMINO 都有竞争力；真实 ALOHA 中 pretrained 版本平均 83.6%，scratch 版本 48.5%，说明大规模异构预训练确实转化成了真实机器人能力。

## 七、个人总结

1. Qwen-VLA 的核心不是某个单点模块，而是把 VLM 认知能力、DiT 连续动作生成、embodiment prompt、多源数据和 staged training 组合成一个可扩展的通用 VLA recipe。

2. 最大优势是统一接口下的跨任务、跨 embodiment 迁移；最大弱点是接触丰富、长时程、失败恢复和真实闭环交互仍不足，且动作语义的统一主要依赖 prompt 和数据分布。

3. 对 VLA/robotics 研究来说，这篇更像是一个强 baseline 和系统配方：以后做 cross-embodiment、human video pretraining、VL+VLA co-training、RL post-training，都可以把它当作重要参照。

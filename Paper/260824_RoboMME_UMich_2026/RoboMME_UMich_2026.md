# RoboMME：系统评测机器人策略的四类记忆能力

> 原标题：RoboMME: Benchmarking and Understanding Memory for Robotic Generalist Policies  
> 作者：Yinpei Dai, Hongze Fu, Jayjun Lee, Yuejiang Liu, Haoran Zhang, Jianing Yang, Chelsea Finn, Nima Fazeli, Joyce Chai  
> 机构：University of Michigan；Stanford University；Figure AI  
> 发表：arXiv:2603.04639v3，2026-05-26；Accepted to ICML 2026  
> 链接：https://arxiv.org/abs/2603.04639  
> 项目页：https://robomme.github.io/  
> 代码：https://github.com/RoboMME/robomme_policy_learning  
> 文件夹日期说明：按本地下载 PDF 的修改日期 `2026-08-24` 命名，而非论文发布日期

---

## 一、研究背景与动机

机器人策略的“记忆”经常被泛化地讨论，但现有评测并没有把真正依赖历史的能力单独测出来。许多长时程 benchmark 虽然轨迹很长，当前观测却已经包含足够信息，策略不需要回忆过去；已有 memory policy 也往往在各自设计的小任务上验证，模型 backbone、输入和成功协议不统一，难以判断到底是哪种 memory 表示有效。

RoboMME 将记忆拆成四种认知维度：**时间记忆**（数次数、顺序和时机）、**空间记忆**（遮挡或场景变化后的物体位置）、**对象记忆**（跨时间保持指代对象一致）和**程序记忆**（复现演示过的动作过程）。论文同时构建统一的 π₀.₅ memory policy suite，比较 symbolic、perceptual、recurrent 三类表示以及三种注入方式。

## 二、核心贡献

1. **提出 RoboMME benchmark**：基于 ManiSkill 构建四个 task suites、16 个非 Markov 长时程操作任务，覆盖时间、空间、对象和程序记忆。
2. **建立可控的数据集与任务 taxonomy**：1,600 条演示、约 770K 高质量 timestep；每个任务 100 个 episode，并通过遮挡、动态交换、视频条件、计数和轨迹模仿制造必须依赖历史的决策。
3. **构建 MME-VLA suite**：在同一个 π₀.₅ backbone 上比较语言子目标、视觉 token 历史、TTT 和 RMT，并测试 memory-as-context、memory-as-modulator、memory-as-expert 三种集成机制。
4. **给出任务依赖性的经验结论**：总体上 FrameSamp+Modul 最强，平均成功率 44.51%；symbolic memory 更擅长计数和高层事件记忆，perceptual memory 更擅长运动和时间敏感任务，recurrent memory 在该配置下表现最弱。

## 三、方法原理

### 3.1 整体框架

RoboMME 本身是 benchmark，MME-VLA 是围绕它建立的实验模型族。整体数据流如下：

```text
历史视频/执行历史 ──> memory representation ──> memory integration
当前 RGB + 语言 ───────────────────────────────> π₀.₅ VLM/action experts ──> action chunk
```

基础 π₀.₅ 使用 VLM expert 融合语言和视觉，action expert 根据 VLM 特征和 flow-matching timestep 生成动作。作者尽量固定 backbone、memory budget 和训练设置，只改变记忆表示或注入位置。

### 3.2 关键技术细节

#### 记忆表示

**Symbolic memory。** 辅助 VLM 根据当前图像和历史 subgoals 生成下一个语言子目标。SimpleSG 只输出自然语言，例如“pick up the green cube”；GroundSG 额外输出前视图中的像素坐标，例如“pick up the green cube at [63,152]”。子目标被追加到 π₀.₅ 的语言 prompt 中。子目标预测器包括 prompt-only Gemini-2.5-Pro、LoRA 微调的 Qwen3-VL-4B，以及 simulator ground-truth Oracle。

symbolic memory 很紧凑、可解释，但依赖子目标质量；语言能表达“做什么”，不一定能表达精确接触和连续运动。

**Perceptual memory。** 从 π₀.₅ 的 SigLIP2 vision encoder 中提取过去图像的视觉 token：

- FrameSamp：每帧 max-pool 到 `4×4=16` tokens，均匀采样 32 帧。
- TokenDrop：使用 `8×8` 网格，每隔 8 个时间步检查 patch 的平均 RGB 差异，阈值为 `1e-4`，保留发生变化的 token。

FrameSamp 保留全局时间分布，TokenDrop 更偏向局部变化，但过度删除可能丢失全局空间关系。

**Recurrent memory。** 将最多 64 帧的历史视觉 token 压缩为固定 512 个 memory tokens：

- RMT：512 个可学习 memory queries，对每帧 token 做 grouped-query cross-attention。
- TTT：使用 `512×512` fast weights 的线性 MLP，通过自监督损失在线更新；两个 TTT heads，学习率 `0.01`，做 query-key normalization 和 gradient clipping `5.0`。

与 RoboTTT 中将 fast weights 直接作为长序列状态不同，这里 TTT 输出最近 `B=512` 个 memory tokens，再接入 π₀.₅。

#### 记忆集成机制

**Memory-as-context：** 将 memory tokens 直接拼到当前 observation 和 language tokens 前面，由 VLM 联合处理。结构改动最小，但输入序列变长。

**Memory-as-modulator：** memory token 通过 attention 产生每层调制参数 `γ, β`，作用于 action expert 的 adaptive LayerNorm：

```text
r_t^k = Attn(Q=current action feature, K=memory, V=memory)
(γ_t^k, β_t^k) = MLP(r_t^k)
ŝ_t^k = γ_t^k ⊙ Norm(s̃_t^k) + β_t^k
```

它不直接改写 VLM token，而是以 feature-wise conditioning 调整动作专家，初始化为 identity modulation，因此更容易保留 π₀.₅ 的原能力。

**Memory-as-expert：** 增加一个独立的 memory expert 处理历史 token，再让 action expert 对 memory、VLM 和 action 三路 block 做 causal attention。表达能力更强，但增加约 190M 参数。

### 3.3 训练与优化

MME-VLA 采用统一 recipe：π₀.₅ base，训练 80K steps，warmup 10K，学习率 `5e-5`，AdamW（β₁=0.9、β₂=0.95、weight decay=0），gradient clip 1.0，EMA 0.999。冻结 SigLIP2 vision backbone，并预计算/缓存视觉 token；只训练 VLM expert、action expert 和新增 memory 参数。

| 模型/方案 | 训练设置 | 关键配置 |
|---|---|---|
| MME-VLA symbolic/perceptual | batch 64，4×A40，约 3–4 天 | 最大语言/memory budget 512 |
| MME-VLA recurrent | batch 16；TTT 用 4×A40，RMT 用 2×H100 | TTT 约 5–6 天，RMT 约 2–3 天 |
| SAM2Act+ | 两阶段各 40K steps，4×A40 | 离散 waypoint action，LoRA rank 16 |
| OpenVLA-OFT | 110K steps，2×H200 | OpenVLA-7B，L1 action regression，LoRA |
| MemoryVLA | 160K steps，2×H200 | DiT-L，16-step EEF action |
| Diffusion Policy | 200K steps，单 GPU | ResNet-18 + SpatialSoftmax，DDPM |

### 3.4 数据使用与维度追踪

| 数据源 | 规模 | 样本单位 | 模态/字段 | 关键维度 | 用途阶段 | 处理方式 |
|---|---:|---|---|---|---|---|
| RoboMME 仿真演示 | 16 tasks × 100 episodes = 1,600 demos；约 770K steps | dense trajectory；训练时随机 episode + 随机 timestep | front/wrist RGB、语言、关节状态、EEF pose、夹爪状态、动作 | 256×256 双视角；joint action 8D 或 EEF action 7D | VLA 训练与评估 | keyframe waypoint replay；5% waypoint noise 后恢复；失败 planner episode 丢弃 |
| Video-conditioned demonstrations | 各 Video/Imitation 任务的历史视频 | 初始视频序列 + 执行 trajectory | 视频帧、配对 proprioception、语言指令 | 执行时每步仍是当前 image；历史最多 64 frames | Permanence/Reference/Imitation | uniform/frame-stride sampling；视频最多可先采到 40 帧 |
| Symbolic subgoal labels | 每个 episode 的 keyframe subgoals | 当前图像 + previous subgoals → next subgoal | simple language 或语言 + `(u,v)` grounding | 最大语言 budget 512 tokens | symbolic memory | Qwen3-VL-4B-Instruct LoRA；batch 48、2 epochs、LR `1e-4` |
| 真实机器人演示 | 350 demos，78,400 steps | 50 PutFruits + 100×其他三任务 | RGB-D、多视角、语言、人工 grounded subgoals | 15 Hz；三相机；7-DoF Franka | real-world transfer | Oculus Quest 2 遥操作；keyframe 手工标注 |

**维度快照**

- Observation：仿真为 front + wrist 两个 RGB 视角，均 `256×256`；默认仅使用 front view 构造 memory，`V=1`。
- Language：π₀.₅ 原始输入为 64 language tokens；symbolic memory 将最大语言预算扩展到 512。
- Visual feature：SigLIP2 输出维度 2048，经轻量 MLP 投影到 π₀.₅ 内部宽度 1024；M-RoPE 为 768 维。
- State：原始 benchmark 包含 joint position、EEF pose、gripper state；但 MME-VLA 统一实验发现 proprioception 无收益，因此训练时关闭。Diffusion Policy 和 OpenVLA-OFT 对照实验保留 proprioception。
- Action：仿真 joint-space 为 8D（7 joints + gripper），EEF-space 为 7D（3D position、Euler orientation、gripper）；MME-VLA 主实验用 joint-space，action horizon 为 20。
- Memory：统一预算 `B=512 tokens`；perceptual FrameSamp 为 32 帧×16 tokens，recurrent 输入最多 64 帧×64 tokens 后压缩为 512 tokens；推理每 16 个执行步追加一帧。
- 归一化：多数方法使用 q1/q99 quantile normalization；Diffusion Policy 使用 min-max normalization。

训练样本对 recurrent memory 的定义尤其重要：从随机 episode 中随机选 timestep，输入最多 64 帧历史；梯度通过 64 个 recurrent steps，RMT 只在最终步监督，TTT 在最后 8 步监督。少于 64 帧时左侧 padding，并用 validity mask 区分 padding。所有视觉 token 预计算并缓存，说明这项 benchmark 的训练成本很大一部分来自 memory 设计，而非重复跑 vision encoder。

## 四、实验与结果

### 4.1 实验设置

RoboMME 使用 ManiSkill tabletop 环境和 7-DoF Franka Panda。每个任务 100 条训练/演示轨迹；最终评估包含 16 个任务，每任务 50 个 evaluation episodes。任务平均长度约 208–1,134 steps，所有任务都刻意让当前观测无法唯一决定动作。

| Suite | 记忆类型 | 任务 | 典型要求 |
|---|---|---|---|
| Counting | Temporal | BinFill、PickXTimes、SwingXTimes、StopCube | 数量、顺序、特定时机 |
| Permanence | Spatial | VideoUnmask、ButtonUnmask、VideoUnmaskSwap、ButtonUnmaskSwap | 遮挡、位置追踪、动态交换 |
| Reference | Object | PickHighlight、VideoRepick、VideoPlaceButton、VideoPlaceOrder | 视觉/动作/语言指代一致性 |
| Imitation | Procedural | MoveCube、InsertPeg、PatternLock、RouteStick | 复现工具使用、方向和运动轨迹 |

此外，作者在 Franka + UMI fin-ray fingers 上进行真实实验：PutFruits、TrackCube、RepickBlock、DrawPattern，分别对应仿真中的 BinFill、VideoUnmask(Swap)、VideoRepick、PatternLock。

### 4.2 主要结果

| 方法 | 平均仿真成功率 | 主要特点 |
|---|---:|---|
| FrameSamp + Modul | **44.51%** | 所有 MME-VLA 变体中总体最佳，性能/计算折中最好 |
| MemER | 42.38% | 使用 keyframe + symbolic subgoal，动态场景变化较强 |
| GroundSG + QwenVL | 最高约 32.70%（MME-VLA symbolic） | 对 counting 和 grounding 任务有效 |
| TTT/RMT recurrent variants | 约 18–22% 平均区间 | 当前浅层 recurrent 集成训练不稳定 |
| π₀.₅ baseline | 17.93% | 没有显式 memory |
| π₀.₅ + past actions | 19.73% | 简单加入过去动作不足以解决记忆问题 |

作者的核心结论不是“某一个 memory 永远最好”，而是任务和记忆形式匹配：

- FrameSamp 通常优于 TokenDrop；后者的激进 token 删除会损失全局空间信息，StopCube 等任务尤其受影响。
- memory-as-modulator 通常优于 context 和 expert，因为它保留 π₀.₅ 的表示路径，仅调节 action expert。
- symbolic memory 在计数、事件显著任务上强；perceptual memory 在运动模仿和时间敏感任务上强。
- GroundSG+Oracle 达到约 84% overall，说明语言子目标的表示容量很强，但即使给出正确 subgoal，StopCube、InsertPeg 等低层精确控制任务仍会失败。

**真实机器人结果：**

| 方法 | PutFruits | TrackCube | RepickBlock | DrawPattern | 总成功 |
|---|---:|---:|---:|---:|---:|
| π₀.₅ | 2/10 | 1/10 | 1/10 | 0/10 | 4/40 |
| GroundSG + QwenVL | 9/10 | 3/10 | 5/10 | 2/10 | 19/40 |
| FrameSamp + Modul | 6/10 | 5/10 | 6/10 | 8/10 | **25/40** |

真实实验复现了仿真趋势：PutFruits 这类计数任务 symbolic 更强，DrawPattern 这类动作轨迹任务 perceptual 更强；TrackCube 中 QwenVL 的 grounding 在动态交换时容易出错。

### 4.3 消融与额外分析

1. **Memory budget**：FrameSamp+Modul 随 memory budget 增大获得稳定收益，额外成本相对温和；关键原因是计算主要花在视觉 token，而非 modulation 本身。
2. **TokenDrop vs. FrameSamp**：TokenDrop 只保留局部发生变化的 patch，虽然减少冗余，但在需要全局布局和物体距离的任务上会误删信息。
3. **Integration mechanism**：modulator 的 identity initialization 和 action-expert 局部注入更适合已有 π₀.₅；expert 增加容量但也增加约 190M 参数，context 则增加序列长度。
4. **Symbolic grounding**：GroundSG 通常优于 SimpleSG，但在不需要精确位置的 PickXTimes 上，错误 grounding 反而可能伤害性能。
5. **外部 VLM 成本**：GroundSG+QwenVL 约为 π₀.₅ 的 3× 计算量，MemER 约 5×；缓存 subgoal 或视觉 token 可以降低开销。
6. **人类上限**：在 oracle planner 将低层动作执行理想化后，18 名参与者、800 个 evaluation episodes 的平均成功率为 90.5%，但在长期保持注意力、StopCube 和轨迹记忆任务上仍会失败，说明 benchmark 并非只是在测机器人低层控制。

## 五、局限性与展望

论文最重要的限制是：RoboMME 主要是 ManiSkill 中的人工设计任务，虽然强调非 Markov 和长时程，但与开放世界家务任务的语义复杂度、视觉多样性和真实接触动力学仍有距离。每个任务的对象集合和结构相对固定，模型可能学习到任务模板，而非真正通用的 episodic memory。

MME-VLA 的 recurrent 结果也不能直接推出“recurrent memory 无效”：作者使用的是轻量 TTT/RMT、从 π₀.₅ fine-tuning 的配置，论文自己将较差表现归因于浅层集成和训练不稳定。更公平的结论是：在当前 backbone、数据和 80K-step recipe 下，perceptual memory 更可靠。

此外，真实机器人只覆盖 4 个任务、350 条示范，且使用单臂 Franka 和固定相机布局。symbolic 方法需要额外 VLM 推理与 grounding 标注，perceptual 方法需要缓存/处理视觉 token，二者在长时间部署中的延迟、错误累积、memory reset 和安全性仍需要更系统的评估。

后续方向包括：让 benchmark 包含更多本体、环境和自然语言变化；使用真实失败轨迹而不只是成功 planner replay；将 symbolic 的事件压缩与 perceptual 的运动细节联合起来；以及针对机器人 memory 设计训练目标，而不是只把通用序列模块接到已有 VLA 上。

## 六、灵魂三问

1. **它解决了什么问题？**  它解决的是机器人 memory 研究缺乏统一问题定义和可比较实验的问题。RoboMME 把“记忆”拆成时间、空间、对象、程序四类，并确保当前观测不足以完成任务。

2. **为什么这么做？**  因为单一 memory 形式无法同时保留事件计数、物体位置、对象身份和连续动作轨迹。语言 subgoal 压缩得好但会丢运动细节，视觉 token 保留细节但成本高，recurrent 状态省内存但可能压缩过度；统一 π₀.₅ suite 能把表示能力与集成方式分开比较。

3. **什么证据最有说服力？**  最有说服力的是任务类型与 memory 类型之间的稳定对应：FrameSamp+Modul 在仿真平均 44.51%，真实 DrawPattern 达到 8/10；GroundSG+QwenVL 在真实计数 PutFruits 达到 9/10，而 π₀.₅ 只有 2/10。这说明 memory 的价值取决于任务需要回忆的是事件、空间状态还是动作过程。

## 七、个人总结

1. RoboMME 的最大价值不是提出一个新的 memory module，而是把机器人记忆从模糊概念变成可拆解、可诊断的 benchmark 维度。
2. 论文给出的工程结论很实用：对 π₀.₅ 这类已有 VLA，轻量 memory-as-modulator + 均匀视觉帧采样是当前较稳妥的起点；单纯 past actions 或浅层 recurrent memory 并不够。
3. 对 VLA 数据工程的启发是，memory 数据不能只按 episode 数量统计，还要标注历史依赖类型、视频/当前帧边界、keyframe/subgoal、动作维度和 padding mask；否则模型的“记忆提升”很难复现，也很难知道究竟记住了什么。

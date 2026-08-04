# X-VLA：用 Soft Prompt 吸收跨机器人异构性的 VLA 框架

> 原标题：X-VLA: Soft-Prompted Transformer as Scalable Cross-Embodiment Vision-Language-Action Model  
> 作者：Jinliang Zheng, Jianxiong Li, Zhihao Wang, Dongxiu Liu, Xirui Kang, Yuchun Feng, Yinan Zheng, Jiayin Zou, Yilun Chen, Jia Zeng, Ya-Qin Zhang, Jiangmiao Pang, Jingjing Liu, Tai Wang, Xianyuan Zhan  
> 机构：Institute for AI Industry Research (AIR), Tsinghua University / Tsinghua University / Peking University  
> 发表：arXiv:2510.10274, 2025-10-11；官方 README 标注 ICLR 2026 accepted  
> 链接：https://arxiv.org/abs/2510.10274  
> 项目页：https://thu-air-dream.github.io/X-VLA/  <!-- 2026-07-31 访问返回 404，论文和 README 仍给出该地址 -->
> Demo：https://sites.google.com/view/xvla  
> 开源：https://github.com/2toinf/X-VLA  
> Checkpoints / datasets：https://huggingface.co/collections/2toINF/x-vla

---

## 一、研究背景与动机

VLA 预训练最难的地方不只是“数据量不够”，而是机器人数据天然很乱：不同 robot arm、不同相机视角、不同 proprioception 字段、不同 action space、不同采集协议混在一起。直接把这些数据塞进一个共享 backbone，模型会同时面对视觉域偏移、动作语义偏移和 embodiment 动力学偏移，训练容易不稳定，下游适配也不一定受益。

已有方法常见处理方式是给不同 embodiment 单独 action head，或者手写 embodiment prompt / domain adapter。但 X-VLA 认为这样还不够：异构性不只存在于 action head，也存在于 observation、proprioception、相机布局和数据采集分布。作者提出的核心想法是很轻的：给每个 data source / hardware setup 学一组 soft prompt，让 prompt 去吸收域差异，而共享 Transformer backbone 专注学习更通用的 action generation 能力。

从 VLA 算法工程角度看，这篇论文的价值在于它把“跨 embodiment 预训练”拆成了一个更可落地的问题：不用把所有机器人硬塞成完全同构的观测，而是保留 domain-specific prompt 和少量 input/output projection，同时共享绝大多数参数。

## 二、核心贡献

1. **提出 soft-prompted Transformer VLA。** 每个数据源/硬件配置有独立 learnable embeddings，用 soft prompt 编码 embodiment-specific 信息，配合标准 Transformer encoder 做多模态融合和 flow-matching action generation。

2. **实现 X-VLA-0.9B 并验证可扩展性。** 最大配置约 0.9B 参数，24 层 Transformer、hidden size 1024；在 290K episodes、7 个 data sources 上训练，论文称在模型大小、数据多样性、数据规模三条轴上 validation prediction error 都没有饱和。

3. **设计面向异构数据的训练 recipe。** 预训练阶段联合优化 backbone 和 soft prompts；适配新 embodiment 时先 prompt warm-up，再 joint policy adaptation；同时对 soft prompt 和 VLM 模块使用更小学习率以减少 pretrained representation drift。

4. **统一并抽象 action 监督。** 动作统一到 end-effector pose：XYZ + Rotate6D + binary gripper；预训练时将未来 4 秒动作下采样成 30 个 anchor points，用更抽象的 trajectory intention 替代过密、噪声大的低层轨迹。

5. **大范围实验覆盖仿真和真机。** 在 6 个 simulation benchmarks 和 3 个 real-world robots 上评估，包括 LIBERO、Simpler、VLABench、RoboTwin-2.0、CALVIN、NAVSIM，以及 WidowX、AgileX、AIRBOT 等真机设置。

## 三、方法原理

### 3.1 整体框架

X-VLA 的输入输出可以概括成：

```text
multi-view RGB + language instruction + proprioception + domain_id
        -> VLM / vision encoder + soft prompt + action/proprio projection
        -> standard Transformer blocks
        -> flow-matching denoising
        -> future action chunk
```

它不是 π0 那种 VLM + action expert 的强分离结构，也不是每个机器人一套独立策略。它更像是在共享 backbone 前加一组“可学习的 embodiment 条件变量”：同一个模型看到 `domain_id` 后取出对应 soft prompt、输入投影和输出投影，然后在统一 Transformer 里完成融合。

### 3.2 关键技术细节

#### Soft Prompt：让数据源自己学 embodiment embedding

论文把每个硬件配置/数据源对应到一组 soft prompts：

```text
P^H = {p_i}_{i=1}^H
```

其中 `p_i` 不是手写文本 prompt，也不是人工定义的 robot descriptor，而是随机初始化后通过端到端训练学出来的 embedding。直觉上，它应该近似编码某个隐藏硬件配置 `h_i` 的信息，包括 arm 类型、相机设置、动作空间和采集域差异。

这和直接加 domain-specific action head 的区别是：soft prompt 在 action generation 的早期就参与 attention，能影响视觉、语言、proprioception 和 noisy action token 的融合；它不是最后一步才补救 action dimension mismatch。

#### 架构：标准 Transformer，而不是复杂专家系统

论文附录说明 X-VLA 采用 Florence-Large 作为 vision-language encoder，并使用 24 层、hidden size 1024 的标准 Transformer backbone 做 action generation。大部分参数跨 embodiment 共享，只有 soft prompt 以及 action-related token 的 input/output linear projections 是 domain-specific；这些非共享参数只占总参数约 0.04%。

视觉输入处理有一个很工程化的选择：主视角进入完整 VLM，额外视角如 wrist view 只送入 vision encoder。作者的理由是当前 VLM 对多视角感知并不天然稳定，如果把所有视角都丢进完整 VLM，可能破坏预训练视觉语言能力。

#### Flow Matching Action Generation

X-VLA 用 flow matching 生成 action chunk。它不是直接回归动作，而是从高斯噪声 action sample 出发，学习 velocity field，把噪声逐步搬运到目标 action chunk。README 的部署接口里 `steps` 控制 denoising steps，例子使用 10。

在模型内部，proprioception、flow-matching time variable 和 noisy action chunk 会拼接/重复后投影到高维 token 空间，再和 soft prompt、视觉 token、语言 token 一起进入 Transformer。最后 control tokens 通过 domain-specific output projection 映射回 action chunk。

#### Action 表示：EEF XYZ + Rotate6D + Gripper

作者把 action supervision 标准化到 end-effector pose：

| 组件 | 维度/形式 | 训练损失 | 说明 |
|---|---|---|---|
| EEF position | XYZ, 3D | MSE | 笛卡尔末端位置 |
| EEF rotation | Rotate6D, 6D | MSE | 避免 Euler / quaternion 不连续 |
| Gripper | binary | BCE | 离散开合状态 |
| Padding | 视 embodiment 而定 | 通常不作为真实控制量 | README 的 EE6D 接口给出 20D，其中单臂用 padding 补齐 |

README 对标准控制接口写得更直接：X-VLA 采用 EE6D control space，action output 是 20-D vector = 3 position + 6 rotation + 1 gripper + 10 padding；单臂场景用 0 padding 维持 20D。client 示例里返回 `30` 个 predicted actions，因此开源部署接口对应的典型输出 shape 是 `30 x 20`。

#### Temporal Downsampling：把低层轨迹变成意图 anchor

这是数据处理里最值得记的一点。作者认为原始低层动作轨迹太细，包含大量人类操作随机性和无意义微动，直接拿来做大规模预训练会让模型过度拟合低层噪声。于是预训练时不预测每个控制周期的完整 EEF pose，而是预测未来 4 秒内的 30 个 anchor points。

这个设计有点像把 VLA action supervision 从“密集控制信号”抽象成“未来轨迹意图”。它牺牲了部分低层控制精度，但提高了预训练时跨数据源共享的语义一致性。真正部署到目标机器人时，再通过 finetuning / post-processing 接回具体控制接口。

### 3.3 训练与优化

训练分两阶段：

1. **Phase I: Pretraining。** 在 290K episodes 的异构数据混合上预训练 X-VLA-0.9B，数据来自 Droid、RoboMind、AgiBot 等开源机器人数据，覆盖 7 个 data sources / hardware setups 和 5 类 robot arms。训练时联合优化 backbone 与每个数据源的 soft prompts。

2. **Phase II: Domain adaptation。** 面向新 embodiment 引入新的 soft prompt。先只 warm up 新 prompt，让它对齐预训练 backbone 的通用特征；然后再 joint policy adaptation，让 backbone 和 prompt 一起适配目标域。

作者还强调两个稳定训练技巧：

- soft prompt 和 vision-language module 用较低 learning rate，避免破坏 VLM 的预训练表示；
- 数据采样不采用简单 round-robin，而是在 domain 间、每个 domain 内 trajectory 间都打乱，以减轻 dominant domain bias。

开源 README 还给了微调参数入口：`freeze_steps` 和 `warmup_steps`，默认示例里 `freeze_steps=1000`、`warmup_steps=2000`。论文附录对两步适配的描述是：先冻结大部分参数训练 prompt/action head，再逐渐恢复到 joint training。

### 3.4 数据使用与维度追踪

X-VLA 的数据设计可以分成两层：论文里的预训练混合数据，以及开源部署/微调接口里的标准化样本格式。对工程复现来说，第二层同样重要，因为它说明 raw dataset 到模型输入不是自然对齐的，需要 handler / domain config / projection 处理。

| 数据源 | 规模 | 样本单位 | 模态/字段 | 关键维度 | 标签/动作 | 用途阶段 | 处理方式 |
|---|---:|---|---|---|---|---|---|
| Droid | 论文未单独报告 | episode / trajectory | RGB、language、proprioception、action | Droid 内含不同 camera setup；论文提到 Droid-Left / Droid-Right prompt 聚类接近 | EEF action 统一后进入 action chunk | 预训练 | 转成统一 EEF 表示，按 data source 配 soft prompt |
| RoboMind | 论文未单独报告 | episode / trajectory | RGB、language、state/action | 具体相机数、图像分辨率未报告 | EEF XYZ + Rotate6D + gripper | 预训练 | 与其他源混合，domain 内/跨 domain shuffle |
| AigBot / AgiBot | 论文未单独报告 | episode / trajectory | robot observation、language、state/action | 覆盖单臂到双臂硬件 | 统一 action 表示 | 预训练 | 作为跨 embodiment 数据混合来源之一 |
| 预训练混合数据 | 290K episodes | episode -> sampled action chunk | multi-view images、language instruction、proprioception、noisy action chunk、domain prompt | 7 data sources，5 类 robot arms；X-VLA-0.9B 为 24 layers / hidden 1024 | future action chunk | Phase I pretraining | flow-matching BC loss；balanced data sampling；temporal downsampling |
| 下游仿真数据 | 各 benchmark 自带 | demonstrations / validation tasks | image、language、proprio、action | LIBERO、Simpler、VLABench、RoboTwin-2.0、CALVIN、NAVSIM | domain-specific action | Phase II adaptation / eval | 新 prompt warm-up + joint adaptation |
| 真机数据 | 任务相关 | demonstrations | WidowX / AgileX / AIRBOT 图像与 proprio/action | AIRBOT 是预训练未见平台；Soft-Fold 为 1,200 trajectories | target robot action | real-world finetune / PEFT | 少量 demos 适配；cloth folding 用 DAgger-style 迭代采集 |
| 开源推理接口 | 单次请求 | current observation -> action sequence | `proprio`、`language_instruction`、`image0`、可选 `image1/image2`、`domain_id`、`steps` | README 示例 image 为 `256 x 256 x 3`，proprio 示例为 7D；标准 EE6D action 为 20D | 返回 30 个 predicted actions | deployment | HTTP server-client；`domain_id` 选择 embodiment/domain |

**样本怎么变成模型监督**

1. **原始 episode 到 action chunk。** 每条 demonstration 先被整理成 observation/state/action 序列。训练时采样当前时间 `t`，输入当前图像、语言、proprioception 和 noisy action chunk，监督目标是未来 action chunk。

2. **action space 对齐。** 不同机器人原始控制接口先映射到 EEF pose：XYZ + Rotate6D + gripper。对单臂/双臂维度不一致的问题，开源 EE6D 接口采用 20D action vector；单臂只使用前 10D 控制量，剩余 10D padding。

3. **时间下采样。** 预训练不直接预测未来 4 秒所有低层控制点，而是抽成 30 个 anchor points。这样每个监督样本更接近“未来动作意图”，减少人类示教微扰对预训练的干扰。

4. **domain conditioning。** 每条样本携带 data source / domain identity，模型据此查询对应 soft prompt 和 action projection。README 推理 payload 中的 `domain_id` 正是这个思想在部署接口中的显式形式。

5. **flow matching 监督。** action chunk 被加噪成 `A^t`，模型学习 velocity field，把噪声 action 朝 ground-truth action chunk 推回去。部署时通过若干 denoising steps 输出 action sequence。

**维度快照**

- Observation: multi-view RGB；主视角进入完整 VLM，额外视角只进 vision encoder；README 示例为 `image0: 256 x 256 x 3`，但论文未报告统一训练分辨率、crop/resize、augmentation。
- Language: natural language instruction；tokenizer、最大 token 长度、mask 规则论文未说明。
- Proprioception: 当前 proprioceptive state，例如 joint positions / EEF pose；README 示例是 7D `proprio`，但通用维度随 domain handler 改变。
- Soft prompt: 每个 data source / hardware setup 一组 learnable embeddings；prompt 维度 `k` 论文以符号表示，未在正文明确给出具体 token 数。
- Action: EE6D control；单个 EEF 为 `3 xyz + 6 Rotate6D + 1 gripper = 10D`；README 标准 action vector 为 20D，含 10D padding。
- Prediction target: future action chunk；开源接口返回 30 个 predicted actions；论文预训练抽象为未来 4 秒 30 个 anchor points。
- Control frequency: 原始 robot control frequency 未统一报告；4 秒 / 30 anchor 约等价于 7.5 Hz 的抽象 action target，不应直接理解为真实低层控制频率。

**预处理链路拆解**

- 统一动作表示：把各源 action 映射到 EEF xyz、Rotate6D rotation 和 binary gripper。
- 低层轨迹抽象：将未来 4 秒 trajectory 下采样为 30 anchor points，降低示教噪声与微动作密度。
- 数据混合：跨 domain shuffle，同时在每个 domain 内部 trajectory shuffle，避免 round-robin 或 dominant domain 导致训练偏置。
- 域条件注入：每个 data source 绑定 soft prompt，另有 domain-specific input/output projection 处理 action-related tokens。
- 下游适配：新 domain 先训练 prompt/action head，再逐步开放 joint training；PEFT 场景使用 LoRA 等少量参数适配。
- 开源数据接入：README 要求为自定义数据准备 meta JSON、实现 domain handler、注册 domain config；这意味着 X-VLA 的“统一”依赖显式数据加载适配层，而不是模型自动吃任意机器人日志。

**工程判断**

X-VLA 对数据的态度比较务实：它没有强行发明一个包含所有机器人自由度的巨大 canonical state/action vector，而是把 action 规约到 EEF 控制，并用 soft prompt + domain projection 处理无法统一的部分。这对跨 embodiment 预训练很友好，但也意味着底层关节空间、移动底盘、人形全身控制这类更复杂 embodiment 可能需要额外 mapping 或更强的 embodiment descriptor。

## 四、实验与结果

### 4.1 实验设置

实验分三条主线：

1. **Scaling experiments。** 观察模型大小、数据多样性和数据规模增加时，held-out validation action prediction error 是否下降。最大配置为 X-VLA-0.9B，使用 290K episodes、7 data sources。

2. **Adaptation experiments。** 在 LIBERO、Simpler、VLABench、RoboTwin-2.0、CALVIN、NAVSIM 等 6 个仿真 benchmark 上评估，也在 WidowX、AgileX、AIRBOT 三类真机平台上测试。

3. **In-depth analysis。** 用 T-SNE 可视化 learned soft prompts，并比较 random prompt、frozen pretrained prompt、adapted prompt 在新平台适配中的差异。

从数据划分角度看，论文的评估更像“预训练 foundation + 下游适配”的系统验证，而不是单一 held-out dataset。尤其真机任务往往需要额外 demos、PEFT 或 full finetuning，因此不能把结果解读成完全 zero-shot 跨机器人部署。

### 4.2 主要结果

#### 仿真 benchmark 总览

| Benchmark | X-VLA 结果摘录 | 对比/意义 |
|---|---:|---|
| LIBERO | README 发布模型标注 98.1%；论文正文提到 93% 适配结果 | 在常见 manipulation suite 上接近/超过强 VLA baseline |
| Simpler-WidowX | README 发布模型标注 95.8%；论文正文 PEFT 段提到 54% / 54.2% 设置 | 不同评估协议差异较大，需要看具体 model checkpoint 和 split |
| CALVIN ABC -> D | README 发布模型标注 4.43 | 多阶段长程操作能力较强 |
| RoboTwin-2.0 | README 发布模型标注 70% | 双臂协调 benchmark |
| VLABench | README 发布模型标注 51.1 score | 更综合的仿真任务分数 |
| NAVSIM | 论文附录表中 PDMS 87.3 | 说明框架被扩展到 autonomous driving style action benchmark |

这里要小心：README 的 checkpoint 表是后续发布版本/具体 checkpoint 的性能摘要，论文正文和附录表的数字来自不同实验设置。笔记里更建议把它们看成“覆盖范围和可用 checkpoint 线索”，不要直接横向比较。

#### 真机结果

| 平台 | 任务类型 | 数据/适配 | 结果要点 |
|---|---|---|---|
| WidowX | BridgeData-v2 风格 pick-and-place | BridgeData finetune 后部署 | 评估 manipulation 和 language instruction following，每个任务 10 次 |
| AgileX | dexterous cloth folding | Soft-Fold 1,200 trajectories | X-VLA-0.9B 达到接近 100% success，约 33 folds/hour |
| AIRBOT | cloth-pick PEFT | 预训练未见平台，200 demos | 用 LoRA 等 PEFT 检验少参数适配能力 |

对 VLA 工程师来说，最有价值的不只是成功率，而是这些任务的部署差异很大：单臂桌面操作、双臂布料折叠、未见平台少样本适配。soft prompt 的目标就是让这些域差异以很小参数成本被吸收。

#### PEFT 与参数效率

论文报告用约 9M 可训练参数，即约 1% full model，就能在 LIBERO 和 Simpler-WidowX 上接近 fully finetuned π0 类模型：正文提到 PEFT 达到 Libero 93%、Simpler-WidowX 54%，而 π0 对应 94.2%、55.7%。这支持了“共享 backbone 已学到 embodiment-agnostic prior，少量 domain-specific 参数足以适配”的说法。

### 4.3 消融实验

#### 异构性处理路径

论文的预备实验比较了多种处理方式：naive mixed training、domain-specific projection、language prompt、soft prompt 等。核心结论是，直接混训会退化，单纯做 projection 或手写 prompt 也不稳；soft prompt 在 validation error 和 Simpler-WidowX downstream adaptation 上更好。

#### 架构设计

附录比较了 DiT、MM-DiT、π0-style decoder 和 X-VLA。validation error 分别为：

| Backbone | Validation Error |
|---|---:|
| DiT | 0.077 |
| MM-DiT | 0.140 |
| π0-style | 0.056 |
| X-VLA | **0.041** |

这说明“标准 Transformer + soft prompt + 合理输入流拆分”在作者的异构数据设置下比更复杂的 multimodal DiT 更稳定。

#### Soft Prompt 表征

T-SNE 显示 7 个 data sources 的 soft prompts 会形成和硬件配置相关的 cluster；Droid-Left / Droid-Right 这类只差主视角的 Franka 设置没有被粗暴分开，说明 prompt 学到的不只是数据源 ID，而可能包含一定硬件相似性。

#### 数据效率

附录 D.3 报告少样本适配：50 demos 达到 92.8% success，10 demos 仍有 91.1%。这类结果说明 prompt + pretrained backbone 对小数据适配有帮助，但也要注意任务域和 evaluation protocol 会强烈影响这个数字。

## 五、局限性与展望

1. **仍需要 domain-specific adaptation。** X-VLA 不是任意新机器人拿来就能 zero-shot 部署。论文也承认当前部署通常还需要收集少量 target demos 做 post-training。

2. **统一 EEF action 降低了全身控制表达力。** EEF XYZ + Rotate6D + gripper 对桌面操作很实用，但对移动底盘、人形全身、灵巧手多指接触而言，可能不如 LingBot-VLA 2.0 这类全身 canonical vector 直接。

3. **关键数据细节披露仍不足。** 论文给出 290K episodes、7 sources、5 robot arm types、30 anchor / 4s，但没有完整列出每个数据源的 episode 数、采样比例、图像分辨率、语言 token 长度、action normalization 统计、filtering 规则。

4. **temporal downsampling 是好用 heuristic，但不是 richer supervision。** 作者在 limitation 中也提到，低维 action label 本身信息有限；30 anchors 能抽象意图，但没有真正引入 3D spatial reasoning、physical dynamics 或 subgoal annotations。

5. **项目页可访问性有问题。** arXiv 和 README 给出的 GitHub Pages 项目页在 2026-07-31 访问返回 404；代码、Hugging Face 集合和 README 可用，但 demo/项目页需要以后再核验。

后续值得关注：

- soft prompt 是否能和显式 robot morphology / URDF / kinematic descriptor 结合；
- 对 mobile manipulation、humanoid、dexterous hand 的 action schema 扩展；
- 用更丰富的自监督目标补足低维 action label；
- 建立更清楚的数据 mixture card，方便比较不同 VLA 预训练 recipe。

## 六、灵魂三问

1. **它解决了什么问题？**

它解决的是跨 embodiment VLA 预训练中的异构性吸收问题。相比只给不同机器人换 action head，X-VLA 把 data source / hardware setup 作为 soft prompt 注入 action generation 早期，让共享 Transformer 在保留通用能力的同时获得 domain-specific 条件信息。

2. **为什么这么做？**

因为机器人数据的差异不只在动作维度，也在相机设置、proprioception 语义、采集策略和任务分布。soft prompt 的好处是参数极少、可端到端学习、能随着数据规模扩展；再配合 EEF action 对齐和 30-anchor temporal downsampling，模型能在混合数据中学到更稳定的跨域行为先验。

3. **什么证据最有说服力？**

最有说服力的是三类证据放在一起：最大配置在 0.9B / 290K episodes / 7 sources 下仍呈现 scaling trend；架构消融中 X-VLA validation error 低于 DiT、MM-DiT 和 π0-style；PEFT 用约 9M 参数就能接近 fully finetuned baseline。这比单个 benchmark SOTA 更能说明 soft prompt 对异构数据预训练有实际价值。

## 七、个人总结

1. X-VLA 的核心不是发明复杂大模型，而是把跨机器人数据里的 domain heterogeneity 交给 learnable soft prompt，并保持 backbone 尽量标准、可扩展。

2. 最大优势是工程接口清楚：`domain_id`、soft prompt、domain projection、EE6D action、30-step output 这些都能落到代码和部署；最大弱点是 action schema 偏 EEF，对全身/灵巧手/移动操作的表达还不够自然。

3. 对 VLA 预训练研究的启发是：数据混合不应该只问“有多少 episodes”，还要问每个 domain 的硬件差异如何进入模型、动作是否被过度低层化、padding/normalization/采样策略是否让模型学到了错误的域偏置。

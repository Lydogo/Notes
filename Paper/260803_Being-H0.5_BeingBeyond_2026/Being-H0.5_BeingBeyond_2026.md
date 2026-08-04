# Being-H0.5：把人类手部交互当成跨本体 VLA 的“共同动作语言”

> 原标题：Being-H0.5: Scaling Human-Centric Robot Learning for Cross-Embodiment Generalization  
> 作者：Hao Luo, Ye Wang, Wanpeng Zhang, Sipeng Zheng, Ziheng Xi, Chaoyi Xu, Haiweng Xu, Haoqi Yuan, Chi Zhang, Yiqing Wang, Yicheng Feng, Zongqing Lu  
> 机构：BeingBeyond Team  
> 发表：arXiv:2601.12993v1, submitted 2026-01-19；项目页日期 2026-01-20；PDF 44 页  
> 链接：https://arxiv.org/abs/2601.12993  
> 项目页：https://research.beingbeyond.com/being-h05  
> 开源：https://github.com/BeingBeyond/Being-H  
> 权重：https://huggingface.co/collections/BeingBeyond/being-h05  

---

## 一句话说清楚

Being-H0.5 的核心不是单纯做大一个 action head，而是把人类手部轨迹、机器人 EEF/joint/base 控制和视觉语言监督放进一个统一序列中训练：人类视频提供可扩展的交互先验，统一 state-action space 负责跨本体对齐，Mixture-of-Flow 增加动作生成容量，MPG/UAC 再把这个统一模型推到真实异构机器人闭环部署里。

## 一、研究背景与动机

VLA 模型要跨机器人本体泛化时，主要卡在两类不一致上：第一是数据规模不对称，多数机器人平台没有足够多、足够杂的交互轨迹；第二是动作空间不对齐，不同机器人可能使用 EEF delta、joint target、移动底盘速度、灵巧手关节等不同控制语言。把这些数据直接拼起来，会让动作监督互相干扰，尤其在 diffusion/flow 类 action chunk 生成里，复杂高 DoF 本体的动作流形更容易偏离有效区域。

作者的判断是：人类手部交互可以作为物理交互的“母语”。人类视频便宜、覆盖场景广、接触行为多，虽然不是机器人控制量，但它包含抓取、接触、工具使用、双手协作等可迁移结构。如果能把人手和机器人都投到一个物理语义对齐的 action space，就可以让低资源机器人从人类交互和高资源机器人数据里借到先验。

这篇论文的目标是构建一个能在多种真实机器人上运行的跨本体 VLA 基础模型，而不是只在单一 benchmark 上做 specialist。它同时解决数据配方、动作表示、动作专家容量和真实部署延迟问题。

## 二、核心贡献

1. **提出 UniHand-2.0 大规模 human-centric 预训练配方**  
   数据总量超过 35,000 小时、400M 样本、120B tokens，覆盖 16,000 小时 egocentric human video、约 14,000 小时 robot manipulation 和 5,000 equivalent hours 的视觉语言理解数据，机器人部分覆盖 30 个本体。

2. **用统一 state-action space 连接人类手部运动和机器人控制**  
   论文把 EEF pose、joint、gripper/finger articulation、base velocity 等放进固定语义 slot，并把 MANO hand wrist/finger 参数映射进去，使 human motion 可以进入同一 action supervision 管线。

3. **提出 Mixture-of-Flow 扩展动作专家容量**  
   MoF 将 action expert 分成共享 foundation experts 和 routed specialized experts，前者学习通用 motor primitives，后者吸收 embodiment/task-specific dynamics，减少多本体联合训练中的负迁移。

4. **面向真实部署提出 ESA、MPG 和 UAC**  
   Embodiment-Specific Adaptation 只更新活跃 slot 的 adapter；Manifold-Preserving Gating 在观测上下文不可靠时收缩 feature-conditioned correction；Universal Async Chunking 用本体相关延迟建模解决 action chunk 推理和执行异步的问题。

5. **在仿真和真实多平台上给出证据**  
   LIBERO specialist 达 98.9%，RoboCasa Human-50 specialist 达 53.9%；真实机器人覆盖 PND Adam-U、FR3+Inspire、Unitree G1+O6、BeingBeyond D1、LeRobot SO-101 五个平台，并报告了弱但明确的 task-embodiment zero-shot 信号。

## 三、方法原理

### 3.1 整体框架

Being-H0.5 是一个 Mixture-of-Transformers 风格的 VLA。它把 multimodal understanding 和 action generation 分开建模，但通过共享 attention 与统一 state-action interface 连接。

| 模块 | 输入 | 输出 | 作用 |
|---|---|---|---|
| Understanding expert | RGB/文本/state/action tokens 的上下文 | 文本、语义表示、共享上下文 | 保留 VLM 的视觉语言理解、规划和 grounding 能力 |
| Action expert | 上下文特征、state/action suffix tokens | 连续 action chunk | 用 flow matching 生成机器人或人类手部运动 |
| Unified state-action space | raw embodiment-specific state/action | 固定语义 slot 向量 | 让人类手部、单臂、双臂、灵巧手、移动底盘共存 |
| Mixture-of-Flow | action expert 中的 flow layers | routed velocity field | 增加容量，同时避免所有本体挤在一个小 action expert 里 |

训练时，所有监督被序列化为 QA-style multimodal token stream：

```text
S = [vision/text/state/action segments]
L = lambda_text * L_text + lambda_act * L_act
```

VQA 和 motion description 使用文本 cross-entropy；human/robot motion generation 和 continuation 使用 action loss。这个统一序列的好处是简单，坏处是很多工程细节被藏在数据 adapter、slot mask 和 attention mask 里，复现时必须追这些细节。

### 3.2 关键技术细节

**1. Unified state-action space**

论文把不同本体的 state/action 写入共享物理语义 slot。核心约束是：

- Cartesian EEF action 表示为统一 world frame 下的 relative delta displacement。
- rotation 统一用 axis-angle，避免 Euler/quaternion 混用。
- joint-space positions 使用 absolute radians。
- 不做传统统计归一化到 `[-1, 1]`，作者认为 1 rad、10 cm 这类物理量本身有意义；只做 outlier filtering。
- human hand motion 通过 MANO 参数进入统一空间，wrist 对齐到 EEF subspace，finger articulations 对齐到 fine-manipulation slots。

这和很多 VLA 的 “zero-pad + dataset-specific normalization + per-embodiment head” 不一样。Being-H0.5 更激进地希望 action channel 本身带物理语义。

**2. Mixture-of-Flow**

MoF 的直觉是：统一 action language 解决“怎么放在一起”的问题，但没有解决“一个 action expert 容量够不够”的问题。低 DoF gripper、双臂、灵巧手和半身 humanoid 的动力学差异很大，如果全靠一个小 flow expert，容易学出平均化的动作场。

MoF 采用两层结构：

| 层级 | 作用 | 我的理解 |
|---|---|---|
| Foundation experts | 共享底层 transformer blocks | 学 reach、grasp、place、collision avoidance 等可迁移运动原语 |
| Specialized experts | Top-K routed expert blocks | 为本体、任务和局部 action subspace 保留专门容量 |

这个设计的优点是 active parameter count 不随专家总数线性增长；缺点是 routing 是否真的按 embodiment/task 形成可解释分工，论文没有给足够细的 expert specialization 分析。

**3. Hybrid human motion representation**

人类手部轨迹既要保留连续精度，又要吸收大规模 noisy video 的抽象行为先验。论文同时训练两种目标：

```text
continuous action chunk: A in R^{T x d}
discrete motion tokens: z in {1, ..., |C|}^{T_z}
L_act = lambda_1 * L_FM + lambda_2 * L_MASK
```

连续分支使用 flow matching，从 Gaussian noise 运输到目标 action chunk；离散分支把 motion chunk 量化成 codebook tokens，再做 masked motion token prediction。两个 target segment 都能看共享 context，但互相不可见，避免模型直接在 continuous/discrete target 之间抄答案。

**4. ESA、MPG 和 UAC**

| 组件 | 解决的问题 | 机制 |
|---|---|---|
| ESA | post-training 时不同本体梯度冲突 | 每个 semantic action slot 有轻量 adapter，只更新当前 embodiment 激活的 slot |
| MPG | 感知 shift 导致 flow denoising 抖动 | 用 SWD 度量 context feature 与 action prior anchor 的差异，得到 gate `g`；不可靠时压低 feature-conditioned residual，保留 ungated prior offset |
| UAC | 推理延迟和控制频率在不同机器人上不同 | 按 embodiment 的控制周期和延迟预算采样 delay，将 chunk 切成 committed prefix 与 predicted postfix，只对 postfix 训练/写入 |

UAC 的工程含义很实在：action chunk policy 在真实机器人上不是“算完再执行”，而是控制线程持续消费 buffer，推理线程异步补 postfix。这个细节对 10 Hz tabletop 到 50 Hz humanoid 这种异构平台尤其重要。

### 3.3 训练与优化

预训练分为三类任务：

| 任务 | 数据来源 | 监督 | 目的 |
|---|---|---|---|
| Motion generation | human video + robot trajectories | 连续 action chunk / motion token | 学视觉语言条件下的可执行动作 |
| Motion description | human interaction traces | 文本描述 | 让运动和语言语义对齐 |
| Motion continuation | 过去 observation/action history | future motion chunk | 学时序连贯的交互动态 |
| VQA / grounding / planning | VL corpora | 文本答案、坐标、规划判断 | 防止 VLM 语义和空间推理退化 |

post-training 面向目标机器人本体，重点不是从头学 skill，而是用 ESA、MPG、UAC 在有限目标数据下保持跨本体先验可用。真实任务每个 task 收集 30-60 分钟 demonstration。

### 3.4 数据使用与维度追踪

#### 3.4.1 数据清单

| 数据源 | 规模 | 样本单位 | 模态/字段 | 关键维度 | 标签/动作 | 用途阶段 | 处理方式 |
|---|---:|---|---|---|---|---|---|
| Egocentric human video | 16,000 小时；134M human samples | video segment / motion chunk | egocentric RGB, estimated hand pose, text annotations | MANO wrist/finger -> unified action slots；具体 `d` 未报告 | continuous action chunk + discrete motion tokens + text | pretrain motion generation / description / continuation | HaWoR hand pose + camera extrinsics；Gemini-2.5 生成 per-second instruction 与 10s chunk intent；motion-quality filtering；manipulation relevance filtering；left-right mirroring |
| Robot manipulation | 约 14,000 小时；约 1.5B frames；30 embodiments | trajectory / action chunk | multi-view RGB, state, robot action, language | unified state/action `R^d`，`d` 未给固定数；EEF/joint/gripper/base/finger slots | continuous action chunk | pretrain 与 post-train | 聚合 OXE、AgiBotWorld、SO100-Community、InternData-M1、RoboMIND、RoboCOIN、LET 等；dedup；frames downsample 到 30%；sim 占比 capped at 26% |
| Visual-text understanding | 5,000 equivalent hours | image/video QA sample | image/video, question, answer, spatial target | 坐标/点/文本 token；图像分辨率依数据源，论文未统一报告 | text answer / bbox / point / planning judgment | pretrain VQA/grounding/planning | LLaVA 系列、FineVision、LLaVA-Video、RefCOCO、RefSpatial、RoboPoint、ShareRobot、RoboRefit、RoboVQA、MolmoAct、A0-ManiSkill、PixMo-Points、AS-V2 等 |
| UniCraftor 新采集数据 | 超过 200 小时，43 tasks | human-centric demonstration | RGB-D, keyframe events, camera extrinsics | 论文未说明统一输入尺寸；强调同步与标定 | interaction trace / motion supervision | 补充自采与未来扩展 | portable data collection，支持深度、外参、关键事件，设计上可扩展 tactile |
| Real-robot post-training demos | 每 task 30-60 min | task demonstration | 目标机器人相机、state/action、instruction | 依本体 active slots；论文未逐项列出 action dim | executable action chunks | embodiment-specific/single generalist post-train | ESA slot-wise adapter；UAC delay-aware chunking |
| LIBERO fine-tuning/eval | 4 suites，每 task 50 demos；eval 每 suite 500 trials | action chunk sample | wrist + third-person RGB, language, action | RGB `224 x 224`；chunk size 8 | benchmark action | simulation specialist/generalist | packed sequence 7,680 tokens/GPU；effective batch 128；specialist 45k steps on 4xA800 |
| RoboCasa Human-50 | 24 household tasks，每 task 50 human demos；eval 每 task 50 trials | action chunk sample | RGB-only, natural-language command | RGB `224 x 224`；无 depth/point cloud | benchmark action | simulation specialist/generalist | specialist RoboCasa-only；generalist LIBERO+RoboCasa 约 2x steps |

#### 3.4.2 机器人数据构成

UniHand-2.0 的 robot manipulation 表格列了 30 个本体，覆盖 single-arm、dual-arm、portable education arm、half-humanoid 和 humanoid。较大的数据源包括 Agibot-G1 约 2,391.7 小时、Franka 约 2,196.4 小时、Leju Kuavo 约 1,198.2 小时、Google Robot 约 1,195.2 小时、Agilex Split ALOHA 约 1,099.1 小时、Piper 约 904.5 小时、Galaxea R1 Lite 约 630.5 小时、Xarm7 约 594.4 小时。

这张表的价值不只是“数据很大”，而是说明作者确实把 camera view、EEF type、real/sim source 和 embodiment taxonomy 当成训练配方的一部分。对跨本体 VLA 来说，这比单纯报告 episode 数更有信息量。

#### 3.4.3 维度快照

- Observation: 仿真 benchmark 使用 multi-view RGB，`224 x 224`；真实机器人包含 ego、wrist、third-person 等配置，具体 resize/同步频率部分场景未说明。
- Language: QA-style prompt/answer；human data 有 per-second instruction 和 10-second chunk intent；VLM 数据包含 VQA、2D grounding、planning/reasoning。
- State: 统一 `s in R^d`，`d` 为共享 semantic slots 数，论文未给出固定公开数值。
- Action: 统一 `a in R^d`；continuous action chunk `A in R^{T x d}`；EEF relative delta，rotation axis-angle，joint absolute radians；无效 slots 置零/不激活。
- Prediction target: text tokens、continuous flow-matching action、masked discrete motion tokens。
- Control/deployment: LIBERO action chunk size 8；真实部署使用 UAC ring buffer，覆盖约 10 Hz 到 50 Hz 平台；每个本体的 delay distribution 论文未列具体参数。

## 四、实验与结果

### 4.1 实验设置

**真实机器人**覆盖五种硬件：

| Embodiment | Structure | DoF | Hand | Vision |
|---|---|---:|---|---|
| PND Adam-U | bimanual + head + waist | 31 | dexterous 6DoF | ZED Mini movable ego, dual-camera |
| Unitree G1 + LinkerBot O6 | bimanual | 26 | dexterous 6DoF | D435 fixed ego |
| FR3 + Inspire Hand | single-arm | 13 | dexterous 6DoF | 2xD435 fixed third-person |
| BeingBeyond D1 | single-arm | 14 | dexterous 6DoF | D435 movable ego |
| LeRobot SO-101 | single-arm | 6 | gripper | D435 fixed third-person |

真实任务分 spatial、long-horizon、bimanual、generalization 四类，共 10 个任务。评估采用 blind black-box inference server：随机布局、随机 policy、操作者不知道当前 policy 身份，每个 preset configuration 下每个 policy 默认评估 20 trials。

**仿真 benchmark**包括 LIBERO 和 RoboCasa。作者区分 specialist 与 generalist：specialist 只在对应 benchmark 数据上训练，generalist 用 LIBERO+RoboCasa 联合训练并直接在两个 benchmark 上评估。

### 4.2 主要结果

**真实机器人 category-level 成功率**

| 方法 | Spatial | Long Horizon | Bimanual | Generalization |
|---|---:|---:|---:|---:|
| Being-H0.5-specialist | 75 | 60 | 55 | 80 |
| Being-H0.5-generalist | 70 | 60 | 45 | 75 |
| pi0.5 | 55 | 45 | 40 | 65 |
| Being-H0.5-scratch specialist | 50 | 35 | 30 | 60 |
| Being-H0.5-scratch generalist | 35 | 25 | 25 | 50 |

要点：

- specialist 通常最强，但 generalist 在 spatial、long-horizon 上非常接近，说明联合训练没有明显崩。
- 相比 pi0.5，优势主要出现在 long-horizon 和 bimanual，这和作者关于统一 action space + robust chunking 的叙述一致。
- scratch generalist 掉得最明显，支持“UniHand-2.0 不是额外数据，而是跨本体联合优化的先验”的判断。

**LIBERO**

| 方法 | L-Spatial | L-Object | L-Goal | L-Long | Avg. |
|---|---:|---:|---:|---:|---:|
| pi0.5 | 98.8 | 98.2 | 98.0 | 92.4 | 96.9 |
| OpenVLA-OFT | 97.6 | 98.4 | 97.9 | 94.5 | 97.1 |
| X-VLA | 98.2 | 98.6 | 97.8 | 97.6 | 98.1 |
| EO1 | 99.7 | 99.8 | 99.2 | 94.8 | 98.2 |
| Being-H0.5 generalist | 97.0 | 98.2 | 99.0 | 96.2 | 97.6 |
| Being-H0.5 specialist | 99.2 | 99.6 | 99.4 | 97.4 | 98.9 |

最强证据是 Long suite：Being-H0.5 specialist 的 97.4% 和 generalist 的 96.2% 都明显高于 pi0.5 的 92.4%，说明 human-centric pretraining 对长程意图和时序稳定性有实际帮助。

**RoboCasa Human-50**

| 方法 | Modality | Pick & Place | Doors/Drawers | Others | Total Avg. |
|---|---|---:|---:|---:|---:|
| GWM | 3D | 14.8 | 54.3 | 49.8 | 39.3 |
| GR00T-N1 | RGB 256 | 18.6 | 50.2 | 39.1 | 36.0 |
| pi0.5 | RGB 256 | 21.5 | 57.8 | 44.9 | 41.4 |
| pi0 | RGB 256 | 14.0 | 53.1 | 58.5 | 42.4 |
| Being-H0.5 generalist | RGB 224 | 40.0 | 73.0 | 52.0 | 53.3 |
| Being-H0.5 specialist | RGB 224 | 36.0 | 71.7 | 57.6 | 53.9 |

RoboCasa 更能体现系统级价值：它是 long-horizon household tasks，且 Being-H0.5 只用 RGB 224，却超过 3D baselines 和其他 VLA。generalist 与 specialist 仅差 0.6 个百分点，说明跨 benchmark 共享在这个设置里没有明显负迁移。

### 4.3 消融实验

| 消融 | 结论 | 细节 |
|---|---|---|
| Human-centric pretraining | 在低数据/冻结组件设置里最明显 | LIBERO 5-shot single-task 中，冻结 Und+ViT 时平均提升 +25.8 pp；L-Long 提升 +41.6 pp |
| Full fine-tuning | 预训练收益变小 | 可训练参数越多，下游数据越容易覆盖预训练先验；部分简单 object task 甚至有轻微负迁移 |
| Action expert plasticity | action expert 不能过度冻结 | 冻结 0-7 层影响小，超过 14 层后明显掉，完全冻结会低于 20% |
| Masked motion token prediction | 作者认为能提升 noisy human motion 的抽象行为先验 | PDF 文本抽取出的表格数值与正文方向存在疑似排版/解析冲突，因此笔记只保留正文结论，不写具体数值 |
| MPG + UAC | 对 long-horizon 和 bimanual 最关键 | 去掉 UAC 会放大执行延迟导致的误差累积；去掉 MPG 会让 noisy context 下双臂协调更抖 |

## 五、局限性与展望

作者没有把局限性单列成很长一节，但从方法和实验可以看出几个明显边界：

1. **统一 action space 的具体维度和 slot contract 没有完全展开**  
   论文解释了 EEF/joint/gripper/base/finger 的语义原则，但没有像一些工程文档那样给出完整 slot index 表。要复现或迁移到新机器人，还需要代码/adapter 级别细节。

2. **人类视频到机器人动作的映射仍依赖估计器和清洗质量**  
   HaWoR、MANO、Gemini annotation、motion filtering 都是关键链路。任何估计偏差都会影响 human motion prior，尤其是 contact-rich 或遮挡严重的片段。

3. **真实 zero-shot 仍是“信号”，不是成熟能力**  
   作者报告 Adam-U 在未见 task-embodiment pair 上有非零成功和 task-consistent 行为，但也明确说成功率低、精度不可靠。这里更像 scaling direction 的证据，而不是可部署保证。

4. **benchmark 成功率不能完全代表开放世界鲁棒性**  
   LIBERO/RoboCasa 很强，但真实机器人仍只覆盖 5 个平台和 10 个任务。触觉、力控、安全约束、失败恢复、人机混合环境都没有被充分展开。

我的判断是：Being-H0.5 的最大价值在数据和接口哲学，而不是某一个模块。它把 human video pretraining、cross-embodiment slot alignment、flow action expert 和 deployment async control 放在同一个 recipe 里，这对后续 VLA 工程很有参考价值。

## 六、灵魂三问

1. **它解决了什么问题？**

它解决的是跨本体 VLA 的数据稀缺和动作空间碎片化问题。相比每个机器人单独训练或用独立 action head 规避差异，Being-H0.5 试图把人类手部运动和机器人控制映射到同一个物理语义动作空间，用大规模 human-centric 数据支撑低资源本体泛化。

2. **为什么这么做？**

因为人类交互视频的规模和场景覆盖远高于机器人数据，而不同机器人虽然结构不同，但抓取、接触、搬运、放置等操作意图有共享结构。统一 action space 让这些共享结构进入同一个监督接口，MoF 则给差异化本体保留专家容量，避免统一模型被动作空间冲突拖垮。

3. **什么证据最有说服力？**

最有说服力的是两类证据叠在一起：LIBERO Long 和 RoboCasa 这类长程 benchmark 上的高成功率，以及真实机器人中 generalist 接近 specialist、scratch generalist 明显掉队的对比。它们共同说明，UniHand-2.0 和统一接口确实在跨本体联合训练中提供了可迁移先验。

## 七、个人总结

1. Being-H0.5 是一篇典型的 scaling-recipe 论文：数据规模、动作表示、专家容量和部署协议一起构成贡献，单看模型结构会低估它。

2. 最大优势是把 human video 当成一等公民来训练 VLA，并且认真处理了本体异构和真实部署延迟；最大弱点是关键 slot/adapter 的工程细节还不够透明，zero-shot 跨本体仍处早期。

3. 对 VLA 研究来说，这篇适合作为 cross-embodiment pretraining 的重要参照：以后做 human video、统一 action space、Mixture action expert、real-time chunking，都可以拿它来对齐问题定义和实验口径。

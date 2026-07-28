# ABot-M0.5：把移动操作拆成世界预测、潜在动作和可执行控制的 WAM

> 原标题：ABot-M0.5: Unified Mobility-and-Manipulation World Action Model  
> 作者：Ronghan Chen, Yandan Yang, Zuojin Tang, Dongjie Huo, Tong Lin, Haoning Wu, Haoyun Liu, Yuzhi Chen, Lulu Zheng, Botai Yuan 等  
> 机构：AMAP CV Lab, Alibaba Group  
> 发表：arXiv:2607.00678v2，2026-07-06 修订；技术报告日期 2026-07-01  
> 链接：https://arxiv.org/abs/2607.00678  
> 项目页：https://amap-cvlab.github.io/ABot-Manipulation/  
> 开源：https://github.com/amap-cvlab/ABot-Manipulation  
> 本地来源：`Paper/2607.00678v2.pdf`

---

## 一、研究背景与动机

移动操作（mobile manipulation）比桌面机械臂难的地方不只是轨迹更长，而是导航和操作的时间尺度、动作空间和误差传播机制都不一样。传统 VLA policy 大多是 reactive policy：看当前图像和语言，直接输出动作 chunk。这个范式在短程 tabletop manipulation 上可以工作，但面对厨房/家庭这类移动操作时会缺少显式未来建模，容易丢失长程上下文。

World Action Model（WAM）试图通过预测未来视频再反推出动作来解决长时程问题，但作者认为已有 WAM 和移动操作之间有三个结构错位：

1. **时间粒度错位**：视频预测通常是 coarse chunk，真实控制需要 frame/step 级别的接触、闭合、对齐和释放。
2. **动作空间错位**：底盘移动和机械臂操作的频率、物理约束和 loss landscape 不同，放在一个 action head 里容易互相干扰。
3. **训练-推理条件错位**：训练 inverse dynamics 时常用 ground-truth future video，推理时只能用模型自己预测出来的 future video，长程 rollout 会累积误差。

ABot-M0.5 的目标就是围绕这三个错位重新设计 WAM：用中间 latent action 对齐视频和动作的粒度，用双层 Mixture-of-Transformers 解耦动作子空间，再用 Dream Forcing 让 inverse dynamics 在训练时就看到模型自己“梦”出来的未来。

## 二、核心贡献

1. **提出移动操作 WAM 的三重对齐视角**：把已有方法的失败归因到 temporal granularity、action space、train-test consistency 三个结构问题，而不是简单归因于模型不够大。

2. **设计 Video → Latent Action → Action 级联生成结构**：在 coarse video latent 和 executable action 之间插入 frame-level latent action，把局部视觉状态转移作为更细粒度、更弱本体绑定的动作表示。

3. **提出 Dual-Level Mixture-of-Transformers（D-MoT）**：第一层解耦 video latent、latent action、executable action 三种 token 流；第二层在 action 内部解耦 mobility action 和 manipulation action，同时保留 joint self-attention 做协调。

4. **提出 Dream Forcing 训练策略**：SFT 后期不再只用真实未来 latent 监督动作，而是先让模型生成 dreamed video/latent-action，再用这些预测 latent 训练 inverse dynamics，减小 rollout 时的 exposure bias。

## 三、方法原理

### 3.1 整体框架

ABot-M0.5 建在 Wan2.2 5B video diffusion backbone 上。输入是语言指令和多视角机器人观测，输出包括三个变量：

```text
future video latent z_{t+1}
frame-level latent action m_t
executable robot action a_t
```

核心生成顺序是：

```text
历史观测/语言 -> z_{t+1} -> m_t -> a_t
```

这比“图像 token 直接接 action head”多了一层显式未来建模和一层局部运动抽象。直觉上：

- `z_{t+1}` 负责回答“场景下一步大概会变成什么样”；
- `m_t` 负责回答“局部视觉状态发生了什么运动变化”；
- `a_t` 负责回答“当前机器人身体要怎么动才能实现这个变化”。

三个阶段都用 Conditional Flow Matching（CFM）训练。以 video latent 为例，模型从噪声 `epsilon` 和干净 latent `z_t` 的插值点预测速度场，loss 是预测速度与目标速度 `z_t - epsilon` 的 MSE。latent action 和 action 也沿用同样的 flow matching 思路，只是条件变量逐步增加。

### 3.2 Intermediate Latent Action

这篇最值得注意的是 latent action 的角色。它不是机器人控制器里的真实动作，也不是语言 action token，而是从连续帧视觉变化中抽出来的局部运动表征：

```text
m_t = E_m(I_t, I_{t+1})
```

作者采用 ALAM 风格的 latent action encoder。训练时给定时间有序的三帧 `o_i, o_j, o_k`，要求 transition embedding 满足近似代数结构：

```text
L_add = ||m_i^k - (m_i^j + m_j^k)||^2
L_rev = ||m_i^j + m_j^i||^2
```

直觉是：从 i 到 k 的长转移应该能被 i 到 j、j 到 k 的短转移相加；反向转移应该抵消正向转移。这个设计让 latent action 更像一个可组合的视觉运动空间，而不是任意压缩码。

训练完成后，只保留 frozen latent action encoder，用它离线给机器人轨迹生成 latent-action label。这样大量没有机器人 action label 的视频也能参与 motion abstraction 学习。对预训练来说，这个点很关键：它把“看视频学运动先验”和“用机器人动作学控制”之间接了一层可监督的桥。

### 3.3 Dual-Level MoT

D-MoT 有两层解耦。

第一层是 **modality-level disentanglement**。video latent、latent action、executable action 各有独立 input projection、timestep embedding 和 output head，但在 Transformer 里通过 self-attention 交换信息。这样可以避免三种 token 语义混在一起，又保留跨模态条件化。

第二层是 **action-level disentanglement**。移动操作里的 action 被分成：

```text
a_t = [a_t^move, a_t^manip]
```

底盘移动是低频、全局、导航式动作；机械臂操作是高频、局部、接触敏感动作。ABot-M0.5 给这两个子空间分配不同 FFN 和 prediction head，减少梯度干扰；但 self-attention 仍然共享，所以模型可以学“底盘移动到哪个位置才方便下一步抓取”这种协调关系。

### 3.4 Dream Forcing

标准 WAM 训练 action 时常用 teacher forcing：

```text
a_t ~ p_a(. | z_{<=t+1}, m_{<=t}, a_{<t}, l)
```

这里的 `z_{t+1}` 和 `m_t` 是 ground truth future latent。问题是推理时没有这些真值，模型只能先预测 future latent，再基于预测 latent 出动作。如果 inverse dynamics 只在干净未来上训练，推理时一遇到 blur、object drift、hallucination，动作就会不稳。

Dream Forcing 把动作条件改成：

```text
a_t ~ p_a(. | z_hat_{t+1}, z_{<=t}, m_hat_t, m_{<t}, a_{<t}, l)
```

也就是训练时先 forward 一次得到 dreamed future video latent 和 dreamed latent action，再第二次 forward 用这些预测 latent 训练 action。它不是把 video model 和 action model 分开，而是在同一个 WAM 里让 inverse dynamics 适应自己的未来预测误差。

### 3.5 训练与优化

训练 recipe 分成三段：

| 阶段 | 训练内容 | 目标 |
|---|---|---|
| World Model Pretraining | 从 Wan2.2 5B 初始化，全参数微调 action-unconditioned future video predictor | 让视频世界模型适应机器人多视角、物体交互和移动视角 |
| Latent Action Model Pretraining | 用 ALAM 风格自监督训练 `E_m`，之后冻结作为离线 label extractor | 从视觉转移中抽取 frame-level motion intent |
| Progressive SFT | SFT1 用真实未来 latent 做稳定联合训练；SFT2 用 Dream Forcing 做 rollout 对齐 | 学 inverse dynamics，并让训练条件靠近推理条件 |

预训练数据来自 OXE、OXE-AugE、Agibot-Beta、RoboCOIN、RoboMind、Galaxea、InternData-A1，并额外包括 RoboNet、BridgeData V2、DROID 等公开数据。作者强调 Galaxea 提供 base mobility 相关数据，InternData-A1 提供合成规模，RoboCOIN/RoboMind 提供跨本体和双臂覆盖。

多视角处理也比较工程化：固定四个 canonical video slots，前两个放第三视角，后两个放 wrist views；多于四个随机采样，少于四个 zero padding，并用 attention mask 排除 padded view。这个设计降低了不同机器人相机配置带来的语义混乱。

总的 SFT loss 是：

```text
L_SFT1 = lambda_z L_z + lambda_m L_m + lambda_a L_a
L_SFT2 = lambda_z L_z + lambda_m L_m + lambda_a L_tilde_a
```

其中 `L_tilde_a` 就是 Dream Forcing 下基于 dreamed latent 的 action CFM loss。这里的 action loss 接在 action branch / inverse dynamics 上；video latent 和 latent action 也各自有 flow matching loss。

## 四、实验与结果

### 4.1 实验设置

ABot-M0.5 在四类环境上评估：

| Benchmark | 作用 |
|---|---|
| RoboCasa365 | 移动操作主 benchmark，包含 atomic / composite / unseen 任务 |
| RoboTwin 2.0 | 多任务双臂操作，含 clean 和 randomized hard setting |
| LIBERO / LIBERO-Plus | 桌面组合操作和视觉扰动鲁棒性 |
| 真实机器人 | Agilex Piper 6-DoF 单臂平台，每任务 50 条真实 demo |

主要指标是 success rate；真实机器人额外报告 process score。

### 4.2 主要结果

**RoboCasa365 pretraining setting：ABot-M0.5 对 composite seen 提升最明显。**

| Method | Average | Atomic-Seen | Composite-Seen | Composite-Unseen |
|---|---:|---:|---:|---:|
| Diffusion Policy | 6.1 | 15.7 | 0.2 | 1.3 |
| π0 | 14.8 | 34.6 | 6.1 | 1.1 |
| π0.5 | 16.9 | 39.6 | 7.1 | 1.2 |
| GR00T-N1.5 | 23.9 | 50.7 | 14.8 | 2.7 |
| RLDX-1 | 33.2 | 63.0 | 27.5 | 5.4 |
| Qwen-RobotManip | 35.9 | 68.6 | 20.1 | 14.9 |
| ABot-M0.5 | 40.4 | 75.9 | 38.3 | 2.7 |
| ABot-M0.5 + Condensed Memory | 46.6 | 79.4 | 48.3 | 7.9 |

ABot-M0.5 的优势主要来自 atomic-seen 和 composite-seen；composite-unseen 不如 Qwen-RobotManip，说明它的长程结构和控制强，但开放语义/新组合泛化仍不是绝对优势。Condensed Memory 提升很大，但论文说细节未来再展开，所以这里不能把它当成完全可复现贡献。

**RoboCasa365 target setting：完整数据和 10% 数据都超过 GR00T-N1.5。**

| Setting | Method | Atomic-S | Composite-S | Composite-U | Average |
|---|---|---:|---:|---:|---:|
| Target 100% | GR00T-N1.5 | 60.6 | 35.0 | 33.3 | 43.7 |
| Target 100% | Fast-WAM | 59.1 | 36.4 | 33.2 | 43.5 |
| Target 100% | Lingbot-VA | 63.5 | 37.3 | 32.1 | 45.1 |
| Target 100% | ABot-M0.5 | 70.6 | 44.3 | 45.6 | 54.2 |
| Target 10% | GR00T-N1.5 | 38.7 | 11.0 | 11.2 | 21.0 |
| Target 10% | ABot-M0.5 | 49.0 | 23.4 | 15.4 | 30.1 |

这组最能体现预训练价值：在 Target 10% 下，ABot-M0.5 仍然有 30.1 平均成功率，而 GR00T-N1.5 是 21.0。

**RoboTwin 2.0：非移动双臂操作上也接近或刷新 SOTA。**

| Model | Clean (Easy) | Randomized (Hard) | Average |
|---|---:|---:|---:|
| π0.5 | 82.70 | 76.80 | 79.75 |
| ABot-M0 | 86.06 | 85.08 | 85.57 |
| Qwen-VLA | 86.10 | 87.20 | 86.65 |
| Fast-WAM | 91.90 | 91.80 | 91.85 |
| Qwen-RobotManip | 93.70 | 94.00 | 93.85 |
| ABot-M0.5 | 94.00 | 94.20 | 94.10 |

这说明 latent action 和 rollout alignment 不只对底盘移动有效，也能改善 contact-rich manipulation。

**LIBERO / LIBERO-Plus：桌面任务达到强竞争水平。**

| Benchmark | ABot-M0.5 |
|---|---:|
| LIBERO average | 99.4 |
| LIBERO-Plus total zero-shot | 83.4 |

LIBERO-Plus 上它超过 Fast-WAM 51.5、ImageWAM 83.1、Cosmos-Policy 82.2，但低于 Qwen-RobotManip-Context 的 91.4。这个对比很有信息量：WAM 路线可以强鲁棒，但 VLM-heavy 的 semantic grounding 在某些扰动上仍有优势。

**真实机器人：每任务 50 条 demo，长程任务比 Fast-WAM 稳。**

| 任务 | ABot-M0.5 结果 |
|---|---:|
| Peg Cylinder | 70% SR / 96% process score |
| Organize Plate | 70% SR |
| Arrange Fruits | 80% SR |
| Cup Stacking | 80% SR |
| Arrange Flower | 60% SR |

Peg Cylinder 对细粒度插入很敏感，ABot-M0.5 高于 π0.5 的 50% SR 和 Fast-WAM 的 30% SR。长程任务上 Fast-WAM 只有 20%-40%，ABot-M0.5 更稳定，作者将其归因于 latent action 和 Dream Forcing 对误差累积的缓解。

### 4.3 消融实验

**latent action 结构消融：直接 video-to-action 明显不够。**

| Training Strategy | Drop | RoboTwin Clean SR |
|---|---:|---:|
| Baseline | - | 87.60 |
| 2-Stage Separate | 0 | 90.86 |
| 2-Stage Channel Concat | 0 | 91.06 |
| 3-Stage Separate | 0.2 | 91.06 |
| 3-Stage Separate | 0 | 94.00 |

最关键的是 3-Stage Separate：video、latent action、action 使用独立 temporal/modal indicators，并通过 attention mask 防止 video token 偷看 latent action token。`p_drop=0.2` 反而下降，说明在这个 cascade 里，训练时随机丢 latent action 会制造推理时不存在的不稳定条件。

**Action-Decoupled MoT：移动操作 composite seen 从 0.34 到 0.48。**

作者在 RoboCasa365 Composite-Seen 子集上对比单一 action modality transformer 和 action-decoupled MoT。分开 mobility / manipulation 后成功率 0.48，高于 baseline 的 0.34，同时收敛更快。这直接支持“底盘和机械臂不要共用一个均质 action head”的判断。

**Dream Forcing：只多 5k steps，就比继续 SFT1 更好。**

| Training Stage | Training Steps | Atomic-Seen |
|---|---:|---:|
| SFT1 Base | 50k | 67.55 |
| SFT1 | +5k / 55k | 66.78 |
| SFT1 | +10k / 60k | 68.90 |
| SFT2 + Dream Forcing | +5k / 55k | 70.56 |

这个消融很干净：同一个 50k warm-start，继续原训练甚至短期下降，而 Dream Forcing 5k steps 提升到 70.56。它说明收益不是“多训一点”，而是 conditioning distribution 变了。

**预训练 + SFT：Target 10% 下从 17.8 到 49.0。**

同样架构和 SFT protocol 下，从 Wan2.2 直接 SFT 只有 17.8；经过 ABot-M0.5 预训练后再 SFT 是 49.0，差 31.2 个点。attention map 也显示，预训练后模型更关注 robot arm 和 interaction region，SFT 再把这些 interaction priors 对齐到目标语义。

## 五、局限性与展望

作者提到的未来方向包括扩展到更广泛真实数据和更多机器人本体、研究 WAM scaling laws、设计更高效的 memory mechanism、优化推理延迟以支持 edge deployment。

我的补充判断：

1. **Composite-Unseen 仍是短板**：RoboCasa365 pretraining 中 ABot-M0.5 的 Composite-Unseen 只有 2.7，低于 Qwen-RobotManip 的 14.9；即使加 Condensed Memory 也只有 7.9。WAM 结构改善长程执行，但没有自动解决开放组合语义。

2. **Condensed Memory 结果暂时不能完全复现**：表 2 里 +Condensed Memory 从 40.4 到 46.6，但论文说细节未来工作再展开。这个数字可以记录，但不能作为当前 technical recipe 的核心证据。

3. **真实实验平台仍偏单臂**：虽然论文主题是 mobility-and-manipulation，但真实部署是 Agilex Piper 6-DoF 单臂平台，且每任务 50 条 demo。对真正移动底盘 + 双臂 + 多房间任务的泛化还需要更多实证。

4. **预训练数据工程依赖很强**：多源机器人数据、synthetic data、固定视角 slot、offline latent action label、Wan2.2 视频骨干，全都需要较重工程基础。普通实验室复现完整 recipe 仍然困难。

## 六、灵魂三问

1. **它解决了什么问题？**

它解决的是移动操作里 WAM/VLA 的结构错位问题。不是单纯“让模型看更多机器人数据”，而是把 coarse world prediction、fine motion intent 和 embodiment-specific control 分层，让长程未来预测能真正服务低层动作。

2. **为什么这么做？**

因为直接从 video latent 到 robot action 会把三个不同层级压到同一个映射里：视频太粗、动作空间太杂、训练时 future 太干净。latent action 负责补上细粒度视觉运动，D-MoT 负责拆开 mobility/manipulation，Dream Forcing 负责让 action branch 习惯自己预测的未来。

3. **什么证据最有说服力？**

最有说服力的是两组消融：Target 10% 下预训练 + SFT 从 17.8 提到 49.0，说明预训练确实学到可迁移 interaction prior；Dream Forcing 从同一 50k checkpoint 继续 5k steps 达到 70.56，而普通 SFT1 继续 5k 只有 66.78，说明 train-test alignment 不是装饰。

## 七、个人总结

1. ABot-M0.5 的核心 idea 是把移动操作的动作生成拆成 **world dynamics → latent motion intent → executable action**，再用训练策略保证这条链在推理时也能稳定工作。

2. 最大优势是 recipe 非常成体系：latent action、action decoupling、Dream Forcing 都对准具体失败模式；最大弱点是开放组合泛化和真实移动操作覆盖还没完全证明，尤其 Composite-Unseen 结果并不强。

3. 对后续 VLA 基座策略来说，这篇最值得借的是“中间动作表征”这条线：动作预训练不一定只能监督机器人 joint/EEF action，也可以先学视觉状态转移里的 motion algebra，再把它接到具体本体动作上。

# MiVLA：用人机互相模仿把人类视频和仿真机器人数据接到 VLA 预训练里

> 原标题：MiVLA: Towards Generalizable Vision-Language-Action Model with Human-Robot Mutual Imitation Pre-training  
> 作者：Zhenhan Yin, Xuanhan Wang, Jiahao Jiang, Kaiyuan Deng, Pengqi Chen, Shuangle Li, Chong Liu, Xing Xu, Jingkuan Song, Lianli Gao, Heng Tao Shen  
> 机构：Tongji University；University of Electronic Science and Technology of China  
> 发表：arXiv:2512.15411v2，2025-12-19 修订  
> 链接：https://arxiv.org/abs/2512.15411  
> 项目/开源：论文与 arXiv 页面未提供明确项目页或代码链接  

---

## 一、研究背景与动机

VLA 的瓶颈仍然是大规模真实机器人数据太贵、太慢、覆盖不够广。π0、π0.5 这类模型证明了真实机器人数据预训练可以带来强泛化，但这种路线需要上万小时机器人 episode，对普通实验室很难复现。

MiVLA 的核心问题是：能不能不依赖大规模真实机器人预训练数据，而用更容易扩展的两类数据替代？

- **仿真机器人数据**：有精确 robot action、任务和 embodiment 可以扩展，但有 Sim2Real gap。
- **人类视频/人类演示数据**：来自真实场景、日常任务丰富，但手和机械臂之间存在相机视角、外观和形态差异。

作者认为，这两类数据的共同点是“行为结构”：人手和机器人末端执行器都在完成抓取、移动、放置、调整等操作。如果能把 human action space 和 robot action space 双向对齐，就可以让同一个 VLA 既学仿真机器人操作多样性，又吸收真实人类行为先验。

这篇论文的目标不是提出一个全新的大 VLM backbone，而是提出一个**人机互相模仿预训练（Human-Robot Mutual Imitation Pre-training）** recipe：把人类演示和仿真机器人演示都转成带互补动作标签的数据，然后训练 diffusion-based VLA 预测本 embodiment 的动作，同时模仿另一个 embodiment 的动作。

## 二、核心贡献

1. **提出 MiVLA 预训练范式**：用 human-to-robot 和 robot-to-human 的互相模仿目标，把人类视频的真实场景行为先验与仿真机器人数据的可控动作监督放入同一个 VLA。

2. **设计双向人机动作空间转换**：以人类 thumb knuckle 和机器人 end-effector pose 为对齐锚点，结合左右手坐标系旋转、IK 求解和人体解剖先验，生成跨 embodiment 的互补动作标签。

3. **采用连续动作 diffusion transformer 架构**：观测侧用 DINOv2、SigLIP、T5 和 proprioception MLP tokenizers，动作侧用 flow-matching diffusion transformer 解码连续 action chunk。

4. **在仿真和三种真实机器人上验证泛化**：RoboTwin-2.0 代表性 20 任务中达到 69% easy / 66% hard，真实 PiPer、ARX、LocoMan 三任务平均 full-task SR 为 55%，相对多个 VLA baseline 有竞争力。

## 三、方法原理

### 3.1 整体框架

MiVLA 的输入是视觉观测、语言指令和 proprioceptive state，输出未来一段 action chunk。它把数据分为两类：

| 数据类型 | 记号 | 内容 | 原生动作 |
|---|---|---|---|
| 仿真机器人演示 | `D_r = {l_r, v_r, a_r}` | 语言指令、机器人视角序列、机器人动作 | robot joint / EEF action |
| 人类演示 | `D_h = {l_h, v_h, a_h}` | 语言描述、人类视觉观测、人类动作 | wrist、thumb、finger 等 human action |

普通 VLA 只学：

```text
P_theta(A_r | O_t^r), P_theta(A_h | O_t^h)
```

MiVLA 则要求模型在看到一个 embodiment 的观测时，同时预测本 embodiment 的未来轨迹，并模仿另一个 embodiment 的轨迹：

```text
P_theta(A_r, A_hat_h | O_t^r), P_theta(A_h, A_hat_r | O_t^h)
```

这就是“互相模仿”的关键：机器人演示不只监督机器人动作，还生成一份人类动作标签；人类演示不只监督人类动作，还生成一份机器人动作标签。于是每条数据都被扩展成跨 embodiment 的监督信号。

### 3.2 预训练架构

MiVLA 架构由三类 observation tokenizers 和一个 diffusion action decoder 组成。

| 模块 | 作用 | 关键设置 |
|---|---|---|
| Vision tokenizers | 编码图像观测 | DINOv2 + SigLIP；每帧 `224 x 224`；共 392 个视觉 token |
| Language tokenizer | 编码任务指令 | T5 tokenizer / encoder |
| State projectors | 编码 proprioception | 三个 MLP，把状态投影成固定数量 token |
| Action decoder | 生成连续动作块 | diffusion transformer；self-attention / cross-attention；flow matching 去噪 |

这个架构的判断比较明确：视觉语言理解不从零训练，直接借用成熟视觉/语言 encoder；真正为机器人控制服务的是 diffusion transformer action decoder。它把 noisy action chunk 作为输入，把 vision/language/state token 作为条件，逐步估计噪声并恢复干净的连续动作。

相比 OpenVLA、RT-2 那种离散 action token autoregressive 预测，MiVLA 选择 diffusion/flow-matching 的理由是连续控制更自然：机械臂关节、EEF pose、手指轨迹都不是天然离散 token，离散化会带来精度损失和 action convention 负担。

### 3.3 动作空间设计

MiVLA 没有把所有 embodiment 强行压成一个物理完全一致的动作语义，而是建立了一个包含 human-specific、robot-specific 和 common EEF 的统一动作空间。

| 动作空间 | 维度 | 含义 |
|---|---:|---|
| Human joints | 48 | 双手 wrist pose：每只手 3D position + 6D orientation，共 18 维；所有 fingertips 的 3D position，共 30 维 |
| Robot joints | 14 | 双臂每臂 6 个关节 + 1 个 gripper |
| End-effector pose | 14 | 每个 embodiment 的 3D position + 4D quaternion，双侧共 14 维 |

这里最值得注意的是 human action 选择：作者没有直接预测完整人体姿态，而是围绕手腕、thumb knuckle、fingertips 建模。这和 manipulation 任务的接触点更相关，也让 human-to-robot 转换可以落到机器人 EEF 和 gripper/joint 空间。

### 3.4 数据处理链路：人机动作双向转换

MiVLA 的数据处理不是简单把 human videos 和 simulated robot data 混在一起，而是先把二者变成“互补动作标签”。

**1. 机器人演示转人类动作（Robot-to-Human）**

给定机器人 EEF pose，作者把 EEF pose 当作人手 thumb knuckle 的参考点，再用机器人坐标系到人手坐标系的旋转矩阵 `R_m` 做坐标转换：

```text
a_h^{thumb} = R_m(a_m^{eef})
```

随后用经验函数 `f_d(.)` 根据人体解剖先验估计 thumb 与其他 fingers 的距离，得到人类手部关节/指尖轨迹。直觉上，这是把机器人“末端执行器如何移动”翻译成人手“拇指和手指大概如何运动”。

**2. 人类演示转机器人动作（Human-to-Robot）**

给定初始机器人 EEF pose，作者用人类 thumb knuckle 相对初始帧的位移作为动作核心，再通过人手坐标系到机器人坐标系的旋转矩阵 `R_h` 转到机器人空间：

```text
a_r^{eef}(t) = a_r^{eef}(0) + R_h(a_h^{thumb}(t) - a_h^{thumb}(0))
```

之后通过 PyBullet 的 optimization-based IK solver，把目标 EEF pose 求成机器人关节动作。直觉上，这是把人手“相对怎么动”映射成机器人“末端应该怎么动”，再用 IK 补上具体关节角。

**3. 每条数据生成双标签**

处理后，数据从单一监督变成双监督：

| 原始数据 | 原生标签 | 合成互补标签 | 训练信号 |
|---|---|---|---|
| 仿真机器人演示 | `A_r*` | `A_hat_h*` | 预测机器人动作 + 模仿人类动作 |
| 人类演示 | `A_h*` | `A_hat_r*` | 预测人类动作 + 模仿机器人动作 |

预训练损失为：

```text
L = l_r2h + l_h2r
l_r2h = ||A_r - A_r*||^2 + ||A_hat_h - A_hat_h*||^2
l_h2r = ||A_h - A_h*||^2 + ||A_hat_r - A_hat_r*||^2
```

论文里 Eq. 6 对 `A_h*` 来源的表述疑似有笔误：它写成 robot demonstrations `D_r`，按上下文应为 human demonstrations `D_h`。

### 3.5 预训练和微调细节

论文给出的训练细节如下：

| 阶段 | 设置 |
|---|---|
| 预训练硬件 | 4 x A100 |
| Optimizer | AdamW |
| Batch size | 每 GPU 32，总 batch size 128 |
| Learning rate | `1e-4` |
| Weight decay | `0.01` |
| Scheduler | constant learning rate + warmup |
| Precision | bf16 mixed precision |
| 微调硬件 | 2 x A100 |
| 微调 batch size | 每 GPU 16，总 batch size 32 |

数据规模方面，作者在结果分析中提到 MiVLA 使用的是“中等规模 mixed data”，约 900 小时，远少于 π 系列超过 10,000 小时真实机器人数据。但论文没有详细披露这 900 小时的来源、human/simulation 比例、过滤规则、动作标注生成质量控制或数据配比 curriculum。对复现来说，这是这篇文章最明显的信息缺口。

下游评测训练数据需要和预训练数据区分：

| 场景 | 数据设置 |
|---|---|
| RoboTwin-2.0 仿真评测 | 50 个任务，每任务 50 条 demonstration，共 2500 条用于训练；测试选 20 个代表任务，含 12 hard、4 middle、4 easy |
| 真实机器人评测 | PiPer、ARX、LocoMan 三个 embodiment；每个真实任务收集 30 条 demonstration 做 post-training |

所以论文的“without real-world robot data”更准确地说是：**预训练阶段不依赖大规模真实机器人数据**；真实机器人结果仍然进行了少量任务级 post-training。

## 四、实验与结果

### 4.1 实验设置

仿真评测使用 RoboTwin-2.0，包含 easy 和 hard 两种模式。hard 模式引入 domain randomization，包括背景/纹理随机化、桌面 distractor、光照变化，以及工作台高度约 `±3 cm` 的几何扰动。

真实机器人评测包含三种形态：

| Robot | 形态 | 任务 |
|---|---|---|
| AgileX PiPer | 单臂 6-DoF | Move bottle onto pad |
| ARX-5 | 单臂 6-DoF | Tidy up umbrella rack |
| LocoMan | 四足底盘 + 轻量双臂组合平台 | Gathering scattered objects |

baseline 包括 ACT、π0、π0.5、H-RDT。作者使用 open-sourced baseline weights，并在相同数据和训练配置下 fine-tune。

### 4.2 主要结果

**RoboTwin-2.0 代表性 20 任务：MiVLA 在 easy/hard 都领先。**

| 方法 | Easy Avg. SR | Hard Avg. SR |
|---|---:|---:|
| ACT | 9% | 8% |
| π0 | 23% | 25% |
| π0.5 | 35% | 53% |
| H-RDT | 36% | 43% |
| MiVLA | 69% | 66% |

这个结果说明 mutual imitation 预训练对 domain randomization 下的稳健性有帮助。尤其 hard 模式中，MiVLA 比 π0.5 高 13 个百分点，比 H-RDT 高 23 个百分点。

**RoboTwin-2.0 全 50 任务：优势仍然存在，但不是所有任务都成功。**

| 方法 | Easy Avg. SR | Hard Avg. SR |
|---|---:|---:|
| ACT | 5.4% | 5.4% |
| π0 | 19.1% | 21.7% |
| π0.5 | 33.6% | 53.8% |
| H-RDT | 27.4% | 37.2% |
| MiVLA | 62.0% | 63.6% |

全 50 任务结果比 20 任务更能说明泛化，但也暴露了边界：例如 `scan object` 上所有方法都是 0%，`stack blocks three` 上 MiVLA 也几乎失败。这意味着 MiVLA 的强项更偏向常规抓取、移动、放置和简单组合操作，不等于解决了所有 long-horizon 或感知依赖任务。

**真实机器人：MiVLA 平均最好，但 π0.5 在单臂任务上更强。**

| 任务 / Robot | π0 SR | π0.5 SR | H-RDT SR | MiVLA SR |
|---|---:|---:|---:|---:|
| Move bottle onto pad / PiPer | 20% | 66% | 36% | 54% |
| Tidy up umbrella rack / ARX | 60% | 75% | 45% | 60% |
| Gathering scattered objects / LocoMan | 50% | 20% | 0% | 50% |
| Average | 43% | 54% | 27% | 55% |

作者的解释是：π0、π0.5 用超过 10,000 小时真实机器人数据预训练，并且数据中包含 PiPer、ARX 这类单臂平台，所以单臂任务适应能力更强；而 LocoMan 是四足 + 双臂的组合 embodiment，对这些模型更陌生，MiVLA 的跨 embodiment 互相模仿在这里更有优势。

完整性和时间也支持这个判断：MiVLA 平均 sub-task completeness 为 69%，平均完成时间 43.0 秒；π0.5 的 completeness 为 64%，时间 54.4 秒。

### 4.3 消融实验

**预训练目标消融：双向 mutual imitation 最关键。**

| 设置 | RoboTwin-2.0 | PiPer | ARX | LocoMan |
|---|---:|---:|---:|---:|
| From scratch | 37% | 0% | 25% | 0% |
| Human pre-train | 43% | 36% | 60% | 0% |
| `l_h2r` | 46% | 30% | 49% | 20% |
| `l_h2r + l_r2h` | 66% | 54% | 60% | 50% |

单独 human pre-train 有帮助，但对 LocoMan 为 0%；加入 human-to-robot 后 LocoMan 到 20%；完整双向目标到 50%。这说明收益不是“多塞了一些人类视频”这么简单，而是跨 embodiment 的互补动作标签确实改变了模型学到的动作先验。

**Few-shot adaptation：20 条左右演示已经能明显受益。**

| 设置 | Move bottle 10 | Move bottle 20 | Move bottle 30 | Umbrella 5 | Umbrella 10 | Umbrella 20 |
|---|---:|---:|---:|---:|---:|---:|
| From scratch | 0% | 0% | 0% | 10% | 10% | 25% |
| Human pre-train | 0% | 14% | 36% | 5% | 40% | 45% |
| `l_h2r` | 12% | 26% | 30% | 35% | 50% | 50% |
| `l_h2r + l_r2h` | 6% | 36% | 56% | 25% | 60% | 55% |

完整 mutual imitation 在 20/30 shot 上优势明显；但 10-shot move bottle 反而不如单向 `l_h2r`，说明双向目标不是在所有低数据点都稳定最优，可能受任务动作分布和合成标签噪声影响。

**泛化评测：主要增强 location generalization，对 object/scene 泛化仍有限。**

| 设置 | Seen | Unseen locations | Unseen objects | Unseen scenes | Average |
|---|---:|---:|---:|---:|---:|
| From scratch | 0% | 0% | 0% | 0% | 0% |
| MiVLA-H2R full | 40% | 33% | 20% | 13% | 30% |
| MiVLA full | 75% | 50% | 30% | 38% | 54% |

MiVLA 对 unseen location 的提升最稳定；unseen object 和 unseen scenes 仍有明显下降。这和作者在 limitation 里承认的 OOD failure mode 一致：新物体形状/纹理、极端初始姿态和强干扰背景仍会破坏抓取和目标理解。

## 五、局限性与展望

作者明确指出 MiVLA 在 OOD 场景中仍有三类失败：新物体、新初始位姿、强干扰背景。尤其对于训练分布外的形状和纹理，模型可能生成不合适的抓取姿态；对于杂乱或异常初始位置，策略可能无法找到可行轨迹；即使做了 domain randomization，显著背景干扰仍可能让模型误解任务目标。

我的补充判断有四点：

1. **预训练数据细节不足**：论文只给出约 900 小时 mixed data 的笼统描述，缺少来源、比例、过滤、标注质量和 curriculum，导致最核心的 pre-training recipe 难以复现。

2. **动作转换依赖几何近似**：thumb knuckle 与 EEF 的锚点选择很实用，但 gripper 与人手多指接触并不等价。对精细灵巧操作、多指接触和力控任务，这个映射可能过粗。

3. **语义理解不如 VLM-heavy 路线**：作者也承认 π0.5 在部分复杂语义 OOD 场景更好，因为它继承了大规模 VLM 的 commonsense 和 semantic grounding。MiVLA 的 diffusion transformer 更擅长视觉-运动模仿，但“what/why”理解较弱。

4. **真实机器人仍需少量 post-training**：MiVLA 的结果不是 zero-shot real robot deployment，而是每个真实任务有 30 条 demonstration。它降低的是大规模真实机器人预训练依赖，不是完全取消真实机器人数据。

未来更有价值的方向是把 MiVLA 这种 cross-embodiment action pretraining 和 VLM semantic grounding 结合起来：VLM 负责目标、affordance、错误恢复和高层计划，diffusion action decoder 负责连续控制，human-robot mutual imitation 负责补足真实机器人数据稀缺。

## 六、灵魂三问

1. **它解决了什么问题？**

它解决的是 VLA 预训练过度依赖大规模真实机器人数据的问题。相比直接用上万小时 robot demos 训练，MiVLA 试图用更容易获得的仿真机器人数据和人类演示数据，通过人机动作双向转换构造跨 embodiment 监督，从而获得可迁移的操作先验。

2. **为什么这么做？**

因为仿真数据有精确动作和多机器人形态，但缺真实场景；人类视频有真实日常行为和场景覆盖，但没有机器人原生动作。mutual imitation 把这两者互补起来：每条机器人数据生成一份人类动作标签，每条人类数据生成一份机器人动作标签，让模型学习“同一行为在不同身体里长什么样”。

3. **什么证据最有说服力？**

最有说服力的是预训练目标消融和 LocoMan 真实机器人结果。完整 `l_h2r + l_r2h` 在 RoboTwin-2.0 从 46% 提到 66%，LocoMan 从单向 20% 提到 50%；真实三平台平均 SR 为 55%，略高于 π0.5 的 54%，而使用的数据规模约 900 小时，远低于 π 系列超过 10,000 小时真实机器人数据。

## 七、个人总结

1. MiVLA 的核心 idea 是：不要把 human data 和 sim robot data 只当成两个 domain，而是通过显式人机动作映射，把它们变成可互相监督的 cross-embodiment action data。

2. 最大优势是预训练 recipe 很清楚地服务于“少真实机器人数据”的目标，尤其对新 embodiment 有启发；最大弱点是数据细节披露不足，且 human hand 到 robot gripper 的转换仍是粗几何近似。

3. 对后续 VLA 研究来说，这篇论文值得和 H-RDT、EgoVLA、Qwen-VLA、π0.5 放在一起看：MiVLA 强在 action-space bridge，Qwen-VLA 强在统一大模型和异构数据 recipe，π0.5 强在真实机器人数据规模和 VLM 语义泛化。真正稳的下一代系统大概率要把这三条线合起来。

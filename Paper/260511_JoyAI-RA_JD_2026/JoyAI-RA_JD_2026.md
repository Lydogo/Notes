# JoyAI-RA 0.1：面向机器人自主性的基础模型

> 原标题：JoyAI-RA 0.1: A Foundation Model for Robotic Autonomy
> 作者：Joy Future Academy（京东 JD），Tianle Zhang、Zhihao Yuan、Dafeng Chi、Peidong Liu 等；通讯作者 Yuzheng Zhuang、Liang Lin
> 发表：arXiv:2604.20100v2，2026 年 4 月
> 链接：https://joyai-ra.github.io/

---

## 一、研究背景与动机

- **开放世界机器人自主性的两大瓶颈**：(1) 任务覆盖所需的数据多样性不足——高质量真机数据采集昂贵，长尾交互、罕见失败模式和场景布局在现有语料中代表性差；(2) 多源行为知识共享时不可避免的"构型鸿沟"（Embodiment Gap）——不同机器人本体的运动学差异让跨构型知识迁移困难
- **现有 VLA 模型的不足**：RT-1、Open X-Embodiment、π0、π0.5、GR00T N1 等已开始探索更广义的预训练和跨构型扩展，但在异构数据源之间做有效的行为知识迁移仍是核心障碍。简单地把多源数据混在一起训练，会因为动作表示不一致和构型差异导致"平均化"或负迁移
- **本文目标**：构建一个 VLA 基础模型，**同时利用网络数据、第一人称人类操作视频、仿真轨迹、真实机器人数据**这四种互补数据源，通过统一动作空间消除构型鸿沟，并用多阶段训练 recipe 保证跨源迁移的稳定性

## 二、核心贡献

1. **多源多层级预训练框架**：在统一训练流程下融合四类互补数据（多模态网络数据、第一人称人类视频、仿真轨迹、真机数据），并按 embodiment 邻近度和数据规模做结构化组织
2. **统一动作空间（Unified Action Space）**：基于相机坐标系的末端执行器表示 + 固定维度向量 + 动作掩码，让单臂夹爪到双臂灵巧手在同一表示下训练，消除构型鸿沟
3. **EgoLive 数据集**：自建的大规模第一人称人类操作数据集，60 FPS RGB 视频、1969 个物体类别、1796 个动作类别，覆盖 1 万+任务，远超 EgoDex 等公开数据集
4. **三阶段训练 Recipe**：VLM Co-Pretraining → VLA Co-Pretraining → Post-Training，在 RoboTwin 2.0、RoboCasa GR1 Tabletop 和真机 AgiBot 基准上全面优于 π0、π0.5、Motus、GR00T-N1.6 等 SOTA

## 三、方法原理

### 3.1 整体框架

JoyAI-RA 由一个 4B 参数的 **预训练 VLM** 和一个 600M 参数的 **感知-动作专家（Perception-Action Expert）** 组成，后者基于 Perceiver 架构构建。

- **输入**：多视角观察 $O_t$、本体感知状态 $s_t$、语言指令 $L$
- **输出**：高层语义文本 $\ell$（如子任务描述）+ 低层连续动作序列 $a_{t:t+H}$
- **训练数据**：四源混合（真机 31% + 仿真 24% + 人类第一视角 33% + 多模态网络 12%）

模型采用先生成高层语义、再以该语义为条件生成动作 chunk 的两段式预测方式。

### 3.2 关键技术细节

#### 统一动作空间（Unified Action Space）

这是消除构型鸿沟的核心抓手，包含两部分设计：

**(1) 相机坐标系末端执行器表示**

跨真机、人类视频重定向、仿真三类带动作标签的数据，统一在相机坐标系下表达末端位姿。6 自由度末端位姿分解为：

$$\text{Pose} = (\underbrace{t \in \mathbb{R}^3}_{\text{平移向量}}, \underbrace{r \in \mathbb{R}^3}_{\text{轴角旋转向量}})$$

直觉解释：相机坐标系下表达动作有两个关键好处。第一，**语义一致性**——同一个动作向量编码的位移和旋转与机器人底座位置、关节构型无关；第二，**视觉对齐**——动作和图像观察共享同一视角，便于视觉条件下的动作预测。

**(2) 固定维度的统一动作向量**

定义一个固定长度的动作向量，覆盖所有数据源中出现的执行器组：左/右臂、左/右灵巧手、左/右夹爪等。对每种构型，没有对应 DoF 的维度在损失中被掩码、不参与梯度回传：

$$\mathcal{L}_{\text{action}} = \sum_i M_i \cdot \|a_i - \hat{a}_i\|^2, \quad M_i \in \{0, 1\}$$

直觉解释：让单臂夹爪、双臂灵巧手等差异巨大的构型可以共用同一套训练张量，差异通过 mask 屏蔽掉，等效于"对齐共有部分、忽略独有部分"。

#### 模型架构：VLM + Perceiver Action Expert

**VLM 主干**：4B 参数预训练 VLM，处理多视角图像 + 语言，产出空间接地的多模态表征 $z_t$，同时支持文本输出（VQA caption、子任务预测、FAST 离散动作 token）。

**Perception-Action Expert**：600M 参数，基于 **Perceiver** 架构，通过 latent bottleneck 高效融合多模态信息。动作生成采用 **Flow Matching** 框架预测条件速度场。

输入潜变量序列拼接三部分：

$$a_t^{0:t+H} = \text{Concat}\bigl(\phi_s(s_t),\ f_{\text{future}},\ \phi_a(\tilde{a}_{t:t+H}, \tau)\bigr)$$

其中 $\phi_s$ 是状态编码器、$f_{\text{future}}$ 是可学习的未来动作占位 token、$\phi_a$ 是带时间步 $\tau$ 的噪声动作编码器，$\tilde{a}_{t:t+H}$ 是高斯噪声与真值动作的插值。

模型预测速度场：

$$v_{t:t+H}^{\text{out}} = f_\theta(z_t, a_t^{0:t+H}, \tau)$$

Expert 内部由若干 **时间感知 Perceiver 注意力块** 堆叠组成。每层先用时间步自适应归一化调制视觉-语言流：

$$\tilde{z}_t = \text{AdaLN}_z(z_t, \tau)$$

再做残差注意力 + 残差 MLP：

$$h'_{t:t+H} = h_{t:t+H} + \text{MHA}(Q=h_{t:t+H},\ K, V=[h_{t:t+H};\tilde{z}_t])$$
$$h^{\text{out}}_{t:t+H} = h'_{t:t+H} + \text{MLP}(h'_{t:t+H})$$

直觉解释：显式的时间条件让模型在不同去噪阶段调整速度预测，提升动作时序一致性和生成稳定性。Perceiver 的 latent bottleneck 设计也比纯 cross-attention 更节省算力。

#### 多源数据的角色定位

| 数据源 | 占比 | 角色 | 关键特点 |
|---|---|---|---|
| 多模态网络数据 | 12% | 语义层 | Cambrian-10M、RefSpatial、Galaxea、Cosmos-Reason1-SFT 等，提供感知和语言先验 |
| 第一人称人类视频（EgoLive） | 33% | 行为先验层 | 自建 60 FPS、1969 物体类别、1796 动作类别、>10k 任务；含逐帧子任务标注 |
| 仿真数据 | 24% | 跨构型动作监督层 | InternData-A1、GenieSim3.0、InternData-M1 等，提供可扩展的动作标签 |
| 真机数据 | 31% | 物理接地层 | Open-X-Embodiment、AgiBot-World、Galaxea Open-World 公开数据 + 自采 JDAgibot |

EgoLive 的关键设计：使用自研手部姿态估计 pipeline 恢复人手轨迹，并 **重定向（Retarget）** 到 ALOHA、Fourier、Agibot G1 等多种机器人构型，使人类数据可以直接进入统一动作空间。

### 3.3 训练与优化

#### 三阶段训练 Recipe

**Stage 1 — VLM Co-Pretraining**：训练 VLM 主干，目标是赋予视觉理解、空间推理、长时规划、离散动作生成能力。

四类数据：
- General VQA（保留原有视觉理解能力）
- Embodied VQA（图像/视频输入，输出 points/bboxes/trajectories，强化空间推理 + 任务分解）
- 跨构型动作数据（通过 FAST [29] 离散 token 学习初步的动作生成）
- 人类视频数据（拓宽视觉输入与动作分布）

损失：自回归 token 预测的 NLL：

$$\mathcal{L}_{\text{VLM Co-Pretraining}}(\theta) = \mathbb{E}_{(x,y)\sim \mathcal{D}}\left[-\sum_{j=1}^{n-1} M_j \log p_\theta(y_{j+1} \mid x_{1:j})\right]$$

其中 $y$ 同时覆盖 VLM 文本响应和 FAST 离散动作 token。

**Stage 2 — VLA Co-Pretraining**：在保留 VQA 能力的前提下，引入动作专家进行连续动作监督。仿真+真机+重定向人类视频三类动作数据全部以统一动作空间表示。

损失 = 自回归损失 + Flow Matching 损失：

$$\mathcal{L}_{\text{VLA Co-Pretraining}}(\theta) = \alpha \cdot \mathbb{E}_{(x,y)\sim \mathcal{D}}\left[-\sum_j M_j \log p_\theta(y_{j+1}\mid x_{1:j})\right] + \mathbb{E}_{\mathcal{D},\tau,\omega}\left[\|\omega - a_{1:H} - f_\theta^a(a_{1:H}^{\tau,\omega})\|^2\right]$$

其中 $a_{1:H}^{\tau,\omega} = \tau a_{1:H} + (1-\tau)\omega$，$\omega \sim \mathcal{N}(0, I)$，$\alpha$ 是平衡两项损失的乘数。

**Stage 3 — Post-Training on Target Robots**：在目标机器人数据上轻量微调，**丢弃自回归损失**，仅用 Flow Matching 损失更新全部参数：

$$\mathcal{L}_{\text{Post-Training}}(\theta) = \mathbb{E}_{\mathcal{D}_{\text{target}},\tau,\omega}\left[\|\omega - a_{1:H} - f_\theta^a(a_{1:H}^{\tau,\omega})\|^2\right]$$

直觉解释：先用 VLM 阶段筑牢"语义-感知-离散动作"的根基，再用 VLA 阶段对齐连续动作分布，最后用 Post-Training 适配目标构型——三阶段从"广"到"专"，避免在大规模异构数据上直接做连续动作监督带来的不稳定。

## 四、实验与结果

### 4.1 实验设置

- **仿真基准**：
  - **RoboTwin 2.0**：50 个任务，2,500 demo 干净场景 + 25,000 demo 重随机化场景；Easy（固定初始）+ Hard（随机化）两档
  - **RoboCasa GR1 Tabletop**：24 个 6DoF 灵巧手家庭场景任务，每任务 50 次 rollout 取平均
- **真机基准**：自建 Real-World AgiBot Benchmark，AgiBot G1 平台，5 场景 6 任务，每任务 20 次试验
- **对比基线**：π0、π0.5、Motus、LingBot-VLA、GR00T-N1.6、Qwen3PI、TwinBrainVLA、DualCoT-VLA、ABot-M0、Being-H0.7 等

### 4.2 主要结果

**RoboTwin 2.0**（50 任务平均成功率）：

| 方法 | Easy | Hard |
|---|---|---|
| π0 | 65.92 | 58.40 |
| π0.5 | 82.74 | 76.76 |
| Motus | 88.66 | 87.02 |
| LingBot-VLA | 88.56 | 86.68 |
| **JoyAI-RA** | **90.48** | **89.28** |

在 Adjust Bottle、Grab Roller、Place Empty Cup 等多个任务上达到 100%。

**RoboCasa GR1 Tabletop**（24 任务平均成功率）：

| 方法 | GR00T-N1.6 | Qwen3PI | TwinBrainVLA | DualCoT-VLA | ABot-M0 | Being-H0.7 | **JoyAI-RA** |
|---|---|---|---|---|---|---|---|
| Success Rate(%) | 47.6 | 43.9 | 54.6 | 55.1 | 58.3 | 49.2 | **63.2** |

在长程任务上提升尤为显著：CanToDrawerClose **+16.0**、MilkToMicrowaveClose **+24.0**、TrayToPot **+18.0**。

**真机 AgiBot Benchmark**：JoyAI-RA 跨任务平均成功率从 π0.5 的 0.62 提升到 **0.74**，在 Headphones、Remedy 等需要精确目标识别和放置的任务上领先最显著。

### 4.3 消融实验

**(1) EgoLive 数据消融（RoboTwin 2.0 Easy）**：

| 配置 | No Pretraining | JDAgibot Only | + EgoLive 10% | + EgoLive Full |
|---|---|---|---|---|
| Success Rate (%) | 81.64 | 77.62 | 81.40 | **87.42** |

关键发现：只用真机数据预训练反而比无预训练略差（77.62 vs 81.64），暗示构型鸿沟带来的负迁移；加入足量人类视频数据后性能跃升 5.78 个百分点。**人类数据需达到足够规模才显现增益**（10% 子集尚不足）。

**(2) 数据源消融（RoboTwin 2.0 Easy）**：

| 配置 | JDAgiBot | EgoDex | EgoLive | Success (%) |
|---|---|---|---|---|
| Baseline | – | – | – | 81.28 |
| + JDAgiBot | ✓ | – | – | 79.20 |
| + Human (Full) | ✓ | ✓ | ✓ | **89.30** |
| EgoDex only | – | ✓ | – | 86.88 |
| EgoLive only | – | – | ✓ | 87.16 |

EgoLive 单独使用就比 EgoDex 稍好；两者叠加可进一步提升（互补关系）。t-SNE 显示 EgoLive 在语义空间分布更广更连续，对应其更丰富的对象/动作长尾覆盖。

**(3) 训练 Recipe 消融**：

| 配置 | Success (%) |
|---|---|
| Baseline（仅 Post-Training） | 81.28 |
| 仅 VLM Co-Pretraining + 人类数据 | 87.84 |
| 仅 VLA Co-Pretraining + 人类数据 | 87.42 |
| **VLM + VLA Co-Pretraining + 人类数据** | **90.48** |
| 同上但 Stage2 去掉仿真数据 | 89.10 |
| 同上且 Stage2 包含仿真数据 | 90.24 |

两个阶段独立都能带来 6+ 点增益，叠加更佳（+9.2 点）；仿真数据贡献约 1.14 点（90.24 → 89.10）。

## 五、局限性与展望

- **真机长程精细任务仍有瓶颈**：在 Cup、Croissant 等需要长程视觉推理和多步精细操控的任务上，π0.5 仍占优；Food Scraps 对所有方法都很难
- **In-Domain EgoLive 数据并非对所有任务都正向**：当人类视频的环境/任务结构与评测场景偏差大时，反而引入冲突监督信号（如 Mouse、Food Scraps 任务）
- **EgoLive 收益尚未饱和**：从 10% 到 100% 仍有约 6% 提升，暗示继续扩大人类视频数据规模是值得的方向
- **未来方向**：进一步扩展人类视频规模、改进低层执行精度、设计对协同操作更友好的架构

## 六、灵魂三问

1. **它解决了什么问题？** 以前的 VLA 模型在多源异构数据上训练时会因为动作表示不一致和构型差异产生负迁移；JoyAI-RA 通过"统一动作空间 + 多阶段训练"让真机、仿真、人类视频、网络数据能在同一模型下互相增益，并把第一视角人类视频规模化做出来（EgoLive 1 万+任务）。

2. **为什么这么做？** 核心设计选择是 **相机坐标系下的固定维度统一动作向量 + 动作掩码**。不这么做（即各构型独立的关节空间）就无法做有效跨构型迁移，人类视频也无法转化为动作监督。论文消融里 "仅 JDAgibot 真机数据反而比 No Pretraining 还差" 直接说明：没有统一表示和构型对齐时，简单堆数据是负迁移的。

3. **什么证据最有说服力？** EgoLive 消融的 scaling 趋势最有说服力：无人类预训练时性能 77.62%，加 10% EgoLive 几乎无帮助（81.40），加全量 EgoLive 跳到 **87.42%**（+9.8 点）。这同时说明了两件事：人类视频经过统一动作空间处理后确实能传递可用的操控先验；且收益依赖数据规模，不是噪声扰动。配合真机基准上 0.62 → 0.74 的端到端提升，证据链完整。

## 七、个人总结

1. JoyAI-RA 的核心 idea 是"**用统一动作空间把人类视频升级为一类正经的动作监督**"——通过相机坐标系 + 固定维度 + mask 三件套，把人手轨迹重定向到机器人空间，让 33% 的人类视频数据真正参与到动作学习而不是仅作为视觉先验。这本质上是把"如何用人类视频"这个老问题做了一次工程化的封装。
2. 最大优势是 **大规模人类视频 + 系统化训练 recipe 的组合**——RoboTwin 2.0 Hard 89.28% 和 RoboCasa GR1 63.2% 都明显优于 π0.5、Motus 等同期 SOTA，且消融充分。不足在于：(a) 论文未深入讨论 in-domain 人类数据为何会产生负迁移的细节，仅作现象描述；(b) 三阶段训练 recipe 较为复杂，复现成本高；(c) 真机基准任务量偏少（6 个任务×20 试），统计稳定性有限。
3. 对后续研究的启发：(a) "构型鸿沟"应作为 VLA 设计的一等问题——把动作表示统一好比堆数据更重要；(b) 人类视频不再是辅助先验，可以是与真机数据等量齐观的核心训练源；(c) 三阶段训练 recipe（VLM 离散动作 → VLA 连续动作 → 目标域专精）是一个可推广的范式，值得在小模型上复现验证。

## 八、与 π 系列的对比要点

| 维度 | π0 / π0.5 / π0.6 / π0.7 | JoyAI-RA 0.1 |
|---|---|---|
| **动作空间** | 关节级 / 末端执行器（按任务切换 control mode） | **相机坐标系** 末端执行器 + 固定维度统一向量 |
| **跨构型策略** | 训练时混合多构型数据，依赖 VLM 学习 | 显式动作空间统一 + 掩码 |
| **人类视频角色** | 主要作为视觉/语义先验 | **与真机等量级的动作监督源**（占 33%） |
| **训练 recipe** | 单阶段 flow matching + 多模态 prompt（π0.7） | 三阶段：VLM Co-Pretrain → VLA Co-Pretrain → Post-Training |
| **动作生成** | Flow Matching + Action Expert | Flow Matching + **Perceiver 架构** Action Expert |
| **核心创新点** | 多模态 prompt 与 episode metadata（π0.7） | 统一动作空间 + 多源 + 多阶段 |

两者代表了 VLA 基础模型的两条不同路线：π 系列偏向"丰富 prompt + 大模型"路线，JoyAI-RA 偏向"统一表示 + 多源数据扩展"路线，互有借鉴价值。

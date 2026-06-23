# ViTRA：用真实生活人类视频做灵巧手 VLA 预训练——全自动管线 + 规模化实验

> 原标题：Scalable Vision-Language-Action Model Pretraining for Robotic Manipulation with Real-Life Human Activity Videos  
> 作者：Qixiu Li, Yu Deng, Yaobo Liang, Lin Luo, Lei Zhou, Chengtang Yao, Lingqi Zeng, Zhiyuan Feng, Huizhi Liang, Sicheng Xu, Yizhong Zhang, Xi Chen, Hao Chen, Lily Sun, Dong Chen, Jiaolong Yang（通讯）, Baining Guo  
> 机构：Tsinghua University; Microsoft Research Asia  
> 发表：arXiv:2510.21571v1, 2025-10-24  
> 链接：https://microsoft.github.io/VITRA/  
> 开源：承诺开源训练数据集与预训练 VLA 模型

---

## 一句话说清楚

ViTRA 的核心判断是：把人类活动视频变成机器人 VLA 预训练数据不需要人工标注——通过 3D 手部/相机重建 + 手腕速度最小值做动作切分 + GPT 标注动作描述，可以全自动地从非结构化第一视角视频中抽出 100 万条原子级 V-L-A episode，预训练后的灵巧手 VLA 模型在完全未见过的环境中有很强的 zero-shot 动作预测能力。

## 一、研究背景与动机

VLA 模型面临 LLM/VLM 预训练曾经经历过的同一道坎：数据不够。但机器人 VLA 的数据问题比语言和视觉都更重——因为动作标注只能由人操作机器人来采集，成本极高，导致现有的 V-L-A 数据（OXE、DROID、AgiBot World 等）在规模、技能多样性、物体种类、场景变化上都远落后于互联网级别的语言/视觉语料。灵巧手（多指手）的 VLA 数据更是几乎空白——到本文撰写时还没有可用于预训练的大规模灵巧手动作数据集。

与此同时，互联网上有海量真实人类活动视频——无需脚本、无需标注、覆盖大量技能、物体和环境。但这些视频是「脏数据」：无切分、时长任意、包含大量无关动作、没有语言指令也没有 3D 动作标签。此前用人类视频做机器人学习的工作（学习视觉表征、affordance、point trajectory、latent action 等）都绕过了直接提取显式 3D 动作标签这一步，因此无法做到和真实机器人 V-L-A 训练数据格式的对齐。

ViTRA 是第一个回答「能不能把非结构化人类视频转成对齐机器人 V-L-A 格式的训练数据」这个问题的。答案是肯定的。

> 与 OpenEgo 的对比：OpenEgo 做的是把已有标注的数据集统一格式，ViTRA 做的是从裸视频自动生成标注。数据上有交叉（都用 Ego4D、Epic-Kitchen），但 ViTRA 完全不依赖已有的人工标注。

## 二、核心贡献

1. **提出全自动人类活动解析框架。** 三阶段管线：3D 运动标注 → 原子级动作分割 → 指令标注。将任意长度的人类手部活动视频转成多条灵巧操作 V-L-A 轨迹，全程无需人工干预。

2. **构造百万级手部 VLA 预训练数据集。** 从 Ego4D、Epic-Kitchen、EgoExo4D、SSV2 的裸视频中提取 1M episodes / 26M frames，覆盖丰富的手部动作、物体、属性和环境多样性，远超现有机器人数据集。

3. **设计灵巧手 VLA 模型架构与训练策略。** 采用 VLM backbone（PaliGemma-2 3B）+ Diffusion Transformer (DiT) action expert 的解耦结构，引入 causal attention denoising、unified single/dual-hand prediction、trajectory-aware augmentation 等设计解决人类视频预训练的特有问题。

4. **系统验证预训练的有效性。** 在零样本 human hand action prediction 中，ViTRA 显著优于 EgoDex lab data、Being-H0 等 baseline；在真实灵巧手机器人实验中（Realman + XHand），微调 1.2K 条真实轨迹后在未见物体和类别上大幅超越 π0、VPP、latent action pretrain 等方法。展示清晰的 data scaling 行为。

## 三、方法原理

### 3.1 整体框架

ViTRA 的核心 pipeline 分为两个层面：

**数据构造层面（三阶段，Fig. 2）：**

1. **3D 运动标注（Motion Labeling）：** 从裸视频中恢复 metric-scale 3D 手部运动和相机运动。
2. **原子级动作分割（Atomic Action Segmentation）：** 基于手腕在 3D 空间中的运动速度最小值来切分视频。
3. **指令标注（Instruction Labeling）：** 对每个分段用 GPT-4.1 根据采样帧+手部轨迹叠加图生成动作描述。

**模型训练层面：**

预训练阶段，用构造的手部 VLA 数据训练一个 VLA 模型做 human hand action prediction；微调阶段，将手部动作空间映射到灵巧机器人手（XHand），用真实机器人轨迹 fine-tune。

### 3.2 关键技术细节

#### 3.2.1 三阶段数据构造管线

**阶段一：3D 运动标注。** 输入是一段无校准的单目自监督视频。首先判断相机是否运动（基于背景光流），然后用 DroidCalib（运动相机）或 MoGe-2 + DeepCalib（静态相机）估计内参并对畸变进行校正。处理后的视频进入两条并行流程：一边用 HaWoR 做逐帧 camera-space 3D 手部重建（MANO 参数模型，输入 6D wrist pose + 关节角），一边用改进版 MegaSAM（将深度先验从原模型替换为 MoGe-2）做视觉 SLAM 估计相机位姿。将 camera-space 手部与 metric-scale 相机位姿结合，得到 world-space 3D 手部序列，再用 spline smoothing 去噪。world-space 序列可以任意转换回任何一帧的相机空间，模拟了大多数机器人数据的静态相机设置。

为提高效率，长视频先切成 20 秒重叠片段分别处理再合并。

**阶段二：原子级动作分割。** 这是整条管线中最巧妙的一步。现有 temporal action segmentation 方法依赖预定义动作类别，或借助 VLM 但定位精度不够。ViTRA 的灵感来自一个自然观察：人类在动作切换时，手腕运动速度通常出现局部最小值——手在结束一个动作轮廓、准备开始下一个动作时会短暂减速甚至停顿。

具体算法：对 world-space 3D 手腕轨迹做平滑处理，以每个点为中心在固定窗口内检测局部速度最小值作为切分点。左右手独立切分，互不干扰。这个方法不需要额外模型推理，不需要预标注文本——非常高效。对往复动作（如擦拭）可能过切分，但后续可通过指令标注后合并处理。

**阶段三：指令标注。** 对每个分段均匀采样 8 帧，在每帧上叠加从当前帧到片段结束的手掌 3D 轨迹投影（在图像上画出手部运动路径）。将这些图像输入 GPT-4.1，要求以祈使句形式描述指定手的动作，并标注无意义片段的动作为 "N/A"。提供原子级片段（而非固定时长截断）对 GPT 的标注准确率有显著提升——因为片段内的动作语义完整，GPT 不需要在片段内自行判断动作边界。手部轨迹叠加图也是关键：只给图像不给运动线索时，GPT 容易误判。

#### 3.2.2 VLA 模型设计

**整体架构（Fig. 3）：** VLM backbone + Diffusion Transformer action expert 的解耦设计。
- VLM：PaliGemma-2 3B（SigLIP 视觉编码器 + Gemma-2 LLM），224² 输入分辨率。
- Action expert：DiT-Base。输入是 cognition feature（VLM 输出的条件信号）、当前手部状态、带噪声的动作 chunk 和 action mask 的拼接。Cognition feature 通过 AdaLN 注入 DiT 各层。
- 额外输入 camera FoV token，帮助模型理解图像的原始宽高比和相机内参。
- 视觉编码器冻结，其余部分端到端训练。

**手部动作空间（Eq. 3）：** 每步动作 102 维：
```
[Δtl, Δrl, θh_l, Δtr, Δrr, θh_r] ∈ R¹⁰²
```
- Δt ∈ R³, Δr ∈ R³：相邻帧间手腕相对平移和旋转(欧拉角)
- θh ∈ R¹⁵ˣ³：MANO 15 个关节的欧拉角（每手 45 维）
- 左右手各 51 维，共 102 维

**统一单/双手动作预测：** 这是应对预训练数据的一个特有设计。预训练数据以单手原子动作为单位，但某些 episode 存在双手重合动作。VLM 始终接收格式化的语言指令：`Left hand: <left-hand action>. Right hand: <right-hand action>`，无动作时为 `None`。Action expert 始终接收双手 noisy action，通过 action mask（0/1）标记哪些维度有监督，不可见维度置零并排除出 loss。这使得同一架构能同时处理纯单手和双手同时操作的 episode。

**Causal Attention Denoising：** 人类手部动作很快，很多 clip 仅约 1 秒（~30 帧）。当 action chunk 长度 N=16 时，很多 chunk 会超出 episode 终点。传统做法是末端补零+双向注意力，但补零会被理解为「停止运动」，对于往复动作（如擦拭）会引入错误的终止信号。ViTRA 用 causal attention 做去噪——每个时间步只关注历史，零填充位置不会反向影响有效位置。零填充位置的 loss 被 mask 掉。

#### 3.2.3 训练策略

**预训练阶段：**
- 先 warm up action expert + cognition token 映射层 + FoV MLP（5K 步），再联合微调 VLM + action expert（80K 步）
- 学习率：action expert 1e-4, VLM 1e-5, batch size 512
- 8×H100 GPU, 2 天

**Trajectory-aware augmentation：** 随机裁剪 + 透视变换 + 改变 FoV/aspect ratio/crop center，保持主点在图像中心。动作序列随相机参数变换同步调整。约束裁剪后手部投影轨迹不超出图像。随机水平翻转（同步调整指令中的左右手），当指令不含颜色词时应用 color jittering。

**微调阶段：**
- 20K 步, batch size 256, lr 1e-5, 8×H100 GPU 约 8 小时
- 将人类手部动作空间视为机器人手的超集，通过 joint topology mapping 把 XHand 关节映射到 MANO 对应的最近关节位置
- 用直接执行的关节命令（而非从记录状态导出的动作标签）作为监督，产生更合理的手-物交互动作

### 3.3 与 latent action pretrain 的区别

论文专门设了一个 baseline：将 ViTRA 的显式 3D action 标签替换为 LAPA 提取的 latent actions。结果表明 latent action 在 seen tasks 上还行，但 unseen 场景下完全失败（0% 成功率）。这是因为 latent action 难以将任务相关运动与背景信息解耦，在预训练-微调 domain gap 较大时发生灾难性遗忘。显式 3D action 在这点上有关键优势——它直接缩小了预训练和微调之间的格式差异。

## 四、实验与结果

### 4.1 预训练数据多样性分析

| 对比维度 | ViTRA (Ours) | EgoDex (lab) | OXE (robot) | DROID | AgiBot World |
|---|---:|---:|---:|---:|---:|
| Episodes | 1M | 338K | ~400K | - | - |
| Frames | 26M | 90M+ | - | - | - |
| 视觉多样性 (OpenImages sim) | **最高** | 低 | 中低 | - | - |
| 指令多样性 (noun/verb/adj) | **最高** | 低 | 中 | - | - |
| 数据来源 | 4 个 ego 数据集裸视频 | 实验室采集 | 20+ 机器人 | 1 机器人 | 1 平台 |

视觉多样性衡量方式：以 OpenImages 作为真实世界视觉多样性参考集，用 DINOv2 特征计算每个数据集与 OpenImages 的最大余弦相似度。ViTRA 不仅绝对相似度最高，而且随着采样 episode 数量增加，相似度上升斜率更陡——说明它对真实世界场景的覆盖更均匀，而非像 OXE 那样呈现碎片化分布。

指令多样性：用 GPT-4.1 从每条指令中提取名词/动词/形容词，统计 distinct words 的 h-index 和 i100-index。ViTRA 在所有指标上显著超过其他数据集。

### 4.2 零样本手部动作预测

在两组评估上测试预训练模型的 zero-shot 能力：

**Grasping benchmark：** 47 个未见环境、396 个物体、RGB-D 图像 + 渲染合成手。指标：预测手指轨迹与目标物体最近点距离 d_hand-obj（越小越好）。

**General action benchmark：** 117 个未见真实环境、手机拍摄。23 位参与者对 30 个随机场景做 top-3 动作排序打分（3/2/1 赋分）。

| 方法 | Grasp avg./med. d_hand-obj (cm) ↓ | General Action User Score ↑ |
|---|---:|---:|
| Initial position | 20.0 / 20.0 | – |
| Being-H0 (8B) | 19.1 / 18.4 | 0.15 |
| EgoDex (lab data) | 17.6 / 18.3 | – |
| Human annotation | 14.1 / 14.1 | 0.96 |
| No augmentation | 11.6 / 10.7 | 1.43 |
| Bidirectional attention | 9.3 / 7.2 | 1.69 |
| **Ours (ViTRA)** | **8.8 / 6.2** | **1.91** |

关键发现：
- 即使用 EgoDex 全量数据（338K lab-captured episodes），效果远不如 ViTRA——数据多样性比数据量更关键。
- 用原始人类标注训练反而劣于自动标注（14.1 vs 8.8）：原始标注的时间粒度或动作粒度与机器人数据不对齐，削弱了指令跟随能力。
- Trajectory-aware augmentation 是提升泛化性的关键设计（去掉后从 8.8 退化到 11.6）。
- Causal attention 比 bidirectional attention 更好（8.8 vs 9.3），因为它正确处理了跨 episode 边界的动作语义。

**Episode 构造策略消融（在 350K 子集上）：**

| 方法 | Avg./med. d_hand-obj (cm) ↓ |
|---|---:|
| Fixed-interval segmentation (1s) | 10.5 / 8.8 |
| No trajectory overlay | 11.7 / 10.7 |
| Ours† | **9.9 / 8.1** |

两点结论：（1）固定时长切分效果更差——GPT 难以在含多个原子动作的 clip 中正确识别每个动作的指令；（2）不给 GPT 看手部轨迹投影导致性能下降明显——运动信息对正确标注很重要。

**Data scaling 行为（Fig. 7）：** 在 grasping 任务上，随着预训练数据从 1% → 10% → 20% → 50% → 100% 扩展，d_hand-obj 近似对数线性下降。虽然 EgoDex 的 episode 数大于 ViTRA 的 20% 甚至 10% 子集，但其性能仍显著落后——核心还是多样性不足。

### 4.3 真实灵巧手机器人实验

**硬件：** Realman RM75 机械臂 + 12-DoF XHand 灵巧手 + RealSense 头部相机。数据采集用双臂遥操作 + MANUS 遥操作手套。

**任务设计（4 类，1.2K 条遥操作轨迹）：**
1. **General pick & place**：将物体放入盒子，3-4 个随机干扰物
2. **Functional grasping**：抓取物体的功能性部位（如手柄）
3. **Pouring**：拿起瓶子、将内容物倒入另一个容器、放回
4. **Sweeping**：从篮子中拿起扫帚、扫垃圾、放回扫帚

**评估设置：**
- Seen Object & BG：微调时见过的物体和背景，随机化位置和干扰物
- Unseen Object & BG：新的物体和背景（分同类型新物体和全新类别）

#### 4.3.1 Seen 场景结果

| 方法 | Pick & Place (40) | Func. Grasp (24) | Pour (8) | Sweep (8) | 平均 |
|---|---:|---:|---:|---:|---:|
| VPP | 57.5 | 29.2 | 12.5 | 0.0 | 24.8 |
| π0 | 37.5 | 25.0 | 75.0 | 50.0 | 46.9 |
| No VLA pretrain | 32.5 | 33.3 | 12.5 | 50.0 | 32.1 |
| Latent action pretrain | 42.5 | 41.7 | 37.5 | 62.5 | 46.0 |
| OXE pretrain | 40.0 | 37.5 | 62.5 | 25.0 | 41.3 |
| **Ours (ViTRA)** | **80.0** | **66.7** | **75.0** | **62.5** | **71.0** |

#### 4.3.2 Unseen 场景结果

| 方法 | Unseen Obj&BG | Unseen Category | 平均 unseen |
|---|---:|---:|---:|
| VPP | 5.2 | 8.3 | 6.8 |
| π0 | 16.1 | 33.3 | 24.7 |
| No VLA pretrain | 10.9 | 12.5 | 11.7 |
| Latent action pretrain | 0.0 | 0.0 | 0.0 |
| OXE pretrain | 7.8 | 12.5 | 10.2 |
| **Ours (ViTRA)** | **64.6** | **70.8** | **67.7** |

几个强力信号：
- Unseen 场景下 ViTRA vs π0 是 64.6% vs 16.1%，**差距远超 seen 场景**（71.0% vs 46.9%）。这说明预训练的价值主要体现于泛化，而非 domain 内的模式匹配。
- Latent action pretrain 在 unseen 上 **完全失败（0%）**——这是整篇论文中最有冲击力的一个数据点。显式 3D action 的意义不在于训练时更好看，而在于从根本上降低了预训练到微调的 domain gap。
- OXE pretrain 在灵巧手上效果不佳——夹爪的数据多样性和机器人动作空间差异导致负迁移。

#### 4.3.3 Data Scaling 与 Robot Performance 的关联（Fig. 10）

在 pick & place 任务上，预训练数据从 10% → 20% → 50% → 100%，seen 和 unseen 成功率均呈现上升趋势。更有意思的是 Fig. 10c：预训练阶段 hand-object distance（human hand prediction 精度）与微调后 robot task success rate 呈**正相关**——意味着零样本手部动作预测 benchmark 可以作为下游机器人性能的有效代理指标，加速 VLA 模型的原型迭代。

## 五、局限性与展望

**作者提出的局限：**
1. 当前数据来自已有的自监督视频数据集，VLA 数据质量受限于 3D 重建算法和 VLM 的能力——存在一定的噪声和不准确。
2. 数据构造和训练目前主要针对短时域/原子级技能，尚未纳入长期任务规划和推理能力。
3. 机器人实验以单手为主，虽然做了简单的双手 handover 演示，但尚未系统验证双手协作场景。
4. 未来计划接入更多视频源（如 Howto100M）、引入更先进的重建技术增强数据质量、构建高层任务结构以支持长程规划。

**我补充的观察：**
- **3D 重建作为最大瓶颈。** 整个管线的质量天花板是第一阶段 3D 运动标注的精度。HaWoR 对于遮挡、快速运动、与物体交互中的手部仍不完美，而 MegaSAM 的相机位姿估计在纹理弱或动态场景下也可能漂移。第二阶段的速度最小值切分可能对静止操作（holding、steady contact）过度放大误差——因为此时速度本来就很低，局部最小值检测的信噪比差。
- **GPT 标注的噪声来源。** 8 帧采样对于快速动作是稀疏的，手部轨迹投影虽然辅助但不可能完全弥补信息损失——GPT 有时会根据视觉上下文推断而非观测（hallucination），这在未见类别上尚可容忍，但放在训练数据中可能产生错误监督。
- **实验覆盖面的差距。** 真实机器人实验虽然好看，但只有 4 个任务和 1.2K 条轨迹——fine-tuning 数据量依然很小。Unseen category 的高成功率（70.8%）令人印象深刻但也需要警惕：unseen categories 的定义只说了「category not encountered」，但没有明确列举，可能存在视觉相似但类别不同的物体。
- **真正的 scaling 在于 pipeline 的跑通门槛。** ViTRA 展示了一条从裸视频到 VLA 预训练数据再到部署的全自动通路。这个通路一旦搭好，加数据就只需要算力——你可以把 Youtube、TikTok、居家录像都扔进去。这一点比单个模型的分数字更有份量。

## 六、灵魂三问

1. **它解决了什么问题？**
   解决了「没有足够大、足够多样的灵巧手 VLA 预训练数据」这个问题，而且是用一种前所未有的方式——不依赖任何人工标注，全自动地从真实生活裸视频中提取 V-L-A 数据。此前用人类视频做机器人学习的工作要么停在视觉/语言层，要么用 latent action 而无法跨越人-机动作空间鸿沟。ViTRA 第一次把「从裸视频到机器人格式的 V-L-A episode」的全自动管线做通了，并且用真实灵巧手实验验证了价值。

2. **为什么这么做？**
   核心设计有三层。第一层：3D 重建把人类手部运动从 2D 图像里抽出来放到 metric 空间——这是"对齐"的基础。第二层：速度最小值切分把非结构化的长视频切成原子级片段——这是"粒度对齐"。第三层：GPT+轨迹叠加做指令标注——这是"语义对齐"。这三个对齐合在一起，使裸人类视频在格式上与机器人 V-L-A 数据几乎没有区别。而对比 latent action 方案为什么不行——latent action 在预训练时学到的动作概念是数据驱动的、与视觉背景纠缠的，到了新场景中被污染、退化为 noise。

3. **什么证据最有说服力？**
    最有力的不是某个数字，而是**三个跨层次的一致性**：（1）预训练数据多样性分析（OpenImages similarity + 指令 h-index）说明 ViTRA 数据确实更丰富——这为后续性能提供了基础解释；（2）Unseen robot tasks 上 ViTRA 64.6% vs π0 16.1%、vs latent action 0%——差距远大于 seen tasks，说明预训练提供的是泛化能力而非 domain 内的 memorization；（3）预训练 hand prediction 精度与下游 robot success rate 的正相关（Fig. 10c）——建立了跨层次、跨 embodiment 的预测性联系。这三组证据从数据质量 → 模型行为 → 下游泛化，构成了一个闭合的归因链条。

## 七、个人总结

1. **核心 idea：** 将人类视频转成 VLA 预训练数据不需要人工标注——通过 3D 手部/相机重建 + 速度最小值动作切分 + GPT 轨迹增强标注，可以全自动地从裸视频中抽出 100 万条原子级 V-L-A episode，这些 episode 在格式上与真实机器人数据完全对齐。

2. **最大优势和风险：** 优势是 pipeline 的全自动可扩展性——每多加一个视频源、多一些 GPU 就能扩大数据规模，没有人工标注瓶颈。风险是 3D 重建精度构成的天花板——如果 HaWoR/MegaSAM 在手物交互、快速运动或弱纹理场景下系统性出错，整个 pipeline 会产生大量劣质训练样本。后续版本的管道质量本质上取决于 SOTA 3D 感知的提升速度。

3. **对后续研究的影响：** ViTRA 为「人类视频 → 机器人 VLA 预训练」提供了第一个完整的 recipe，同时也呼应了 H-RDT、EgoMimic 等同期工作的趋势。下一步的自然演化方向是：（a）引入更多视频源（YouTube、Howto100M 等）做更大规模预训练；（b）把 atomic skill 组合成层级化的长程任务结构；（c）用更先进的 3D 重建（如 DUSt3R、MASt3R、MonST3R）升级管线，降低重建噪声。这也对 OpenEgo 这类数据集工作提出了新要求——统一的数据格式应该能兼容自动生成的标注，而不只是人工标注的组合。

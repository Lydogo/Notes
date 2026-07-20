# Do as I Do：把日常人类视频转成灵巧手机器人数据

> 原标题：Do as I Do: Dexterous Manipulation Data from Everyday Human Videos  
> 作者：Bhawna Paliwal, Haritheja Etukuru, William Liang, Pieter Abbeel, Nur Muhammad “Mahi” Shafiullah, Jitendra Malik  
> 机构：UC Berkeley  
> 发表：arXiv:2606.19333v1, 2026-06-17  
> 链接：https://arxiv.org/abs/2606.19333  
> 项目页：https://do-as-i-do.com  
> 开源：论文/项目页未明确给出代码仓库链接

---

## 一、研究背景与动机

机器人学习最缺的是可扩展、可执行的数据，尤其是多指灵巧手。真实遥操作昂贵且依赖专家，仿真探索又需要环境和奖励设计。相比之下，人类视频极其丰富，但大多数只是单目 RGB 视频，缺少深度、手部关键点、物体 3D 模型和机器人动作。

Do as I Do 试图把“观察数据”转为“机器人可执行数据”：从日常人类手物交互视频中重建 4D hand-object interaction，再重定向到多指机器人手和机械臂上。论文强调其目标不是只处理干净实验室数据，而是处理 egocentric、exocentric、互联网视频甚至生成视频。

## 二、核心贡献

1. **提出从单目 RGB 人类视频到灵巧手轨迹的完整 pipeline**：先重建手和物体，再用动力学感知重定向生成可执行机器人动作。
2. **把 SAM 3D 改造成物体视频跟踪器**：通过 guided diffusion 固定物体形状、逐帧更新 6-DoF pose，使遮挡和运动模糊下的物体跟踪更稳定。
3. **增强 noisy reference 下的动态重定向**：在 sampling-based optimization 中加入 warmup steps、random force perturbation 和 transition reward，提高抓取和放置阶段的稳定性。
4. **展示真实机器人可播放轨迹**：从互联网、第一视角和生成视频中过滤出 500 条高质量轨迹，并在双 UR3e + Sharpa Wave hands 上展示 10 类真实 rollouts。

## 三、方法原理

### 3.1 整体框架

Do as I Do 包含两个阶段：

| 阶段 | 输入 | 输出 | 关键工具 |
| --- | --- | --- | --- |
| Reconstruction | 单目 RGB hand-object 视频 | 3D 手、物体 mesh、物体 6-DoF pose 序列 | HaWoR, SAM 3, MoGe, SAM 3D |
| Retargeting | 重建得到的 hand-object reference | 机器人手臂动作轨迹 | MuJoCo / sampling-based optimization |

第一阶段解决“视频里人和物怎么动”，第二阶段解决“机器人怎样用自己的身体做同样的事”。

### 3.2 关键技术细节

**物体跟踪 via Guided Diffusion** 是重建部分的核心。SAM 3D 原本是单图像到 3D 的生成模型，如果逐帧独立使用，会导致每帧 mesh 和 pose 不连续。作者的做法是固定 anchor frame 的物体形状，只在后续帧中采样 pose，并在 diffusion / flow matching 推理过程中把形状 block nudging 到固定形状、pose block nudging 到上一帧 pose。

pose guidance 不是固定常数，而是用 2D point tracks 估计物体旋转速度，自适应调整指导强度。每帧会采样多个 pose candidate，再用加权 SE(3) 距离聚类和 mask-IoU 选择共识 pose。这样避免每次都计算昂贵的条件 log-density。

**手物对齐** 处理独立重建带来的尺度不一致。作者把 HaWoR 的手尺度当作近似 metric ground truth，用 MoGe 深度和可见手部/物体质心估计尺度与平移，再把物体放到与手一致的空间中。

### 3.3 训练与优化

重定向不是简单做 kinematic retargeting，因为人手和机器人手形态不同，视频 reference 又没有接触力。Do as I Do 使用 MPPI-style sampling-based optimization，在物理仿真中跟踪 reference，同时追求动态可行性。

为处理 noisy reconstructed references，作者加入三项设计：

| 组件 | 解决的问题 | 直觉 |
| --- | --- | --- |
| Warmup Steps | 第一帧可能已经是不可恢复的错误抓取状态 | 先让机器人手调整到稳定姿态，再开始跟踪 reference |
| Random Force Perturbation | 短时看似能跟踪但抓取不稳 | 对 rollout 加随机力，鼓励更鲁棒的接触 |
| Transition Reward | 抓起/放下等离散接触转换容易错过 | 对应处于 resting / in-hand 的 reference 阶段，加失败惩罚 |

这种设计很实用：它不假设完美 MoCap reference，也不依赖手工抓取启发式，而是让优化器在仿真里找到更稳定的动作。

## 四、实验与结果

### 4.1 实验设置

重建评估使用 DexYCB 的 160 条视频和 HOI4D 的 12 条视频。为了隔离物体级别性能，实验提供 ground-truth hands，主要评估 object reconstruction 和 tracking。baseline 包括 HO、IHOI、HORSE、MCC-HO、G-HOP，以及 FoundationPose、Any6D 这类 6-DoF object tracker。

论文还收集 150 个更接近日常视频分布的 benchmark，来源包括互联网视频、egocentric datasets 和 generated videos。由于没有 ground-truth pose，作者用 3 名志愿者进行 head-to-head human preference。

重定向评估使用 655 条 in-the-wild reconstructed references，以及 OakInk2 的 1,352 条干净双手 MoCap 轨迹。成功标准是平均位置误差小于 0.1m、平均旋转误差小于 0.5 rad。机器人平台为 22-DoF Sharpa Wave hand；真实部署使用双 UR3e 机械臂 + Sharpa Wave hands，50 Hz 控制。

### 4.2 主要结果

**重建结果**：

| 方法 | DexYCB F-5 | DexYCB F-10 | DexYCB CD | HOI4D F-5 | HOI4D F-10 | HOI4D CD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FoundationPose | 0.69 | 0.89 | 0.89 | 0.71 | 0.91 | 0.49 |
| Any6D | 0.69 | 0.88 | 0.97 | 0.71 | 0.91 | 0.50 |
| G-HOP | 0.31 | 0.49 | 8.11 | 0.69 | 0.91 | 0.63 |
| Do as I Do | 0.71 | 0.93 | 0.66 | 0.72 | 0.91 | 0.49 |

在人类偏好评估中，raters 在 150 个 in-the-wild 视频里有 67% 选择 Do as I Do，18% 选择 FoundationPose，15% 为 tie。去掉 tie 后，Do as I Do 的 win rate 为 79%；75% 视频三名评审一致，Fleiss' kappa 为 0.65。

**重定向结果**：

| 方法 | Reconstruction Success | Pos | Rot | OakInk2 Success | Pos | Rot |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Annealed Sampling | 0.25 | 0.08 | 0.40 | 0.72 | 0.08 | 0.32 |
| + Warmup | 0.66 | 0.06 | 0.28 | 0.77 | 0.06 | 0.25 |
| + Perturbation | 0.67 | 0.06 | 0.30 | 0.79 | 0.03 | 0.14 |
| + Transition Reward | 0.71 | 0.05 | 0.28 | 0.81 | 0.03 | 0.15 |

在 noisy reconstruction 数据上，成功率从 25% 提升到 71%；在干净 OakInk2 MoCap 上，也从 72% 提升到 81%。这说明三个重定向组件不仅救 noisy reference，也能改善干净 reference 的动态可行性。

**真实部署与数据过滤**：

作者最终得到 500 条人工验证的高质量灵巧操作轨迹，来源为互联网 53%、egocentric 31%、generated 16%。真实机器人展示了 whisking、pouring、dusting、squeezing、tamping、erasing、stirring、hammering、spreading、picking 等 10 类动作。

数据过滤 playbook 很值得注意：从 100DOH 抽样 2,000 个 10 秒 clips，只有 187 个包含有意义 hand-object interaction，最终只有 83 个（4%）通过重建质量检查。作者认为不做预处理会带来约 20 倍有效数据惩罚。

### 4.3 消融实验

物体跟踪消融显示：

| Pose Guidance | Candidate Selection | DexYCB F-10 | DexYCB CD | HOI4D F-10 | HOI4D CD |
| --- | --- | ---: | ---: | ---: | ---: |
| Fixed | Clustering | 0.91 | 0.74 | 0.91 | 0.50 |
| Adaptive | Random | 0.91 | 0.74 | 0.87 | 0.66 |
| Adaptive | Log-likelihood | 0.93 | 0.65 | 0.91 | 0.49 |
| Adaptive | Clustering | 0.93 | 0.66 | 0.91 | 0.49 |

自适应 pose guidance 和 clustering 选择基本达到 log-likelihood 选择的质量，但后者计算成本高很多；论文称 clustering 与 likelihood 相当且最高可快 30 倍。

重定向消融更清楚：warmup 是最大增益来源，把 reconstructed reference 的成功率从 0.25 提到 0.66；perturbation 和 transition reward 带来更稳的接触和最终 0.71 成功率。

## 五、局限性与展望

作者明确指出几个限制：

1. **只假设刚体物体**：对布料、软物体、液体等非刚体交互不适用。
2. **依赖单目深度近似准确**：如果 MoGe 类深度估计失败，手物距离和尺度会直接出错。
3. **手物接触存在歧义**：单目 RGB 很难区分真实接触和视觉遮挡。
4. **没有完整场景重建**：当前只重建手和一个物体，不能处理障碍物、铰链、抽屉等环境约束。
5. **仿真动力学有上限**：MuJoCo 中可行不代表真实世界完全可行，尤其是接触丰富的多指操作。

未来方向很明确：把场景级重建、铰接物体、非刚体和更强 sim-to-real 加进来，才能把人类视频真正变成大规模机器人学习数据。

## 六、灵魂三问

1. **它解决了什么问题？**  
它解决的是单目人类视频无法直接作为灵巧手机器人数据的问题。以前的方法通常需要深度、MoCap、手部关键点或干净实验设置，而 Do as I Do 面向更开放的日常 RGB 视频。

2. **为什么这么做？**  
因为视频到机器人数据要同时解决重建和本体差异。先用视觉基础模型恢复手物 4D 轨迹，再用物理仿真中的 sampling-based optimization 生成机器人动作，比直接做几何映射更能处理接触和动态可行性。

3. **什么证据最有说服力？**  
最有说服力的是重定向消融和真实部署：noisy reconstructed references 上成功率从 25% 到 71%，并且最终在双 UR3e + Sharpa hands 上执行 10 类真实动作，说明 pipeline 不只是视觉指标更好，而能转化为可播放机器人轨迹。

## 七、个人总结

1. Do as I Do 的核心是把“看人做事”拆成可验证的工程链条：手物重建、尺度对齐、动态重定向、人工过滤、真实部署。
2. 最大优势是敢处理真正杂乱的数据源，并给出数据过滤效率分析；最大弱点是仍停留在刚体、单物体、近似深度和手工验证轨迹阶段。
3. 对 VLA/机器人数据 scaling 的启发是，人类视频确实有潜力，但中间的数据转化成本很高；可用数据比例、重建质量和接触可执行性会决定它是否真的能替代遥操作。

# BridgeVLA++：用时空记忆补全 3D VLA 的“下一步”和“落点”

> 原标题：BridgeVLA++: A Data-Efficient, Generalizable, and Memory-Augmented Vision-Language-Action Framework for 3D Manipulation  
> 作者：Peiyan Li, Yuze Zhu, Yixiang Chen, Qisen Ma, Yuan Xu, Jiabing Yang, He Guan, Yan Huang, Hongtao Wu, Xiao Ma, Tao Kong, Liang Wang, Tieniu Tan  
> 机构：中国科学院自动化研究所模式识别国家重点实验室（NLPR）/ 中国科学院大学 / FiveAges；部分作者在 ByteDance Seed 工作期间完成贡献  
> 发表：arXiv:2608.05042v1，2026-08-05；投稿 TPAMI。BridgeVLA 的 NeurIPS 2025 扩展  
> 链接：[arXiv](https://arxiv.org/abs/2608.05042) | [HTML](https://arxiv.org/html/2608.05042v1) | [项目主页](https://bridgevla-plus.github.io/)  
> 开源：项目页与论文未报告公开代码仓库（截至该论文 v1）。

---

## 一、研究背景与动机

3D 操作策略擅长利用几何结构，却常与预训练视觉语言模型（Vision-Language Model, VLM）的 2D 图像输入分布不匹配；把 3D 动作直接离散成 token 又会切断“观测空间位置 - 动作空间位置”的对应。原始 BridgeVLA 的做法是将 RGB-D 重建的点云正交投影为多视图图像，并预测语言条件的 2D 热图，再回投得到 3D 末端位置。

但其决策仍是 Markov 的：只看当前帧。因此会在两类任务失败：一是相似画面出现在不同操作阶段，模型不知道下一子目标；二是夹爪或已抓物体遮住目标区域，当前局部观测无法精定位。BridgeVLA++ 的目标是在不破坏原有 VLM 输入输出对齐及样本效率的前提下，补上显式记忆。

## 二、核心贡献

1. **保持 VLM 原生 2D 对齐的 3D VLA。** 以三正交视图的热图预测取代直接 3D 回归/动作 token，将预训练与下游动作学习统一为视觉定位问题。
2. **可扩展的语言条件热图预训练。** 用 RoboPoint 的 12 万检测样本，把文本指向的目标框转换为归一化 Gaussian 热图；随后迁移至机器人动作微调。
3. **阶段匹配的双记忆。** 粗阶段用时间记忆判断“接下来做什么”，细阶段用初始点云的重渲染空间记忆判断“具体在哪做”；二者都在 VLM visual-token 空间以 cross-attention 注入。
4. **跨设置验证。** 覆盖 RLBench、COLOSSEUM、GemBench、RMBench、MemoryBench 及 Franka / Dobot 两套真实平台；同时展示双臂扩展与低示范数训练。

## 三、方法原理

### 3.1 整体框架

一条专家轨迹记为 $\tau=(l,\{(o_t,a_t)\}_{t=1}^{H})$。$o_t$ 是一台或多台标定 RGB-D 相机观测，单臂下一关键帧动作为

$$
a_t=(x_t,R_t,g_t,c_t),\quad x_t\in\mathbb{R}^3,\ R_t\in SO(3),\ g_t,c_t\in\{0,1\}.
$$

其中分别为末端目标平移、旋转、夹爪状态与碰撞规避标志（没有该标签的 benchmark 不预测 $c_t$）。策略在稀疏关键帧调用，低层规划器执行后刷新观测。

处理链路：RGB-D -> 彩色点云 -> top/front/right 三张正交投影 -> PaliGemma（SigLIP + Gemma）与语言指令 -> 每视图平移热图。对工作空间采样候选 3D 点，将其投影到三张热图并求和，最大分数即为粗平移；以其为中心裁剪、放大、重渲染后用同权重主干再做一次细定位。全局 token 与预测点附近的局部 token 送 MLP，输出连续 6D rotation、夹爪和碰撞标志。

这不是“将点云编码给 LLM”。主干前向只接收投影 RGB 图和语言，刻意不输入关节/末端 proprioception 或逐像素 3D 坐标，以保留预训练 VLM 的图文分布。

### 3.2 关键技术细节

**热图接口。** RoboPoint 中每个目标框中心生成截断 Gaussian；多目标热图平均并归一化。模型把语言条件图像 token 还原为二维网格，用 convex upsampling 输出与输入同分辨率的空间 softmax 热图，采用像素级交叉熵。动作微调时，真值关键帧平移投影到每个视图，继续用同一热图损失；粗、细两阶段都受监督。热图提供比单一 3D 坐标更密集的监督，并将几何先验压回 VLM 熟悉的 2D 空间。

**时间记忆 $\mathcal{T}_t$，服务粗定位。** 缓存初始锚点视图 $A_0$、最近 $n=2$ 个关键帧，以及自适应挑选的子目标关键帧。子目标 gate 在已融合记忆的 token 上预测保留概率，只缓存真正带来新任务进度信息的帧；RMBench 最多 12 个 slot（2 个邻近帧 + 10 个子目标），其他 benchmark 无子目标标签，只保留 2 个邻近帧。缓存的是语言条件的 encoded token，非原始图像，形状为 $V\times N\times d$。

**空间记忆 $\mathcal{S}_t$，服务细定位。** 将 episode 初始点云 $P_0$ 在当前预测 waypoint 和当前 zoom 下按需重渲染。它与当前局部视图使用同一虚拟相机，因此逐视图对齐；当前 token 仅查询同视图的空间 token。初始点云提供未被夹爪遮挡的几何，当前帧保留最新状态，两者互补而不是用旧场景替换新场景。

每个 memory injection block 有两层 attention：当前 token 作 query，memory 作 key/value，随后 self-attention 与 FFN 融合，输出形状不变，仍可直接接原热图与动作头。时间、空间模块与子目标 gate 分别约 168M、84M、18M 参数。双臂时共享 VLM 和两种记忆，只复制每臂的热图/动作头及各自 coarse-to-fine crop。

### 3.3 训练与优化

训练分两阶段：先冻结 PaliGemma，只训练 convex upsampler、动作头、memory block 与 gate；后续解冻主干共同训练，但 SigLIP vision encoder 和语言 token embedding 始终冻结。2D 热图预训练仅做一次，所有下游微调从这套权重 warm-start；新增 memory 模块和 gate 只在微调时从零训练。

单臂演示被 PerAct 规则切成连续关键帧 transition：机器人静止、夹爪状态改变或轨迹末帧即为关键帧；RMBench 使用双臂版本。训练会对点云与真值动作共同做 SE(3) 增广（平移与 yaw），并对第二阶段 zoom 加扰动。启用 memory 时关闭随机平面图像增强，工作空间固定边界而非按帧点云均值居中，避免破坏当前图与缓存 token 的像素对应。

目标函数为 $L=L_{trans}+L_{rot}+L_{gripper}+L_{collision}$：平移是粗/细热图交叉熵，旋转为 6D rotation 还原矩阵与真值的平方 Frobenius 距离，后两项为二分类交叉熵。

### 3.4 数据使用与维度追踪

| 数据源 | 规模 | 样本单位 | 模态/字段 | 关键维度 | 标签/动作 | 用途阶段 | 处理方式 |
|---|---:|---|---|---|---|---|---|
| RoboPoint detection split | 120K | 图文检测样本 | 单图、目标文本、bbox | 原图/token 长度未报告 | bbox 中心 -> 截断 Gaussian 热图 | 2D heatmap pretrain | 多目标平均归一化；空间 softmax + CE |
| RLBench | 18 tasks x 100 demos | 相邻关键帧 transition | 4 路 RGB-D、语言 | 传感器 128x128；渲染为 3 x 224x224 | $x\in\mathbb{R}^3$、6D rot、gripper、collision | finetune/eval | 点云、三正交投影、coarse-to-fine；SE(3) aug |
| COLOSSEUM | 论文表中每任务 100 demos | 同上 | 4 路 RGB-D、语言 | 传感器 128x128；渲染 3 x 224x224 | 同 RLBench | finetune / OOD eval | 12 类未见视觉扰动 |
| GemBench | 每 variation 100 demos | 同上 | 4 路 RGB-D、语言 | 传感器 256x256；渲染 3 x 224x224 | 平移、6D rot、gripper | finetune / 组合泛化 eval | collision head 关闭 |
| RMBench | 9 个双臂任务，各 50 demos | 双臂关键帧 transition | 4 路 RGB-D、分段语言 | 传感器 224x224；渲染 3 x 224x224；K=12 | 每臂一套 $(x,R,g)$，子目标标签 | finetune / memory eval | 双臂关键帧规则；段末为正子目标 |
| MemoryBench | 每任务 100 demos | 单臂关键帧 transition | 4 路 RGB-D、语言 | 传感器 128x128；渲染 3 x 224x224；K=2 | 平移、6D rot、gripper | finetune / memory eval | 无子目标标注，gate 关闭 |
| 真实 Franka / Dobot | 每任务/指令 10 demos | 论文未完整报告 | ZED 2i RGB-D、语言 | 相机/渲染具体维度未报告 | 与平台相符的关键帧动作 | real-robot train/eval | Franka 13 任务；Dobot 每设置 10 trials |

**维度快照**

- Observation：一或多台标定 RGB-D -> 点云 -> 固定三视图（top/front/right）224x224；RLBench 原始为 4 路相机。点数、RGB 归一化与语言最大长度均未报告。
- Language：RoboPoint 是目标描述 prompt；操作数据是任务指令。tokenizer、模板、最大 token 数与 mask 规则未报告。
- State：不将 proprioception、关节角或末端 pose 输入 VLM；内部动作平移是 3 维。
- Action：单臂为平移 3D + 连续 6D 旋转 + 二值 gripper + 可选 collision；双臂是左右两套动作。控制频率、坐标系、夹爪编码语义与动作执行后处理未报告。
- Prediction target：下一稀疏关键帧；平移以三视图的高斯热图监督，细阶段结果执行。

数据视角的关键点是：2D 大规模检测数据并不直接教机器人轨迹，而是先把 VLM 的输出接口改造成“语言指向空间位置”；少量机器人 demo 再把这个接口与几何回投、姿态和夹爪监督绑定。记忆训练只从同一 expert trajectory 的先前观测构建，没有额外视频预训练或跨 episode memory。

## 四、实验与结果

### 4.1 实验设置

仿真包括：RLBench 的 18 个 Franka Panda 操作任务（4 RGB-D 相机，100 demos/task，5 次评估各 25 episode）；COLOSSEUM 的 12 个 OOD 扰动轴及 GemBench 的新物体/颜色组合；双臂长期记忆 RMBench；单臂 MemoryBench。真实环境包括 7-DoF Franka Research 3 和 6-DoF Dobot CR5A，均由静态 ZED 2i depth camera 观测。Franka 的 13 个任务每任务仅 10 demo，并额外评估 distractor、lighting、background、height、未见 object-skill combination、未见 category；Dobot 记忆任务也按每条指令 10 demo 和各设置 10 trial。

### 4.2 主要结果

| 场景 | 指标 | BridgeVLA | BridgeVLA++ | 关键对比 |
|---|---:|---:|---:|---|
| RLBench 18 tasks | Avg. SR | 90.5% | 93.7% | 超过 SAM2Act 的 86.8%；memory 还提升一般操作 |
| COLOSSEUM OOD | Avg. SR | 64.0% | 65.2% | BridgeVLA 比 RVT-2 高逾 7 pt |
| GemBench | Avg. SR | 50.0% | 51.1% | 记忆未牺牲组合泛化 |
| RMBench 双臂记忆 | Avg. SR | 18.9% | 96.0% | 比 MemoryWAM 高 13.0 pt |
| MemoryBench 单臂记忆 | Avg. SR | 未报告 | 99.7 +/- 0.3% | 证明不只适用于双臂 |
| Dobot 记忆任务，Basic | SR | 20.0% | 93.3% | SAM2Act+ 为 30.0% |
| Dobot 记忆任务，扰动平均 | SR | 15.8% | 78.3% | 背景/高度/光照仍保持优势 |

最强证据并非单纯 RLBench 的数点增益，而是同一 base policy 在 RMBench 从 18.9% 提升至 96.0%，且真实 Dobot 由 20.0% 到 93.3%。这更直接隔离了“任务确实需要 episode memory”与“模型只是更大”的区别。

### 4.3 消融实验

| 消融 | 结果 | 含义 |
|---|---:|---|
| 直接回归位置，去热图 decoder | RLBench 90.5% -> 31.4% | 2D 热图的稠密监督和输入输出空间对齐是核心 |
| 向 VLM 融合逐像素 3D position | 90.5% -> 56.2% | 更多 3D 特征反而破坏预训练图像分布 |
| 6D rotation 改离散 Euler | 90.5% -> 88.2% | 连续旋转更利于精细姿态且避免 gimbal lock |
| 去 spatial memory | RLBench 93.7% -> 92.0%；Sort Shape 72.0% -> 60.8% | 它主要解决遮挡下的局部几何 |
| 去 temporal memory | RMBench 96.0% -> 21.3% | 长时任务的决定性组件 |

## 五、局限性与展望

**论文呈现的边界：** 空间记忆取 episode 初始点云，适合“初始时看得清、随后被机械臂遮住”的情况；若关键目标本身在中途才出现、被移动或初始建图错误，旧几何会不可靠。当前缓存仍以关键帧为粒度，子目标 gate 的强监督只来自 RMBench 的分段语言标注。

**实现与部署层面的推断：** 该方法依赖标定 RGB-D、点云质量和固定工作区；它将 3 张 224x224 正交图输入大型 PaliGemma，且额外增加约 270M memory 参数，真实机器人闭环时延/吞吐未报告。论文也未说明动作坐标系、控制频率、点云清洗与跨 embodiment 归一化，复现硬件结果仍有工程缺口。更强的方向是用可更新的 3D scene memory 处理动态物体，并把 memory 写入/置信度与失败恢复结合。

## 六、灵魂三问

1. **它解决了什么问题？** 不是泛泛地给 VLA 加历史 token，而是在 3D 操作中区分“下一步子目标依赖过去”与“当前局部被遮挡无法定位”两种失败模式。它保留 BridgeVLA 的少样本几何接口，同时使策略不再完全 Markov。

2. **为什么这么做？** BridgeVLA 已把点云和动作共同桥接到 2D 热图空间，直接塞入 3D 特征会让 VLM 偏离图文预训练分布。于是时序和空间历史也保持为同一个 visual-token 表示：粗阶段读历史来选区域，细阶段读重渲染初始点云来恢复可见几何，和 coarse-to-fine 的职责严格对齐。

3. **什么证据最有说服力？** RMBench 的 18.9% -> 96.0% 及 Dobot 基础设置 20.0% -> 93.3% 是最干净的记忆增益证据；去时间记忆后 RMBench 降到 21.3%，而去空间记忆主要伤害遮挡精定位任务，进一步说明双记忆不是冗余堆叠。

## 七、个人总结

1. BridgeVLA++ 的核心价值是把 VLM 对齐问题处理得很彻底：**预训练、动作监督和记忆读取都留在 image-token / heatmap 域**，3D 只在点云投影与候选点回投处显式出现。
2. 相比把长历史塞进 transformer context，它的时间记忆有明确的 slot 预算和子目标选择；相比只记图像，它的空间记忆用相同 virtual camera 对齐初始点云，针对性很强。代价是对 RGB-D 标定、初始观测和固定场景假设的依赖也更强。
3. 对 VLA 预训练/数据工程的启示是：可先收集便宜的 2D grounding 数据训练一个可转移的空间输出接口，再用小规模 robot demos 学 3D action binding；这比把无关模态直接拼到 VLM 输入端更值得优先验证。后续应重点测量这种接口在开放世界视频、动态场景与非固定相机上的迁移边界。

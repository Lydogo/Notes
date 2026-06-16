# H-RDT：用人类操作数据增强双臂机器人操作

> 原标题：H-RDT: Human Manipulation Enhanced Bimanual Robotic Manipulation
> 作者：Hongzhe Bi, Lingxuan Wu, Tianwei Lin, Hengkai Tan, Zhizhong Su, Hang Su, Jun Zhu（清华大学 + 地平线机器人 Horizon Robotics）
> 发表：arXiv preprint，2025 年 8 月（arXiv:2507.23523v2）
> 链接：https://arxiv.org/abs/2507.23523

---

## 一、研究背景与动机

- **领域核心问题**：模仿学习（Imitation Learning）做机器人操作的根本瓶颈是**高质量演示数据稀缺**。遥操作（teleoperation）昂贵且依赖熟练操作员；UMI、动捕等替代方案基础设施复杂、数据质量不一致，规模都上不去。
- **现有方法不足**：当前主流 VLA 模型（OpenVLA、RDT、π0、π0.5）走的是"跨形态预训练"路线，在 Open X-Embodiment、AgiBot World 这类聚合数据集上训练。但不同机器人形态（morphology）和动作空间（action space）差异巨大，统一训练效果受限；而且这些机器人数据总量仍然偏小、质量异构。
- **本文出发点**：人类操作视频是天然的、海量的、低成本的"演示数据库"。EgoDex 数据集已经提供了 829 小时（338K episodes）的第一人称视角操作视频 + 逐帧 3D 手部姿态标注。本文核心目标：**把人类作为一种"统一的本体"（unified embodiment），从大规模人类操作视频中学到可迁移的操作先验，再 fine-tune 到任意机器人形态上**。

## 二、核心贡献

1. **首个系统化框架**，把大规模第一人称人类操作视频（带 3D 手部姿态标注）作为机器人策略学习的预训练数据源，规模上比 EgoMimic（2K demos）和 HAT（27K demos）大一两个数量级（338K trajectories）。
2. **模块化 diffusion transformer 架构**：vision/language 编码器 + transformer backbone 在预训练阶段学到的多模态表征可以**完整继承**，仅需重新初始化 state/action adapter 和 action decoder 即可适配任意机器人形态。
3. **两阶段训练范式 + flow matching**：第一阶段在 EgoDex 上预训练 48 维人手动作；第二阶段在目标机器人上 fine-tune，支持任意 dual-arm 形态（Aloha-Agilex、Franka、ARX5、UR5+UMI 等）。
4. **充分实验证据**：在仿真（RoboTwin 2.0）和真实平台（5 种机器人）上，相比从零训练分别有 +13.9%（仿真）和 +40.5%（真机）的成功率提升，超越 RDT、π0 等 SOTA 方法。

## 三、方法原理

### 3.1 整体框架

H-RDT 是一个 **2B 参数的 diffusion transformer**，输入为多视角 RGB 图像、proprioceptive state、语言指令，输出为长度为 $H$ 的未来动作序列 $a_{t:t+H}$。整体由五个模块组成：

```
RGB 图像 → DinoV2 + SigLIP → 图像 cross-attention →
语言 → T5-XXL → 语言 cross-attention →            ┐
state → StateAdapter (MLP) ──────────────────────┤→ Transformer Backbone (LLaMA-3 风格) → ActionDecoder (MLP) → v_θ
noisy action a^τ → ActionAdapter (MLP) ──────────┘                       ↑
                                                              flow time τ → AdaLN
```

关键架构选择：
- **图像/语言走 cross-attention 注入**（参考 RDT），避免与 state/action 走 self-attention 时的模态不平衡。
- **state 和 action 拼接后走 self-attention**，让模型学到"从当前本体感知 + 噪声动作 → 干净动作"的映射。
- **flow time τ 通过 AdaLN 调制**（参考 DiT）。

### 3.2 关键技术细节

#### (1) 统一的人手动作表示（48 维）

为解决人和机器人之间的形态差异，作者**不**采用"flow as transit"（仅高层物体运动）或"显式重定向"（强假设运动学结构），而是把人手动作压缩成一个 **48 维向量**，作为多数机器人末端执行器动作空间的"超集"：

- 双侧腕部位姿（位置 3D + 6D 旋转表示）：$2 \times 9 = 18$ 维
- 双手所有指尖 3D 坐标：$30$ 维

这个表示的好处：
- 双侧腕位姿与典型机器人 EE pose 直接对齐，迁移时不损失语义。
- 指尖坐标承载抓取、夹持构型等"通用可迁移特征"，即使目标机器人末端只是平行夹爪（gripper），这些信息在预训练阶段也能让 transformer 学到合理的"操作语法"。

#### (2) Flow Matching 训练目标

采用 flow matching 而非传统 diffusion，理由是**训练更稳、推理更快**。给定真实动作 $a^*_{t:t+H}$，构造直线流路径：

$$a^\tau = \tau \cdot a^*_{t:t+H} + (1-\tau) z, \quad z \sim \mathcal{N}(0, I)$$

直觉：在 $\tau$ 从 0 到 1 时，沿一条"噪声 → 真实动作"的直线插值。模型 $v_\theta$ 学到沿这条路径的速度场（vector field）：

$$\mathcal{L}_{FM} = \mathbb{E}_{\tau, z, a^*, c} \| v_\theta(a^\tau, \tau, c) - (z - a^*_{t:t+H}) \|^2$$

其中 $c = \{o_t, s_t, l\}$ 是图像、state、语言条件。**目标向量 $z - a^*$ 就是从真实动作指向噪声的方向**，模型学到的是反向（噪声→真实）的速度。

推理：从纯高斯噪声 $a_0 \sim \mathcal{N}(0, I)$ 出发，用确定性 ODE 求解器走 5 步，每步 $a_{t+\Delta t} = a_t + \Delta t \cdot v_\theta(a_t, t, c)$，$\Delta t = 0.2$。仅 5 步 NFE（Number of Function Evaluations）就能在 30Hz 控频下实时跑。

#### (3) 两阶段训练范式

**Stage 1：人类数据预训练**
- 数据：完整 EgoDex（338K+ trajectories，194 个不同操作任务，829 小时）。
- 输入：单个第一人称视角 + 语言指令 + 上一帧 48 维人手 state。
- 输出：未来 $H$ 步的 48 维人手动作序列。
- 目标：让 vision encoder + language encoder + transformer backbone 学到"看图 + 听指令 → 推理操作意图 → 生成手部轨迹"的通用知识。

**Stage 2：跨形态 fine-tune**
- 选择性权重迁移：
  - **保留** vision encoder、language encoder、transformer backbone（已学到的视觉/语义/操作先验）。
  - **重新初始化** StateAdapter、ActionAdapter、ActionDecoder（适配目标机器人的 state/action 维度，例如 14 维 dual-arm + 平行夹爪）。
- 这种选择性迁移使得**同一个预训练模型可以适配任意双臂机器人**，只需替换头尾的 MLP 即可。

### 3.3 训练与优化

- 优化器：AdamW，lr=1e-4，wd=0.01，grad clip=1.0。
- 精度：bfloat16 混合精度 + gradient accumulation。
- 模型规模：2B 参数；hidden=2176，16 层，16 头注意力，GQA k/v heads=8。
- 图像：编码为 196 个 patches。
- 语言：T5-XXL，最大 1024 tokens。
- Flow matching：$\tau \in [0, 0.999]$ 均匀采样训练，5 步 ODE 推理。
- 训练规模：仿真单任务 10K steps × 4 H100；多任务 30K steps × 4 H100；真机 ARX5 100K × 4 H100、UR5+UMI 20K × 8 H100、Aloha-Agilex 50K × 8 H20。

## 四、实验与结果

### 4.1 实验设置

- **仿真**：RoboTwin 2.0 平台，13 个单任务（Easy/Hard mode）+ 45 个多任务（Hard mode）；评估 Aloha-Agilex-1.0 和 Franka-Panda 两种平台。
- **真机**：Aloha-Agilex-2.0（dual Piper）做毛巾折叠和杯子放置；ARX5 dual-arm 做 113 个 few-shot pick-place（每任务 1-5 个 demo）；UR5+UMI 做双臂外卖袋放置（4 个 subtask）。
- **基线**：RDT、π0、w/o human（自身去掉人类预训练）。

### 4.2 主要结果

**真机毛巾折叠**（Aloha-Agilex-2.0）：

| 方法 | 完整成功率 |
|---|---|
| RDT | 40% |
| w/o human | 0% |
| **H-RDT** | **52%** |

**真机杯子放置**（需空间推理选择左右手）：

| 方法 | 完整成功率 |
|---|---|
| RDT | 28% |
| w/o human | 20% |
| **H-RDT** | **64%** |

**Few-shot 真机实验**（ARX5，每任务 1-5 demo，113 任务）：

| 方法 | 平均成功率 |
|---|---|
| RDT | 16.0% |
| π0 | 31.2% |
| w/o human | 17.6% |
| **H-RDT** | **41.6%** |

**多任务仿真**（RoboTwin 2.0 Hard mode，45 任务）：

| 方法 | 平均成功率 |
|---|---|
| RDT | 28.8% |
| π0 | 48.4% |
| w/o human | 67.2% |
| **H-RDT** | **87.2%**（+20.0% over scratch） |

**跨形态泛化**：H-RDT 在 Aloha-Agilex-1.0 上 87.2%，在 Franka-Panda 上 62.9%（分别比从零训练 +20.0% 和 +18.9%），证明同一个预训练 backbone 能稳定迁移到不同形态。

### 4.3 消融实验

论文消融主要体现在 "w/o human"（同模型同结构，不做人类预训练）这个 baseline：
- 单任务仿真：H-RDT 比 w/o human 在 Easy/Hard 都高 8.4%。
- 多任务仿真：H-RDT 比 w/o human 高 20.0%（多任务场景人类先验收益更大）。
- Few-shot 真机：H-RDT 比 w/o human 高 24%（数据稀缺场景人类先验收益最大）。

**结论**：增益主要来自**人类预训练**这一项，而非架构本身或 flow matching 的功劳——同样的架构没有人类预训练时表现就退回到 baseline 水平。

## 五、局限性与展望

- **作者提到的局限**：原文没有专门的 limitations 章节，但从数据/方法可以看出：
  - 仅使用单视角第一人称图像预训练，未来可扩展多视角。
  - 仅以 48 维 hand pose 作为人类动作表示，缺少接触力、物体本体姿态等更细粒度信号。
  - Stage 2 重新初始化 adapter 意味着真机 fine-tune 仍然需要一定数量的演示（虽然显著少于 baseline）。

- **潜在改进方向**（个人补充）：
  - 把 EgoDex 之外的 Ego4D、HOI4D、Hot3D 等更大规模视频纳入预训练。
  - 引入物体接触/力觉信号补充 hand pose。
  - 把 ActionDecoder 设计成"任意机器人 query 适配器"，实现一次预训练即可零样本对接新形态。

## 六、灵魂三问

1. **它解决了什么问题？**
   之前要做一个能干各种活的双臂机器人模型，必须靠遥操作攒昂贵的真机数据，且换一个机器人就基本要重训；现在能用海量的"人戴 AR 眼镜做家务"视频先把通用操作语义灌进 transformer，再用极少（甚至 1-5 条）真机 demo 把模型适配到任意双臂机器人。

2. **为什么这么做？**
   最关键的设计是**用 48 维双手 pose（双腕 + 双手指尖）作为统一的人类动作表示**。它既不是"flow"那种过度抽象（丢失动作参数），也不是"hand-to-robot retargeting"那种过度具体（强假设目标机器人结构），而是一个**双臂机器人 EE pose 的"超集"**——预训练学到的腕位姿语义可以直接对齐机器人末端，指尖语义在 fine-tune 时通过 ActionDecoder MLP 映射到具体夹爪动作。这样既保留了人类数据的丰富信息，又保持了对任意机器人形态的兼容性。

3. **什么证据最有说服力？**
   **ARX5 few-shot 实验**：113 个任务每任务只有 1-5 条 demo 的极端数据稀缺场景下，H-RDT 41.6% vs π0 31.2% vs w/o human 17.6%。这个实验最能证明"人类先验"真的迁移过来了——因为在 5 条 demo 这种极少数据下，模型几乎不可能从机器人数据本身学到操作策略，能成功只能靠预训练带过来的"知道怎么抓东西、怎么放东西"。同时 w/o human 相比 H-RDT 掉了 24 个点，把"是不是架构本身的功劳"也证伪了。

## 七、个人总结

1. **核心 idea**：把"统一的人手动作（48 维 wrist + fingertip）"作为人和机器人之间的桥梁，用 EgoDex 大规模预训练 + 模块化 fine-tune，让 VLA 模型摆脱"机器人数据稀缺"的诅咒。

2. **优势与不足**：
   - **优势**：训练范式优雅、迁移成本低、数据成本低；模块化 adapter 设计让任意双臂机器人都能受益于同一个预训练 backbone；few-shot 场景下提升显著，对实际部署最有价值。
   - **不足**：仍局限于双臂 EE 控制场景，对灵巧手、单臂、移动底盘等形态扩展未验证；预训练只用 EgoDex 一个数据源，规模上仍有上限；缺少与 GR00T、π0.5 等更新模型的对比。

3. **启发**：
   - 对**研究**：印证了"人类视频是 VLA 的 ImageNet"这个方向的可行性，后续工作可以朝着"更大规模视频预训练 + 更通用的人类动作 token"方向走（参考 EgoDex、Hot3D、Ego-exo4d 数据集）。
   - 对**实际应用**：对于像 PICO ego pipeline 这种自采第一人称数据的项目，H-RDT 的 ActionDecoder/StateAdapter 模块化设计是个直接可复用的范式——预训练阶段固定 48 维人手表示，fine-tune 阶段为每个机器人形态训独立的 adapter，可以避免"换机器人就重训"的工程成本。

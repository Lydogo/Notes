# MagicVLA Dynamic / SGM：动作与世界表征联合学习

> 本文说明模型原理与主要设计。具体尺寸、loss权重和分支可见关系随实验配置变化。
> 项目复述见 [项目笔记](../../Note_Project.md)；基础公式见 [基础知识](../../Note_Basics.md)。

模块级细节见 [SGM 模块与训练细节](02_SGM_模块与训练细节.md)，包括state编码、DeltaNet、预测头、loss、mask与数值稳定性。

## 1. 要解决什么问题

Base VLA根据图像、语言和状态生成动作。单独的动作监督告诉模型“应该怎样动”，但不会显式要求它预测场景的视觉变化、几何结构或运动关系。

Dynamic / SGM在动作专家旁加入2D、3D World Experts，用额外监督学习这些表征，再让动作分支读取它们。核心问题是：**视觉与运动预测能否提供更有用的动作条件？** 是否改善任务表现，仍需闭环实验判断。

这里的World Expert预测latent特征，不直接生成RGB视频，也不是在外部规划器中反复模拟多个候选动作。

## 2. 从 base 到多专家结构

Base包含Qwen VLM和连续Action Expert。Dynamic保留这条动作路径，新增两条并行的World分支。

```text
当前图像 + 任务文本 + state/mask
                ↓
           Qwen VLM prefix
                ↓ 提供条件 K/V
       ┌────────┼────────┐
    2D World  3D World  Action
       │        │        │
   视觉特征  几何/运动特征  动作 velocity
```

| 分支 | 直接输入 | 预测目标 |
|---|---|---|
| VLM | 图像、语言和状态条件 | 条件表征；训练时可带语言/FAST CE |
| Action | 带噪动作、flow time、state | 动作空间中的flow velocity |
| 2D World | 可学习queries，加位置/视角信息 | 未来图像的DINOv3 patch特征 |
| 3D World | 可学习queries，加位置信息 | Track4World的geometry与motion特征 |

World queries是待更新的特征槽位，不是真实未来帧的编码。它们通过attention读取当前条件和允许交互的专家信息，学习预测teacher给出的目标。

代表性配置使用50个动作token、每个头部视角256个2D queries，以及256个3D queries。query数控制特征分辨率和计算量，不等于未来视频帧数。

## 3. 信息在哪里交互

各分支采用混合层结构：DeltaNet层独立更新，在full-attention层交换信息。代表结构为24层，每4层出现一次full attention，共6次交互。

对某个专家来说：

```text
Q = 当前专家的 query 投影
K/V = VLM prefix + 配置允许读取的专家 K/V
输出 = Attention(Q, K, V) → 当前专家的输出投影
```

交互发生在主干内部，不是各专家完整运行后再拼接最终结果。不同分支的hidden size可以不同，但参与同一次attention的head维度需要兼容。

默认关系如下：

| Query → K/V | VLM | 2D World | 3D World | Action |
|---|---|---|---|---|
| VLM | causal | — | — | — |
| 2D World | ✓ | ✓ | ✓ | ✓ |
| 3D World | ✓ | ✓ | ✓ | ✓ |
| Action | ✓ | ✓ | ✓ | ✓ |

VLM不读取专家token。三个专家可以相互读取，但可通过 `joint_attention_visibility` 改成其他结构。例如关闭Action对2D的读取，就能研究动作是否主要依赖3D信息。

“Action能读取World”与“Action一定有效利用World”不同，需要屏蔽、交换或打乱分支等对照实验判断。

## 4. Teacher 如何提供监督

Teacher是冻结的目标生成器；World Expert是需要训练的预测器。两者职责不同。

### 2D：预测未来视觉表征

```text
未来图像 → 冻结 DINOv3 → patch target
当前条件 → 2D World Expert → predicted patch latent
                                ↓
                          特征匹配 loss
```

目标是让World queries学到与未来外观相关的表示，而不必预测每个RGB像素。

### 3D：学习几何与运动表征

```text
当前图像 + 未来图像 → 冻结 Track4World
                       ├─ geometry latent
                       └─ motion latent
当前条件 → 3D World Expert → 两个预测头 → 特征匹配 loss
```

geometry和motion分别提供场景几何与当前到未来的运动关系监督，不应都简称为“未来3D坐标”。具体目标是teacher中间特征，而非直接的机器人joint标签。

### 时间对齐

World target应与动作chunk对应的时间跨度一致。若50个动作点来自2倍上采样，它们对应约25个源帧间隔；不能仅按动作token数选取未来视频帧。

未来图像只用于训练目标生成，不能作为部署输入。越过episode尾部时，可复用终止帧作为World target；动作padding是否监督由数据配方决定。

## 5. 训练目标与梯度

机器人batch的目标可概括为：

```text
L_robot = λ_action L_flow
        + λ_2d L_2d
        + λ_3d (λ_geometry L_geometry + λ_motion L_motion)
        + λ_fast L_FAST
```

语言batch另用VLM-SFT CE。哪些损失开启、权重大小及是否调度，由对应训练配方决定。

Flow Matching沿用base：

```text
x_t = (1-t) × action + t × noise
velocity target = noise - action
```

这里的velocity是生成过程中的变化率，不是机械臂的物理关节速度。有效维度和有效时间位置通过mask参与loss。

### 三种控制不要混淆

| 控制项 | 改变什么 | 不代表什么 |
|---|---|---|
| `joint_attention_visibility` | 各专家可读取的K/V，训练和推理均生效 | 不直接决定参数是否冻结 |
| `action_world_stream_mask` | 训练时随机屏蔽Action对部分World信息的读取 | 不删除World Expert或teacher |
| `knowledge_insulation` | 隔离机器人loss到VLM的梯度 | 不必然冻结整个VLM |

开启KI时，Action/World分支仍读取VLM条件，但机器人loss不通过该路径更新VLM；VLM可以由FAST或语言CE学习。关闭KI后允许联合更新，实际更新哪些参数还取决于冻结设置。

部分配方逐步降低辅助loss权重，使后期训练更侧重动作目标。这是权重调度，不是把辅助分支从模型中移除。

## 6. 推理时还需要哪些模块

部署不需要真实未来帧，也不需要DINOv3 / Track4World teacher；训练好的World Experts仍保留在动作生成路径中。

```text
当前观测 → VLM 前向一次 → 缓存交互层 prefix K/V
                                  ↓
masked noise → World/Action 专家前向 → velocity → Euler 更新
                    ↑                                │
                    └──────── 多次迭代 ───────────────┘
                                  ↓
                     反归一化 → 恢复动作 → 执行
```

由于VLM不读取动作或World suffix，其条件可以在固定观测的一轮生成中缓存。专家间存在交互时，World/Action的特征仍需随专家前向更新，不能认为World只算一次，或把它的开销等同于base。

观测变化后重建条件。动作输出还需使用对应版本的normalization、delta恢复和维度mask；模型生成的归一化张量不是直接可执行的机器人命令。

## 7. 32D 与 34D 分别理解

Dynamic是模型结构扩展，32D/34D是动作接口版本，两者不是同一个概念。

| 内容 | 32D版本 | 34D版本 |
|---|---|---|
| 左joint / gripper | `0:6` / `6` | `0:7` / `7` |
| 右joint / gripper | `7:13` / `13` | `8:15` / `15` |
| 左EEF xyz / rot6d | `14:17` / `17:23` | `16:19` / `19:25` |
| 右EEF xyz / rot6d | `23:26` / `26:32` | `25:28` / `28:34` |

34D为每臂保留最多7个joint槽位。6-joint数据映射后，多出的槽位无效；Ego人手数据没有真实arm joint，相关槽位仍关闭mask。

源数据可以保持32D落盘，再由reader映射到34D。两版分别使用自己的动作索引、输入输出层、mask和归一化统计。理解共同原理时可以复用数据流图，复现实验时应选择同一版的整套配置。

## 8. 与 memory 的关系

| 方向 | 信息来源 | 主要问题 |
|---|---|---|
| Memory | 已经发生的历史观测 | 当前图像不足以确定任务状态 |
| Dynamic / SGM | 训练时的视觉、几何与运动预测监督 | 动作条件是否包含有用的场景变化信息 |

两者可以组合，但Dynamic本身不等于长期记忆。评价时分别检查：历史是否被使用、World分支是否影响动作，以及这些变化是否带来闭环收益。

## 9. 复习要点与代码入口

复述时按四步说清楚：**在base旁增加什么 → 目标由谁提供 → 分支如何交互 → 部署保留什么。** 不必背每个实验的loss权重。

代码入口：

- [模型](../../../magicvla/src/models/magicvla_sgm/modeling_magicvla_sgm.py)：experts、joint attention与缓存推理。
- [配置](../../../magicvla/src/models/magicvla_sgm/configuration_magicvla_sgm.py)：分支可见性、KI、loss与时间跨度。
- [Teacher](../../../magicvla/src/models/magicvla_sgm/world_teachers.py)：2D/3D目标构造。
- [在线推理](../../../magicvla/scripts/infer/robodojo_dynamic_server.py)：观测和动作接口。

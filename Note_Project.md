# 核心项目与面试复盘

本文保留超维动力、魔法原子项目和全部面试复盘。早期机械臂及Isaac Lab/RL实践见[辅助项目记录](Note_OtherProjects.md)，通用概念见[基础知识](Note_Basics.md)。

- [超维动力](#project-chaowei)
- [魔法原子](#project-magicatom)
- [面试复盘](#interviews)

# 一、项目实战

<a id="project-chaowei"></a>

## 超维动力工作总结

### PI05 归一化统计

基础查阅：[动作表示与归一化](Note_Basics.md#basic-normalization)。

#### 1. norm_stats 是什么

`compute_norm_stats.py` 会输出 `norm_stats.json`，包含 `state`（机器人当前状态）和 `actions`（动作指令）两组统计量，每组4个值：

| 字段 | 含义 | 计算方式 |
|---|---|---|
| `mean` | 均值 | 全数据加权滑动平均 |
| `std` | 标准差 | √(E[x²] - E[x]²) |
| `q01` | 第1百分位数 | 直方图近似 |
| `q99` | 第99百分位数 | 直方图近似 |

为什么需要：每个关节的角度范围相差几百倍，直接喂给模型会让loss和梯度被波动大的维度主导，波动小的维度信号被淹没。归一化把每个维度除以自己的标准差拉到同一尺度，让模型对每个关节同等关注。

#### 2. `compute_norm_stats.py` 计算速度优化

**核心优化：跳过视频解码**

`compute_norm_stats` 只需要 `state` 和 `actions` 的统计量，根本用不到图像；但v1直接复用 `LeRobotDataset`，每次取样都会走 `_query_videos` 把对应帧的mp4解码出来，纯属浪费CPU。

v2的思路是用一个子类覆盖 `_query_videos`，让它直接返回与真解码同shape的全零张量。下游 `RepackTransform` / `AlohaInputs` / `DeltaActions` 等transform看到的dict结构和维度跟原来完全一致，不需要任何改动；而图像本来就是transform pipeline的旁路，不参与 `state`/`actions` 的统计，因此 `norm_stats` 数值结果与v1同分布，是纯加速优化。

实测提速：`pico_ego_V7` 是av1编码、1536×2048视频，mp4解码极吃CPU，v1在8-worker下几乎是CPU-bound；v2只读parquet，全量遍历1500w帧从6–10小时压到1小时以内（5–10×）。

### PI0 训练流程笔记

PI0 / PI0.5 / PI0.7都是Physical Intelligence出的视觉-语言-动作（VLA）模型，把机器人控制建模成"条件生成"问题：给定多视角图像、语言指令和本体状态，输出未来H步的连续动作序列。

#### 0. 家族总览

| 模型 | 时间 | 核心改动 |
|---|---|---|
| PI0 | 2024.10 | PaliGemma VLM + Action Expert + Flow Matching，10K小时同质数据 |
| PI0-FAST | 2025 | 用FAST tokenizer把动作离散化成token，走纯自回归路径 |
| PI0.5 | 2025.04 | FAST离散预训练 + flow matching后训练；异构多源数据联合训练；层次化高低层推理 |
| PI0.7 | 2026 | Steerable：多模态prompt（subgoal图、episode metadata）+ 知识隔离KI |

#### 1. 模型架构：双专家如何与 VLM 交互

基础查阅：[Attention](Note_Basics.md#basic-attention)、[Masked Attention](Note_Basics.md#basic-masked-attention)。

##### 1.1 整体结构

- **Prefix侧（VLM专家）**：SigLIP编码图像 → image embedding；Gemma-2B词嵌入 → language embedding。约2.7B参数，从PaliGemma初始化。
- **Suffix侧（Action Expert）**：`state_proj` → 状态embedding；`action_in_proj` + time MLP → 动作 + 时间embedding。Gemma-300m架构，随机初始化。
- **输出头**：`action_out_proj` → 预测向量场 `v_t`，shape `[B, 50, action_dim]`。

##### 1.2 双专家在 Transformer 内的交互方式

PI0在**同一个Transformer内**用两组独立权重（类似MoE），每一层做的事：

- **Q / K / V投影 + FFN各自独立**（VLM用PaliGemma权重，action expert用随机初始化的小权重）。
- 两边的token **拼成一条序列做联合self-attention**——action expert的Q可以查到VLM prefix的K/V，把视觉语义"拉过来"。
- attention出来之后FFN各走各的。

**信息流向用attention mask控制**：

| 区块 | 可看见 |
|---|---|
| Prefix（image + language） | 仅Prefix（双向） |
| State token | Prefix + 自己 |
| Action tokens | Prefix + state + 全部action token（action内部双向） |

prefix看不到suffix——信息**单向流向** action expert，不污染VLM预训练分布。这也是prefix KV可以缓存的原因（10步Euler推理时prefix只算一次）。

PI0.5 / PI0.7沿用这个双专家骨架，但监督路径有变化：

- **PI0.5**：VLM同时通过FAST离散动作token的交叉熵被监督；action expert token不去看FAST token，避免两种动作表示泄漏。
- **PI0.7（知识隔离KI）**：action expert可以attention访问VLM全部激活，但**梯度不回传到VLM**。VLM只由FAST离散交叉熵监督，避免连续flow损失干扰视觉语言表征。

#### 2. Flow Matching vs Diffusion

基础查阅：[Flow Matching](Note_Basics.md#basic-flow)。

两者都是"从噪声生成数据"的连续生成模型，本质都在学一条把高斯分布变换到数据分布的路径，区别在路径设计：

| 维度 | Diffusion (DDPM) | Flow Matching |
|---|---|---|
| 前向过程 | 反复加噪：`x_t = √α_t · x_0 + √(1-α_t) · ε` | 直线插值：`x_τ = τ·noise + (1-τ)·action` |
| 学习目标 | 预测噪声 `ε` 或score `∇log p_t` | 预测向量场 `v_τ = noise - action`（直线方向上的速度） |
| 数学框架 | SDE / 马尔可夫链 | ODE / 连续归一化流 |
| 时间步采样 | 一般均匀 | PI0用 `Beta(1.5, 1.0)` 偏向小 τ |
| 推理 | DDIM/DPM-Solver，20-50步 | Euler ODE，PI0只10步 |
| 训练稳定性 | β/α 调度敏感 | 直线路径更稳，loss更平 |

##### PI0 的具体实现

构造样本（fp32算）：

- `noise ~ N(0, I)`，`τ ~ Beta(1.5, 1.0)` 缩放到 `[0.001, 1.0]`。
- `x_τ = τ · noise + (1 - τ) · actions`。
- 目标向量场：`u_τ = noise - actions`。

Loss：`L = E[||v_θ(x_τ, τ, condition) - u_τ||²]`。

直觉：`τ → 0` 时 `x_τ ≈ actions`（接近目标），`τ → 1` 时 `x_τ ≈ noise`（接近纯噪声）。`Beta(1.5, 1.0)` 偏向小 τ 是因为接近目标那段决定最终精度，需要重点训。

##### PI0 选 Flow Matching 的理由

- 训练目标更简单（不用复杂噪声调度）。
- 推理快——10步Euler就收敛，diffusion一般要20-50步。
- 与机器人50Hz控频匹配（4090上推理 ~73ms）。

#### 3. 数据流：从观测到 loss

##### 3.1 输入

> 以下以PI0论文默认的**双臂场景**（如Franka双臂 / Aloha-AgileX）为例，state/action都是 **16维**；不同embodiment维度不同，但openpi实现里都会被pad到 `max_state_dim = max_action_dim = 32` 统一进Transformer（这也是3.5节输出为 `[B, 50, 32]` 的原因）。

- `observation`：
  - **图像** `[B, V, 3, 224, 224]`：V个视角的RGB（典型双臂配置V=3：head + 左wrist + 右wrist），每张224×224给SigLIP；
  - **state** `[B, 16]`：当前本体感觉（proprioception），双臂 = 左臂7 joint pos（关节角度，单位rad）+ 1 gripper（开合0/1或归一化连续值）+ 右臂7 joint pos + 1 gripper；
  - **language tokens** `[B, L]`：任务指令（如 "pick up the red cup"），经Gemma tokenizer编码。

- `actions` `[B, 50, 16]`：未来H=50步的 **action chunk**，每一步同样16维（双臂joint + gripper）。action一般用**绝对joint目标**或 **delta joint**（取决于具体配置）；gripper维度一般是0/1开合或归一化连续开度。

> 关于16维到32维的padding：进入 `action_in_proj` 前，会把16维零填充到32维（`max_action_dim`），这样同一个PI0模型可以无缝吃不同embodiment的数据（单臂7维、Franka双臂16维、Aloha-AgileX 14维等），只在最后做action时按真实维度截断。state也是同样的pad处理。

##### 3.2 Prefix Embedding（图像 + 语言）

- SigLIP：Conv2d patch embedding（fp32）→ 位置编码（fp32）→ cast bf16 → 12层Transformer → 输出 `[B, 256, dim]` bf16。
- 语言：Gemma-2B `embed_tokens`（bf16）→ 乘 `sqrt(dim)` 缩放。
- 拼接后attention mask全0 = 双向，图像和语言互相可见。

`patch_embedding` 和 `position_embedding` 故意保留fp32：图像信息进入模型的第一个瓶颈，精度损失会传播到所有后续层。

##### 3.3 Suffix Embedding（状态 + 动作 + 时间）

- 状态：`state_proj = Linear(16 → width)`，fp32。
- 时间：正弦位置编码把标量 `τ ∈ [0, 1]` 编码为高维向量。
- 动作：`action_in_proj(x_τ pad 到 32 维)` → 与 `time_emb` 拼接 → MLP（`Linear → SiLU → Linear`）融合。
- Suffix内部attention mask为causal：state和action各token只能看到自身及之前的。

##### 3.4 联合 Transformer 前向

- Attention mask结构见1.2。
- 双专家：每层prefix/suffix各算自己的Q/K/V后拼起来做联合attention，FFN各自独立。
- 数值稳定关键算子强制fp32：**Softmax、RoPE三角函数、RMSNorm方差**。bf16下这几个算子会累积明显数值漂移。

##### 3.5 输出与 Loss

```
suffix_out → cast fp32 → action_out_proj → v_t [B, 50, 32]
loss = MSE(u_τ, v_t)        # fp32 下算，避免 bf16 平方溢出
loss.backward()
```

`v_t` 应逼近 `u_τ = noise - actions`。

#### 4. 为什么能输出 Action Chunk

##### 4.1 chunk 是什么

PI0一次预测的不是单步动作，而是未来H=50步序列 `A_t = [a_t, ..., a_{t+49}]`，叫 **action chunk**。控制时取前25步执行，然后下一次推理。

##### 4.2 架构为什么能支持

- Suffix里直接放H个action token，每个 `action_in_proj` 投影一个时间步的noisy action。
- Action内部双向attention，可以建模50步之间的时序依赖。
- Flow matching的向量场输出shape天然是 `[B, H, action_dim]`，**一次性预测整个chunk的速度场**，没有自回归的串行依赖。

对比OpenVLA：走自回归离散token，每个时间步都要解一个token，输出50步动作需要50次串行decode，无法满足50Hz控频。PI0论文里OpenVLA在灵巧任务上"几乎完全失败"就是这个原因。

##### 4.3 推理：10 步 Euler ODE

从纯噪声 `x ~ N(0, I)`（`τ=1`）出发，分10步沿向量场走回 `τ=0`：

```
for step in range(10):
    x = x - dt * v_θ(x, τ, condition)   # dt = 0.1
    τ -= dt
```

之所以10步够：

- Flow matching直线路径收敛快。
- Prefix KV可缓存，只算一次；10步只重复算suffix的attention/FFN。
- 4090上总耗时 ~73ms（图像编码14ms + 观测forward 32ms + 10步去噪27ms）。

#### 5. State 表示的两种正交选择

PI系列里state字段有两个完全独立的维度可以调，**不要把它们搞混**：

##### 5.1 编码方式：`discrete_state_input`

控制 **state怎么进模型**：

- `discrete_state_input=True`（PI0.5默认）：state走**文本token路径**。tokenizer把state离散化成整数序列拼到prompt文本，例如 `Task: xxx, State: 12 87 200 ...; Action:`，跟task一起走VLM bidirectional attention。
- `discrete_state_input=False`（PI0默认）：state走**连续向量路径**。经过 `state_proj` 线性投影变成连续token拼到action expert的suffix里。

实际工程里 `pi05_pico` 当前是 `pi05=True` + `discrete_state_input=False`，这是非默认组合——骨架是PI0.5，但state走PI0风格连续投影。

##### 5.2 内容语义：state 字段里装什么

控制 **state字段里放什么数**，与编码方式无关：

- 当前proprioception（标准）：state[t] = 当前关节角度。
- `action[t-1]`（变体）：state[t] = 上一帧的动作指令。

##### 5.3 两个维度可任意组合

| state内容 | state编码 | 说明 |
|---|---|---|
| 当前proprioception | 离散文本token | PI0.5标准组合 |
| 当前proprioception | 连续投影 | PI0 / 当前pi05_pico |
| `action[t-1]` | 离散文本token | 变体A |
| `action[t-1]` | 连续投影 | 变体B |

要把state改成 `action[t-1]`，**正确做法是改数据侧**，不是动 `discrete_state_input`：

- 在数据transform链路加一步 `RepackTransform` 或自定义transform，在LeRobot dataset层面或 `LeRobotPicoEgoDataConfig` 的 `repack_transforms` 里替换 `state` 字段。
- 首帧没有 `t-1`，用零向量或第0帧action自身填充。
- **`norm_stats` 必须同步替换**：原来state和action各有mean/std；改成action分布后必须用action的norm stats给替换后的state归一化，否则模型看到的输入分布偏移。

一句话：`discrete_state_input` 改的是"state张量怎么进模型"，"上一帧action当state"改的是"state张量里放什么数"，两者正交不互相替代。

#### 6. PI0-FAST：动作离散化 Tokenizer

PI0用flow matching输出连续动作；PI0-FAST走另一条路——把动作变成离散token，让自回归VLM直接预测。

##### 6.1 为什么不能直接对动作做 BPE

机器人动作在时域上**高度相关**（手臂位置变化平滑），直接BPE会得到极长的低熵token序列，浪费上下文。

##### 6.2 FAST 的两步压缩

1. **DCT频域转换**：对一段动作做**离散余弦变换（Discrete Cosine Transform）**，把时序信号从时域转到频域。机器人动作的高频分量很小，可以直接丢掉只保留低频系数。
2. **BPE字节对编码**：对剩下的低频系数做BPE，得到有限的"动作词表"——类似LLM词表，每个token代表一段动作的某种"模式"。

##### 6.3 在 PI0.5 / PI0.7 中的作用

PI0.5同时训练两种动作预测路径，联合loss：

`L = H(FAST_tokens) + α · ||v_θ - u_τ||²`

- 预训练阶段 `α=0`，只用FAST离散监督，训练效率高、适合大规模异构数据。
- 后训练阶段 `α=10`，启用action expert + flow matching，精度高、推理快。

PI0.7把这种"FAST监督VLM + flow matching监督action expert"做成永久的双轨结构（即KI），并且VLM梯度不被action expert污染。

#### 7. 训练工程细节

##### 7.1 梯度裁剪 + AdamW

- `clip_grad_norm_`：全局L2范数超过 `max_norm=1.0` 时等比例缩小。
- AdamW：`β1=0.9, β2=0.95, ε=1e-8, wd=1e-10`。

精度影响：bf16参数下m/v只有2-3位有效数字，`lr=2.5e-5` 时微小更新会被吞掉（`1.0 + 2.5e-7 = 1.0`）；fp32下更新正确保留（`1.0 + 2.5e-7 = 1.00000025`）。所以**优化器状态必须fp32**，参数可以bf16 + master copy fp32。

##### 7.2 学习率调度

前1000步线性warmup到 `peak_lr=2.5e-5`，之后cosine decay到 `end_lr=2.5e-6`。

##### 7.3 JAX 显存：XLA 环境变量

```bash
# export XLA_PYTHON_CLIENT_ALLOCATOR=platform   # 不要再设这个
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
```

`platform` 模式是"用多少分配多少"会有波动，跟 `MEM_FRACTION=0.9` 的预分配冲突，导致0.9不生效。统一只设 `MEM_FRACTION=0.9` 预先分配，行为更稳定。

#### 8. 完整数据流图

```
数据加载 (fp32)
  → 采样 noise + time (fp32)
  → 构造 x_τ 和目标 u_τ (fp32)
  → SigLIP 编码图像 (fp32→bf16)
  → Gemma 词嵌入 (bf16)
  → state/action/time 编码 (fp32)
  → 统一 cast bf16 进 Transformer
  → 双专家 Transformer (bf16, Softmax/RoPE/RMSNorm 用 fp32)
  → 输出 cast fp32 → action_out_proj → v_t [B, 50, 32]
  → MSE loss (fp32) → backward → clip_grad → AdamW.step
  → 定期保存 checkpoint
```

#### 9. PI0.5 相对 PI0 的改进

| 维度 | PI0 | PI0.5 |
|---|---|---|
| 训练数据 | 10K小时同质遥操作 + OXE | 异构联合训练：MM + ME + CE + HL（高层标注）+ WD（网络数据）+ VI（口头指令），97.6% 数据不来自目标平台 |
| 训练阶段 | 单阶段：预训练 → 后训练 | 两阶段：FAST离散token自回归预训练 → 加入action expert + flow matching后训练 |
| 推理范式 | 一次给出action chunk | 层次化：同一模型先输出子任务文本（"拿起盘子"），再基于子任务输出动作 |
| State默认编码 | 连续投影（`discrete_state_input=False`） | 离散文本token（`discrete_state_input=True`） |
| 泛化能力 | 任务级泛化（同环境） | 开放世界泛化：在**全新真实家庭**完成10-15分钟清洁任务 |

核心take-away：PI0.5把VLA重新定义成一个**同时能输出文本（子任务、FAST动作token）和连续动作（flow matching）的统一模型**，靠异构数据联合训练 + 层次化推理实现开放世界泛化。

#### 10. PI0.7 相对前作的不同

关键词是 **Steerable**——同一个通用模型可以通过prompt精确控制"怎么做"。

##### 10.1 架构变化

- VLM骨干升级到Gemma-3 4B + 400M视觉编码器。
- 新增 **MEM视频历史编码器**：最多4个摄像头 × 6帧历史时空压缩成固定数量token。
- Action expert扩到860M（PI0/PI0.5是300M）。
- 总参数约5B。

##### 10.2 核心创新：多模态 Prompt

除语言指令外，prompt还可包含：

- **子任务指令** `ℓ̂_t`：当前要做的语义子任务文本。
- **子目标图像** `g_t`：BAGEL 14B世界模型生成的近未来期望状态图像，专门解决"语言描述不清楚视觉细节"的问题。
- **Episode metadata**：速度（离散化步数）、质量（1-5分）、错误标签、控制模式。**这是steerable的核心抓手**——训练时给真实标签，推理时设为"最高质量/最快速度/无错误"来引导模型输出最优行为。

训练时各组件随机dropout（subgoal 75%、metadata 15%、子任务30%），让模型推理时可以灵活使用任意子集。

##### 10.3 知识隔离（KI）

VLM只由FAST token的离散交叉熵监督，action expert可以attention访问VLM的全部激活，但**梯度不回传到VLM**。VLM训练更稳定，避免连续flow loss干扰视觉语言表征。

##### 10.4 涌现能力

- **跨构型零样本迁移**：BiPi → UR5e折T恤，80% 成功率，匹配顶级人类遥操作员。
- **组合泛化**：通过语言coaching完成训练中从未见过的任务（空气炸锅、压面壶等）。
- **混合质量数据的scaling**：去掉metadata时加更多数据反而性能下降；有metadata时持续提升——证明metadata条件化解锁了数据规模的scaling效应。

##### 10.5 PI0 → PI0.5 → PI0.7 内在逻辑

| 维度 | PI0 | PI0.5 | PI0.7 |
|---|---|---|---|
| 解决的核心问题 | 灵巧操作的高频动作生成 | 开放世界场景泛化 | 多策略steerable控制 |
| 数据策略 | 高质量同质遥操作 | 异构多源（含网络数据） | 混合质量 + metadata条件化 |
| Prompt | 语言指令 | 语言 + 自动生成子任务 | 语言 + 子任务 + subgoal图 + metadata |
| 关键设计 | Flow matching + 双专家 | FAST预训练 + 层次化推理 | Episode metadata + 知识隔离 |

### pico ego pipeline

完整处理流程见[数据处理：超维Pico](Note_DataPipeline.md#data-chaowei)，包括时间匹配、曲率gripper、质量分级和标签语义。

把PICO头显采集的第一人称视角原始数据（视频 + tracking + 片段标注）转换成 **LeRobot v2.1** 数据集，用于VLA模型（pi0.5）预训练。

#### 输入与输出

**输入**：每个采集会话目录包含

- `CameraRecord_*.mp4`：原始头显视频
- `trackingData_*.txt`：JSON Lines格式的手部追踪
- `camera_params*.json`：相机内参 + 畸变参数
- `*_segments_description.json`：人工标注的片段（skill、interacting_hand、target_object等）
- `quality_inspection.json`：质检报告（可选）

**输出**：标准LeRobot数据集（`data/` / `videos/` / `meta/`），`observation.state` 与 `action` 均为 **20D**（每只手 `xyz(3) + 6D rotation(6) + gripper(1) = 10D`，6D旋转用Zhou et al. 2019的"旋转矩阵前两列展平"）。

#### 流水线步骤

入口 `run_pipeline.py` 递归扫描顶层目录下所有会话，按以下步骤逐个处理：

| 步 | 内容 | 脚本 |
|---|---|---|
| **Q** | 质量过滤（硬过滤 + 软评分0~5）；硬过滤未通过直接跳过 | `quality_filter.py` |
| **0** | 修正tracking时间戳（补偿管线延迟140ms） | `00_correct_tracking_time.py` |
| **1** | 并行：① tracking TXT → HDF5（只保留左右手）；② 视频去畸变 | `01_trackingdata_to_hdf5.py` / `01_video_undistort.py` |
| **2** | 按标注切分episodes：视频段 + 动作H5，可选帧率转换（如25→30fps，`setpts=N/fps/TB -bf 0`） | `02_video_hdf5_segment.py` |
| **3** | H5 → Parquet：构造20D state/action（xyz+6D rot+gripper），同时生成 `tasks.jsonl` | `03_hdf5_to_parquet.py` |
| **4** | 并行：① 整理LeRobot `data/`；② 整理LeRobot `videos/`（仅复制重组，不重编码） | `04_lerobot_data_generate.py` / `04_lerobot_video_generate.py` |
| **5** | 生成LeRobot `meta/`（info.json、stats、episodes.jsonl、tasks.jsonl等） | `04_lerobot_meta_generate.py` |
| **6**（可选） | `--auto-merge` 合并所有单会话数据集到 `_merged/`，**视频用symlink** 节省空间 | `05_merge_lerobot_datasets.py` |

#### 质量过滤的两层设计

1. **硬过滤**：任一命中直接淘汰
   - 相机标定无效（去畸变会崩）
   - 视频 < 5s（切分后没意义）
   - 视频-tracking时长比偏离 `[0.9, 1.1]`（同步出问题）
   - 双手missing ratio都 > 90%
   - 语义噪声里有invalid段
   - 关节ROM违规 > 100帧

2. **软评分**：从视觉 / 动作 / 时序 / 内容四个维度打0~100分 → 映射到0~5的quality数值，写入 `task_prompt` 前缀：

   `quality: 5; skill: Hover; hand: both; target_object: shelf; type: human; <原始描述>`

   下游训练时可按quality筛选/加权（pi0.7那种multimodal prompting的思路）。

#### 关键设计点

- **20D action维度** 是为了对齐VLA输入；6D rotation而非欧拉/四元数，避免不连续性。
- **视频帧率统一在Step 2完成**，Step 4不再重编码，避免重复transcoding损失质量。
- **中间产物（`_middle/` 与 `_segments/`）默认结束清理**，`--debug` 保留方便排查。
- **OSS FUSE写视频问题**：所有ffmpeg / cv2.VideoWriter输出必须先写本地FS再 `cp`，已封装在 `staged_writer`（见上节）。
- **并行**：单会话内Step 1 / Step 4并行；多会话之间用 `--workers` 控制ThreadPoolExecutor。

### 基于 OpenPI 0.5 开发两套 Policy：Egocentric 与 UMI/遥操

为了让Pico第一视角数据和松灵双臂的UMI/遥操数据共用同一个 π0.5模型，开发了 `pi05_pico` 和 `pi05_kaiumi` 两套policy。**核心原则**：π0.5 backbone与base checkpoint完全共享，差异只放在policy的input/output transform层——让两种数据用同一份权重起步，方便阶段式迁移（pico pretrain → kaiumi midtrain → 遥操posttrain）。

#### 1. 两套 Policy 对照

| 维度 | `pi05_pico` | `pi05_kaiumi` |
|---|---|---|
| 数据来源 | Pico VR头显第一视角 | 松灵双臂UMI采集 / 遥操 |
| state/action维度 | **20D**：(3 xyz + 6 6D-rot + 1 gripper) × 2手 | **14D**：(6 joint + 1 gripper) × 2手 |
| 动作空间 | 末端位姿（手部tracking解出） | 关节角度 |
| 相机 | 仅 `cam_high`（单目第一视角） | `cam_high` + 双wrist（三相机全启用） |
| delta mask | `(3, -7, 3, -7)`：xyz delta，rot/gripper absolute | `(6, -1, 6, -1)`：joint delta，gripper absolute |
| Inputs/Outputs | `PicoEgoInputs/Outputs`（新写） | `AlohaInputs/Outputs`（直接复用aloha_policy） |
| `adapt_to_pi` | — | `True` |

#### 2. 共同的模型骨架

```python
Pi0Config(
    pi05=True,                          # PI0.5 骨架（FAST 离散监督 + adaRMSNorm）
    action_horizon=ACTION_HORIZON,
    paligemma_variant="gemma_2b",       # USE_LORA=True 时切 gemma_2b_lora
    action_expert_variant="gemma_300m",
    discrete_state_input=False,         # state 走 PI0 风格连续投影
)
# 都用 PI05_BASE_CHECKPOINT_PATH 起步 + CosineDecaySchedule + ema_decay=None
```

`discrete_state_input=False` 是**非PI0.5默认**的组合——骨架是PI0.5，但state走PI0的连续 `state_proj`。原因是Pico的末端位姿和UMI的关节角度都是连续物理量，离散tokenize反而失真。

#### 3. Pico Ego 的两个关键设计点

**(1) 单目兼容三相机base ckpt**（最有意思的工程取舍）：base是按3相机训的，但Pico只有第一视角。直接改模型结构会破坏base权重，所以走mask路线——把两路wrist用零图填充，对应 `image_mask` 设为 `False`，attention自动忽略这两路。base ckpt完全不动就能吃单目数据。

**(2) 旋转选6D + absolute，xyz选delta**：xyz是欧氏空间的平移量，delta物理上就是位移，最好学；旋转用6D（连续可微无双覆盖）+ absolute（绕开SO(3) 上delta怎么定义的坑）。这套维度选择是踩了"四元数 + 全delta"的坑之后定下来的，详见下方"遇到的问题 #2"。

#### 4. KaiUmi 的设计：直接复用 Aloha 接口

松灵双臂的形态（6-DoF + gripper × 2 = 14D）和Aloha完全一致，所以 `kaiumi_policy.py` 直接派生自 `aloha_policy.py`，保留 `adapt_to_pi=True`：

- `_joint_flip_mask`：把Aloha joint约定翻成 π0内部约定（部分joint符号反转）；
- `_gripper_to/from_angular`：Aloha gripper是线性归一化（米），π0是角度归一化（弧度），双向换算来自Interbotix datasheet。

inputs做正向（数据集 → 模型），outputs做逆向（模型 → 真机）。三相机和base ckpt完全对齐，无需任何trick。

#### 5. 关键设计要点速查

| 设计点 | 选择 | 一句话原因 |
|---|---|---|
| 模型架构 | 完全共享 π0.5 backbone | base ckpt复用 + 阶段式迁移 |
| state编码 | 连续投影（非默认） | 连续物理量，离散化反而失真 |
| Pico单目 | 零图 + image_mask=False | 不改模型结构兼容三相机ckpt |
| 旋转表示 | 6D rotation + absolute | 连续可微 + 绕开SO(3) delta（详见踩坑 #2） |
| LoRA / 全参 | `USE_LORA` 环境变量切换 | LoRA节省显存适合小数据 |
| EMA | `ema_decay=None` | 微调阶段EMA收益有限 |

### 遇到的问题以及一些细节

#### 1. OSS FUSE 写 MP4 时 moov atom 丢失

- 问题现象：用ffmpeg或cv2.VideoWriter把mp4直接写到OSS FUSE挂载路径（`/mnt/pico_data`）时，文件能写出但ffprobe报moov缺失、播放器打不开，典型报错 `Error writing trailer: Invalid argument`。

- 原因：MP4文件由两个核心atom组成——`mdat` 存编码数据（编码过程中顺序追加），`moov` 存每帧偏移和编解码参数等索引（必须等所有帧编完才能算出）。主流muxer的标准流程是先占位写 `mdat`，结束时构造 `moov` 写到文件尾，再seek回头把 `moov` 搬到文件首部做faststart重排，方便边下边播。而OSS对象存储本身不可变（PutObject是原子全量写），ossfs2用multipart upload模拟追加，只支持顺序append和顺序read，不支持回头改写已写过的偏移。所以一旦muxer在close时做faststart重排，那一刻就必然失败。

- 解决方案：用staged writer模式，让seek发生在本地FS上，写完整后再一次性顺序传到OSS。即encoder先把完整mp4写到本地高速FS（tmpfs或CPFS），再用 `cp` 顺序复制到OSS FUSE。`cp` 对OSS FUSE来说就是把整个文件作为一次multipart upload写出，全程顺序无seek，因此能成功。项目里统一封装在 `python/staged_writer.staged_oss_output`（context manager），face_blur、VideoSplitRefiner等所有写mp4的refiner都套这个wrapper。

#### 2. Pico action 表示踩坑：从「四元数 + 全 delta」到「6D rotation + xyz delta + 旋转 absolute」

基础查阅：[旋转表示与相对位姿](Note_Basics.md#basic-rotation)。

- 问题现象：Pico ego policy最初版本用 `xyz (3) + 四元数 (4) + gripper (1) = 8D` 作为单手动作表示，并且 **xyz、四元数都做delta**。训练时旋转维度loss长期不下降、推理时手部姿态明显抖动甚至跳变，xyz维度反而正常。

- 原因（两个独立但叠加的问题）：

  - **四元数双覆盖（±q表示同一旋转）**：单位四元数 `q` 和 `-q` 几何上代表同一个旋转，但欧氏数值上差了一倍模长。Pico头显的手部tracking在相邻帧偶尔会输出符号翻转的q（解算时挑了相反的半球），如果直接 `Δq = q_t - q_{t-1}`，正常情况下是接近0的小向量，符号翻转那帧就会突然变成一个 |Δq| ≈ 2的"伪大旋转"。训练数据里混了这种**完全虚假的大目标**，模型既学不到真规律，也压不住梯度。
  - **四元数分量差不等于旋转增量**：直接相减无法保证合法旋转，也受符号翻转影响。几何增量可用旋转矩阵组合或四元数乘法表示，需要三维误差向量时再取log map。本项目选择absolute rotation-6D，简化目标和恢复过程。
  - 附加问题：相比xyz这种本来就在欧氏空间的物理量，旋转delta对模型来说还要额外学一个非线性流形上的减法操作，难度更高。

- 解决方案：把单手动作从8D改成 **10D = `xyz (3) + 6D rotation (6) + gripper (1)`**，双手合20D，并调整delta mask为 `make_bool_mask(3, -7, 3, -7)`，即**只xyz做delta，6D rot和gripper全部absolute**：

  - **旋转表示换成6D rotation**（Zhou et al. 2019，旋转矩阵前两列展平）：连续可微、没有双覆盖、用Gram-Schmidt就能反解出合法的旋转矩阵，对回归非常友好。
  - **旋转改成absolute（不做delta）**：直接预测下一时刻的目标旋转矩阵，彻底绕开SO(3) 上delta怎么定义这个坑。代价是模型每帧都要从头预测姿态，但实测6D表示足够稳定，没有性能下降。
  - **xyz仍然delta**：xyz是平移量，本来就在欧氏空间，delta物理上就是位移（≈ 速度 × dt），对模型最友好。
  - **gripper仍absolute**：开/合是绝对状态，delta没有物理意义。

- 教训：VLA/IL的旋转表示：
  - 本项目使用 **6D rotation** 回归姿态；其他表示也可使用，但需处理欧拉角奇异性、四元数符号一致性或旋转矩阵约束；
  - 需要旋转增量时，明确参考坐标系和组合顺序；不要把表示分量的直接相减当作几何旋转差；
  - delta / absolute应按语义组选择，旋转组不能任意拆开；本项目采用位移delta、旋转absolute、夹爪absolute。

### 优化方向

两条根据近期论文产生、可在Pico ego + Kaiumi双policy上落地的思路。

#### 1. 时间监督不平衡：让 ego 数据重监督关键帧

- **问题**：当前pico ego pipeline把所有帧均匀送进 `pi05_pico` 训练。但ego视频大段是「悬停 / 接近 / 稳定搬运」等低信息帧，真正决定任务的「对齐 / 接触 / 抓取 / 释放」瞬间占比很小——等于把算力大头喂给了低价值帧。

- **参考论文**：FrameSkip（arXiv:2605.13757，2026）。dataloader层用AVI（动作变化）+ VAC（视觉-动作错位）+ TPI（任务进度先验）+ gripper过渡4个轻量信号给每帧打分，按retention ratio剪枝；**不动模型架构、不动loss、不动推理**。r=20% 时三benchmark平均66.5% → 76.15%。

- **落地路径**：
  - 在pipeline Step 3（H5 → Parquet）后做一次离线打分，先只用最便宜的AVI + gripper-aware（VAC和GMM-TPI第一版可省），把分数写成parquet的 `importance` 列。
  - 在LeRobot dataloader加一层index remapping：按当前retention ratio用二分查找把请求timestep映射到最近的保留timestep；openpi训练侧（`pi05_pico` / `pi05_kaiumi`）一行不改。
  - 配合现有 `quality: 0~5` 软分形成「trajectory级 × frame级」双层数据分配——`quality` 决定整条轨迹的采样权重，`importance` 决定轨迹内的帧采样。

#### 2. 三阶段微调的先验丢失：用先验保留式适配替代 full fine-tune

- **问题**：当前 `base ckpt → Pretrain(Ego) → Midtrain(UMI) → Posttrain(遥操)` 是串行full fine-tune，每一阶段都会把上一阶段（含 π0.5 base）学到的广泛motor / scene先验**覆写成当前阶段的窄分布**。Posttrain后policy在新光照 / 桌高 / 物体位置等OOD下鲁棒性不足，few-shot真机数据时尤其明显。注意KI（PI0.7）思路只冻VLM，**action expert的motor prior仍会被改写**，不能直接解决这个问题。

- **参考论文**：PriorVLA（arXiv:2605.10925，2026）。把预训练VLA看成两类只读先验源（VLM = scene prior，action expert = motor prior）。
  - **Dual Action Experts**：预训练AE复制成frozen Prior Expert + trainable Adaptation Expert，只有AE输出进loss和轨迹更新，PE仅作为motor只读源。
  - **Expert Queries**：Scene / Motor / Action 3组可学习token + attention mask，让AE单向读取两类先验（MQ不许看VLM prefix，避免被scene特征淹没）。
  - 25% 可训参数全面赢过full fine-tune；真机few-shot OOD 10% → 32%（3.2×）。

- **落地路径**：
  - 优先在最敏感的Posttrain(遥操) 阶段替换：冻住Midtrain出来的AE当Prior Expert，复制一份作Adaptation Expert训练；VLM同时冻。
  - 3组Expert Queries接入openpi双专家attention mask（兼容现有prefix/suffix mask结构，新增3段可学习token + 对应的单向mask即可）。
  - 先在「Midtrain ckpt + 10–50 demo遥操数据」的few-shot设定下对比full fine-tune vs PriorVLA-style适配的OOD成功率，作为最便宜的可行性验证。
  - 推理多一次PE forward是已知成本，先用chunked control摊薄，后续再考虑PE蒸馏。

<a id="project-magicatom"></a>

## 魔法原子工作总结-VLA算法工程师-magicvla预训练方向

### 项目总框架

**版本范围：** 本节将 **32D base版本**与**后续34D训练版本**分别记录。第1节说明32D数据落盘契约，第2节说明原32D base；第3节介绍Dynamic/SGM，并单列34D配方。两版的动作索引、模型输入输出、mask和归一化统计各自配套，不相互替换。Dynamic/SGM是模型结构扩展，本身也支持32D，不能与34D画等号。

```text
数据处理
  ├─ EgoDex：virtual-hand EEF、坐标对齐与质量控制
  ├─ Hy-UMI：相机参数估计与 ARX5 retarget
  ├─ 仿真数据
  └─ 真机数据
        ↓ 统一数据格式 / action 与 state 表示 / 质量控制

MagicVLA-base pretrain
  ├─ VLM backbone
  ├─ action expert
  ├─ 多模态输入与 action chunk 输出
  ├─ flow matching / 其他监督
  └─ 多源数据路由与训练框架
        ↓ base checkpoint

MagicVLA Dynamic / SGM
  ├─ Action + 2D World + 3D World experts
  ├─ DINOv3 / Track4World 冻结 teacher 监督
  └─ 多分支 attention、梯度隔离与动作部署

后训练：memory 能力探索
  ├─ memory 方向
  ├─ RoboMME 方法整理与接入
  ├─ DM05 方法整理与接入
  └─ 长时任务、遮挡、历史信息和 OOD 能力评估
```

数据处理统一异构输入，base学习动作先验；Dynamic加入视觉、几何和运动表征监督，memory利用历史观测缓解当前帧歧义。能力收益需分别通过实验验证。

<a id="magic-data"></a>

### 1. 数据处理与预训练数据

框架与生产细节见[数据处理：魔法原子](Note_DataPipeline.md#data-magicatom)，两条pipeline的区别见[对比表](Note_DataPipeline.md#data-comparison)。

**一句话回答：** 32D数据版本把第一视角人手、Hy-UMI、真机和仿真数据接入LeRobot v2.1、统一相机key和逐维mask契约；我主要负责Hy-UMI的 `cam_high` 无标定相机参数估计，以及将UMI双手EEF轨迹投影为ARX5双臂joint标签。

#### 1.1 预训练数据构成

本节对应的32D base配方把8个机器人数据源按样本量混合为一个robotics source，再以robotics:EO VLM-SFT = 9:1按batch交错。机器人数据使用行为克隆/flow matching，EO提供视觉语言监督。

| 类别 | 数据源 | 机器人/相机特点 | 作用 |
|---|---|---|---|
| Ego | EgoDex | 人手第一视角，仅 `cam_high` | 学习第一视角操作和手部运动先验 |
| UMI | Hy-Embodied UMI table_000/001 | 双手第一视角，`cam_high` + 双wrist | 大规模人类示教，retarget为ARX5 |
| 真机 | RoboDojo real：ARX X5、PiPER、PiPER-X | 三相机、双臂 | 对齐真实机器人动力学和关节控制 |
| 仿真 | RoboTwin2.0、RoboDojo sim | 三相机、跨机器人/任务 | 扩展任务、场景和轨迹覆盖 |
| 真机 | Galaxea R1 Lite | head + 双wrist，读取时映射为统一相机key | 增加embodiment多样性 |
| VLM | EO Robo2VLM SFT | 图文/视频问答 | 保留和增强视觉语言能力 |

机器人数据在source内按物理样本量 `concat_shuffle`，不人为把小数据集重复到和大数据集一样多；每个source独立做normalization，不能将人手、ARX5和PiPER的统计量混用。

#### 1.2 32D 版本的数据契约

```text
state[t]：当前机器人状态
action[t:t+50]：未来 50 步动作 chunk
dim_mask：该维度是否真实存在并参与输入/loss
camera_valid：当前样本实际具备哪些相机
```

| 索引 | 维度 | 语义 |
|---|---:|---|
| `0:6` | 6 | 左臂joint |
| `6` | 1 | 左gripper |
| `7:13` | 6 | 右臂joint |
| `13` | 1 | 右gripper |
| `14:17` | 3 | 左EEF在 `cam_high` 坐标系的xyz |
| `17:23` | 6 | 左EEF rotation-6D |
| `23:26` | 3 | 右EEF在 `cam_high` 坐标系的xyz |
| `26:32` | 6 | 右EEF rotation-6D |

设计为32D的原因：

- **统一模型接口**：不同机器人、joint控制和EEF控制可共用同一个action expert、checkpoint和action tokenizer。
- **joint与EEF互补**：前14D是可直接执行的双臂控制量；后18D把动作放到图像观察坐标系，提供更强的视觉几何对应。
- **mask而非假零值**：Hy-UMI的兼容版本只有前14D、部分源没有EEF或缺少wrist图像，均右侧补零并关闭相应mask。mask同时进入state embedding和flow loss，避免把“未测量的0”误当成“中位姿态/真实动作”。
- **统一旋转语义**：EEF使用连续rotation-6D；训练中的 `chunk_delta` 对平移/joint构造相对量，gripper保持绝对状态，rotation group用合法旋转组合处理，避免直接相减四元数。

#### 1.3 总体 pipeline

`/pfs/user/pretrain_data_pipeline` 是数据处理仓库。每种source只实现reader，公共transforms负责质量检查、坐标处理、retarget和写盘：

```text
Lance / HDF5 / 原始 LeRobot
  → source reader：episode、图像、原始 state/action
  → quality check / 异常修复
  → 坐标系变换、retarget、next-step action
  → 32D pack + state_mask/action_mask
  → LeRobot v2.1：Parquet + 实际相机视频 + meta / quality / mask
  → 按训练版本生成 normalization statistics
```

训练reader再统一相机key为 `cam_high / cam_left_wrist / cam_right_wrist`；缺失相机使用 `camera_valid` 屏蔽，而不把零图像作为真实观测。

#### 1.4 EgoDex：人手轨迹到 virtual-hand EEF

基础查阅：[坐标变换](Note_Basics.md#basic-transforms)、[旋转表示](Note_Basics.md#basic-rotation)。

**目标：** 从单路头部视频、相机位姿/内参和人手关键点生成EEF state/action，接入32D数据契约。这里保留人手运动监督，不做机械臂IK；12个arm joint槽位补零且mask=0，开合量写入 `6/13`，双手EEF写入 `14:23/23:32`。

**人手表示。** 当前生产分支使用 `virtual_hand`：以wrist与指根均值的中点作为palm原点，以palm指向中指尖构造approach轴，再用小指尖方向正交化构造其余轴。拇指与食指尖距离单独决定开合量：`clip((d-0.01)/(0.12-0.01), 0, 1)`，其中1表示张开。位姿轴和夹爪标量各有明确的几何来源。

**坐标与动作生成顺序：**

```text
关键点 → virtual-hand pose + openness
       → 质量检查与尖峰/不连续修复
       → 世界坐标约定统一 + EEF 局部轴对齐
       → action[t] = EEF[t+1]，同步移动 action 可见性标签
       → state[t]、action[t] 均表达在当前相机 C_t 坐标系
       → rotation-6D + 32D pack + state/action mask
```

世界系变换同时作用于手和相机；局部轴对齐采用右乘 `Rx(90°)`，只改变姿态。相机系目标为 `T_Ct_E(t+1) = inverse(T_W_C(t)) @ T_W_E(t+1)`，避免头部运动造成参考系混用。数据侧保存绝对next-step target，训练reader再构造chunk delta和归一化目标。

**质量控制。** 检查图像可用性、EEF可见性、位置异常、速度/角速度突变、静止/冻结和场景运动。只对指定的尖峰与不连续轨迹修复，其余质量标签保留用于筛选和审计。两只手的EEF分别按可见性设置mask；action使用下一帧可见性。当前virtual-hand pack的两个开合量槽位没有绑定EEF可见性mask，不能概括为“整只手所有维度一起失效”。

**生产版本。** 普通配置可裁剪首尾低信息段，保持内部轨迹连续；`120core` 配置使用deferred media，不裁剪帧区间，先写Parquet、元数据和媒体清单，再单独物化视频。断点恢复与媒体完整性共同保证图像和标签对齐。两版均只导出实际存在的 `cam_high`。

**方案边界。** 仓库还保留 `egodex_hand_alignment` 等转换器，其原点和轴定义不同，不能复用virtual-hand的轴修正。后续34D配方通过reader映射此32D数据，并配置EgoDex动作2倍语义上采样；这不改变磁盘数据的32D定义。

代码入口：

- [人手 EEF 转换器](../pretrain_data_pipeline/src/pretrain_data_pipeline/data_io/eef_converter.py)：`compute_virtual_hand()`。
- [普通生产配置](../pretrain_data_pipeline/configs/run/production/egodex_virtual_hand_quality_production.yaml) / [deferred-media 配置](../pretrain_data_pipeline/configs/run/production/egodex_virtual_hand_quality_production_120core.yaml)。
- [ActionPostprocess](../pretrain_data_pipeline/src/pretrain_data_pipeline/transforms/action_postprocess/pipeline.py)：next-step、相机系、rotation-6D与pack。

#### 1.5 Hy-UMI 原始数据与清洗

Hy-UMI原始数据是Lance-backed LeRobot v3，`table_000` 和 `table_001` 各约1.16万episode、约1,079万帧，原始30 FPS。每帧包含三路 `424x240` RGB、16D双手跟踪状态和2D gripper command：

```text
raw state = [L_xyz(3), L_quat_xyzw(4), L_gripper(1),
             R_xyz(3), R_quat_xyzw(4), R_gripper(1)]
raw action = [L_gripper_command, R_gripper_command]
```

处理时将四元数从 `xyzw` 统一为 `wxyz`；夹爪把原始 `0 mm=open, 90 mm=closed` 转为 `1=open, 0=closed`。测得gripper state与下一步gripper command分开保存，不能相互替代。

质量控制先检查三路图像描述符、夹爪范围、EEF可见性、位置异常、速度/角速度突变和静止段；只对异常EEF轨迹插值修复，再进入IK。视频、EEF与action共用同一帧索引，正式生产配置保持30 FPS、原始 `424x240` 分辨率。

#### 1.6 本人工作一：`cam_high` 无标定相机参数估计

Hy-UMI没有官方 `cam_high` 内外参，且每帧可稳定利用的几何对应只有左右两个UMI设备。目标不是逐episode盲拟合，而是估计table级共享参数，并按session/batch做小范围refinement。

1. 从多帧灰度图取时间中值作为背景；以亮桌面区域为搜索范围。
2. 用 `max(|I-background|, background-I)` 同时保留运动和暗色证据，形态学去噪后取两个连通域；左右手尝试两种匹配，选总重投影误差更小的一种。
3. 以针孔模型优化15个变量：相机旋转/平移6D、共享焦距1D、主点2D、左右device offset各3D。offset解决3D跟踪原点和图像暗块质心并非同一点的问题。
4. 优化目标是所有有效对应的pixel residual的最小60% trimmed mean，降低遮挡、设备重叠和blob误检的影响；从经过验证的seed多次Nelder-Mead优化，而不是随机初始化。
5. 标定输出 `T_W_C`（`cam_high -> UMI world`）和K。训练EEF通过 $T_C^E=(T_W^C)^{-1}T_W^E$ 转到相机系；**device offset只用于标定，不写入EEF标签**。

table_000的全局标定为 `fx=fy=235.7 px`，重投影中位误差约41 px、held-out约37 px。由于2D blob是“手+设备”的质心而非动捕原点，存在约15-20 px的误差地板；验收以跨任务overlay为主，数值residual只作汇总。头戴相机跨session会变化，因此后续以table全局K/offset为先验，session主要refine rotation。

#### 1.7 本人工作二：UMI EEF 到 ARX5 joint 投影

基础查阅：[IK 与阻尼伪逆](Note_Basics.md#basic-ik)。

目标是把人手EEF示教变成可由ARX5执行的14D双臂标签，而不是把人手坐标直接当成robot joint。

```text
UMI EEF pose
  → 统一到 UMI world/task frame
  → 手部局部轴对齐 ARX5 TCP
  → 固定虚拟 ARX5 base
  → 双臂 DLS IK + joint limit + step limit + collision check
  → [L_joint1..6, L_gripper, R_joint1..6, R_gripper]
```

- UMI world和ARX5 task frame都采用 `+X forward, +Y left, +Z up`，所以world到task为identity；但UMI local hand axes与ARX5 TCP不同，必须右乘固定 `hand_to_ee` 置换矩阵，否则姿态标签错误。
- ARX5 base不是相机外参。通过代表性episode搜索一套 `task_from_root`，以IK失败、碰撞、位置/姿态residual为主目标，并用双臂左右对称和朝向先验打破近似解；同一session/batch固定base，避免跨episode的joint语义漂移。
- 每臂使用6-DoF URDF chain的阻尼最小二乘IK，限制单步joint变化 `0.12 rad`，并检查关节限位和双臂碰撞。table_000小批量搜索得到base约为 `[0.107, 0, -0.704] m`、yaw约 `-10 deg`；采样验证128/128 IK成功、0碰撞。
- joint state/action使用IK得到的结果；相机系EEF标签保留 **送入IK的目标EEF**，不再用FK回算覆盖，避免URDF TCP偏差和IK residual污染视觉几何监督。

#### 1.8 面试回答要点

- **EgoDex如何接入？** 从手部关键点构造virtual-hand EEF和开合量，生成当前相机系下的next-step target；无机器人joint的槽位关闭mask，再按训练版本完成映射和归一化。
- **你做了什么？** 负责Hy-UMI的无标定 `cam_high` 参数估计和UMI EEF到ARX5 joint retarget，使人类第一视角示教可进入统一32D预训练。
- **最大难点？** 两个3D点对应两个无标签图像blob，焦距、位姿和设备偏置高度耦合；因此用跨episode共享参数、trimmed residual、显式左右匹配和overlay验收，而非逐帧/逐episode全参数拟合。
- **为什么要32D + mask？** 既保留joint的可执行性和EEF的视觉对齐，又让不同embodiment共用模型接口；mask解决异构数据中“缺失维度”和“数值为零”不可区分的问题。

<a id="magic-base"></a>

### 2. MagicVLA-base pretrain 模型架构（32D 版本）

**一句话回答：** MagicVLA-base用Qwen3.5-2B承担视觉语言理解，用一个460M的连续action expert生成32D、50-step动作chunk；两者只在Qwen的full-attention层进行单向joint attention，因此既复用VLM先验，又避免把连续控制离散成语言token。

#### 2.1 设计思路

- **分工而非单塔硬做**：视觉、语言和任务理解已有强Qwen先验；动作是连续高频轨迹，直接预测velocity比量化为token更自然。于是VLM做条件前缀，action expert做flow matching。
- **兼容Qwen3.5的混合骨干**：Qwen3.5的24层按 `3 x Gated DeltaNet + 1 x full attention` 重复6次。线性注意力是递推结构，不能安全地把两种token直接拼接；所以仅在6个full-attention层融合，其余18层两支独立运行。
- **异构embodiment可共训**：数据层统一为32D，但不是强行假装每个机器人都有全部维度。state/action mask、相机有效位和source-specific normalization同时进入模型和loss。
- **保留通用视觉语言能力**：机器人flow loss不直接冲击VLM；通过knowledge insulation隔离梯度，VLM主要由FAST action CE和EO VLM-SFT的next-token CE更新。

#### 2.2 模型结构与信息流

基础查阅：[RMSNorm](Note_Basics.md#basic-rmsnorm)、[SwiGLU](Note_Basics.md#basic-swiglu)、[RoPE](Note_Basics.md#basic-rope)。

```text
cam_high + left/right wrist + task text + 32D state/mask
                         ↓
Qwen3.5-2B VLM prefix (24 layers, hidden 2048)
                         ↓ 仅 6 个 full-attention 层提供 K/V
32D noisy action[50] + flow time + state condition
                         ↓
Qwen3.5-style action expert (24 layers, hidden 1024, about 460M)
                         ↓
velocity[50, 32]  -- reverse flow --> future action chunk
```

| 模块 | 实现细节 | 作用 |
|---|---|---|
| 多模态prefix | Qwen原生图像编码；三相机letterbox到 `256x256`；文本为task/embodiment条件 | 形成场景与任务语义 |
| state条件 | `state(32) + state_dim_mask(32)` 经MLP投成1个连续prefix token；同时以additive condition加到每个action token | 避免把32个数字展开为约294个文本token；显式区分缺失维度与归一化后的零值 |
| action expert | 输入 `noisy_action[50,32]`，加action position、flow-time embedding和state condition；24层、width `1024`、SwiGLU `3072` | 直接建模连续chunk，不依赖动作词表 |
| Hybrid schedule | 18个Gated DeltaNet层分别更新VLM/action；6个full-attention层共享attention计算 | 以较低成本让动作读取视觉语言上下文，并保持与Qwen预训练层型一致 |
| output head | RMSNorm + linear，输出每个动作位置的32D velocity | 供flow matching训练和Euler反演 |

full-attention中序列固定为 `[VLM prefix, action suffix]`，mask是非对称的：

| query \ key | VLM prefix | action suffix |
|---|---|---|
| VLM prefix | causal + valid | 禁止 |
| action suffix | 全部有效prefix | chunk内双向 |

因此action可以使用视觉语言条件和整个未来chunk的协同信息；VLM不读取连续action suffix；训练时追加的FAST动作token用于CE，必须从AE可读取的prefix中排除，避免目标泄漏。两支hidden size虽为 `2048/1024`，但full-attention的head规格兼容，attention后再分别走各自的output projection。

##### 2.2.1 Qwen 与 Action Expert 关键参数对照

以下按 `magicvla/configs/train/magicvla_base_pretrain_robot_group_8data_eo_wandb_200k.yaml` 汇总。**Qwen指VLM的语言骨干，不包括视觉塔。** 本次环境未找到配置指定的Qwen checkpoint；Qwen hidden size依据仓库文档，层型与注意力规格依据 `MagicVLABasePolicy._validate_hybrid_layout()` 的强制匹配检查，未直接读取checkpoint核验。

| 整体参数 | Qwen语言骨干 | Action Expert |
|---|---|---|
| hidden size：每个token的主干宽度 | 2048 | 1024 |
| 层数 | 24 | 24 |
| 层型排列 | `[DeltaNet × 3 → Full Attention] × 6` | 相同 |
| DeltaNet / Full Attention层数 | 18 / 6 | 18 / 6 |
| SwiGLU中间维度 | 本次未核实 | 3072 |
| token数量 | 图像、文本和状态构成的prefix长度L | 50个动作时刻 |
| 主干特征形状 | `[B, L, 2048]` | `[B, 50, 1024]` |

**Full Attention规格：主干宽度不同，但投影后的头规格一致。**

| 参数 | Qwen | Action Expert |
|---|---|---|
| Q头数 | 8 | 8 |
| K/V头数 | 2 | 2 |
| 每头维度 | 256 | 256 |
| Q总宽度 | 2048 | 2048 |
| K、V各自总宽度 | 512 | 512 |
| 注意力输出投影 | `2048 → 2048` | `2048 → 1024` |

8个Q头共享2组K/V，即每4个Q头共享一组K/V（GQA）。AE的 `q_proj` 同时生成Q和gate，因此实际线性层输出为 `4096 = 2048 Q + 2048 gate`。

在第4、8、12、16、20、24层，Qwen当前层输入经归一化及其自身K/V投影生成条件；AE用自己的投影生成Q/K/V。两支K/V分别为 `[B, 2, L, 256]` 和 `[B, 2, 50, 256]`，沿序列维拼为 `[B, 2, L+50, 256]`，供AE的Q读取。AE在一次注意力计算中同时读取VLM条件与动作chunk；Qwen仍单独运行原生层，不读取动作suffix。

**DeltaNet规格：两支分别运行，不交换K/V。**

| 参数 | Qwen | Action Expert |
|---|---|---|
| Q/K头数 | 16 | 16 |
| V头数 | 16 | 16 |
| Q/K每头维度 | 128 | 128 |
| V每头维度 | 128 | 128 |
| Q、K、V各自总宽度 | 2048 | 2048 |
| 因果卷积核大小 | 4 | 4 |

DeltaNet的16头与Full Attention的8个Q头属于不同模块。层型和头规格对齐是当前实现的设计选择，不意味着两支共享全部参数，也不是所有VLA都必须如此设计。

**AE输入与输出：**

```text
带噪动作 [B, 50, 32] → Linear(32, 1024) → [B, 50, 1024]
                         + 位置 embedding [1, 50, 1024]
                         + state MLP 输出 [B, 1024]，广播到 50 个位置
                         + flow-time embedding [B, 1024]，广播到 50 个位置
                         ↓
                     24 层混合骨干
                         ↓
               RMSNorm → Linear(1024, 32)
                         ↓
                  velocity [B, 50, 32]
```

AE的state MLP为 `32 → 1024 → 1024`；VLM prefix的状态投影是另一套MLP，输入包含state与mask。**32是动作空间维度，1024是AE主干宽度，256是Full Attention每头维度。** flow time表示噪声阶段，与chunk内的动作位置不同；输出velocity是flow空间变化率，不等同于关节物理速度。

代码入口（均位于 `magicvla/src/models/magicvla_base/modeling_magicvla_base.py`）：`QwenHybridActionExpert` 组装专家，`embed_inputs()` 注入输入条件，`QwenJointFullAttention` 处理K/V交互，`_run_joint_trunks()` 按层调度两支，`_validate_hybrid_layout()` 校验规格兼容性。

#### 2.3 32D 训练与推理契约

基础查阅：[Flow Matching](Note_Basics.md#basic-flow)、[Masked MSE](Note_Basics.md#basic-masked-mse)、[归一化](Note_Basics.md#basic-normalization)。

训练机器人batch的目标是归一化后的 `action[50,32]`。采样 $t\sim\mathrm{Beta}(1.5,1.0)$ 并截断到 $[0.001,0.999]$，构造 $x_t=(1-t)a+t\epsilon$，模型预测 velocity $\epsilon-a$。loss只在 `~action_is_pad & action_dim_mask` 的元素上计算，可选提高前几个可执行horizon的权重。

- **chunk delta**：joint/平移使用相对当前state的delta；gripper保持绝对命令；两组EEF rotation-6D用 $R_{target}R_{state}^{T}$ 组合，而非逐元素相减。
- **共同训练**：8个机器人source用flow matching，EO VLM-SFT用Qwen CE；当前主配方为robotics:EO=`9:1`。FAST是辅助的动作token CE，不参与部署时的动作生成。
- **Knowledge insulation**：action expert读取detached VLM prefix，flow gradient不更新VLM；关闭KI时可做完全端到端共同优化。当前8-source配方开启KI，VLM通过FAST/EO loss更新。
- **推理**：从masked Gaussian noise开始，默认10次Euler reverse-flow。VLM prefix与每个full-attention层的K/V对噪声步骤无关，先计算一次并缓存；每一步都重新施加action mask，保证训练与推理都不会在不存在的embodiment维度上产生噪声或速度。
- **部署闭环**：按source的quantile stats反归一化，再按delta/rotation规则还原到action target。故checkpoint必须携带normalization metadata，只有权重不能正确执行动作。

#### 2.4 实际排障记录

| 问题 | 根因 | 修复与防回归 |
|---|---|---|
| full fine-tune第一次真实forward直接报 `AttributeError` | joint trunk从decoder layer读取 `block_type`；Transformers 5.5.4改名为 `layer_type`，而旧fake test恰好复制了错误假设 | 改从checkpoint的 `text_config.layer_types` 读取层调度，并在初始化校验action expert与VLM的24层schedule一致；测试模拟真实layer缺少该属性 |
| 缺失维度被当作“中位姿态” | quantile normalization后 `0` 是范围中点。旧逻辑把masked state清零后丢弃mask，RoboTwin2右EEF仅约41.3% 帧有效 | 将32D state mask与state一起输入MLP；action mask同时控制noise、flow loss和每一轮推理更新 |
| EEF rotation的delta语义错误但loss不报错 | 旧实现直接相减rotation-6D，结果不在SO(3)，且同一手腕运动会随参考坐标变化 | 改为 $R_{rel}=R_{target}R_{state}^{T}$ 后再转rotation-6D；把rotation规则写入norm-stats signature，拒绝复用旧统计量 |
| 多卡训练有效随机性不足，resume后更新几乎停滞 | 所有rank用同一全局RNG，flow time/noise完全相同；同时optimizer load把bf16参数对应的fp32 master/moments强转回bf16 | rank-aware seed保留数据source同步而区分模型随机数；resume后显式恢复fp32 optimizer state，并在关闭autocast的fp32 delta-rule路径测试 |
| FAST辅助目标和部署动作不一致 | episode尾部padding被置零后仍送入FAST tokenizer；归一化零值并非“静止”，生成了伪造的回中位动作token | tokenizer按sample截断/前向填充invalid tail；flow与FAST共享pad语义。checkpoint保存并强制校验per-source normalization/delta metadata |

#### 2.5 面试回答要点

以下按当前32D、50-step MagicVLA-base配置回答；`B` 为batch size，`L` 为多模态条件prefix长度。

**1. VLA的视觉编码部分怎么做？**

三路相机图像先等比例缩放并补边到 `256×256`，再由Qwen原生processor和视觉编码器处理：切成patch，经过视觉骨干提取特征，再通过合并、投影形成宽度为 `2048` 的视觉token，插入语言序列的图像占位位置。切patch只是第一步，视觉token数要以processor产生的图像网格为准。

**2. 视觉token的位置信息怎么嵌入？**

视觉编码器内部处理patch的空间位置；进入语言骨干后，`compute_3d_position_ids()` 根据图像网格等信息生成时间、高度、宽度位置ID，再通过Qwen的多模态旋转位置编码作用于注意力Q/K。**RoPE虽使用正弦、余弦，但它旋转的是Q/K，不是简单把正余弦向量加到token embedding上。**

**3. 自注意力本身不编码位置，动作序列怎么处理？**

AE为50个动作位置设置可学习的embedding，形状为 `[1,50,1024]`，加入动作特征；full-attention中还对动作Q/K使用RoPE。不同动作位置的embedding不同，同一组位置embedding在batch间共享。动作位置表示未来第几步，flow-time embedding表示当前噪声阶段，两者不能混淆。各位置共享网络参数和当前样本的state/time条件，不是共享动作值。

**4. Flow matching做动作生成，输入来自哪里？**

AE的直接输入为带噪动作 `x_t [B,50,32]`、flow time `t [B]` 和当前state `[B,32]`；图像、文本和状态形成的VLM prefix提供条件。训练时构造 `x_t=(1-t)a+tε`，目标velocity为 `ε-a`；推理时从高斯噪声开始迭代更新。两支逐层运行，在6个full-attention层拼接K/V，最终预测 `[B,50,32]` 的velocity。**不是两支完整跑完后再拼特征；flow matching是训练和生成方法，不是backbone后额外的模块。**

**5. Flow matching使用的VLM向量怎么理解？**

它是一串经过上下文融合的条件特征 `[B,L,2048]`，表达图像、任务文本与当前状态；AE读取对应full-attention层的中间特征生成的K/V，而非只读取最终一个向量。`state(32)` 与 `mask(32)` 拼成64维后，经MLP形成1个状态token。真实action chunk不属于AE可读取的prefix；训练追加的FAST动作token仅用于语言CE，AE不能读取，避免答案泄漏。

**6. 交互注意力的Q、K、V来自哪里？**

Q来自AE；K/V来自VLM prefix和AE两部分，使用各自的投影后沿token维拼接：`Q=Q_action`，`K=[K_vlm;K_action]`，`V=[V_vlm;V_action]`。因此是非对称joint attention：一次注意力同时读取外部条件和动作chunk，VLM自身不读取动作suffix。

| 张量 | 形状 |
|---|---|
| AE Q | `[B,8,50,256]` |
| VLM K、V | 各为 `[B,2,L,256]` |
| AE K、V | 各为 `[B,2,50,256]` |
| 拼接K、V | 各为 `[B,2,L+50,256]` |

**7. 推理为什么比较快？K/V cache起什么作用？**

当前默认用10次Euler更新，每轮同时预测整段50步动作的velocity，无需将动作离散后逐token自回归生成。固定观测下，VLM不读取带噪动作，因此其条件不随 `x_t` 和 `t` 变化；视觉编码与VLM prefix只需计算一次，缓存6个交互层的K/V，后续只运行较小的AE。AE自身Q/K/V随迭代变化，仍需每轮重算；观测、任务或状态变化后需重建VLM cache。10步是当前配置，实际延迟还取决于硬件、prefix长度和执行后端。

```text
视觉编码 + VLM 前向一次 → 缓存 6 层 prefix K/V
                                  ↓
高斯噪声 [B,50,32] → AE + Euler 更新 × 10 → 动作 chunk [B,50,32]
```

**其他设计与工程追问：**

- **为什么不直接让VLM输出动作token？** 50步32D轨迹是连续、强时序相关的控制量。flow expert在连续空间生成更合适，FAST只作为保护VLM表征的辅助监督。
- **为什么只在部分层joint attention？** Qwen的DeltaNet是递推线性注意力，强行拼接会破坏其状态语义；full-attention层支持标准Q/K/V融合，6次交互已能把条件传给action branch。
- **最重要的工程原则？** 32D不只是padding shape；mask、归一化、delta规则、checkpoint metadata和推理反归一化必须是一套契约。否则训练loss正常，部署动作仍可能是错误的。

### 3. MagicVLA Dynamic / SGM

本节用于项目复述；完整的信息流、训练与推理说明见 [Dynamic / SGM 原理](MagicAtom/04_Dynamic模型/01_MagicVLA_Dynamic_SGM.md)。

**一句话回答：** 在base的动作专家旁增加2D、3D world experts，用冻结teacher提供视觉、几何和运动latent监督，并在full-attention层与动作分支交互。目标是增强动作条件表征；闭环收益需单独验证。

#### 3.1 模型结构与监督

```text
当前图像 + 文本 + state/mask → Qwen VLM prefix
                                  ↓ 各 full-attention 层提供 K/V
                     2D World ↔ 3D World ↔ Action
                         ↓          ↓          ↓
                    视觉 latent  几何/运动 latent  action velocity
```

箭头表示可配置的信息通路，实际可见关系由配置决定。各专家在DeltaNet层分别运行，在full-attention层读取允许的K/V；VLM自身保持原生causal attention，不读取专家token。

| 分支 | 输入/预测 | 训练监督 |
|---|---|---|
| Action | 带噪action chunk、state、flow time → velocity | Flow Matching |
| 2D World | 可学习视觉queries → patch latent | 冻结DINOv3编码未来图像 |
| 3D World | 可学习queries → geometry / motion latent | 冻结Track4World处理当前—未来帧对 |
| VLM | 图像、任务与状态条件 | 配置启用的FAST CE / VLM-SFT CE |

这里预测的是特征表示，并非直接生成未来RGB视频。未来帧进入teacher监督路径；部署仅提供当前观测。future offset可按各source的action chunk时间跨度推导：50个动作点在2倍上采样源上对应25个源帧，不能统一解释成50帧视频。

#### 3.2 信息可见性与梯度路径

基础查阅：[Masked Attention](Note_Basics.md#basic-masked-attention)。

默认的专家可见关系如下；每一行都可由 `joint_attention_visibility` 配置：

| Query \ K/V | VLM prefix | 2D World | 3D World | Action |
|---|---|---|---|---|
| VLM | causal | 不可见 | 不可见 | 不可见 |
| 2D World | 可见 | 可见 | 可见 | 可见 |
| 3D World | 可见 | 可见 | 可见 | 可见 |
| Action | 可见 | 可见 | 可见 | 可见 |

三个开关分别控制不同问题：

- `joint_attention_visibility`：固定信息通路，训练和推理都生效。例如部分SFT配置关闭Action/3D对2D的读取。
- `action_world_stream_mask`：训练时随机屏蔽Action对某个World分支的读取，用于依赖性消融；不等于删除该专家。
- `knowledge_insulation`：控制机器人loss是否通过条件路径更新VLM。开启时隔离Flow/World梯度，VLM可由FAST/语言CE更新；关闭后允许联合优化，但实际可训练参数还取决于冻结开关。

#### 3.3 独立的 34D 训练版本

**本节仅适用于34D配方，不修改第1、2节的32D定义。** 代表配置为 [SGM action34 配方](../magicvla/configs/train/magicvla_sgm_pretrain_robot_group_23data_eo_action34_stride25.yaml)。多源数量应按启用条目与数据分片实际清点，不能只依据文件名。

| 索引 | 34D版本语义 |
|---|---|
| `0:7`、`7` | 左臂最多7个joint、左gripper |
| `8:15`、`15` | 右臂最多7个joint、右gripper |
| `16:19`、`19:25` | 左EEF xyz、rotation-6D |
| `25:28`、`28:34` | 右EEF xyz、rotation-6D |

32D源经reader显式映射到上述槽位；6-joint双臂数据的额外joint槽 `6/14` 无效，人手数据的arm joint槽仍无效。归一化、rotation groups、gripper absolute规则和推理恢复均使用34D索引。32D/34D模型的输入输出形状不同，checkpoint与norm statistics不可直接混用。

该代表配置使用50-step chunk、25帧anchor stride，EgoDex采用2倍语义上采样。Flow权重为1，FAST/2D/3D辅助权重启用调度：前20% 保持最大值，中间60% 余弦下降，最后20% 保持最小值。其余配方是否启用调度应分别读取配置。

#### 3.4 推理与训练一致性

推理关闭DINO/Track4World teacher的构造，保留学习到的2D/3D experts。固定观测下可缓存VLM prefix K/V；World和Action streams仍参与专家前向，不能直接沿用base的计算成本估计。

动作从masked noise开始进行多步Euler更新，再按对应source的统计和delta/rotation规则恢复。维度mask、相机预处理、动作选择和checkpoint metadata必须一致；仅监督joint的配置不能用于未监督的EEF控制。`action14in32`、34D下的joint/EEF子集实验要分别说明槽位映射，不能按文件名推断。

轨迹尾部也属于数据契约：SGM支持保留尾部anchor，以终止帧作为越界的World target；重复末动作的padding是否参与Flow loss由 `supervise_episode_end_padding` 控制。这与第2节原32D base配方的padding描述分别记录。

#### 3.5 验证与代码入口

验证分两层：单元/契约检查覆盖梯度路径、stream mask、teacher-free推理和动作恢复；能力评估比较action-only、加入2D/3D、不同可见关系及KI设置的闭环成功率与成本。open-loop动作误差和teacher latent loss只能作为诊断，不能替代闭环结果。

- [模型实现](../magicvla/src/models/magicvla_sgm/modeling_magicvla_sgm.py)：`MagicVLASGMPolicy`、joint trunks与缓存推理。
- [配置定义](../magicvla/src/models/magicvla_sgm/configuration_magicvla_sgm.py)：可见性、stream mask、辅助loss调度。
- [Teacher 实现](../magicvla/src/models/magicvla_sgm/world_teachers.py)：DINOv3与Track4World目标。
- [RoboDojo 服务](../magicvla/scripts/infer/robodojo_dynamic_server.py)：在线状态、图像与动作转换。

<a id="magic-memory"></a>

### 4. 后训练：memory 方向

这一阶段的目标不是重新预训练VLA，而是以RoboDojo轨迹为数据，对已有base policy做后训练 / SFT，让模型在动作预测时利用当前帧以前的视觉信息。实验中既有Pi0.5，也有MagicVLA / Hy-VLA类基座的尝试；下文的脚本和源码路径以当前能核对到的Pi0.5、Hy-VLA实现为准。动作chunk、动作空间和flow-matching监督原则上保持不变，主要比较的是“历史信息放在哪里、怎样压缩、怎样注入动作专家”。

三个方向可以按下面的关系来理解：

| 方向 | 历史信息进入模型的位置 | 主要机制 | 当前状态 |
|---|---|---|---|
| Hy-VLA-style-mem | 视觉编码器内部 | 6帧视频输入 + 时空注意力 | 已完成代码和RoboDojo后训练尝试 |
| RoboMME FrameSamp+Modul | action expert | 均匀采样 / 首帧特征 + cross-attention + RMSNorm调制 | 已完成多组采样与keyframe ablation |
| DM05-style memory | VLM prefix | 稀疏长历史帧 + SigLIP pooling | 正在验证 |

#### 4.1 Hy-VLA-style-mem：把历史帧作为短视频输入视觉塔

**一句话原理：** 不把过去帧简单拼成更多语言token，而是把同一相机的历史图像组成一个短视频，在视觉编码器中先做时间建模，再把融合后的当前帧特征交给后面的VLM和action expert。

**数据和输入。** 每个RoboDojo样本从当前时刻所在episode内取一个固定长度的图像窗口。代码配置为 `img_history_size=6`、`img_history_interval=20`，顺序是oldest → current；严格按代码口径，6帧总数包含最后的当前帧，可以在面试中概括为“把过去约6个时间点的图像历史送入模型”。训练时可以在每个时间间隔对应的小区间内采样，评估时使用确定的历史索引；episode开头不足的部分会落到第0帧，并用mask标识无效历史。历史只来自当前episode，不读取未来帧。

**模型改动。** `Hy-Embodied-0.5-VLA` 对原视觉塔增加video encoder路径：输入从单帧 `(B,C,H,W)` 变为 `(B,K,C,H,W)`。在视觉transformer的部分block中插入 `SpaceTimeBlock`，先对同一空间patch沿时间做causal attention，再做空间attention，并加入时间sinusoidal embedding；不同相机和不同空间位置不会互相混淆。经过指定层后只保留当前帧token，历史信息已经在视觉塔内部汇入当前帧表示，因此下游action expert接口不需要改变。

**为什么这样设计。** 视觉侧时空注意力适合捕捉遮挡前后的物体位置、运动方向和接触过程，同时通过“中间融合、末端只保留当前帧”控制token数和后续计算量。它的局限是窗口较短，主要解决短时动态和当前帧歧义，不负责跨episode的长期任务记忆；而且历史帧会直接增加视觉塔前段的计算和显存。

**面试表述：** “我在Hy-VLA分支上做了RoboDojo后训练，把每个相机的当前帧和过去5个采样时刻组成6帧短视频，在视觉encoder的部分层加入时空attention。时间attention使用causal mask，保证当前只能看过去；后面丢弃历史token，只保留当前帧的融合特征，所以不改action expert和动作输出接口。”

#### 4.2 RoboMME FrameSamp+Modul：离线特征采样，再调制 action expert

基础查阅：[Cross-Attention](Note_Basics.md#basic-cross-attention)、[RMSNorm](Note_Basics.md#basic-rmsnorm)。

**一句话原理：** 先用冻结的视觉塔离线提取整条episode的 `cam_high` 特征，训练样本只取其中一部分历史帧；再把历史视觉特征和时空位置编码投影成memory token，让action token cross-attend到memory，并用memory产生的scale / shift调制action expert的RMSNorm。

**历史采样。** `frame_memory.py` 中的 `even_sampling_indices()` 在episode起点至当前帧的闭区间内均匀取样，包含首帧和当前帧；`framesamp_budget` 固定memory token总预算，未使用位置右侧padding并由 `static_mask` 屏蔽。标准 `framesamp_modul` 配置通常是 `budget=512`、每帧16个token，也就是最多约32个采样帧；历史视觉特征保存在 `framesamp_features` 中，训练时不再重复跑历史图像的视觉encoder。当前三路相机仍走Pi0.5的普通输入路径，memory主要来自top-head / `cam_high`。

在此基础上做了两类ablation：

- **加入首帧：** 使用 `framesamp_sampling_strategy="first_frame"`，只提供episode的第0帧。`train_robodojo_mem_keyframe_256.sh` 对应的 `pi05_robodojo_mem_keyframe_256_v2` 配置使用256个token、每帧256个token，即用一张未做空间pooling的首帧作为memory；它重点验证“任务初始场景 / 初始物体信息”是否比长历史更有用。
- **提高keyframe采样比例：** 在数据采样层读取 `is_key_frame`，通过 `keyframe_mode="boost"` 提高关键帧权重，同时仍保留普通帧，避免模型只看到关键帧。不同实验中尝试过 `keyframe_boost=2` 和v2中的 `15`；这改变的是训练样本分布，不改变单个样本的动作监督。

**模型注入。** 每个采样帧的视觉embedding与3D sinusoidal temporal/spatial position embedding拼接，再经过 `PerceptualMemory` 投影到action expert的hidden size。Gemma action分支在transformer block中对memory做cross-attention；得到的memory condition继续经过 `MemoryRMSNorm` 生成scale和shift，调制action expert的FFN输入。也就是说，历史不直接塞进VLM的主prefix，而是作为action expert生成动作时的额外条件；`action_horizon`、动作维度和flow-matching loss都不变。

**优缺点。** 这种方式把历史视觉计算离线化，并用固定token budget控制训练成本；memory与动作分支直接交互，适合需要根据过去观测选择动作的任务。代价是需要维护episode级feature cache、采样索引、位置编码和padding mask的一致性；如果只提高keyframe权重，也可能损失普通过渡状态，因此必须与均匀采样和current-only做对比。

**面试表述：** “我复现并扩展了RoboMME的FrameSamp+Modul。历史帧先用base Pi0.5的SigLIP离线编码，在线只读取固定预算的历史token；模型用时空位置编码区分帧和空间位置，再让action expert cross-attend这些memory，并通过RMSNorm的scale/shift做调制。我还比较了均匀历史、只加首帧，以及提高 `is_key_frame` 采样权重三种数据策略。”

#### 4.3 DM05-style memory：稀疏长历史作为 VLM prefix

**一句话原理：** 不只看短窗口，而是在同一个episode内按较大的时间间隔抽取一段严格过去的top-head图像，把每帧压缩成少量视觉token后，和当前图像、语言一起放入VLM prefix，让模型在生成动作前形成更长时间尺度的场景状态表示。

**当前实现。** 以 `kai0-robodojo-dm05style-mem/train_robodojo_5tasks_dm05style_mem_20_25_keyframe_boost25.sh` 对应的Pi0.5五任务配置为例，参数为 `history_frames=20`、`history_stride=25`：每个当前样本携带20张严格过去的 `cam_high` 帧，时间间隔为25个action step，当前帧仍由普通相机输入提供。每张历史帧经过共享SigLIP后，将视觉token grid平均池化到4×4网格，变成每帧16个token，总共320个history token；`history_is_pad` 用来屏蔽episode开头不存在的历史。

**与前两个方向的区别。** 这里历史token直接追加到VLM prefix，与当前图像和语言共同参与prefix attention，再由action expert使用最终的条件表示；它不是FrameSamp那种只在action expert内部cross-attend的memory，也不是Hy-VLA那种在视觉塔内部做时空attention。它更强调“记住较长时间范围内的任务状态”，例如物体在早期出现过什么、任务进度如何，而不是只恢复当前帧附近的运动细节。

**验证重点。** 该方向已有实现与实验配置，闭环收益仍需对应评估记录支持。主要需要检查长历史token是否挤压当前图像 / 语言的有效上下文、padding mask是否正确，以及稀疏采样间隔是否适合不同任务；后续应至少做current-only、短历史、20帧历史和不同pooling比较，并观察动作成功率与长时任务表现。

**实验边界。** 仓库已有多任务和keyframe boost配置；配置存在不等于效果已验证。20帧、stride=25覆盖最远500个数据帧，秒数需按数据FPS换算。闭环收益以对应checkpoint、任务和评估记录为准。

#### 4.4 三个方向的统一训练和比较方法

**MagicVLA分支补充。** 原仓库除mem-v0/v1外，feat/history还实现了压缩历史图像的Base/SGM prefix和部署侧ring buffer；model/magicvla_streamer包含观测动作历史与专家递归状态。它们与上述Pi0.5实现分别记录，分支commit、机制和边界见[MagicVLA Memory分支实现](MagicAtom/01_视觉记忆/04_MagicVLA_Memory分支实现.md)。

先区分两个采样层次：history sampling决定单个样本读取哪些历史，`keyframe_boost` 决定哪些当前anchor更常被训练。FrameSamp的均匀采样包含当前帧，DM05的历史分支只取严格过去帧，两者都不能读取未来。

三个方向都从已有base checkpoint初始化，在RoboDojo轨迹上预测当前时刻开始的action chunk。应在同一基座、数据划分、动作契约和训练预算下，各自对比current-only baseline；跨基座结果不能直接归因于memory。面试中可以按四个维度总结：

- **信息放置位置：** Hy-VLA放在vision encoder内，FrameSamp+Modul放在action expert，DM05-style放在VLM prefix。
- **时间范围：** Hy-VLA是短时密集窗口，FrameSamp是固定预算的可配置采样，DM05-style是稀疏但更长的历史。
- **计算方式：** Hy-VLA在线参与视觉前向；FrameSamp历史特征离线缓存；DM05-style仍需对历史图像做SigLIP编码，但通过pooling控制token数。
- **验证目标：** 分别检查短时遮挡/运动趋势、首帧信息回忆、长时物体与任务状态记忆；这些是实验假设，需通过成功率和历史屏蔽/打乱对照验证。

比较时记录任务成功率、跨seed波动、推理延迟、峰值显存和实际history token数；缺少结果时标为待验证。

最重要的工程契约是：历史帧索引不能读到未来，训练和推理的时间顺序必须一致，首帧padding要有mask，token budget / pooling / position embedding必须与checkpoint配套；否则loss可能正常下降，但部署时模型看到的memory与训练语义不一致。

代码入口：[DM05 历史视觉 token](../kai0-robodojo-dm05style-mem/src/openpi/models/pi0.py)、[五任务实验脚本](../kai0-robodojo-dm05style-mem/train_robodojo_5tasks_dm05style_mem_20_25_keyframe_boost25.sh)、[FrameSamp 采样与特征组织](../kai0-robodojo-frame-mem/src/openpi/models/frame_memory.py)。

基础公式与最小实现见 [基础知识：VLA / Transformer 模块](Note_Basics.md#basic-vla)。

<a id="interviews"></a>

# 二、面试复盘

## 25.10.10 海恒智能 机械臂算法工程师

### 1. ros加moveit2 怎么做一些完整的运动规划和控制？

整个链路分四层：模型 → 感知 → 规划 → 控制。

**1) 机器人模型与配置**

- **URDF**：机器人描述文件，定义几何结构、关节、连杆、传感器。
- **SRDF**：MoveIt2自动生成，在URDF基础上加语义信息（规划组、虚拟关节、碰撞对白名单等）。

**2) 感知与环境建模**

- **传感器数据**：订阅深度相机（如Kinect）、激光雷达的点云。
- **环境表示**：Costmap，通常订阅ROS2话题实时更新。

**3) 运动规划（Motion Planning）**

规划请求（Planning Request）包含：

- **起始状态**：机器人当前关节角度
- **目标状态**：目标末端位姿或目标关节角度
- **路径约束**：例如保持末端姿态不变
- **障碍物信息**：来自环境建模

规划器（Planners）：MoveIt2集成了多种算法，包括采样式（RRT/RRT* 等）和优化式（CHOMP/TrajOpt等）。

逆运动学求解器（IK Solvers）：规划过程中频繁调用，把末端目标位姿反解成关节角度。MoveIt2默认使用KDL或TRAC-IK。

**项目中的体现**：

- `panda_pick_n_place.py` 中 `self._panda.solve_ik(self._end_effector_target)` 直接计算关节目标；
- 也可以通过 `move_group` 接口发送规划请求，由MoveIt2自动选规划器并生成轨迹。

**4) 控制层**
最终轨迹经 `ros2_control` 下发给各controller（关节位置 / 力矩等），驱动真实或仿真机器人执行。
### 2. opencv中使用了哪一些算法？
1）边缘检测（比如 Canny）找出墙面上的“线条”
2）轮廓检测（findContours）找出抹头的边缘位置  <!-- TODO: 确认「抹头」是否应为「抹布头」或具体物体名 -->
3）霍夫直线变换（Hough Lines）拟合出这两条线的角度


### 3. CAN通信两个节点在主线上无法通信，怎么排查问题？

1）从软件角度：工作中遇到的实际bug案例，调度代码里面屏蔽了
2）硬件角度：示波器看差分波形，看看显性隐性电平对不对；监听抓ACK故障位

### 4. 之前项目使用的CAN通信波特率是多少？

500 kbit/s注意单位

### 5. 多个模块在ROS中，是怎么管理的？

1）引入组件（Component）机制，通过rclcpp_components实现运行时动态加载节点为共享库
2）Docker容器化

### 6. 节点启动是怎么做的？

launch文件

### 7. ROS的通讯机制是什么？（分布式）

分布式、异步、多对多

### 8. PID的三个字母分别代表什么意思？有什么作用？

P：比例，消除当前误差；
I：积分，消除稳态残差；
D：微分，预测未来误差变化，抑制超调。

### 9. PID控制算法和模糊控制算法相比有什么优势和劣势？

| 维度 | PID | 模糊控制 |
|---|---|---|
| 原理 | 线性反馈：误差的P / I / D三项加权求和作为控制量 | 基于模糊逻辑：误差与误差变化率模糊化（"正大""负小"...）→ 规则库推理 → 解模糊化输出 |
| 模型依赖 | 可通过实验整定；模型有助于分析与调参 | 可用经验规则，不要求精确解析模型 |
| 优点 | 结构简单、响应快、稳定性好 | 鲁棒性强、贴近人类经验 |
| 缺点 | 对系统模型和参数变化敏感 | 规则设计依赖经验，调试复杂 |
| 适用场景 | 模型明确、线性、控制精度要求高 | 非线性、时变、难建模、不确定性高 |

**选型思路**：结合控制目标、非线性程度、测量噪声和调参成本选择；有无精确模型不是唯一标准。

### 10. 机械臂出现轨迹抖动，或者说关节不连续，有可能是因为什么原因造成的？

1）轨迹规划层（Trajectory Planning）

- 轨迹平滑性不足（路径不连续或加速度跳变），贝塞尔曲线B样条插值
- 逆运动学（IK）求解不稳定，对于冗余自由度机械臂IK解不唯一，导致奇异点附近IK解算器输出跳变
2）运动控制层（Motion Control）

-  控制器增益设置不当（PID / 力矩环震荡）
现象：关节在目标位置附近高频抖动（小幅度振荡），尤其在低速或静止时明显。
原因：位置环或速度环PID增益过高，导致系统震荡；力矩环带宽过高，激发结构柔性模态；未做摩擦补偿或前馈控制，导致稳态误差 + 积分饱和。

- 采样频率不一致或通信延迟

### 11. 如果是关节的解不是唯一的，这个时候应该怎么做？

关节限位 + 避障代价

### 12. Docker的主要步骤

- 1.创建Dockerfile ：定义基础镜像、安装依赖、配置环境变量等
- 2.编写启动脚本 ：在项目中有docker/entrypoint.bash ，用于容器启动时执行的命令 
设置ROS2环境变量、构建工作空间、启动相关节点

- 3.运行脚本 ： docker/run.bash用于简化Docker容器的启动
这种方式的优点是：

- 环境一致性：所有开发者使用相同的环境
- 依赖隔离：避免系统依赖冲突
- 便于部署：可以轻松在不同机器上运行

### 13. git基本操作

常用指令：

| 指令 | 用途 |
|---|---|
| `git clone <url>` | 下载仓库 |
| `git status` / `git diff` | 查看修改状态和未暂存差异 |
| `git add <file>` | 将文件修改加入暂存区 |
| `git commit -m "说明"` | 提交暂存区内容 |
| `git log --oneline -n 10` | 查看最近提交 |
| `git switch -c <branch>` / `git switch <branch>` | 新建并切换分支 / 切换已有分支 |
| `git fetch` / `git pull` | 获取远端更新 / 获取并整合更新 |
| `git push` | 推送本地提交 |
| `git merge <branch>` | 合并另一分支 |
| `git cherry-pick <commit>` | 应用指定提交 |
| `git stash push` / `git stash pop` | 临时保存 / 恢复未提交修改 |
| `git restore --staged <file>` | 取消暂存，保留工作区修改 |
| `git revert <commit>` | 新建提交，撤销某次提交的效果 |

`git restore <file>`会丢弃该文件未暂存的修改；`git reset --hard`会丢弃已跟踪文件的未提交修改，使用前先确认需要保留的内容。

## 26.1.26 iData具身智能算法

### 1. 手眼标定相关，怎么校准，有没有自动校准

采集机器人与标定板在多种位置、姿态下的观测，求解相机与机器人之间的固定变换，再用未参与求解的数据检查误差。采样要有足够的平移与旋转变化，不能只固定深度或沿单一轴运动。原理见[相机与手眼标定](Note_Basics.md#basic-calibration)。

### 2. 为什么用五次多项式，和三次多项式比有什么优点

基础查阅：[轨迹与控制](Note_Basics.md#basic-control)。

- 选五次多项式的核心是补齐了加速度的边界约束，实现位-速-加全连续：三次多项式仅能约束位、速，加速度无约束导致拼接处跳变，有硬件冲击
- 五次多项式可同时约束两端的位置、速度和加速度。多段轨迹在边界条件匹配时可保持加速度连续，但不会自动保证段间jerk连续；若需限制jerk，应额外设计约束或采用相应轨迹生成方法。

### 4. VLA和传统的规控相比有什么优势？

- 传统规控感知、决策控制分离，依赖精确的运动学动力学模型，PID,MPC
- VLA端到端策略，泛化性好(相对来说)，多模态数据，预训练大模型

### 5. 脚部电机互斥机制是什么，怎么实现的

- 通过标志位实现腿部电机和手臂电机的双向互斥，确保二者不能同时运动。

### 6. rl训练大概做了什么？奖励函数怎么设定的？

主要在Isaac Lab中用PPO做机械臂抓取任务，从Lift扩展到Grasp & Pull，重点是观测设计、奖励调整和失败分析。

奖励按任务过程组织：接近物体、姿态对齐、抬升或拉到目标位置，并加入动作平滑等约束。训练中检查是否出现“靠近但不抓”“完成后继续抬升”等奖励投机，再调整权重或让阶段奖励在完成后衰减。Curriculum用于逐步增加难度，效果结合成功率和实际轨迹判断。

具体过程见[RL项目实践](Note_OtherProjects.md#rl-project)，通用原理见[RL基础](Note_Basics.md#basic-rl)。

### 7. sim2real中的难点是什么，会存在哪些问题，应该怎么解决？

- 物理误差：摩擦系数、关节间隙、电机延迟、力控精度误差、环境扰动（如气流、震动、光照变化）
- 视觉误差：仿真画面无噪点、无运动模糊，真实相机有曝光、白平衡、畸变；仿真物体纹理单一，真实世界有反光、阴影、遮挡
- 动作误差：动作执行有延迟比如指令移动10cm，实际只动9.5cm）；反馈信号（如力传感器、视觉反馈）有噪声、采样延迟
- 解决方法：
    1.域随机化（Domain Randomization）
    2.域适应（Domain Adaptation）

### 8. 机械臂做路径规划的时候，怎么避免碰撞的

先在规划场景中建立机器人、障碍物和自碰撞模型，再对候选路径及插值轨迹做碰撞检查。关节限位只限制可达范围，笛卡尔插值也不自动保证无碰撞；执行前还要检查速度、加速度和环境变化。

## 26.2.11 星尘智能

### 1. go函数里面的原理？笛卡尔路径规划的原理？

- go() = plan() + execute()
    plan路径规划，OMPL(RRTconnect)
    execute执行规划好的路径
    包含碰撞、逆运动学KDL、执行控制
- 笛卡尔路径：
    线性插值+ik解算

### 2. RL相关知识 什么是离线学习，什么是在线学习，大概各自的算法有哪些？

- 在线强化学习（Online RL）：智能体一边和环境交互，一边学习，用的是自己刚产生的新数据。PPO、SAC、TD3
- 离线强化学习（Offline RL）：只用现成的数据集学习，不和环境交互。CQL、IQL、TD3+BC

### 3. VLA数据有什么采集方法？

- 遥操作，通过手柄 / VR / 键盘 / 鼠标远程操控机器人，同步采集相机图像流、语言指令、机器人关节动作序列，事后自动标注或人工补语言描述OpenVLA
- 仿真数据采集

### 4. 逆运动学解算的原理是什么？用的什么方法？

基础查阅：[IK 原理](Note_Basics.md#basic-ik)。

- 逆运动学：已知末端位姿，求关节角。解算本质是解非线性方程组。
    1.解析法：快、专用、靠几何推导
    2.数值法：通用、迭代、靠雅可比 / 优化
- 6轴工业臂：解析IK（几何法）
- 7轴及以上冗余臂：数值IK（雅可比 + 阻尼最小二乘）
- 仿真、规划、RL、VLA：数值IK / 优化型IK
- ROS、MoveIt：用的是TRAC-IK、KDL数值求解器

### 5. 冗余自由度有什么解算方法

- 雅可比伪逆 + 零空间投影
零空间可以在不影响末端的前提下：让臂远离障碍物、远离关节极限、远离奇异点、让运动更平滑

- 阻尼最小二乘法（公式见 [基础知识中的 IK](Note_Basics.md#basic-ik)）

### 6. sim2real中的难点是什么，会存在哪些问题，应该怎么解决？

见26.1.26第7题。

### 7. 设计强化学习策略的时候有哪些方法论？

按以下顺序设计即可：

1. 明确成功、失败和超时条件，区分任务终止与时间截断。
2. 观测覆盖任务所需状态，动作范围与控制频率匹配机器人能力。
3. 用完成奖励确定目标，用接近、对齐等稠密奖励引导探索，避免奖励之间冲突。
4. 先跑通简单场景，再逐步增加初始位置、物体属性等变化。
5. 同时看成功率、失败轨迹、各奖励项和PPO训练指标，再决定调奖励还是超参数。

不必堆很多奖励项，先保证每一项都服务于任务目标。

### 8. PPO算法相关,大致介绍一下

见[RL基础：PPO与GAE](Note_Basics.md#basic-rl)；项目公式与代码见[PPO实践记录](Note_OtherProjects.md#ppo-practice)。

## 26.3.20 魔法原子 VLA算法工程师 一面

### 1. 采集的 ego-centric 数据以及 umi 数据是怎么接入模型的？

先统一转换成Lerobot 2.1格式数据集，再通过两套policy，π0.5 backbone和base ckpt完全共享，差异只在input/output transform层：

- `pi05_pico`：单目第一视角，20D末端位姿（每只手3 xyz + 6 6D rotation + 1 gripper），缺失的双wrist用零图 + `image_mask=False` 补齐。
- `pi05_kaiumi`：三相机，14D关节空间（每只手6 joint + 1 gripper），直接复用Aloha接口做joint flip和gripper角度换算。
模型内部把所有state/action pad到32维统一进Transformer，输出再截断回真实维度。

### 2. 这两种数据是怎么 align 在一起的？

**分层align，能align的硬align，不能align的用阶段式训练桥接**：

- **能align的**：模型架构（同一份Pi0Config + base ckpt）、数据格式（都用LeRobot v2.1）、Tensor维度（pad到32D）、图像通道（三相机接口）——全部强制对齐。
- **不强行align的**：action物理空间。Pico是末端位姿、Kaiumi是关节空间，物理意义不同，硬映射会丢信息。

action空间靠**三阶段课程式训练**桥接：

```
base ckpt → Pretrain (Ego) → Midtrain (UMI) → Posttrain (遥操) → final policy
            数据量最大        过渡真机分布      精细 fine-tune
```
按"通用 → 半专用 → 精细"顺序学习，比把三类数据混训稳得多——混训时高方差的ego数据会淹没遥操精细信号。最终双臂操作任务完成率89%。

## 26.3.22 魔法原子 VLA算法工程师 二面

### 1. 既然 Pretrain 阶段冻结了 action expert，那 pretrain 的输入输出是什么？用什么监督？

**输入 / 输出 / 监督跟全参微调完全一样**，flow matching MSE一路不变（`src/openpi/models/pi0.py::compute_loss`）：

- **输入**：`cam_high` 图像（缺的双wrist用零图 + `image_mask=False`）+ language tokens + 20D state + ground-truth action chunk `[B, 50, 20]`。
- **输出**：`v_t = action_out_proj(suffix_out[:, -H:])`，即未来50步的向量场。
- **监督**：`L = ‖v_t − (noise − actions)‖²`。

**"冻结action expert"的实际范围比字面小**

`_build_freeze_filter()` 里 `TRAIN_ACTION_EXPERT=false` 只冻PaliGemma内部双专家中 `.*_1.*` 后缀的Gemma-300M expert权重；模型外层的 `action_in_proj / state_proj / action_out_proj / time_mlp` 等小投影头**不在freeze范围内，仍然可训**。Pretrain实际训练的是：**SigLIP + Gemma-2B LLM + 这些小投影头**。

**冻结到底改变了什么——只换梯度的消费方**

Loss还是作用在 `v_t` 上，梯度反向穿过整个双专家Transformer。冻结的是action expert的参数更新，不是其对输入的梯度传播；通过双专家联合attention，**梯度仍然流回VLM**，驱动VLM学到"能让frozen action expert解码出ego动作"的视觉/语言表征。

**为什么这么设计**

- Action expert在 `pi05_base` 里已经学到了通用motor prior，用Pico ego这种噪声大、视角新的数据全参微调会污染它；
- 真正需要适配的是VLM——第一人称视角与遥操第三人称差异极大，必须重学；
- 需要区分参数冻结与 **KI（Knowledge Insulation）**：冻结AE决定哪些参数更新；KI决定动作loss是否回传到VLM。开启KI时，VLM仍可由FAST或语言CE更新，不能将其直接等同于“冻结VLM”。

### 2. 那 pi05 本身的 VLM 是靠什么监督？这个监督具体指什么？

要分两个语境：

**(a) PI实验室预训练 `pi05_base` 时**

VLM被 **FAST离散动作token的下一token交叉熵**直接监督（详见前文「PI0训练流程笔记」§6.3）。两阶段：

| 阶段 | α | VLM监督 | Action expert |
|---|---|---|---|
| FAST预训练 | 0 | FAST token下一token交叉熵 | 关 |
| Flow matching后训练 | 10 | 仍被FAST交叉熵监督 | flow matching MSE |

联合损失：`L = H(FAST_tokens) + α · ‖v_θ − u_τ‖²`

FAST token是把连续action chunk经DCT + BPE离散化得到的整数序列；监督VLM就是让它把这个序列像"句子"一样一个个吐出来——本质就是next token prediction，复用LLM的训练范式。

**(b) 我们在ego数据上做Pretrain时**

openpi这套代码 `compute_loss` 只算flow matching MSE，**FAST路径没启用**。VLM没有独立监督，只能靠MSE通过双专家联合attention反向传播。冻结action expert是将适配集中到VLM的一种训练选择，并非VLM能够学习的必要条件；是否冻结应结合梯度路径、数据量和对照实验决定。

**"监督"具体指什么**

= 数据集里有"标准答案" → 模型预测与答案的差 = loss → 梯度 → 更新参数。同一份action chunk真值，可以走两种监督路径：

- 离散：FAST token序列 → 交叉熵
- 连续：action chunk张量 → flow matching MSE

PI0.5 base训练时两条路径并存（FAST给VLM、MSE给action expert + VLM）；下游Pretrain (Ego) 阶段只剩MSE这一路。

## 26.9.9 TeleAI 具身智能算法研究员 秋招一面

### 1.具身智能长期发展需要积累什么？

我更看重数据质量、动作表示、真实部署和评测之间的闭环。超维的实验说明，数据量增加不一定带来收益，数据必须与目标机器人的动作空间和时间轴对齐；魔法原子的工作则是先建立统一数据接口、VLA base model和后训练评测，再用下游结果反向迭代。

## 26.9.10 智元具身算法实习一面

### 1.超维动力-Ego、UMI、遥操分别有多少数据？为什么UMI更有效？

超维使用的量级是：Ego约500–600小时；UMI约1000条，每条约1分钟；遥操约100条，每条约1分钟。UMI与遥操的动作表示更接近，UMI经过IK可以转成目标机械臂的joint，因此在实际实验中的收益比Ego更明显。这个现象还同时受到数据质量、视角和动作表示的影响，不能只归因于数据量。

### 2.超维动力-Pico在Ego和UMI中分别做什么？为什么UMI位姿更可靠？

Ego使用头显视频和手部tracking；UMI将tracker固定在假爪上，Pico直接提供刚体6DoF位姿，训练图像来自固定摄像头，再通过IK得到机器人joint。刚体tracker的跟踪链路比完整人手关键点估计更直接，通常更稳定；项目中没有做严格的设备精度对照。

### 3.超维动力-为什么把UMI的EEF轨迹转成joint？一份数据能否复用于不同机器人？

UMI原始数据保留的是假爪EEF轨迹。根据最终demo或遥操使用的机械臂构型，可以换用对应的运动学模型和IK，将同一份EEF示教retarget成不同机器人的joint数据，从而减少重复采集。超维当前实际落地的是松灵双臂；“可面向不同构型重新映射”是pipeline的复用设计，尚不等于已经完成所有构型的验证。换构型时仍需重新检查坐标系、TCP、工作空间、关节限位、碰撞和轨迹连续性。

### 4.超维动力-Ego数据的时间戳为什么要补偿？140ms 是怎么来的？

视频和手部tracking不是同一条采集链路：摄像头曝光和帧写入、传输或缓存、tracking算法处理、结果写盘都会引入延迟，帧率和调度差异还可能造成抖动。因此画面中的手部动作与tracking轨迹会错开。在那一批Ego数据上，我们通过图像与tracking的overlay对齐检查，估计出约140ms是较合适的补偿量，并在切片前用时间戳修正脚本统一时间轴。这个数值是针对该批数据的经验校准，不应理解为所有数据都适用的固定系统常数。

### 5.超维动力-除了时间同步，Ego数据还怎样做质量控制？

先做硬过滤：无效相机标定、视频过短、视频与tracking时长不匹配、双手大量缺失、动作范围异常等直接剔除；再从视觉、动作、时序和任务内容进行软评分。对局部offset随数据变化的问题，当前主要依靠可视化和批次级校准，没有把动态offset估计作为已完成能力。

### 6.魔法原子-Ego人手如何映射到机器人gripper？

EgoDex的`virtual_hand`分支用拇指尖与食指尖的三维距离表示手的开合程度，再映射为连续的gripper开合标签：

```text
d = ||p_index_tip - p_thumb_tip||
gripper = clip((d - d_closed) / (d_open - d_closed), 0, 1)
```

生产配置中，`d_closed=0.01 m`、`d_open=0.12 m`：距离不超过1 cm记为闭合0，不小于12 cm记为张开1，中间线性插值。左右手分别计算，得到连续开合量；这两个阈值是配置参数，不是所有人手通用的标定常数。

state保存当前帧开合量，action由下一帧目标生成。32D数据中，左右gripper分别写入索引`6/13`；人手没有真实机械臂关节，相关joint槽位保持无效。EEF姿态另由手部关键点构造，不用这一个距离推断姿态。

这里生成的是归一化开合标签，不是电机角度，也不是完整五指重定向；部署到具体夹爪时，还需按设备接口转换量程和开闭方向。该开合公式本身没有时序滤波或开闭滞回。

代码：[compute_virtual_hand](../pretrain_data_pipeline/src/pretrain_data_pipeline/data_io/eef_converter.py)、[生产配置](../pretrain_data_pipeline/configs/run/production/egodex_virtual_hand_quality_production_120core.yaml)。

## 26.9.11 蚂蚁灵波具身算法实习一面

### 1.魔法原子-32D action space 是什么？为什么同时保留joint和EEF？

32D由双臂joint/gripper的14D和双手EEF的18D组成；每侧EEF是xyz位置加6D rotation。joint提供可执行的机器人控制量，EEF提供与视觉几何更直接的表示。不同数据缺失的维度通过mask屏蔽，避免把补零误当成真实状态。当前实际的UMI retarget只落地到ARX5；统一32D接口保留了后续接入其他数据源的空间。

### 2.魔法原子-为什么统一到头部相机坐标系？UMI有腕部相机怎么办？

统一的是EEF标签的表达坐标系，不是只使用头部相机。cam_high更容易作为Ego与机器人数据的共同视觉参考；UMI仍可保留双wrist图像，缺失或无效相机通过camera mask处理。joint仍按目标机器人定义，不会变换到相机坐标系。

### 3.魔法原子-没有官方相机内外参时，如何估计 Hy-UMI 的参数？如何验证？

从图像中提取左右设备的blob，即连通区域，与已知世界坐标下的双手3D轨迹建立多帧对应，优化针孔模型的内外参，同时估计tracking原点与图像blob质心之间的device offset。使用跨episode共享参数和鲁棒重投影目标降低误检影响，再做session级小范围修正。没有官方真值时，主要用重投影残差和跨任务overlay验收；这些指标不能直接等价为真实3D位姿误差。

### 4.魔法原子-VLM、Action Expert和Flow Matching的信息流是什么？为什么采用π/Qwen风格？

图像、文本和state/mask形成VLM条件；Action Expert接收50步、32D的noisy action，并加入action position、flow time和state条件。两支按三层Gated DeltaNet加一层Full Attention的结构重复运行，在Full Attention层由action读取VLM条件，最后预测32D flow velocity，再通过反向积分生成动作chunk。选择 π/Qwen风格主要是为了复用视觉语言先验，并用连续action expert建模动作chunk；同时保持已有hybrid结构，便于在有限资源下快速建立可训练、可评测的baseline。没有做完整的架构横向消融，因此不宣称它普遍优于其他VLA架构。

### 5.魔法原子-Ego、UMI、真机和仿真数据如何混训？mask与robot tag分别解决什么？

所有数据统一到同一套相机key、32D state/action语义和LeRobot数据格式，并按source独立做归一化。action mask和state mask表示哪些维度真实存在，camera mask表示哪些图像有效；robot tag用来说明当前embodiment。mask解决缺失值问题，robot tag解决构型条件问题，两者不能互相替代。当前UMI的实际joint映射目标是ARX5。

### 6.魔法原子-为什么可能出现EEF-only优于joint-only和EEF+joint？

这是部分微调实验中的观察，不是普遍规律。预训练中的Ego、UMI原生表示更接近EEF，IK生成的joint可能带来额外误差；同时预测两种表示也可能增加优化负担。要固定数据、训练预算和执行方式做受控消融，才能区分表示、标签质量和多目标监督的影响。

### 7.absolute action和delta action怎么选择？

基础查阅：[旋转增量](Note_Basics.md#basic-rotation)、[动作表示](Note_Basics.md#basic-normalization)。

应按物理量分别决定。超维Ego使用xyz delta、6D rotation absolute、gripper absolute；UMI和遥操使用joint delta、gripper absolute。魔法原子中平移和joint使用相对量，旋转按合法旋转组合处理，gripper保持绝对状态。没有完整重跑预训练的absolute/delta消融，已有比较主要集中在微调阶段的action representation。

### 8.魔法原子·Memory-历史帧怎么融合？FrameSamp的token budget和关键帧怎么理解？

不同分支的实现不同：MagicVLA v0使用零门控temporal delta，v1使用先时间、后空间的attention融合六帧；Hy-VLA在视觉塔中加入时空attention；π0.5 FrameSamp+Modul则对历史图像做离线视觉编码和空间压缩，再让Action Expert cross-attend历史token并做调制。

FrameSamp的基线是在当前帧以前均匀采样历史，固定budget决定每帧保留多少token、最多覆盖多少帧；提高首帧或is_key_frame权重属于额外消融，不是自动理解语义关键帧。若关键帧定义依赖未来动作，训练和推理之间会产生信息不一致。

### 9.魔法原子·Memory-真正的memory应该带来什么能力？如何验证？

基础查阅：[模型评测](Note_Basics.md#basic-evaluation)。

目标是利用过去信息判断任务进度，记住已经不可见的目标或遮挡前的状态，而不是单纯增加历史图像。应构造“当前观测相近、历史不同、正确动作不同”的任务，并比较无历史、正确历史、打乱历史和移除关键历史的结果。只有依赖历史的任务稳定受益，才能说明模型确实使用了memory。

### 10.不同数据集的任务组织和数据工厂应该怎样设计？

数据不只是小时数，还要覆盖完整交互过程、阶段边界、失败恢复和真正需要历史的长程任务。采集时应统一设备、坐标和时间戳，控制丢帧、异常头动和手部不可见；标注可采用自动标注、人工抽检和反馈修正。结合超维经验，UMI更适合承担中间阶段的目标机器人动作对齐，真机数据用于最终执行适配。

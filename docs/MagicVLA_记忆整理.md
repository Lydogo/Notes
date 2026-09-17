# MagicVLA Memory整理

本文汇总MagicVLA短时视觉记忆、proprio记忆和长期记忆的实现思路。当前最可靠的能力是固定窗口的短时视觉记忆；动态事件记忆、语言摘要和Test-Time Training仍属于后续方向。

相关笔记：

- [MagicVLA短时视觉记忆](../MagicAtom/01_视觉记忆/02_MagicVLA_短时视觉记忆.md)
- [π0.5 FrameSamp实现](../MagicAtom/01_视觉记忆/03_π0.5_FrameSamp_Modul_实现.md)
- [MagicVLA视觉记忆完整记录](../MagicAtom/99_详细记录/01_MagicVLA_视觉记忆_完整记录.md)
- [MagicVLA预训练实验](MagicVLA_预训练实验.md)
- [统一数据处理笔记](../Note_DataPipeline.md)
- [基础知识](../Note_Basics.md)

## 1. 结论先行

MagicVLA Memory建议分阶段实现：

1. **P0：修复数据契约**。明确历史帧、当前state、有效mask和真实时间差，建立memory专用诊断集。
2. **P1：短时记忆基线**。使用K=6、约1秒间隔的视觉历史，先验证模型是否真正使用历史。
3. **P2：固定预算压缩**。把历史压缩成固定数量的memory token，控制VLM token和推理延迟。
4. **P3：事件记忆**。保留任务起始锚点和少量关键帧，再训练动态选择器。
5. **P4：语义长记忆**。在有子任务、成功/失败标签的长流程数据上维护递归文本摘要。

不要直接从P3或P4开始。只有当模型已经能在成对样本中利用短时历史，动态选择和语言摘要才有明确的验证基础。

## 2. MagicVLA-Base基础架构

### 2.1 模型主干

基本链路为：

~~~text
多路相机 + 任务语言 + 当前state
                 ↓
           Qwen3.5-2B VLM
                 ↓
             VLM prefix
                 ↓
       MagicVLA Action Expert
                 ↓
          action velocity
~~~

| 分支 | 作用 |
|---|---|
| Qwen VLM | 编码图像、语言、机器人信息和当前state |
| Action Expert | 根据VLM prefix、当前state和带噪action chunk预测速度 |
| 混合层 | 每3层Gated DeltaNet后接1层Full Attention，共6组跨分支交互 |

Action Expert为独立的24层、宽度1024的动作主干。Full Attention层中，action query读取VLM prefix的K/V；VLM不读取action suffix。因此Memory如果进入VLM prefix，会同时影响VLM表示和action-to-prefix的计算量。

当前实现中的关键入口：

- [modeling_magicvla_base.py](/pfs/user/magicvla/src/models/magicvla_base/modeling_magicvla_base.py:1783)
- [processor_magicvla_base.py](/pfs/user/magicvla/src/models/magicvla_base/processor_magicvla_base.py:236)
- [lerobot_history.py](/pfs/user/magicvla/src/data/lerobot_history.py:8)

### 2.2 动作与state

Flow Matching训练时，从动作a和噪声epsilon构造：

~~~text
x_t = (1 - t) a + t epsilon
v*  = epsilon - a
L_flow = MSE(v_theta, v*)
~~~

动作chunk的长度、有效维度和归一化统计必须与checkpoint绑定。32D和34D是两个独立动作版本，分别维护动作索引、dimension mask、统计量、动作头和部署脚本，不能混用。

当前state有两条路径：

1. 通过投影得到一个连续token，加入VLM prefix；
2. 通过state projection和timestep embedding调制Action Expert。

Memory版本应保持当前state路径不变。过去state如果启用，应单独投影为历史token，不能把T个state展平后继续标记为Current state。

### 2.3 当前代码状态

LeRobot reader已经可以按anchor读取连续历史，并返回时间偏移；processor也能保留TCHW图像形状。但当前模型的[prepare robotics prefix](/pfs/user/magicvla/src/models/magicvla_base/modeling_magicvla_base.py:1783)仍会对四维图像取最后一帧，再以单张image送入VLM。因此只修改obs_steps或数据配置，不能自动启用视觉Memory。

启用Memory前需要明确三件事：

1. 历史图像使用多image、Qwen video还是时间压缩模块；
2. 当前state和历史state分别进入哪条网络；
3. history_valid、history_dt和相机有效mask如何传入模型。

## 3. 短时视觉记忆

### 3.1 输入约定

默认短时窗口为6帧、约1秒间隔：

~~~text
t-5s, t-4s, t-3s, t-2s, t-1s, t
~~~

每个样本应同时返回：

- 图像：[camera, time, channel, height, width]；
- state：[time, state_dim]；
- history_valid：历史位置是否是真实观测；
- history_dt：相对于当前帧的真实时间差。

历史按oldest-first排列，当前帧固定在最后，当前帧的dt为0。episode开头不足的槽位可以复制首帧保持固定shape，但必须由valid mask屏蔽，不能把复制帧当成真实历史。不同相机分别沿时间维处理，不直接混合相机和时间。

### 3.2 v0：带gate的短时记忆

v0在Qwen视觉塔的第3、7、11、15层加入同一空间patch的因果时间注意力。每个head使用零初始化gate，将时间分支作为对原始value的增量：

~~~text
V_used = V + tanh(g) × (V_temporal - V)
~~~

初始化时g为0，所以K=1时可以退化到current-only路径；训练后每个head再决定是否使用历史。

v0在视觉塔中段丢弃过去帧，只保留当前帧token送入后续VLM。因此：

- VLM侧视觉token数量不增加；
- 视觉塔前半段计算量仍随历史帧数增加；
- 历史帧不会进入文本主干；
- v0没有动态选帧、语言摘要、proprio history或在线ring buffer。

原记录中的v0训练配置冻结原始vision tower，只训练Action Expert、VLM非视觉部分和temporal gate，并使用flow loss。它适合作为低侵入、checkpoint-safe的短时记忆基线，但可学习的时序容量有限。

### 3.3 v1：Time-Then-Space

v1取消v0的零初始化per-head gate，改为在视觉塔中直接学习时间融合：

1. 对同一空间位置的K帧做causal temporal attention；
2. 用融合后的value做正常spatial attention；
3. 在第20个vision block后删除过去帧；
4. 后4层只处理融合历史后的当前帧。

v1的Temporal Block位于第4、8、12、16、20个Vision Block。历史只能读取自身和更早帧，当前帧可以读取全部有效历史。连续history_dt在Q/K投影前提供时间信息，当前帧时间编码为0。

v1的训练随机化包括：

- current-only probability：约25%的样本只保留当前帧；
- history dropout：随机屏蔽部分历史槽位；
- 同一样本的不同相机共享history mask；
- eval时关闭随机dropout。

History dropout只改变Temporal Attention的可见性，不会跳过历史帧的vision计算，因此主要用于鲁棒性训练，不能当作加速方法。v1解冻Vision Encoder，使QKV、MLP和norm可以适应机器人视频；这会增加灾难性遗忘和训练成本，需要配合较小学习率及current-only样本。

v0和v1是两个不同实现版本，不能混合描述。具体启用状态以对应代码commit和训练配置为准。

## 4. 为什么不把历史直接拼成普通输入

有三条视觉路线：

| 路线 | 方式 | 优点 | 局限 | 用途 |
|---|---|---|---|---|
| A 多image | 每帧独立编码后拼接 | 改动小，适合验证历史是否有效 | VLM token和延迟随K线性增加 | correctness对照 |
| B Qwen video | 每个相机以video输入 | 复用Qwen原生视频路径和预训练 | 不等同于MEM，token仍随K增加 | P1基线 |
| C 时间压缩 | 在视觉塔或adapter中压缩历史 | 输出固定预算，适合部署 | 需要新增训练模块和严格的K=1兼容测试 | P2生产方向 |

Qwen3.5视觉patch embedding带有temporal kernel。直接把单帧输入改成video，可能改变patch数量、位置编码和checkpoint输入分布。因此Qwen native video应作为独立基线；如果追求MEM式的same-patch causal temporal attention，需要单独实现视觉forward并验证K=1的feature/logits parity。

## 5. Proprio Memory

短时视觉和机器人state解决的问题不同：

- 视觉历史适合表示遮挡、物体位移和接触前后的外观变化；
- state history适合表示关节变化、末端速度和夹爪状态。

推荐将过去K-1个state逐帧归一化，再投影为连续memory token，并加入history_dt和history_valid：

~~~text
STATE_CURRENT = state_history[:, -1, :]
STATE_HISTORY = state_history[:, :-1, :]
~~~

当前state继续走原有Action Expert condition。历史state只作为额外memory输入，不改变当前state接口。episode起点必须使用valid mask，不能依赖重复首帧或默认零值推断历史是否存在。

## 6. 从短时记忆到长期记忆

### 6.1 记忆层级

| 层级 | 时间范围 | 保存内容 | 典型用途 | 优先级 |
|---|---|---|---|---|
| 短时感知记忆 | 0.1至10秒 | 稠密图像和state变化 | 遮挡、滑落、运动方向 | P1 |
| 稀疏事件记忆 | 10秒至数分钟 | 关键帧和事件证据 | 物体已放置、容器已处理 | P3 |
| 语义任务记忆 | 数分钟以上 | 子任务、成败和当前阶段摘要 | 长流程规划，避免重复步骤 | P4 |
| 跨episode参数记忆 | 天至月 | 技能和世界知识更新 | lifelong learning | 暂不覆盖 |

### 6.2 完整MEM的高低层接口

论文中的完整MEM包含低层动作策略和高层语言记忆策略：

- 低层读取短时稠密视觉、proprio和当前子任务，持续输出action chunk；
- 高层读取当前观察、任务目标和旧摘要，生成下一子任务及新摘要；
- 高层按subtask或事件更新，不应在每个相机帧都生成文本。

当前MagicVLA已有低层Action Expert，但没有独立的高层策略、递归摘要m_t、动态子任务l_t和成功/失败监督。因此，加入固定历史帧不能直接称为完整MEM。

若要实现长记忆，需要准备：

1. 子任务边界或高层调用时刻；
2. 子任务成功、失败、纠正和重复执行记录；
3. 离线摘要teacher，用于生成可覆盖、可压缩的文本记忆；
4. 推理侧按episode维护m_t，并在reset时清空。

### 6.3 动态事件记忆

动态选帧不应只按图像变化或novelty。关键帧应同时考虑：

- 未来动作是否需要该视觉证据；
- 夹爪、接触和物体状态是否发生变化；
- 是否进入新的任务phase；
- 与已有memory的冗余程度；
- 策略不确定性、失败或重试信号。

建议的最小buffer为“起始锚点+最多4个高位相机关键帧”。使用固定slot、valid mask、时间戳和score，加入按秒定义的cooldown。训练先使用离线teacher keyframe，再逐渐切换到selector预测，避免部署时误差累积。

训练时不要把可变episode bank放在模型或collator的全局字典中。当前DataLoader会shuffle，DDP的rank和worker也不共享状态，容易产生跨episode泄漏。训练优先使用每个anchor显式带齐历史；在线buffer只由policy wrapper或environment管理。

## 7. 训练策略与梯度边界

### 7.1 Knowledge Insulation

启用Knowledge Insulation时，flow gradient可能在VLM prefix进入Action Expert前被detach。若memory只拼接到VLM prefix，而没有独立辅助损失或expert直连分支，memory参数可能没有有效梯度。

推荐的梯度路径：

- 通用VLM参数继续由FAST CE或VLM co-train更新；
- vision backbone初期冻结或使用较小学习率；
- memory adapter、proprio encoder、selector和expert-memory projection始终列入trainable allowlist；
- flow loss训练Action Expert及其memory分支；
- temporal auxiliary loss训练VLM可见的memory token；
- selector单独使用BCE、focal或ranking loss。

每次训练都应检查memory参数反向传播后的梯度是否非零，并记录各loss的有效mask比例。

### 7.2 训练课程

**Phase 0：数据和诊断集**

- 加入history_valid、真实history_dt和current/history state split；
- 构造当前图像近似、但历史和正确动作不同的成对样本；
- 覆盖自遮挡、物体离开视野、抓取失败后重试和多物体选择；
- 严格按episode划分训练和评测，避免轨迹泄漏。

**Phase 1：短时基线**

- K=6，间隔约1秒；
- 约25%的样本使用current-only；
- 其余样本随机丢弃部分历史，并记录真实dt；
- 保持FAST/action flow，必要时增加过去state或past-token辅助预测。

**Phase 2：固定预算压缩**

- 将历史压缩为8、16或32个memory token；
- 当前帧token保持不变；
- adapter或gate零初始化；
- 训练current-only任务上的feature/output distillation；
- 比较不同K、时间间隔和memory token数量。

**Phase 3：动态事件buffer**

- 离线生成event/keyframe标签；
- 先训练策略消费teacher keyframe；
- 再训练selector并模拟写入延迟；
- 最后使用episode-sequential rollout验证reset和在线更新。

**Phase 4：语义长记忆**

只在有明确子任务、成败标签和摘要调用点时进行。摘要需要固定长度、可覆盖和可过期，避免错误事实无限累积。

## 8. 目标架构

~~~text
episode observations
  ├─ dense recent window: K=6, all cameras, about 1 s stride
  │    └─ Qwen video or temporal compressor
  ├─ sparse event buffer: anchor + selected keyframes
  │    └─ fixed slots + valid + dt + task-aware retrieval
  └─ proprio history: K-1 states + valid + dt
                 ↓
          fixed-budget memory tokens
          ├─ VLM/temporal objective
          └─ trainable memory K/V for Action Expert
                 ↓
      current state + noisy action chunk → Action Expert
~~~

运行时MemoryState应属于policy wrapper或environment，而不是全局模型：

~~~text
episode_id
dense_frames, dense_states, dense_timestamps, dense_valid
keyframes, keyframe_timestamps, keyframe_scores, keyframe_valid
~~~

每个并行环境拥有独立slot。timestamp必须单调，episode切换必须reset，空slot用mask填充。训练checkpoint默认不保存在线episode memory，恢复rollout时再明确是否恢复。

## 9. 实验矩阵与验收标准

### 9.1 最小消融

| ID | 视觉历史 | proprio | 动态keyframe | 目的 |
|---|---|---|---|---|
| B0 | current-only | current | 无 | 原checkpoint基线 |
| B1 | 无 | K=6 | 无 | 测量proprio history单独贡献 |
| B2 | K=3多image | K=3 | 无 | 检查数据和消费链路 |
| B3 | K=6 Qwen video | K=6 | 无 | P1主基线 |
| B4 | K=6 fixed adapter | K=6 | 无 | 验证固定预算压缩 |
| B5 | B4 | K=6 | teacher keyframe | 验证策略是否消费事件记忆 |
| B6 | B4 | K=6 | predicted keyframe | 验证部署闭环 |

时间和容量可以先比较K={1,3,6,12}、间隔{0.1,0.5,1.0,3.0}秒、memory token数{8,16,32}，不必做完整笛卡尔积。

### 9.2 指标

能力指标：

- closed-loop task success和subtask progress；
- memory-specific paired accuracy；
- 遮挡后目标保持、失败恢复和避免重复步骤；
- 正确历史、打乱历史、错误历史和全mask历史之间的性能差异；
- 非memory任务的性能保持率。

效率与正确性指标：

- 实际视觉token和prefix token；
- 峰值显存、吞吐和p50/p95推理延迟；
- 控制频率和action staleness；
- future leakage、episode reset和跨环境污染；
- selector选择结果与事件的对应关系。

推进标准：

- P1→P2：正确历史明显优于shuffled history，且memory任务优于B0；
- P2→P3：固定压缩接近或超过P1，同时满足延迟和显存预算；
- teacher→predicted：预测keyframe与teacher keyframe的策略差距可控；
- 上线前：非memory技能无明显回退，跨episode泄漏为零。

## 10. 代码改造清单

### 数据与processor

- 在[lerobot_history.py](/pfs/user/magicvla/src/data/lerobot_history.py:8)统一历史索引、oldest-first、valid和dt；
- processor分离STATE_CURRENT和STATE_HISTORY，禁止展平时间维；
- 相机缺失mask扩展到camera×time；
- 对K、时间顺序、当前帧位置和chunk边界增加断言。

### 模型与配置

- 在[configuration_magicvla_base.py](/pfs/user/magicvla/src/models/magicvla_base/configuration_magicvla_base.py)区分none、multi-image、qwen-video、adapter和mem-vision；
- 在[modeling_magicvla_base.py](/pfs/user/magicvla/src/models/magicvla_base/modeling_magicvla_base.py:1783)明确历史图像是丢弃、拼接还是进入video；
- 为memory模块设置独立的trainable allowlist和optimizer group；
- 在KI模式下明确detach边界，并检查memory梯度；
- 保存memory schema、动作版本、归一化统计和配置快照。

### 推理与测试

- 新增MemoryState、reset、observe和build_model_input接口；
- 支持多环境slot隔离和时间倒退检查；
- 测试K=1 parity、全历史mask、future leakage、DDP确定性和checkpoint加载；
- 动态selector增加cooldown、pending/commit和debug trace。

## 11. 当前限制与待补充方向

- v0/v1主要解决秒级视觉历史，不能称为长期记忆；
- 固定间隔对抓取接触、夹爪滑动等高速事件可能过稀；
- 当前方案对proprio history、失败纠正和长流程阶段信息覆盖不足；
- Qwen native video不等同于MEM式causal temporal encoder；
- 动态buffer需要新的数据组织和在线推理接口；
- TTT和RoboTTT方向尚未完成代码核验，暂不纳入主训练方案。

TTT方向的基本设想是在Action Expert中加入小型可更新模块，把模块参数作为fast weights，在训练和推理时用自监督损失持续更新。当前记录的候选目标是Key-Value Binding，但还缺少稳定的更新频率、reset策略、计算预算和动作收益验证，后续应单独建实验。

## 12. 参考资料

- [MEM：Physical Intelligence论文记录](../Paper/260717_MEM_PhysicalIntelligence_2026/MEM_PhysicalIntelligence_2026.md)
- [MagicVLA视觉记忆完整记录](../MagicAtom/99_详细记录/01_MagicVLA_视觉记忆_完整记录.md)
- MemoryVLA
- EventVLA
- Learning Long-Context Diffusion Policies via Past-Token Prediction
- [RoboTTT](https://arxiv.org/pdf/2607.15275)

论文中的性能数字属于作者报告，不能直接外推为MagicVLA收益。具体模型结构、数据比例、延迟和显存需要在实际checkpoint、processor和部署GPU上重新测量。

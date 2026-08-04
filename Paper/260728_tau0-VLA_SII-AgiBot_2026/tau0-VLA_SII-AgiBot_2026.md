# tau0-VLA：用世界模型把长程 VLA 的“下一步子任务”变成可扩展推理

> 原标题：τ0-VLA: a Hierarchical Robot Foundation Model with World-Model-Guided Test-Time Computation  
> 作者：Xiaowei Cai, Yunuo Cai, Bingao Chen, Jingxiao Chen, Zhi Chen, Siyuan Feng, Tengyu Hou, Jingshun Huang, Han Jiang, Runkun Ju, Dong Li, Mingxiang Li, Shaowei Li, Xinchen Li, Yifan Li, Yi Liu, Zhongyuan Liu, Jianlan Luo, Junwen Miao, Ruiqi Ni, Buqing Nie, Mingjie Pan, Xinlin Ren, Jianheng Song, Jiaxu Wang, Peiqi Wang, Sen Wang, Xiaoyan Wang, Dafeng Wei, Dongming Wu, Pengwei Xie, Pu Yang, Hangjian Ye, Xiangyu Yue, Jinyu Zhang, Qinglin Zhang, Xueyong Zhao, Yue Zhou  
> 机构：Shanghai Innovation Institute, Agibot Finch, The Chinese University of Hong Kong  
> 发表：Technical report / project release, 2026-07-27；PDF metadata creation date 为 2026-07-29；远端 PDF Last-Modified 为 2026-07-28  
> 链接：https://tau0-vla.github.io/tau0-vla.pdf  
> 项目页：https://tau0-vla.github.io/  
> 开源：https://github.com/sii-research/tau-0-vla  
> 权重：https://huggingface.co/sii-research/tau-0-vla  

---

## 一句话说清楚

tau0-VLA 不是只把低层 VLA 做大，而是把长程任务里的“下一条语言子任务应该是什么”抽出来，交给一个带 execution memory 的高层 VLM；当高层不确定时，它用 proposal model 生成候选子任务，用 world model 想象执行后的 head-camera 图像，用 value model 打分，再让 reflective model 结合候选分支生成最终子任务，最后由低层 VLA 输出连续 action chunk。

---

## 一、研究背景与动机

长程家庭任务的难点不只是“抓得准”，而是“知道现在该做哪一步”。清理房间、做奶茶、做番茄炒蛋这类任务持续几分钟，包含十几到二十几步，机器人需要记住哪些步骤已经完成、哪些失败需要重试、哪些状态变化肉眼不明显。

已有 VLA 或 hierarchical VLA 通常有两类问题：

1. **直接执行整条任务指令**：低层 policy 一直拿完整 instruction 做 action generation，需要自己隐式判断当前阶段。任务越长，stage tracking 越容易漂。
2. **固定算力的高层规划**：高层 VLM 一次 forward 给出下一步子任务，没有显式比较多个候选，更没有在执行前评估候选会把环境变成什么样。
3. **world model 多用于执行后的子目标生成或动作选择**：一些方法已经用未来图像辅助低层控制，但 tau0-VLA 把 visual consequence 放到“commit 下一条子任务之前”，用于高层搜索和反思。

这篇论文的目标是：在低层 VLA 保持统一跨本体 action interface 的前提下，让高层 next-subtask prediction 支持 test-time computation，即困难决策可以花更多算力，简单决策直接通过。

---

## 二、核心贡献

1. **提出一个双时间尺度的 hierarchical VLA 系统**  
   高层 policy 负责 execution memory、下一子任务生成和 test-time search；低层 VLA 负责把选中的子任务落到连续动作。两层之间的接口是自然语言 subtask。

2. **把 next-subtask generation 变成 world-model-guided search**  
   在不确定时，高层不是只采样一个答案，而是执行 propose -> predict -> evaluate -> reflect：候选子任务先被 world model 映射到预测终态图像，再由 value model 评价任务进展，beam search 保留高分分支。

3. **低层 VLA 使用统一 40D state/action 表示支持多机器人本体**  
   40D slot 同时覆盖双臂 EEF、双臂关节、夹爪、腰部和移动底盘。不同机器人通过 mask 激活自己有效的维度，避免为每个本体单独加 output head。

4. **给出较完整的数据工程链路**  
   低层使用 40,115 小时异构真实机器人数据并混入多模态数据；高层通过 L1/L2/L3 标注、Gemma4-31B-it 预标注、keyframe 抽取、memory perturbation 和过滤构造训练样本。Appendix 报告过滤后保留 40.4M clean samples。

---

## 三、方法原理

### 3.1 整体框架

系统每个 logical inference step 做两件事：

1. 高层 policy `mu` 接收任务指令、执行记忆、上一条子任务和当前多视角观测，输出更新后的 memory `M_t` 和当前子任务 `z_t*`。
2. 低层 policy `pi_theta` 接收多视角 RGB、proprio state、子任务语言和控制元数据，输出未来 `H` 步 action chunk。

形式上，低层 VLA 是：

```text
a_{t:t+H-1} = pi_theta(o_t, s_t, c_t, eta)
```

其中 `o_t` 是 multi-view observation，`s_t` 是 proprioceptive state，`c_t` 在 direct execution 中等于完整任务指令，在 hierarchical execution 中等于高层生成的子任务 `z_t*`，`eta` 是 robot type / control mode / whole-body control 等文本元数据。

高层 context 是：

```text
h_t = (l, M_{t-1}, z_{t-1}*, o_t)
```

proposal model 先生成直接建议：

```text
(z_t^dir, M_t) = P(h_t)
```

如果 token-confidence router 判断模型足够自信，就走 fast route，直接令 `z_t* = z_t^dir`。如果不够自信，就进入 test-time computation。

### 3.2 关键技术细节

**1. 高层由四个模型组成**

| 组件 | 初始化/类型 | 输入 | 输出 | 作用 |
|---|---|---|---|---|
| Proposal model `P` | Qwen3.5-9B 初始化的 robot-pretrained VLM | task instruction、multi-view observation、memory、previous subtask | `<think>`、更新后的 `<memory>`、候选 `<subtask>` | 直接提出下一子任务，并维护执行记忆 |
| World model `W` | Step1X-Edit 初始化 | head-camera RGB + candidate subtask | predicted terminal head-camera image | 想象执行该子任务后的视觉结果 |
| Value model `V` | Qwen3.5-9B 系列 VLM | global instruction + candidate subtask + predicted image | 5 档质量等级映射到 `[0.05, 0.95]` 标量 | 判断候选是否推动任务进展 |
| Reflective model `F` | Qwen3.5-9B 系列 VLM | real context + retained branch summaries | final subtask `z_t*` | 基于搜索结果反思并提交最终子任务 |

**2. TTC 的搜索空间是 language subtask，不是 action**

Action-level search 可以处理局部控制不确定性，但搜索树会很深，而且很难覆盖数分钟任务。Language-only planning 的抽象层级够高，但不接地气。tau0-VLA 选择 subtask 作为搜索单元：它比 action chunk 稀疏，也足以让 world model 预测出有意义的视觉后果。

TTC 的核心过程：

```text
root context h_t
  -> P 采样 N 个候选子任务
  -> W 为每个候选预测 terminal head-camera image
  -> V 为每个候选打分
  -> beam search 保留 top-B 分支
  -> 重复到深度 D
  -> F 基于 retained branches 生成最终子任务
```

论文 Figure 1/2 的示例里展示了 `N = 3` 的单层 expansion；真实推理可通过 branching factor `N`、beam width `B`、depth `D` 调整预算。

**3. Execution memory 是可修正状态，不是只追加日志**

高层模型不仅生成下一步，还要维护 `<memory>`。为了让它能处理真实部署中的错误，训练数据里会人为扰动 input memory：

- memory 落后于视觉状态：需要 catch up。
- memory 过度乐观：需要 rollback。
- 执行失败但 memory 未记录：需要 error-think 和 recovery step。

这点很关键，因为长程任务里一些动作结果不容易从当前图像判断。例如番茄炒蛋里的“加盐”几乎不改变视觉外观，单看 observation 很难知道是否已经做过，显式 memory 可以避免重复加盐或漏加盐。

**4. 低层 VLA：Qwen3.5-2B + MoT action expert + flow matching**

低层 policy 使用 Qwen3.5-2B 视觉语言骨干，接 Mixture-of-Transformers（MoT）action expert。Action token 和 backbone token 在 full-attention layer 中交互，但由不同 Transformer stream 处理。训练目标是 conditional flow matching，从噪声 action chunk 积分到可执行 action chunk。

推理时：

- action chunk shape 是 `(H, 40)`。
- 论文报告 `H = 30`。
- flow-matching inference 使用 10 个 uniform Euler updates。
- 部署控制 loop 约 `30 Hz`，高层后台约每 `1 s` 刷新一次 subtask cache。

### 3.3 训练与优化

**低层 policy 三阶段训练**

| 阶段 | 名称 | 数据 | 训练策略 | 目的 |
|---|---|---|---|---|
| Stage 1 | Knowledge-isolated co-training | 机器人 action data + 多模态数据 | action-loss gradient 在 backbone 接口处隔离 | 先让 action expert 学控制，同时保护 VLM 语义能力 |
| Stage 2 | End-to-end co-training | 同上 | 取消 KI，全模型端到端优化 | 让视觉、语言、动作表示进一步耦合 |
| Stage 3 | Task-specific adaptation | 目标任务小规模 demos | 针对目标 embodiment、相机、物体配置、success criteria 微调 | 获得每个部署任务的可执行 checkpoint |

**高层 policy 训练**

Proposal、value、reflective model 都从同一个 robot-pretrained Qwen3.5-9B VLM checkpoint 独立 finetune；world model 从 Step1X-Edit 初始化并单独训练。

| 模型 | 监督信号 | 关键点 |
|---|---|---|
| Proposal `P` | 对齐或扰动后的 memory、当前 observation、ground-truth next subtask | 学会更新 memory 并生成下一子任务 |
| World `W` | subtask segment 的 start/end head-camera RGB | 学会给定候选子任务预测完成后的终态图像 |
| Value `V` | offline rollout 中的候选 subtask + predicted image + ground-truth next step | 多选 VQA 形式，5 档质量映射到标量 |
| Reflective `F` | retained branches、predicted future states、scores、ground-truth next subtask | 学会不盲目复制第一候选，而是结合分支后果生成最终子任务 |

### 3.4 数据使用与维度追踪

#### 3.4.1 数据清单

| 数据源 | 规模 | 样本单位 | 模态/字段 | 关键维度 | 标签/动作 | 用途阶段 | 处理方式 |
|---|---:|---|---|---|---|---|---|
| 内部低层机器人数据 | 约 23.4K 小时 | trajectory / action chunk | multi-view RGB, proprio state, language, action | state/action 统一映射到 40D；action chunk `(30, 40)` | 连续控制命令 | 低层 Stage 1/2 预训练，Stage 3 任务适配 | 40D scatter、mask、relative action、normalization |
| AGIBOT G1 内部数据 | 21.9K 小时 | trajectory | G1 多视角 RGB、proprio、双臂/移动控制 | 统一 40D；有效维度按 route mask | joint / EEF / gripper / base 等 | 低层训练主力数据 | native layout -> dim registry -> 40D |
| AGIBOT G2 内部数据 | 585 小时 | trajectory | 论文未逐字段展开 | 统一 40D | native robot commands | 低层训练 | 同一采样预处理链路 |
| ARX AC One 内部数据 | 578 小时 | trajectory | wrist cameras + fixed central camera, proprio | 双 6-DoF arms + grippers 映射到 40D | native commands | 低层训练与跨本体评估 | 机器人 adapter 映射 |
| Franka 内部数据 | 347 小时 | trajectory | 3 个 RGB cameras, proprio | 双 7-DoF arms + grippers 映射到 40D | native commands | 低层训练与 Tidy Makeup Table 评估 | 机器人 adapter 映射 |
| 公开机器人数据 | 约 16.7K 小时 | trajectory | public robot dataset modalities | 统一 40D；具体每源维度未报告 | action supervision | 低层训练 | 同一 sampling/preprocessing pipeline |
| open-source UMI 数据 | 9.25K 小时 | UMI-style recording | RGB/video + action/proprio，具体字段未完全展开 | 论文未逐项说明 | contact-rich manipulation skills | 低层训练补充 | 转入统一训练管线 |
| 多模态 co-training 数据 | 未报告 | image/text 或 robot-centric perception sample | instruction following, visual grounding, spatial/depth reasoning | 取决于 VLM 输入；论文未报告统一尺寸 | 多模态监督 | Stage 1/2 保护 VLM 语义/视觉能力 | 与 action data interleave |
| 高层任务/阶段/子任务标注 | 过滤后 40.4M clean samples；保留 88.26% | subtask-boundary sample / keyframe sample | top_head, hand_left, hand_right，同步视角；task/stage/subtask 文本；memory | 高层视觉分辨率未报告；三视角 keyframe | `<think>`, `<memory>`, `<subtask>` | Proposal / reflective / high-level eval | Gemma4-31B-it pre-label，dirty filtering，held-out frames |
| World model start-end frames | 未单独报告 | subtask transition pair | head-camera RGB 起点、head-camera RGB 终点、candidate subtask | 单 head-camera image；分辨率未报告 | terminal image target | World model | 按 subtask temporal range 抽取 start/end |
| Offline simulated rollouts | 未报告 | imagined branch | proposed subtask sequence + predicted images + scores | branch depth `D`，width `B`，factor `N` | value labels / reflective target | Value / reflective model | P 与 W 递归生成，VQA 打分监督 |
| 任务特定 demos | 小规模，未报告具体数量 | task demo | 目标机器人相机、state、action、语言 | 同低层 40D | action chunk | Stage 3 task-specific adaptation | 针对任务、视角、物体配置微调 |

#### 3.4.2 一个低层训练样本长什么样

按开源 data pipeline，一个 unified route 的 batch contract 是：

| key | shape / type | 含义 |
|---|---|---|
| `state` | `float32 (B, 40)` | 当前 proprio / robot state，已按 40D contract scatter 和 normalize |
| `action` | `float32 (B, H, 40)` | 未来 action chunk，论文报告 `H = 30` |
| `state_mask` | `float32 (B, 40)` | 当前样本哪些 state slots 有效 |
| `action_mask` | `float32 (B, 40)` | 哪些 action slots 参与训练 loss；沿 `H` 个 action steps broadcast |
| `images[cam]` | `uint8 (B, height, width, 3)` | HWC RGB；相机名、顺序、数量必须与 config / adapter / deployment 一致 |
| `prompt` | `list[str]` 长度 `B` | language command，可能来自整任务 instruction 或 subtask annotation |

这意味着每个 action supervision 不是单步动作，而是从 anchor frame 开始的未来 `30 x 40` 连续动作块。不同机器人没有独立 action head，而是靠 `action_mask` 激活有效 slots；无效 slots 置零且不计入 flow-matching loss。

#### 3.4.3 40D state/action slot 设计

下面用代码/模型侧常见的 0-index slot 表示；论文 Table IV 用的是 1-index，语义一致。

| slot | 语义 | 表示 |
|---:|---|---|
| `0:3` | left EEF position | XYZ，单位 metre |
| `3:9` | left EEF orientation | Rotation 6D，即旋转矩阵前两列 |
| `9:12` | right EEF position | XYZ，单位 metre |
| `12:18` | right EEF orientation | Rotation 6D |
| `18` | left gripper | 1 个 native opening scalar |
| `19` | right gripper | 1 个 native opening scalar |
| `20:22` | waist | 最多 2 个 adapter-declared values |
| `22:24` | chassis velocity | 最多 2 个 native velocity values |
| `24:32` | left arm joints | 最多 8 个关节，radians |
| `32:40` | right arm joints | 最多 8 个关节，radians |

几个工程细节很重要：

- 关节少于 8 个的机械臂填前面的 slots，剩余 slots mask 为 0。
- EEF 和 arm joint 不会同时表示同一只手臂的 motion：native EEF 存在时激活 EEF slots 并关闭 arm-joint slots；没有 EEF 时用 arm joints 作为 fallback。
- arm joint action 是 `action - current_state` 的关节 delta。
- EEF action 是相对当前 EEF pose 的 position / rotation delta，其中 position delta 在当前 EEF frame 下表达。
- gripper、waist、chassis velocity 在开源文档里说明保持 absolute。
- public v1 serving 只支持 joint-control checkpoints；EEF 可以进入训练/数据管线，但公开 server 不支持 EEF serving。

#### 3.4.4 低层 action 生成与 mask loss

低层 action dimension `d_a = 40`。对每个未来 step `j in {0, ..., H-1}`，有：

```text
a_{t+j}, epsilon_{t+j} in R^40
M in {0,1}^{40 x 40}
```

`M` 是 diagonal action mask。flow path 只在 active channels 上插值和监督：

```text
a_tau = tau * M * epsilon + (1 - tau) * M * a
u = M * (epsilon - a)
```

loss 也只看 active channels，并按 `H * tr(M)` 归一化。推理时从 masked Gaussian noise 开始，用 10 个 Euler steps 从 `tau = 1` 积分到 `tau = 0`，每一步都重新应用 action mask，最后再把 inactive slots 清零。

这个设计的好处是：同一个模型可以同时见到 mobile base、双臂、Franka、ARX、G1 等异构控制；代价是 adapter / registry / mask / norm_stats 任何一个错位都会改变动作语义。

#### 3.4.5 高层数据构造：L1/L2/L3 到 `<think>/<memory>/<subtask>`

高层监督来自已有 task instruction、stage description、executable-subtask annotation、segmented demonstrations 和 videos。论文把 instruction 分为三级：

| 层级 | 来源 | 用途 |
|---|---|---|
| L1 | episode-level high-level instruction | 全局任务目标 |
| L2 | key_frame / subtask_frame 中的 stage/subtask interval | 阶段级结构和过滤 |
| L3 | instruction_segments 的 frame-level subtask instruction | 低层 prompt 或高层 target subtask |

构造步骤：

1. **Pre-labeling**：用 `google/Gemma4-31B-it` 根据 carried memory、previous subtask 和当前图像生成 `<think>` 与 `<memory>`；target `<subtask>` 复用 executable-subtask annotation。
2. **Keyframe extraction**：按 subtask temporal range 用 ffmpeg 抽取三路同步视角：`top_head`、`hand_left`、`hand_right`。
3. **VQA assembly**：把同步图像、生成字段和子任务标注组装成高层 VQA-style training examples。

清洗流程包括：

- cross-task contamination filter：避免把另一个任务里的对象/约束污染到当前任务。
- empty-`<subtask>` detector：移除空子任务样本。
- memory-contamination filter：避免错误 memory 继续污染下游。
- 三层 dirty policy：dirty ratio 超过 90% 的 task 整体排除，10%-90% 的 task 按 episode 过滤，低于 10% 的 task 只删具体坏样本。

最终报告：丢弃 `11.74%` episodes，保留 `40.4M` clean samples，占 `88.26%`。

#### 3.4.6 Memory perturbation 的样本族

| Family | 采样位置 | Input -> target memory | Target subtask | 对应部署错误 | Mix |
|---|---|---|---|---|---:|
| within-subtask | segment `n` 内任意位置 | `M_n -> M_n` | segment `n` | 正常执行 | 58% |
| transition | segment `n` 尾部 | `M_n -> M_{n+1}` | segment `n+1` | 完成后进入下一子任务 | 15% |
| catch-up | segment `n` 头部 | `M_{n-1} -> M_n` | segment `n` | memory 落后于视觉状态 | 10% |
| rollback | segment `n` 后段 | `M_{n+1...n+3} -> M_n` | retry segment `n` | memory 过度乐观 | 12% |
| error-think | annotated failure frame | `M_n -> type-dependent` | recovery step | 未察觉执行失败 | 5% |

Rollback instances 被限制在 `10%-15%`，避免模型过度不信任正确 memory。论文还提到对 L1/L2 task/stage instructions 做六维 visually grounded augmentation，以增强指令多样性和 zero-shot steerability。

#### 3.4.7 LeRobot / adapter / deployment contract

开源仓库要求数据是 LeRobot v3.0。核心文件包括 `meta/info.json`、`meta/stats.json`、`tasks.parquet`、`data/*.parquet` 和 `videos/{camera}/...mp4`。

关键约束：

- `observation.state` 和 `action` 是 flat vectors，必须带 `field_descriptions`。
- `observation.images.<camera>` 是 `[H, W, 3]` video stream；具体分辨率由数据和 config 决定。
- unified route 需要 `dim_registry.json` 把 native columns scatter 到 40D slots。
- camera key、left/right 语义、顺序和数量必须在 adapter、YAML、deployment 中完全一致。
- 常见图像变换包括 `ColorJitter` 和 `ResizeWithPad(224, 224)`，但论文没有说明主训练实际统一分辨率。
- normalization statistics 是 per-embodiment 40D；inactive slots 不参与估计。
- 训练和评估的 frame filter 要匹配，例如 L3 policy 需要保证 action horizon tail 仍在 instruction segment 内。

#### 3.4.8 维度快照

```text
High-level observation: multi-view RGB；keyframe 构造使用 top_head / hand_left / hand_right
World model input: single head-camera RGB + candidate subtask
World model output: predicted terminal head-camera RGB
Language: task instruction / previous subtask / generated subtask / <think> / <memory>
Low-level state: R^40
Low-level action per step: R^40
Action chunk: R^{30 x 40}
Masks: state_mask R^40, action_mask R^40
Flow inference: 10 Euler updates
Control loop: 约 30 Hz
High-level cache refresh: 约每 1 s
Evaluation trials: 每个 method-task entry 10 次真实机器人 trial
```

---

## 四、实验与结果

### 4.1 实验设置

**机器人平台**

| 平台 | 本体 | 传感器 | 用途 |
|---|---|---|---|
| AGIBOT G1 | wheeled humanoid，omnidirectional base，双 7-DoF arms，parallel-jaw grippers | head-mounted RGB-D / fisheye cameras，每只手腕一个 camera | 四个主要长程任务 |
| ARX AC One | 双 6-DoF X5 arms，parallel-jaw grippers | wrist cameras + fixed central camera | Book Organization、Collect Laundry |
| Bimanual Franka Research 3 | 双 7-DoF torque-controlled arms，自定义 3D printed grippers | 3 个 RGB cameras | Tidy Makeup Table |

**任务**

| 任务 | 步数 | 平台/性质 | 典型时长 | 评估重点 |
|---|---:|---|---|---|
| Clean Room | 25 | mobile manipulation | 约 8 min | 房间切换、进度记忆、物品整理 |
| Prepare Ingredients | 14 | mobile manipulation | 约 4 min | 冰箱/食材/工具操作 |
| Tomato and Egg Stir Fry | 22 | mobile manipulation / cooking | 约 10 min | 烹饪顺序、不可见状态，如加盐 |
| Make Milk Tea | 13 | fixed manipulation | 约 3 min | 顺序装配、盖盖/插吸管等接触丰富动作 |
| Collect Laundry | 5 | ARX mobile manipulation | 约 1 min | 跨本体低层执行 |
| Tidy Makeup Table | 2/2/4 | bimanual Franka | 约 30 s | 指令条件下的对象/手臂/顺序选择 |
| Book Organization | 3 | ARX bimanual | 约 1 min | in-domain/OOD high-level next-subtask prediction |

物理机器人实验报告 success rate（SR）和 progress。每个 method-task entry 都是 10 次独立真实机器人 trial。Progress 按 prerequisite graph 上的 milestones 计分，首次完成给 1，失败后重试完成或部分完成可给 0.5，跳过必需 milestone 会阻断其后继。

### 4.2 主要结果

**长程任务：hierarchical Plan Once 明显优于 direct execution**

| Method | Clean Room SR / Progress | Prepare Ingredients SR / Progress | Stir Fry SR / Progress | Milk Tea SR / Progress | Avg. SR / Progress |
|---|---:|---:|---:|---:|---:|
| GR00T N1.7 | 0/10 / 59.80% | 1/10 / 68.57% | 0/10 / 24.32% | 0/10 / 28.46% | 2.50% / 45.29% |
| LingBot-VLA | 0/10 / 66.60% | 0/10 / 35.00% | 0/10 / 12.27% | 0/10 / 63.85% | 0.00% / 44.43% |
| pi0.5 | 4/10 / 86.20% | 2/10 / 73.93% | 0/10 / 49.77% | 3/10 / 82.31% | 22.50% / 73.05% |
| tau0-VLA direct | 4/10 / 92.80% | 2/10 / 66.43% | 0/10 / 65.00% | 5/10 / 96.15% | 27.50% / 80.10% |
| tau0-VLA hierarchical Plan Once | 5/10 / 94.80% | 4/10 / 82.86% | 4/10 / 81.82% | 5/10 / 91.92% | 45.00% / 87.85% |

最强信号在 Tomato and Egg Stir Fry：direct tau0-VLA 是 `0/10`，hierarchical Plan Once 到 `4/10`。这说明提升不只是低层控制能力，而是高层子任务选择和记忆对长程顺序很关键。

**跨本体 direct execution：低层 VLA 也有强基线能力**

| Method | Collect Laundry | Cotton Pad | Eyelash Curler | Makeup Puff |
|---|---:|---:|---:|---:|
| GR00T N1.7 | 4/10 / 76.00% | 10/10 / 87.50% | 8/10 / 77.50% | 7/10 / 52.50% |
| LingBot-VLA | 2/10 / 35.00% | 9/10 / 67.50% | 3/10 / 22.50% | 3/10 / 33.75% |
| pi0.5 | 9/10 / 88.00% | 9/10 / 85.00% | 8/10 / 85.00% | 7/10 / 73.75% |
| tau0-VLA | 10/10 / 97.00% | 10/10 / 95.00% | 9/10 / 92.50% | 10/10 / 95.00% |

这些任务较短，不使用高层 decomposition / memory / TTC，所以更像是在测低层 40D unified VLA 的执行和指令跟随能力。

**TTC：更多 test-time compute 提升 next-subtask accuracy**

| Setting | Plan Once | Best-of-N | TTC |
|---|---:|---:|---:|
| Make Milk Tea | 64.7% | 70.0% | 87.3% |
| Book Organization In-Domain | 66.0% | 83.0% | 88.0% |
| Book Organization OOD | 50.0% | 57.5% | 74.0% |
| Clean Room | 72.0% | 74.0% | 87.0% |

OOD Book Organization 是最干净的证据：初始书本排列没在训练里出现过，Plan Once 只有 `50.0%`，TTC 到 `74.0%`。这支持作者的核心论点：把候选子任务先映射成视觉后果，再进行 value-guided branch selection，比一次性 VLM 预测更稳。

**TTC 在真实闭环中也提升 success**

| Method | Make Milk Tea | Book Organization | Clean Room |
|---|---:|---:|---:|
| Plan Once | 5/10 / 91.92% | 6/10 / 66.67% | 5/10 / 94.80% |
| TTC | 7/10 / 95.38% | 9/10 / 93.33% | 7/10 / 97.60% |

这里低层 policy 固定，差异来自高层是否使用 TTC。Book Organization 从 `6/10` 到 `9/10`，progress 从 `66.67%` 到 `93.33%`，说明 open-loop next-subtask accuracy 的提升能转化为闭环执行收益。

### 4.3 消融实验

论文主要给了三类机制分析：

1. **Plan Once vs Best-of-N vs TTC**  
   Best-of-N 使用同一个 world/value model 评估多个一跳候选，但没有多步 branch expansion 和 reflective commitment。四个 setting 中 Best-of-N 都有收益，但 TTC 更高，说明多步 consequences 与 reflection 不是装饰项。

2. **Compute vs accuracy 曲线**  
   Figure 5 显示 Make Milk Tea 和 Book Organization 的 accuracy 随 PFLOPs/sample 增加快速上升，之后趋于饱和。结论是中等预算区间性价比最好，继续堆算力边际收益下降。

3. **Execution memory 的价值**  
   项目页报告 revisable memory 带来 `+11.0` percentage points 的 next-subtask accuracy。PDF Appendix 详细解释了 memory perturbation 的五类样本，但提取文本中没有看到完整 memory ablation 表格；因此我把它视为项目页主张而不是表格级结论。

---

## 五、局限性与展望

**论文自身或开源范围中的限制**

1. **公开权重只覆盖低层 VLA policy**  
   Hugging Face model card 明确说明 checkpoint 是 low-level policy initialization；完整 tau0-VLA 还需要高层 proposal/reflection/world/value models，这些不在该 checkpoint 内。

2. **仍然需要 target-specific post-training**  
   论文和 model card 都把公开 checkpoint 定位成任务/本体微调初始化，不是任意机器人任意任务 zero-shot controller。

3. **public v1 serving 只支持 joint-control**  
   EEF 数据可以用于训练和本地数据流程，但公开 server 不支持 EEF action slices。这对想直接部署 EEF policy 的工程用户很关键。

4. **数据细节仍有未报告项**  
   虽然给了总小时数、部分机器人拆分、40D layout、40.4M clean high-level samples，但没有公开每个公开数据集的完整列表、采样配比、图像主训练分辨率、语言 token 长度、各阶段训练步数/学习率等。

**我的推断性限制**

1. **TTC 依赖 world model 的视觉后果质量**  
   如果候选子任务的关键状态变化在 head-camera 图像中不可见，或者 world model 对接触/遮挡/液体/烹饪状态预测不准，value model 的打分会受影响。

2. **高层约每 1 s 刷新，适合子任务级，不适合高速 reactive correction**  
   论文通过异步 cache 保护低层 30 Hz 控制，但高层仍是慢系统；快速碰撞规避、力控失败恢复仍要靠低层 controller 或安全层。

3. **40D 统一接口把复杂性转移到了 adapter contract**  
   跨本体泛化不是免费获得的。camera order、native column indices、mask、normalization、relative restoration、SDK order 任何错位都会造成严重部署问题。

4. **真实任务 trial 数仍偏少**  
   每个 method-task 是 10 次真实机器人 trial，已经很贵，但统计置信度有限。尤其 Stir Fry 这种长程烹饪任务，`4/10` 说明方向有效，同时也说明接触丰富、状态不可见的步骤仍是硬瓶颈。

---

## 六、灵魂三问

1. **它解决了什么问题？**

它解决的是长程 VLA 里“当前该执行哪条子任务”的推理瓶颈。低层 action policy 即使能稳定执行，也可能在错误阶段执行正确动作；tau0-VLA 用 execution memory 和 TTC 让高层在 commit 前比较多个子任务及其视觉后果。

2. **为什么这么做？**

因为 subtask 是一个比 action 更稀疏、比纯语言计划更接地的搜索单元。对每个候选子任务预测 terminal image，再用 value model 评价，可以把 test-time compute 用在真正不确定的高层决策上；而低层仍保留统一 40D action chunk 接口，避免高层搜索直接碰连续控制空间。

3. **什么证据最有说服力？**

最有说服力的是两个互相支撑的结果：OOD Book Organization 的 next-subtask accuracy 从 Plan Once 的 `50.0%` 提到 TTC 的 `74.0%`，真实闭环 Book Organization 又从 `6/10` 提到 `9/10`。这说明 TTC 不只是 open-loop judge 上好看，而是能转化为真实机器人成功率。

---

## 七、个人总结

1. tau0-VLA 的核心思想可以概括为：**低层 VLA 负责执行，高层 VLM 负责可修正记忆和可扩展子任务推理，world model 提供 commit 前的视觉后果证据**。

2. 最大优点是数据和接口做得工程化：40,115 小时低层数据、40D unified state/action、mask loss、LeRobot v3.0 contract、memory perturbation 和高层样本过滤都比较具体。最大弱点是完整 TTC 系统的关键高层组件没有随低层 checkpoint 一起公开，复现完整论文结果会有落差。

3. 对 VLA 算法工程很有启发的一点是：跨本体不一定要靠“更智能的模型自己学会一切”，也可以靠清晰的 slot contract、mask、normalization 和 adapter 把动作空间先对齐；长程能力也不一定靠低层 action horizon 变长，而是可以把高层 subtask decision 做成可 test-time scaling 的闭环。

---

## 八、和近期 VLA 工作的关系

| 论文/系统 | 更关注什么 | 和 tau0-VLA 的差异 |
|---|---|---|
| pi0 / pi0.5 | flow-based action chunking、开放世界泛化 | tau0-VLA 的低层借鉴类似 action-chunk flow 思路，但主贡献在高层 TTC |
| LingBot-VLA | pragmatic VLA foundation model | tau0-VLA 在长程任务中显式引入 execution memory 和 subtask-level TTC |
| X-VLA | soft prompt 跨本体统一 Transformer | X-VLA 侧重跨数据源/本体的 soft prompt alignment；tau0-VLA 更强调 40D contract 和高层 world-model search |
| WorldVLA / WLA-0 | world modeling 与 action/语言联合预测 | tau0-VLA 把 world model 放在高层子任务 commit 前，用于候选分支比较 |
| VLA-Reasoner / RoboMonkey | test-time sampling / verification | tau0-VLA 的搜索对象是 language subtask，而不是低层 action trajectory |


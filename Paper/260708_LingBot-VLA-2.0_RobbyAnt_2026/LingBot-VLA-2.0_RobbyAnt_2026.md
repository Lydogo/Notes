# LingBot-VLA 2.0：把 VLA 从基础模型推向真实机器人应用

> 原标题：From Foundation to Application: Improving VLA Models in Practice  
> 作者：Wei Wu, Fangjing Wang, Fan Lu, He Sun, Shi Liu, Yunnan Wang, Yibin Yan, Yong Wang, Shuailei Ma, Xinyang Wang, Yibin Liu, Shuai Yang, Tianxiang Zhou, Kejia Zhang, Lei Zhou, Cheng Su, Nan Xue, Bin Tan, Han Zhang, Youchao Zhang, Fei Liao, Xing Zhu, Yujun Shen, Kecheng Zheng  
> 机构：RobbyAnt / Ant Digital Technologies 相关团队  
> 发表：arXiv:2607.06403, 2026-07-07  
> 链接：https://arxiv.org/abs/2607.06403  
> 项目页：https://technology.robbyant.com/lingbot-vla-v2  
> 开源：https://github.com/robbyant/lingbot-vla-v2  
> Checkpoints：https://huggingface.co/collections/robbyant/lingbot-vla-v2

---

## 一、研究背景与动机

VLA 基础模型已经在实验室 benchmark 上证明了潜力，但真实部署会遇到更硬的几类问题：

1. **跨任务和跨机器人泛化不够。** 实际机器人数据来自不同平台、不同相机、不同控制空间、不同采集质量；模型不能只在一个双臂平台或少数任务上好看。
2. **动作空间过窄。** 很多 VLA 还围绕单臂/双臂末端执行器建模，而真实机器人需要控制头部、腰部、移动底盘、灵巧手等更多自由度。
3. **动态场景里缺少未来感。** 只看当前帧反应式输出动作，很难处理长程移动操作、物体状态变化和动作后果预测。

LingBot-VLA 2.0 的核心目标不是提出单个漂亮模块，而是把 VLA 系统往真实应用推进：更大更干净的数据、更宽的统一动作空间、更强的时序/未来预测监督。

## 二、核心贡献

1. 构建约 60,000 小时预训练数据：从约 90,000 小时机器人数据中筛出 50,000 小时高质量轨迹，覆盖 20 种机器人配置；另加入约 10,000 小时 egocentric human videos。

2. 设计 55 维统一状态/动作表示：覆盖 arm joint、end-effector pose、gripper、dexterous hand、waist、head、mobile base 等 body-part component，使模型能支持超出标准双臂的全身动作空间。

3. 在 action expert 中引入 token-level sparse MoE：采用 shared expert + routed experts，并用 auxiliary-loss-free load balancing 避免额外负载均衡损失干扰动作学习目标。

4. 提出 dual-query distillation：用 LingBot-Depth 提供几何监督，用 DINO-Video 提供因果时序语义监督，让模型同时学习当前和未来 observation 的表征。

5. 在 GM-100 双臂 benchmark 和两个长程移动操作任务上验证：LingBot-VLA 2.0 在 generalist setting 下整体优于 LingBot-VLA 1.0、π0.5 和 GR00T N1.7。

## 三、方法原理

### 3.1 整体框架

LingBot-VLA 2.0 的框架可以理解成四层：

1. **数据层**：统一清洗机器人轨迹和第一视角人类视频，生成语言标注和标准化动作。
2. **动作表示层**：把不同 embodiment 的控制量映射到一个 55 维 canonical vector。
3. **模型层**：VLM understanding expert + MoE action expert，后者负责从视觉/语言/状态中生成连续动作。
4. **辅助监督层**：dual-query distillation 同时对当前和未来 query 做 depth/video representation 蒸馏。

这篇报告的“工程味”很重：它关心的不是某个模块单点提分，而是整个 VLA 系统从数据、动作、模型到评估的闭环。

### 3.2 关键技术细节

#### 大规模数据处理：先收很多，再用规则和人工筛掉不可信样本

机器人数据从约 90,000 小时筛到 50,000 小时，主要过滤：

- action/state 的 jerk、velocity、acceleration Z-score 异常；
- 超过 95% 时间几乎静止的 episode；
- 视频与状态不对齐：用 URDF 将机器人状态投影到图像平面，再由人工检查 mismatch；
- 模糊、严重遮挡、掉帧、多视角不同步等视频质量问题。

egocentric 数据从约 20,000 小时筛到 10,000 小时，流程包括：

- VLM 预筛掉非第一视角、无交互、无物体、非操作者手部等视频；
- 对有标签数据做时间戳、坐标系和轨迹完整性整理；
- 对无动作标签视频用 egocentric SLAM + hand pose estimation 重建世界坐标系下的手部轨迹；
- 过滤有效手部帧少于 20%、SLAM 不稳定、轨迹突变或违反人体约束的片段。

#### 55 维统一动作/状态向量

统一表示包含：

| 组件 | 维度 |
|---|---:|
| arm joint position | 14 |
| end-effector pose | 14 |
| gripper position | 2 |
| hand joint position | 12 |
| waist position | 4 |
| head position | 2 |
| mobility signal | 3 |
| reserved | 4 |
| 总计 | 55 |

这个设计的关键不是所有机器人都有 55 维动作，而是用 padding/mask 让不同 embodiment 落到同一个 action schema 中。这样模型能在统一接口下同时学习单臂、双臂、半人形、人形、移动平台和灵巧手数据。

#### 自动语言标注

作者使用 Qwen3.6-27B 对 manipulation video 做自动分段和语言标注。每个视频有 video-level instruction，每个 subtask 有：

- 原子动作类别；
- 交互对象；
- 简短 instruction；
- 时间边界。

闭集动作词表包含 15 个 primitive manipulation actions，如 move、pour、push、open、close、fold、wipe、stir、cut、press、attach、detach 等，以及 transit、idle、other 三个辅助标签。

#### MoE action expert

LingBot-VLA 2.0 在 action expert 的 transformer block 中把 FFN 替换为 sparse MoE。每层包含：

- 一个 shared expert：保留通用动作/控制 prior；
- 多个 routed experts：为不同 token 提供专门建模容量；
- token-choice top-K routing：每个 token 独立选择多个 expert；
- sigmoid router：避免 softmax 带来的过强专家竞争；
- auxiliary-loss-free load balancing：用 routing correction bias 调整专家负载，而不是加显式负载均衡 loss。

论文的 scaling experiment 显示，在 active parameter count 匹配时，MoE 版本比 dense 版本有更低 training loss 和 validation action error，说明收益不是单纯来自总参数更多，而是稀疏激活更有效地分配了容量。

#### Dual-query distillation：同时学当前和未来

作者在视觉/文本 token 后附加两个 learnable queries：

- `Q_t`：对齐当前 observation；
- `Q_{t+T}`：预测未来 horizon T 的 observation，T 对应 action chunk size。

两个 teacher：

1. **LingBot-Depth**：提供当前/未来帧的 depth representation，用 L1 loss 蒸馏几何信息。
2. **DINO-Video**：基于 DINOv3 扩展 causal temporal attention 和 3D-RoPE，在 5M 视频 clips 上训练，提供 motion-aware causal video representation，用 L2 loss 蒸馏语义时序信息。

这个设计背后的直觉是：机器人动作不是图像分类。它既需要当前几何，也需要预判动作之后场景会怎么变化。dual-query distillation 就是在 VLA 内部塞进“当前空间 + 未来动态”的辅助学习目标。

### 3.3 训练与优化

预训练数据覆盖 20 种 embodiment，包括 Franka、Flexiv Rizon 4、AgileX、ARX Lift2、UR7e、AgiBot G1、Galbot G1、Galaxea R1Pro/R1Lite、Astribot S1、Unitree G1、Fourier GR-2、AgiBot A2 等。

训练动作目标和归一化也做了专门消融：

- relative joint action 明显优于 absolute joint action；
- EEF 和 joint action 平均成功率接近，但不同任务偏好不同；
- MeanStd normalization 优于 MinMax 和 Q01-Q99；
- L2 loss 平均优于 L1，但 L1 在 contact-rich 的 Squeeze Ketchup 上更稳。

## 四、实验与结果

### 4.1 实验设置

实验分为两类：

1. **GM-100 双臂 benchmark**：选取 9 个任务，在 generalist mixed-training setting 下评估。单个 policy 同时训练所有任务，指标为 progress score 和 success rate。
2. **长程移动操作 benchmark**：两个机器人 embodiment，两个任务：
   - Astribot S1：Sort objects into refrigerator；
   - Cobot Magic-ARX X5：Stove cleaning。

长程任务同时评估 in-domain（ID）和 out-of-distribution（OOD）。OOD 设置包括初始机器人位置前后左右 ±10 cm 扰动；冰箱分类任务还替换 unseen object categories。每个 task-setting 评估 15 次。

### 4.2 主要结果

#### GM-100 双臂 generalist 结果

| 平台 | 方法 | Progress | Success |
|---|---|---:|---:|
| Agilex Cobot Magic | GR00T N1.7 | 36.3 | 17.8 |
| Agilex Cobot Magic | π0.5 | 59.1 | 32.2 |
| Agilex Cobot Magic | LingBot-VLA-1.0 | 58.2 | 30.0 |
| Agilex Cobot Magic | LingBot-VLA-2.0 | **66.2** | **34.4** |
| Galaxea R1 Pro | GR00T N1.7 | 16.4 | 5.6 |
| Galaxea R1 Pro | π0.5 | 27.4 | 8.9 |
| Galaxea R1 Pro | LingBot-VLA-1.0 | 32.7 | **15.6** |
| Galaxea R1 Pro | LingBot-VLA-2.0 | **34.6** | **15.6** |

要点：

- Agilex Cobot Magic 上，LingBot-VLA-2.0 比 LingBot-VLA-1.0 高 8.0 progress / 4.4 success，比 π0.5 高 7.1 / 2.2。
- Galaxea R1 Pro 上，2.0 的 progress 最好，success 与 1.0 持平，但高于 π0.5。
- success rate 绝对值仍不高，说明 GM-100 generalist setting 难度很大，很多任务能部分完成但最后精确放置、释放或收尾失败。

#### 长程移动操作结果

| Embodiment | Task | Setting | LingBot-VLA-2.0 | π0.5 |
|---|---|---|---:|---:|
| Astribot S1 | Sort objects into refrigerator | In-domain | **77.1 / 60.0** | 65.3 / 46.7 |
| Astribot S1 | Sort objects into refrigerator | OOD | **37.0 / 13.3** | 30.3 / 6.7 |
| Cobot Magic-ARX X5 | Stove cleaning | In-domain | **84.3 / 66.7** | 79.9 / 60.0 |
| Cobot Magic-ARX X5 | Stove cleaning | OOD | **67.5 / 40.0** | 62.5 / 33.3 |

表中数字为 progress / success。要点：

- LingBot-VLA-2.0 在两个 embodiment、两个 setting 下都超过 π0.5。
- OOD 下都有明显掉点，尤其冰箱分类任务，因为同时改变初始位姿和目标物体类别。
- Stove cleaning 的 OOD 更稳，可能因为对象和场景结构没有变，主要考验初始位置扰动恢复。

#### DINO-Video teacher 质量

| Model | Params(M) | Composite Human ↑ | Composite Robot ↑ | RoboCOIN ↓ | AgiBotWorld-Beta ↓ |
|---|---:|---:|---:|---:|---:|
| V-JEPA 2 | 303.89 | 80.35 | 70.43 | 0.32 | 0.33 |
| DINOv3 | 303.13 | 76.19 | 69.06 | 0.22 | 0.24 |
| DINO-Video | 303.13 | 80.21 | **71.97** | **0.20** | **0.19** |

DINO-Video 在 4 个 LARYBench 指标中 3 个最好，支持它作为 robotics-aware temporal teacher 的合理性。

### 4.3 消融实验

#### Action target

relative joint action 将平均 success 从 33.7 提到 55.0。作者用动作统计解释：relative target 把 global joint-configuration regression 变成 local motion regression，动作标准差显著缩小；pooled distribution 下 absQpos std 约 0.80，relQpos std 约 0.28。

#### Action space

EEF 和 joint action 平均成功率接近：56.0 vs 55.0，但任务偏好不同。

- Barcode Scan：joint 更好，joint action distribution 更贴近 pooled distribution。
- Squeeze Ketchup：EEF 更好，接触丰富的端点运动更适合笛卡尔空间表示。
- Scoop Rice：EEF 更好，即使 distribution gap 更大，说明物理结构不只由统计分布决定。
- Take Bowl from Microwave：joint 更好，可能因为姿态、可达性和关节配置约束更关键。

结论很务实：不存在绝对最优 action space，要看任务物理结构和数据分布。

#### Normalization

| Normalization | 平均 Success | 解释 |
|---|---:|---|
| MinMax | 47.5 | 把大多数样本压到很窄范围，动作分辨率下降 |
| Q01-Q99 | 47.4 | 比 MinMax 动态范围更大，但仍偏压缩 |
| MeanStd | **55.0** | 保留长尾纠偏动作，动态范围最大 |

#### Loss

L2 平均成功率 55.0，高于 L1 的 46.4。作者认为多数 relQpos 是零附近的小连续修正，L2 更适合拟合高密度区域；但 L1 在 Squeeze Ketchup 上更好，符合接触任务中 heavy-tailed motion 更需要鲁棒性的直觉。

## 五、局限性与展望

论文没有强写 limitation，但从结果看，边界很清楚：

1. 成功率离“可靠部署”还有距离。GM-100 上 progress 明显高于 success，说明模型经常能做到中间步骤，但在最后精确放置、释放、关门、复位等环节失败。

2. 数据和工程成本很高。60,000 小时数据、20 embodiment、人工质量检查、自动标注、视频 teacher、depth teacher，这套系统不是轻量 recipe。

3. 跨 embodiment 仍然困难。Galaxea R1 Pro 的结果明显低于 Agilex Cobot Magic，说明相机视角、运动学、action alignment 和平台动态仍是难点。

4. OOD 泛化还不稳。冰箱长程任务从 ID 60.0 success 掉到 OOD 13.3，说明 unseen object + 初始位姿扰动叠加后，长程恢复能力仍然脆。

5. 模块贡献耦合。数据规模、动作空间、MoE、dual-query distillation 都在变，虽然有部分消融，但很难完全拆清每个模块对最终 benchmark 的独立贡献。

后续值得关注：

- 更系统的跨平台 action representation 选择机制；
- 显式恢复策略或 closed-loop correction，用来解决 progress 高但 success 低的问题；
- 对 dynamic scenes 的更强 causal world model；
- 降低数据处理和人工 QA 成本，让系统 recipe 更容易复现。

## 六、灵魂三问

1. **它解决了什么问题？**

它解决的是 VLA 从实验室基础模型走向真实机器人应用时的“三座山”：数据和 embodiment 泛化不够、动作空间覆盖不够、对未来动态的预测不够。LingBot-VLA 2.0 用大规模高质量数据、统一全身动作表示、MoE action expert 和 dual-query distillation 一起处理这个系统性问题。

2. **为什么这么做？**

因为真实机器人不是单一双臂末端控制器。它有头、腰、底盘、手，有多源异构数据，也有长程任务的未来后果。单纯扩大模型或加 benchmark 分数不够；需要把数据、动作接口和预测式表征一起对齐到部署需求。

3. **什么证据最有说服力？**

最有说服力的是长程移动操作结果：LingBot-VLA-2.0 在 Astribot S1 和 Cobot Magic-ARX X5 上、ID 和 OOD 下都超过 π0.5。这比单纯 GM-100 提分更重要，因为它直接覆盖移动底盘、长程子任务、物体交互和 OOD 初始位置扰动。

## 七、个人总结

1. 核心想法一句话：LingBot-VLA 2.0 是一个偏“应用工程系统”的 VLA recipe，用数据清洗、统一动作空间、MoE 容量分配和未来预测蒸馏来提升真实机器人可用性。

2. 最大优势是系统完整，覆盖从数据到部署评估的关键链路；最大弱点是成本高、复现门槛高，而且 generalist success rate 仍说明真实可靠性远未解决。

3. 对 VLA 研究的启发是：未来的 VLA 竞争不会只是谁的 backbone 更大，而是谁能把 embodiment、action schema、数据质量和 future-aware learning 更好地接成一个工程闭环。

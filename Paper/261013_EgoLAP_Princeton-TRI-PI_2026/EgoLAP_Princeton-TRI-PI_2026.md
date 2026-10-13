# EgoLAP：用物理语言把人类第一人称动作经验迁移给机器人

> 原标题：EgoLAP: Learning from Egocentric Human Data through Language-Action Reasoning  
> 作者：Lihan Zha、Shresth Grover、Tenny Yin、Samuel M. Bateman、Hengkai Pan、Mengchao Zhang、Aykut Onol、Allen Z. Ren、Dhruv Shah、Anirudha Majumdar  
> 机构：Princeton University、Toyota Research Institute、Physical Intelligence  
> 发表：arXiv:2610.08726v1，2026-10-06  
> 来源：[arXiv](https://arxiv.org/abs/2610.08726v1) · [项目页](https://ego-lap.github.io/)  
> 阅读范围：本地 v1 PDF 正文与附录；指标均为作者报告，未独立复现。

---

**一句话说清楚**：EgoLAP 将人手与机器人末端的一秒运动都转换成同一套结构化自然语言动作，再用“为什么当前状态需要这段运动”的局部物理推理作为训练监督；文本目标只训练 VLM，真正执行时由 300M flow-matching action expert 直接生成连续动作，无需解码文本。

**阅读判断**：这篇论文把“语言动作比离散动作 codec 更适合迁移人类示教”的假设做成了相对干净的 matched comparison。最关键结果来自 14 任务零样本仿真和新构型双臂 YAM 真机：语言动作与运动级推理互相增强。限制是动作被压成一秒窗口的粗量化语言，覆盖面仍集中在双臂桌面操作。

## 一、研究背景与动机

人类第一人称视频规模大、操作行为丰富，理论上能补足机器人示教昂贵且窄的问题。但人手和机器人的形态、运动学、控制接口不同，直接把人手轨迹当机器人动作标签会产生 embodiment gap。现有路线常依赖手姿估计、关键点轨迹、编辑后视频或离散动作 token；这些中间表示会带入专用处理器的误差，也未必落在 VLM 已学到的语义空间。

EgoLAP 的核心主张是：人和机器人不必共享低层控制量，应该共享“动作意图”。它把动作的 **what** 表示为可组合的语言动作，把 **why** 表示为结合接触、几何和物体 affordance 的局部运动推理，再保留连续动作专家负责实际控制。

## 二、核心贡献

1. **人机共享语言动作**：将约一秒的末端/手部位姿变化确定性地转成自然语言，避免手工动作标签或另训离散 codec。
2. **运动级推理监督**：生成描述当前接触关系、可供性和预期物理效果的局部 rationale，与动作语言组成 action chain-of-thought。
3. ** matched 实验检验表示选择**：固定主干、数据、训练预算和连续 action expert，只替换 FAST、OAT、language-action 表示，比较人类经验迁移效果。
4. **文本监督不增加控制时解码开销**：文本理由和语言动作只作为 backbone 训练目标，连续 expert 在推理时直接读 observation/state 表征。

## 三、方法原理

### 3.1 整体框架

模型使用 PaliGemma-2B VLM backbone 和从头训练的 300M action expert，采用 Mixture-of-Transformers 式联合注意力。输入最多三路 RGB（base + 两个腕部相机）、指令和离散化状态。VLM 在机器人与人类样本上预测语言动作，并在有标注的子集上预测 rationale；action expert 用 flow matching 学连续动作块。

注意力设计是去除推理时文本生成的关键：action expert 可以 attend observation、instruction、state 前缀，但不能 attend rationale 或语言动作 token；连续动作损失传回 VLM 的梯度在边界停止。故文本确实训练 backbone 的视觉语义表征，但控制不依赖模型先生成一段理由或动作句子。

### 3.2 关键技术细节

**动作帧先按具身类型对齐。** 单臂且有腕部相机时以末端坐标系作参考；双臂和无腕部相机单臂以 base frame 作参考；人手轨迹以头戴相机 frame 作参考。每个一秒窗口取净平移、相对旋转和夹爪/手状态变化。这样统一的是面向视角下的运动语义，不是严格相同的机器人/人类物理坐标系。

**确定性语言语法。** 平移按纵向、竖直、横向排序，取整到厘米；旋转转成外禀 XYZ Euler 角并按 5 度取整；零分量省略；夹爪/手状态用 open、close 或 maintain 表达。双臂固定输出 Left arm 与 Right arm 两段；单臂的左右前缀在训练时随机采样。量化使文本标签可规模化自动产生，也会损失小幅精细运动和快速接触过程。

**运动级 rationale 不是高层任务计划。** 标注解释当前看到的对象/手部关系、接触或 affordance，以及这段动作带来的即时物理效果。比如描述手怎样压住/拉动物体，而不是只说“完成整理桌面”。标注由 Gemini 离线生成，输入含任务、按时间排序的图像/action anchors、近期动作历史和 ground-truth 语言动作。训练时只用主 rationale，不用模型同时生成的其他解释字段。

**随机化 rationale 上下文与损失。** 有 rationale 标签的样本中，50% 直接预测语言动作，25% 将 rationale 作为 teacher-forced 前缀但不对 rationale 计 loss，25% 同时使用并监督 rationale。语言动作每次都监督。这减少策略只在推理文本齐全时才会动作的依赖。

### 3.3 训练与优化

| 项目 | 设置 |
|---|---|
| VLM / action expert | PaliGemma-2B / 300M，后者从头初始化 |
| 损失 | 文本 autoregressive loss 权重 0.8；连续 flow matching 权重 1.0；action loss 不向 VLM 反传 |
| 训练 | batch 2048，40k steps，64 TPU v6e，Adam，2500-step warmup 后 constant；EMA 0.999 |
| 图像/状态 | 224×224；每张图 256 个 SigLIP patch tokens；state 归一化后均匀离散到 256 档并串入 prompt |
| 推理 | 30-step action chunk，14 维；标准配置 10 Euler integration steps；推理不解码语言 |
| 后训练 | ABC 仿真上 GRPO，只更新语言 backbone，冻结 vision/action expert；以解析出的语言动作与示范动作距离做奖励 |

### 3.4 数据使用与维度追踪

| 数据组 | Batch 权重/规模 | 样本与监督 |
|---|---|---|
| ABC YAM 双臂机器人 | 30% batch | 连续 14 维动作 + 语言动作；作为测试零样本仿真的训练机器人域 |
| EgoVerse 第一人称人类视频 | 25% batch，MECKA/Scale/Aria 子集 | 双手轨迹转语言动作；只有 MECKA 子集带运动级 rationale |
| 其他双臂机器人 | 25% batch，含 AgiBot、Galaxea R1 Lite、MolmoAct2 YAM | 连续动作与语言动作 |
| 单臂 OXE 数据 | 20% batch，DROID 约占 12% | 单臂动作映射到 14 维双臂槽位，另一臂补零；DROID 带 rationale |
| 总原始数据 | 约 10k 小时 | rationale 覆盖 MECKA 20% + DROID 12% batch；其余只有动作监督 |

**维度快照**

- 观测：最多 3 路 RGB，224×224；单臂缺少视角时补零并用 mask。
- 状态：20 维，双臂各 10 维（末端位置、连续 6D 旋转、夹爪）；单臂或人类可用 10 维并省略尾部 padding。
- 连续动作：30×14，即一秒窗口，每臂 7 维 Cartesian action；单臂另一半补零。论文未在主文统一报告 Hz，附录数据 anchors 有约 30Hz的人手采样信息。
- 模型输入：三相机最多 768 个图像 patch tokens，加最多 320 个文本/state tokens；VLM width 2048；action expert width 1024。
- 语言动作量化：平移厘米级、旋转 5 度级，按固定词序生成；因此是粗粒度监督目标，不是部署 action decoder。
- 训练归一化：per-category 1st–99th percentile bounds；图像随机裁剪、旋转、颜色增强；腕相机 dropout 0.1，随机相机 mask 0.2。

**训练样本构建链路**：机器人轨迹的末端状态/动作与人类 head-camera frame 中的左右手轨迹，先投到各自约定的 egocentric frame；由轨迹窗口自动生成连续 action target 和离散语言动作；MECKA 与 DROID 的少数窗口再由 Gemini 根据图像、指令、动作与短历史生成 rationale。文本作为 VLM 目标，连续块作为 action expert 目标。

## 四、实验与结果

### 4.1 实验设置

真机使用训练中未见过臂间距和相机配置的双臂 YAM，5 个任务（简单/复杂 pick-place、折裤子、交接放置、擦拭），每任务 10 次，复杂 pick-place 12 次，报告 mean task progress。仿真基于 ABC 的 14 个双臂任务，未用仿真数据训练，每任务 100 episodes，共 1,400 episodes，报告 success rate。主比较包括 EgoLAP、EgoFAST、EgoOAT；这些策略共享数据、预处理、主干和连续动作专家，主要替换文本动作目标。多数训练 40k steps，EgoLAP 最终训练至 30k。

### 4.2 主要结果

| 设定 | EgoLAP | 对照/结论 |
|---|---:|---|
| 零样本真机任务进度，有语言动作无 rationale | 44.5% | FAST 35.2%，OAT 18.0% |
| 零样本仿真成功率，有语言动作无 rationale | 16.4% | FAST 1.6%，OAT 2.7% |
| 加运动级 rationale 后真机进度 | 80.1% | 相对无 rationale +35.6 点；五个任务均提升 |
| 加运动级 rationale 后仿真成功率 | 38.3% | 相对无 rationale +21.9 点；所有任务组均提升 |
| 人类数据增益，无 rationale | 7.2% → 16.4% 仿真成功率 | 2.3×；同一语言动作表示，去掉人类数据为 robot-only 对照 |
| 交叉具身 skill retrieval | Robot→Human R@1 51% → 76% | rationale +25 点；反向 42% → 63% |

最有说服力的是控制变量的表示对比与 rationale ablation：相同 continuous action expert 下，语言动作明显超过 FAST/OAT；语言动作加运动级推理后又大幅提升。该结论适用于论文的训练混合和双臂桌面任务，尚不能说明自然语言动作在所有控制域都优于连续 tokenization。

### 4.3 消融实验

- **人类数据与推理互补**：robot-only 从 7.2% 提升到 18.4%；robot+human 从 16.4% 提升到 38.3%。人类数据增益在有 reasoning 时更大。
- **推理格式对照**：将 subtask、object box、visual trace 放在动作文本前的复合 ECoT 目标单独仅 5.3%，低于无推理语言动作 16.4%；motion-level reasoning 单独 38.3%，两者叠加为 14.4%。
- **表示依赖性**：rationale 对 FAST 仿真仅 +7.1 点且真机从 35.2% 降至 26.1%；OAT 增益也较小。这支持语义动作目标与局部物理推理有协同，而非“多加文本监督就会更好”。
- **GRPO**：冻结 action expert 与 vision encoder 后，语言动作奖励仍提升 ABC 仿真任务；只采样动作文本比同时采样 rationale 的增益更大（+19.4% 对 +12.7%），作者将其归因于训练时未见过 ABC rationale。

## 五、局限性与展望

作者明确指出语言动作目前固定总结一秒窗口，厘米/5 度取整可能不适合快动作、接触丰富或精细控制；评估集中在双臂桌面操作；RL 只在仿真进行，优化的是动作匹配 proxy 而非真实任务奖励。

我认为另一个关键限制是语言标签的 frame 约定仍带有具身专用元数据：语言语义共享并不等于坐标转换/视觉标定自动消失。人类 rationale 只覆盖 EgoVerse 的 MECKA 子集，机器人 rationale 只覆盖 DROID；人类数据带来的改进也依赖于特定 10k 小时混合配方。真机任务进度的每项试验数为 10–12，后续需要更大任务分布和独立团队复现。

值得继续验证的方向包括自适应语言粒度、短时精细控制标签、把任务级成功反馈用于联合调整 rationale/action expert，以及在更多机器人构型和移动操作上测试数据迁移。

## 六、灵魂三问

1. **它解决了什么问题？** 第一人称人类动作轨迹不能直接作为机器人控制目标。EgoLAP 用面向视角的一秒语言动作统一意图，并用物理 rationale 表达动作发生的局部原因。

2. **为什么这么做？** 语言是预训练 VLM 已具备的表示空间，手与机械臂都能以同一词汇描述近似运动意图；连续 action expert 保留精确控制路径，而不要求生成文本后再解析执行。

3. **什么证据最有说服力？** 在相同架构和连续动作监督下，language-action + rationale 达到 80.1% 真机进度和 38.3% 零样本仿真成功率；rationale 对 FAST/OAT 的增益远小于对语言动作的增益，说明关键在监督接口的组合。

## 七、个人总结

1. EgoLAP 的关键不是让机器人执行自然语言，而是把自然语言当成人和机器人共同训练 VLM 的运动监督，再由连续 expert 负责控制。
2. 最大优势是 ablation 较清楚地分离了人类数据、动作表示和 reasoning 格式；主要弱点是动作语言的时间/精度瓶颈以及评测任务范围有限。
3. 对 VLA 数据工程而言，最可借鉴的是“先把人机轨迹投到一致视角框架，再用确定性语法生成标签”，以及文本标签只训练 backbone、连续动作梯度隔离的设计。

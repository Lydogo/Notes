# FrameSkip：从更少但更具信息量的帧学习 VLA 模型

> 原标题：FrameSkip: Learning from Fewer but More Informative Frames in VLA Training
> 作者：Bin Yu, Shijie Lian, Xiaopeng Lin, Zhaolong Shen, Yuliang Wei, Changti Wu, Hang Yuan, Haishan Liu, Bailing Wang, Cong Huang, Kai Chen（哈工大 + 中关村学院 ZGCA + 中关村 AI 研究院 + 多家高校 + DeepCybo）
> 发表：arXiv preprint，2026 年 5 月（arXiv:2605.13757v1）
> 链接：https://arxiv.org/abs/2605.13757

---

## 一、研究背景与动机

- **领域核心问题**：VLA（Vision-Language-Action）模型当前几乎都假设"每一帧机器人演示数据都同等有价值"，然后把整段轨迹的所有帧均匀采样进训练。但现实中机器人演示是**时间稀疏不均衡**的——大段是"接近物体""稳定搬运"这种低变化帧，真正决定任务成败的"对齐 / 接触 / 抓取 / 释放"等关键瞬间只占很小比例。

- **现有方法不足**：
  - 模型架构层面（OpenVLA、π0、GR00T 等）只在 backbone、action head、数据 mixture 上下功夫，没人系统研究过"轨迹内部如何分配监督"。
  - 数据 curation 层面（Remix、Sci-zor）只做轨迹级筛选/重加权，**轨迹内部的帧仍然被一视同仁**。
  - TGM-VLA 关注 keyframe 过采样问题，但仅适用于 keyframe-based 架构。

- **本文出发点**：把"帧选择"重新定义成**"在固定优化预算下，把监督信号重新分配到最关键的瞬间"** 的问题，而不是单纯的数据压缩。在 VLA 训练全流程中插入一个 dataloader 层的轻量级帧选择模块，**不动模型架构、不动 loss、不动推理**。

## 二、核心贡献

1. **首次把帧级监督分配作为 VLA 训练的优化对象**，明确指出"时间监督不平衡"（temporal supervision imbalance）是当前 VLA 训练中一个被严重忽视的问题。
2. **提出 FrameSkip**，一个架构无关的 dataloader 层框架，用四种轻量级轨迹线索（动作变化 + 视觉-动作一致性 + 任务进度先验 + 夹爪过渡保留）给每帧打分并按目标保留率剪枝。
3. **系统性消融**：在 RoboCasa-GR1、SimplerEnv、LIBERO 三个 benchmark 上做了 retention ratio、importance metric、warmup steps 的全面消融，验证收益来自**"哪里被监督"** 而不仅仅是"看了更少帧"。
4. **强结果**：在仅保留 20% 帧的设置下，三 benchmark 平均成功率从 66.50%（full-frame）提升到 76.15%。

## 三、方法原理

### 3.1 整体框架

FrameSkip 完全在数据层（dataloader）做事，分三步：

```
[离线预处理] 每条轨迹 → 计算 AVI/VAC/TPI 三类帧重要性分数 → 按 r 剪枝并缓存帧索引
                    ↓
[训练阶段]   Phase 1 Warmup: 全帧训练（Nwarm 步）
            Phase 2 Pruned Sampling: 5 个剪枝 mini-batch + 1 个全帧 mini-batch 交替
                    ↓
[索引重映射] dataloader 用 binary search 把请求 timestep → 最近的保留 timestep
                    ↓
[VLA 训练]   架构、loss、推理一行不改
```

输入：原始 LeRobot 格式机器人演示数据集；输出：训练好的 VLA 策略，推理时无任何额外开销。

### 3.2 关键技术细节

#### (1) 三个互补的帧重要性信号

**Action Variation Importance (AVI)**——捕捉局部动作动态：

$$\text{AVI}(t) = \| a_t - a_{t-1} \|_2 + \lambda \cdot \text{MeanVar}(a_{t+1:t+k})$$

其中 $k=3$、$\lambda=0.1$。第一项是当前动作与上一步的差，第二项是后 $k$ 步动作的均值方差。**直觉**：动作突变的帧（比如夹爪闭合瞬间）打高分，因为这些帧对应"决策切换点"。

**Visual-Action Coherence (VAC)**——捕捉视觉变化和动作变化的"耦合度"：

$$\text{VAC}(t) = \frac{\| v_t - v_{t-1} \|_2}{\| a_t - a_{t-1} \|_2 + \epsilon}$$

其中 $v_t$ 是 DINOv2 提取的视觉特征。**直觉**：当动作变化不大但视觉变化很大时（比如手碰到了物体导致物体动了），说明这一帧承载了重要的"环境交互信号"，光看 action 抓不到这种事件。为了控制成本，VAC 在稀疏采样的视频帧上算然后线性插值回完整序列长度。

**Task Progress Importance (TPI)**——编码任务的弱结构先验：

$$q(p) = \sum_{m=1}^M \pi_m \mathcal{N}(p; \mu_m, \sigma_m^2), \quad \text{TPI}(t) = \frac{q(p_t)}{\max_s q(p_s)}$$

其中 $p_t = (t-1)/(T-1)$ 是归一化进度。**直觉**：人类操作的关键事件（对齐、抓取、释放）在轨迹中的位置有规律可循。论文用 5% 训练数据上的"关键阶段中心"标注拟合一个 1D GMM，把这种结构知识做成离线先验。如果没标注，可以退化成简单的"中段更重要"高斯先验：$\text{TPI}(t) = \exp(-(p_t - 0.5)^2 / \sigma^2)$。

**关键**：TPI 的标注**仅用于离线打分预处理**，VLA 模型本身和评估流程都接触不到这些标注，避免泄漏。

**综合得分 + 夹爪过渡保留**：

$$I(t) = \alpha \cdot \widehat{\text{AVI}}(t) + \beta \cdot \widehat{\text{VAC}}(t) + \gamma \cdot \widehat{\text{TPI}}(t)$$

默认 $\alpha=0.6, \beta=0.2, \gamma=0.2$（AVI 主导，VAC 和 TPI 辅助）。所有分数先做 trajectory 内 min-max 归一化。最后再乘上"夹爪/末端状态绝对变化因子"做 gripper-aware 加成，保证抓取/释放瞬间几乎不会被剪掉。

#### (2) Ratio-Aware 帧剪枝

给定保留率 $r$，目标保留帧数：

$$K_r = \max(K_{min}, \lfloor rT \rfloor)$$

先按 $(1-r)$ 分位数取阈值 $\theta_r$，保留 $\{t \mid I(t) \geq \theta_r\}$。再加三条工程约束：

- **强制保留**：首帧、末帧、夹爪过渡帧、动作变化最大的 top 10% 帧。
- **数量微调**：分位数法保留数量偏离 $K_r$ 时按重要性顺序加/减帧。
- **时间一致性**：可选地填补保留帧之间的"超大间隙"，避免轨迹被剪到不连续。

预剪枝结果按多个 $r$ 缓存（key 是 importance/pruning 配置），多次实验复用。

#### (3) Warmup + Pruned Sampling with Anchors

训练分两阶段：

- **Phase 1 Warmup**：前 $N_{warm}$ 步用 $r=1.0$ 的全帧训练，给策略一个稳定的初始视觉-动作 grounding。
- **Phase 2 Pruned Sampling**：之后大部分 mini-batch 来自压缩视图（$r<1$，主实验 $r=0.2$），少数 mini-batch 仍来自全帧视图作为"context anchor"。主实验比例是 5:1（5 个剪枝 batch + 1 个全帧 batch）。

**直觉**：剪枝 batch 让梯度集中在关键瞬间，全帧 batch 偶尔刷新一下完整轨迹上下文，防止过拟合到太稀疏的转折点。注意——**优化目标和模型一行不改**，改的只是"哪些 timestep 主导梯度"。

#### (4) 训练集成（dataloader 索引重映射）

每个采样的训练 step：
1. 标准 LeRobot 索引 → 拿到 (trajectory, original timestep)。
2. 按当前 active retention ratio，从缓存里查这条 trajectory 的保留索引列表。
3. **二分查找**把请求的 timestep 映射到第一个不早于它的保留 timestep（如果到末尾就 fallback 到最后一个）。
4. 用原 dataloader 加载这个 frame，过原 transform/collate pipeline。

整个干预完全在 dataloader 内部，**对 LeRobot dataset 长度、trajectory 索引空间、混合数据集采样权重都透明**，可以无缝接入任意 VLA 训练框架（论文用的是 StarVLA + Qwen3-4B-VL-Instruct 理解专家 + DiT + flow matching 动作专家）。

### 3.3 训练与优化

- **VLA 架构**：StarVLA 的双专家结构。Understanding expert = Qwen3-4B-VL-Instruct（多模态隐状态）；Action expert = DiT + flow matching（连续动作生成）。
- **训练**：Global batch size 128，8×H100 + DeepSpeed ZeRO-2。三个 benchmark 各自的优化步数：RoboCasa-GR1 100K、SimplerEnv 60K、LIBERO 30K。
- **FrameSkip 默认设置**：$r=0.2$，pruned:full = 5:1，warmup 5000 步。
- **预处理成本**：DINOv2 在每条轨迹最多 16 个稀疏视频帧上算 VAC 后插值回去；GMM-TPI 用 5% trajectories 的 stage center 标注拟合 3 组件 GMM。

## 四、实验与结果

### 4.1 实验设置

- **Benchmarks**：RoboCasa-GR1（GR1 双臂灵巧手，24 个 tabletop 任务，24K demos）；SimplerEnv（WidowX，4 个 held-out 任务，BridgeV2 训练）；LIBERO（Franka 单臂，4 个 task suites）。
- **基线**：Full-Frame Training（同架构同 schedule，但用全帧），以及多个 SOTA VLA（GR00T N1.5/N1.6、π0、π0.5、TwinBrainVLA、PhysBrain、LangForce、ABot-M0 等）。

### 4.2 主要结果

**三 benchmark 主结果**：

| Benchmark | Full-Frame | FrameSkip | 提升 |
|---|---|---|---|
| RoboCasa-GR1 (24 tasks avg) | 47.80% | **59.50%** | +11.7 |
| SimplerEnv (4 tasks avg) | 55.20% | **71.55%** | +16.4 |
| LIBERO (4 suites avg) | 96.50% | **97.40%** | +0.9 |
| **Macro-average** | **66.50%** | **76.15%** | **+9.65** |

LIBERO 提升小是因为基线已经接近天花板（96.5%）；RoboCasa-GR1 和 SimplerEnv 这两个更难、更接近真实分布的 benchmark 上提升幅度都很大。

特别值得注意的是 **SimplerEnv**：FrameSkip 71.55% 已经超过了 LangForce（66.5%）、TwinBrainVLA（64.5%）、π0.5（57.1%）等所有列出的方法，**仅靠改训练数据选择策略就拿到了第一**。

### 4.3 消融实验

**(1) Retention ratio $r$**（RoboCasa-GR1）：

| r | 10% | 20% | 30% | 40% | 50% | 60% | 100% |
|---|---|---|---|---|---|---|---|
| Avg | 55.00 | 59.50 | 59.50 | 56.75 | **59.75** | 55.92 | 47.80 |

**结论**：所有剪枝设置都比全帧好；最佳点在 $r=50\%$，但 $r=20\%$-30% 已经几乎饱和。说明**全帧训练真的在浪费监督信号**，即便保留率压到 10% 也仍优于全帧。

**(2) Importance metric** 消融（三 benchmark 平均）：

| 变体 | RoboCasa-GR1 | SimplerEnv | LIBERO | Avg |
|---|---|---|---|---|
| Random | 47.67 | 56.51 | 96.30 | 66.83 |
| AVI | 54.25 | 57.29 | 97.05 | 69.53 |
| AVI+TPI | 57.42 | 59.90 | 97.00 | 71.44 |
| AVI+VAC | 58.75 | 65.08 | 97.15 | 73.66 |
| AVI+VAC+TPI | 59.00 | 67.33 | 97.20 | 74.51 |
| **FrameSkip Full**（+gripper） | **59.50** | **71.55** | **97.40** | **76.15** |

**结论**：每个组件都有正向贡献。Random（同样 20% 保留率）只能拿到 66.83%，比全帧的 66.50% 几乎没提升；而 AVI+VAC+TPI+gripper 的完整 FrameSkip 拿到 76.15%。**这一对比直接证明：收益不是"看更少帧能正则化"带来的，而是"看哪些帧"决定的**。

**(3) Warmup steps**（RoboCasa-GR1）：从 2500 到 15000 步只波动 1 个点（最佳 5000 步 59.50%），FrameSkip 对这个超参不敏感。

## 五、局限性与展望

- **作者讨论的局限**：原文没有专门 limitations 章节，但隐含的限制：
  - GMM-TPI 需要少量 stage center 标注（5% 数据），完全无监督场景下退化为简单高斯先验。
  - VAC 依赖 DINOv2 离线特征提取，预处理有一次性算力成本。
  - 实验全部在仿真 benchmark，未在真实机器人上验证。
  - 仅用 StarVLA + Qwen3-4B + DiT 一种架构组合验证，未在 OpenVLA、π0 等其他 VLA backbone 上验证 plug-and-play 能力。

- **潜在改进方向**（个人补充）：
  - 把 TPI 替换成轻量级"任务进度预测器"（比如基于 BERT 的语言-轨迹对齐），完全去掉人工标注。
  - 把帧重要性做成 in-the-loop 自适应（训练中根据梯度大小或 loss 动态调整保留率）。
  - 把 FrameSkip 应用到大规模真实 VLA 预训练数据集（比如 Open X-Embodiment、BridgeV2、AgiBot World）上验证 scalability。

## 六、灵魂三问

1. **它解决了什么问题？**
   之前训练 VLA 模型时，所有人都默认"轨迹里每一帧都同等重要"，于是训练时间和算力被大段"接近物体"和"稳定搬运"这种没什么信息的帧浪费掉，关键的"对齐 / 抓取 / 释放"瞬间反而被淹没；现在能用一个**纯 dataloader 层、不改架构和 loss** 的剪枝策略，把训练监督集中到关键帧上，相同算力下三 benchmark 平均成功率从 66.5% 提到 76.15%。

2. **为什么这么做？**
   最关键的设计是**用三个互补的、轻量级的、与 VLA 架构无关的信号给帧打分**（AVI 抓动作突变 + VAC 抓视觉-动作错位 + TPI 抓任务进度结构），而**不去训一个学习型的"重要性预测网络"**（像 Sci-zor 那样）。理由是：(a) 学习型方法又引入一个需要训练和验证的子模型，复杂度爆炸；(b) 这三个信号是物理可解释的，能直接对应"接触""抓取"等机器人操作语义；(c) 离线一次性算完缓存好，训练时零开销。同样关键的设计是**"5:1 剪枝+全帧"双轨采样**——纯剪枝会丢失全局上下文，纯全帧又解决不了不平衡问题，5:1 是一个工程上简单且鲁棒的折衷。

3. **什么证据最有说服力？**
   **Importance metric 消融表（Table 5）**：在同样 20% 保留率下，Random 只比 Full-Frame 高 0.33 个点（66.83% vs 66.50%），而 FrameSkip Full 比 Full-Frame 高 9.65 个点（76.15%）。这 9.32 个点的差距**完全来自"选哪些帧"**——不是"看的少所以正则化好"，也不是"训练 batch 多样性提高"，就是"把监督分配到关键帧"这个 idea 本身有效。同时 AVI → +TPI → +VAC → +gripper 每一步都稳定提升，说明四个信号都不是冗余的。这个消融把方法的因果链验证得很彻底。

## 七、个人总结

1. **核心 idea**：VLA 训练应当从"全帧均匀采样"转向"基于帧重要性的监督再分配"——用 AVI（动作突变）+ VAC（视觉-动作错位）+ TPI（任务进度）+ gripper 过渡保留四个轻量级信号，在 dataloader 层剪枝，不改任何模型结构。

2. **优势与不足**：
   - **优势**：plug-and-play 程度极高（理论上对任何 VLA 框架都能直接套）；推理零开销；预处理成本可控（DINOv2 只在稀疏帧上跑）；消融做得透彻，逻辑链严谨；在三个差异很大的 benchmark 上都有稳定提升，说明方法不是过拟合某个特定数据集。
   - **不足**：实验全在仿真，缺真机验证；只在一个 VLA 架构上跑过，所谓"架构无关"还需要更多 backbone 验证；GMM-TPI 仍需少量标注；缺少与 Sci-zor、TGM-VLA 等同类数据 curation 方法的直接对比。

3. **启发**：
   - 对**研究**：提示我们 VLA 领域还有很多"训练数据组织"层面的低垂果实——大家都在卷模型规模、卷动作 tokenization、卷 chain-of-thought，但底层"如何分配监督"的问题反而没人系统研究过。可以推广到"长序列 VLA 训练的多帧时序聚合""视频生成 → action 蒸馏的关键帧选择"等方向。
   - 对**实际应用**：对于像 PICO ego pipeline、LeRobot pipeline 这种自采机器人/人类数据的项目，FrameSkip 的方法几乎可以直接迁移——只需在 LeRobot dataset 的 dataloader 上加一层 index remapping 就能用，能显著降低训练时长和算力成本，特别适合数据量大但任务关键事件稀疏的场景（比如 PICO ego pipeline 处理几百小时第一人称视频）。

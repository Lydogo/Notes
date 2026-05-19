# RoboTwin 2.0：面向鲁棒双臂操作的可扩展数据生成器与基准

> 原标题：RoboTwin 2.0: A Scalable Data Generator and Benchmark with Strong Domain Randomization for Robust Bimanual Robotic Manipulation
> 作者：Tianxing Chen, Zanxin Chen, Baijun Chen, Zijian Cai, Yibin Liu, Zixuan Li, ..., Ping Luo, Yao Mu（HKU MMLab + 上海交大 + Shanghai AI Lab + 清华 + TeleAI + Lumina EAI 等 16 家机构，HKU 与 SJTU 同为牵头单位）
> 发表：arXiv preprint，2025 年 6 月（arXiv:2506.18088v2，2025 年 8 月 v2）
> 链接：https://arxiv.org/abs/2506.18088 ｜ https://robotwin-platform.github.io

---

## 一、研究背景与动机

- **领域核心问题**：要训出能跨场景、跨形态、跨任务的双臂 VLA 基础模型，必须有**高质量 + 高多样性 + 大规模**的演示数据。但真机遥操作贵且慢，覆盖所有任务/物体/形态在工程上不可行。
- **现有合成数据 pipeline 的不足**：
  1. **没有自动质量控制**——没有 expert-level 的闭环验证，生成的轨迹混进很多失败/次优抓取，污染策略学习。
  2. **域随机化过于表层**——场景过于干净均匀，缺少现实里的杂乱物、光照变化、模糊语言指令等关键扰动，sim-to-real 转移差。
  3. **不考虑跨形态差异**——不同双臂平台的运动学差异巨大（Piper 6-DoF 偏好侧抓，Franka 7-DoF 适合垂直抓），现有合成数据不编码这种形态相关的可达性和抓取策略。
- **本文出发点**：构建一个**完整的数据生成 + 基准框架**，解决上述三大痛点。具体目标：用 MLLM 自动生成任务代码 + 系统化域随机化 + 形态感知抓取适配，规模化地造数据，并在统一基准上评估策略泛化。

## 二、核心贡献

1. **MLLM + 仿真闭环的自动代码生成 pipeline**：两个 agent（代码生成 + VLM 观察）通过 10 次仿真执行 → VLM 诊断 → 代码修复的循环，实现自我改进的 expert 代码合成；ASR 比 vanilla 提升 10.9%（71.3% vs 60.4%）。
2. **五维域随机化**：杂乱物体、光照、背景纹理（11K 张库）、桌面高度、语言指令多样化，构成强 sim-to-real 训练分布。
3. **形态感知抓取适配**：每个物体标多个候选抓取姿态（不同方向 + 抓轴），低 DoF 平台（Piper、Aloha-AgileX、ARX-X5）受益显著（最高 +22.7%）。
4. **大规模开源资源**：
   - **RoboTwin-OD** 物体库：147 类 / 731 个物体，富语义+操作标注。
   - **100K+ 预采集轨迹** × 50 任务 × 5 种双臂平台。
   - **统一 benchmark**：50 任务 × Easy/Hard 两种评估设置 + 在线 leaderboard。

## 三、方法原理

### 3.1 整体框架

```
自然语言任务描述
   ↓
[MLLM 代码生成 Agent]  ←┐
   ↓                  │ 修正反馈
执行 × 10 仿真试验      │
   ↓                  │
[VLM 观察 Agent]      │
   ↓                  │
执行日志 + 失败定位 ───┘
   ↓ （success ≥ 0.5 或迭代 > 5 终止）
专家代码 → 配合域随机化批量生成轨迹
   ↓
策略训练 / Benchmark 评估
```

构建在两个基础设施之上：
- **RoboTwin-OD** 物体资产库（语义+操作点标注）。
- **预定义技能 API 库**（grasp_actor、place_actor、move_by_displacement 等）。

### 3.2 关键技术细节

#### (1) MLLM + 仿真闭环的代码生成

两个 agent 协作：

- **Code Generation Agent**：输入任务名 + 自然语言描述 + API 列表 + 函数调用示例 + 分层约束规范 → 输出 Python 程序（一系列 API 调用）。
- **VLM Observer Agent**（moonshot-v1-32k-vision-preview）：对 10 次执行的关键帧做逐步检查，定位失败步骤并诊断原因（抓取失败 / 放置错位 / API 误用 / 逻辑错误）。

迭代终止条件：
- 10 次执行中成功率 ≥ 0.5（设定阈值）→ 通过；
- 迭代 5 次仍不通过 → 放弃。

**两类反馈**喂回 Code Agent：
- 定量执行日志（成功/失败 + 错误类型）；
- 定性 VLM 诊断（失败步骤的视觉描述和修复建议）。

效果（Tab. 1）：

| 配置 | RoboTwin 1.0 ASR | RoboTwin 2.0 ASR |
|---|---|---|
| Vanilla（一次性生成） | 47.4% | 62.1% |
| + 执行日志反馈（FB） | 60.4% | 66.7% |
| + 多模态反馈（MM FB） | 63.9% | **71.3%** |

RoboTwin 2.0 不仅 ASR 高，迭代轮数也少（MM FB 1.76 vs 1.0 的 2.42），且代码更短（569 tokens vs 1236）。

#### (2) 五维域随机化

| 维度 | 做法 |
|---|---|
| **场景杂乱** | 从 RoboTwin-OD 抽取 distractor 物体；碰撞感知放置；与任务相关物体语义/视觉相似的会排除 |
| **背景纹理** | 11K 张高质量纹理库（LLM 生成描述 → SD v2 生成 20K → 人工筛选剩 11K） |
| **光照** | 随机化光色、类型、强度、位置；色温变化对物体外观影响巨大 |
| **桌面高度** | 在物理合理范围内均匀采样，影响相机视角和空间关系 |
| **语言指令** | MLLM 生成多套任务模板 + 物体的多种描述（形状/纹理/功能/部件等），每条轨迹独立采样组合 |

例：`Move Can Pot` 任务的指令可以从模板 `Use {a} to place {A} to the left of {B}` 派生为「Use left arm to place sauce can to the left of gray kitchenpot」、「Use left arm to place white plastic lid sauce can to the left of kitchenpot for boiling and cooking」等多种说法。

#### (3) 形态感知抓取适配

不同臂型可达性差异大：

| 形态 | DoF | 偏好抓取 |
|---|---|---|
| Franka / UR5 | 7-DoF | 垂直 top-down |
| Piper / ARX-X5 / Aloha-AgileX | 6-DoF | 侧抓为主 |

做法：每个物体标注**多个候选抓取姿态**（覆盖多个抓轴和接近方向）+ 角度扰动（偏向高可达性方向）+ 并行运动规划。

效果（Tab. 2）：

| 形态 | RoboTwin 1.0 | RoboTwin 2.0 | Δ |
|---|---|---|---|
| Aloha-AgileX | 65.1% | 78.8% | **+13.7%** |
| Piper | 2.4% | 25.1% | **+22.7%** |
| Franka | 67.3% | 67.2% | -0.1% |
| UR5 | 57.6% | 57.1% | -0.5% |
| ARX-X5 | 68.6% | 74.2% | +5.6% |

收益集中在低 DoF 平台——验证了"提供更多 feasible grasp options"对低 DoF 限制的缓解效果；高 DoF 已经够灵活所以没收益。

### 3.3 RoboTwin-OD 物体库

- **147 类 / 731 个物体**：
  - 534 个 in-house RGB→3D 重建（Rodin 平台）+ convex decomposition；
  - 153 个来自 Objaverse；
  - 44 个 articulated 物体来自 SAPIEN PartNet-Mobility。
- **每个物体的标注**：
  - **15 条多样化语言描述**（形状/纹理/功能/部件/粒度）；
  - **关键点-轴**：placement points、functional points、grasp points、grasp axes。
- **50 任务**全部双臂协作场景，支持 5 种形态。

## 四、实验与结果

### 4.1 实验设置

- **5 个 robot 形态**：Aloha-AgileX、ARX-X5、Piper、Franka、UR5。
- **5 个策略 baseline**：ACT、DP、RDT、Pi0、DP3。
- **3 类评估**：
  1. 代码生成质量（10 任务，ASR / Top5-ASR / CR-Iter / Token）；
  2. 域随机化对策略鲁棒性的影响（仿真）；
  3. Sim-to-Real 真机迁移（4 任务 × COBOT-Magic 双臂）。

### 4.2 主要结果

#### 域随机化对策略鲁棒性的影响（Tab. 3）

在 32 任务上预训练 9600 条专家轨迹，分 clean / domain-randomized 两组：

| 模型 | Pretrain | +Clean fine-tune | +Rand. fine-tune |
|---|---|---|---|
| RDT 平均 | 18.8% | 14.6% | **24.8%** |
| Pi0 平均 | 22.5% | 24.9% | **29.1%** |

**关键观察**：clean fine-tune 几乎无提升（甚至降）；只有 domain-randomized 数据才让模型获得对环境扰动的鲁棒性。证明合成数据本身不是 sim-to-sim gap 的问题，而是要看**数据是否覆盖了真实世界的变化**。

#### Sim-to-Real 真机实验（Tab. 4）

4 个任务（Stack Bowls / Handover Block / Pick Bottle / Click Bell），三种训练配置：

| 配置 | Unseen Bg + Cluttered（最难） |
|---|---|
| 10 条 clean 真机 demo | 9.0% |
| 10 条真机 demo + 1000 条 RoboTwin 2.0 合成 | **42.0%（+33.0%）** |
| 仅 1000 条 RoboTwin 2.0 合成（zero-shot） | 29.5%（+20.5%） |

**核心发现**：
- **few-shot 场景**：1000 条合成 + 10 条真机 → 367% 相对提升（10→42 在最难配置上）。
- **zero-shot 场景**：纯合成数据已有 228% 相对提升。
- 越难的场景（unseen background + cluttered）提升越大，证明 RoboTwin 2.0 的域随机化**精准对准了真实场景的痛点**。

#### Benchmark 评估（Tab. 10 全表 50 任务）

5 个策略在 50 任务 × Easy（clean）/ Hard（域随机化）两种条件下的平均成功率：

| 策略 | Easy avg | Hard avg | Δ |
|---|---|---|---|
| ACT | 29.7% | 1.7% | -28.0 |
| DP | 28.0% | 0.6% | -27.4 |
| RDT | 34.5% | 13.7% | -20.8 |
| Pi0 | 46.4% | 16.3% | -30.1 |
| DP3 | 55.2% | 5.0% | -50.2 |

**核心观察**：
- 非预训练模型（ACT/DP/DP3）在 Hard 设置下几乎全军覆没（<5%）；
- 预训练 VLA（RDT/Pi0）有更强韧性，但 Easy→Hard 仍掉 20-30 个点；
- DP3 在 Easy 下用 3D 信息拿到最佳少样本表现（55.2%），但严重依赖完美点云分割，Hard 下掉到 5%；
- 表明 VLA 预训练带来的 prior 有效，但**数据多样性仍是 scaling 的关键瓶颈**。

### 4.3 消融实验

- **架构改进（不开 feedback）**：RoboTwin 2.0 vs 1.0 在 zero-shot 代码生成上：AST 相似度 +21.06%、CodeBERT +1.08%、Unixcoder +5.97%；代码 token 从 1236 降到 569（缩 54%）。
- **VLM observer 开销**：每次观察约 6894 tokens（6295 input + 599 output），是 plug-and-play 模块，只在执行失败时触发。
- **多模态反馈 vs 仅执行日志**：MM FB 比 FB 在 RoboTwin 2.0 上 +4.6 ASR，证明 perceptual 诊断带来了执行日志看不见的失败模式定位。

## 五、局限性与展望

- **作者讨论的局限**：
  - VLM observer 的诊断能力还不够稳定——manually 标注的 130 个执行序列上，error detection 的 F1 只有 0.302（recall 高但 precision 低，容易把成功误判为失败）；error localization accuracy 仅 30%。
  - 部分细粒度任务（如 Hanging Mug、Click Alarmclock、Open Microwave）的代码生成成功率为 0%，对极依赖几何精确性的任务仍束手无策。
  - 真机实验只覆盖了 4 个任务和 1 个平台（COBOT-Magic），未在更多形态上验证 sim-to-real。
- **潜在改进方向**：
  - 用更强的多模态 reasoning 模型（如 GPT-4V、Gemini 2）替代当前 moonshot-v1，提升 observer 诊断准确率；
  - 把 RoboTwin 2.0 与真实世界 RL（如 π*0.6 / π0.7 的自主 rollout）结合，让 sim 数据和 real rollout 互补；
  - 扩展到更复杂的长 horizon 任务（多物体 + 多阶段）和移动操作场景。

## 六、灵魂三问

1. **它解决了什么问题？**
   之前合成数据 pipeline 要么没人审（混进失败轨迹）、要么场景太干净（sim-to-real 直接崩）、要么没考虑形态差异（同一套代码低 DoF 臂跑不通）；RoboTwin 2.0 用 MLLM+VLM 闭环造代码 + 五维域随机化 + 形态感知抓取，一站式造出能让策略 zero-shot 跨域 +228%、few-shot +367% 的高质量合成数据。

2. **为什么这么做？**
   最关键的设计选择是**让 VLM 观察执行过程并诊断失败原因**，而不是只用执行日志（成功/失败二值）。理由：很多真实失败（如抓取角度偏 5 度、放置稍微歪了、物体语义混淆）执行日志根本看不出来，但这些恰恰是 LLM 修复代码最需要的信号。同样关键的是**形态感知的多候选抓取**——给每个物体标 N 个候选 pose 而不是 1 个，让低 DoF 平台总能找到一个可达姿态，否则像 Piper 这种 6-DoF 平台直接掉到 2.4%。

3. **什么证据最有说服力？**
   **Tab. 4 真机 sim-to-real 实验**：unseen background + cluttered 场景下，10 条真机 demo 只有 9%，加 1000 条合成数据干到 42%（+33 个点）；甚至纯合成（zero-shot）也有 29.5%（+20.5）。这个对比直接证明 RoboTwin 2.0 的合成数据**真的能在最难的真实场景下补足真机数据缺失**。同时 Tab. 3 里 RDT/Pi0 加 clean fine-tune 几乎没提升（甚至降），加 randomized fine-tune 才大涨——把"是不是 sim-to-sim gap"这个干扰因素也证伪了，差距确实在"数据多样性"这一维。

## 七、个人总结

1. **核心 idea**：把"合成数据生成"从手写脚本升级成一个**完整的闭环系统**——MLLM 生成代码 → 仿真执行 → VLM 多模态诊断 → 自动修复 → 配五维域随机化 → 形态感知适配。整个 pipeline 几乎不需要人工介入，规模化造出能直接 sim-to-real 的数据。

2. **优势与不足**：
   - **优势**：完整度高（资产库 + 数据生成器 + benchmark + 100K 轨迹全部开源）；真机验证扎实（4 任务覆盖 stack/handover/pick/click 四大类操作）；50 任务的 leaderboard 体量足够大，能成为社区共同评估的事实标准。架构设计有 generality——MLLM agent 框架可以直接迁移到其他模拟器。
   - **不足**：VLM observer 性能本身偏弱（F1 0.302、localization 30%），未来如果用更强的 reasoning 模型重做这部分应该还能涨；真机实验仅 COBOT-Magic 一个平台；某些细粒度任务（Hanging Mug 等）代码生成 0% 完全失败，说明纯靠 LLM 的"语言→动作代码"对几何高敏任务仍力不从心。

3. **启发**：
   - 对**研究**：印证了"合成数据 + 域随机化"这条路在 VLA 时代仍然有效（甚至比 π0.5 那种海量真机数据成本低得多），但**关键是数据要"难"**——clean 数据再多也没用，只有匹配真实世界扰动分布的数据才能让模型学到鲁棒策略。
   - 对**实际应用**：对像 PICO ego pipeline 这种自己造数据的项目，RoboTwin 2.0 的五维域随机化清单（杂乱物 / 光照 / 纹理 / 桌面高度 / 语言）可以直接当成数据增广的"checklist"用；MLLM+VLM 闭环代码生成的设计模式（生成→执行→诊断→修复）也是值得复用的 agent 架构。
   - 对**面试**：是 VLA 数据生成方向必读的 benchmark 论文，跟 π0/π0.5/π0.7 在数据规模上的"暴力扩展"形成对比——一个走真机收集，一个走合成增广。

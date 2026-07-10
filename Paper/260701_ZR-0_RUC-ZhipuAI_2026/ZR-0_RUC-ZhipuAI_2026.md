# ZR-0：用密集 Embodied CoT 监督塑形跨本体 VLA 表征

> 原标题：Training Vision-Language-Action Models with Dense Embodied Chain-of-Thought Supervision  
> 作者：Haoyang Li, Guanlin Li, Youhe Feng, Chen Zhao, Zhuoran Wang, Yang Li, Qizhe Wei, Shifeng Bao, Haitao Shen, Yihan Zhao, Tong Yang, Jing Zhang  
> 机构：Renmin University of China, Zhipu AI  
> 发表：arXiv:2606.30552v1, 2026-06-29  
> 链接：https://arxiv.org/abs/2606.30552  
> 开源：https://github.com/RUCKBReasoning/ZR-0  

---

## 一、研究背景与动机

跨本体迁移（cross-embodiment transfer）是 VLA 模型做通用机器人策略时绕不开的问题。不同机器人平台的关节数、控制接口、状态维度、动作语义都不同。已有方法常用 zero-padding、per-embodiment normalization 或手工统一 action space，让不同数据可以塞进一个模型，但这更多是格式层面的统一，不等于模型真的学到了跨机器人共享的语义表征。

ZR-0 的观察是：低层 state/action 空间强依赖本体，但操作任务中的高层认知流程大多共享，例如场景理解、目标物识别、任务进度判断、未来步骤规划和子任务分解。论文因此用密集具身思维链监督（Embodied Chain-of-Thought, ECoT）去训练 VLM stream，让跨本体数据先在“可语言化、可解释的操作认知过程”上对齐，再把这种表征供给 action expert 生成连续动作。

## 二、核心贡献

1. 提出 ZR-0，一个 2.6B 参数端到端 VLA 模型，用 dense ECoT supervision 对齐跨本体 VLM 表征。
2. 采用 System 2 / System 1 双流结构：Qwen3-VL-2B-Instruct 负责 ECoT 表征学习，500M DiT action expert 负责 flow matching 连续动作 chunk。
3. 通过 cross-attention mask 让 action expert 只看 input prompt features，不依赖生成后的 ECoT tokens，因此推理时完全跳过 autoregressive ECoT generation。
4. 在 ProcCorpus-60M 上预训练，覆盖约 60M frames、1,000 小时、400K+ trajectories，ECoT 标注覆盖 96.8% frames，并在 LIBERO、RoboTwin 2.0、RoboCasa GR-1 Tabletop 和真实 xArm 上评估。

## 三、方法原理

### 3.1 整体框架

ZR-0 是一个 dual-stream VLA：

- **System 2：VLM stream**，初始化自 Qwen3-VL-2B-Instruct，输入多视角图像和任务指令，训练时生成结构化 ECoT。
- **System 1：Action expert**，一个 Diffusion Transformer（DiT）动作专家，输入 VLM hidden features、机器人 state 和 noisy action chunk，通过 flow matching 输出连续动作 chunk。

推理时流程很短：输入图像、语言和机器人状态，VLM 做一次前向得到 prompt features，action expert 用 denoising steps 生成动作 chunk，然后执行并闭环 replanning。ECoT 只在训练时作为监督信号出现，推理时不生成文字推理。

### 3.2 关键技术细节

**Dense ECoT supervision。** 每一帧 ECoT 是一个结构化序列，包含六类信息：

| ECoT 组件 | 作用 |
|---|---|
| Scene Description | 强化物体识别和场景理解 |
| Progress Assessment | 判断任务已经完成到哪一步 |
| Future Plan | 预测剩余步骤，支持长时序规划 |
| To-Do Actions | 把 future plan 分解成原子子任务，是跨本体对齐的关键 |
| Target Objects | 用 JSON bounding boxes 做目标物 grounding |
| Discrete Actions | 用 FAST tokenizer 产生离散动作 token，桥接高层推理和低层控制 |

其中 To-Do Actions 是论文最核心的跨本体设计：同一个“抓取蓝色盘子”“把盘子放进架子”可以适用于 Franka、xArm、ALOHA 或 humanoid，低层关节怎么动则交给 action expert 学。

**ECoT 训练、推理跳过。** 论文的一个重要工程点是 cross-attention mask。虽然 VLM 训练时会自回归生成 ECoT，但 DiT action expert 被限制只能 attend 到 task instruction 和 image prompt 对应的 VLM features，不能 attend 到 ECoT tokens。这样 action expert 学到的是被 ECoT loss 塑形后的 prompt 表征，而不是依赖推理时必须生成的文字链条。

**DiT action expert。** Action expert 包含 state encoder、action encoder、DiT blocks 和 action decoder。每个 block 使用 1 层 self-attention 加 3 层 cross-attention，相比 GR00T N1 的 1:1 比例更强调从 VLM features 吸收图像和语言信息。动作通过 flow matching 学习 denoising vector field，输出 continuous action chunks。

**VL data co-training。** 除机器人 ECoT 数据外，作者混入 CapsFusion、Pixmo 等通用 VL 数据，只训练 VLM 的语言建模，不参与动作预测。目的不是增加机器人技能，而是减少 action-only fine-tuning 导致的 VLM 感知和语言能力遗忘，尤其对 OCR、目标识别和开放词汇指令有帮助。

### 3.3 训练与优化

预训练数据是 ProcCorpus-60M，来自 DROID、Bridge、Fractal、RH20T、Open X-Embodiment subsets 等，约 60M frames、1,000 小时、400K+ trajectories。状态和动作统一 padding 到 64 维，对 padding 维度 mask loss；每维用训练集 1st/99th percentile 做 min-max normalization。

训练目标是：

- ECoT next-token prediction loss，只更新 VLM 参数；
- Flow matching action loss，更新 action expert，并通过 VLM features 回传到 VLM；
- 总损失 `L = L_ntp + alpha L_fm`，预训练时 `alpha = 5`。

模型总参数约 2.6B：VLM 2.1B，DiT action expert 500M。预训练 action horizon `H = 32`，global batch size 1,024。后训练时 LIBERO 使用 `H = 10`，RoboTwin 2.0、RoboCasa 和真实 xArm 使用 `H = 16`。论文报告单张 A6000 上 action chunk 约 90 ms，结论中写单张 H100 上约 100 ms。

## 四、实验与结果

### 4.1 实验设置

评估覆盖三类仿真本体和一个真实平台：

- **LIBERO**：单臂任务，40 tasks，1,693 training trajectories，评估 Spatial/Object/Goal/Long。
- **RoboTwin 2.0**：ALOHA 双臂，50 tasks，每任务 50 clean demos 和 500 randomized demos。
- **RoboCasa GR-1 Tabletop**：GR-1 humanoid tabletop，24 tasks，单模型多任务训练。
- **Real xArm**：4 个真实任务，超过 2,000 条遥操作轨迹，50+ objects，5 Hz 控制频率；每任务 10 次评估，用 progress score。

### 4.2 主要结果

| Benchmark | ZR-0 | 最强/关键对比 | 结论 |
|---|---:|---:|---|
| LIBERO Avg. | 97.8 | MolmoAct2 97.2, GR00T-N1.7 97.0 | 在接近饱和 benchmark 上小幅领先 |
| LIBERO-10 | 96.4 | pi0.5 92.4 | 长时序 suite 增益明显 |
| RoboCasa GR-1 Tabletop Avg. | 69.3 | JoyAI-RA 63.2 | humanoid tabletop 上领先 6.1 点 |
| RoboTwin 2.0 Clean | 88.70 | Motus 88.66, LingBot-VLA 88.56 | 与强基线接近并略高 |
| RoboTwin 2.0 Randomized | 87.98 | Motus 87.02, LingBot-VLA 86.68 | 随机化鲁棒性较好 |
| Real xArm Avg. progress | 76.0 | pi0.5 67.8 | 真实任务高 8.2 分 |

RoboCasa 的结果比较有信息量：ZR-0 在大量 pick-and-place 任务上明显领先，例如 CuttingboardToTieredbasket 80 vs. JoyAI-RA 36、PlacematToPlate 88 vs. 38、PlateToPan 89 vs. 46。但它在 6 个 Close tasks 上弱于 JoyAI-RA，例如 BottleToCabinetClose 39 vs. 84、CanToDrawerClose 47 vs. 90。这说明 ECoT 对预训练中常见的 pick-and-place 技能迁移很好，但不能凭空补足稀缺操作原语。

真实 xArm 结果如下：

| Method | Pick & Place | Hang Cups | Clean Table | Push Blocks | Avg. |
|---|---:|---:|---:|---:|---:|
| pi0.5 | 56.7 | 85.0 | 63.3 | 66.1 | 67.8 |
| ZR-0 | 66.7 | 70.0 | 73.4 | 94.0 | 76.0 |

最强的真实任务证据是 Push Blocks：ZR-0 得 94.0，pi0.5 为 66.1。这个任务需要识别木块上的字母并按 OCR/语言目标操作，比较符合 ECoT + VL co-training 保留 VLM 语义能力的预期。反过来，Hang Cups 上 ZR-0 低于 pi0.5，说明精细对准和灵巧控制仍更依赖动作监督规模，而不是高层推理。

### 4.3 消融实验

| 设置 | LIBERO-Spatial | LIBERO-Object | LIBERO-Goal | LIBERO-10 | Avg. |
|---|---:|---:|---:|---:|---:|
| Full FT | 97.4 | 99.4 | 98.0 | 96.4 | 97.8 |
| FT w/o ECoT | 96.8 | 98.6 | 94.8 | 92.6 | 95.7 |

这个消融把 ECoT next-token prediction objective 去掉，但保持同样模型结构和 flow-matching action loss。平均下降 2.1 点，LIBERO-10 下降 3.8 点，说明 ECoT 对长时序任务更有帮助。不过消融只在 LIBERO 上做，缺少对 RoboCasa/RoboTwin/real robot 的同等 ablation，也没有拆分六类 ECoT 组件各自贡献。

## 五、局限性与展望

作者在 Discussion 里给出三个方向。第一，预训练机器人数据约 1,000 小时，远低于 pi0 的 10,000+ 小时、LingBot-VLA 的 20,000 小时和 Qwen-RoboManip 的 30,000+ 小时；RoboCasa Close tasks 的失败也说明数据覆盖不足会限制技能迁移。第二，ECoT 可以扩展到人类第一视角视频，因为 scene description、planning、sub-task decomposition 和 object grounding 不依赖机器人动作标签。第三，逐帧 dense ECoT annotation 成本高，需要研究选择性标注 informative frames。

我的额外判断是，ZR-0 证明的是“ECoT 作为表征监督有价值”，但还没有充分证明“ECoT 六组件都必要”。它的核心消融偏少，尤其没有对 VL co-training、attention mask、1:3 cross-attention 比例、ECoT 组件拆分做系统验证。另一个部署层面的限制是，ECoT 标注 pipeline 依赖强 VLM 自动生成，错误标注是否会在大规模预训练里积累偏差，论文没有深入讨论。

## 六、灵魂三问

1. **它解决了什么问题？**

   它试图解决跨本体 VLA 训练中“格式统一但语义不统一”的问题。不同机器人低层 state/action 维度不可直接共享，但场景理解、进度判断、任务规划和子任务分解是共享的；ZR-0 用 dense ECoT 把这些共享认知过程作为训练监督。

2. **为什么这么做？**

   因为显式生成 CoT 推理会增加推理延迟并可能引入 autoregressive error，而完全不用推理监督又容易让 VLM 表征只被 action loss 弱监督塑形。ZR-0 的折中是训练时用 ECoT 塑形 prompt features，推理时 action expert 只读取 prompt features，从而保留表征收益并跳过文本生成。

3. **什么证据最有说服力？**

   LIBERO 消融最干净：去掉 ECoT 后平均从 97.8 降到 95.7，LIBERO-10 从 96.4 降到 92.6，说明 ECoT 对长时序任务确有帮助。真实 xArm 的 Push Blocks 94.0 vs. pi0.5 66.1 也很有说服力，因为它对应 OCR 和语义理解能力，而这正是 VL co-training + ECoT 应该改善的部分。

## 七、个人总结

1. ZR-0 的核心不是“让机器人边想边做”，而是“用可语言化的具身推理过程训练 VLM 表征，然后推理时直接做动作”。
2. 最大优势是设计上避开了 CoT 推理的推理延迟和错误累积；最大弱点是消融不够细，难判断收益来自 ECoT 的哪一部分、VL co-training 的哪一部分，还是数据构成差异。
3. 和最近的 LA4VLA、ACE-Ego-0 放在一起看，它们都在试图补足 action-only VLA 监督太薄的问题：LA4VLA 显式强化语言-动作，ACE-Ego-0 扩大人类动作覆盖，ZR-0 则用 dense reasoning 监督强化跨本体共享表征。

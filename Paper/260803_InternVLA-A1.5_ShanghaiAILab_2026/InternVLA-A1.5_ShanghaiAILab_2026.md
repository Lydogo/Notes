# InternVLA-A1.5：用训练期 latent foresight 给 VLA 注入世界模型先验

> 原标题：InternVLA-A1.5: Unifying Understanding, Latent Foresight, and Action for Compositional Generalization  
> 作者：Haoxiang Ma, Junhao Cai, Xiaoxu Xu, Hao Li, Yuyin Yang, Yang Tian, Jiafei Cao, Hongrui Zhu, Zherui Qiu, Zhaxizhuoma, Yuqiang Yang, Jiaqi Peng, Xueyuan Wei, Yangkun Zhu, Jiahao Jiang, Xing Gao, Hanqing Wang, Feng Yuan, Kailin Li, Xueyue Zhu, Tai Wang, Yan Ding, Jiangmiao Pang, Jia Zeng, Jingjing Zhang, Bowen Zhou, Yao Mu, Chunhua Shen, Weinan Zhang  
> 机构：Physical Intelligence Team, Shanghai AI Laboratory  
> 发表：arXiv:2607.04988v1, submitted 2026-07-06；PDF 24 页  
> 链接：https://arxiv.org/abs/2607.04988  
> 项目页：https://internrobotics.github.io/internvla-a15.github.io/  
> 开源：https://github.com/InternRobotics/InternVLA-A-series  
> 权重：https://huggingface.co/collections/InternRobotics/internvla-a15  

---

## 一句话说清楚

InternVLA-A1.5 的关键设计是：保留一个原生 Qwen-3.5 2B VLM 持续做 VQA、subtask 和 FAST action token 训练，同时接一个 460M lightweight unified expert 生成连续 action chunk；训练期用 50 个 learnable foresight tokens 去“查询”冻结 WAN2.2-5B 视频生成模型的未来动态先验，部署时完全丢掉视频分支，只保留实时 flow action policy。

## 一、研究背景与动机

统一机器人模型通常想同时获得两种能力：VLM 的语义理解和 world/action model 的未来动态感。但把这些目标硬塞进同一个训练里会出现三个问题。

第一，很多 VLA 在加入 action 或 future prediction 后不再持续训练 VQA/语言任务，导致 pretrained VLM 的语义和指令跟随能力漂移。第二，语言建模、future latent regression、flow action prediction 的 loss 形式不同，联合训练容易相互干扰。第三，很多 future prediction 模块从零开始学像素或 latent 生成，没有利用大型视频生成模型里已经学到的时空动态先验。

InternVLA-A1.5 的目标是把理解、未来感和动作统一起来，但不让未来生成拖垮实时控制：让 foresight 只在训练期监督 action expert 学到 dynamics-aware representation，部署时不做视频生成。

## 二、核心贡献

1. **提出 VLM preserving 的两阶段训练 recipe**  
   Stage 1 保持 Qwen-3.5 2B VLM 的原生 chat-template 和 next-token objective，在机器人数据和 VQA 数据上共同训练，目标包括 answer、subtask 和 FAST action tokens。

2. **把 future prediction 改写为 latent-querying 问题**  
   不是让 policy 自己生成未来图像，而是用 learnable foresight tokens 产生条件 embedding，输入冻结 WAN2.2-5B 视频生成模型，通过 video loss 反传到 foresight tokens 和 unified expert。

3. **用轻量 unified expert 生成连续 action chunk**  
   Stage 2 加入 460M 参数 expert，经 shared full attention 接收 VLM context，然后用 flow matching 输出 continuous action chunk，避免 Stage 1 autoregressive FAST tokens 的控制延迟和精度限制。

4. **给出清楚的数据配方和采样权重**  
   机器人数据 1.2M episodes / 861M frames，来自 6 个源；multimodal co-training 数据约 3M samples，分 General QA、Box QA、Point QA、Trajectory QA；机器人流和多模态流按固定 0.15:0.85 采样。

5. **仿真、真实、zero-shot 都有较强结果**  
   六个 simulation benchmarks 总体最强或很有竞争力；真实任务中在 held-out instruction bindings 和 MOF 长程化学流程上明显领先；foresight ablation 显示视频监督和 foresight tokens 对 zero-shot/动态任务最有价值。

## 三、方法原理

### 3.1 整体框架

InternVLA-A1.5 是一个 Mixture-of-Transformers 结构，由 VLM backbone 和 unified expert 组成。

| 模块 | 规模/来源 | 输入 | 输出 | 推理时是否保留 |
|---|---|---|---|---|
| VLM backbone | Qwen-3.5 2B | multi-view images, instruction, control mode, discretized state | VQA answer, subtask, optional FAST action tokens, hidden context | 是 |
| Unified expert | 460M，Qwen-3.5-Text 风格小模型 | VLM context, foresight tokens, action query tokens | flow velocity for continuous action chunk | 是 |
| Video generator | WAN2.2-5B, frozen | foresight embeddings as condition, current+future video latent | video flow-matching supervision | 否 |
| FAST action branch | 2048 action vocabulary | action chunk tokenized into discrete sequence | next-token action supervision | 推理时不 decoded |

VLM 和 expert 都有 hybrid attention 结构：3 个 Gated DeltaNet linear attention layers 加 1 个 standard full attention layer。两者只通过 shared full attention layer 交互，各自保留独立 linear attention layers。

### 3.2 关键技术细节

**1. Stage 1：VLM transferring**

每个 robot timestep 的输入包括：

```text
K-view observations o_t
language instruction l
control mode m in {<joint>, <end_effector>, <vqa>}
proprioceptive state q_t in R^D, D <= 32
```

图像按 Qwen3.5 vision pipeline 插入 `<|vision_start|><image_pad><|vision_end|>` tokens；不足的 camera views 被 mask。state 每维用 256 bins 在 `[-1, 1]` 内离散化。action chunk 先用 FAST tokenizer 转成离散 token，action vocabulary size 为 2048，并追加到 VLM 原词表，共享 embedding table 和 language head。

robot sample 的 label 是 subtask description + FAST action segment；VQA sample 的 label 是 answer span。所有 label 都用同一个 next-token cross-entropy，没有额外 head 或 loss weight。

**2. Stage 2：latent foresight + continuous action**

Stage 2 插入 `M` 个 learnable foresight tokens，论文实践中 `M = 50`。这些 tokens 通过 unified expert 读取 VLM context，输出 `Z_t^f`，再映射成 WAN 的 conditioning embedding `C_t^f`。

视频监督不是预测整段像素，而是：

```text
V_t in R^{(1+N) x H_I x W_I x 3}, N = 4 future frames
x_1 = WAN-VAE(V_t)
x_s = (1-s) * x_0 + s * x_1
L_video = || u(x_s, C_t^f, s) - (x_1 - x_0) ||^2
```

WAN 是冻结的，所以梯度只走 conditioning pathway。这是论文最漂亮的点：policy 不学“怎么生成视频”，只学“当前上下文中哪些未来信息能让一个强视频模型生成正确未来”。

连续 action branch 同时训练 flow matching：

```text
a_tau = (1 - tau) * epsilon + tau * a_{t:t+H}
tau ~ Beta(1.5, 1.0)
L_action = || v_theta(a_tau, H_t, Q^f) - (a_{t:t+H} - epsilon) ||^2
```

推理时从 Gaussian action chunk 开始，用 Euler integration 从 `tau=0` 积分到 `1`，得到连续动作块。

**3. Attention masking**

VLM tokens 维持 Qwen3.5 causal attention。unified expert 中 foresight tokens 和 noisy action embeddings 作为不同 token group：

- 组间 causal：foresight 看 VLM context；action embeddings 看 VLM context 和 foresight。
- 组内 bidirectional：foresight tokens 互相看，action tokens 也互相看，符合 flow matching 一次 denoise 整个 chunk 的非自回归性质。
- 训练时 FAST action tokens 存在，但 unified expert 被 mask 不能看 FAST span，防止连续 action generator 偷看 ground-truth discrete action。
- 推理时 FAST action tokens 不 decoded，expert 复用 VLM context KV cache。

### 3.3 训练与优化

| 阶段 | 训练对象 | 数据 | Loss | 超参 |
|---|---|---|---|---|
| Stage 1 Pretrain | VLM backbone | robot + VQA mixture | `L_stage1` next-token CE | batch 1024, lr 5e-5, 300K steps, warmup 2K, wd 0.01, bf16 |
| Stage 2 Pretrain | VLM + unified expert + foresight tokens | robot + VQA mixture + future frames | `L_stage1 + alpha L_video + beta L_action` | batch 1024, lr 5e-5, 600K steps, `alpha=1`, `beta=10`, foresight tokens 50, action chunk 50 |
| Posttrain | downstream policy | benchmark/real demos | 同 Stage 2，可选择保留 video branch 继续调 foresight | batch 128, lr 5e-5 -> 5e-6 cosine, 60K steps |

论文报告 action chunk 为 50；RoboTwin 2.0 appendix 中实际执行 chunk size 为 18，这是 benchmark-specific execution setting，不等同于预训练通用 chunk 长度。

### 3.4 数据使用与维度追踪

#### 3.4.1 数据清单

| 数据源 | 规模 | 样本单位 | 模态/字段 | 关键维度 | 标签/动作 | 用途阶段 | 处理方式 |
|---|---:|---|---|---|---|---|---|
| InternData-A1 | 587,946 episodes / 395.9M frames / weight 0.20 | episode / action chunk | synthetic robot observations, state, action, instruction | unified action slots；具体 action dim 继承 InternVLA-A1，论文未展开 | continuous action, future frames, FAST action tokens | pretrain robot stream | 转入 unified action space；作为 synthetic foundation |
| AgiBotWorld | 112,988 episodes / 206.3M frames / weight 0.25 | episode | real robot multi-view data | morphology-specific slots padded to shared layout | 同上 | pretrain robot stream | real data upweighted |
| UMI | 377,018 episodes / 201.3M frames / weight 0.10 | episode | real-world UMI trajectories | 论文未逐项说明 | 同上 | pretrain robot stream | 统一 action space |
| DROID | 95,658 episodes / 27.6M frames / weight 0.15 | episode | real robot manipulation | 论文未逐项说明 | 同上 | pretrain robot stream | 小源上采样 |
| Galaxea | 19,085 episodes / 25.0M frames / weight 0.20 | episode | real robot manipulation | 论文未逐项说明 | 同上 | pretrain robot stream | 小源上采样 |
| RoboMind 1.0 | 8,638 episodes / 5.4M frames / weight 0.10 | episode | real robot manipulation | 论文未逐项说明 | 同上 | pretrain robot stream | 小源上采样 |
| InternVLA-M1 General QA | 637K samples | image/video QA | captioning, VQA, OCR, knowledge grounding | text answer | next-token text | multimodal co-training | 统一 QA 格式 |
| InternVLA-M1 Box QA | 879K samples | grounding QA | image + referring expression | absolute image bbox coordinates | text-form bbox | multimodal co-training | 统一 QA 格式 |
| InternVLA-M1 Point QA | 832K samples | grounding QA | image + spatial query | absolute image point coordinates | text-form points | multimodal co-training | 统一 QA 格式 |
| InternVLA-M1 Trajectory QA | 684K samples | trajectory QA | image + instruction | 2D end-effector waypoints | waypoint text | multimodal co-training | 统一 QA 格式 |

#### 3.4.2 采样策略

机器人数据本身是长尾分布，最大源 InternData-A1 有 395.9M frames，最小源 RoboMind 1.0 只有 5.4M frames。论文使用两级 grouped sampling：

- 每个 source 是一个 group。
- group 内按 `(#frames)^gamma` 采样，`gamma=1` 退化成 frame-proportional sampling。
- group 间权重先用 Re-Mix 得到，再人工调整，小的 real-world sources 被上采样，synthetic source 被下调。
- robot corpus 和 multimodal corpus 以固定 `0.15:0.85` 混合。

这个比例看起来有点反直觉：多模态样本占大头。作者的解释是保持 VLM semantic/grounding 能力，避免 action/foresight 目标侵蚀语义 backbone。也就是说，InternVLA-A1.5 的训练不是“机器人数据越多越好”，而是把语义稳定性当作控制能力的前置条件。

#### 3.4.3 训练样本形状

**Stage 1 robot sample**

| 字段 | shape / 表示 | 说明 |
|---|---|---|
| Images | `K` views，Qwen3.5 vision tokens | padded views 被 mask；原始分辨率/resize 论文未说明 |
| Instruction | text tokens | task instruction |
| Control mode | `<joint>` / `<end_effector>` | 指定 action space |
| State | `q_t in R^D, D <= 32` | 每维 256-bin over `[-1, 1]`，离散 token |
| Label | subtask text + FAST action tokens | action vocab size 2048 |

**Stage 2 sample**

| 字段 | shape / 表示 | 说明 |
|---|---|---|
| VLM context | image + instruction + state + optional subtask | 提供语义和当前观测 |
| Foresight tokens | `50 x d` learnable tokens | `d` 是 unified expert hidden dim，论文未给具体 hidden size |
| Video target | `(1+N) x H_I x W_I x 3`, `N=4` | 当前帧 + 4 个未来帧，WAN-VAE 编码后算 loss |
| Action target | `a_{t:t+H}`，pretrain `H=50` | 连续 action chunk，用 flow matching |
| FAST tokens | discrete action sequence | 训练 VLM；continuous expert 不能 attend |

**维度快照**

- Observation: `K`-view images；camera count 可变，padded view masked；图像 resize 论文未报告。
- Language: Qwen3.5 native chat template；robot labels 包含 subtask + FAST action；VQA labels 是 answer。
- State: `D <= 32`，uniform 256-bin discretization over `[-1, 1]`。
- Action: Stage 1 FAST tokens，vocab size 2048；Stage 2 continuous action chunk，pretrain chunk length 50；RoboTwin execution chunk size 18。
- Foresight: 50 learnable tokens；future video target 4 future frames；WAN2.2-5B frozen。
- Loss weights: `alpha=1` for video，`beta=10` for action。

## 四、实验与结果

### 4.1 实验设置

真实机器人包含四个任务：Sort Tubes、Insert Tubes、Move Tubes 和 MOF。前三个是 instruction following，通过 held-out tube-target bindings 测 compositional grounding；MOF 是长程 chemistry procedure。所有方法使用同样 demos 和协议，每个 condition 多次随机物体位置试验。部署在单张 NVIDIA RTX 5090 上，静态图、SDPA 和 flash linear attention 下单步推理约 0.1s。

仿真评测覆盖六个 benchmark：

| Benchmark | 关注能力 | 设置 |
|---|---|---|
| LIBERO | single-arm spatial/object/goal/long | 每 suite 10 tasks，50 demos/task，eval 每 suite 500 rollouts |
| LIBERO-Plus | camera/language/layout/lighting 等扰动 | LIBERO checkpoint zero-shot |
| RoboTwin 2.0 | 50 个双臂任务，clean/randomized | full train split，共 27,500 demos；eval 10,000 rollouts |
| DOMINO | dynamic manipulation | RoboTwin checkpoint zero-shot；另报 fine-tuned |
| EBench | mobile manipulation / long horizon / dexterous | 26 tasks，多维泛化 |
| SimplerEnv | WidowX real-to-sim | 4 manipulation tasks |

### 4.2 主要结果

**真实机器人**

| 任务 | pi0.5 | Motus | InternVLA-A1.5 |
|---|---:|---:|---:|
| Sort Tubes | 77.8 | 64.8 | 75.9 |
| Insert Tubes | 51.7 | 44.2 | 72.5 |
| Move Tubes | 72.7 | 56.2 | 80.5 |
| MOF | 29.3 | 0.0 | 76.4 |

要点：

- Sort Tubes 更接近普通 pick-and-place，pi0.5 略高。
- Insert/Move Tubes 要求更精细的目标绑定和孔位/位置 grounding，InternVLA-A1.5 明显领先。
- MOF 是最强证据：76.4% vs pi0.5 29.3%、Motus 0.0，说明 subtask prediction + local dynamics foresight 对长程状态变化有价值。
- held-out instruction bindings 上，InternVLA-A1.5 在三项 instruction-following 任务都最好，说明不是简单 replay seen binding。

**Simulation headline**

| Benchmark | 指标 | InternVLA-A1.5 |
|---|---|---:|
| SimplerEnv WidowX | Avg SR | 80.8 |
| RoboTwin 2.0 | Avg SR | 93.2 |
| DOMINO zero-shot | SR / MS | 27.7 / 39.8 |
| DOMINO fine-tuned | SR / MS | 29.3 / 42.5 |
| LIBERO | Avg SR | 98.9 |
| LIBERO-Plus zero-shot | Total | 84.8 |
| EBench Test | SR / Score | 35.2 / 49.5 |

**LIBERO**

| 方法 | Spatial | Object | Goal | Long | Avg |
|---|---:|---:|---:|---:|---:|
| pi0.5 | 98.8 | 98.2 | 98.0 | 92.4 | 96.9 |
| Xiaomi-Robotics-0 | 98.8 | 100.0 | 98.8 | 97.2 | 98.7 |
| LingBot-VA | 98.5 | 99.6 | 97.2 | 98.5 | 98.5 |
| InternVLA-A1.5 | 98.6 | 99.8 | 98.6 | 98.4 | 98.9 |

**RoboTwin / DOMINO**

| 方法 | RoboTwin Clean | RoboTwin Rand. | RoboTwin Avg | DOMINO zero-shot SR | DOMINO zero-shot MS |
|---|---:|---:|---:|---:|---:|
| pi0.5 | 82.7 | 76.8 | 79.8 | 7.5 | 20.4 |
| InternVLA-A1 | 89.4 | 89.6 | 89.5 | 未报告 | 未报告 |
| LingBot-VA | 92.9 | 91.5 | 92.2 | 24.1 | 36.1 |
| Qwen-VLA-Instruct | 未报告 | 未报告 | 未报告 | 26.6 | 39.5 |
| InternVLA-A1.5 | 93.3 | 93.0 | 93.2 | 27.7 | 39.8 |

### 4.3 消融实验

| 消融 | LIBERO | LIBERO-Plus | RoboTwin | DOMINO |
|---|---:|---:|---:|---:|
| InternVLA-A1.5 | 98.9 | 84.8 | 93.2 | 27.7 |
| w/o video loss | 97.9 | 78.0 | 91.1 | 25.3 |
| w/o foresight tokens | 98.6 | 77.9 | 90.2 | 23.8 |

结论很清楚：foresight 相关设计对 in-distribution LIBERO 帮助有限，但对 LIBERO-Plus 和 DOMINO 这类 distribution shift / dynamic manipulation 更重要。也就是说，它不是主要提高静态 imitation 的拟合上限，而是在扰动和动态场景下提供更好的局部未来感。

训练效率分析显示，在 RoboTwin 同一 SFT 设置下，InternVLA-A1.5 比 pi0.5 和 InternVLA-A1 收敛更快、最终 loss 更低。论文把这解释为预训练 representation 让 downstream adaptation 的优化地形更友好。

## 五、局限性与展望

作者明确指出两个限制：

1. **foresight supervision 只有一个 action chunk 的短 horizon**  
   它学到的是局部 dynamics prior，还不是长程 imagination 或显式 world-model planning。MOF 任务表现好不等于已经解决复杂长程规划。

2. **视频生成模型冻结且通用**  
   WAN2.2 的先验受自身预训练覆盖限制。如果 embodied scene、相机视角、接触状态在视频模型里覆盖不足，foresight token 能查询到的动态也会受限。

我的补充判断是：0.15:0.85 的 robot/multimodal 采样比例很值得关注。它说明 InternVLA-A1.5 把语义保持放在很高优先级，但也可能限制 action-heavy 数据的充分利用。另一个工程缺口是 unified action space 继承自 InternVLA-A1，本文没有展开完整 slot 定义，新平台迁移仍需要读代码和前作。

## 六、灵魂三问

1. **它解决了什么问题？**

它解决的是 unified VLA 中语义理解、未来预测和连续动作生成互相拉扯的问题。传统做法要么牺牲 VLM 语义，要么在部署时做昂贵视频生成，要么让未来预测从零学。InternVLA-A1.5 把 foresight 做成训练期 latent supervision，既吸收世界模型先验，又不增加推理延迟。

2. **为什么这么做？**

因为机器人控制需要未来动态，但实时闭环不适合每步生成视频。learnable foresight tokens 是一个很聪明的折中：让 policy 学会从当前 VLM context 中提取“能被视频模型解释为正确未来”的 latent code，再把这个 code 交给 continuous action head 使用。

3. **什么证据最有说服力？**

最有说服力的是 MOF 真实长程任务和 foresight ablation。MOF 上 InternVLA-A1.5 达 76.4%，明显高于 pi0.5 的 29.3% 和 Motus 的 0.0；去掉 video loss 或 foresight tokens 后，LIBERO-Plus 和 DOMINO 掉幅最大，说明未来动态先验确实主要作用在扰动、动态和长程场景。

## 七、个人总结

1. InternVLA-A1.5 的核心是“训练期借用世界模型，推理期不背世界模型”。这个设计比直接部署 video-action model 更工程友好。

2. 最大优势是语义保持、latent foresight 和连续 action generation 的组合非常干净；最大弱点是 foresight horizon 短、依赖冻结通用视频模型，且 action space 细节没有在本文完全展开。

3. 对 VLA 研究来说，它和 Being-H0.5 形成了有趣对照：Being-H0.5 更强调 human-centric cross-embodiment 数据和 action language，InternVLA-A1.5 更强调 native VLM preservation 与 training-only world-model supervision。

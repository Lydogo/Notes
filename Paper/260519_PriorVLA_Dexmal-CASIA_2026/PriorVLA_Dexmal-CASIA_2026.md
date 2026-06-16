# PriorVLA：面向视觉-语言-动作模型的先验保留式微调

> 原标题：PriorVLA: Prior-Preserving Adaptation for Vision-Language-Action Models
> 作者：Xinyu Guo, Bin Xie（Project leader, Dexmal）, Wei Chai, Xianchi Deng, Tiancai Wang, Zhengxing Wu（中科院自动化所，通讯）, Xingyu Chen（中关村学院，通讯）
> 机构：中科院自动化所 + Dexmal + UCAS + 中关村学院 + 南航
> 发表：arXiv preprint，2026 年 5 月（arXiv:2605.10925v1）
> 链接：https://arxiv.org/abs/2605.10925 ｜ https://priorvla.github.io/

---

## 一、研究背景与动机

- **领域核心问题**：大规模预训练 VLA（π0、π0.5、OpenVLA）作为通用机器人基座已经很强了，但下游任务**必须微调**才能用——而当前主流的"full fine-tuning"做法是把整个模型当成"初始化"，所有参数都更新，结果是把**预训练学到的广泛先验偏移成了狭窄的下游分布模式**，OOD 泛化和 few-shot 能力被严重削弱。
- **现有微调方法的不足**：
  1. **LoRA / 参数受限更新**：只是减少可训练参数量，没有显式保护 forward-pass 里的预训练表征；
  2. **冻 VLM + 训 action 侧**（KI、π0.5 部分思路）：保护了语言部分，但 action expert 本身的预训练 motor prior 还是会被改写；
  3. **Continual learning 正则化 / 参数 merge**：以"约束"为主，没有提供"读取并复用"预训练先验的机制。
- **本文出发点**：把预训练 VLA 看成**两类先验源**而非初始化——VLM 提供 **scene prior**（任务相关的视觉语言结构），action expert 提供 **motor prior**（动作生成规律）。微调应该是"学会**读取并整合**这两类先验"，而不是"用少量数据把它们覆写掉"。

## 二、核心贡献

1. **重新框定 VLA 微调范式**：从"用预训练做初始化"转向"保留并复用预训练 forward-pass 先验"。
2. **PriorVLA 框架**，两个耦合设计：
   - **Dual Action Experts（DAE）**：把预训练 AE 复制成两份——一份冻结作 **Prior Expert**（只读先验源），一份训练作 **Adaptation Expert**（下游专用）。
   - **Expert Queries（EQ）**：3 组可学习 token——Scene Queries 从 VLM 抓 scene prior，Motor Queries 从冻结的 PE 抓 motor prior，Action Queries 在 AE 内部把两者整合。
3. **以 25% 的可训练参数量**全面超越 full fine-tuning，在 RoboTwin 2.0-Hard 上 +11，LIBERO 上 99.1%，真机 8 任务上 ID/OOD 分别 +12/+16，few-shot 10 demo 设置下 +24/+22。

## 三、方法原理

### 3.1 整体框架

```
        ┌──────────────────────────────────────────┐
        │                  VLM (frozen)            │
图像 + 文本 + state → [OBS tokens] + [Scene Queries(SQ)] → 抓 scene prior
        └──────────────────────────────────────────┘
                       │ KV cache (scene prior)
                       ▼
        ┌─────────────────────┐  ┌─────────────────────┐
噪声 ã → │  Prior Expert (PE)  │  │ Adaptation Expert    │ ← Action Queries (AQ)
        │       FROZEN        │  │      (trainable)     │
        │  + Motor Queries(MQ)│  │  ← Scene Queries KV  │
        └─────────┬───────────┘  │  ← Motor Queries KV  │
                  │ KV cache     └──────────┬───────────┘
                  │ (motor prior)            │
                  └──────────────────────────┘
                          ▼
          只用 AE 的 denoising output 更新轨迹 → 最终 action chunk
```

构建在 **π0.5** 基座之上（SigLIP + Gemma-2B VLM + Gemma-300M flow-matching AE）。PE 的输出**不参与最终动作，也不参与 loss**，它只是个 read-only forward path，其内部表征通过 Motor Queries 暴露给 AE。

### 3.2 关键技术细节

#### (1) Dual Action Experts：分离"保留先验" vs"任务特化"

预训练 AE 复制两份，参数都从同一个 checkpoint 初始化：

- **Prior Expert（PE，冻结）**：保留预训练 denoising dynamics 完整不动，作为 motor prior 的只读来源。
- **Adaptation Expert（AE，训练）**：负责下游 action 生成，唯一驱动 denoising 轨迹更新的分支。

在每一步去噪 $\tau$，两个 expert 都拿同一个噪声 chunk $\tilde a^\tau$，但**只有 AE 的输出 $f^\tau_{Ada}$ 被用来更新轨迹**：

$$\tilde a^{\tau+1} = \text{FM}(\tilde a^\tau, f^\tau_{Ada})$$

PE 在轨迹上跑一遍只是为了产生 motor-prior 表征，输出直接丢弃。

#### (2) Expert Queries：通过 attention mask 实现的"先验读取接口"

3 组可学习 token，通过精心设计的 attention mask（Fig. 3）限定信息流向：

**Scene Queries (SQ)**：插在 VLM 输入序列里，自注意力可看 OBS 和其他 SQ：

$$h^{l+1}_{sq} = \text{Attn}(Q^l_{sq}, K^l_{obs}\|K^l_{sq}, V^l_{obs}\|V^l_{sq})$$

其层级 KV cache 就是给 AE 用的 **scene prior 接口**。

**Motor Queries (MQ)**：插在 PE 里。**关键设计**：PE 的噪声 action token 只能看 OBS 和自己，**不能看 SQ 和 MQ**（保护预训练 denoising 路径不被污染）；MQ 自注意力 + 单向读 PE 的噪声 action token：

$$h^{l+1}_{mq} = \text{Attn}(Q^l_{mq}, K^l_{mq}\|K^l_{\tilde a_{pe}}, V^l_{mq}\|V^l_{\tilde a_{pe}})$$

MQ 的 KV cache 就是 **motor prior 接口**。这种"单向只读"设计既保住了 PE 的预训练前向路径不变，又能把内部表征导出。

**Action Queries (AQ)**：和 AE 的噪声 action token 在同一个 block 里双向自注意力，并能读 OBS / SQ / MQ：

$$h^{l+1}_{aq,\tilde a_{ae}} = \text{Attn}(Q^l_{aq,\tilde a_{ae}}, K^l_{aq,\tilde a_{ae}}\|K^l_{obs}\|K^l_{sq}\|K^l_{mq}, V^l_{aq,\tilde a_{ae}}\|V^l_{obs}\|V^l_{sq}\|V^l_{mq})$$

**关键设计**：AE 不能直接看 PE 的原始噪声 action 状态（KV），所有 motor 信息必须经 MQ 这个**压缩接口**进来。实证表明直接访问 PE 原始状态会让训练不稳定，MQ 提供了一个更紧凑稳定的桥梁。

> AQ 本身不解码成动作，它专门负责"组织多源先验"，让 AE 的噪声 action token 拿到整合好的特征再做 denoising。

#### (3) 为什么 MQ 不能看 SQ / VLM prefix？

作者在 Appendix A.4 给出原因：VLM prefix 有大量 token，PE 的 action 路径只有少数 token；如果让 MQ 同时看 VLM prefix，MQ 会被 scene 特征**淹没**（dominated by scene features），失去"motor prior 专属接口"的作用。所以用 attention mask 强制 MQ 只看 PE 的 action 表征。

### 3.3 训练与优化

- **可训练参数**：Adaptation Expert + 3 组 Expert Queries + VLM vision encoder；其余 VLM 参数和 Prior Expert 全部冻结。**总可训参数 ≈ full fine-tuning 的 25%**。
- **损失**：标准 flow-matching MSE，只施加在 AE 的 denoising prediction 上。PE 输出在训练和推理都丢弃，从不进入 loss；EQ 也没有任何辅助 loss。
- **优化器**：AdamW + global-norm grad clip = 1.0；可训参数 fp32 存，冻结参数 bf16 存。
- **分组学习率乘子**：SQ ×2.0，MQ ×4.0，AQ ×4.0，其余 ×1.0（EQ 是新初始化的，需要更大 lr 跟上）。
- **关键超参**：
  - RoboTwin 2.0：H=50, batch=32, peak lr 2.5e-5, 30k steps, EMA 0.99
  - LIBERO：H=10, batch=256, peak lr 5e-5, 30k steps, EMA 0.999
  - 真机：H=50, batch=32, peak lr 2.5e-5, 30k steps, EMA 0.99
- **训练时长**（8×H20 或 8×A100）：RoboTwin 单任务 ~5.6h，真机单任务 ~5h，LIBERO 单 suite ~23.6h；**全部比 π0.5 baseline 还快**（因为可训参数少）。

## 四、实验与结果

### 4.1 实验设置

- **三类基准**：
  - **RoboTwin 2.0**（13 task subset，Aloha-AgileX 双臂仿真）：Easy = ID，Hard = OOD（背景随机化 + 桌面杂乱 + 光照 + 桌高扰动），每任务 300 trials。
  - **LIBERO**（4 suite × 10 task，Franka 单臂）：每 task 50 rollouts。
  - **真机**（8 task × 2 embodiment）：Franka 单臂（D435 third-person + D435 wrist）+ AC-One 双臂（top-view D435 + 双 D405 wrist）。每任务 20 ID + 20 OOD trials。OOD 同时扰动 4 个维度：Light / Background / Object Position / Table Height（升高 2cm）。
- **基线**：DP、RDT、π0、π0-FAST、π0.5、GR00T-N1/N1.7、OpenVLA-OFT、DreamVLA、UniVLA、MemoryVLA、F1、DD-VLA、GE-Act 等。
- **数据规模**：标准 50 demos/任务（RoboTwin）、few-shot 10 demos、large 250 demos。

### 4.2 主要结果

#### RoboTwin 2.0 仿真（Tab. 1）

| 方法 | Easy 平均 | Hard 平均 |
|---|---|---|
| DP | 36 | 0 |
| RDT | 44 | 17 |
| π0 | 62 | 22 |
| π0.5 | 67 | 42 |
| **PriorVLA** | **77 (+10)** | **53 (+11)** |

13 个任务里 PriorVLA 几乎全面领先，OOD（Hard）提升最显著。

#### 数据规模消融（Tab. 2）

| 数据规模 | π0.5 Easy / Hard | PriorVLA Easy / Hard |
|---|---|---|
| Few (10 demos) | 29 / 20 | **41 (+12) / 31 (+11)** |
| Standard (50) | 67 / 42 | **77 (+10) / 53 (+11)** |
| Large (250) | 89 / 59 | 88 (-1) / **65 (+6)** |

**核心规律**：Easy 上 large-data 时差距消失（数据多了 prior 不那么关键），但 Hard 上**全部数据规模下 PriorVLA 都赢**，说明先验保留对 OOD 始终有效。

#### LIBERO（Tab. 3）

| 方法 | Spatial | Object | Goal | Long | Avg |
|---|---|---|---|---|---|
| π0.5 | 98.8 | 98.2 | 98.0 | 92.4 | 96.9 |
| OpenVLA-OFT | 97.6 | 98.4 | 97.9 | 94.5 | 97.1 |
| **PriorVLA** | **99.4** | **99.8** | **99.4** | **97.6** | **99.1** |

即使在已经饱和的 benchmark 上仍能继续涨点。

#### 真机实验（Tab. 4 / 5）

| 设置 | 评估 | π0.5 | PriorVLA |
|---|---|---|---|
| Standard data | ID | 69 | **81 (+12)** |
| Standard data | OOD | 41 | **57 (+16)** |
| Few-shot (10 demos) | ID | 24 | **48 (+24)** |
| Few-shot (10 demos) | OOD | 10 | **32 (+22)** |

**few-shot OOD 是真正的"杀手锏"**：只用 10 条 demo，OOD 成功率从 10% 提到 32%（3.2x），完美对应论文"data-efficient + generalizable"的卖点。

### 4.3 消融实验

#### Prior Expert 消融（Tab. 6a）

| 变体 | 参数量 | Easy | Hard |
|---|---|---|---|
| w/o PE（去掉 PE→AE 通路） | 0.85B | 75 | 42 |
| Random PE（用随机权重冻住） | 0.85B | 75 | 43 |
| Trainable PE（PE 也训练） | 1.28B | 73 | 44 |
| **Full PriorVLA** | 0.85B | **77** | **49** |

**关键发现**：
- 把 PE 换成**随机权重**几乎没差（43 vs 42 Hard）——证明收益**真的来自预训练 motor prior**，而不是"多加了一个 frozen 分支"的正则化效果；
- 把 PE 也训练反而更差——证明"保留稳定先验源"比"两个 expert 都适配"更有效；
- w/o PE 和 w/o MQ 数值几乎相同（PE 只通过 MQ 接到 AE，所以等价）。

#### Expert Queries 消融（Tab. 6b）

| SQ | MQ | AQ | Easy | Hard |
|---|---|---|---|---|
| ✗ | ✗ | ✗ | 61 | 28 |
| ✗ | ✓ | ✓ | 70 | 30 |
| ✓ | ✗ | ✓ | 75 | 42 |
| ✓ | ✓ | ✗ | 71 | 43 |
| **✓** | **✓** | **✓** | **77** | **49** |

- **去掉 SQ 对 Hard 影响最大**（49→42），说明 scene prior 对 OOD 泛化的贡献最关键；
- 全部去掉 EQ 时性能崩盘（28% Hard），证明**只有 frozen PE 而没有 learnable 接口是没用的**——必须有 query 才能"用上"先验。

#### VQA 探针（Appendix G）

定性实验显示：**full fine-tuning 后的模型对 VQA 问题（识别 / 计数 / 描述）输出乱码**（"Ꮁ媄𝖨퐹牁"这种），而 PriorVLA 微调后仍能正确回答。直接证明 full fine-tuning 会**破坏 VLM 的语言生成能力**，而 PriorVLA 因为冻住了 VLM 主干所以保留了通用能力。

## 五、局限性与展望

- **作者讨论的局限**：
  1. RoboTwin 2.0 只跑了 13 个任务（每个任务一个独立模型，全 50 个太贵）；
  2. 真机 OOD 是 4 个维度**联合扰动**（贴近部署但难以解耦各因素贡献）；
  3. **推理多了 PE 的 forward 开销**——虽然 chunked control 下可控，但确实增加了部署成本；
  4. 缺少对 scene prior 和 motor prior 在不同 layer / denoising step 上**如何涌现、互动、演化**的细粒度分析。
- **潜在改进方向**：
  - 把 PE 的 forward 做**缓存或蒸馏**，去掉推理开销；
  - 跨任务训练而非"每任务一个模型"，看能否进一步扩展；
  - 用更系统的 probing 实验研究 prior 在 transformer 不同层的分布；
  - 把 PriorVLA 思路推广到非 flow-matching 的 VLA（如 FAST tokenizer 的离散 action）。

## 六、灵魂三问

1. **它解决了什么问题？**
   之前对 VLA 做下游微调，要么 full fine-tune 把预训练的广泛先验搞坏（OOD 直接崩、VQA 输出乱码），要么 LoRA 只是减参数没保护表征；PriorVLA 用"冻结 Prior Expert + 可学习 Expert Queries"的接口设计，**显式把预训练 forward-pass 的 scene 和 motor 先验当成可读取资源**，做到了 25% 参数训练就能在 OOD 和 few-shot 上吊打 full fine-tune（真机 few-shot OOD 从 10% 干到 32%）。

2. **为什么这么做？**
   最关键的设计选择是**把同一个预训练 AE 复制成 frozen PE + trainable AE 两份**——不是凭空加 frozen 分支（消融里 Random PE 几乎无效），而是利用预训练 AE 里**已经学会的 motor dynamics**作为只读的"参考答案"。配合 MQ 这个单向接口（不让 MQ 看 VLM prefix，避免被 scene 特征淹没），让 motor prior 干净地流到 AE。比"冻 VLM 训 action 侧"（KI）多了一层 motor 侧的先验保护——KI 只能管 scene，PriorVLA 同时管 scene + motor。

3. **什么证据最有说服力？**
   **Tab. 6a 的 Random PE 消融**：把 Prior Expert 的权重换成随机数（保持参数量、保持 frozen、保持架构），Hard 性能从 49% 掉到 43%，跟"完全去掉 PE"几乎一样。这直接排除了"只是加了个 frozen 正则化"的假说——**收益就是来自预训练 motor prior 本身**。配合**真机 few-shot OOD 10%→32%**（Tab. 5），证明这个设计在"数据最少 + 分布最远"的最难场景下增益最大，恰好对准论文核心 motivation。

## 七、个人总结

1. **核心 idea**：把 VLA 微调从"覆写预训练"重新定义为"读取预训练 forward-pass 先验"。具体实现是 **Dual Action Experts（一冻一训）+ 三组 Expert Queries（SQ/MQ/AQ）**，用 attention mask 把 scene prior 和 motor prior 单向导出给 trainable 分支。架构上是 π0.5 的"侧分支增强版"，思想上是把 LoRA / KI / continual learning 的目标统一成了"显式先验复用"。

2. **优势与不足**：
   - **优势**：(1) 25% 参数训练但全面赢 full fine-tune，性价比极高；(2) OOD 和 few-shot 增益压倒性（few-shot OOD 真机 3.2x），完美对接实际部署痛点；(3) VQA 实验直接证明预训练能力**真的被保住了**（其他方法都没这么直观的证据）；(4) 训练比 baseline 更快（参数少）。
   - **不足**：(1) 推理多一个 PE forward，部署有额外开销，作者没给出缓存/蒸馏的优化方案；(2) 每任务训一个独立模型，没探索"跨任务共享 PriorVLA"；(3) RoboTwin 13/50 任务覆盖率不全；(4) 整篇 paper 的 ablation 主要在 RoboTwin 上做，LIBERO 和真机的细粒度消融较少。

3. **启发**：
   - **对研究**：把 VLA 微调思路从"参数高效"（LoRA）推进到"**表征高效**"——既然 forward-pass 里有现成的好东西，就别覆盖它，**搭个接口读出来用**。这个思想完全可以迁移到其他大模型微调场景（LLM agent、VLM 多模态推理）。
   - **对面试**：和 KI（π0.7）形成鲜明对比——KI 是"插一层冻住 gradient"，PriorVLA 是"复制一份当只读源 + learnable query 读"；KI 偏 scene，PriorVLA scene+motor 都覆盖。两篇放一起讲能很好地展示"如何在 VLA 微调里保护预训练"的两条技术路线。
   - **对实际应用**：如果手头是 π0.5 base + 少量真机数据要做下游适配，PriorVLA 的"冻一份 + train 一份 + 接 query"思路可以直接复用，特别适合**真机数据稀缺 + 真实部署有 OOD 风险**的场景。配合 RoboTwin 2.0 这种合成数据可以进一步放大优势。

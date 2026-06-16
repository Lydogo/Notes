# GR00T N1：面向通用人形机器人的开源基础模型

> 原标题：GR00T N1: An Open Foundation Model for Generalist Humanoid Robots
> 作者：NVIDIA GEAR / GR00T 团队（研究负责人 Linxi "Jim" Fan, Yuke Zhu）
> 发表：arXiv:2503.14734v2，2025-03
> 链接：https://arxiv.org/abs/2503.14734

---

## 一、研究背景与动机

- **核心问题**：人形机器人需要一个"通用大脑"，但与 LLM/VLM 不同，机器人没有"互联网级"的人形数据可用；任何单一硬件的轨迹量都远不够撑起通用模型。
- **现有方法不足**：
  - Open X-Embodiment 等跨 embodiment 数据集存在严重的"数据孤岛"——传感器、自由度、控制模式各异，难以直接训成一个统一策略。
  - 纯模拟生成（MimicGen / DexMimicGen）有 sim2real gap；纯真机遥操作成本和时间线性增长。
  - 已有 VLA（RT-2, π0 等）多数还停留在单一/少量 embodiment，且对"无动作的人类视频"利用不充分。
- **目标**：构建一个**开源、跨 embodiment、能从异构数据混合训出来**的通用人形 VLA 基础模型，下游用少量数据即可适配新任务。

## 二、核心贡献

1. **双系统 VLA 架构**：System 2（Eagle-2 VLM）做感知/语义理解 + System 1（DiT + flow matching）做高频动作生成，端到端联合训练，VLM 与 DiT 通过 **cross-attention** 解耦（区别于 π0 的 mixture-of-experts）。
2. **数据金字塔训练范式**：把"真机轨迹（顶）+ 模拟轨迹 + 神经生成轨迹（中）+ 人类第一视角视频（底）"统一进同一个 flow matching 目标里。
3. **Latent action + IDM 双路伪标签**：用 VQ-VAE 在所有视频上学一份共享的潜在动作码本，使"无动作"的人类视频也能像一种新 embodiment 一样参与训练；同时训练 IDM 生成连续伪动作。
4. **数据规模**：50k H100 GPU 小时预训练；总数据 **592.9M 帧 / 约 8,376 小时**，其中神经生成视频 827 小时（用 WAN2.1-I2V-14B + LoRA 把 88 小时真机数据扩 10×）。开源 GR00T-N1-2B checkpoint、数据与基准。

## 三、方法原理

### 3.1 整体框架

输入：

- 多视角 RGB 图像（224×224，pixel shuffle 后每帧 64 个 token）
- 任务语言指令
- 机器人本体感知状态 $q_t$（不同 embodiment 维度不同）
- 噪声动作 chunk $A_t^\tau$（chunk 长度 $H=16$）

输出：去噪后的 $H$ 步动作 chunk。

```
图像 ─┐
      ├─► Eagle-2 VLM (System 2, 10Hz) ──► φ_t (第 12 层中间嵌入)
文本 ─┘                                            │
                                                  ▼ cross-attn
状态 q_t ──► State Encoder (embodiment-specific MLP) ─┐
                                                       ├─► DiT × N (self-attn) ──► Action Decoder ──► a_t … a_{t+H-1}
噪声 A_t^τ ─► Action Encoder (含 τ 编码) ─────────────┘                                  (120Hz)
```

模型参数 2.2B，其中 VLM 1.34B；推理 16 步动作 chunk 在 L40 上约 63.9ms。

### 3.2 关键技术细节

**(1) Vision-Language 模块（System 2）**

- 用 NVIDIA **Eagle-2** VLM（SmolLM2 LLM + SigLIP-2 图像编码器）。
- 关键经验：取 LLM **第 12 层**的中间嵌入而不是最后一层——更快、下游成功率反而更高。直觉：最后一层过度向"下 token 预测"特化，丢失了对动作生成有用的稠密视觉/空间信息。

**(2) Diffusion Transformer 模块（System 1）**

- DiT 块交替使用 self-attention（动作 + 状态自关注）和 cross-attention（与 VLM 输出 φ_t 交互），结构借鉴 Flamingo / VIMA。
- 每个 embodiment 一套独立的 state encoder MLP + action encoder MLP + action decoder MLP（处理不同维度的状态/动作空间）；DiT 主干共享。

**(3) Flow Matching 目标**

给定真值动作 chunk $A_t$、时间步 $\tau \in [0,1]$、噪声 $\epsilon \sim \mathcal{N}(0, I)$：

$$A_t^\tau = \tau A_t + (1-\tau)\epsilon$$

$$\mathcal{L}_{fm}(\theta) = \mathbb{E}_\tau \left[ \| V_\theta(\varphi_t, A_t^\tau, q_t) - (\epsilon - A_t) \|^2 \right]$$

直觉：模型不直接预测动作，而是预测从噪声到真值的"流速场" $\epsilon - A_t$。采用 $p(\tau) = \text{Beta}((s-\tau)/s; 1.5, 1),\ s=0.999$，让训练偏向小 $\tau$（更难的、更接近真值的去噪步）。

推理：从 $A_t^0 \sim \mathcal{N}(0, I)$ 出发，用前向 Euler 迭代 **K=4 步**即可：

$$A_t^{\tau + 1/K} = A_t^\tau + \frac{1}{K} V_\theta(\varphi_t, A_t^\tau, q_t)$$

**(4) Latent Action 码本（跨 embodiment 关键）**

- 训练一个 VQ-VAE：encoder 吃 $x_t, x_{t+H}$ 输出 latent action $z_t$；decoder 吃 $z_t, x_t$ 重建 $x_{t+H}$。
- 训练时不同 embodiment 的视频共享同一份码本 → 学到一份"通用 latent 动作语义"（论文 Fig. 4：不同机器人 + 人手"右臂向左移动"对应同一个 latent）。
- 推理时把 encoder 当 IDM 用，给视频打 latent 标签；这些标签被当作"LAPA"这种特殊 embodiment 参与 flow matching 训练。

**(5) Neural Trajectory（神经轨迹扩增）**

- 用 WAN2.1-I2V-14B + LoRA 在 88h 真机数据上微调，给定初始帧 + 新语言 prompt 生成"反事实视频"，扩到 827 小时。
- 再用商用多模态 LLM 做帧采样判别（8 帧）→ 过滤不符合指令的视频 → 失败的重新打 caption。
- 这些视频没有动作真值 → 用 **latent action** 或单独训的 **IDM** 打伪动作标签。

**(6) 与 π0 的关键架构区别**

| 维度 | π0 | GR00T N1 |
|---|---|---|
| VLM ↔ action expert 耦合 | mixture-of-experts（共享 attention） | cross-attention |
| 动作模块 | flow matching expert | DiT + flow matching |
| 多 embodiment 处理 | 同一套 head | embodiment-specific MLP projector |
| 优点 | 参数共享、attention 中信息密度高 | VLM/动作模块可独立换、对多 embodiment 友好 |

### 3.3 训练与优化

**预训练**

- 50,000 H100 GPU 小时，最多 1024 GPU；NVIDIA OSMO 调度 + Ray 容错。
- batch 16,384，gradient steps 200,000，lr 1e-4 (cosine, warmup 0.05)。
- **冻结**：VLM 的 text tokenizer；**解冻**：vision encoder + DiT + 所有投影头。
- 真机数据用真实动作 + latent action 双标签；人类视频只用 latent action；神经轨迹用 latent + IDM。

**后训练（每个下游 embodiment 单独微调）**

- batch 128 或 1024，steps 20k–60k；其余超参与预训练一致。
- 单 A6000 即可微调：只调 adapter + DiT 时 batch 可到 200；连同 vision encoder 一起调时 batch ≤ 16。
- 可选 **co-training with neural trajectories**：1:1 混合真机数据 + 神经轨迹。

**辅助 loss**：OWL-v2 检测出目标物体 bbox，模型从 VLM 最后一层 token 上线性回归归一化中心坐标，$L_{det} = \|x_{pred} - x_{gt}\|^2$，总 loss = $L_{fm} + L_{det}$。

## 四、实验与结果

### 4.1 实验设置

- **三个模拟基准**：RoboCasa Kitchen（24 task，Franka 单臂）、DexMimicGen（9 task，双臂/灵巧手/GR-1）、自建 GR-1 Tabletop（24 task，GR-1 + Fourier 灵巧手）。
- **真机基准**：Fourier GR-1，4 类任务：Pick-and-Place / Articulated / Industrial / Multi-Agent Coordination，共 13 个子任务。
- **基线**：BC-Transformer（RoboMimic）、Diffusion Policy。
- **数据消融**：模拟用 30 / 100 / 300 demo/task；真机用 10% / 全量。
- **评估**：模拟 100 trials 取最近 5 ckpt 最佳；真机 10 trials/task。

### 4.2 主要结果

**模拟基准（100 demos/task 平均成功率）**

| 模型 | RoboCasa | DexMG | GR-1 | 平均 |
|---|---|---|---|---|
| BC-Transformer | 26.3% | 53.9% | 16.1% | 26.4% |
| Diffusion Policy | 25.6% | 56.1% | 32.7% | 33.4% |
| **GR00T-N1-2B** | **32.1%** | **66.5%** | **50.0%** | **45.0%** |

GR-1 Tabletop 上领先 DP **17.3%**，预训练价值最显著。

**真机 GR-1 humanoid（13 任务平均）**

| 模型 | Pick-Place | Articulated | Industrial | Coordination | 平均 |
|---|---|---|---|---|---|
| Diffusion Policy (10%) | 3.0% | 14.3% | 6.7% | 27.5% | 10.2% |
| Diffusion Policy (Full) | 36.0% | 38.6% | 61.0% | 62.5% | 46.4% |
| **GR00T-N1-2B (10%)** | 35.0% | 62.0% | 31.0% | 50.0% | **42.6%** |
| **GR00T-N1-2B (Full)** | **82.0%** | **70.9%** | **70.0%** | **82.5%** | **76.8%** |

关键观察：**GR00T-N1 用 10% 数据 = 42.6%，已逼近 DP 用 100% 数据的 46.4%**，仅差 3.8 个点 → 大规模异构预训练显著提升数据效率。

**预训练 zero-shot 评估（双手协调 + 新物体新容器）**

- 双手协调 pick-place（左手交右手放架子）：76.6% (11.5/15)。
- 未见过的物体放未见过的容器：73.3% (11/15)。

### 4.3 消融实验

**Neural Trajectory 协训**

| 设置 | 提升 |
|---|---|
| RoboCasa 30 demo | +4.2% |
| RoboCasa 100 demo | +8.8% |
| RoboCasa 300 demo | +6.8% |
| 真机 GR-1（8 task 平均） | +5.8% |

**LAPA（latent） vs IDM 伪标签**

- 低数据（30）：LAPA 略优——IDM 训练数据少，伪动作不准。
- 高数据（100、300）：IDM 反超且差距拉大——更多数据使 IDM 与真值更对齐。
- 经验：真机高数据场景只用 IDM 即可。

## 五、局限性与展望

- **任务范围**：目前局限于短时程桌面操作，未覆盖 loco-manipulation（移动 + 操作）。
- **VLM backbone**：可换更强的 VLM 提升空间推理 / 语言理解 / 自适应能力。
- **神经轨迹质量**：视频生成模型仍难保证多样性 + 物理一致性同时满足，counterfactual 场景的物理合理性是瓶颈。
- **架构探索**：还需要研究新架构和预训练策略以进一步提升鲁棒性与泛化。

## 六、灵魂三问

1. **它解决了什么问题？**
   之前没有一个开源 VLA 能同时吃下"多种真机 + 模拟 + 神经生成视频 + 人类第一视角视频"四类异构数据并跨多 embodiment 工作；GR00T N1 把它们统一进同一个 flow matching 训练管线，下游用 10% 数据就能逼近 baseline 用全量数据的水平。

2. **为什么这么做？**
   最关键的设计选择不是 DiT 也不是双系统，而是 **latent action 码本 + embodiment-specific projector + cross-attention**。这三件事一起允许"无动作的视频"和"动作语义完全不同的机器人"被映射到同一个 loss 上——而不是像 π0 那样需要 MoE 共享 attention、也不是像 Octo 那样把 VLM 当冻结特征提取器。换句话说：他们没用 MoE 是因为想保留"换 VLM、换 action expert"的灵活性；他们用 latent action 是因为只有这样才能塞进 Ego4D 这种没动作的海量人类视频。

3. **什么证据最有说服力？**
   真机 GR-1 上 **GR00T-N1 (10% 数据) = 42.6% vs DP (100% 数据) = 46.4%**，只差 3.8 个点；同时 DP (10%) 只有 10.2%。这一个对照同时证明了：① 预训练有效（同模型 10% 数据飙升 4 倍），② 模型架构本身也强（同等数据下完胜 DP），③ 数据效率是真的——这是评价"基础模型"最直接的指标。

## 七、个人总结

1. **最核心 idea**：把"数据金字塔（真机 + 模拟 + 神经 + 人类视频）"通过 **latent action 码本 + IDM 伪动作**变成统一格式，再用 **VLM cross-attention DiT + embodiment-specific MLP** 的双系统架构端到端训出来。本质是"工程化地把所有能用的视频都塞进一个 flow matching loss"。
2. **优势**：开源、跨 embodiment 易扩展、数据效率高、cross-attention 比 MoE 解耦更灵活；预训练 + 少量微调的范式在真机上得到充分验证。
3. **不足**：仍限于短时程桌面操作；神经生成视频的物理一致性是隐忧；latent action 的语义可解释性弱；没有真正的"通用对话/规划"能力（VLM 只做感知）。
4. **对项目的启发**：
   - **Latent action 码本**思路完全可以借鉴到自家 ego 数据 Pretrain——用同样的 VQ-VAE 把人手视频和机器人视频对齐到共享 latent 空间，减少 sim2real / human2robot 差距。
   - **Embodiment-specific projector** 是处理多机器人/多 owner 数据的简洁方案，可直接套用到 pico_ego 多采集者数据上。
   - **VLM 中间层取 embedding** 这个细节是免费午餐：自家 PI0 / PI0.5 也值得试一下取 SmolLM/Gemma 第 ~12 层而非最后一层，看下游成功率是否能涨。
   - **视频生成做反事实扩增**值得跟进——用 WAN/CogVideo 把 88h 数据扩 10× 这条路在 ego 数据上同样适用，配合 IDM 打伪标签就能补足"高重要性帧不足"的问题（与 FrameSkip 思路互补）。

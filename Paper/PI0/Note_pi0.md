# $π_0$: Vision-Language-Action (VLA) Model Notes

## 前言
最近闲下来了开始整理一下之前看过的论文，先从VLA$π_0$模型开始，本文仅对整体的模型框架进行整理，按照我自己的思路对整个架构进行了自上而下的分解，同时补充了一些细节，看完后应该可以对vla技术有大致上的理解，至少面试的时候扯几句是没问题的。

论文题目：$π_0$: A Vision-Language-Action Flow Model for General Robot Control
论文地址：https://arxiv.org/abs/2410.24164
代码仓库:https://github.com/Physical-Intelligence/openpi

## Layer 1: VLA 模型概览 (Vision-Language-Action)

$π_0$ 是一个典型的 VLA（视觉-语言-动作）模型，旨在将视觉感知、语言指令和机器人本体状态直接映射为具体的控制动作。

### 1.1 模型输入与输出

**输入 (Observation $o_t$)**
模型在时刻 $t$ 接收以下多模态输入：
1.  **视觉观测量 ($I_t$)**：包含多帧 RGB 图像（每台机器人配置 2~3 个相机）。
2.  **文本指令 ($\ell_t$)**：语言 Token 序列，描述任务目标。
3.  
4.  **机器人状态 ($q_t$)**：机器人的本体感受状态（如关节角向量）。

形式化表示：
$$o_{t}=[I_{t}^{1}, ..., I_{t}^{n}, \ell_{t}, q_{t}]$$

**输出 (Action Chunk $A_t$)**
模型输出一个**未来动作块**，包含未来 $H$ 个时间步的动作序列（本研究中 $H=50$）：
$$A_{t}=[a_{t}, a_{t+1}, ..., a_{t+H-1}]$$

### 1.2 架构总览

$π_0$ 的核心架构由一个预训练的 VLM（视觉语言模型）和一个专门的 Action Expert 组成。

![pi0framework_Overview](pi0framework_Overview.png)

*   **Backbone**: Pretrained VLM (SigLIP 300M + Gemma 2.6B)
*   **Head**: Action Expert (300M Parameters)

---

## Layer 2: 核心组件详解 (SigLIP + Gemma + Action Expert)

$π_0$ 的处理流程可以概括为：**SigLIP 编码视觉 -> Gemma 理解语义 -> Action Expert 生成动作**。

### 2.1 Vision Encoder: SigLIP
*   **作用**：将摄像头捕捉的原始像素 (Pixels) 压缩成具有高层语义的 Visual Tokens。它将海量的像素压缩成了具有高层语义的向量，极大地降低了机器人学习新任务所需的动作数据量。
*   **处理流程**：
    1.  所有图像缩放到 224 尺寸。
    2.  通过 SigLIP 将图像切分为固定大小的 Patches 并生成 Tokens。
    3.  Tokens **不进行 Pooling**，直接保留空间特征作为 Gemma 的输入，最多支持3张图片。
*   **优势**：相比于 CLIP，SigLIP 在处理大规模视觉数据时效率更高，能够为后续模型提供“画面左上角有个杯子”这类细粒度的语义信息。

### 2.2 VLM Backbone: Gemma
*   **作用**：作为大脑，负责语义对齐与常识推理。它理解“把杯子拿起来”这一指令的含义，并结合视觉信息进行推理。
*   **工作机制**：
    *   接收 SigLIP 输出的 Visual Tokens 和文本指令 Tokens。
    *   **KV Cache**：在推理阶段，Gemma 只进行一次前向传播（Prefill），计算并缓存 KV Cache，不直接输出动作，而是将上下文信息传递给 Action Expert。
    *   **Attention Mask**：通过 `mask_ar` 数组控制序列中 Token 的可见性（区分双向注意力与因果注意力）。
*   **配置参考**：
    ```python
    if variant == "gemma_300m":
        # 311M params
        return Config(
            width=1024,
            depth=18,
            mlp_dim=4096,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
        )
    if variant == "gemma_2b":
        return Config(
            width=2048,
            depth=18,
            mlp_dim=16_384,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
        )
    ```

### 2.3 Action Expert
*   **作用**：在 VLM 宽泛的语义引导下，专注于计算高频、精细的动作细节。
*   **输入**：
    *   Robot State Token（机器人当前状态）。
    *   Action Token（带噪声的动作序列）。
    *   来自 Gemma 的上下文信息（通过 KV Cache）。
*   **生成机制**：使用 **Flow Matching** 算法。它不直接预测动作值，而是预测一个**速度向量 (Vector Field)**，指明如何修正当前的噪声，使其逐步逼近真实的物理动作轨迹。

---

## Layer 3: 深度解析与补充

本部分对上述组件的原理与实现细节进行补充。

### 3.1 视觉预训练：SigLIP vs CLIP

**CLIP (Contrastive Language-Image Pre-training)**
*   **原理**：使用 **Softmax Loss**。将图片和文字映射到同一空间，计算 Batch 内所有 $N \times N$ 对的相似度。对于某一张图片 $v_i$，它与正确文字 $t_i$ 的匹配概率是用 Softmax 计算的：
    $$L_{v \to t} = -\log \frac{\exp(\text{sim}(v_i, t_i) / \tau)}{\sum_{j=1}^{N} \exp(\text{sim}(v_i, t_j) / \tau)}$$
*   **局限**：分母部分 $\sum_{j=1}^{N} \exp(...)$ 表明，Softmax 强制要求必须把一张图片和当前 Batch 里所有的文字都对比一遍，进行全局归一化（概率和为1）。这导致计算量大且难以扩展到超大 Batch。

**SigLIP (Sigmoid Loss for Language-Image Pre-training)**
*   **原理**：使用 **Sigmoid Loss**。将每一对图文匹配视为独立的“二分类”问题（是/否匹配），不再要求正样本在整个 Batch 中胜出。
    $$L = - \sum_{i,j} \log \sigma (z_{ij} \cdot \text{Label})$$
*   **优势**：
    1.  **解耦**：不需要全局归一化，计算第 $i$ 对的损失时，完全不需要看第 $k$ 对的数据，易于大规模并行训练。
    2.  **性能**：实验证明，这种简单的“判断题”逻辑在处理大规模视觉数据时，比 CLIP 的“多选题”逻辑学得更快、更准。

### 3.2 VLM 核心机制

#### Attention Mask 机制
通过 `mask_ar` 数组，可以灵活控制一个长序列 Token 中哪些可以相互看见（bi-directional），哪些只能看左边（causal）。这对于融合不同模态的输入至关重要。

```python
def make_attn_mask(input_mask, mask_ar):
    """Adapted from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` bool[?B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: bool[?B, N] mask that's true where previous tokens cannot depend on
        it and false where it shares the same attention mask as the previous token.
    """
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)
```

#### Image & Text Encoding (KV Cache Prefill)
在推理开始时，模型首先对前缀（图像+文本）进行一次前向传播，计算并填充 KV Cache。这个 Cache 会被后续的 Action Expert 复用，避免了重复计算。

```python
# first fill KV cache with a forward pass of the prefix
observation = _model.preprocess_observation(None, observation, train=False)
prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
positions = jnp.cumsum(prefix_mask, axis=1) - 1
# 调用底层 LLM (Gemma)，只为计算 kv_cache
_, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)
```

### 3.3 Action Expert 实现细节

#### Token 处理
**Robot State Token**: 直接通过一个线性层映射得到。
```python
state_token = self.state_proj(obs.state)[:, None, :]
```

**Action Token**: 将带噪声的动作 `noisy_actions` 和 Flow Matching 的时间步 `timestep` 结合。时间步通过正弦余弦位置编码嵌入，然后与动作 Token 拼接，经过一个 MLP 混合。
```python
action_tokens = self.action_in_proj(noisy_actions)
# embed timestep using sine-cosine positional encoding
time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)

# mix timestep + action information using an MLP
time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)
action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
action_time_tokens = self.action_time_mlp_in(action_time_tokens)
action_time_tokens = nnx.swish(action_time_tokens)
action_time_tokens = self.action_time_mlp_out(action_time_tokens)
action_expert_tokens = action_time_tokens
```

#### Flow Matching
**基本原理**: Flow Matching 学习一个向量场 $v_t(x)$，它定义了从噪声分布 $x_1$ 到真实数据分布 $x_0$ 的路径。
*   路径定义: $x_t = (1-t)x_0 + t x_1$
*   预测目标 (速度): $\frac{dx_t}{dt} = x_1 - x_0$

**推理过程 (Inference)**: Action Expert 通过一个 `while_loop` 循环迭代地修正噪声，生成最终动作。关键在于，这个循环只调用小型的 Action Expert，并复用 VLM 的 KV Cache，实现了高效推理。

```python
# Action Expert在不重新运行VLM的情况下工作
def step(carry):
    x_t, time = carry
    # 嵌入后缀 (动作 + 时间)
    suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
        observation, x_t, jnp.broadcast_to(time, batch_size)
    )
    
    # 构造 Attention Mask
    suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
    prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
    full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
    
    positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
    
    # 调用小模型 (Action Expert) 进行推理，利用 KV Cache
    # 第1个参数是None，表示只对后缀进行计算
    (prefix_out, suffix_out), _ = self.PaliGemma.llm(
        [None, suffix_tokens],
        mask=full_attn_mask,
        positions=positions,
        kv_cache=kv_cache, # 复用 VLM 的 Cache
        adarms_cond=[None, adarms_cond],
    )
    
    # 预测速度向量 v_t
    v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])
    
    # 更新状态 (Euler Step)
    return x_t + dt * v_t, time + dt

def cond(carry):
    x_t, time = carry
    return time >= -dt / 2

# 从纯噪声 (time=1.0) 开始循环去噪
x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
return x_0
```

**损失计算 (Loss)**: 训练时，模型的目标是让预测的速度向量 $v_{\theta}$ 逼近真实的速度 $u_t = \text{noise} - \text{actions}$。
公式:
$$L^{\tau}(\theta)=\mathbb{E}_{p\left(A_{t} | o_{t}\right), q\left(A_{t}^{\tau} | A_{t}\right)}\left\| v_{\theta}\left(A_{t}^{\tau}, o_{t}\right)-u\left(A_{t}^{\tau} | A_{t}\right)\right\| ^{2}$$

```python
def compute_loss(
    self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
) -> at.Float[at.Array, "*b ah"]:
    preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
    observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

    # 构造带噪声的动作 x_t 和目标速度 u_t
    batch_shape = actions.shape[:-2]
    noise = jax.random.normal(noise_rng, actions.shape)
    time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
    time_expanded = time[..., None, None]
    x_t = time_expanded * noise + (1 - time_expanded) * actions
    u_t = noise - actions

    # 一次大的前向传播，同时计算前缀和后缀
    prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
    suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)
    input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
    ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
    attn_mask = make_attn_mask(input_mask, ar_mask)
    positions = jnp.cumsum(input_mask, axis=1) - 1
    
    (prefix_out, suffix_out), _ = self.PaliGemma.llm(
        [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond]
    )
    v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

    # 计算预测速度 v_t 和真实速度 u_t 的均方误差
    return jnp.mean(jnp.square(v_t - u_t), axis=-1)
```

---

## 流程总结

$π_0$ 模型的工作流程可以拆解为以下几个步骤：

1.  **视觉编码 (SigLIP)**：看到杯子。
2.  **语义理解 (Gemma VLM)**：理解指令“把杯子拿起来”，并结合视觉信息进行常识推理。
3.  **动作细化 (Action Expert)**：在 VLM 的引导下，专注计算拿起杯子所需的具体动作特征。
4.  **轨迹生成 (Flow Matching)**：将抽象的动作特征转化为平滑、物理真实的运动轨迹，将随机信号“推”向正确的动作序列。


## 参考资料
*   **Flow Matching 讲解**: [https://www.bilibili.com/video/BV142C2BSEbm/](https://www.bilibili.com/video/BV142C2BSEbm/)
*   **pi0 论文讲解**: [https://www.bilibili.com/video/BV142uFzWEdY/](https://www.bilibili.com/video/BV142uFzWEdY/)

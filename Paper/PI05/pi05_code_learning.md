# OpenPI / π₀.₅ 代码架构学习笔记

## 第一步：项目总览与核心抽象层

### 1.1 项目是什么

openpi 是 Physical Intelligence 开源的机器人 VLA（Vision-Language-Action）模型库，包含三种模型：

| 模型 | 类型 | 核心思路 |
|------|------|----------|
| **π₀** | Flow Matching VLA | 用 flow matching 生成连续动作序列 |
| **π₀-FAST** | 自回归 VLA | 用 FAST tokenizer 将动作离散化后自回归生成 |
| **π₀.₅** | Flow Matching VLA（升级版） | 在 π₀ 基础上引入 **knowledge insulation** 和架构改进，提升开放世界泛化能力 |

**关键点**：π₀.₅ **不是**一个独立的模型文件，而是通过 `Pi0Config(pi05=True)` 在 π₀ 的代码中切换启用。

### 1.2 目录结构全景

```
openpi/
├── scripts/                    # 入口脚本
│   ├── train.py               # JAX 训练入口
│   ├── train_pytorch.py       # PyTorch 训练入口
│   └── serve_policy.py        # 推理服务入口（WebSocket）
│
├── src/openpi/                # 核心库
│   ├── models/                # 模型定义（JAX/Flax NNX）
│   │   ├── model.py          #   基类：ModelType, BaseModelConfig, BaseModel, Observation
│   │   ├── pi0_config.py     #   Pi0Config（含 pi05 开关）
│   │   ├── pi0.py            #   Pi0 模型实现（π₀ 和 π₀.₅ 共用）
│   │   ├── pi0_fast.py       #   Pi0-FAST 模型
│   │   ├── gemma.py          #   Gemma 语言模型骨干
│   │   ├── siglip.py         #   SigLIP 视觉编码器
│   │   ├── tokenizer.py      #   PaliGemma / FAST tokenizer
│   │   └── lora.py           #   LoRA 适配器
│   │
│   ├── models_pytorch/        # PyTorch 版模型实现
│   │   ├── pi0_pytorch.py    #   PI0Pytorch（含 pi05）
│   │   └── gemma_pytorch.py  #   Gemma PyTorch 版
│   │
│   ├── policies/              # 策略层：连接模型与具体机器人
│   │   ├── policy.py         #   Policy 基类
│   │   ├── policy_config.py  #   create_trained_policy() 工厂
│   │   ├── aloha_policy.py   #   ALOHA 机器人输入/输出映射
│   │   ├── droid_policy.py   #   DROID 机器人输入/输出映射
│   │   └── libero_policy.py  #   LIBERO 仿真输入/输出映射
│   │
│   ├── training/              # 训练基础设施
│   │   ├── config.py         #   所有 TrainConfig 定义（pi05_aloha, pi05_droid 等）
│   │   ├── data_loader.py    #   数据加载
│   │   ├── optimizer.py      #   优化器
│   │   ├── checkpoints.py    #   checkpoint 保存/加载
│   │   └── weight_loaders.py #   预训练权重加载
│   │
│   ├── transforms.py          # 数据变换管线
│   ├── serving/               # WebSocket 推理服务
│   └── shared/                # 共享工具
│       ├── normalize.py      #   动作归一化
│       ├── image_tools.py    #   图像处理
│       └── download.py       #   checkpoint 下载
│
├── examples/                   # 各机器人平台使用示例
│   ├── aloha_real/            # ALOHA 真机
│   ├── droid/                 # DROID（Franka）
│   ├── libero/                # LIBERO 仿真
│   └── simple_client/         # 最简推理客户端
│
└── packages/openpi-client/     # 独立客户端 SDK
```

### 1.3 核心抽象层：model.py

整个架构围绕 `src/openpi/models/model.py` 中定义的几个核心抽象构建：

#### ModelType 枚举

```python
class ModelType(enum.Enum):
    PI0 = "pi0"
    PI0_FAST = "pi0_fast"
    PI05 = "pi05"
```

这是全局模型类型标识，后续的 policy、transform、config 都根据它来做分支判断。

#### Observation 数据结构

```python
@struct.dataclass
class Observation(Generic[ArrayT]):
    images: dict[str, Float[ArrayT, "*b h w c"]]       # 图像输入，key 为相机名
    image_masks: dict[str, Bool[ArrayT, "*b"]]          # 图像有效性掩码
    state: Float[ArrayT, "*b s"]                        # 机器人低维状态（关节角等）
    tokenized_prompt: Int[ArrayT, "*b l"] | None        # 语言指令 token
    tokenized_prompt_mask: Bool[ArrayT, "*b l"] | None  # 语言 token 掩码
```

**设计意图**：模型始终接收统一格式的输入，与具体机器人平台解耦。不同机器人的差异由 policy 层的 transform 来处理。

标准图像输入有三路：
- `base_0_rgb`：基座相机
- `left_wrist_0_rgb`：左腕相机
- `right_wrist_0_rgb`：右腕相机

如果某路相机不存在，通过 `image_masks` 屏蔽即可，模型仍然能正常运行。

#### Actions 类型

```python
Actions = Float[ArrayT, "*b ah ad"]  # ah=action_horizon, ad=action_dim
```

动作是一个 `[batch, 时间步数, 动作维度]` 的张量。π₀.₅ 用 flow matching 去噪声生成这个连续动作序列。

#### BaseModelConfig 基类

```python
@dataclasses.dataclass(frozen=True)
class BaseModelConfig(abc.ABC):
    action_dim: int       # 动作空间维度
    action_horizon: int   # 动作序列长度（一次预测多少步）
    max_token_len: int    # 语言 prompt 最大 token 数

    def create(self, rng) -> BaseModel: ...       # 创建模型实例
    def load(self, params) -> BaseModel: ...      # 从 checkpoint 加载
    def load_pytorch(self, ...) -> PI0Pytorch: ... # 加载 PyTorch 版
    def inputs_spec(self, ...) -> (Observation, Actions): ... # 输入形状规格
```

这是**工厂模式**：Config 既是配置容器，又负责创建对应的模型实例。

#### BaseModel 基类

```python
@dataclasses.dataclass
class BaseModel(nnx.Module, abc.ABC):
    def compute_loss(self, rng, observation, actions, *, train=False) -> loss: ...
    def sample_actions(self, rng, observation, **kwargs) -> Actions: ...
```

所有模型必须实现两个方法：
- `compute_loss`：训练时计算损失
- `sample_actions`：推理时采样动作

### 1.4 Pi0Config：π₀ 与 π₀.₅ 的分水岭

`Pi0Config` 继承自 `BaseModelConfig`，是理解 π₀.₅ 的关键：

```python
@dataclasses.dataclass(frozen=True)
class Pi0Config(BaseModelConfig):
    paligemma_variant: Variant = "gemma_2b"        # 语言模型骨干（2B 参数）
    action_expert_variant: Variant = "gemma_300m"   # 动作专家（300M 参数）
    action_dim: int = 32
    action_horizon: int = 50
    pi05: bool = False                              # ← 核心开关
    discrete_state_input: bool = None               # 是否将状态离散化为 token
```

当 `pi05=True` 时，有三个关键行为变化：

| 特性 | π₀ (`pi05=False`) | π₀.₅ (`pi05=True`) |
|------|-------------------|---------------------|
| **状态输入方式** | 连续向量，作为 action expert 的前缀输入 | 离散化为文本 token，拼接到语言 prompt 中 |
| **时间步注入方式** | 标准 RMSNorm | **adaRMSNorm**（自适应归一化，用 flow matching 时间步调制） |
| **max_token_len** | 48 | 200（因为 token 中包含了状态信息，序列更长） |

`__post_init__` 中的自动推断逻辑：

```python
def __post_init__(self):
    if self.max_token_len is None:
        object.__setattr__(self, "max_token_len", 200 if self.pi05 else 48)
    if self.discrete_state_input is None:
        object.__setattr__(self, "discrete_state_input", self.pi05)
```

### 1.5 数据流总览（从输入到输出）

```
用户输入 (机器人特定格式)
    │
    ▼
Policy 层 (aloha_policy / droid_policy / libero_policy)
    │  ── 将特定格式映射为统一的 Observation
    ▼
Transforms (transforms.py)
    │  ── 图像 resize、归一化、prompt tokenize、状态归一化
    ▼
Model (pi0.py / pi0_pytorch.py)
    │  ── SigLIP 编码图像 → Gemma 处理语言+图像 → Action Expert 生成动作
    ▼
Transforms (逆变换)
    │  ── 动作反归一化
    ▼
Policy 层 (输出映射)
    │  ── 转为机器人特定的控制命令
    ▼
机器人执行
```

### 1.6 本步小结

| 概念 | 作用 | 文件位置 |
|------|------|----------|
| `ModelType` | 全局模型类型枚举 | `models/model.py` |
| `Observation` | 统一输入格式 | `models/model.py` |
| `Actions` | 统一输出格式 | `models/model.py` |
| `BaseModelConfig` | 模型配置+工厂基类 | `models/model.py` |
| `BaseModel` | 模型实现基类（loss + sample） | `models/model.py` |
| `Pi0Config` | π₀/π₀.₅ 配置，`pi05=True` 切换 | `models/pi0_config.py` |
| Policy 层 | 连接具体机器人与统一模型接口 | `policies/` |
| Transforms | 数据预处理/后处理管线 | `transforms.py` |

**下一步预告**：深入 `pi0.py`，看 π₀.₅ 的模型架构——SigLIP 视觉编码、Gemma 语言处理、Action Expert 以及 Flow Matching 是如何组合在一起的。

---

## 第二步：深入 π₀.₅ 模型架构实现（pi0.py）

`pi0.py` 只有 280 行，但浓缩了整个 VLA 的核心。π₀ 和 π₀.₅ 共用这个文件，通过 `self.pi05` 标志切换行为。

### 2.1 模型初始化：`__init__` 中的组件组装

```python
# pi0.py 第 66-103 行
class Pi0(BaseModel):
    def __init__(self, config: Pi0Config, rngs: nnx.Rngs):
        self.pi05 = config.pi05

        # ① 创建双专家 Gemma
        llm = _gemma.Module(
            configs=[paligemma_config, action_expert_config],  # 两套权重
            adarms=config.pi05,                                 # π₀.₅ 启用 adaRMS
        )

        # ② 创建 SigLIP 视觉编码器
        img = _siglip.Module(num_classes=paligemma_config.width, variant="So400m/14")

        self.PaliGemma = nnx.Dict(llm=llm, img=img)

        # ③ 投影层（因 pi05 不同而分叉）
        self.action_in_proj  = Linear(action_dim → expert_width)    # 动作输入投影（共用）
        self.action_out_proj = Linear(expert_width → action_dim)    # 动作输出投影（共用）

        if config.pi05:
            self.time_mlp_in  = Linear(expert_width → expert_width) # 时间步 MLP（用于 adaRMS）
            self.time_mlp_out = Linear(expert_width → expert_width)
        else:
            self.state_proj        = Linear(action_dim → expert_width)     # 状态投影
            self.action_time_mlp_in  = Linear(2*expert_width → expert_width) # 动作+时间 混合MLP
            self.action_time_mlp_out = Linear(expert_width → expert_width)
```

**π₀.₅ 的架构核心**：模型由四个组件构成：

| 组件 | 作用 | 参数量级 |
|------|------|----------|
| **SigLIP** (`So400m/14`) | 将 224×224 图像编码为 256 个视觉 token | ~400M |
| **PaliGemma** (`gemma_2b`) | 语言+视觉融合的主干 Transformer | ~2B |
| **Action Expert** (`gemma_300m`) | 处理动作 token 的专用 Transformer | ~300M |
| **投影层** | 动作/时间步的输入输出映射 | 很小 |

### 2.2 双专家架构：Gemma 的 "configs 列表" 机制

这是 openpi 最精妙的设计。`gemma.py` 中的 `Module` 接收一个 **configs 列表**：

```python
# gemma.py 第 340-343 行
class Module(nn.Module):
    configs: Sequence[Config]  # 每个 expert 一个 config
```

在 pi0.py 中，传入了两个 config：
- `configs[0]` = `gemma_2b`（PaliGemma 主干，处理图像+语言）
- `configs[1]` = `gemma_300m`（Action Expert，处理动作 token）

**每一层 Transformer Block 内部**，两套权重各自独立处理各自的 token，但**共享同一个 attention 的 KV cache**：

```
Block 内部流程（gemma.py Block.__call__）：

xs = [paligemma_tokens, action_expert_tokens]  # 两组 token

for i, x in enumerate(xs):
    x = RMSNorm[i](x, adarms_cond[i])   # 各自的 LayerNorm

pre_attn = [norm_pg, norm_ae]
post_attn, kv_cache = Attention(pre_attn, ...)  # ← 共享注意力计算！

for i, (x, config) in enumerate(zip(xs, configs)):
    x = RMSNorm[i](x, adarms_cond[i])
    x = FFN[i](x)                        # 各自的 FFN
```

**关键理解**：两个 expert 在同一个 Transformer 的每一层中**并行运行**，各自有独立的 LayerNorm 和 FFN 权重，但在 Attention 中通过共享 KV 实现跨 expert 的信息交互。Action Expert 的 token 可以 attend 到 PaliGemma 的图像/语言 token，从而获取视觉-语言上下文来生成动作。

### 2.3 Token 序列拼接：prefix 与 suffix

模型将所有输入组织为**两段 token 序列**：

#### Prefix（`embed_prefix`）：视觉 + 语言

```python
# pi0.py 第 106-137 行
def embed_prefix(self, obs):
    tokens = []
    # ① 每路图像 → SigLIP → 256 个视觉 token
    for name in obs.images:
        image_tokens, _ = self.PaliGemma.img(obs.images[name])
        tokens.append(image_tokens)     # [b, 256, 2048]
        ar_mask += [False] * 256        # 图像 token 互相可见（双向注意力）

    # ② 语言 prompt → Gemma embedder → 文本 token
    tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
    tokens.append(tokenized_inputs)
    ar_mask += [False] * token_len      # 与图像互相可见（双向注意力）
```

三路图像（各 256 token）+ 语言 prompt = prefix 序列，全部使用**双向注意力**（`ar_mask=False`）。

**π₀.₅ 的关键差异**：状态信息（关节角等）被离散化为文本 token，混入了这段 prefix 的语言 prompt 里。而 π₀ 的状态是连续向量，放在 suffix 中。

#### Suffix（`embed_suffix`）：动作 + 时间步（+ π₀ 的状态）

```python
# pi0.py 第 139-186 行
def embed_suffix(self, obs, noisy_actions, timestep):
    # ===== π₀ 独有：状态作为连续 token =====
    if not self.pi05:
        state_token = self.state_proj(obs.state)[:, None, :]   # [b, 1, width]
        tokens.append(state_token)
        ar_mask += [True]  # prefix 不能看到 state（单向）

    # ===== 动作 token（π₀ 和 π₀.₅ 共用）=====
    action_tokens = self.action_in_proj(noisy_actions)         # [b, horizon, width]
    time_emb = posemb_sincos(timestep, ...)                    # 正弦余弦编码时间步

    # ===== 时间步注入方式分叉 =====
    if self.pi05:
        # π₀.₅：时间步经 MLP → 作为 adaRMS 条件，不直接拼入 token
        time_emb = swish(time_mlp_in(time_emb))
        time_emb = swish(time_mlp_out(time_emb))
        adarms_cond = time_emb                     # 传给 Gemma 的 RMSNorm
        action_expert_tokens = action_tokens       # 纯动作 token
    else:
        # π₀：时间步与动作 concat 后经 MLP 混合
        action_time = concat([action_tokens, time_tokens], dim=-1)
        action_expert_tokens = swish(mlp(action_time))
        adarms_cond = None                         # 不用 adaRMS

    ar_mask += [True] + [False] * (horizon - 1)   # prefix 不能看动作，动作 token 之间互相可见
```

### 2.4 adaRMSNorm vs 标准 RMSNorm：π₀.₅ 的关键创新

**标准 RMSNorm**（π₀ 使用，`cond=None`）：

```python
# gemma.py 第 113-125 行
class RMSNorm:
    def __call__(self, x, cond):
        normed = x / sqrt(mean(x²) + ε)
        if cond is None:
            return normed * (1 + scale), None      # 仅学习一个 scale 参数
```

**adaRMSNorm**（π₀.₅ 使用，`cond=time_emb`）：

```python
        # cond 不为 None 时，走 adaptive 分支
        modulation = Dense(cond)                   # cond → [scale, shift, gate]
        scale, shift, gate = split(modulation, 3)
        normed = normed * (1 + scale) + shift      # 用时间步动态调制
        return normed, gate                        # gate 用于残差连接
```

区别的直觉：
- **标准 RMSNorm**：每层的归一化参数是固定的，时间步信息只能通过 token 内容隐式传递
- **adaRMSNorm**：时间步直接调制每一层的 scale/shift/gate，在**每个 Transformer 层**都显式告诉 Action Expert "当前去噪到什么程度了"。这类似于 DiT（Diffusion Transformer）中的 adaptive LayerNorm

gate 机制用于残差连接：

```python
# gemma.py 第 453-459 行
def _gated_residual(x, y, gate):
    if gate is None:
        return x + y           # 标准残差（π₀）
    return x + y * gate        # 门控残差（π₀.₅），gate 由时间步控制
```

### 2.5 注意力掩码设计：谁能看到谁

`make_attn_mask` 通过 `ar_mask` 构建灵活的注意力模式：

```python
# ar_mask 中 False=双向可见，True=只能被后面的 token 看到

# 完整序列的 ar_mask 示例（π₀.₅）：
# [img1: 0,0,...,0 | img2: 0,0,...,0 | img3: 0,0,...,0 | lang: 0,0,...,0 | actions: 1,0,...,0]
#  ←── prefix（全部双向）──→                                              ←── suffix ──→
```

信息流向规则：

| 源 → 目标 | 是否可见 | 原因 |
|-----------|---------|------|
| 图像 ↔ 图像 | ✅ 双向 | 图像 token 互相参考 |
| 图像 ↔ 语言 | ✅ 双向 | VLM 需要跨模态融合 |
| 图像/语言 → 动作 | ✅ 可见 | 动作生成需要感知上下文 |
| 动作 → 图像/语言 | ❌ 不可见 | 感知不应被当前噪声动作污染 |
| 动作 ↔ 动作 | ✅ 双向 | 动作序列内部需要协调 |

### 2.6 训练与推理的前向传播对比

#### 训练（`compute_loss`）

```python
# pi0.py 第 189-214 行
def compute_loss(self, rng, observation, actions, *, train=False):
    noise = normal(actions.shape)
    time = beta(1.5, 1) * 0.999 + 0.001            # 偏向大 t 采样
    x_t = time * noise + (1 - time) * actions       # 线性插值
    u_t = noise - actions                            # 真实速度

    # prefix + suffix 拼接后做一次完整前向传播
    (prefix_out, suffix_out), _ = self.PaliGemma.llm(
        [prefix_tokens, suffix_tokens], mask=attn_mask,
        adarms_cond=[None, adarms_cond]             # PaliGemma 不用 adaRMS，Action Expert 用
    )
    v_t = self.action_out_proj(suffix_out[:, -action_horizon:])

    return MSE(v_t, u_t)                            # 预测速度 vs 真实速度
```

训练时 prefix 和 suffix **一次性拼接** 送入 Gemma，效率高。

#### 推理（`sample_actions`）

```python
# pi0.py 第 217-279 行
def sample_actions(self, rng, observation, *, num_steps=10):
    # 第一步：prefix 单独前向，缓存 KV cache
    _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], ...)

    # 第二步：循环 10 步去噪，每次只算 suffix
    def step(carry):
        x_t, time = carry
        suffix_tokens = embed_suffix(obs, x_t, time)
        (_, suffix_out), _ = self.PaliGemma.llm(
            [None, suffix_tokens],     # prefix 传 None，用 KV cache
            kv_cache=kv_cache,         # 复用缓存
            adarms_cond=[None, adarms_cond],
        )
        v_t = action_out_proj(suffix_out[:, -action_horizon:])
        return x_t + dt * v_t, time + dt   # 欧拉积分

    x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))  # t: 1→0
    return x_0
```

推理时 prefix 只算一次，10 次去噪迭代只计算 suffix 部分，**KV cache 复用**是关键的效率优化。

### 2.7 整体架构图

```
                          π₀.₅ 模型架构
═══════════════════════════════════════════════════════════

输入层
────────────────────────────────────────────────────────
  base_0_rgb ──→ SigLIP ──→ 256 tokens ─┐
  left_wrist ──→ SigLIP ──→ 256 tokens ─┼──→ Prefix tokens
  right_wrist ─→ SigLIP ──→ 256 tokens ─┤     (双向注意力)
  语言 prompt ──→ Gemma Embed ──→ N tokens ─┘
  [π₀.₅: 状态也在这里，作为文本 token]

                    ┌─── PaliGemma (2B) 处理 ───┐
                    │    图像+语言 token          │
Gemma Transformer   │                            │ ← 共享 Attention KV
(18层, 共享结构)    │                            │
                    └─── Action Expert (300M) ───┘
                         动作 token
                         [π₀.₅: adaRMSNorm 注入 time]
                         [π₀:   时间步拼入 token]

输出层
────────────────────────────────────────────────────────
  suffix_out[:, -action_horizon:] ──→ action_out_proj ──→ v_t (速度预测)
      │
      │  × 10步欧拉积分（推理时）
      ▼
  去噪后的动作序列 [batch, 50, 32]
```

### 2.8 本步小结

| 概念 | 实现 | 文件位置 |
|------|------|----------|
| SigLIP 视觉编码 | `So400m/14`，每图输出 256 token | `siglip.py` |
| 双专家 Gemma | configs 列表 `[2B, 300M]`，共享 Attention KV | `gemma.py` |
| adaRMSNorm | `cond` 不为 None 时用时间步调制 scale/shift/gate | `gemma.py` RMSNorm |
| Prefix (embed_prefix) | 图像 + 语言，双向注意力 | `pi0.py` L106-137 |
| Suffix (embed_suffix) | 动作 + 时间步（π₀.₅: adaRMS，π₀: concat MLP） | `pi0.py` L139-186 |
| 训练 (compute_loss) | 构造 x_t → 预测速度 v_t → MSE(v_t, u_t) | `pi0.py` L189-214 |
| 推理 (sample_actions) | KV cache 复用 + 10 步欧拉积分 | `pi0.py` L217-279 |
| 门控残差 | `x + y * gate`（π₀.₅），`x + y`（π₀） | `gemma.py` _gated_residual |

**下一步预告**：训练配置与数据管线——`training/config.py` 中的 `TrainConfig`、`transforms.py` 中的数据预处理、以及 Policy 层如何将具体机器人数据映射为模型输入。

---

## 第三步：训练配置、数据管线与 Policy 层

本步以 **LIBERO** 和 **DROID** 两个具体平台为例，带入实际代码讲解数据如何从原始数据集经过层层变换，最终变成模型能消费的 `Observation` + `Actions`。

### 3.1 TrainConfig：以 pi05_libero 为例

`training/config.py` 中每个 `TrainConfig` 就是一套完整的训练/推理方案。看 `pi05_libero` 的实际定义：

```python
# training/config.py L743-763
TrainConfig(
    name="pi05_libero",
    # ① 模型配置：π₀.₅，action_horizon=10（一次预测10步），关闭状态离散化
    model=pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False),
    # ② 数据配置：LIBERO 数据集 + 数据变换
    data=LeRobotLiberoDataConfig(
        repo_id="physical-intelligence/libero",
        base_config=DataConfig(prompt_from_task=True),
        extra_delta_transform=False,
    ),
    # ③ 训练超参
    batch_size=256,
    lr_schedule=CosineDecaySchedule(warmup_steps=10_000, peak_lr=5e-5, ...),
    optimizer=AdamW(clip_gradient_norm=1.0),
    ema_decay=0.999,
    # ④ 预训练权重
    weight_loader=CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
    num_train_steps=30_000,
)
```

对比 `pi05_droid` 的配置：

```python
# training/config.py L629-641
TrainConfig(
    name="pi05_droid",
    model=pi0_config.Pi0Config(action_horizon=15, pi05=True),  # DROID 预测 15 步
    data=SimpleDataConfig(
        assets=AssetsConfig(asset_id="droid"),
        data_transforms=lambda model: _transforms.Group(
            inputs=[droid_policy.DroidInputs(model_type=ModelType.PI05)],  # ← DROID 的变换
            outputs=[droid_policy.DroidOutputs()],
        ),
        base_config=DataConfig(prompt_from_task=True),
    ),
)
```

**关键区别**：两个配置的 `model` 都是 `Pi0Config(pi05=True)`，但 `data` 字段完全不同——不同机器人需要不同的数据变换管线。

### 3.2 数据变换管线：三层架构

数据从原始格式到模型输入经过三层变换。以 LIBERO 训练为例，看 `LeRobotLiberoDataConfig.create` 的实际代码：

```python
# training/config.py L291-355（LeRobotLiberoDataConfig.create 方法）
def create(self, assets_dirs, model_config):
    # ===== 第一层：repack_transforms（训练专用，键名映射）=====
    repack_transform = _transforms.Group(inputs=[
        _transforms.RepackTransform({
            "observation/image": "image",             # LeRobot 数据集的字段名
            "observation/wrist_image": "wrist_image",  #   ↓
            "observation/state": "state",              # 映射为统一的字段名
            "actions": "actions",                      #   ↓
            "prompt": "prompt",                        # 给后续变换使用
        })
    ])

    # ===== 第二层：data_transforms（训练+推理共用，机器人特定）=====
    data_transforms = _transforms.Group(
        inputs=[libero_policy.LiberoInputs(model_type=model_config.model_type)],
        outputs=[libero_policy.LiberoOutputs()],
    )

    # ===== 第三层：model_transforms（由 ModelTransformFactory 自动生成）=====
    model_transforms = ModelTransformFactory()(model_config)

    return DataConfig(
        repack_transforms=repack_transform,
        data_transforms=data_transforms,
        model_transforms=model_transforms,
        ...
    )
```

这三层变换的执行顺序是：`repack → data_transforms → Normalize → model_transforms`。

### 3.3 第一层 repack_transforms 代码详解

**作用**：把不同数据集的字段名映射为统一名称。

```python
# transforms.py L80-101
@dataclasses.dataclass(frozen=True)
class RepackTransform(DataTransformFn):
    structure: at.PyTree[str]  # 映射规则

    def __call__(self, data: DataDict) -> DataDict:
        flat_item = flatten_dict(data)                     # 把嵌套字典展平
        return jax.tree.map(lambda k: flat_item[k], self.structure)  # 按规则重组
```

LIBERO 数据集原始格式是 `"observation/image"`，repack 后变成 `"image"`。之所以需要这步，是因为不同数据集的命名习惯不同（DROID 用 `"observation/exterior_image_1_left"`，ALOHA 用 `"observation.images.cam_high"`），但后续的 `LiberoInputs` / `DroidInputs` 期望看到统一的键名。

**只在训练时使用**——推理时数据直接来自机器人环境，键名由 policy 层控制。

### 3.4 第二层 data_transforms 代码详解：Policy 文件

这是整个管线最关键的一层，定义在 `policies/` 目录下。每个机器人平台一个文件，每个文件包含 `Inputs`（输入变换）和 `Outputs`（输出变换）两个类。

#### 例 1：LiberoInputs（libero_policy.py L29-83）

```python
# libero_policy.py
@dataclasses.dataclass(frozen=True)
class LiberoInputs(transforms.DataTransformFn):
    model_type: _model.ModelType  # 根据模型类型做不同处理

    def __call__(self, data: dict) -> dict:
        # 步骤 1：解析图像格式
        # LeRobot 存储格式是 float32 [C,H,W]，需要转为 uint8 [H,W,C]
        base_image = _parse_image(data["observation/image"])       # [224,224,3] uint8
        wrist_image = _parse_image(data["observation/wrist_image"]) # [224,224,3] uint8

        # 步骤 2：构建模型要求的三路图像输入
        # LIBERO 只有 2 路相机，第三路（右腕）用全零填充
        inputs = {
            "state": data["observation/state"],       # [8] float（7关节+1夹爪）
            "image": {
                "base_0_rgb": base_image,             # 第三人称相机
                "left_wrist_0_rgb": wrist_image,      # 腕部相机
                "right_wrist_0_rgb": np.zeros_like(base_image),  # ← 填零
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.False_,       # ← 标记为无效，模型会忽略
            },
        }

        # 步骤 3：传递动作（训练时有，推理时无）
        if "actions" in data:
            inputs["actions"] = data["actions"]

        # 步骤 4：传递语言指令
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs
```

输出变换更简单——截取有效维度：

```python
# libero_policy.py L86-100
@dataclasses.dataclass(frozen=True)
class LiberoOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        # 模型输出 [horizon, 32]，但 LIBERO 只需要前 7 维（6关节+1夹爪）
        return {"actions": np.asarray(data["actions"][:, :7])}
```

#### 例 2：DroidInputs（droid_policy.py L30-74）

DROID 和 LIBERO 的区别在于：状态是 joint_position + gripper_position 分开存储的。

```python
# droid_policy.py
@dataclasses.dataclass(frozen=True)
class DroidInputs(transforms.DataTransformFn):
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        # 步骤 1：拼接状态（DROID 的关节和夹爪是分开的）
        gripper_pos = np.asarray(data["observation/gripper_position"])
        if gripper_pos.ndim == 0:
            gripper_pos = gripper_pos[np.newaxis]  # 标量 → [1]
        state = np.concatenate([data["observation/joint_position"], gripper_pos])  # [7]+[1]=[8]

        base_image = _parse_image(data["observation/exterior_image_1_left"])
        wrist_image = _parse_image(data["observation/wrist_image_left"])

        # 步骤 2：根据模型类型选择不同的图像映射
        match self.model_type:
            case _model.ModelType.PI0 | _model.ModelType.PI05:
                # π₀/π₀.₅：标准三路（base, left_wrist, right_wrist）
                names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
                images = (base_image, wrist_image, np.zeros_like(base_image))
                image_masks = (np.True_, np.True_, np.False_)  # 右腕无效
            case _model.ModelType.PI0_FAST:
                # FAST 模型用不同的图像命名，且不 mask 填充图像
                names = ("base_0_rgb", "base_1_rgb", "wrist_0_rgb")
                images = (base_image, np.zeros_like(base_image), wrist_image)
                image_masks = (np.True_, np.True_, np.True_)

        inputs = {
            "state": state,
            "image": dict(zip(names, images, strict=True)),
            "image_mask": dict(zip(names, image_masks, strict=True)),
        }
        ...
        return inputs
```

```python
# droid_policy.py L77-81
@dataclasses.dataclass(frozen=True)
class DroidOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        # DROID 只取前 8 维（7关节+1夹爪），丢弃 pad 到 32 维的多余部分
        return {"actions": np.asarray(data["actions"][:, :8])}
```

#### 例 3：AlohaInputs（aloha_policy.py L24-87）

ALOHA 是双臂机器人，最复杂——需要坐标系翻转和夹爪空间转换：

```python
# aloha_policy.py
@dataclasses.dataclass(frozen=True)
class AlohaInputs(transforms.DataTransformFn):
    adapt_to_pi: bool = True  # 是否转换到 pi 内部坐标系

    def __call__(self, data: dict) -> dict:
        # 步骤 1：坐标系转换（ALOHA 标准坐标 → pi 训练时的坐标）
        data = _decode_aloha(data, adapt_to_pi=self.adapt_to_pi)
        # 内部做了两件事：
        #   - _joint_flip_mask() 翻转某些关节方向（[1,-1,-1,1,1,1,1,1,-1,-1,1,1,1,1]）
        #   - _gripper_to_angular() 将夹爪从线性空间转到角度空间

        # 步骤 2：四路相机 → 三路标准输入
        images = {
            "base_0_rgb": in_images["cam_high"],           # 头顶相机
        }
        # 动态处理：有腕部相机就用，没有就填零
        for dest, source in {"left_wrist_0_rgb": "cam_left_wrist",
                              "right_wrist_0_rgb": "cam_right_wrist"}.items():
            if source in in_images:
                images[dest] = in_images[source]
                image_masks[dest] = np.True_
            else:
                images[dest] = np.zeros_like(base_image)
                image_masks[dest] = np.False_

        ...
        return inputs
```

**三个平台 Inputs 的设计模式完全一致**：

| 步骤 | LiberoInputs | DroidInputs | AlohaInputs |
|------|-------------|-------------|-------------|
| 状态处理 | 直接传递 `[8]` | 拼接 joint+gripper `[7]+[1]` | 坐标翻转+夹爪空间转换 `[14]` |
| 图像映射 | 2路→3路，右腕填零 | 2路→3路，按模型类型分支 | 4路→3路，动态填充 |
| 图像格式 | float32[C,H,W]→uint8[H,W,C] | 同左 | 同左 |
| 输出截取 | `[:, :7]` | `[:, :8]` | `[:, :14]` + 逆坐标转换 |

### 3.5 第三层 model_transforms 代码详解

由 `ModelTransformFactory` 根据 `ModelType` 自动生成，对比 π₀ 和 π₀.₅：

```python
# training/config.py L113-138（ModelTransformFactory.__call__）
def __call__(self, model_config):
    match model_config.model_type:
        case ModelType.PI0:
            return Group(inputs=[
                InjectDefaultPrompt(self.default_prompt),
                ResizeImages(224, 224),
                TokenizePrompt(
                    PaligemmaTokenizer(model_config.max_token_len),  # max_token_len=48
                    # 注意：没有 discrete_state_input 参数
                ),
                PadStatesAndActions(model_config.action_dim),
            ])
        case ModelType.PI05:
            return Group(inputs=[
                InjectDefaultPrompt(self.default_prompt),
                ResizeImages(224, 224),
                TokenizePrompt(
                    PaligemmaTokenizer(model_config.max_token_len),  # max_token_len=200
                    discrete_state_input=model_config.discrete_state_input,  # ← π₀.₅ 独有
                ),
                PadStatesAndActions(model_config.action_dim),
            ])
```

π₀.₅ 多了 `discrete_state_input` 参数，它控制 tokenizer 是否将状态离散化拼入 prompt。

对应 tokenizer 中的实际代码：

```python
# tokenizer.py L22-48（PaligemmaTokenizer.tokenize）
def tokenize(self, prompt, state=None):
    cleaned_text = prompt.strip().replace("_", " ").replace("\n", " ")
    if state is not None:
        # π₀.₅ 路径：状态量化为 256 bin，拼入文本
        discretized_state = np.digitize(state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
        state_str = " ".join(map(str, discretized_state))
        full_prompt = f"Task: {cleaned_text}, State: {state_str};\nAction: "
        # 结果如："Task: pick up the cup, State: 180 92 128 64 200 ...;\nAction: "
        tokens = self._tokenizer.encode(full_prompt, add_bos=True)
    else:
        # π₀ 路径：纯文本 + "\n" 作为分隔符
        tokens = self._tokenizer.encode(cleaned_text, add_bos=True) + self._tokenizer.encode("\n")
    # 填充或截断到 max_token_len
    ...
    return np.asarray(tokens), np.asarray(mask)
```

### 3.6 Policy 推理流程代码详解

`policies/policy.py` 中的 `Policy.infer` 是推理的入口，逐行对应：

```python
# policy.py L67-106
@override
def infer(self, obs: dict, *, noise=None):
    # ① 深拷贝输入（防止变换修改原始数据）
    inputs = jax.tree.map(lambda x: x, obs)

    # ② 执行所有输入变换（data_transforms → Normalize → model_transforms）
    inputs = self._input_transform(inputs)

    # ③ 添加 batch 维度并转为 JAX 数组
    inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)

    # ④ 转为模型输入结构体
    observation = _model.Observation.from_dict(inputs)

    # ⑤ 调用 model.sample_actions（flow matching 10 步去噪）
    outputs = {
        "state": inputs["state"],
        "actions": self._sample_actions(sample_rng, observation, **sample_kwargs),
    }

    # ⑥ 去掉 batch 维度，转回 numpy
    outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)

    # ⑦ 执行所有输出变换（Unnormalize → data_transforms.outputs）
    outputs = self._output_transform(outputs)
    return outputs
```

`create_trained_policy`（`policy_config.py` L16-94）是最常用的入口函数，它按顺序组装完整的变换链：

```python
# policy_config.py L75-89
return Policy(
    model,
    transforms=[                              # 输入变换链（按执行顺序）
        *repack_transforms.inputs,            #  1. 键名映射
        InjectDefaultPrompt(default_prompt),  #  2. 注入默认 prompt
        *data_config.data_transforms.inputs,  #  3. LiberoInputs / DroidInputs
        Normalize(norm_stats, ...),           #  4. 归一化
        *data_config.model_transforms.inputs, #  5. resize + tokenize + pad
    ],
    output_transforms=[                       # 输出变换链（镜像顺序）
        *data_config.model_transforms.outputs,#  5'. （PI0/PI05 无输出变换）
        Unnormalize(norm_stats, ...),         #  4'. 反归一化
        *data_config.data_transforms.outputs, #  3'. LiberoOutputs / DroidOutputs
        *repack_transforms.outputs,           #  1'. 键名反映射
    ],
)
```

### 3.7 完整数据流追踪：π₀.₅ LIBERO 推理

用具体的数据形状追踪整个流程：

```
LIBERO 仿真环境传入：
{
    "observation/image": [224,224,3] uint8,      # 第三人称相机
    "observation/wrist_image": [224,224,3] uint8, # 腕部相机
    "observation/state": [8] float,               # 7关节 + 1夹爪
    "prompt": "open the middle drawer of the cabinet"
}
    │
    ▼  LiberoInputs.__call__()
    │  ├─ _parse_image(): 确保 uint8 [H,W,C] 格式
    │  ├─ 填充 right_wrist_0_rgb = zeros([224,224,3])
    │  └─ 设置 image_mask: right_wrist = False
{
    "state": [8],
    "image": {"base_0_rgb": [224,224,3], "left_wrist_0_rgb": [224,224,3],
              "right_wrist_0_rgb": zeros[224,224,3]},
    "image_mask": {"base_0_rgb": True, "left_wrist_0_rgb": True,
                   "right_wrist_0_rgb": False},
    "prompt": "open the middle drawer of the cabinet"
}
    │
    ▼  Normalize(quantile归一化)
    │  state: [8] → 每个维度映射到 [-1, 1]
    │
    ▼  ResizeImages(224, 224)
    │  （本例已是 224×224，跳过）
    │
    ▼  TokenizePrompt(discrete_state_input=False)
    │  注意：pi05_libero 配置中 discrete_state_input=False
    │  所以走 π₀ 路径："open the middle drawer...\n" → token 序列
    │  tokenized_prompt: [200] int32（π₀.₅ 的 max_token_len=200）
    │
    ▼  PadStatesAndActions(action_dim=32)
    │  state: [8] → [32]（后 24 维填 0）
    │
    ═══ Observation.from_dict() → Pi0.sample_actions() ═══
    │  10 步 flow matching 去噪
    │  输出 actions: [1, 10, 32]（10步 × 32维）
    │
    ▼  Unnormalize(quantile反归一化)
    │  actions 从 [-1,1] 映射回原始尺度
    │
    ▼  LiberoOutputs.__call__()
    │  actions[:, :7] → [10, 7]（截取前 7 维：6关节+1夹爪）
    │
    ▼  返回给 LIBERO 仿真环境执行
```

### 3.8 本步小结

| 概念 | 作用 | 代码位置 |
|------|------|----------|
| `TrainConfig` | 训练/推理全局配置入口 | `training/config.py` L466-557 |
| `LeRobotLiberoDataConfig` | LIBERO 数据集配置+三层变换组装 | `training/config.py` L281-355 |
| `RepackTransform` | 键名映射（数据集字段→统一字段） | `transforms.py` L80-101 |
| `LiberoInputs` | LIBERO 输入变换（2路图像→3路） | `libero_policy.py` L29-83 |
| `DroidInputs` | DROID 输入变换（拼接 state，按模型分支） | `droid_policy.py` L30-74 |
| `AlohaInputs` | ALOHA 输入变换（坐标翻转+夹爪转换） | `aloha_policy.py` L24-87 |
| `*Outputs` | 输出截取（去掉 pad 维度） | 各 policy 文件 |
| `ModelTransformFactory` | 按模型类型生成 tokenize+resize+pad | `training/config.py` L106-163 |
| `PaligemmaTokenizer` | π₀.₅ 状态离散化拼入 prompt | `tokenizer.py` L14-48 |
| `Policy.infer` | 推理入口：变换→模型→逆变换 | `policy.py` L67-106 |
| `create_trained_policy` | 一站式工厂：加载模型+组装变换链 | `policy_config.py` L16-94 |

**下一步预告**：训练脚本与推理服务——`scripts/train.py` 的训练循环、`scripts/serve_policy.py` 的 WebSocket 推理服务、以及如何实际微调和部署 π₀.₅。

---

## 第四步：训练脚本与推理服务

### 4.1 JAX 训练脚本：scripts/train.py

这是主训练入口，280 行代码覆盖了完整的训练循环。

#### 启动方式

```bash
# 用 tyro CLI 选择预定义配置并启动训练
uv run python scripts/train.py pi05_libero --exp_name my_experiment

# 覆盖配置参数
uv run python scripts/train.py pi05_libero --exp_name my_exp --batch_size 64 --num_train_steps 50000
```

`_config.cli()` 使用 tyro 解析命令行，第一个参数是配置名（如 `pi05_libero`），后续参数可覆盖 `TrainConfig` 的任意字段。

#### main 函数流程（train.py L194-280）

```python
def main(config: TrainConfig):
    # ① 初始化分布式
    mesh = sharding.make_mesh(config.fsdp_devices)     # FSDP 分片网格

    # ② 初始化 checkpoint 管理
    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir, overwrite=config.overwrite, resume=config.resume
    )

    # ③ 创建数据加载器
    data_loader = _data_loader.create_data_loader(config, sharding=data_sharding, shuffle=True)
    data_iter = iter(data_loader)
    batch = next(data_iter)  # 预取第一个 batch

    # ④ 初始化训练状态（模型 + 优化器 + EMA）
    train_state, train_state_sharding = init_train_state(config, init_rng, mesh, resume=resuming)

    # ⑤ JIT 编译 train_step
    ptrain_step = jax.jit(
        functools.partial(train_step, config),
        in_shardings=(replicated, train_state_sharding, data_sharding),
        out_shardings=(train_state_sharding, replicated),
        donate_argnums=(1,),  # 复用 train_state 的内存
    )

    # ⑥ 训练循环
    for step in range(start_step, config.num_train_steps):
        train_state, info = ptrain_step(train_rng, train_state, batch)  # 一步训练
        batch = next(data_iter)                                          # 预取下一个 batch
        if step % config.save_interval == 0:
            _checkpoints.save_state(checkpoint_manager, train_state, data_loader, step)
```

#### train_step 函数（train.py L136-191）

单步训练的核心逻辑：

```python
def train_step(config, rng, state, batch):
    # ① 从状态中恢复模型
    model = nnx.merge(state.model_def, state.params)
    model.train()

    # ② 定义损失函数
    def loss_fn(model, rng, observation, actions):
        chunked_loss = model.compute_loss(rng, observation, actions, train=True)
        return jnp.mean(chunked_loss)

    # ③ 前向传播 + 反向传播（只对可训练参数求梯度）
    diff_state = nnx.DiffState(0, config.trainable_filter)  # 冻结 filter
    loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(model, rng, obs, actions)

    # ④ 优化器更新
    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)

    # ⑤ EMA 更新
    if state.ema_decay is not None:
        ema_params = ema_decay * old_ema + (1 - ema_decay) * new_params

    return new_state, {"loss": loss, "grad_norm": ..., "param_norm": ...}
```

关键设计点：
- `nnx.DiffState` + `config.trainable_filter`：实现 LoRA 微调时只对 LoRA 参数求梯度，冻结参数不参与反向传播
- `donate_argnums=(1,)`：告诉 JAX 可以复用 `train_state` 的内存，避免峰值内存翻倍
- EMA（指数移动平均）：推理时用 EMA 参数代替原始参数，输出更稳定

#### init_train_state（train.py L84-133）

初始化流程：

```python
def init_train_state(config, init_rng, mesh, *, resume):
    def init(rng, partial_params=None):
        # ① 创建模型（随机初始化所有参数）
        model = config.model.create(model_rng)

        # ② 合并预训练权重（覆盖随机初始化的参数）
        if partial_params is not None:
            state.replace_by_pure_dict(partial_params)

        # ③ 冻结参数转 bfloat16（节省内存）
        params = nnx_utils.state_map(params, config.freeze_filter,
                                      lambda p: p.replace(p.value.astype(jnp.bfloat16)))

        return TrainState(params=params, opt_state=tx.init(...), ema_params=params)

    # 加载预训练权重
    partial_params = _load_weights_and_validate(config.weight_loader, params_shape)

    # FSDP 分片后初始化
    train_state = jax.jit(init, out_shardings=state_sharding)(init_rng, partial_params)
```

### 4.2 PyTorch 训练脚本：scripts/train_pytorch.py

提供与 JAX 版等价的 PyTorch 训练，支持多 GPU（DDP）和多节点训练。

#### 启动方式

```bash
# 单 GPU
python scripts/train_pytorch.py pi05_libero --exp_name my_exp

# 多 GPU（单节点，4卡）
torchrun --standalone --nnodes=1 --nproc_per_node=4 \
    scripts/train_pytorch.py pi05_libero --exp_name my_exp

# 多节点
torchrun --nnodes=2 --nproc_per_node=8 --node_rank=0 \
    --master_addr=<ip> --master_port=<port> \
    scripts/train_pytorch.py pi05_libero --exp_name my_exp
```

#### 与 JAX 版的关键差异

```python
# train_pytorch.py L309-622（train_loop 核心）

# ① DDP 初始化（JAX 版用 FSDP mesh）
use_ddp, local_rank, device = setup_ddp()
model = PI0Pytorch(model_cfg).to(device)
if use_ddp:
    model = torch.nn.parallel.DistributedDataParallel(model, ...)

# ② 加载预训练权重（JAX 版用 weight_loader）
if config.pytorch_weight_path is not None:
    safetensors.torch.load_model(model, model_path)

# ③ 训练循环
for observation, actions in loader:
    observation = jax.tree.map(lambda x: x.to(device), observation)
    actions = actions.to(device)

    losses = model(observation, actions)     # 前向传播
    loss = losses.mean()
    loss.backward()                          # 反向传播

    grad_norm = clip_grad_norm_(model.parameters(), max_norm=config.optimizer.clip_gradient_norm)
    optim.step()
    optim.zero_grad(set_to_none=True)

# ④ checkpoint 保存（用 safetensors 格式）
safetensors.torch.save_model(model, "model.safetensors")
```

| | JAX 版 (train.py) | PyTorch 版 (train_pytorch.py) |
|---|---|---|
| 并行策略 | FSDP（模型分片） | DDP（数据并行） |
| JIT | `jax.jit` 编译 train_step | 无 JIT |
| 权重格式 | Orbax checkpoint | safetensors |
| EMA | 支持 | 不支持 |
| LoRA 冻结 | `trainable_filter` | 未实现 |
| 多节点 | 不支持 | `torchrun` 原生支持 |

### 4.3 推理服务：scripts/serve_policy.py

推理部署分两部分：**服务端**（加载模型，暴露 WebSocket 接口）和**客户端**（发送观测，接收动作）。

#### 服务端启动

```bash
# 使用默认 checkpoint（推荐）
uv run python scripts/serve_policy.py --env DROID
uv run python scripts/serve_policy.py --env LIBERO

# 使用自定义 checkpoint
uv run python scripts/serve_policy.py \
    --policy.config pi05_droid \
    --policy.dir gs://openpi-assets/checkpoints/pi05_droid \
    --default_prompt "pick up the red cup"
```

#### 服务端代码流程（serve_policy.py）

```python
# serve_policy.py L58-76 — 默认 checkpoint 映射
DEFAULT_CHECKPOINT = {
    EnvMode.ALOHA:  Checkpoint(config="pi05_aloha",  dir="gs://...checkpoints/pi05_base"),
    EnvMode.DROID:  Checkpoint(config="pi05_droid",  dir="gs://...checkpoints/pi05_droid"),
    EnvMode.LIBERO: Checkpoint(config="pi05_libero", dir="gs://...checkpoints/pi05_libero"),
}

# serve_policy.py L99-117 — main 流程
def main(args):
    # ① 创建 Policy（加载模型 + 组装变换链）
    policy = create_policy(args)
    # 内部调用 create_trained_policy(config, checkpoint_dir)

    # ② 启动 WebSocket 服务
    server = WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,  # 默认 8000
    )
    server.serve_forever()
```

#### WebSocket 服务实现（websocket_policy_server.py）

```python
# websocket_policy_server.py L48-83
async def _handler(self, websocket):
    packer = msgpack_numpy.Packer()
    await websocket.send(packer.pack(self._metadata))  # 发送元数据

    while True:
        # ① 接收客户端观测（msgpack 编码的 numpy 数组）
        obs = msgpack_numpy.unpackb(await websocket.recv())

        # ② 调用 policy.infer(obs)（完整的 变换→模型→逆变换 流程）
        action = self._policy.infer(obs)

        # ③ 返回动作 + 耗时统计
        action["server_timing"] = {"infer_ms": infer_time * 1000}
        await websocket.send(packer.pack(action))
```

协议：基于 WebSocket，数据用 msgpack + numpy 序列化，单连接长链接，延迟低。

#### 客户端使用（examples/simple_client/main.py）

```python
# 客户端只需几行代码
from openpi_client import websocket_client_policy

# ① 连接服务端
policy = websocket_client_policy.WebsocketClientPolicy(host="0.0.0.0", port=8000)

# ② 发送观测，获取动作
obs = {
    "observation/state": np.random.rand(8),
    "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
    "observation/wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
    "prompt": "pick up the cup",
}
action = policy.infer(obs)
# action["actions"] 就是模型输出的动作序列
```

### 4.4 完整的微调 → 部署流程

以 π₀.₅ 微调 LIBERO 为例：

```bash
# 第一步：计算归一化统计量（首次使用新数据集时需要）
uv run python scripts/compute_norm_stats.py --config-name pi05_libero

# 第二步：训练（JAX 版）
uv run python scripts/train.py pi05_libero \
    --exp_name libero_finetune \
    --num_train_steps 30000 \
    --batch_size 256

# 或训练（PyTorch 版，多 GPU）
torchrun --nproc_per_node=4 scripts/train_pytorch.py pi05_libero \
    --exp_name libero_finetune_pt

# 第三步：启动推理服务
uv run python scripts/serve_policy.py \
    --policy.config pi05_libero \
    --policy.dir checkpoints/pi05_libero/libero_finetune/29999

# 第四步：客户端连接并执行
python examples/simple_client/main.py --env LIBERO --host <server_ip> --port 8000
```

### 4.5 部署架构图

```
┌────────────────────┐       WebSocket        ┌─────────────────────────────┐
│   机器人控制端     │◄═══════════════════════►│     推理服务器              │
│  (openpi-client)   │   msgpack + numpy       │   (serve_policy.py)         │
│                    │                          │                             │
│  obs = get_obs()   │  ──── 发送观测 ────►    │  Policy.infer(obs)          │
│  act = infer(obs)  │  ◄─── 返回动作 ────     │   ├─ input_transforms       │
│  execute(act)      │                          │   ├─ model.sample_actions   │
│                    │                          │   └─ output_transforms      │
└────────────────────┘                          └─────────────────────────────┘
     可以在任意设备                                  需要 GPU（≥8GB）
     只需 openpi-client 包                           需要完整 openpi 环境
```

**Client-Server 分离**的设计意图：
- 机器人端只需安装轻量的 `openpi-client` 包（纯 Python，无 JAX/PyTorch 依赖）
- GPU 推理集中在服务器端，可以远程部署
- WebSocket 协议延迟低（通常 <100ms），适合实时控制

### 4.6 本步小结

| 概念 | 作用 | 文件位置 |
|------|------|----------|
| `scripts/train.py` | JAX 训练入口，FSDP + JIT + EMA | `scripts/train.py` |
| `train_step` | 单步训练：loss → grad → update → EMA | `scripts/train.py` L136-191 |
| `init_train_state` | 创建模型 + 加载预训练权重 + 冻结参数 | `scripts/train.py` L84-133 |
| `scripts/train_pytorch.py` | PyTorch 训练入口，DDP 多 GPU 支持 | `scripts/train_pytorch.py` |
| `scripts/serve_policy.py` | 推理服务入口，创建 Policy + 启动 WebSocket | `scripts/serve_policy.py` |
| `WebsocketPolicyServer` | WebSocket 服务实现，接收观测返回动作 | `serving/websocket_policy_server.py` |
| `openpi-client` | 轻量客户端 SDK，无 GPU 依赖 | `packages/openpi-client/` |
| `DEFAULT_CHECKPOINT` | 各平台的默认 π₀.₅ checkpoint | `scripts/serve_policy.py` L59-76 |

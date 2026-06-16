# OpenPI-0.5 源码精读（egocentric 版）

> 适用代码库：`/cpfs_infra/shared/liyiduo/ws/egocentric_openpi05_model`
>
> 阅读路径：`model.py` → `transforms.py` → `policies/*.py` → `pi0_config.py` → `pi0.py` → `gemma.py` / `tokenizer.py` → `training/config.py`
>
> 目标：把"数据怎么进来 → 模型怎么算 → 训练怎么串"一次讲清。

---

## 0. 总体架构鸟瞰

OpenPI-0.5（π0.5）= **PaliGemma 视觉语言主干 + Gemma-300M Action Expert + Flow Matching 动作头**。

```
┌────────── 数据集 (LeRobot / RLDS) ──────────┐
│  observation.images.* / observation.state    │
│  action / task                               │
└──────────────────┬───────────────────────────┘
                   │ ① RepackTransform           (键名重映射)
                   ▼
┌────────── 机器人专属变换 ──────────┐
│  *Inputs / *Outputs                 │
│  (DroidInputs, LiberoInputs,        │
│   AlohaInputs, H1Inputs ...)        │
└──────────────────┬──────────────────┘
                   │ ② Normalize / DeltaActions
                   ▼
┌────────── 模型通用变换 ──────────┐
│  ResizeImages / TokenizePrompt   │
│  PadStatesAndActions             │
└──────────────────┬───────────────┘
                   │ ③ 形成 model.Observation 的字典形式
                   ▼
┌────────────────── Pi0 / Pi0.5 模型 ──────────────────┐
│  SigLIP (224×224) ─┐                                 │
│  Prompt Tokens ────┼──► PaliGemma (Gemma-2B) ─┐      │
│  State (pi0 only) ─┘                          │      │
│                                               ▼      │
│  Noisy Action  ─► Action Expert (Gemma-300M)         │
│              ◄──  Flow-Matching v_t                  │
└──────────────────┬───────────────────────────────────┘
                   │ ④ Unnormalize / AbsoluteActions
                   ▼
            最终动作序列 (action_horizon × action_dim)
```

`Pi0` 与 `Pi0.5` 共用同一份代码，靠 `Pi0Config.pi05` 这一个开关切换：

| 维度 | π0 | π0.5 |
|------|----|------|
| state 输入 | 连续向量 → action expert 第一个 token | 离散化 256 bins → 拼到 prompt 文本里 |
| 时间步 t 注入 | 与 action token 拼接后过 MLP | adaRMSNorm 调制每层 |
| `max_token_len` 默认 | 48 | 200 |
| `discrete_state_input` | False | True |

---

## 1. 数据流主线：`model.py` 定义"模型唯一认识的输入格式"

文件：`src/openpi/models/model.py`

### 1.1 模型类型与硬编码常量

```36:50:src/openpi/models/model.py
class ModelType(enum.Enum):
    """Supported model types."""

    PI0 = "pi0"
    PI0_FAST = "pi0_fast"
    PI05 = "pi05"


# The model always expects these images
IMAGE_KEYS = (
    "base_0_rgb",
    "left_wrist_0_rgb",
    "right_wrist_0_rgb",
)


# This may need change if we release a small model.
IMAGE_RESOLUTION = (224, 224)
```

无论上游机器人有几个相机，**模型一律按 `base_0_rgb / left_wrist_0_rgb / right_wrist_0_rgb` 三路 224×224 输入设计**。缺的相机用零图 + `image_mask=False` 占位。

### 1.2 `Observation` —— 模型输入的标准结构

`from_dict` 是数据流接入模型前的最后一步，把 transform 输出的嵌套 dict 校验并转成 `Observation`：

```112:132:src/openpi/models/model.py
    @classmethod
    def from_dict(cls, data: at.PyTree[ArrayT]) -> "Observation[ArrayT]":
        """This method defines the mapping between unstructured data (i.e., nested dict) to the structured Observation format."""
        # Ensure that tokenized_prompt and tokenized_prompt_mask are provided together.
        if ("tokenized_prompt" in data) != ("tokenized_prompt_mask" in data):
            raise ValueError("tokenized_prompt and tokenized_prompt_mask must be provided together.")
        # If images are uint8, convert them to [-1, 1] float32.
        for key in data["image"]:
            if data["image"][key].dtype == np.uint8:
                data["image"][key] = data["image"][key].astype(np.float32) / 255.0 * 2.0 - 1.0
            ...
```

字段一览：

| 字段 | 形状 | 说明 |
|------|------|------|
| `images` | `dict[str, [b,h,w,c]]` | 取值范围 `[-1, 1]` |
| `image_masks` | `dict[str, [b]]` | True 表示该相机本步有效 |
| `state` | `[b, s]` | 已归一化、padding 到 `action_dim` 的本体感知 |
| `tokenized_prompt` | `[b, l]` | SentencePiece 后的 token id |
| `tokenized_prompt_mask` | `[b, l]` | 区分实际 token 与 padding |
| `token_ar_mask` / `token_loss_mask` | 仅 PI0_FAST 用 | 自回归 / loss 区分 prefix vs postfix |

### 1.3 `preprocess_observation` —— 进入模型前的统一预处理

该函数是 `compute_loss` / `sample_actions` 的第一行。做三件事：

1. 缺图警告 + `resize_with_pad` 统一到 224×224；
2. **训练时**对每路相机做数据增强（默认 augmax 链；启用 `augmentation_config` 时走自定义透视/平移）；
3. 补齐缺失的 `image_mask`（默认全 True）。

增强链来自 `augmentation_config.py`，可按相机分别配置：见 `get_h1_strong_augmentation()`，对头部相机启用透视+10% 平移，对腕部相机只做颜色抖动。

### 1.4 `BaseModel` / `BaseModelConfig`

抽象基类约定两个必须实现的方法：

```550:561:src/openpi/models/model.py
    @abc.abstractmethod
    def compute_loss(
        self,
        rng: at.KeyArrayLike,
        observation: Observation,
        actions: Actions,
        *,
        train: bool = False,
    ) -> at.Float[at.Array, "*b ah"]: ...

    @abc.abstractmethod
    def sample_actions(self, rng: at.KeyArrayLike, observation: Observation, **kwargs) -> Actions: ...
```

`Pi0` / `Pi0FAST` 都实现这两个接口。`BaseModelConfig.create()` 用 `nnx.Rngs` 实例化模型，`load()` 把 orbax 还原出的 PyTree 灌进去（这是热加载预训练 checkpoint 的入口）。

---

## 2. `transforms.py` —— 数据流水线的"积木"

文件：`src/openpi/transforms.py`

### 2.1 设计模式

- 所有变换都实现 `DataTransformFn` 协议：`__call__(data: dict) -> dict`，输入输出都是嵌套 numpy 字典。
- `Group(inputs=..., outputs=...)`：训练只用 `inputs`；推理同时用 `inputs`（喂模型）和 `outputs`（把模型输出转换回机器人空间，例如反归一化、把 delta 转回绝对动作）。
- `compose()` 把多个 transform 串成 `CompositeTransform`，按顺序应用。

### 2.2 通用变换

| 类 | 作用 |
|----|------|
| `RepackTransform(structure)` | 按目标 dict 模板把扁平化后的 `'observation/state'` 之类的键拍成 `{"state": ..., "image": {...}}` |
| `InjectDefaultPrompt(prompt)` | 数据没带 prompt 时填默认 |
| `Normalize / Unnormalize` | 标准化或分位数归一化（π0.5 默认用 q01/q99 → `[-1,1]`） |
| `ResizeImages(h,w)` | 统一图像分辨率（一般 224） |
| `DeltaActions(mask) / AbsoluteActions(mask)` | 在选定维度上把绝对动作 ↔ 相对动作（夹爪通常保持绝对） |
| `PadStatesAndActions(model_action_dim)` | 把机器人真实 `state/action` 维度（7、14、16、23）右补零到模型固定的 `action_dim`（默认 32） |
| `TokenizePrompt` | π0/π0.5 用，调用 `PaligemmaTokenizer` |
| `TokenizeFASTInputs / ExtractFASTActions` | π0-FAST 专用，把动作离散成 token 再嵌进 prompt |
| `PromptFromLeRobotTask` | 从 LeRobot dataset 的 `task_index` 拿任务字符串 |

### 2.3 `_normalize_quantile` —— π0.5 的标准做法

```141:145:src/openpi/transforms.py
    def _normalize_quantile(self, x, stats: NormStats):
        assert stats.q01 is not None
        assert stats.q99 is not None
        q01, q99 = stats.q01[..., : x.shape[-1]], stats.q99[..., : x.shape[-1]]
        return (x - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0
```

把动作 / state 截断到分位数区间再线性映射到 `[-1, 1]`，比 z-score 更鲁棒。**这一点很关键：tokenizer 把 state 离散到 256 bins 时，假定输入已落在 `[-1, 1]`。**

### 2.4 `pad_to_dim` & `make_bool_mask`

二者是把"真实机器人 → 模型 32 维空间"的两件法宝：
- `pad_to_dim` 在维度不够时右侧补零；
- `make_bool_mask(6, -1, 6, -1)` 生成 `(T,T,T,T,T,T,F,T,T,T,T,T,T,F)`，告诉 `DeltaActions` 哪几维要做 delta（关节）哪几维保持绝对（夹爪）。

---

## 3. policies/*.py —— "我家机器人怎么映射到 base_0_rgb"

每种机器人一个 `*Inputs` + `*Outputs` 子类。它们做两件事：

1. 把数据集里的相机/关节键名重排成 `model.Observation` 期望的标准键名；
2. 把缺失的相机用零图占位、缺失的 mask 用 `False`。

### 3.1 `LiberoInputs` —— 教科书式范例

```42:84:src/openpi/policies/libero_policy.py
    def __call__(self, data: dict) -> dict:
        ...
        base_image = _parse_image(data["observation/image"])
        wrist_image = _parse_image(data["observation/wrist_image"])

        inputs = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
            },
        }

        if "actions" in data:
            inputs["actions"] = data["actions"]

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs
```

要点：
- `right_wrist_0_rgb` 用零图占位，`image_mask` 设为 `False`，模型在注意力里会屏蔽这部分 token；
- π0-FAST 因为没有 image mask 机制，所以让占位图也参与计算（设为 `True`）。

`LiberoOutputs` 只取前 7 维，把 model 的 32 维输出还原成 Libero 真实动作维度：

```95:100:src/openpi/policies/libero_policy.py
    def __call__(self, data: dict) -> dict:
        # Only return the first N actions ...
        # For Libero, we only return the first 7 actions (since the rest is padding).
        return {"actions": np.asarray(data["actions"][:, :7])}
```

### 3.2 `DroidInputs` —— 处理 PI0_FAST 的差异

```47:58:src/openpi/policies/droid_policy.py
        match self.model_type:
            case _model.ModelType.PI0 | _model.ModelType.PI05:
                names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
                images = (base_image, wrist_image, np.zeros_like(base_image))
                image_masks = (np.True_, np.True_, np.False_)
            case _model.ModelType.PI0_FAST:
                names = ("base_0_rgb", "base_1_rgb", "wrist_0_rgb")
                images = (base_image, np.zeros_like(base_image), wrist_image)
                image_masks = (np.True_, np.True_, np.True_)
```

DROID 只有外视 + 单腕，π0/π0.5 把它放到 `base_0_rgb + left_wrist_0_rgb`，π0-FAST 走另一套命名约定。`state = concat(joint_position[7], gripper_position[1])` 共 8 维。

### 3.3 `Policy.infer()` —— 推理串联

`policies/policy.py` 把整个推理过程串起来，重要片段：

```132:181:src/openpi/policies/policy.py
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:
        try:
            inputs = jax.tree.map(lambda x: x, obs)
            inputs = self._input_transform(inputs)

            inputs = self._ensure_gpu_buffers(inputs)
            ...
            observation = _model.Observation.from_dict(inputs)
            ...
            outputs = {
                "state": inputs["state"],
                "actions": self._sample_actions(
                    sample_rng_or_pytorch_device, observation, **sample_kwargs
                ),
            }
            ...
            outputs = self._output_transform(outputs)
```

注意这里有一个 ego 版本独有的优化：`_ensure_gpu_buffers()` 第一次推理时分配固定形状的 GPU 缓冲区，之后所有步都原地写入，避免每步重新分配显存。

---

## 4. `pi0_config.py` —— π0 vs π0.5 的所有开关都在这里

```19:46:src/openpi/models/pi0_config.py
@dataclasses.dataclass(frozen=True)
class Pi0Config(_model.BaseModelConfig):
    dtype: str = "bfloat16"
    paligemma_variant: _gemma.Variant = "gemma_2b"
    action_expert_variant: _gemma.Variant = "gemma_300m"

    action_dim: int = 32
    action_horizon: int = 50
    max_token_len: int = None  # type: ignore
    # Pi05 has two differences from Pi0:
    # - the state input is part of the discrete language tokens rather than a continuous input that is part of the suffix
    # - the action expert uses adaRMSNorm to inject the flow matching timestep
    pi05: bool = False
    discrete_state_input: bool = None  # type: ignore

    augmentation: aug_config.AugmentationConfig = dataclasses.field(
        default_factory=aug_config.get_default_augmentation
    )

    def __post_init__(self):
        if self.max_token_len is None:
            object.__setattr__(self, "max_token_len", 200 if self.pi05 else 48)
        if self.discrete_state_input is None:
            object.__setattr__(self, "discrete_state_input", self.pi05)
```

关键开关：

- `pi05`：一处开关同时影响"state 进文本"和"adaRMSNorm 注入时间步"两件事；
- `paligemma_variant ∈ {gemma_2b, gemma_2b_lora}`：是否对 LLM 主干启用 LoRA；
- `action_expert_variant ∈ {gemma_300m, gemma_300m_lora}`：动作专家是否 LoRA；
- `action_dim=32`：模型固定空间，所有真实机器人都 pad 到 32；
- `action_horizon=50`：默认一次预测 50 步动作（egocentric 配置里通常被 `ACTION_HORIZON` env var 覆盖）。

`get_freeze_filter()` 用 NNX 的 `PathRegex` 把"匹配 `.*lora.*` 的参数 + 匹配 `.*llm.*_1.*`（动作专家）的参数"组合冻结/可训练 —— 这就是 LoRA 微调的实现机制。

`inputs_spec()` 用 `jax.ShapeDtypeStruct` 描述模型期望的输入形状，给 `nnx.eval_shape` 做参数初始化用。

---

## 5. `pi0.py` —— 模型主体（训练 + 推理 = Flow Matching）

文件：`src/openpi/models/pi0.py`

### 5.1 网络组件

```67:104:src/openpi/models/pi0.py
class Pi0(_model.BaseModel):
    def __init__(self, config: pi0_config.Pi0Config, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.pi05 = config.pi05
        self.augmentation = config.augmentation
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        # TODO: rewrite gemma in NNX. For now, use bridge.
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
                adarms=config.pi05,
            )
        )
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, True] if config.pi05 else [False, False])
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        ...
        self.PaliGemma = nnx.Dict(llm=llm, img=img)
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        if config.pi05:
            self.time_mlp_in = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        else:
            self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)
```

要点：
- **一个 Gemma `Module`，里面装两个 expert**（PaliGemma-2B 和 Action Expert-300M），共享注意力的 head_dim、num_heads、num_kv_heads，但 width / mlp_dim 各异；
- SigLIP `So400m/14` 把 224×224 图片切 patch、过 ViT，输出宽度对齐 PaliGemma 的 `width=2048`；
- π0.5 没有 `state_proj`（state 走文本），多了一对 `time_mlp_*` 用来生成 adaRMS 的条件向量；
- π0 有 `state_proj`（把连续 state 投影成一个 token），并且把 time embedding 与 action token 拼起来过 MLP。

### 5.2 prefix / suffix 双段拼接

模型把整个 token 序列分成两段：

- **prefix**：图像 patch tokens + 文本 tokens（双向注意力，可作 KV cache）
- **suffix**：状态 token（仅 π0）+ 动作 token（自回归依赖 prefix）

`embed_prefix` 见 `pi0.py:106-138`：

```106:138:src/openpi/models/pi0.py
    def embed_prefix(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        input_mask = []
        ar_mask = []
        tokens = []
        # embed images
        for name in obs.images:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)

            tokens.append(image_tokens)
            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=image_tokens.shape[1],
                )
            )
            # image tokens attend to each other
            ar_mask += [False] * image_tokens.shape[1]

        # add language (aka tokenized inputs)
        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            # full attention between image and language inputs
            ar_mask += [False] * tokenized_inputs.shape[1]
```

`ar_mask`：False 表示与上一 token "同一 attention block"（双向），True 表示开启新块（只能看到之前的）。这种"块状自回归"思路来自 PaliGemma：图像和语言在 prefix 里互相完全可见，但动作 token 不能反向影响图像/语言。

`embed_suffix` 见 `pi0.py:140-187`：

- **π0**：先 push 一个由 `state_proj(state)` 得到的 state token；再把 noisy action 经 `action_in_proj` 投影、与 sin/cos 时间嵌入 concat，过 `action_time_mlp_*`。
- **π0.5**：跳过 state token；时间嵌入单独走 `time_mlp_*` → `adarms_cond` 注入到 `Gemma.Block` 的 `RMSNorm` 中；动作 token 直接是 `action_in_proj(noisy_actions)`。

> 关键技巧：suffix 里"第一个动作 token" `ar_mask=True`、之后全 False。也就是动作内部互相可见（双向），但与 prefix 之间是单向的。

### 5.3 `compute_loss` —— Flow Matching 训练目标

```190:221:src/openpi/models/pi0.py
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(
            preprocess_rng, observation, train=train, augmentation_config=self.augmentation
        )

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

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

        return jnp.mean(jnp.square(v_t - u_t), axis=-1)
```

逐行解释：
1. 时间 `t ~ Beta(1.5, 1)`：偏向"中间到接近真值"的样本，这是 π0 论文的设定；
2. 线性插值 `x_t = t·noise + (1-t)·actions`，目标速度 `u_t = noise - actions`；注意"t=1=噪声、t=0=数据"的方向，与 diffusion 的常用约定相反；
3. 同一次 forward 把 prefix 和 suffix 一起跑（高效），其中 `prefix → expert 0`、`suffix → expert 1`；
4. 取 suffix 输出的最后 `action_horizon` 个位置过 `action_out_proj`，得到预测速度 v_t；
5. Loss = MSE(v_t, u_t)，按维度平均后保留 `[*b, ah]` 形状（外层会再做 mean）。

### 5.4 `make_attn_mask` —— 块状 attention 的核心实现

```19:44:src/openpi/models/pi0.py
def make_attn_mask(input_mask, mask_ar):
    """
    [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
        themselves and the last 3 tokens have a causal attention.
    [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks.
    """
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)
```

通过 cumsum 巧妙地实现"同一块互相可见、新块只能看历史"。结合 input_mask 屏蔽 padding，就得到最终注意力 mask。

### 5.5 `sample_actions` —— Flow Matching 推理（欧拉积分）

```223:286:src/openpi/models/pi0.py
    def sample_actions(
        self, rng, observation, *, num_steps=10, noise=None,
    ) -> _model.Actions:
        observation = _model.preprocess_observation(None, observation, train=False)
        dt = -1.0 / num_steps
        ...
        # first fill KV cache with a forward pass of the prefix
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        def step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(...)
            ...
            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens], mask=full_attn_mask, positions=positions,
                kv_cache=kv_cache, adarms_cond=[None, adarms_cond],
            )
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])
            return x_t + dt * v_t, time + dt

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0
```

要点：
1. **先单独跑 prefix 一次**，把 KV cache 缓存起来 —— 推理 N 步时图像/文本只算一次；
2. while_loop 内部从 t=1（纯噪声）走到 t≈0（动作），步长 `dt = -1/N`；
3. 每步只把 suffix 喂进去（`[None, suffix_tokens]`），expert 0 不再计算，节省一半算力；
4. 默认 `num_steps=10` 就足够，时间复杂度 ≈ 一次 prefix + 10 次 suffix。

---

## 6. `gemma.py` —— 双 expert Transformer 主干

文件：`src/openpi/models/gemma.py`

### 6.1 双 expert 的实现思想

`Module.configs: Sequence[Config]` 同时持有多个 expert 的 config。每一层 `Block` 内：

- 每个 expert 有自己的 RMSNorm、QKV 矩阵、FFN、输出投影；
- **但 Q/K/V 在 sequence 维度上拼接，做一次共享的 self-attention** —— 这就是 expert 间共享注意力的实现。

```201:202:src/openpi/models/gemma.py
        q, k, v = (jnp.concatenate(y, axis=1) for y in zip(*qkvs, strict=True))
```

这一行就是 π0 的核心 trick：图像/语言 token 走 PaliGemma 的 QKV，动作 token 走 Action Expert 的 QKV，但拼接在序列维上做同一次 attention，让动作 token 能 cross-attend 到图像/语言。

### 6.2 RMSNorm 与 adaRMSNorm

```112:131:src/openpi/models/gemma.py
class RMSNorm(nn.Module):
    @nn.compact
    def __call__(self, x, cond):
        ...
        if cond is None:
            scale = self.param("scale", nn.initializers.zeros_init(), (x.shape[-1]))
            normed_inputs = normed_inputs * (1 + scale)
            return normed_inputs.astype(dtype), None  # return in original dtype

        # adaptive RMSNorm
        modulation = nn.Dense(x.shape[-1] * 3, kernel_init=nn.initializers.zeros, dtype=dtype)(cond)
        scale, shift, gate = jnp.split(modulation[:, None, :], 3, axis=-1)
        normed_inputs = normed_inputs * (1 + scale) + shift
        return normed_inputs.astype(dtype), gate
```

- 普通 RMSNorm：学一组 scale；
- adaRMSNorm：给一个 `cond` 向量（π0.5 里就是 time embedding），生成 `(scale, shift, gate)` —— **scale/shift 改归一化输出，gate 给残差**。`_gated_residual` 用 `x + y * gate` 替代标准残差 `x + y`。

这是 π0.5 把"时间步 t"注入网络的方式，类似 DiT 的 adaLN-Zero。

### 6.3 RoPE + GQA 注意力

`_apply_rope` 在 Q、K 上独立加 RoPE 位置编码（KV cache 里的位置已 baked in）。`num_heads=8, num_kv_heads=1` 是 Gemma 的极端 GQA（multi-query attention），8 个 query head 共享 1 个 KV head，KV cache 显存大幅减小。

### 6.4 `nn.scan` 实现层堆叠

```365:381:src/openpi/models/gemma.py
        self.layers = nn.scan(
            block_cls,
            variable_axes={"params": 0},
            split_rngs={"params": True, "dropout": True},
            in_axes=(0, nn.broadcast, nn.broadcast, nn.broadcast, nn.broadcast),
            length=self.configs[0].depth,
        )(...)
```

**所有 18 层共享一份编译过的 Block** —— 用 `nn.scan` 在参数 axis=0 堆叠权重，配合 `nn.remat` 做 checkpoint。这让模型代码"像一层"那么紧凑，编译速度也快。

### 6.5 命名约定 —— LoRA / 预训练加载的关键

```443:450:src/openpi/models/gemma.py
def _name(name, i):
    # the first expert's weights to have no suffix (e.g., "attn"), so that they
    # can be loaded seamlessly from the existing PaliGemma checkpoint. subsequent
    # experts will have a suffix (e.g., "attn_1") ...
    if i == 0:
        return name
    return f"{name}_{i}"
```

PaliGemma 公开 checkpoint 里只有一组权重（无后缀），加载时正好命中 expert 0；动作专家（expert 1）的权重 key 形如 `attn_1`、`mlp_1`，是新初始化的。这就是 `_build_freeze_filter` 用 `.*_1.*` 匹配动作专家、用 `.*lora.*` 匹配 LoRA 适配层的依据。

---

## 7. `tokenizer.py` —— prompt 与 state 的"文本化"

文件：`src/openpi/models/tokenizer.py`

### 7.1 `PaligemmaTokenizer.tokenize`

```29:55:src/openpi/models/tokenizer.py
    def tokenize(self, prompt: str, state: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        cleaned_text = prompt.strip().replace("_", " ").replace("\n", " ")
        if state is not None:
            # This is the Pi05 format, where the state is part of the discrete language input.
            discretized_state = np.digitize(state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
            state_str = " ".join(map(str, discretized_state))
            full_prompt = f"Task: {cleaned_text}, State: {state_str};\nAction: "
            tokens = self._tokenizer.encode(full_prompt, add_bos=True)
        else:
            # This is the Pi0 format, where the state is part of the continuous action expert input.
            # tokenize "\n" separately as the "start of answer" token
            tokens = self._tokenizer.encode(cleaned_text, add_bos=True) + self._tokenizer.encode("\n")
```

- **π0**：prompt → "do something\n"，state 走另一条连续输入；
- **π0.5**：state 离散化到 0~255，用空格分隔写进文本，最终 prompt 形如：
  ```
  Task: pick up the red cup, State: 128 130 ... 64;
  Action: 
  ```
  全部走 SentencePiece，喂到 PaliGemma 词表。

`max_token_len` 在 π0.5 里默认 200，因为 state 嵌入文本会显著拉长 prompt。

### 7.2 `FASTTokenizer`

π0-FAST 把动作也离散成 token（用 HuggingFace 上的 `physical-intelligence/fast` 模型），拼到 `Action: ...|` 中，做 next-token prediction。这套流程的 mask 见 `TokenizeFASTInputs`：`ar_mask` 在 prefix 全 0（双向）、postfix 全 1（自回归），`loss_mask` 只在 postfix 上算。

---

## 8. `training/config.py` —— 把所有部件粘起来

### 8.1 三个核心 dataclass

| 类 | 作用 |
|----|------|
| `DataConfig` | 跑通数据流要的一切：`norm_stats`、三组 `Group`（repack / data / model transforms）、`use_quantile_norm` 等 |
| `DataConfigFactory` | 工厂模式 + tyro CLI；`SimpleDataConfig` / `LeRobotLiberoDataConfig` / `LerobotH1DataConfig` 是具体子类 |
| `TrainConfig` | 顶层训练配置：模型、weight loader、优化器、学习率调度、batch size、freeze filter、是否 EMA、wandb 等 |

### 8.2 `ModelTransformFactory` —— 决定 model_transforms

```222:247:src/openpi/training/config.py
    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        match model_config.model_type:
            case _model.ModelType.PI0:
                return _transforms.Group(inputs=[
                    _transforms.InjectDefaultPrompt(self.default_prompt),
                    _transforms.ResizeImages(224, 224),
                    _transforms.TokenizePrompt(_tokenizer.PaligemmaTokenizer(model_config.max_token_len)),
                    _transforms.PadStatesAndActions(model_config.action_dim),
                ])
            case _model.ModelType.PI05:
                return _transforms.Group(inputs=[
                    _transforms.InjectDefaultPrompt(self.default_prompt),
                    _transforms.ResizeImages(224, 224),
                    _transforms.TokenizePrompt(
                        _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                        discrete_state_input=model_config.discrete_state_input,
                    ),
                    _transforms.PadStatesAndActions(model_config.action_dim),
                ])
            case _model.ModelType.PI0_FAST:
                ...
```

π0/π0.5 在数据侧的唯一区别就是：`TokenizePrompt` 是否传 `discrete_state_input=True`。

### 8.3 完整的 transform 链

数据从 LeRobot dataset 出来到喂模型的完整顺序：

```
原始 LeRobot dict
   │
   │ data.repack_transforms.inputs
   ▼  (RepackTransform 把 observation.images.cam_high 等键名重排)
{"image":{...}, "state":..., "actions":..., "prompt":...}
   │
   │ data.data_transforms.inputs
   ▼  (LiberoInputs/H1Inputs/AlohaInputs ... + DeltaActions)
{标准三相机 + 标准状态 + delta 动作}
   │
   │ Normalize(norm_stats)
   ▼
{[-1,1] 范围的 state/action}
   │
   │ data.model_transforms.inputs
   ▼  (Resize / TokenizePrompt / PadStatesAndActions)
{含 tokenized_prompt、pad 到 32 维的 state/action}
   │
   │ Observation.from_dict
   ▼
Pi0.compute_loss(observation, actions)
```

推理时模型输出后反向走一遍：`model_transforms.outputs` → `Unnormalize` → `data_transforms.outputs`（包含 `AbsoluteActions` 还原 delta、并截取真实维度）。

### 8.4 `_build_freeze_filter()` —— LoRA / 部分微调控制

```111:144:src/openpi/training/config.py
    if USE_LORA:
        # With LoRA: non-LoRA params always frozen; carve out trainable exceptions
        exceptions = []
        if TRAIN_SIGLIP:
            exceptions.append(_img)
        if TRAIN_LLM:
            exceptions.append(nnx.All(_lora, nnx.Not(_expert)))
        if TRAIN_ACTION_EXPERT:
            exceptions.append(nnx.All(_expert, _lora))
        ...
```

通过 `TRAIN_SIGLIP / TRAIN_LLM / TRAIN_ACTION_EXPERT / USE_LORA` 四个 env var，可以自由组合"训哪个组件 + 是否 LoRA"，逻辑全是 `nnx.filterlib` 的正则匹配。

### 8.5 `TrainConfig` 与三套 ego 配置

`_CONFIGS` 列表里 ego 项目主要关心：

| name | 机器人 | 关键设定 |
|------|--------|----------|
| `pi05_h1` | H1 人形 23DOF | `LerobotH1DataConfig` + `delta_action_mask = (1,2,2,7,-2,7,-2)`，3 相机 |
| `pi05_kaiumi` | Kaiumi 双臂 14D | `LeRobotKaiUmiDataConfig`，3 相机，delta=(6,-1,6,-1) |
| `pi05_pico` | Pico 第一人称 16D | `LeRobotPicoEgoDataConfig`，单相机，全维 delta |

它们共享：
- **基础权重**：`weight_loader=CheckpointWeightLoader(PI05_BASE_CHECKPOINT_PATH)` 从 `gs://openpi-assets/checkpoints/pi05_base/params` 加载；
- **`pi05=True`** + Gemma-2B（可选 LoRA）+ Gemma-300M action expert；
- **freeze filter**：由 env vars 决定；
- **CosineDecaySchedule**：peak_lr=2e-4、warmup 100、decay 3000、decay_lr 1e-6（H1 配置）。

---

## 9. 整体收束：把训练和推理串起来

### 9.1 训练一步发生了什么

```
DataLoader  ─►  (RepackTransform)            键名重排
            ─►  (*Inputs + DeltaActions)      delta + 三相机标准化
            ─►  (Normalize w/ q01,q99)        归一化到 [-1,1]
            ─►  (Resize + Tokenize + Pad)     图像 224、prompt token、维度 pad
            ─►  Observation.from_dict
            ─►  Pi0.compute_loss
                    │
                    ├─ preprocess_observation：augmax 增强
                    ├─ embed_prefix：SigLIP + PaliGemma.embed
                    ├─ embed_suffix：x_t = t*noise + (1-t)*action
                    ├─ Gemma 双 expert + adaRMSNorm（pi05）一次 forward
                    └─ MSE(v_t, noise - action) → 梯度 → optax → params 更新
```

### 9.2 推理一帧发生了什么

```
机器人观测  ─►  Policy._input_transform
            ─►  Observation.from_dict
            ─►  Pi0.sample_actions
                    │
                    ├─ embed_prefix → 一次 forward 填 KV cache
                    └─ for step in range(num_steps):
                           embed_suffix(x_t, t)
                           forward suffix only with KV cache
                           x_t += dt * v_t,  t += dt
            ─►  Policy._output_transform
            ─►  Unnormalize → AbsoluteActions → 取真实动作维 → 机器人执行
```

---

## 10. 二次开发常见入手点速查表

| 想做的事 | 改哪里 |
|---------|--------|
| 接入新机器人 | 写一份 `*Inputs/*Outputs`（参考 `libero_policy.py`），在 `config.py` 加一个 `*DataConfig` + `TrainConfig` |
| 改相机数量 | 在 `*Inputs` 里把没有的 view 用零图 + `image_mask=False` 占位；模型那边 `IMAGE_KEYS` 一律保持三个 key |
| 改 action 维度 | 数据侧写 delta mask 区分关节/夹爪；模型侧通过 `action_dim=32` 自动 pad；输出侧只取前 N 维 |
| 启用 / 关闭 LoRA | `USE_LORA` env var；模型 variant 切到 `gemma_2b_lora` / `gemma_300m_lora` |
| 只训动作头 | `TRAIN_LLM=false TRAIN_SIGLIP=false TRAIN_ACTION_EXPERT=true` |
| 改增强策略 | `Pi0Config(augmentation=get_h1_strong_augmentation())` 之类 |
| π0 ↔ π0.5 切换 | 仅需 `Pi0Config(pi05=True/False)` |
| 把动作生成步数加多 | `Policy(sample_kwargs={"num_steps": 20})` |
| 改归一化策略 | `DataConfig(use_quantile_norm=False)` 退回 z-score |

---

## 11. 一个常被忽视的细节合集

1. **Beta(1.5, 1) 时间分布**：训练时 t 偏向接近 0（接近真值），有助于学到精细的速度场。
2. **flow matching 方向**：`t=1=噪声、t=0=数据`，与 DDPM 相反。注释里也吐槽了"sorry"。
3. **`make_attn_mask` 的 cumsum 技巧**：用一行 cumsum 实现块状 attention，比写 N 段 if-else 优雅得多。
4. **`_name(name, i)` 命名规则**：让 PaliGemma 公开权重无缝加载到 expert 0；动作专家自动命中 `_1` 后缀 → 是 freeze filter 能写得这么短的根本原因。
5. **expert 间共享 attention**：靠 `jnp.concatenate(qkvs, axis=1)` 一行实现，是 π 系列的灵魂之笔。
6. **GQA 极端化**：8 query heads ↔ 1 KV head，KV cache 体积小 8 倍，flow matching 推理才能在 10 步内"几乎免费"。
7. **EMA**：`pi05_h1` 等微调配置 `ema_decay=None` 关闭 EMA（LoRA 微调没必要）；预训练时 `ema_decay=0.99`。
8. **GPU buffer 复用**：`Policy._ensure_gpu_buffers` 是 ego 版本独有的推理优化，避免高频 `device_put` 导致显存碎片。

---

读完这套源码后，可以这样理解 π0.5：**它本质上是一个把"图像 + 文本（含状态）"喂到 PaliGemma 做条件，再用一个轻量 Action Expert 走 Flow Matching 在 32 维连续动作空间里去噪的模型**。两端用统一的 `Observation` 抽象、用 `Group` 把数据流拼起来，再用 `freeze_filter` 优雅地切换微调粒度 —— 整个仓库非常模块化，研究新机器人时基本只动 `policies/*.py` 和 `training/config.py` 就够了。

# MagicVLA 短时视觉记忆实现说明：v0 与 magicvla-mem-v1

本文只记录 MagicVLA 已实现过的两版短时视觉记忆：`v0` 和当前代码中的
`magicvla-mem-v1`。论文调研、长期 memory bank、事件 keyframe、文本摘要等未落地
方案不在本文讨论范围内。

当前代码只保留 `magicvla-mem-v1`；v0 的说明根据 Git 基线和 v0 训练配置保留，便于
理解历史实验与比较结果。两版都属于“每个样本显式携带固定长度历史”的无状态短时
视觉记忆，不在模型内部维护跨推理步的 episode buffer。

## 1. 共同的数据与模型契约

### 1.1 记号

| 记号 | 含义 | 当前配置 |
|---|---|---:|
| `B` | 每个 rank 的 batch size | v0 为 60，v1 为 48 |
| `V` | 配置的相机数 | 3 |
| `S` | batch 中有效的“样本-相机”序列数，`S <= B*V` | 三相机全有效时为 `3B` |
| `K` | 每条序列的观测帧数 | 6 |
| `P` | 每帧 merger 前的视觉 patch 数 | `16*16=256` |
| `H_v` | Qwen3.5 vision hidden size | 1024 |
| `N_h` | vision attention heads | 16 |
| `D_h` | 每个 vision head 的维度 | `1024/16=64` |
| `H_l` | merger 后、进入语言模型的 hidden size | 2048 |

下文中的 block index 均为 **0-based**。Qwen3.5-2B 的视觉塔共有 24 个 block，索引为
`0..23`。例如 block index 19 是第 20 层。

### 1.2 历史帧采样

三个数据源均配置：

```yaml
obs_steps: 6
obs_interval_seconds: 1.0
```

reader 根据数据集 FPS 计算帧步长：

```text
stride = max(1, round(obs_interval_seconds * fps))
offsets = [-5*stride, -4*stride, -3*stride, -2*stride, -stride, 0]
```

因此每个 anchor 返回 oldest-first 的 6 帧，目标时间通常约为
`[-5 s, -4 s, -3 s, -2 s, -1 s, 0 s]`。实际 `dt` 使用量化后的
`offset/fps`，不假定 FPS 一定能精确表示 1 秒。

episode 开头不足 6 帧时，负索引会 clamp 到 episode 的第 0 帧，但同时返回两个显式
字段，防止重复像素被当成真实历史：

- `observation.history_valid`: `[K] bool`，requested frame 是否真实存在；
- `observation.history_dt`: `[K] float32`，相对当前 anchor 的秒数，非正、oldest-first，
  且最后一个值严格为 0。

reader 对单条样本的主要输出 shape 为：

```text
image/<camera>                 [K, 3, H, W]
observation.state             [K, 32]
observation.state_dim_mask    [K, 32]
action                        [50, 32]
observation.history_valid     [K]
observation.history_dt        [K]
```

collate 后相应增加 batch 维。processor 会把 `state[-1]` 和
`state_dim_mask[-1]` 作为当前 state，因此两版都只实现**视觉历史**；过去 5 个 state
不会作为 proprio memory token 输入模型。action target 仍为当前 anchor 开始的
`[B, 50, 32]` action chunk。

历史索引只在当前 episode 内构造，不读取未来帧，也不会跨 episode。

### 1.3 从 batch 到视觉序列

模型先按 `sample -> valid camera -> oldest-to-current frame` 的顺序展平图像。无效相机
不会形成 memory sequence。整理后的辅助张量为：

```text
memory_sequence_valid       [S, K]
memory_sequence_dt          [S, K]
memory_sequence_group_ids   [S]       # v1 新增；值为原 batch sample index
```

`sequence_group_ids` 用于让同一样本的多相机共享随机 history mask。例如同一个机器人
样本的 high、left wrist、right wrist 三路相机会一起进入 current-only 模式，或丢弃
相同的历史 slot，而不是各自独立随机化。

每个 256×256 RGB 图像经当前 Qwen image processor 后：

```text
pixel_values per frame      [P, 1536] = [256, 2*3*16*16]
image_grid_thw per frame    [3]       = [1, 16, 16]
```

其中 Qwen3.5 vision config 为 `patch_size=16`、`temporal_patch_size=2`。单图 processor
按 Qwen 的单图规则形成 1536 维 patch 输入。把全部 memory 图像打包后：

```text
memory_pixel_values         [S*K*P, 1536]
memory_grid_thw             [S*K, 3]
                            每一行都是 [1, 16, 16]
```

视觉 `patch_embed` 及绝对位置编码之后：

```text
hidden_states               [S*K*P, H_v]
                            [S*K*256, 1024]
```

普通 Qwen vision block 通过 `cu_seqlens` 把每张图像划为独立 segment，只在单帧的
256 个空间 patch 内做 attention。只有配置为 temporal block 的层会显式 reshape 出
`[S, K, P, ...]` 并在时间维交互。

### 1.4 固定的语言侧 token 预算

两版都会在视觉塔中途删除过去帧，只保留每条相机序列的当前帧：

```text
drop 前     [S, K, P, 1024]
取 [:,-1]   [S, P, 1024]
展平后      [S*P, 1024]
```

剩余 vision block 只处理当前帧。最后 Qwen spatial merger 使用
`spatial_merge_size=2`，把每个 2×2 patch 合并为一个语言侧视觉 token：

```text
merger 输入                  [S*256, 1024]
merger 输出                  [S*64, 2048]
每个有效相机进入 VLM         64 个 token
```

因此 v0、v1 与 current-only 在 VLM prefix 中使用相同数量的视觉 token。固定的是
**语言侧 token 数**，不是视觉塔计算量：drop 之前仍需处理 `K=6` 倍的帧，并额外执行
时间 attention。

模型会比较 memory packing 中每条序列的 current-frame grid 与 chat template 为当前
图像生成的 placeholder grid；不一致会直接报错。memory encoder 的输出随后替换当前
图像 placeholder 对应的 embedding。

## 2. v0 实现

### 2.1 配置与层分布

v0 配置文件为：

`configs/train/magicvla_base_sft_robodojo_short_mem_v0_from88000_fla_bs60_w8_pf2.yaml`

核心参数：

```yaml
memory_mode: mem_vision
short_memory_frames: 6
short_memory_interval_seconds: 1.0
temporal_block_indices: [3, 7, 11, 15]
past_drop_layer: 16
temporal_time_max_period_seconds: 64.0
freeze_vision_encoder: true
knowledge_insulation: false
```

它在第 4、8、12、16 个 vision block（index `3,7,11,15`）插入 temporal delta
branch，并在执行完 block 15 后 drop：

```text
block 0..15   处理 K=6 帧，其中 3/7/11/15 带 temporal delta branch
drop          只保留 current frame
block 16..23  只处理 current frame
merger        每相机输出 64 个 token
```

以“每个 block 处理的 frame 数”粗略计算，主视觉 block 工作量下界约为
`(16*6 + 8)/24 = 4.33` 倍 current-only；这个估算尚未计入 4 个 temporal block 的
额外 QKV 和 attention。

### 2.2 zero-gated temporal delta branch

设进入 temporal block 的输入为：

```text
X                         [S*K*P, 1024]
Q, K_attn, V_attn         [S, K, P, 16, 64]
```

注意这里用 `K_attn` 表示 key，避免与帧数 `K` 混淆。v0 首先完整执行一次原始 Qwen
空间 attention：

```text
A_base = SpatialAttention(Q, K_attn, V_attn)
```

然后再次调用相同 block 的 QKV projection，专门计算 temporal delta branch。因此每个
temporal block 的 QKV projection 实际执行两次：一次在原始 `block.attn` 内，一次在
新增分支内。

连续时间编码宽度为每头维度 64：

```text
E(dt)                     [S, K, 64]
temporal Q/K/V            [S, P, 16, K, 64]
temporal logits           [S, P, 16, K, K]
```

`E(dt)` 使用 `age=-dt` 的 sinusoidal embedding，周期在 1 到 64 秒间按对数分布；
cosine 分量使用 `cos(angle)-1`，所以当前帧 `dt=0` 的时间向量严格为全零。v0 把它加到
每个 head 的 temporal Q/K 上，不加到 V 或 block residual：

```text
Q_t = Q + E(dt)
K_t = K_attn + E(dt)
M   = softmax((Q_t @ K_t^T) / sqrt(64) + causal/valid mask) @ V_attn
DeltaV = M - V_attn
```

时间 attention 只在同一 `sequence + spatial patch + head` 内沿 K 帧计算，不在相机间
或不同空间位置间混合。causal 条件为：

```text
allowed(query_t, key_t) = dt_query >= dt_key AND history_valid[key_t]
```

因此当前帧可读取全部有效过去帧，过去帧不能读取未来帧。invalid slot 不能作为 key，
其 temporal 输出随后也被置零。

每个 temporal block 有一个可训练的 per-head gate：

```text
gate[index]               [16]
alpha = tanh(gate)        [16]
```

v0 共 4 个 gate，即只新增 `4*16=64` 个参数。gate 全部零初始化。`DeltaV` 乘
`alpha` 后再做一次空间 attention，作为差分注入：

```text
A_delta = SpatialAttention(Q, K_attn, alpha * DeltaV)
Y = X + A_base + Proj_no_bias(A_delta)
Y = Y + MLP(Norm2(Y))
```

delta projection 故意不重复添加 `block.attn.proj.bias`。初始 `alpha=0` 时，
`A_delta=0`，所以无论 K 是否大于 1，整个 temporal block 都严格走原始空间路径。
这让 v0 从 current-only checkpoint 启动时非常保守。

### 2.3 v0 的训练行为与限制

v0 设置 `freeze_vision_encoder: true`，Qwen vision 的 patch embedding、24 个 block、
QKV、projection、MLP、norm 和 merger 均冻结。新增 gate 位于 vision module 外，仍可由
flow-matching loss 更新。VLM 语言部分和 action expert 并未因该选项冻结。

这也形成了 v0 的主要能力瓶颈：

- temporal branch 只能通过 64 个 gate 学习每层、每个 head 应注入多少差分；
- 用于时间/空间匹配的共享 QKV 全部冻结，不能适应机器人视频中的跨帧对应关系；
- zero gate 保证初始化稳定，但训练初期历史信号完全被关断，学习信号必须先穿过 gate；
- 没有 history dropout 或 current-only 随机化；
- drop 发生在第 16 层，只有前 16 层能用历史，后 8 层只细化当前帧表示。

v0 checkpoint loader 对 current-only checkpoint 使用“有限 strict”加载：只允许缺少
`short_memory_vision.temporal_gates.*`，其他 missing/unexpected key 仍报错。

## 3. magicvla-mem-v1 实现

### 3.1 配置与层分布

当前 v1 配置文件为：

`configs/train/magicvla_base_sft_robodojo_magicvla_mem_v1_from88000.yaml`

核心参数：

```yaml
memory_mode: mem_vision
short_memory_frames: 6
short_memory_interval_seconds: 1.0
temporal_block_indices: [3, 7, 11, 15, 19]
past_drop_layer: 20
temporal_time_max_period_seconds: 64.0
history_dropout_prob: 0.20
current_only_probability: 0.25
freeze_vision_encoder: false
knowledge_insulation: false
```

它在第 4、8、12、16、20 个 vision block 做 time-then-space attention，并在执行完
block 19 后 drop：

```text
block 0..19   处理 K=6 帧，其中 3/7/11/15/19 为 time-then-space block
drop          只保留 current frame
block 20..23  只处理 current frame
merger        每相机输出 64 个 token
```

按主视觉 block 的 frame 数粗略计算，工作量下界约为
`(20*6 + 4)/24 = 5.17` 倍 current-only。相比 v0，多保留历史 4 层，并增加第 20 层的
一次 temporal fusion；但 v1 的 temporal block 只执行一次 QKV，而不是 v0 的两次。

### 3.2 参数为零新增的 time-then-space attention

v1 删除了 v0 的 `temporal_gates`、原始 base attention 加 delta attention 的双分支，
改为每个 temporal block 只执行一次 QKV 和一次空间 attention。它复用 Qwen 原 block
的 norm、QKV、attention output projection 和 MLP，不创建任何新 Parameter，也不新增
state-dict key。

输入仍为：

```text
X                         [S*K*P, 1024]
reshape(X)                [S, K, P, 1024]
```

v1 的连续时间编码宽度从 v0 的 per-head 64 改为完整 hidden size 1024：

```text
E(dt)                     [S, K, 1024]
X_timed                   [S, K, P, 1024]
X_timed = reshape(X) + E(dt)[:, :, None, :]
```

1024 维由 512 组 sinusoidal 频率构成，周期从 1 到
`temporal_time_max_period_seconds=64` 秒按对数分布。仍使用 `sin(angle)` 与
`cos(angle)-1`，因此 `E(0)` 精确为零。时间向量只参与该 block 的 norm/QKV 路径；
block residual 加的仍是原始 `X`，不会把时间向量永久累加到 hidden state。

经 `Norm1 -> QKV -> spatial rotary` 后：

```text
Q, K_attn, V_attn         [S, K, P, 16, 64]
temporal Q/K/V            [S, P, 16, K, 64]
temporal logits           [S, P, 16, K, K]
M                         [S, P, 16, K, 64]
```

Q/K 先使用 Qwen 原有的二维空间 rotary embedding；随后同一个空间 patch 沿时间维做
causal attention：

```text
M = softmax((Q @ K_attn^T) / sqrt(64) + causal/valid mask) @ V_attn
```

mask 规则与 v0 相同：当前帧能读取全部有效历史，过去帧不能看未来，invalid history
不能作为 key。invalid query 对应的 `M` 会被置零。

接着把时间混合后的 `M` 当作空间 attention 的 value，而 Q/K 仍来自当前帧各自的
spatial Q/K：

```text
spatial Q/K/V             [S*K, 16, P, 64]
A = SpatialAttention(Q, K_attn, M)
Y = X + Proj(A)
Y = Y + MLP(Norm2(Y))
```

这就是“time-then-space”：先让每个 patch 的 value 汇聚因果历史，再让每帧的所有空间
patch 读取这些带历史的 value。与 v0 不同，历史不是一个 gated residual delta，而是
temporal block 唯一的 attention value 路径。

当 `K=1` 时，时间 softmax 只有一个元素，`M=V_attn`；同时当前时间编码为零，所以该
block 数值上退化为原始单图 Qwen vision block。当前测试覆盖了该兼容性。

### 3.3 训练期 history mask 随机化

v1 在进入视觉塔前，基于已有 `history_valid` 生成 train-only mask。eval/inference
模式不执行随机化，当前帧在所有模式下都强制有效。

随机化按 batch sample 分组，同一样本的所有有效相机共享结果，顺序如下：

1. 以 `current_only_probability=0.25` 的概率把该样本全部 `K-1` 个过去 slot 置为
   invalid；
2. 对未被上一步统一屏蔽的历史 slot，再以 `history_dropout_prob=0.20` 独立丢弃；
3. 强制当前 slot `valid=True`。

对一个原本有效的指定历史 slot，其边缘保留概率为：

```text
(1 - 0.25) * (1 - 0.20) = 0.60
```

这两个 mask 用于训练模型同时保持 current-only 能力，并避免过度依赖某一个固定的
历史位置。它们只改变 attention 可见性，**不会跳过历史图像解码、预处理、
patch embedding 或 drop 前的 vision block**，因此不能直接节省 CPU 内存或视觉计算。

### 3.4 v1 的训练与 checkpoint 行为

v1 设置 `freeze_vision_encoder: false` 和 `knowledge_insulation: false`。flow-matching
loss 可以通过 VLM prefix 回传到整个 vision tower，使共享 QKV、projection、MLP、
norm、patch embedding 和 merger 都能适应 temporal mixing。优化器基础 LR 为 `2e-5`，
`vlm_lr_scale=0.1`，因此 vision/VLM 参数使用 `2e-6`，action expert 使用基础 LR。

由于 memory wrapper 没有参数，原始 step-88000 current-only checkpoint 可以
`strict=True` 直接加载，无需 missing-key allowlist。反过来，带有
`short_memory_vision.temporal_gates.*` 的 v0 训练 checkpoint 不能作为 v1 strict
checkpoint 直接加载，因为这些 key 对 v1 是 unexpected keys。当前 v1 按要求仍从同一个
原始 step-88000 model-only checkpoint 开始。

## 4. v0 与 v1 的核心差异

| 项目 | v0 | magicvla-mem-v1 |
|---|---|---|
| 代码状态 | 已被替换，仅保留历史说明 | 当前实现 |
| vision 深度 | 24 blocks | 24 blocks |
| temporal blocks（0-based） | `[3,7,11,15]` | `[3,7,11,15,19]` |
| 历史 drop | block 15 后，即前 16 层后 | block 19 后，即前 20 层后 |
| drop 后 current-only blocks | 8 层 | 4 层 |
| temporal 结构 | 原空间 attention + gated delta branch | 单一路径 time-then-space attention |
| temporal block 的 QKV 次数 | 2 次 | 1 次 |
| 时间编码 shape | `[S,K,64]`，加到每头 temporal Q/K | `[S,K,1024]`，在 norm/QKV 前加到 hidden |
| 新增参数 | 4×16=64 个 zero-init gates | 0 |
| vision tower | 冻结 | 解冻 |
| 初始历史影响 | gate=0，历史完全不影响输出 | K>1 时历史立即进入 value 路径 |
| K=1 兼容 | zero gate 下等于原空间路径 | temporal length=1 时等于原空间路径 |
| 随机 history mask | 无 | 20% slot dropout + 25% current-only |
| 多相机随机一致性 | 不适用 | 同一样本三相机共享 mask |
| VLM 每相机视觉 token | 64 | 64 |
| checkpoint 加载 | 允许只缺 gate keys | 不新增 key，可直接 strict 加载原始 checkpoint |

v1 的设计取舍是：放弃 v0 最保守的 zero-gate 启动，换取更直接的历史信息路径，并通过
解冻 vision tower 让跨帧 QKV 真正获得训练信号。current-only 随机化和 history slot
dropout 用于约束鲁棒性，而原始 checkpoint 的兼容性由“零新增参数 + K=1 退化”保证。

## 5. 多源数据加载与 CPU 内存

v0 YAML 没有设置 `mixing_level: sample`。三个顶层数据源会各自创建 DataLoader，再由
外层 loader 按 source 交错取 batch。按 v0 文件中的实际值：

```text
3 sources
batch_size=60 per rank
num_workers=8 per source per rank
prefetch_factor=2
pin_memory=true
```

若使用 8 个 rank，这相当于最多 `3*8*8=192` 个 worker；每个 worker 都有自己的数据集、
视频 decoder/cache 和 prefetch queue。每个 batch 又包含最多
`60*3*6=1080` 张 256×256 图像，因此 host memory 占用会非常大。v0 YAML 顶部注释中的
“6 workers”与实际 `num_workers: 8` 不一致，应以字段值 8 为准。

v1 改为 sample-level mixing：

```yaml
data:
  batch_size: 48
  num_workers: 2
  prefetch_factor: 1
  mixing_level: sample
training:
  pin_memory: false
```

factory 的实际结构为：

```text
3 source datasets
    -> ConcatDataset
    -> WeightedMixtureSampler(weights=[1,1,1])
    -> EpochBatchSampler(batch_size=48)
    -> 1 physical DataLoader per rank
    -> InterleavedLoader（内部只有这 1 个 iterator）
```

所以 v1 不是为每个 source 分别创建 DataLoader。8 rank 时共有 16 个 worker，每个 rank
只有一个 loader 和 2 个 prefetch worker；`pin_memory=false` 也避免把大批 K-frame tensor
长期固定在 pinned host memory。三个 source 的权重仍均为 1，sampler 在 sample 层面
保持等权混合。

需要注意：v1 的 per-rank batch 从 60 降为 48，若使用 8 GPU 且没有额外梯度累积，
global batch 会从 480 变为 384。这是当前为控制 host/GPU memory 做出的明确变化。

## 6. 当前实现边界与已验证契约

两版都没有实现：

- 模型内部或 policy wrapper 中的 recurrent/ring-buffer memory；
- 动态 keyframe 选择、检索或淘汰；
- 长期视觉 memory、文本摘要 memory；
- 历史 proprio/state token；
- 跨相机 temporal attention；
- 通过随机 mask 实际跳过历史帧计算。

当前 v1 的测试覆盖以下关键契约：

- history 必须为 `[B,K]`、oldest-first、`dt<=0`，且 current 有效并满足 `dt=0`；
- 不跨 episode 的历史索引、episode 开头 valid mask 和真实 dt；
- 相机输入必须为 `[B,K,C,H,W]`，所有 memory frame 使用一致单图 grid；
- causal temporal attention 不读取未来或 invalid history；
- 同一样本多相机共享随机 mask，eval 模式关闭随机化；
- current frame 永不被丢弃；
- 在配置层后只输出 current-frame token；
- `K=1` 与原 Qwen 单图 block 数值兼容；
- 有效历史能改变当前输出，并能让共享 vision QKV 获得非零梯度；
- sample-level mixing 最终只创建一个物理 DataLoader。

对应实现入口：

- memory encoder：`src/models/magicvla_base/short_memory.py`
- 配置约束：`src/models/magicvla_base/configuration_magicvla_base.py`
- batch packing 与视觉 embedding：`src/models/magicvla_base/modeling_magicvla_base.py`
- 历史数据契约：`src/data/lerobot_history.py`
- 单 DataLoader 多源混合：`src/data/factory.py`
- v0 recipe：`configs/train/magicvla_base_sft_robodojo_short_mem_v0_from88000_fla_bs60_w8_pf2.yaml`
- v1 recipe：`configs/train/magicvla_base_sft_robodojo_magicvla_mem_v1_from88000.yaml`

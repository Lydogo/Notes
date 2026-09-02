# π0.5 FrameSamp+Modul 实现详解

本文按训练时的真实执行顺序，说明本仓库中 FrameSamp+Modul 的完整实现。

## 1. 最重要的结论

这套实现包含两条彼此独立的视觉路径：

1. π0.5 原有视觉路径：当前时刻的头部、左腕和右腕三路图像进入普通 VLM prefix。
2. FrameSamp 记忆路径：当前时刻及之前的 `cam_high` 历史特征构成固定 512-token memory，只供 action expert 使用。

因此，历史帧不是第四张 VLM 图片，也不会增加 PaliGemma prefix 的图像数量。它们通过每层 action expert 中新增的 memory cross-attention 影响动作预测。

整体数据流如下：

```text
当前三路 RGB ──> 原 π0.5 SigLIP/VLM prefix ───────────────┐
                                                          │
历史 cam_high 视频 ──> 离线 SigLIP ──> 4x4 池化特征       │
                                      │                   │
                                      └─> FrameSamp       │
                                          + 3D 位置编码   │
                                          + 右侧 padding  │
                                                  │       │
                                                  v       v
                                          512 memory tokens
                                                  │
                                      每层 action expert Modul
                                                  │
                                                  v
                                           flow-matching 动作损失
```

## 2. 固定配置与张量尺寸

本任务使用以下配置：

| 配置 | 数值 | 含义 |
| --- | ---: | --- |
| `framesamp_budget` | 512 | 每个训练样本固定的 memory token 数 |
| `framesamp_tokens_per_frame` | 16 | 每个历史帧保留 4×4 个视觉 token |
| 最大历史帧数 | 32 | `512 / 16` |
| `framesamp_image_dim` | 2048 | SigLIP patch token 维度 |
| `framesamp_position_dim` | 768 | 3D 时间/空间位置编码维度 |
| `framesamp_state_dim` | 8 | 与 RoboMME 配置一致的占位维度，当前不参与编码 |
| `framesamp_memory_dim` | 1024 | action expert 的宽度 |
| memory view | 1 | 只使用 `observation.images.cam_high` |

一个 batch 的最终 memory 输入为：

```text
static_image_emb: [B, 512, 2048] float32
static_pos_emb:   [B, 512,  768] float32
static_state_emb: [B, 512,    8] float32
static_mask:      [B, 512]       bool
```

配置构造阶段还会检查：该功能只能用于 π0.5；budget 必须能整除每帧 token 数；每帧 token 数只能是 4、16 或 64；位置维度必须能被 6 整除；memory width 必须等于 action expert width。模型开关与数据缓存路径也必须同时启用或同时关闭。

## 3. 第一步：离线预存头部相机视觉特征

入口脚本：

```bash
./precompute_robodojo_match_and_pick_framesamp_features.sh
```

它调用 `scripts/precompute_robodojo_framesamp_features.py`，默认处理：

```text
/pfs/public/RoboDojo_lerobot_v21_video_single_task/
RoboDojo_lerobot_v21_video_match_and_pick_from_conveyor_offical
```

默认缓存输出到：

```text
/pfs/public/RoboDojo_lerobot_v21_video_single_task/
RoboDojo_lerobot_v21_video_match_and_pick_from_conveyor_offical/
framesamp_features/pi05_base_cam_high_4x4
```

### 3.1 加载视觉编码器

预存脚本创建一个未开启 memory 的普通 `Pi0Config(pi05=True)`，加载：

```text
/pfs/public/Models/pi05_base/params
```

随后只保留 JIT 后的 `model.vision_encode`。因此，历史特征来自训练开始前的 π0.5 base SigLIP，不会在微调过程中变化。

### 3.2 逐帧解码和预处理

对每个 episode 的 `cam_high` 视频：

1. 按数据集 FPS 为每帧生成时间戳 `frame_index / fps`。
2. 使用 LeRobot `decode_video_frames` 解码 RGB。
3. 兼容 uint8 `[0,255]` 和 float `[0,1]` 两种返回格式。
4. 转换为 `[-1,1]`。
5. 使用 `resize_with_pad` 调整到 `224×224`。
6. 输入 π0.5 SigLIP，得到 `[batch, 256, 2048]`。

这里的 256 个 token 对应 16×16 patch 网格。

### 3.3 4×4 空间池化

`pool_spatial_tokens(..., 16)` 将 16×16 网格划分为 4×4 个区域。每个输出 token 是原网格中对应 4×4 token 区域的均值：

```text
[batch, 256, 2048] -> [batch, 16, 2048]
```

最终每个 episode 保存：

```text
episode_XXXXXX/image_emb_4x4.npy
shape = [episode_length, 16, 2048]
dtype = float16
```

缓存使用 FP16 减少磁盘占用；训练取样时会转换为 FP32，再由模型按配置使用 BF16 计算。

### 3.4 缓存完整性和断点续跑

根目录下的 `manifest.json` 记录：

- 数据集路径；
- base checkpoint 路径；
- camera key；
- FPS；
- 每帧 token 数和特征维度；
- 每个 episode 的长度；
- 缓存是否全部完成。

脚本开始时先写 `complete=false`，全部 episode 成功后才改为 `true`。单个 episode 先写 `.partial.npy`，完成后再原子重命名，避免把中断文件当成有效缓存。

再次运行时，形状和 dtype 正确的 episode 会直接跳过。训练侧拒绝加载 `complete!=true` 的缓存。

训练侧会校验 manifest 版本、每帧 token 数、特征维度、完成状态，以及每个 episode 文件的帧数。manifest 中的 dataset、camera 和 checkpoint 路径用于记录来源，目前不会与训练 config 自动逐项比较，因此不要把形状相同但来源不同的缓存目录误填到 config 中。

当前缓存已经完成：100 个 episode、43,366 帧，约 2.7 GiB。

## 4. 第二步：为每个训练样本构造 FrameSamp memory

`FrameSampFeatureDataset` 包装原来的 LeRobot/Magiclab dataset。原数据集仍负责返回当前图像、状态、动作、prompt、`episode_index` 和 `frame_index`；包装器额外读取相应 episode 的视觉缓存。

当前实现只能使用一个 dataset root，因为 episode index 到缓存目录的映射没有包含多数据集 namespace。

episode 特征通过 NumPy mmap 读取，不会一次把所有 2.7 GiB 加载进内存。每个 worker 最多维护 16 个 episode 的 LRU 映射。

### 4.1 均匀采样规则

最大帧数为：

```text
max_frames = 512 / 16 = 32
```

对当前帧 `t`：

- 当 `t < 32` 时，使用 `[0, 1, ..., t]` 的全部历史；
- 当 `t >= 32` 时，使用 `np.linspace(0, t, 32, dtype=int32)` 均匀采样 32 帧。

这与 RoboMME 的 `even_sampling_indices` 一致。`linspace` 的两个端点固定，因此：

- 第 0 帧始终存在；
- 当前帧始终存在；
- 中间历史随 episode 推进被均匀稀疏采样。

例如：

```text
t = 0   -> [0]
t = 5   -> [0,1,2,3,4,5]
t = 31  -> 0 到 31，共 32 帧
t = 100 -> 0 到 100 均匀取 32 帧，包含 0 和 100
```

这就是“首帧信息始终保留”的具体保证。

### 4.2 token 排列和右侧 padding

每个采样帧有 16 个 token，按“帧优先、帧内空间 token 次序”展平：

```text
frame_0 的 16 token,
frame_1 的 16 token,
...
frame_current 的 16 token
```

早期帧数不足 32 时，在右侧补零到 512 token，并用 `static_mask` 标记有效部分：

```text
valid_tokens = sampled_frame_count * 16
static_mask[:valid_tokens] = True
static_mask[valid_tokens:] = False
```

从 `t=31` 开始已经有 32 帧，512 个位置全部有效。

## 5. 第三步：生成 RoboMME 3D 位置编码

每个历史视觉 token 都有一个 768 维的时间/空间正弦编码。

因为 `768 / 6 = 128`，它由六组 128 维编码组成：

```text
时间: sin(t), cos(t)               -> 256 维
纵向: sin(y), cos(y)               -> 256 维
横向: sin(x), cos(x)               -> 256 维
合计                                -> 768 维
```

时间频率基数为 10,000，空间频率基数为 1,000。

4×4 池化网格对应原 16×16 patch 网格中的中心坐标：

```text
2, 6, 10, 14
```

因此，同一帧内 16 个 token 的时间编码相同、空间编码不同；不同帧相同空间位置的空间编码相同、时间编码不同。

注意：这里使用原始 episode `frame_index` 作为时间位置，而不是采样后的 0、1、2 编号，所以长时间跨度仍会反映在位置编码里。

## 6. 第四步：memory 字段穿过数据变换链

FrameSamp 字段在原数据变换中容易被 `RepackTransform` 丢弃，因此实现增加了三层显式透传：

1. `OPTIONAL_REPACK_PRESERVE_KEYS` 保留四个 `static_*` 字段。
2. `AgilexInputs` 将它们复制到模型输入。
3. `Observation.from_dict` 和 `preprocess_observation` 将它们保存到最终 Observation。

这四个字段不属于普通图像字典，不参与当前 RGB 的 crop、rotate 或 color jitter；也不使用 RoboDojo state/action 的归一化统计。

与此同时，当前时刻三路相机仍执行原来的数据处理：

```text
cam_high       -> base_0_rgb
cam_left_wrist -> left_wrist_0_rgb
cam_right_wrist-> right_wrist_0_rgb
```

因此当前 `cam_high` 有两个角色：

- 作为当前图像进入正常 VLM prefix；
- 其冻结特征作为最新历史帧进入 FrameSamp memory。

## 7. 第五步：PerceptualMemory 编码

`PerceptualMemory` 与 RoboMME 的 FrameSamp 配置一致，且 `use_pos_emb=true`、`use_state_emb=false`。

计算过程为：

```text
position_features = SiLU(Linear(768 -> 768)(static_pos_emb))
features = concat(static_image_emb, position_features)
memory_tokens = Linear(2048 + 768 -> 1024)(features)
```

输出：

```text
memory_tokens: [B, 512, 1024]
memory_mask:   [B, 512]
```

两个 Linear 的 kernel 都使用 `normal(stddev=0.02)` 初始化。

`static_state_emb` 仍按 RoboMME 接口提供，但当前配置明确不使用它；其内容全部为零，不会进入 concat。保留该字段是为了维持 representation 接口和未来扩展兼容性。

## 8. 第六步：Modul 如何进入 action expert

π0.5 的 PaliGemma 模块有两个 expert：

1. VLM expert，处理图像和语言 prefix；
2. action expert，处理 flow-matching 的动作 suffix。

memory 只传给第二个 expert：

```python
mem_seq  = [None, memory_tokens]
mem_mask = [None, memory_mask]
```

所以 VLM expert 的计算图不直接增加 512 个历史 token。

### 8.1 MemoryAttention

在每一层 action expert 中，动作 token 作为 query，memory token 作为 key/value：

```text
query:  action hidden states [B, T, 1024]
key/value: memory tokens     [B, 512, 1024]
```

π0.5 action expert 的实际注意力配置为：

```text
num_heads    = 4
num_kv_heads = 1
head_dim     = 256
width        = 1024
```

具体步骤：

1. 使用同一个 `MemoryRMSNorm` 分别归一化 action 和 memory。
2. action 生成 Q，memory 生成 K/V。
3. memory 的 RoPE 位置为 `[0, ..., 511]`。
4. action query 的 RoPE 位置接在 memory 之后，从 512 开始。
5. `static_mask=false` 的 padding token 在 softmax 前被屏蔽。
6. cross-attention 输出每个 action token 对应的 1024 维 `mem_mod_vec`。

### 8.2 条件调制

`mem_mod_vec` 不作为普通 residual 直接加到 action hidden state，而是生成归一化的 scale 和 shift：

```text
[scale, shift] = Linear(1024 -> 2048)(mem_mod_vec)
normalized_x = RMSNorm(x)
x_modulated = normalized_x * (1 + scale) + shift
```

该 modulation Linear 使用 `normal(stddev=0.002)` 的小随机初始化，和 RoboMME 一致。

### 8.3 在 Transformer block 中的位置

每层顺序是：

```text
原 self-attention
    -> attention gated residual
    -> MemoryAttention
    -> MemoryRMSNorm 条件调制
    -> 原 pi0.5 timestep AdaRMSNorm
    -> 原 action FFN
    -> FFN gated residual
```

也就是说，memory modulation 位于每一层 action expert 的 FFN 之前，不替换原来的时间条件 AdaRMSNorm。

Gemma 层使用 `nn.scan`，因此新增 memory 参数沿 18 层堆叠，每层都有独立的 MemoryAttention 和 modulation 参数。

## 9. 第七步：训练前向和损失

π0.5 原来的 flow-matching 训练逻辑保持不变：

1. 对动作采样噪声 `noise`。
2. 采样 flow 时间 `time`。
3. 构造带噪动作 `x_t`。
4. 目标速度为 `u_t = noise - actions`。
5. 当前图像和 prompt 构成 prefix。
6. `x_t` 和 time 构成 action suffix。
7. FrameSamp memory 先编码一次，然后在 18 层 action expert 中重复使用。
8. `action_out_proj` 输出预测速度 `v_t`。
9. 使用原来的平方误差 flow-matching loss。

因此，FrameSamp+Modul 改变的是 action expert 的条件信息，不改变动作标签、噪声构造、输出维度或 loss 定义。

## 10. Base checkpoint 与新增参数

训练从原 π0.5 base checkpoint 加载已有参数：

```text
/pfs/public/Models/pi05_base/params
```

base checkpoint 中没有以下新增参数：

- `mem_encoder`；
- 每层 `mem_attn`；
- 每层 `mem_rms_norm`；
- 每层 FFN 前的 memory modulation Dense。

`CheckpointWeightLoader` 允许名称包含 `mem_` 的缺失参数保留训练初始化值，同时严格加载已有 π0.5 参数。这样不会要求人为制作一个带空 memory 权重的新 base checkpoint。

训练配置冻结所有 `img` 参数，即 SigLIP 不更新。以下部分仍参与训练：

- PaliGemma/Gemma 参数；
- action expert；
- PerceptualMemory；
- 每层 MemoryAttention 和 modulation；
- action 输入、时间和输出投影。

## 11. 训练配置

配置名：

```text
pi05_robodojo_match_and_pick_framesamp_modul
```

关键训练参数：

```text
global batch size: 64
num_workers:       4
FSDP devices:      8（由启动脚本传入）
peak_lr:           2.5e-5
decay_lr:          2.5e-5
warmup_steps:      500
num_train_steps:   10,000
gradient clipping: 1.0
EMA:               0.999
save_interval:     1,000
```

归一化参数继续复用：

```text
assets_dir = /pfs/public
asset_id   = RoboDojo_lerobot_v21_video
```

学习率、warmup 和训练步数沿用本仓库原 RoboDojo 单任务 SFT 配置；memory 架构参数、batch size、worker 数和 EMA 对齐 RoboMME 的 memory 训练配置。

启动命令：

```bash
cd /pfs/user/kai0-robodojo-frame-mem
./train_robodojo_official_100_match_and_pick_framesamp_modul.sh
```

## 12. 与“只输入首帧”的区别

旧思路是把首帧当作额外图片放进 VLM prefix。当前实现不是这样。

当前实现的行为是：

- 第 0 帧永远保留；
- 当前帧永远保留；
- 中间最多保留 30 个均匀分布的历史帧；
- 历史总长度固定为最多 32 帧、512 token；
- 历史只调制 action expert。

因此，它既保证任务最重要的首帧信息始终存在，也允许模型利用传送带任务中后续视觉状态的变化。

每次训练输入不会“不断多一张 VLM 图”。VLM prefix 始终是当前三路图像；FrameSamp memory 的物理形状也始终是 512 token，变化的只是有效 mask 和被均匀选中的历史内容。

## 13. 与 RoboMME 原实现的对应关系和差异

核心架构对应关系：

| RoboMME | 本仓库 |
| --- | --- |
| `MemoryBuffer.prepare_frame_sampling` | `prepare_framesamp_memory` |
| `even_sampling_indices` | `even_sampling_indices` |
| `PosEmb3D` | `position_embeddings_3d` |
| `FeatureEncoder.encode_perceptual_memory` | `PerceptualMemory` |
| `MemoryAttention` | `gemma.MemoryAttention` |
| `MemoryRMSNorm` | `gemma.MemoryRMSNorm` |
| `integration_type=modulation` | `use_memory_modulation=True` |

保持一致的关键点：512 budget、单视角、每帧 16 token、均匀采样、3D 位置编码、状态不参与、1024 memory width、每层 FFN 前 modulation、EMA 0.999。

需要明确的工程差异：

1. 本实现只为当前 RoboDojo 单任务数据预存 `cam_high`。
2. 缓存落盘为 FP16；模型训练计算仍按 π0.5 dtype 执行。
3. 训练数据、动作 horizon、学习率和总步数来自 RoboDojo SFT，不是 RoboMME 原论文实验数据和 80k-step schedule。
4. 在线推理 policy 的 episode memory buffer/reset 协议尚未实现；当前完成的是训练链路和模型侧 memory 接口。

因此，本实现复现的是 FrameSamp+Modul 的方法和模型逻辑，不应把不同数据与训练 schedule 下的最终指标理解为对 RoboMME 论文数值的逐点复现。

## 14. 推理侧还缺什么

模型的 `sample_actions` 已能接收准备好的四个 `static_*` 字段，并在所有去噪步中复用同一组 memory token。

真正在线部署还需要一个有 episode 状态的 policy：

1. 每步提取当前 `cam_high` 的冻结 SigLIP 特征；
2. 放入当前 episode 的历史 buffer；
3. 使用相同 FrameSamp 和 3D 位置编码构造 512-token memory；
4. 在 episode reset 时清空 buffer；
5. 保证在线 frame index/FPS 与训练数据语义一致。

详细注意事项见 `docs/framesamp_modul_inference.md`。

## 15. 关键文件索引

```text
src/openpi/models/frame_memory.py
    FrameSamp、空间池化、3D 位置编码、PerceptualMemory

scripts/precompute_robodojo_framesamp_features.py
    离线视频解码和 SigLIP 特征预存

src/openpi/training/framesamp_dataset.py
    mmap 缓存读取和逐样本 memory 构造

src/openpi/models/gemma.py
    MemoryAttention、MemoryRMSNorm、每层 Modul

src/openpi/models/pi0.py
    memory 编码及训练/采样路径接入

src/openpi/models/model.py
    Observation memory 字段和预处理透传

src/openpi/models/pi0_config.py
    FrameSamp 模型参数、输入 spec、视觉冻结规则

src/openpi/training/config.py
    RoboDojo 单任务训练配置和缓存路径

src/openpi/training/data_loader.py
    原数据集与 FrameSampFeatureDataset 的组合

src/openpi/training/weight_loaders.py
    base checkpoint 与新增 memory 参数合并
```

## 16. 已完成的验证

当前实现已完成以下验证：

- FrameSamp 首帧/当前帧保留测试；
- 早期历史右侧 padding 和 mask 测试；
- 4×4 空间池化测试；
- 3D 位置编码测试；
- 缓存 dataset 包装测试；
- memory 字段经过 Repack、AgilexInputs 和 Observation 的透传测试；
- dummy π0.5 FrameSamp+Modul 完整 loss 测试；
- base checkpoint 与新增 memory 参数合并测试；
- 4 卡完整模型 FSDP 初始化和一次前向、反向、AdamW 更新测试；
- 100 个 episode、43,366 帧缓存的形状和有限值检查。

正式集群训练使用当前 8 卡启动脚本和 EMA 0.999。


下面按“一个训练样本从数据集出来以后，如何一路进入 π0.5 action expert”的顺序说明。

整体结构可以概括为：

```text
cam_high 视频
    │
    ├── 离线：π0.5 SigLIP → 256 个视觉 token → 4×4 pooling → 16 token/frame
    │
    └── 训练时：
        episode_index + frame_index
            ↓
        FrameSamp 采样历史帧
            ↓
        固定 512 token memory
            ↓
        3D 时间/空间位置编码
            ↓
        PerceptualMemory
            ↓
        [B, 512, 1024]
            ↓
        action expert 每一层 cross-attention
            ↓
        Modul scale/shift
            ↓
        action prediction
```

---

## 1. 离线预计算 SigLIP 视觉特征

代码位于：

[precompute_robodojo_framesamp_features.py](/pfs/user/kai0-robodojo-frame-mem/scripts/precompute_robodojo_framesamp_features.py:58)

首先加载 π0.5 base 的视觉编码器：

```python
def _load_vision_encoder(params_path: pathlib.Path):
    config = pi0_config.Pi0Config(pi05=True)
    params = _model.restore_params(params_path, dtype=jnp.bfloat16)
    model = config.load(params)
    return nnx_utils.module_jit(model.vision_encode)
```

`vision_encode` 实际调用 π0.5 的 SigLIP：

```python
def vision_encode(self, images):
    image_tokens, _ = self.PaliGemma.img(images, train=False)
    return image_tokens
```

输入一帧：

```text
[224, 224, 3]
```

SigLIP 输出：

```text
[256, 2048]
```

因为 SigLIP 使用 16×16 的视觉 patch 网格：

```text
256 = 16 × 16
```

预处理过程：

```python
images = images / 255.0
images = images * 2.0 - 1.0
images = image_tools.resize_with_pad(images, 224, 224)

image_tokens = encode(images)
pooled_tokens = frame_memory.pool_spatial_tokens(image_tokens, 16)
```

注意：这里的历史视觉特征使用的是 base π0.5 的 SigLIP，并且训练中通过 `freeze_image=True` 冻结视觉编码器。因此离线缓存和训练时视觉编码器是一致的。

---

## 2. 4×4 mean pooling 逻辑

代码位于：

[frame_memory.py](/pfs/user/kai0-robodojo-frame-mem/src/openpi/models/frame_memory.py:29)

```python
def pool_spatial_tokens(tokens: jax.Array, target_tokens: int) -> jax.Array:
    source_tokens = tokens.shape[-2]
    source_side = int(source_tokens**0.5)
    target_side = int(target_tokens**0.5)

    pool_size = source_side // target_side
    leading_shape = tokens.shape[:-2]

    flat = tokens.reshape(
        (-1, source_side, source_side, tokens.shape[-1])
    )

    pooled = nnx.avg_pool(
        flat,
        window_shape=(pool_size, pool_size),
        strides=(pool_size, pool_size),
    )

    return pooled.reshape(
        (*leading_shape, target_tokens, tokens.shape[-1])
    )
```

当前配置为：

```text
source_side = 16
target_side = 4
pool_size = 4
```

因此每个 4×4 输出 token 是原始 4×4 patch token 的平均：

```text
16×16 token grid
    ↓ 每 4×4 区域平均
4×4 token grid
```

输出维度：

```text
[256, 2048] → [16, 2048]
```

这不是 attention pooling，也不是可学习 pooling，而是固定的 mean pooling。

每个 episode 最终保存为：

```text
[episode_length, 16, 2048]
```

当前缓存中：

```text
100 episodes
43,366 frames
每帧 16×2048
dtype=float16
```

---

## 3. 每个训练样本如何取历史帧

代码位于：

[framesamp_dataset.py](/pfs/user/kai0-robodojo-frame-mem/src/openpi/training/framesamp_dataset.py:94)

每个 dataset item 中包含：

```python
episode_index = item["episode_index"]
frame_index = item["frame_index"]
```

然后从对应 episode 的离线缓存中取出：

```python
memory = prepare_framesamp_memory(
    self._episode_embeddings(episode_index),
    frame_index,
    budget=self._budget,
    tokens_per_frame=self._tokens_per_frame,
    position_dim=self._position_dim,
    state_dim=self._state_dim,
)
```

实际核心函数位于：

[frame_memory.py](/pfs/user/kai0-robodojo-frame-mem/src/openpi/models/frame_memory.py:86)

```python
def prepare_framesamp_memory(
    episode_image_embeddings,
    frame_index,
    *,
    budget=512,
    tokens_per_frame=16,
    position_dim=768,
    state_dim=14,
):
    max_frames = budget // tokens_per_frame
    sampled_indices = even_sampling_indices(
        frame_index,
        max_frames,
    )
```

当前参数：

```text
budget = 512
tokens_per_frame = 16
max_frames = 512 / 16 = 32
```

因此每个训练样本最多保留 32 帧历史。

---

## 4. FrameSamp 的均匀采样规则

代码位于：

[frame_memory.py](/pfs/user/kai0-robodojo-frame-mem/src/openpi/models/frame_memory.py:18)

```python
def even_sampling_indices(frame_index: int, max_frames: int):
    if frame_index < max_frames:
        return np.arange(frame_index + 1, dtype=np.int32)

    return np.linspace(
        0,
        frame_index,
        max_frames,
        dtype=np.int32,
    )
```

### 4.1 当前帧小于 32

例如：

```text
frame_index = 10
```

采样结果：

```text
[0, 1, 2, ..., 10]
```

一共 11 帧：

```text
11 × 16 = 176 个有效 token
```

剩余位置右侧 padding：

```text
有效 token: 176
padding:    336
总长度:     512
```

### 4.2 当前帧等于 31

```text
frame_index = 31
```

采样：

```text
[0, 1, 2, ..., 31]
```

刚好 32 帧：

```text
32 × 16 = 512 token
```

没有 padding。

### 4.3 当前帧大于 31

例如：

```text
frame_index = 100
```

采样：

```python
np.linspace(0, 100, 32, dtype=np.int32)
```

得到 32 个均匀分布的 frame index，必然包含：

```text
第一个：0
最后一个：100
```

因此 FrameSamp 具有两个重要性质：

```text
首帧永远保留
当前帧永远保留
```

对于当前 RoboDojo episode，长度大约为 254～598 帧，所以大部分训练样本会采用：

```text
首帧 + 中间均匀历史帧 + 当前帧
```

---

## 5. 固定长度 memory 和 mask

采样后生成固定大小的数组：

```python
valid_tokens = len(sampled_indices) * tokens_per_frame
image_dim = sampled_images.shape[-1]

image_memory = np.zeros(
    (budget, image_dim),
    dtype=np.float32,
)

position_memory = np.zeros(
    (budget, position_dim),
    dtype=np.float32,
)

state_memory = np.zeros(
    (budget, state_dim),
    dtype=np.float32,
)

memory_mask = np.zeros(
    (budget,),
    dtype=np.bool_,
)
```

然后把有效帧放到前面：

```python
image_memory[:valid_tokens] = sampled_images.reshape(
    valid_tokens,
    image_dim,
)

position_memory[:valid_tokens] = sampled_positions.reshape(
    valid_tokens,
    position_dim,
)

memory_mask[:valid_tokens] = True
```

最终输出：

```text
static_image_emb: [512, 2048]
static_pos_emb:   [512, 768]
static_state_emb: [512, 8]
static_mask:      [512]
```

其中：

```text
static_mask=True  → 有效 memory token
static_mask=False → padding token
```

因为 sampled frame 的顺序是从首帧到当前帧，所以：

```text
static_image_emb[0:16]   = 首帧视觉 token
static_image_emb[-16:]   = 当前帧 token
```

在 early frame 阶段，当前帧可能不是最后的物理位置 512，而是有效区域中的最后 16 个 token。

---

## 6. 3D 时间和空间位置编码

代码位于：

[frame_memory.py](/pfs/user/kai0-robodojo-frame-mem/src/openpi/models/frame_memory.py:46)

```python
def position_embeddings_3d(
    frame_indices,
    *,
    spatial_size=4,
    dim=768,
    temporal_base=10_000,
    spatial_base=1_000,
):
```

768 维被拆成：

```text
时间维度：
768 / 6 × 2 = 256

空间 y 维度：
768 / 6 × 2 = 256

空间 x 维度：
768 / 6 × 2 = 256

总计：
256 + 256 + 256 = 768
```

### 6.1 时间位置

```python
width = dim // 6

frequency_power = np.arange(width) / (width - 1)

temporal_omega = 1.0 / (
    temporal_base ** frequency_power
)

temporal_phase = np.einsum(
    "t,d->td",
    frame_indices,
    temporal_omega,
)

temporal = np.concatenate(
    [
        np.sin(temporal_phase),
        np.cos(temporal_phase),
    ],
    axis=-1,
)
```

对于第 0 帧：

```text
frame_index = 0
```

对于当前帧：

```text
frame_index = 当前原始 frame index
```

使用的是原始时间 index，而不是 0～31 的采样序号。因此即使历史被压缩采样，模型仍然知道这些帧在原 episode 中的绝对时间位置。

### 6.2 空间位置

当前是 4×4 token grid：

```python
stride = 16 // spatial_size
centers = stride * np.arange(spatial_size) + stride / 2
```

当：

```text
spatial_size = 4
stride = 4
```

空间中心位置为：

```text
[2, 6, 10, 14]
```

然后分别计算 y 和 x 的 sin/cos：

```python
spatial = np.concatenate(
    [
        np.sin(y_phase),
        np.cos(y_phase),
        np.sin(x_phase),
        np.cos(x_phase),
    ],
    axis=-1,
)
```

最后把时间位置复制到每个空间 token，把空间位置复制到每个时间帧：

```python
temporal = np.repeat(
    temporal[:, None, :],
    spatial_size * spatial_size,
    axis=1,
)

spatial = np.repeat(
    spatial[None, :, :],
    len(frame_indices),
    axis=0,
)
```

最终位置编码形状：

```text
[num_sampled_frames, 16, 768]
```

再 flatten 成：

```text
[num_sampled_frames × 16, 768]
```

---

## 7. PerceptualMemory 如何把输入变成 1024 维

代码位于：

[frame_memory.py](/pfs/user/kai0-robodojo-frame-mem/src/openpi/models/frame_memory.py:138)

初始化：

```python
self.pos_proj = nnx.Linear(
    position_dim,
    position_dim,
    rngs=rngs,
    dtype=dtype,
    kernel_init=_KERNEL_INIT,
)

self.encoder_static = nnx.Linear(
    image_dim + position_dim,
    memory_dim,
    rngs=rngs,
    dtype=dtype,
    kernel_init=_KERNEL_INIT,
)
```

当前维度：

```text
position_dim = 768
image_dim + position_dim = 2048 + 768 = 2816
memory_dim = 1024
```

forward：

```python
def __call__(
    self,
    image_embeddings,
    position_embeddings,
    state_embeddings,
):
    del state_embeddings

    position_features = nnx.silu(
        self.pos_proj(position_embeddings)
    )

    features = jnp.concatenate(
        [
            image_embeddings,
            position_features,
        ],
        axis=-1,
    )

    return self.encoder_static(features)
```

对应的维度变化：

```text
static_image_emb: [B, 512, 2048]
static_pos_emb:   [B, 512, 768]

pos_proj:
[B, 512, 768] → [B, 512, 768]

concat:
[B, 512, 2048 + 768]
= [B, 512, 2816]

encoder_static:
[B, 512, 2816] → [B, 512, 1024]
```

重要的是：

```python
del state_embeddings
```

当前 FrameSamp+Modul 配置并没有使用 memory state embedding。`static_state_emb` 只是为了保持和 RoboMME 数据结构兼容。

---

## 8. Memory 如何进入 π0.5

代码位于：

[pi0.py](/pfs/user/kai0-robodojo-frame-mem/src/openpi/models/pi0.py:153)

```python
def embed_memory(self, obs):
    if not self.use_framesamp_modul:
        raise ValueError(...)

    memory_tokens = self.mem_encoder(
        obs.static_image_emb,
        obs.static_pos_emb,
        obs.static_state_emb,
    )

    return memory_tokens, obs.static_mask
```

输出：

```text
memory_tokens:     [B, 512, 1024]
memory_token_mask: [B, 512]
```

标准 π0.5 的当前观测仍然走原来的 prefix 路径：

```text
cam_high 当前帧
cam_left_wrist 当前帧
cam_right_wrist 当前帧
prompt
state token
```

FrameSamp memory 不会被拼接进 prefix。

训练时：

```python
if self.use_framesamp_modul:
    memory_tokens, memory_token_mask = self.embed_memory(observation)
    mem_seq = [None, memory_tokens]
    mem_mask = [None, memory_token_mask]
```

这里：

```python
mem_seq = [None, memory_tokens]
```

代表：

```text
第一个 expert：PaliGemma language/vision expert，不使用 memory
第二个 expert：action expert，使用 memory
```

所以当前实现只调制 action expert，不调制语言视觉 prefix expert。

---

## 9. Modul 的核心：MemoryAttention

代码位于：

[gemma.py](/pfs/user/kai0-robodojo-frame-mem/src/openpi/models/gemma.py:306)

输入：

```text
x:
[B, T, 1024]
```

这里的 `x` 是 action expert 当前层的 action token。

```text
memory:
[B, 512, 1024]
```

先做维度检查：

```python
if memory.shape[-1] != config.width:
    raise ValueError(...)
```

当前 action expert 配置：

```text
width = 1024
num_heads = 4
num_kv_heads = 1
head_dim = 256
```

### 9.1 Q/K/V 投影

```python
q_einsum = lora.Einsum(
    shape=(config.num_heads, config.width, config.head_dim),
    name="q_einsum_mem",
    ...
)

kv_einsum = lora.Einsum(
    shape=(
        2,
        config.num_kv_heads,
        config.width,
        config.head_dim,
    ),
    name="kv_einsum_mem",
    ...
)
```

对应：

```text
Q:
[B, T, 1024] → [B, T, 4, 256]

K/V:
[B, 512, 1024] → [B, 512, 1, 256]
```

这里使用的是 GQA 形式：

```text
4 个 query heads
1 个 key/value head
```

### 9.2 RMSNorm

```python
rms_norm = MemoryRMSNorm(name="mem_rms_norm")

x = rms_norm(x)
memory = rms_norm(memory)
```

这个 `MemoryRMSNorm` 会先做普通 RMS normalization：

```python
var = jnp.mean(
    jnp.square(x.astype(jnp.float32)),
    axis=-1,
    keepdims=True,
)

normalized = x * jnp.reciprocal(
    jnp.sqrt(var + 1e-6)
)
```

### 9.3 RoPE 位置

memory token 的位置：

```python
k_positions = jnp.arange(memory_length)
```

也就是：

```text
0, 1, 2, ..., 511
```

action query 的位置从 memory 长度之后开始：

```python
q_positions = jnp.arange(
    memory_length,
    memory_length + x.shape[1],
)
```

也就是：

```text
512, 513, 514, ...
```

然后对 Q/K 应用 RoPE：

```python
q = _apply_rope(q, positions=q_positions)
k = _apply_rope(k, positions=k_positions)
```

这里的 RoPE 是 cross-attention 内部的 token 序列位置，不是前面的 3D 图像时空位置编码。两者作用不同：

```text
3D position embedding：
表示原始视频帧的时间和空间位置

RoPE：
表示 memory token 和 action query 在 cross-attention 序列中的位置
```

### 9.4 Cross-attention logits

```python
logits = jnp.einsum(
    "BTKGH,BSKH->BKGTS",
    q,
    k,
    preferred_element_type=jnp.float32,
)
```

逻辑上是：

```text
每个 action token
    查询
512 个历史 memory token
```

logits 形状：

```text
[B, 1, 4, T, 512]
```

### 9.5 Padding mask

```python
masked_logits = jnp.where(
    memory_mask[:, None, None, None, :],
    logits,
    -2.3819763e38,
)
```

padding token 的 logits 被设置为极小值，因此 softmax 后几乎不会获得权重。

这保证了：

```text
early frame 样本中未填充的 memory token 不会参与 attention
```

### 9.6 得到 memory condition

```python
probabilities = jax.nn.softmax(
    masked_logits,
    axis=-1,
).astype(x.dtype)

encoded = jnp.einsum(
    "BKGTS,BSKH->BTKGH",
    probabilities,
    v,
)

encoded = einops.rearrange(
    encoded,
    "B T K G H -> B T (K G) H",
)

output = out_einsum(
    "BTNH,NHD->BTD",
    encoded,
)
```

输出：

```text
memory_condition: [B, T, 1024]
```

它和 action token 的形状一致，但它不是直接加到 action token 上，而是用于后续 modulation。

---

## 10. Modul 的 scale/shift 处理

在 `Block` 中：

[gemma.py](/pfs/user/kai0-robodojo-frame-mem/src/openpi/models/gemma.py:414)

```python
memory_condition, memory_mass = MemoryAttention(
    config=config,
    name="mem_attn",
)(
    x,
    mem_seq[-1],
    mem_mask[-1],
)

x = MemoryRMSNorm(
    name="mem_rms_norm_ffn"
)(
    x,
    memory_condition,
)
```

`MemoryRMSNorm` 的条件分支：

```python
modulation = nn.Dense(
    x.shape[-1] * 2,
    kernel_init=nn.initializers.normal(stddev=0.002),
    dtype=dtype,
)(cond)

scale, shift = jnp.split(
    modulation,
    2,
    axis=-1,
)

return (
    normalized * (1 + scale) + shift
).astype(dtype)
```

当前 action expert width 是 1024，所以：

```text
memory_condition: [B, T, 1024]
Dense:
[B, T, 1024] → [B, T, 2048]

split:
scale: [B, T, 1024]
shift: [B, T, 1024]
```

最终：

```text
x_modulated
=
RMSNorm(x) * (1 + scale)
+
shift
```

这就是 Modul 的核心。

它不是：

```python
x = x + memory
```

也不是：

```python
x = concat([x, memory])
```

而是：

```text
memory → cross-attention → condition
condition → scale/shift
scale/shift → 调整 action token 的特征分布
```

---

## 11. Modul 插入在 Transformer 的什么位置

完整的 `Block` 顺序是：

```python
# 1. 原始 attention
pre_attn = ...
post_attn, kv_cache = attn(...)
xs = gated_residual(xs, post_attn)

# 2. action expert 的 memory modulation
memory_condition, memory_mass = MemoryAttention(...)
x = MemoryRMSNorm(x, memory_condition)

# 3. 原始 FFN 前的 AdaRMSNorm
x, gate = RMSNorm(
    name=_name("pre_ffw_norm", i)
)(
    x,
    adarms_cond[i],
)

# 4. 原始 FFN
x = lora.FeedForward(...)(x)

# 5. FFN residual
xs = gated_residual(xs, out)
```

因此准确位置是：

```text
原始 self-attention residual
    ↓
FrameSamp MemoryAttention
    ↓
MemoryRMSNorm scale/shift
    ↓
π0.5 timestep AdaRMSNorm
    ↓
FFN
```

并且这个 Modul 在 action expert 的每一层都执行。

Gemma action expert 深度为 18，因此一轮完整 forward 中会有 18 次：

```text
action token → memory cross-attention → modulation
```

---

## 12. 训练时的完整 forward

代码位于：

[pi0.py](/pfs/user/kai0-robodojo-frame-mem/src/openpi/models/pi0.py:380)

首先是 flow matching 的 noisy action：

```python
noise = jax.random.normal(noise_rng, actions.shape)

time = (
    jax.random.beta(time_rng, 1.5, 1, batch_shape)
    * 0.999
    + 0.001
)

x_t = (
    time_expanded * noise
    + (1 - time_expanded) * actions
)

u_t = noise - actions
```

然后构造标准 π0.5 prefix 和 suffix：

```python
prefix_tokens, prefix_mask, prefix_ar_mask = \
    self.embed_prefix(observation)

suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = \
    self.embed_suffix(observation, x_t, time)
```

构造 attention mask：

```python
input_mask = jnp.concatenate(
    [prefix_mask, suffix_mask],
    axis=1,
)

ar_mask = jnp.concatenate(
    [prefix_ar_mask, suffix_ar_mask],
    axis=0,
)

attn_mask = make_attn_mask(
    input_mask,
    ar_mask,
)
```

然后计算 memory：

```python
memory_tokens, memory_token_mask = \
    self.embed_memory(observation)

mem_seq = [None, memory_tokens]
mem_mask = [None, memory_token_mask]
```

调用 LLM/action expert：

```python
(prefix_out, suffix_out), _, memory_attention_mass = \
    self.PaliGemma.llm(
        [prefix_tokens, suffix_tokens],
        mask=attn_mask,
        positions=positions,
        adarms_cond=[None, adarms_cond],
        mem_seq=mem_seq,
        mem_mask=mem_mask,
        return_memory_stats=True,
    )
```

最后动作输出：

```python
v_t = self.action_out_proj(
    suffix_out[:, -self.action_horizon :]
)
```

训练目标：

```python
loss_action = _apply_action_loss_mask(
    jnp.mean(
        jnp.square(v_t - u_t),
        axis=-1,
    ),
    loss_mask,
)
```

因此 memory 分支的梯度路径是：

```text
action loss
  ↓
action_out_proj
  ↓
action expert FFN
  ↓
MemoryRMSNorm
  ↓
MemoryAttention
  ↓
PerceptualMemory
  ↓
FrameSamp image/position features
```

视觉特征本身是离线缓存的，所以不会反向更新 SigLIP。

---

## 13. 首帧 attention mass 日志的含义

为了判断模型是否真正使用首帧，`MemoryAttention` 额外统计：

```python
first_frame_tokens = min(16, memory_length)

first_frame_mass = jnp.mean(
    jnp.sum(
        probabilities[..., :first_frame_tokens],
        axis=-1,
    ),
    axis=(1, 2, 3),
).astype(jnp.float32)
```

因为当前每帧 16 个 token，且首帧永远排在 memory 开头：

```text
memory[0:16] = 首帧
```

所以统计的是：

```text
所有 action query
所有 attention head
所有 Transformer layer
对首帧 16 token 的平均 attention mass
```

如果值接近：

```text
16 / 512 = 0.03125
```

说明 attention 接近均匀分布。

如果显著高于：

```text
0.03125
```

说明模型倾向于关注首帧。

不过 attention mass 高不等于一定带来动作收益，因此还需要结合首帧交换 loss。

---

## 14. 首帧交换 loss 的逻辑

训练脚本中会固定一个 monitoring batch，然后交换 batch 内样本的首帧：

```python
swapped_first_frame = jnp.roll(
    observation.static_image_emb[:, :16],
    shift=1,
    axis=0,
)

swapped_memory = observation.static_image_emb.at[
    :, :16
].set(swapped_first_frame)
```

注意这里只交换：

```text
首帧 image feature
```

而没有交换：

```text
首帧位置编码
```

因此首帧的位置仍然是 frame 0，只有首帧内容被替换成了另一个样本的内容。

然后使用相同的随机数分别计算：

```python
correct_loss, _ = model.compute_loss_with_info(
    rng,
    observation,
    actions,
    train=False,
)

swapped_loss, _ = model.compute_loss_with_info(
    rng,
    swapped_observation,
    actions,
    train=False,
)
```

相同 RNG 的作用是保证：

```text
flow-matching noise 相同
timestep 相同
动作 target 相同
```

唯一变化是首帧内容。

指标：

```python
validation_first_frame_swap_loss_delta = \
    swapped_loss - correct_loss
```

解释：

```text
delta > 0：
错误首帧导致 loss 增大，模型可能使用了首帧

delta ≈ 0：
首帧可能没有被使用，或者当前任务不依赖首帧

delta < 0：
首帧交换反而更容易，可能说明模型尚未学到正确对应关系
```

---

## 15. 当前实现中明确没有做的事情

当前机制不是“只保留首帧”。

它实际使用的是：

```text
首帧
+ 均匀历史帧
+ 当前帧
```

当前配置：

```text
每帧 16 token
最多 32 帧
总 memory 512 token
```

它也不是在线递归 memory：

```text
每个训练样本直接根据 episode_index/frame_index
从离线缓存中构造完整历史
```

当前还没有完成标准 policy wrapper 中的在线历史维护。模型级 `sample_actions` 已经支持 memory，但实际部署时还需要：

```text
episode reset
→ 当前 cam_high 编码
→ 加入 history buffer
→ FrameSamp 采样
→ 构造 static_image_emb/static_pos_emb/static_mask
→ 传入 sample_actions
```

---

## 16. 最终张量总结

对于当前训练配置，一个 batch 的主要张量如下：

```text
当前视觉输入：
cam_high             [B, 224, 224, 3]
cam_left_wrist       [B, 224, 224, 3]
cam_right_wrist      [B, 224, 224, 3]

FrameSamp 输入：
static_image_emb     [B, 512, 2048]
static_pos_emb       [B, 512, 768]
static_state_emb     [B, 512, 8]
static_mask          [B, 512]

PerceptualMemory 输出：
memory_tokens        [B, 512, 1024]

action expert：
action_tokens        [B, T, 1024]
Q                    [B, T, 4, 256]
K/V                  [B, 512, 1, 256]
attention logits     [B, 1, 4, T, 512]
memory condition     [B, T, 1024]
scale/shift          [B, T, 1024] each
```

一句话总结：

```text
FrameSamp 决定“从历史中选哪些视觉 token”，
PerceptualMemory 决定“如何把这些 token 映射到 action expert 维度”，
Modul 决定“如何用历史信息改变每个 action token 的特征分布”。
```

目前代码实现的核心路径与 RoboMME 的 FrameSamp+Modul 结构是一致的，差异主要在于 RoboDojo 数据、训练步数、batch size，以及当前尚未完成在线推理 wrapper。
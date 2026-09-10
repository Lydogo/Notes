# Action Chunk 的 Per-Timestep Normalization：原理与 MagicVLA 实现

## 1. 结论先行

这个方案解决的不是“不同 action dimension 量纲不同”这一常规问题，而是另一个被普通 per-dimension normalization 忽略的问题：在 `chunk_delta` 表示下，同一个 action dimension 在 chunk 的不同位置具有不同尺度。

对当前时刻为 $\ell$、chunk 内位置为 $h$ 的动作，MagicVLA 使用：

$$
\Delta a_{\ell,h,d}=a^{\mathrm{abs}}_{\ell+h,d}-q_{\ell,d}
$$

其中 $q_\ell$ 是生成整个 chunk 时的当前机器人状态。较近位置只累积了很短时间的位移，通常变化小；较远位置累积时间更长，分布更宽。如果所有 $h$ 共用 action dimension $d$ 的一组统计量，宽度会主要由 chunk 后部决定：前部信号被压在归一化空间零点附近，后部则更容易触及或超过边界。

因此，统计量应由原来的 `[D]` 扩展为 `[H, D]`：每个 chunk position 和 action dimension 使用自己的 normalizer。这里的 “per-timestep” 特指 **chunk 内相对位置 $h$**，不是环境轨迹的绝对时间 $\ell$，也不是 flow matching 的噪声时间 $t$。

本分支已经实现了这一思路，但有两个重要限定：

1. 论文使用每个 `(h,d)` 的 mean/std；当前 MagicVLA 实现为了兼容已有 bounded quantile normalization，使用每个 `(h,d)` 的 q01/q99。
2. 当前实现是实验开关，默认关闭，仓库现有训练 YAML 尚未启用。训练统计主链路已打通，但部分 open-loop 可视化脚本仍假设统计量是 `[D]`，上线推理前需要同步修正，详见第 9 节。

## 2. 来源与证据边界

方案来源是 2025 BEHAVIOR Challenge 冠军技术报告：

- 论文：[Task adaptation of Vision-Language-Action model: 1st Place Solution for the 2025 BEHAVIOR Challenge](https://arxiv.org/abs/2512.06951)
- 关键章节：[4.5 Delta Action Space with Per-Timestamp Normalization](https://arxiv.org/html/2512.06951#S4.SS5)
- 冠军代码：[behavior-1k-solution](https://github.com/IliaLarchenko/behavior-1k-solution)
- 官方 2025 榜单：[BEHAVIOR Challenge Leaderboard](https://behavior.stanford.edu/challenge/archive/2025/leaderboard.html)

论文定义为：

$$
\mu_{h,d}=\mathbb{E}[\Delta a_{h,d}],\qquad
\sigma_{h,d}=\mathrm{Std}[\Delta a_{h,d}]
$$

$$
\widetilde a_{h,d}=\frac{\Delta a_{h,d}-\mu_{h,d}}{\sigma_{h,d}}
$$

论文明确说明，chunk 前部 delta 小、后部更分散，所以按 chunk position 分开统计；velocity 和 gripper position 不采用这种 per-timestamp normalization。论文没有给这个单项改动的完整消融，因此应把它视为有合理机制和冠军实践支撑的假设，而不是已经在 MagicVLA 上被证明必然提升的结论。

本分支 commit `31d5b32` 给出了本地初步测量：在一个 RoboDojo 数据集、chunk size 50 的设置中，共用 quantile band 时各位置 clipping 从前部约 0.46% 上升到尾部约 4.06%；改为 per-position band 后约为 2.09% 到 1.89%。总 clipping 约 1.83% 与 1.88%，基本不变。这个现象符合 q01/q99 的性质：方案主要是把 clipping 从 chunk 尾部重新均匀分配，而不是消灭 clipping。该数字来自 commit 描述，不等同于下游成功率提升。

## 3. 为什么普通 per-dimension normalization 会压缩 chunk 前部

### 3.1 旧方案

旧统计量只有 action dimension 轴：

$$
L_d=Q_{0.01}(\{\Delta a_{n,h,d}\}_{n,h}),\qquad
U_d=Q_{0.99}(\{\Delta a_{n,h,d}\}_{n,h})
$$

所有 anchor $n$ 和所有 chunk position $h$ 被混在一起。归一化为：

$$
z_{n,h,d}=\mathrm{clip}\left(
2\frac{\Delta a_{n,h,d}-L_d}{U_d-L_d}-1,-1,1
\right)
$$

由于后部 delta 通常更大，$U_d-L_d$ 主要反映后部尺度。前部一个有意义的小位移除以同一大尺度后，会变成很接近 0 的数。

### 3.2 新方案

新统计量保留 chunk position：

$$
L_{h,d}=Q_{0.01}(\{\Delta a_{n,h,d}\}_n),\qquad
U_{h,d}=Q_{0.99}(\{\Delta a_{n,h,d}\}_n)
$$

$$
z_{n,h,d}=\mathrm{clip}\left(
2\frac{\Delta a_{n,h,d}-L_{h,d}}{U_{h,d}-L_{h,d}}-1,-1,1
\right)
$$

例如某个关节的三个位置分别大致分布在 `[-0.01, 0.01]`、`[-0.10, 0.10]`、`[-0.50, 0.50]`。若共用最后这一宽度，前部 `0.008` 只映射到约 `0.016`；若使用自己的 band，它映射到约 `0.8`。不同位置的典型变化因此占据相近的 normalized dynamic range。

### 3.3 对训练目标的实际影响

这不只是输入数值换了尺度。对 quantile band 内部忽略 clamp，有：

$$
\frac{\partial z_{h,d}}{\partial a_{h,d}}=\frac{2}{U_{h,d}-L_{h,d}}
$$

所以 normalized-space MSE 对 raw-space error 的隐式权重与 $1/(U_{h,d}-L_{h,d})^2$ 成正比。per-timestep normalization 会提高模型对前部小幅但立即执行动作的分辨率，同时避免只因尾部尺度大就让其主导整个 dimension 的标度。

对 flow matching 来说，目标 action chunk 各位置进入更相近的数值尺度，固定尺度的噪声和 velocity loss 不再面对“前部接近零、尾部接近边界”的强烈异方差问题。但它不会自动建模时间相关性，也不会替代 loss mask、temporal correlation 或执行时的 receding horizon 策略。

## 4. 什么时候成立，什么时候不该用

这个方法依赖一个关键前提：action 表示是相对同一 anchor state 的累计位移，即 `chunk_delta`。

- `chunk_delta`：`a[l+h] - q[l]`。尺度通常随 $h$ 增长，适合按位置统计。
- `frame_delta`：`a[l+h] - a[l+h-1]`。每一行多为单步变化，没有同样的累计增长机制，本实现不允许启用。
- `abs`：绝对 action 本身不会因位于 chunk 后部就天然变大，强行分片只会减少每个统计 cell 的样本量，本实现不允许启用。
- gripper 或其他 absolute dimension：没有累计 delta 语义，应继续共用全 horizon 的 band。
- velocity dimension：若每个位置表示局部速度而非累计位移，同样不应仅因 chunk position 分开统计；论文也将 velocity 排除。

对于 rot6d，当前数据路径没有直接做向量减法，而是通过 `compose_relative_rot6d` 计算相对旋转。只要整组 rotation dimension 被配置为相对量，它在语义上属于 delta dims；部分维度相对、部分维度绝对会被拒绝。

## 5. 当前分支的端到端实现

### 5.1 开关进入 Dataset

配置入口是每个 LeRobot v2.1 source 的 `params.action_stats_per_horizon`：

```yaml
data:
  datasets:
    - type: lerobot
      modality: robotics
      root: /path/to/lerobot_v21
      lerobot_version: v2.1
      params:
        chunk_size: 50
        action_representation: chunk_delta
        action_delta_indices: [0, 1, 2, 3, 4, 5]
        action_stats_per_horizon: true
        norm_stats_path: /path/to/norm_per_horizon.json
```

`src/data/backends/lerobot.py` 将这个参数传给 `LeRobotV21Dataset`。构造函数立即验证它只能与 `action_representation: chunk_delta` 同时使用。

注意：应按每个 source 单独计算统计量。不同 embodiment、控制频率、action 选择、插值方式和 quality filter 的分布不同，不应把它们无条件合成一个 normalizer。

### 5.2 统计数据必须经过真实训练变换

统计不是直接读取 parquet 里的绝对 action，而是对采样 anchor 调用 `_build_action_chunk()`。因此统计值已经经过和训练样本一致的：

1. action dimension 选择与右侧 padding；
2. action chunk 构造与 episode-end padding；
3. quality mask 和 dimension mask；
4. semantic temporal upsampling / interpolation；
5. `chunk_delta` 转换；
6. rot6d 相对旋转组合；
7. gripper 从 delta indices 中排除。

这是实现正确性的核心。若在原始 absolute action 上先算统计，再在训练时转换 delta，统计分布与模型实际看到的变量将不是同一个变量。

### 5.3 两条统计生产路径

仓库支持两条路径，二者应得到相同结果：

**Dataset 启动时估计**

`LeRobotV21Dataset._estimate_action_stats_per_horizon()` 均匀采样最多 `action_stats_samples` 个 eligible anchors，默认 10,000。它暂存：

- `chunks`: `[N, H, D]` float32；
- `masks`: `[N, H, D]` bool。

然后每个 `(h,d)` 只从 mask 有效的 anchor 中计算 `count/mean/std/min/max/q01/q99`。默认 `N=10000, H=50, D=32` 时，注释估计主 float buffer 约 64 MB，不解码视频。

**离线 artifact 脚本（推荐生产使用）**

`scripts/data/compute_lerobot_v21_norm.py` 使用 `MaskedStreamingStats((H,D))` 流式计算 moments，并用确定性抽样的 anchor 收集 quantile。默认：

- mean/std/min/max：最多 100,000 个均匀 anchor；
- q01/q99：其中最多 10,000 个均匀 anchor；
- 不读取视频；
- 输出 JSON stats artifact，并可同时输出 anchors / episodes sidecar cache。

推荐直接读取训练 YAML，保证所有 source 参数完全一致：

```bash
conda run -n magicvla python scripts/data/compute_lerobot_v21_norm.py \
  --config configs/train/your_train_config.yaml \
  --dataset-index 0 \
  --source-index 0 \
  --output /path/to/norm_per_horizon.json
```

如果 `dataset-index` 指向的不是 dataset group，则去掉 `--source-index`。直接 root 模式可用：

```bash
conda run -n magicvla python scripts/data/compute_lerobot_v21_norm.py \
  --root /path/to/lerobot_v21 \
  --chunk-size 50 \
  --action-representation chunk_delta \
  --action-stats-per-horizon \
  --params-json /path/to/all_other_transform_params.json \
  --output /path/to/norm_per_horizon.json
```

复杂数据源优先使用 config 模式，避免遗漏 delta indices、rotation、upsampling、mask 或 quality rules。

### 5.4 安全回退（carve-outs）

把 `[D]` 拆成 `[H,D]` 会减少每个 cell 的样本，并可能人为制造极窄 band。`apply_horizon_carveouts()` 同时计算一份 pooled full-horizon `[D]` 统计作为 fallback：

1. **非 delta dimensions**：将 pooled `[D]` 的统计沿 H 复制。这使 gripper 等绝对量仍然只有一套有效 normalizer，只是物理存储形状保持 `[H,D]`。
2. **退化 delta slice**：若某个 `(h,d)` 的 q01-q99 宽度小于该 dimension pooled 宽度的 1%，就用 pooled q01/q99 替换。绝对下限为 `1e-6`。
3. **零标准差**：std 至少 clamp 到 `1e-6`，避免 artifact validator 拒绝或除零。
4. **无有效样本的 cell**：初始化为 neutral stats，例如 q01/q99 为 `[-1,1]`；正常训练中对应 mask 会阻止它贡献 loss。

artifact 的 sampling metadata 会记录：被平铺的绝对维度、被放宽的 slice 数、阈值，以及第一行和最后一行 band 宽度比，便于监控数据漂移。

### 5.5 Artifact 防错与 checkpoint 元数据

统计布局被写入 `norm_stats_signature()`：

- 旧布局：`per_dim`，数组形状 `[D]`；
- 新布局：`per_horizon_dim`，数组形状 `[H,D]`。

加载 artifact 时既比较完整 signature，又逐项检查精确 shape。这样旧 `[D]` artifact 不会因广播而被新训练静默接受，chunk size 改变也会明确报错。

训练 checkpoint 的 normalization metadata 同样写入 `action_stats_layout`、chunk size、action representation、delta indices 和统计值，使独立 serving 端能够恢复同一份 normalization contract。

### 5.6 训练时的广播

`magicvla_base` processor 中 `_normalize()` 接收 action `[H,D]` 和 q01/q99 `[H,D]`，逐元素执行 quantile normalization。state 统计仍是 `[D_state]`，按普通广播工作。对二维 band，代码显式要求 band shape 与 action shape 完全相等，防止 chunk size 不一致。

`magicvla_vqa` processor 的数学运算同样能对 `[H,D]` 正确逐元素广播，但目前没有 `magicvla_base` 的显式 shape guard。

## 6. 训练与推理必须是一对可逆变换

训练：

$$
z_{h,d}=2\frac{a_{h,d}-L_{h,d}}{U_{h,d}-L_{h,d}}-1
$$

推理 inverse normalization：

$$
\widehat a_{h,d}=\frac{\widehat z_{h,d}+1}{2}
(U_{h,d}-L_{h,d})+L_{h,d}
$$

随后，只有 delta dimensions 需要加回当前 anchor state：

$$
\widehat a^{\mathrm{abs}}_{\ell+h,d}
=\widehat{\Delta a}_{\ell,h,d}+q_{\ell,d}
$$

rot6d relative rotation 应使用对应的旋转组合逆过程，不能简单相加；gripper 等 absolute dimensions 不加 anchor state。

这里最容易出现的线上错误是：训练使用 `[H,D]`，serving 却把 stats reshape 为 `[H*D]` 或仍取一维 `[D]`。这种错误可能直接 shape failure，也可能在某些广播条件下静默产生错误动作。因此 serving 端应检查：

- `action_stats_layout == per_horizon_dim`；
- stats shape 精确等于 checkpoint 的 `(chunk_size, action_dim)`；
- 当前 source/embodiment 与 stats 对应；
- inverse normalization 在恢复 delta 之前完成；
- 恢复 delta 使用生成该 chunk 时的同一个 anchor state。

## 7. 与论文原方案的差异

| 项目 | 冠军论文 | 当前 MagicVLA 分支 |
|---|---|---|
| action 表示 | `chunk_delta` | `chunk_delta` |
| stats 轴 | `[H,D]` | `[H,D]` |
| normalization | mean/std | q01/q99 映射到 `[-1,1]` 并 clamp |
| 排除项 | velocity、gripper | 非 delta dimensions（包含配置排除的 gripper）平铺 pooled band |
| 退化保护 | 论文段落未详述 | 小于 pooled band 1% 的 quantile slice 回退 |
| 兼容控制 | 未在论文详述 | layout 写入 signature，严格检查 shape |
| 默认状态 | 冠军方案的一部分 | feature flag，默认关闭 |

保持 quantile 的主要原因是 MagicVLA 当前模型、processor、bounded output 和既有统计资产围绕 `[-1,1]` 建立。它不是论文 mean/std 的逐字复现，而是保留核心的 `(h,d)` 统计轴后做的系统适配。

另一个影响是时间平滑性：论文指出 FAST tokenizer 仍使用 global quantile normalization，因为 per-timestamp normalization 会让相邻位置应用不同 affine transform，从而破坏 normalized sequence 的时间平滑性，不利于 DCT 压缩。MagicVLA 若同时训练 FAST 辅助目标，应单独核查 FAST 的 normalization contract，不能默认连续 action expert 和 FAST 必须共享 `[H,D]` stats。

## 8. 建议的落地步骤与 A/B 验证

### 8.1 启用步骤

1. 只选择 `action_representation: chunk_delta` 的 source。
2. 核对 `action_delta_indices`，确保 gripper、velocity、padding 和 absolute commands 不在其中。
3. 在目标训练 YAML 的每个适用 source 中加入 `action_stats_per_horizon: true`。
4. 用同一 YAML 离线重算 norm artifact；不要复用旧 `[D]` artifact。
5. 检查脚本输出的 `tiled_absolute_dims`、`narrow_slices_widened`、`span_ratio_first_to_last`。
6. 修正并测试所有 serving/eval inverse-normalization 消费者。
7. 从同一初始化 checkpoint 做 flat 与 per-horizon A/B；模型配置、数据、seed、训练预算和 inference 参数保持一致。

### 8.2 至少应记录的离线指标

按 `(h,d)` 和按 h 汇总：

- raw q01/q99 span；
- normalized occupancy，例如 p01-p99 覆盖 `[-1,1]` 的比例；
- clipping rate；
- normalized MSE 与 inverse 后 raw-unit MSE；
- 第一行/最后一行 span ratio；
- fallback slice 数量和有效样本 count；
- 各 action dimension 的误差，而不只看全局平均。

重点不是追求总 clipping 降到零。q01/q99 本来就会产生约 2% 的尾部 clipping。更关键的是尾部 clipping 是否不再随 h 系统性上升，以及前部有意义的变化是否不再挤在零附近。

### 8.3 在线指标

- task success / q-score；
- chunk 第一个实际执行动作的 error 和 jitter；
- chunk 尾部预测 error（即使 receding horizon 不一定执行到尾部）；
- 不同执行长度或 action compression 下的稳定性；
- gripper event precision/recall；
- 不同 embodiment/source 是否出现收益相反的情况。

建议同时做三个 sanity baseline：flat `[D]`、完整 per-horizon `[H,D]`、只对明确的累计 joint/pose delta dims 做 per-horizon。第三个通常最能隔离 gripper/velocity 混入造成的副作用。

## 9. 当前实现的已知边界与修改建议

### 9.1 现有 YAML 未启用

仓库搜索只在代码、脚本和测试中发现 `action_stats_per_horizon`，没有训练配置实际设置为 true。也就是说，本分支提供了能力，但尚未形成可直接启动的实验配置和对应 stats artifact。

### 9.2 两个 open-loop inverse normalization 仍按 `[D]` 编写

`scripts/infer/eval_openloop_video.py::_denormalize()` 对 stats 调用 `.reshape(-1)`；二维 `[H,D]` 会变成 `[H*D]`，无法和 `[N,H,D]` action 正确广播。

`scripts/eval_openloop_magicvla_vqa.py` 把 prediction reshape 成 `[-1,D]` 后与 stats 运算；`[H,D]` stats 也无法按 anchor 正确对应。它们应保留 chunk 轴，例如统一要求 prediction 为 `[...,H,D]`，并直接和 `[H,D]` stats 运算，而不是 flatten stats 或先抹掉 H。

这不影响 dataset processor 的训练归一化，但会影响这些脚本的 raw-unit 指标和可视化。正式实验前应先补测试：随机 `[B,H,D]` raw chunk 做 normalize → denormalize round trip，验证最大误差在浮点容差内。

### 9.3 `mean_std` 模式的退化保护不完整

非 delta dimensions 会把 mean/std 一起平铺，所以是安全的；但对退化的 delta slice，当前 `apply_horizon_carveouts()` 只回退 q01/q99，mean/std 仍可能只是 clamp 到 `1e-6`。如果模型 normalization 配置改为 `mean_std`，这会放大很小的噪声。

因此当前实现应视为主要针对仓库常用的 `normalization: quantile`。若要复现论文的 mean/std，应增加基于 pooled std 的阈值，并同时回退该 slice 的 mean/std；还应给 `mean_std` processor 加上与 quantile 路径一致的 `[H,D]` shape 检查。

### 9.4 VQA processor 缺少显式 shape guard

`magicvla_vqa` 的 `_normalize()` 对正确 `[H,D]` 可正常广播，但没有明确拒绝错误 chunk size。建议复用 `magicvla_base` 的 rank/shape 检查，避免某些尺寸碰巧可广播时静默使用错误 band。

### 9.5 统计样本量随 H×D 分片

每个 cell 的有效样本从大约 `N×H` 降为 `N`，episode 尾部 padding、quality mask 和稀有 dimension 会进一步降低 count。应在 artifact 验收中设置最低 count 告警，而不仅依赖 1% span fallback。对低频 gripper event，保持 pooled band 尤其重要。

## 10. 测试覆盖与本次验证

`tests/test_horizon_normalizer.py` 覆盖了 13 个场景，包括：

- flat `[D]` 与 per-horizon `[H,D]` shape；
- band 随位置增宽；
- absolute dimension 使用 pooled band；
- 窄 slice fallback；
- layout signature 和错误 shape 拒绝；
- 仅允许 `chunk_delta`；
- reader 在线估计与离线脚本逐元素一致；
- artifact 写入/读取 round trip；
- 实际 normalize 输出范围。

本次在 `magicvla` Conda 环境执行：

```bash
conda run -n magicvla python tests/test_horizon_normalizer.py
```

环境中 `torch`、`numpy` 可用，但 `pyarrow` 和 `pytest` 不可用，所以测试文件的依赖检查跳过了全部 13 项，用例没有实际运行。补齐依赖后建议执行：

```bash
conda run -n magicvla python -m pytest -q tests/test_horizon_normalizer.py
```

另外使用当前 `magicvla_base._normalize()` 做了一个不依赖 pyarrow 的 `[H=3,D=2]` 数值 smoke test。三个位置使用不同 q01/q99 band，输出 shape 保持 `(3,2)`；normalize 后按同一 `[H,D]` band inverse，最大 round-trip error 为 `2.98e-08`。这验证了 processor 的逐位置广播和基本可逆性，但不能替代上述 dataset/artifact 集成测试。

在未获得真实测试通过结果前，不能把 commit 描述中的 “322 passed” 当成本机当前环境的验证结论。

## 11. 最终判断

这个方向与问题机制是匹配的：`chunk_delta` 的不同 horizon 位置确实不是同分布变量，把它们强行共用 `[D]` normalizer 会浪费前部动态范围并把尾部推向 clipping。将 stats 扩为 `[H,D]` 是比额外手工 loss weight 更直接、也更容易保持 train/inference 对称的处理。

当前分支的主体设计是合理的，尤其是：统计走真实 action 构造路径、absolute dims carve-out、窄 band fallback、artifact signature/shape 防错以及离线/在线统计一致性测试。不过它仍是“可实验”的实现而不是完整生产闭环：需要新增实际启用配置和 artifact，修复 inverse-normalization 脚本，补齐 mean/std 情况的保护，并在真实训练 A/B 中验证 task 指标。最稳妥的第一步是只对明确的累计 delta dimensions 启用 quantile per-horizon normalization。

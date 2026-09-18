# MagicVLA Memory分支实现

本次于2026-09-17检查原仓库的本地Git分支快照。远端fetch因TLS连接失败未完成，因此下表表示本地已有版本，不保证是远端最新代码。检查使用git show，未切换或修改代码分支；未运行模型训练或闭环评测。

## 1. 分支与机制

| 分支 | commit | 实现 | 历史进入哪里 |
|---|---|---|---|
| magicvla-mem | 032faf9 | 早期causal short-memory | Qwen视觉塔 |
| magicvla-mem-v0 | eb4a3e2 | 带零初始化gate的时间差分 | Qwen视觉塔 |
| magicvla-mem-v1 | c1a40d3 | 无新增参数的time-then-space | Qwen视觉塔 |
| feat/history | 82d85bd | DM05式历史prefix，支持Base和SGM | VLM prefix |
| model/magicvla_streamer | 25c7bc6 | Streamer及Streamer-WS | 观测动作序列、专家递归状态或共享主干 |

main中的普通Base入口只保留当前图像，不能据此判断整个仓库没有Memory。feat/opt_memory与feat/opt_memory_bk主要处理优化器和checkpoint内存占用，不能仅按分支名归为策略记忆。

## 2. mem-v0与mem-v1：在视觉塔压缩短历史

共同输入为6帧、约1秒间隔的三相机图像。每个相机独立做同空间patch的时间融合；history_valid屏蔽无效历史，history_dt提供真实时间差。processor只取当前state，两版均未输入历史proprio token。

| 对比 | v0 | v1 |
|---|---|---|
| 时间融合 | 原空间路径加gated temporal delta | temporal attention混合V，再做spatial attention |
| 融合层，按第几层计 | 4、8、12、16 | 4、8、12、16、20 |
| 删除历史帧 | 第16层后 | 第20层后 |
| 新增参数 | 64个per-head gate | 0，复用原norm/QKV/projection/MLP |
| 视觉塔 | 配方冻结原参数 | 配方解冻原参数 |
| 初始化 | gate为0，多帧初始不影响当前输出 | 多帧立即参与融合；K=1时退化为单图路径 |

v1的时间编码先加到hidden，再经过norm和QKV投影，因此会影响Q、K和V；当前帧的时间编码为零。时间attention允许读取自身和有效过去帧。最终只保留当前帧送入merger，语言侧视觉token数不增加，视觉计算量仍增加。

v1配方开启20%的history slot dropout和25%的current-only随机化，同一样本的相机共享mask，eval时关闭。KI关闭、FAST关闭，由flow loss训练视觉塔、VLM和Action Expert。零新增参数不等于零初始化分支，也不代表K>1时与单帧checkpoint输出一致。

### 配置快照变化

本地c1a40d3的v1 YAML字段为：每rank batch=64、workers=12、prefetch=2、sample-level mixing、pin_memory=false、80,000步，基础LR=2e-5，VLM倍率0.1。旧详细记录中的48/2/1是另一配置记录，不应当作此commit的字段值；资源参数变化也不代表Memory公式改变。

实现入口：[v1 short_memory.py](https://gitlab.magiclab.top/vla/magicvla/-/blob/c1a40d32dd32eadd1e74cdfacf24dfa7cbe27c69/src/models/magicvla_base/short_memory.py)。张量与公式详解见[完整记录](../99_详细记录/01_MagicVLA_视觉记忆_完整记录.md)，其中旧配方数值按历史快照阅读。

## 3. feat/history：DM05式稀疏历史prefix

### 3.1 历史怎样进入模型

history recipe v2默认取cam_high的20张严格过去帧，1 Hz采样；当前三路图像仍单独输入。每张过去帧经过共享Qwen视觉塔，在merger后将网格平均池化为4×4，即16个token。

```text
过去cam_high → Qwen vision → 每帧4×4 pooling → 最多320个历史token
                                                        ↓
历史说明与占位符 → 当前三路图像 → 任务/本体/当前state → VLM prefix
                                                        ↓
                                              Action Expert / SGM
```

20张历史帧不包含当前帧。这里固定的是历史上限，实际prefix会随有效历史数量增长；它与mem-v1“只输出当前帧视觉token”不同。recipe v2按时间顺序排列，不逐帧写时间戳。

### 3.2 为什么不简单padding再mask

缺失历史不写入prefix，模型只编码有效帧。原因是Qwen混合主干含GatedDeltaNet：把零token插在真实token中间，即使Full Attention屏蔽它，也可能推进递归状态。实现会检查真实token之间不能出现padding。

history_frames=0直接调用Base构造逻辑，作为current-only对照。recipe v2明确拒绝history_state=true和旧版mask_padded_history设置；仓库保留的早期state历史代码，不代表当前recipe支持proprio memory。

### 3.3 已有部署buffer

HistoryFrameRing按环境编号维护deque，按oldest-first返回过去图像。RoboDojo server已接入：先读取历史构造输入，再记录当前观测，并在reset时清空。

ring本身不看时间戳，每次replan追加一帧，因此replan间隔必须与训练采样间隔匹配。例如25 FPS、1 Hz历史对应每25步重规划；不能在不同控制频率下仍把每次追加视为1秒。

这里已有观测缓存，但没有学习型写入、关键帧选择、文本摘要或跨episode记忆。buffer负责提供历史，模型每次显式消费窗口。

### 3.4 与SGM组合

MagicVLASGMHistoryPolicy将HistoryPrefixMixin与MagicVLASGMPolicy组合，沿用SGM主干。历史放入共享prefix后，Action Expert与可读取prefix的2D/3D World Expert都能使用它。

这说明Memory与Dynamic已有代码级组合，并非仅为设想；但类和配方存在不等于闭环收益已验证。history recipe版本与32D/34D动作版本分别管理，不能互相推导。

代码入口：[HistoryPrefixMixin](https://gitlab.magiclab.top/vla/magicvla/-/blob/82d85bd8570b7d69394b64f1d62ead8835079af7/src/models/magicvla_base_history/modeling_magicvla_base_history.py)、[HistoryFrameRing](https://gitlab.magiclab.top/vla/magicvla/-/blob/82d85bd8570b7d69394b64f1d62ead8835079af7/src/models/magicvla_base_history/history_ring.py)、[SGM组合](https://gitlab.magiclab.top/vla/magicvla/-/blob/82d85bd8570b7d69394b64f1d62ead8835079af7/src/models/magicvla_sgm_history/modeling_magicvla_sgm_history.py)。

## 4. Streamer：观测与动作历史

### 4.1 magicvla_streamer

该模型以连续unit组织观测和动作，有两条历史路径：

- VLM序列追加观测与过去动作的FAST token；预测当前unit时只读取之前已提交的动作记录。
- Action Expert的GatedDeltaNet跨unit传递递归矩阵和卷积尾部。去噪过程读取旧状态，随后用clean action前向更新状态。

训练使用GT动作写入历史，并对历史FAST token做dropout。推理有StreamState和stream_step接口，但需注意注释与执行逻辑的区别：stream_step直接用本次生成的x更新专家状态并编码FAST，再返回动作；该接口没有等待机器人实际执行后的反馈。

因此，如果部署端裁剪、混合或中断动作，需同步修正提交历史，不能直接称为“始终记录实际执行动作”。专家递归状态大小固定，也不代表总内存固定：观测、state和FAST前缀仍会累积，当前实现重算VLM。

### 4.2 magicvla_streamer_ws

同分支还有WS变体：去掉独立Action Expert和FAST，以共享主干处理连续动作。clean序列组织为图像、state和连续动作latent，当前带噪块读取过去clean上下文。

history_units开启时，输入需要显式提供HISTORY_STATES、HISTORY_ACTIONS和HISTORY_VALID等历史字段。joint_block实验还联合预测未来视觉/state变化；这些预测目标用于训练和去噪，不直接作为已发生的历史提交。

该路线同时建模观测与动作上下文，不能归为mem-v1的视觉融合，也不能称为TTT fast-weight更新。实现入口：[Streamer](https://gitlab.magiclab.top/vla/magicvla/-/blob/25c7bc6529c69a6e9ba64c0997c47275ae2a9ecc/src/models/magicvla_streamer/modeling_magicvla_streamer.py)、[Streamer-WS](https://gitlab.magiclab.top/vla/magicvla/-/blob/25c7bc6529c69a6e9ba64c0997c47275ae2a9ecc/src/models/magicvla_streamer_ws/modeling_magicvla_streamer_ws.py)。

## 5. 如何理解这些实现

| 路线 | 记忆内容 | 主要成本与边界 |
|---|---|---|
| mem-v0/v1 | 短时视觉变化 | 前段视觉计算增加，语言视觉token固定 |
| Base/SGM history | 稀疏过去图像 | 视觉编码和prefix增加，已有部署ring |
| Streamer | 观测、动作记录与专家递归状态 | 历史提交需和执行对齐，VLM重算成本随历史增加 |
| Streamer-WS | 显式观测、state、连续动作历史 | 共享主干逐块计算，训练和推理需提供一致历史 |

已有代码证明这些机制可被配置和调用；是否改善任务，还需要对应checkpoint、闭环成功率以及正确/打乱/屏蔽历史对照。本次不补写未核验的训练结果。

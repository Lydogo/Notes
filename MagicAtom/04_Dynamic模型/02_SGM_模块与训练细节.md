# SGM 模块与训练细节

> 先看 [模型整体原理](01_MagicVLA_Dynamic_SGM.md)。本文按前向数据流拆解模块，尺寸以常见的hidden=1024、chunk=50配置为例；动作维度D可分别为32或34。

## 1. 图像、语言和状态怎样进入 VLM

图像经过与训练一致的缩放补边和Qwen processor，再进入视觉塔、merger和语言骨干。不同相机的可用性单独记录，缺失图像不能被当作真实的黑色场景。

连续state有两条路径：

| 路径 | 编码方式 | 用途 |
|---|---|---|
| VLM prefix | `[state, state_dim_mask]` 拼接后经MLP，形成状态token | 让视觉语言上下文理解当前机器人状态和缺失维度 |
| Action Expert | state经独立MLP，广播加到各动作位置 | 给动作生成提供直接的当前状态条件 |

两套MLP参数独立。归一化后的零可能是正常中间值，所以VLM状态token显式携带mask；不能仅通过数值是否为零判断状态是否存在。

## 2. Action Expert 输入与输出

输入张量为 `x_t [B,H,D]`、`state [B,D]`、`t [B]`：

```text
h_action = Linear(x_t) + action_position
         + broadcast(time_embedding(t) + state_MLP(state))

h_action → 混合主干 → RMSNorm → Linear(hidden,D) → velocity[B,H,D]
```

action position是chunk中的位置；flow time表示当前噪声阶段。两者都影响动作特征，但不能互换。这里的状态/时间条件采用相加方式注入，不应统称为AdaLN。

未来50个动作位置一次生成，网络参数在位置间共享。它们通过动作位置embedding、attention和当前带噪数值区分各自角色。

基础：[Flow Matching](../../Note_Basics.md#basic-flow)、[RoPE](../../Note_Basics.md#basic-rope)。

## 3. 混合主干：DeltaNet 与 full attention

每个专家层可概括为：

```text
h = h + Mixer(RMSNorm(h))
h = h + SwiGLU(RMSNorm(h))
```

Mixer随层型切换。DeltaNet层在各专家内部处理序列，full-attention层按可见关系读取其他分支；多个专家结构相似，不表示共享全部权重。

### DeltaNet 的作用

当前实现先对Q/K/V投影做短因果卷积，再使用门控delta rule更新状态。可把它理解为：用已有状态预测当前value，再根据预测残差写入修正；衰减门控制旧信息保留多少，写入门控制新信息更新多少。

它通过矩阵状态压缩序列信息，避免显式保存完整的两两attention矩阵。这里的递推是层内序列计算，不能直接理解为跨控制周期持久保存的机器人记忆。

DeltaNet是因果计算；动作chunk在full-attention层允许双向读取。因而“动作位置能相互协同”不意味着每种层都采用双向attention。

### Full attention 的作用

各分支先使用自己的投影形成Q/K/V，再在兼容的head维度上组织交互。例如8个Q heads、2个K/V heads的GQA中，每组K/V被多个Q heads使用。hidden=1024不要求Q总宽度也为1024；输出投影负责映回分支hidden size。

RMSNorm控制特征尺度，SwiGLU做逐token的非线性通道变换，残差连接保留输入信息。它们与attention分工不同。

基础：[Attention](../../Note_Basics.md#basic-attention)、[RMSNorm](../../Note_Basics.md#basic-rmsnorm)、[SwiGLU](../../Note_Basics.md#basic-swiglu)。

## 4. 2D queries 与预测头

单视角16×16网格对应256个queries：

```text
query = learned_token + row_embedding + column_embedding
        + camera_embedding（启用时）
```

行列embedding标明空间位置，相机embedding区分视角。多个batch样本共享初始query参数，但读取不同观测后得到不同特征。

经过主干后，`RMSNorm → LayerNorm → Linear` 将专家特征投影到DINOv3的目标宽度。输出按视角与patch对齐，而不是把整幅图像压成一个监督向量。

## 5. 3D queries、双预测头与目标整理

3D分支也使用learned queries加行列embedding。共享主干输出分别进入geometry和motion两个MLP：

- geometry head：LayerNorm、Linear、GELU、Linear。
- motion head：LayerNorm、Linear、SiLU、Linear。

两个head输出相同宽度，不表示目标语义相同。前者匹配几何latent，后者匹配运动latent。

Teacher特征需先与query网格对齐，当前代码采用两种不同处理：

| 目标 | 原始形状 | 整理方式 | 典型结果 |
|---|---|---|---|
| Geometry | `[B,324,1024]`，即18×18网格 | area resize到16×16 | `[B,256,1024]` |
| Motion | `[B,256,32,32]`，256是通道数 | 将每个2×2空间块拼入通道 | `[B,256,1024]` |

Motion的处理保留了局部块内的特征，不是简单平均池化。只让两个张量shape一样还不够，必须保持patch排列与目标语义对应。

## 6. 三类 loss 分别比较什么

| Loss | 比较方式 | 主要含义 |
|---|---|---|
| Flow | 有效动作元素上的velocity MSE | 学习从噪声恢复动作的向量场 |
| 2D | 有效patch上的 `1 - cosine(pred,target)` | 侧重特征方向一致性 |
| 3D | prediction/target分别按最后一维LayerNorm，再算MSE | 减少特征整体尺度与偏置的影响 |

3D loss分别计算geometry与motion，再加权组合。Teacher target停止梯度；3D误差以fp32计算。分母使用有效元素或token数，避免无效区域越多，loss就看起来越小。

这些loss量纲与范围不同，不能仅比较数值大小判断哪个任务更难。辅助loss下降也不能直接等同于动作成功率提高。

基础：[Masked MSE](../../Note_Basics.md#basic-masked-mse)。

## 7. Mask 的五个层次

| Mask | 控制范围 | 典型用途 |
|---|---|---|
| state dimension | 输入状态维度 | 区分缺失测量与真实零值 |
| action dimension / time | 动作元素与chunk位置 | 限定noise、loss和生成中的有效动作 |
| view / world validity | 相机或World目标 | 排除缺失视角、无效teacher监督 |
| prefix validity | VLM token | 排除padding，控制有效条件范围 |
| stream visibility | 专家间的信息边 | 指定Action/2D/3D能读取谁 |

维度无效、时间补齐、目标失败和分支屏蔽是不同原因，不能用一个总mask代替。尤其是episode尾部的重复动作，有的配方将其作为“保持终点”监督，有的将其排除，不能统一认定为错误padding。

基础：[Masked Attention](../../Note_Basics.md#basic-masked-attention)。

## 8. FAST、梯度隔离与 teacher 生命周期

FAST将动作编码成离散token，用语言CE给VLM提供辅助监督；连续控制仍由Action Expert生成。训练时，专家读取的prefix必须排除FAST答案token。

KI同时涉及条件特征与条件K/V投影的梯度路径，不能只看某一处 `detach()` 就判断隔离是否完整。检查时分别观察Action、2D、3D loss能否更新VLM；FAST/语言CE是否更新VLM则是另一条路径。

DINOv3与Track4World始终作为冻结teacher。模型可以在部署时关闭teacher构造，但保留World Experts；省去teacher权重不等于省去所有world计算。

## 9. 缓存与数值稳定性

固定观测下，VLM prefix不读取专家输出，所以每个交互层的VLM K/V可缓存。每轮flow更新仍从当前带噪动作构建action hidden，并重新运行专家主干。缓存初始World queries不等于缓存最终World输出。

DeltaNet的递推与求解路径对数值精度敏感。代码在关键计算中显式使用fp32，并控制autocast；仅将输入转换为float32，不保证其后的矩阵乘法不会被autocast降精度。

排查顺序建议为：输入/目标是否有限 → mask是否正确 → loss分项 → 梯度路径 → 混合精度与执行后端。不要仅凭总loss正常就认为全部分支工作正常。

## 10. 源码导航

| 内容 | 入口 |
|---|---|
| Action embedding、RMSNorm、SwiGLU、DeltaNet | [base 模型](../../../magicvla/src/models/magicvla_base/modeling_magicvla_base.py) |
| Queries、投影头、loss、joint trunks、缓存 | [SGM 模型](../../../magicvla/src/models/magicvla_sgm/modeling_magicvla_sgm.py) |
| Teacher target与网格整理 | [world_teachers](../../../magicvla/src/models/magicvla_sgm/world_teachers.py) |
| 配置与可见关系 | [SGM 配置](../../../magicvla/src/models/magicvla_sgm/configuration_magicvla_sgm.py) |

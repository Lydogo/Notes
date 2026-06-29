# T-Rex：用触觉高频闭环补上 dexterous manipulation 的最后一段反应能力

> 原标题：T-Rex: Tactile-Reactive Dexterous Manipulation  
> 作者：Dantong Niu, Zhuoyang Liu, Zekai Wang, Boning Shao, Zhao-Heng Yin, Anirudh Pai, Yuvan Sharma, Stefano Saravalle, Ruijie Zheng, Jing Wang, Ryan Punamiya, Mengda Xu, Yuqi Xie, Yunfan Jiang, Letian Fu, Konstantinos Kallidromitis, Matteo Gioia, Junyi Zhang, Jiaxin Ge, Haiwen Feng, Fabio Galasso, Wei Zhan, David M. Chan, Yutong Bai, Roei Herzig, Jiahui Lei, Fei-Fei Li, Ken Goldberg, Jitendra Malik, Pieter Abbeel, Yuke Zhu, Danfei Xu, Jim Fan, Trevor Darrell  
> 机构：UC Berkeley, NVIDIA, Stanford, Panasonic, La Sapienza University, ItalAI  
> 发表：arXiv:2606.17055v1，2026-06-15；本地 PDF 为 v1，arXiv 页面已有 v2  
> 链接：https://arxiv.org/abs/2606.17055  
> 项目：https://tactile-rex.github.io/  
> 开源：论文称 T-Rex Dataset open-source；项目页为主要入口  

---

## 一、研究背景与动机

当前 VLA 模型大多依赖视觉和语言，已经能做不少桌面操作，但在真正接触丰富的 dexterous manipulation 上仍容易失败。典型任务包括插卡、开锁、挤牙膏、翻页、拆牌、拧灯泡等：视觉能告诉机器人“物体在哪里”，但不能稳定告诉它接触力、微滑移、局部变形和当前是否卡住。

T-Rex 的出发点是：人类灵巧操作的关键不是只看见，而是能在接触发生后以高频触觉反馈快速修正动作。现有 tactile policy 的问题有三类：数据稀缺、VLA 架构频率太低、触觉 encoder 多停留在静态或简单融合。论文希望用一个中训练（mid-training）阶段，让已有大规模 human egocentric pretraining 的 VLA 获得 tactile-reactive 能力，而不是从零收集海量 visuo-tactile 数据。

## 二、核心贡献

1. **提出 T-Rex Dataset**：100 小时 tactile-synchronized bimanual dexterous manipulation 数据，覆盖 200+ 日常物体、22 个 motor primitives，并包含 RGB、触觉信号、机器人状态、动作和语言指令。

2. **提出 T-Rex Model**：Mixture-of-Transformer-Experts（MoT）架构，包括 latent expert、action expert 和 tactile expert，把低频 visuomotor planning 与高频 tactile refinement 解耦。

3. **提出 asynchronous tactile-reactive cascaded flow matching**：action expert 先低频去噪到中间 timestep，tactile expert 复用缓存的视觉语言上下文，在更高频率下根据实时触觉完成剩余去噪，实现触觉反应式闭环。

4. **构建 12 个真实接触丰富任务 benchmark**：覆盖 force control、deformable object manipulation、bimanual coordination 和 force-deformation reasoning；T-Rex 平均成功率 65%，比最强 baseline EgoScale 的 35% 高 30 个百分点。

## 三、方法原理

### 3.1 整体框架

T-Rex policy 输入 RGB 观测、语言指令、触觉力历史和触觉 deformation map，输出未来 action chunk。硬件平台是 Dexmate Vega-1 双臂机器人，配两只 22-DoF Sharpa Wave dexterous hands；视觉包括头部 ZED 相机和两个腕部相机；每只手有五个 fingertip tactile sensors。

模型有三个 expert：

| Expert | 作用 | 运行频率/特点 |
|---|---|---|
| Latent expert | 预测未来视觉 latent | 提供 temporally grounded visual context |
| Action expert | 低频 action denoising | 约 5 Hz，生成基础 dexterous manipulation plan |
| Tactile expert | 高频 tactile refinement | 约 20 Hz，利用实时触觉修正动作 |

核心机制是 cascaded flow matching：action expert 负责从纯噪声去噪到 `tau_split`，tactile expert 接手从 `tau_split` 去噪到干净动作。这样视觉语言大模型不需要每个高频控制 tick 都重跑，触觉 expert 可以复用 cached context。

### 3.2 关键技术细节

**Spatial-temporal tactile encoder** 同时编码两类触觉信息：

| 输入 | 编码方式 | 目的 |
|---|---|---|
| 最近 16 帧 per-finger 6D force history | VQ-VAE temporal force encoder | 捕捉接触动态、减轻传感器漂移，用离散 token 表示 force pattern |
| 当前 force vector | 直接 projection | 保留瞬时接触强度 |
| 当前 deformation map | 轻量 CNN / ResNet-style encoder | 捕捉边缘、滑移、剪切和局部形变 |

**异步高频 refinement** 是论文最有价值的设计。推理时总共 10 个 Euler steps，`tau_split = 0.4`。action expert 低频执行 6 步，生成中间状态和 KV cache；tactile expert 在一个 action chunk 内的 `{0, 4, 8, 12}` offset 高频触发，每次用最新触觉 token 和缓存上下文完成 4 步 terminal denoising，并更新要执行的 action chunk。

这个设计直接对应现实机器人控制中的频率错配：视觉/语言规划慢，触觉反馈快。如果把触觉简单拼进 VLA backbone，高频响应会被大模型推理成本拖住；如果完全分离，又难保留 VLA 的语义和视觉泛化。T-Rex 在两者之间做了一个工程上很聪明的折中。

### 3.3 训练与优化

T-Rex 使用三阶段训练：

| 阶段 | 数据 | 训练目标 |
|---|---|---|
| Human egocentric pretraining | 22,889 小时 egocentric human video | 训练 latent/action experts，获得语义 grounding 和粗粒度 visuomotor prior |
| Tactile grounded robot mid-training | 100 小时 T-Rex teleoperation 数据 | 对齐机器人可执行动作与触觉接触动态，训练 tactile expert |
| Skill-specific post-training | 每个下游任务约 100 条 demo | 适配复杂或任务特定技能 |

训练目标包括 action expert flow loss、tactile expert flow loss 和 future visual latent prediction loss：

```text
L = L_act + lambda_tac L_tac + lambda_future L_future
```

其中 `lambda_tac = 1.0`，`lambda_future = 0.5`。action expert 和 tactile expert 回归同一个 velocity target，但作用在不同的 flow timestep 区间上。

## 四、实验与结果

### 4.1 实验设置

所有真实实验都在 Dexmate Vega-1 + Sharpa Wave dexterous hands 上完成。动作空间为双臂相对 end-effector delta control 加手指 absolute joint control，action dimension 为 62，action chunk 为 16。每个任务评测 16 次，物体位置和旋转随机，报告按任务平均的 success rate。

对比方法包括 ViTacFormer、Reactive Diffusion Policy（RDP）、Tactile-VLA、EgoScale、π0.5、π0.5 + tactile。所有方法使用相同机器人 setup、action space 和 evaluation protocol。

12 个任务包括 Flip Page、Transfer Egg、Wipe Plate、Apply Paste、Split Cup、Sort Mahjong、Open Lock、Refill Tablet、Acid-Base Neutralization、Extract Card、Deal Poker、Screw Bulb，覆盖精细力控、形变感知、插拔/抽取、双手协调等情况。

### 4.2 主要结果

**T-Rex 在 12 个接触丰富任务上显著超过 baseline。**

| 方法 | Flip Page | Transfer Egg | Wipe Plate | Apply Paste | Split Cup | Sort Mahjong | Open Lock | Refill Tablet | Acid-Base Neut. | Extract Card | Deal Poker | Screw Bulb | Avg. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ViTacFormer | 9 | 0 | 4 | 1 | 4 | 7 | 0 | 0 | 0 | 2 | 2 | 1 | 3 |
| RDP | 12 | 8 | 18 | 2 | 6 | 9 | 2 | 0 | 0 | 1 | 2 | 7 | 6 |
| Tactile-VLA | 38 | 14 | 24 | 0 | 21 | 27 | 8 | 0 | 9 | 4 | 11 | 18 | 15 |
| EgoScale | 68 | 44 | 34 | 38 | 33 | 36 | 19 | 12 | 43 | 41 | 28 | 18 | 35 |
| π0.5 | 36 | 17 | 28 | 13 | 18 | 32 | 5 | 1 | 24 | 8 | 9 | 11 | 17 |
| π0.5 + tactile | 8 | 9 | 27 | 2 | 4 | 14 | 2 | 0 | 7 | 3 | 0 | 0 | 6 |
| T-Rex | 96 | 75 | 69 | 66 | 78 | 65 | 47 | 41 | 76 | 70 | 57 | 35 | 65 |

最值得注意的是，直接给 π0.5 拼 tactile 反而从 17% 掉到 6%。这说明触觉不是“多加一个模态就行”，而需要合适的编码、时序和控制频率设计。

**触觉模态和动态编码是关键。**

| 配置 | Flip Page | Apply Toothpaste | Split Cup | Open Lock | Extract Card | Screw Lightbulb | Avg. |
|---|---:|---:|---:|---:|---:|---:|---:|
| Full Model | 96 | 66 | 78 | 47 | 70 | 35 | 65 |
| w/o Tactile | 76 | 39 | 58 | 23 | 34 | 20 | 42 |
| MLP Force + Deform | 89 | 58 | 72 | 44 | 58 | 29 | 58 |
| Deform only | 82 | 57 | 71 | 36 | 55 | 25 | 54 |
| MLP Force + VQVAE Force | 92 | 63 | 65 | 38 | 67 | 28 | 59 |
| w/o Async | 92 | 61 | 73 | 45 | 59 | 30 | 60 |

去掉触觉平均下降 23 个百分点；不用 asynchronous refinement 下降 5 个百分点。说明最大增益来自触觉本身，其次来自动态 force encoding 与异步高频控制。

### 4.3 消融实验

| 消融点 | 结果 | 含义 |
|---|---|---|
| w/o tactile | Avg. 65 -> 42 | 接触丰富任务必须依赖触觉反馈 |
| tactile representation | full model 优于 force/deform 简化版本 | force history、instant force、deformation map 互补 |
| w/o async | Avg. 65 -> 60 | 高频 tactile expert 的异步执行带来稳定增益 |
| tau split | 中间 split 最好 | split 太小则 action prior 不足，太大则 tactile expert 修正空间不足 |
| T-Rex dataset vs task-specific data | T-Rex motor primitive 数据泛化更好 | 数据应覆盖组合式基础接触行为，而不只是任务 demo |
| post-training demo 数量 | mid-training 后低数据区间更强 | 100 小时 tactile mid-training 提高 data efficiency |
| 三阶段 recipe | full recipe 平均 65，去掉阶段明显下降 | human pretraining 提供语义/粗动作先验，tactile mid-training 对齐真实接触控制 |

训练 recipe 的六任务平均结果从 18、34、45 逐步到 full model 的 65，说明两个阶段都不是装饰：human egocentric pretraining 解决 broad prior，tactile robot mid-training 解决接触动力学。

## 五、局限性与展望

作者指出两个主要限制。第一，长时程、高精度接触协调任务仍很难，尤其当 teleoperation 本身困难时，未来可能需要 RL 或在线交互式 refinement。第二，tactile-reactive manipulation 受硬件限制很强，包括传感器形变、设备间 calibration drift，以及缺少 palm-level dense sensing。

我的补充判断是：T-Rex 的实验很强，但平台和传感器依赖也很强。它证明了 tactile mid-training + high-frequency expert 的价值，但跨不同 tactile sensor、不同 dexterous hand、不同接触材料的迁移还没有被充分证明。另一个问题是 benchmark 全是真实机器人，这很有价值，但也意味着可复现实验成本高。

## 六、灵魂三问

1. **它解决了什么问题？**

它解决的是 dexterous manipulation 中视觉 VLA 对接触状态反应慢、不知道力和形变的问题。相比只靠视觉的 VLA 或简单 tactile fusion，T-Rex 明确把 tactile feedback 放到高频闭环 refinement 里，让机器人能在接触后快速修正动作。

2. **为什么这么做？**

因为视觉语言规划和触觉反应天然频率不同。T-Rex 用 action expert 做低频基础计划，用 tactile expert 复用缓存上下文做高频末端去噪，既保留大规模 VLA/egocentric pretraining 的泛化能力，又避免每次触觉更新都重跑沉重视觉 backbone。

3. **什么证据最有说服力？**

最有说服力的是 12 个真实接触任务的主表和触觉消融：T-Rex 平均 65%，最强 baseline EgoScale 35%；去掉触觉后降到 42%，直接给 π0.5 加 tactile 只有 6%。这同时证明了触觉重要，也证明了触觉需要正确的架构和训练方式。

## 七、个人总结

1. T-Rex 的核心思想是把 tactile-reactive control 从“多模态输入”提升为“独立高频控制路径”：触觉不是给 VLA 看一眼，而是在执行过程中持续修正动作。

2. 最大优势是真实接触任务上的效果非常硬，尤其是 force/deformation-sensitive skill；最大弱点是高度依赖触觉硬件和 teleoperation 数据，跨硬件泛化仍是问号。

3. 对后续 VLA/机器人研究来说，T-Rex 提醒我们：视觉语言模型可以负责语义和粗计划，但接触丰富操作最终需要高频、局部、物理信号驱动的控制层。它和 Qwen-VLA 这类统一大模型路线是互补的，而不是替代关系。

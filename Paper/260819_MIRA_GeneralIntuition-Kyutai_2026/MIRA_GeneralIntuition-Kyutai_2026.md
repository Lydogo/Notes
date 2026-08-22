# MIRA：让四个玩家共同受动作控制的实时交互世界模型

> 原标题：Multiplayer Interactive World Models with Representation Autoencoders  
> 作者：Anthony Hu 等 27 人  
> 机构：General Intuition；Kyutai；Epic Games；École nationale des ponts et chaussées  
> 发表：arXiv:2607.05352v2，2026-07-07 修订；Technical Report  
> 链接：https://arxiv.org/abs/2607.05352  
> 项目页/演示：https://mira-wm.com/  
> 开源代码：https://github.com/mira-wm/mira  
> 数据集：https://huggingface.co/datasets/kyutai/rocket-science  
> 本地来源：`Paper/2607.05352v2.pdf`

---

## 一、研究背景与动机

世界模型的核心不是生成“看起来像真的”视频，而是学习在动作干预下环境如何演化。已有交互式视频/世界模型大多把其他智能体当作环境的一部分，只显式接受一个玩家的动作。这会带来三个问题：模型不知道场景变化应归因于哪个玩家；多玩家动作组合容易互相纠缠；不同视角之间可能出现物体位置和事件不一致。

MIRA（Multiplayer Interactive World Models with Representation Autoencoders）把问题放到 2v2 Rocket League 中：输入四个玩家的同步第一视角、四条动作流和短历史，预测四个视角未来如何共同变化。论文的目标是同时做到三件事：动作可控、跨视角物理一致、长时自回归稳定，并在实时速度下生成。

选择 Rocket League 的好处是有快速而紧耦合的物理交互、精确动作日志和 privileged physics state；代价是它仍是单一游戏、单一 bot 家族和近乎确定性的环境，距离真实机器人世界还有明显差距。

## 二、核心贡献

1. **提出多玩家交互世界模型**：联合建模四个玩家的视角与动作流，而不是只控制一个玩家，把其他玩家的行为从“环境噪声”变成显式条件。

2. **提出基于预训练表示的 Representation Autoencoder latent**：用冻结 DINOv3-L 特征构造时空压缩 latent，在 10 Hz latent 空间预测，再解码为 20 fps 视频。

3. **提出面向长时稳定和实时性的训练/推理组合**：diffusion forcing 缩小 teacher-forcing 与自回归推理的分布差异，progressive self-distillation 将 flow matching 采样压缩到少数步，再结合 KV cache 和滚动上下文实现实时生成。

4. **建立不只看画质的评估体系并开源**：除 gFID/gFVD/gFDD 外，使用物理状态 probe、Action Recoverability Ratio（ARR）、跨视角一致性和人工评测，发布约 10,000 match-hours 数据、代码与在线 demo。

## 三、方法原理

### 3.1 整体框架

每个玩家 (p) 有一条视频流和一条键盘动作流。共享 codec encoder 将每个视角编码成 latent 序列；四个 latent 沿高度方向 tiled 成一个联合网格。世界模型在联合 latent 上根据过去窗口和四名玩家动作预测下一 latent，最后由共享 codec decoder 分别还原四个玩家视角。

```text
四路同步视频 + 四路动作
        ↓
共享 DINOv3-L 表示 codec：20 fps / 1280×720 → 10 Hz latent
        ↓
四路 latent 沿高度拼接
        ↓
diffusion-forcing latent DiT + 四路动作条件
        ↓ 逐 latent 自回归，KV cache，20-step context
四路解码视频：20 fps
```

训练时一次处理短 clip 的多个 latent frame；推理时每次只生成一个 latent frame，并通过 decoder 一次还原两帧视频。因此模型的训练目标与在线 rollout 形式不同，但二者都在 latent 空间工作。

### 3.2 表示 codec：预测空间比像素空间更重要

冻结的 DINOv3-L/16 逐帧提取中间层特征。作者将若干中间 block 的特征平均，再加上最深层特征，使 latent 同时保留空间细节和语义信息。一个 learned linear bottleneck 将 1024 通道压到 32 通道，并使用 (2\times2\times2) patchifier：

- DINOv3 原始 patch：16×16 像素；额外空间下采样 2× 后，一个 latent token 覆盖 32×32 像素。
- 时间下采样 2×：20 fps 视频变为 10 Hz latent。
- latent 是 deterministic continuous representation，没有 KL、latent noise 或 GAN loss。
- decoder 先把 latent 空间上采样回 16×16 feature grid，再用空间双向、时间因果的 ViT 解码；最后时间上采样 2×回 20 fps。

codec loss 为 L1 重建、LPIPS 和 P-DINO feature consistency 的加权和。两项 perceptual loss 使用梯度范数自适应平衡，避免原始 loss 尺度主导训练。

关键判断是：重建最清晰的 latent 不一定最适合世界模型生成。from-scratch feature extractor 的 PSNR/SSIM 更好，但 latent 更难预测、长时漂移更严重；冻结预训练 DINO latent 的生成质量和稳定性更好。

### 3.3 Latent flow matching、diffusion forcing 与实时推理

对目标 latent (z_1) 和高斯噪声 (z_0)，构造

```text
zτ = (1 − τ) z0 + τ z1
```

网络预测从噪声流向数据的 velocity (z_1-z_0)。与普通 teacher forcing 不同，diffusion forcing 为每个时间 frame 独立采样 flow time：同一训练 clip 中同时存在干净、轻噪和重噪 frame。这样模型训练时已经见过不完美历史，减少 rollout 中误差逐步积累。

为降低采样成本，模型通过 progressive self-distillation 学习用一个大步近似两个半步的平均 velocity。论文报告 PSD 在少步 regime 明显优于未蒸馏 baseline，使每个 latent frame 只需少数 flow steps。

在线推理的关键设置：

- latent 自回归窗口 (T=20)，即约 2 秒历史；每步滚动一格。
- Transformer 使用 streaming KV cache，避免重复计算历史 token。
- decoder 只依赖最近 3 个 latent frame，保持常数级解码成本。
- 初始由真实短 clip warm-up cache，之后全部由模型生成。
- 5B 模型在单张 Nvidia B200 上端到端约 70 ms/latent step，每步产生两帧，约 35 ms/frame，达到 20 fps。

### 3.4 多玩家条件建模

四路 latent 在高度方向 tiled，空间 attention 横跨所有视角，因此一个视角中的球、车辆或 HUD 可以和其他视角中的对应实体共同建模。每个玩家的九键动作先经过独立 embedding，再按固定玩家顺序拼接，送入 AdaLN；同一个条件向量广播到所有空间位置。

动作不是直接告诉模型“哪种动作影响哪个像素”，而是让模型从联合视角和时序中学习归因关系。训练中随机把某玩家动作替换成 learned absent token，使模型可以在推理时让未控制玩家继续表现出合理行为。

训练采用两阶段：先用单玩家视角和单条动作流预训练，再 warm-start 到四玩家联合训练。固定计算预算下，多玩家从零训练会崩溃；单玩家预训练后再继续多玩家训练能恢复稳定性。预算更大时，从零训练可行，但 warm-start 仍有收益。

### 3.5 数据使用与维度追踪

| 数据源 | 规模 | 样本单位 | 模态/字段 | 关键维度 | 用途阶段 | 处理方式 |
|---|---:|---|---|---|---|---|
| Rocket League 2v2 bot matches | 约 10,000 match-hours；82,983 matches | match；训练窗口约 4 s | 四路 RGB、四路键盘动作、events、physics state | RGB 1280×720，20 fps；四路；动作 9 维 multi-hot | codec、single-player WM、multiplayer WM | 原始视频 30 fps→20 fps；动作/视频按 frame 对齐；坏 match 丢弃 |
| privileged physics state | 同上，120 Hz | 每个 physics tick | 球、四辆车、比分、时钟、事件 | 位置/速度/姿态/boost/接触状态等；完整 schema 见论文附录 | 仅评估，不输入 WM | 原始 120 Hz 锁步记录；不作为训练输入 |

数据由同一个 Nexto self-play bot 的四个独立实例生成，分布在 Champions Field、Forbidden Temple、Deadeye Canyon 三张地图。原始信号为视频 30 fps、physics 120 Hz、bot action 15 Hz；训练前统一成 20 fps 视频和逐帧动作。动作映射为九个键：`W A S D Q E Space LShift LControl`，每帧是 9 维 multi-hot；数据集 loader 中若下采样时间窗口，会对窗口内动作 OR，保持 frame alignment。

**维度快照**

- Observation：`frames (P,T,C,H,W)`，P=4，RGB，1280×720，20 fps；论文实验中的单视角 codec 输入也使用 288×512 crop/resize 设置。
- Language：无语言条件。
- State：physics state 只作 ground truth；probe 最终读出球和四辆车的 50 个位置、四元数和线速度量。
- Action：`actions (P,T,9)`，九键 multi-hot；latent 每一步覆盖两帧动作，先 pooling 成 latent-rate action embedding。
- Latent：空间一个 token 覆盖 32×32 像素，通道 32，10 Hz；四玩家 latent 沿高度 tiled。
- Prediction target：下一 latent frame；训练 clip 中每一帧独立 flow time，推理中单 latent 自回归。
- Context：world model 固定 20 latent frames；decoder 使用最近 3 个 latent frames。

数据没有语言、机器人 proprioception、连续末端位姿或显式 action chunk。它更接近“多视角视频 + 离散控制 + 物理状态评估”的交互模拟器，而不是 VLA 的 instruction-conditioned action policy。

## 四、实验与结果

### 4.1 实验设置

生成质量使用 gFID、gFVD、gFDD；codec 重建使用 PSNR、SSIM、LPIPS、P-DINO 及对应 reconstruction Fréchet distances。世界模型默认在 4 s horizon 报告生成指标，并考察最长 5 min 的 drift。动作可控性用 ARR：生成视频中动作 probe 的 AP 除以 codec reconstruction 中的 AP，ARR=1 表示达到重建上限。

物理 probe 在真实 encoded latent 上训练，测试时读取 generated rollout 的激活并预测球/车辆状态；这比直接看视频画质更能检验模型是否保留动力学信息。

### 4.2 主要结果

| 对比 | gFID ↓ | gFVD ↓ | gFDD ↓ | ARR ↑ |
|---|---:|---:|---:|---:|
| Latent（MIRA） | 10.7 | 163.1 | 0.55 | 0.91 |
| Pixel-space plain | 104.9 | 1456.3 | 16.19 | 0.61 |
| Pixel-space JiT recipe | 81.0 | 961.2 | 17.05 | 0.49 |

latent 相比 pixel-space 不是小幅改进，而是生成质量约一个数量级的差距；像素模型还会在约一秒内变成扭曲纹理。说明实时交互世界模型首先需要一个适合动力学预测的表示空间。

diffusion forcing 对 teacher forcing 也有明显优势：4 s horizon 的 gFID 为 10.7 vs 32.5，gFVD 为 163.1 vs 944.1；5 min rollout 中 teacher forcing 明显漂移，而 diffusion forcing 基本保持平稳。

模型规模从 100M、300M、1B、2.5B 增至 5B 时，FID/FVD 单调改善、收敛更快；球位置 probe 误差从 100M 的 2130 降到 5B 的 1448，但 2.5B 到 5B 已出现收益递减。训练数据方面，在固定 100k steps 下，约 50 小时以下的 unique data 会导致画质和 ARR 一起崩溃；数据继续增加后 gFID 先饱和，但 ARR 仍继续上升，说明动作忠实度比表观质量更吃数据。

多玩家结果的核心不是单个 headline 数字，而是行为：四个视角中的球、车辆、goal explosion、demolition 和 HUD 事件保持互相一致；单玩家模型则更容易忘记离开视野的车辆，甚至把车辆和球混在一起。动作 dropout 还使未被控制的玩家能继续按训练到的行为策略移动。

### 4.3 消融实验

- **预训练表示**：冻结 DINOv3-L 的 codec 生成 gFID 10.7；from-scratch feature extractor 为 22.5，且 5 min drift 更严重。后者重建 PSNR 反而为 32.2，高于 MIRA 的 29.7，说明重建指标不能替代 latent 可生成性指标。
- **时间压缩**：20 fps latent 改为 10 Hz 后，生成质量基本不变，却把世界模型序列长度减半，是实时速度的重要工程收益。
- **感知损失**：LPIPS 与 P-DINO 互补；去掉二者会导致生成崩溃，二者同时使用后不需要 GAN loss。
- **教师强制 vs diffusion forcing**：diffusion forcing 同时解决短期生成质量和长 horizon drift。
- **多玩家训练配比**：多玩家从零训练在小预算下崩溃；先 single-player warm-start 再 multiplayer continuation 最稳健。
- **可控性与频率**：ARR 与人工 action-adherence preference 高度相关（Pearson 0.84，Spearman 0.93）；罕见的 reverse、air-roll 动作 ARR 更低，动作覆盖率直接影响可控性。

## 五、局限性与展望

作者明确承认：

1. 只有一个游戏、三张地图和一个 Nexto bot 家族，行为多样性有限；模型学到的是 bot 风格，不是完整的人类策略分布。
2. 只有 20 latent frames 的上下文，导致 clock、score、off-screen vehicle 等需要长期记忆的状态会漂移或遗忘。
3. 稀有事件欠采样：静止的球会被模型错误地推动，kickoff 中也可能在没有按键时产生 boost/jump；goal replay 这种脚本过渡会失真。
4. Rocket League 近乎确定且状态可由动作完全决定，难度低于真实世界中的部分可观测、多 embodiment、多传感器场景；没有语言、触觉、proprioception 或真实机器人闭环验证。

从机器人/VLA 角度看，最值得延伸的是：把“多视角共享 latent + 多主体动作归因 + physics probe”迁移到多机器人协作、手-物交互和人机协作场景；同时需要更长记忆、稀有事件重采样、真实人类/机器人行为混合，以及显式不确定性建模。

## 六、灵魂三问

### 1. 它解决了什么问题？

它解决的是单玩家世界模型无法正确处理多主体因果归因的问题：四个玩家的动作会同时改变共享场景，模型必须知道谁撞了球、谁被 demolition、各个视角应如何看到同一个事件。MIRA 将四条动作流和四路视角联合预测，使世界模型从单人可控视频生成器变成多主体交互模拟器。

### 2. 为什么这么做？

核心选择是把预测放到冻结 DINOv3 构造的低维 latent，而不是像素空间；再用 diffusion forcing 训练模型面对不完美历史，用 distillation/KV cache 把采样成本压到实时范围。多玩家视角 tiled 到同一空间 attention 网格，则为共享状态提供直接的跨视角约束。实验显示这些选择分别对应画质/可控性、长时稳定性、实时速度和视角一致性的瓶颈。

### 3. 什么证据最有说服力？

最有说服力的是三组互相补强的结果：latent 对 pixel-space 的 gFID 10.7 vs 81–104.9；diffusion forcing 在 5 分钟 rollout 中没有 teacher forcing 那样的漂移；四路模型能在不同摄像机中一致生成 goal/demolition，并在 ARR、physics probe 和人工评测上体现动作与物理状态，而不只是画面更清晰。

## 七、个人总结

1. **一句话总结**：MIRA 的关键不是 5B 参数，而是找到一个“易生成、可跨视角对齐、能接受多路动作”的 latent dynamics 空间，并围绕长时 rollout 做训练和系统优化。
2. **最大优点/最大弱点**：优点是把视觉质量、动作遵循、物理表示和实时交互放在同一套评估中；弱点是数据分布高度单一，20-frame memory 和 bot-only 行为限制了真实世界外推。
3. **对 VLA/机器人研究的启示**：未来机器人 world model 的数据单元不应只保存单视角 `(image, action)`，还应尽量保存同步多视角、多主体动作、事件和可用于离线 probe 的 privileged state；但这些 state 应用于评估/表示学习验证时要与模型实际输入严格分开。

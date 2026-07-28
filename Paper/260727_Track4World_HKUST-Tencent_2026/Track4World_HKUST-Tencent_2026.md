# Track4World：从单目视频恢复世界坐标系下的全像素 3D 轨迹

> 原标题：Track4World: Feedforward World-centric Dense 3D Tracking of All Pixels  
> 作者：Jiahao Lu, Jiayi Xu, Wenbo Hu, Ruijie Zhu, Chengfeng Zhao, Sai-Kit Yeung, Ying Shan, Yuan Liu  
> 机构：The Hong Kong University of Science and Technology；ARC Lab, Tencent PCG  
> 发表：arXiv:2603.02573v2，2026-03-05 修订；GitHub README 标注 ECCV 2026  
> 链接：https://arxiv.org/abs/2603.02573  
> 项目页：https://jiah-cloud.github.io/Track4World.github.io/  
> 开源：https://github.com/TencentARC/Track4World  
> 本地来源：`Paper/2603.02573v2.pdf`

---

## 一、研究背景与动机

理解视频里的 4D dynamics 需要知道每个像素在三维空间里怎么运动。对机器人来说，这类能力对应的是更底层的时空几何：物体移动、相机自运动、手-物接触、遮挡后重现、动态目标轨迹。如果视觉系统只能给出 2D optical flow 或稀疏点轨迹，机器人策略很难直接获得稳定的 3D motion prior。

已有方法有两个主要限制：

1. **稀疏点跟踪不够完整**：SpatialTracker、SpatialTrackerV2、DELTA 等能做 3D tracking，但通常只跟踪第一帧上的 sparse points，无法覆盖后续新出现的像素和物体。
2. **dense 3D tracking 太慢或太重**：TrackingWorld 这类 optimization-based dense pipeline 可以覆盖全帧，但依赖 2D flow、mask、depth priors 等多模块融合，计算成本高，也不容易端到端学习统一时空先验。

Track4World 的目标是做一个 feedforward 模型：输入单目视频，输出全像素、世界坐标系下的 3D tracking。它不是直接预测每个像素完整长轨迹，而是先估计任意帧对之间的 dense 2D/3D scene flow，再把 pairwise flow 融合成全局轨迹。

## 二、核心贡献

1. **提出 feedforward world-centric dense 3D tracking 框架**：基于 VGGT-style ViT 的全局 3D scene representation，从单目视频恢复每个像素在世界坐标系下的 3D 轨迹。

2. **设计 2D-to-3D correlation scene flow decoder**：用图像平面的 2D correlation 引导 3D flow 更新，避免传统 3D kNN search + cross-attention 的高计算成本。

3. **采用 sparse-to-dense 和 arbitrary-pair flow 设计**：先在 1/8 分辨率 anchor points 上迭代更新，再学习上采样到 dense flow；同时支持任意帧对，不局限相邻帧。

4. **用 2D-3D joint supervision 缓解 3D 标注稀缺**：模型同时预测 2D flow 和 3D scene flow，因此可以借助大量 2D optical flow / point tracking 数据提升 3D motion generalization。

## 三、方法原理

### 3.1 整体框架

Track4World 输入一段单目视频：

```text
{I_i | i = 1 ... T}
```

输出是每个像素在世界坐标系中的 3D trajectory。整体流程分四步：

```text
视频帧
  -> VGGT-style 3D geometry encoder
  -> global scene representations: geometry features / point clouds / camera poses
  -> sparse-to-dense 2D-3D scene flow decoder
  -> pairwise flows fusion
  -> world-centric dense 3D tracking
```

这里的关键选择是：**不直接回归所有像素的全时序轨迹**。那样会有大量冗余，训练和推理都很重。作者把问题分解成任意帧对的 scene flow 估计：

```text
source frame i -> target frame j 的 2D flow + 3D scene flow
```

得到 pairwise flow 后，再根据需求构造长程 tracking。比如要跟踪第一帧到所有后续帧，就估计 reference-to-target flows；要全像素全帧 dense tracking，就把相邻帧 flow chain 起来。

### 3.2 Sparse-to-Dense Scene Flow Decoder

直接在全分辨率每个像素上做 iterative correlation 太贵。Track4World 先把 point cloud 和 feature map 下采样到 1/8 分辨率，在 sparse anchor points 上做迭代，再通过 learned pixel-shuffle upsampling 恢复 dense flow。

这带来两个好处：

- 计算量可控，不需要对每个原始像素做 3D kNN；
- anchor points 仍保留全局 geometry features 和 image context，最后可以恢复到 dense motion field。

### 3.3 2D-to-3D Correlation

传统 3D tracking 方法常做显式 3D 空间相关：

```text
source 3D point -> target 3D point kNN search -> cross-attention -> 3D flow update
```

这个过程对 dense points 很容易爆显存。Track4World 的关键设计是先在 2D 图像平面做 correlation，再把 2D flow update lift 到 3D：

1. 在 source image 和 target image 上构建 geometric / semantic correlation volumes；
2. 用 GRU-style recurrent operator 迭代更新 2D flow 和 visibility；
3. 根据当前 2D target position 从 target point map 中插值得到 3D 坐标；
4. 用 lifted target samples、source context、3D spatial similarity 和 historical 3D flow prior 预测 3D flow update。

复杂度上，传统 3D matching 至少需要 `O(N log N + N*k)`，dense global attention 甚至接近 `O(N^2)`；Track4World 的 warp-sampling 通过 2D 坐标查表，接近 `O(N)`。这也是它能做 dense tracking 的核心。

### 3.4 2D-3D Joint Supervision

因为 3D scene flow 标注很稀缺，只靠 3D 数据容易过拟合或泛化差。Track4World 的结构天然同时产生：

```text
2D flow / visibility / confidence
3D scene flow / 3D trajectory
```

于是训练可以混合使用 optical flow、2D point tracking、scene flow、3D point tracking 数据。

2D branch 的 loss 包括：

```text
L_2D = trajectory L1 + visibility BCE + confidence BCE
```

3D branch 的 loss 包括：

```text
L_3D = 3D trajectory L1 + scene-flow smoothness
```

其中 trajectory loss 对可见点权重大，对不可见点仍给 0.2 的弱监督；3D smoothness 在 anchor samples 上做局部邻域约束，避免 flow field 在空间上抖动。

这个 joint supervision 的价值不只是“多一个辅助 loss”，而是让 2D 数据成为 3D motion learning 的稳定支架。对机器人视频而言，这类设计很有意义：真实 3D 动作/scene flow label 难拿，但 2D tracking 或 optical flow 数据便宜得多。

### 3.5 训练与优化

补充材料给出的训练分两阶段：

| 阶段 | 内容 | 数据 | 设置 |
|---|---|---|---|
| Geometry stage | 训练/微调 geometry estimation backbone | Kubric-3D、GTA-SfM、V-KITTI、ARKitScenes、BlinkVision、DynamicStereo、TartanAir、ScanNet、Hypersim 等 | 8 x 40GB GPU，AdamW，StepLR，LR `1e-4`，100k steps，约 1 周 |
| Motion stage | 冻结 geometry module，训练 motion estimation module | AutoFlow、FlyingChairs、OmniWorld、HD1K、Spring、TartanAir、VIPER、Kubric、Monkaa、Driving、V-KITTI、DynamicStereo、PointOdyssey 等 | 8 x 40GB GPU，AdamW，OneCycleLR，peak LR `1e-4`，100k steps，约 5 天 |

几何 encoder 是 backbone-agnostic，可以用 MoGe、Pi3、DA3 等初始化。默认报告使用 DA3 初始化。为了提升时序一致性，作者会对 geometry backbone 做针对性微调，比如给 monocular model 加 global attention layers 和 camera pose tokens，或更新 3D reconstruction model 的中间层。

## 四、实验与结果

### 4.1 实验设置

Track4World 评估覆盖五类能力：

| 任务 | 数据集/指标 |
|---|---|
| Scene & optical flow | Kubric-3D、KITTI、BlinkVision；Abs Rel、EPE3D、EPE2D、AccS/AccR |
| 3D tracking | PointOdyssey、ADT、PStudio、DriveTrack；APD，L-16 / L-50 |
| 2D tracking | Kinetics、RoboTAP、RGB-Stacking；AJ、delta_vis avg、OA |
| Point map estimation | GMU Kitchen、Monkaa、Sintel、ScanNet、Kubric-3D、KITTI、TUM；Abs Rel、delta < 1.25 |
| Camera pose estimation | Sintel、Bonn；ATE、RTE、RRE |

### 4.2 主要结果

**Scene / optical flow：in-domain 和 out-of-domain 都领先。**

| Dataset | Metric | Previous Strong Baseline | Track4World |
|---|---|---:|---:|
| Kubric-3D short | EPE3D ↓ | OpticalExpansion 0.2093 / ZeroMSF 0.3528 | 0.1537 |
| Kubric-3D long | EPE3D ↓ | OpticalExpansion 0.7037 / ZeroMSF 1.2182 | 0.4808 |
| KITTI | EPE3D ↓ | ZeroMSF 0.1823 / OpticalExpansion 0.2419 | 0.0742 |
| BlinkVision | EPE3D ↓ | ZeroMSF 0.3937 / OpticalExpansion 0.4406 | 0.1135 |
| Kubric-3D short | EPE2D ↓ | RAFT 6.7974 | 1.8685 |
| KITTI | EPE2D ↓ | GMFlowNet 4.6977 / RAFT 5.4150 | 2.5722 |

最有说服力的是 KITTI 和 BlinkVision 这两个 out-of-domain 结果：Track4World 的 EPE3D 明显低于 ZeroMSF、Any4D、V-DPM，说明 2D-to-3D correlation 不是只在合成数据上有效。

**3D tracking：camera/world 两个坐标系都强。**

| Setting | Best Baseline Avg L-16 | Track4World Avg L-16 | Best Baseline Avg L-50 | Track4World Avg L-50 |
|---|---:|---:|---:|---:|
| Camera coordinate | STV2 0.2400 / ZeroMSF 0.4782 | 0.5712 | ZeroMSF 0.4390 | 0.5469 |
| World coordinate | V-DPM 0.5142 / Any4D 0.5043 | 0.5636 | V-DPM 0.4668 | 0.5323 |

world coordinate 结果尤其重要，因为它把相机自运动和物体真实运动分开。对移动机器人视频，这比 camera-centric tracking 更有用：机器人一边移动一边观察，背景不应该被误认为动态物体。

**2D tracking：虽然目标是 3D，但 2D branch 也达到强水平。**

| Dataset | Metric | CoTracker3 | Track4World |
|---|---|---:|---:|
| Kinetics | AJ ↑ | 55.8 | 59.1 |
| Kinetics | OA ↑ | 88.3 | 90.6 |
| RoboTAP | AJ ↑ | 66.4 | 70.9 |
| RoboTAP | OA ↑ | 90.8 | 93.3 |
| RGB-Stacking | AJ ↑ | 71.7 | 78.2 |
| RGB-Stacking | OA ↑ | 91.1 | 92.3 |

RoboTAP 和 RGB-Stacking 都和机器人操作更接近。Track4World 在这些数据上强，说明几何分支没有牺牲 2D tracking 的实用性。

**Point map / camera pose：几何基础没有拖后腿。**

| Task | Track4World 结果 |
|---|---:|
| Point map Avg Abs Rel ↓ | 0.0552 |
| Point map Avg delta < 1.25 ↑ | 0.9440 |
| Sintel ATE/RTE/RRE ↓ | 0.119 / 0.054 / 0.309 |
| Bonn ATE/RTE/RRE ↓ | 0.009 / 0.009 / 0.604 |

几何本身不是论文唯一贡献，但如果 point map 和 pose 不稳，world-centric trajectory 会被相机误差污染。这里的结果说明 Track4World 的 geometry backbone 微调对 motion estimation 有支撑作用。

### 4.3 消融实验

**2D supervision 是 3D flow 的关键支架。**

| Setting | Abs Rel ↓ | delta < 1.25 ↑ | EPE3D ↓ |
|---|---:|---:|---:|
| w/o 2D Supervision | 0.1021 | 0.9199 | 0.6511 |
| Full | 0.0474 | 0.9607 | 0.2056 |

去掉 2D supervision 后 EPE3D 从 0.2056 坏到 0.6511，是所有消融里最明显的崩塌之一。这说明 abundant 2D data 对 3D motion generalization 不是锦上添花。

**decoder 组件都在服务 2D-to-3D lifting。**

| Setting | EPE3D ↓ |
|---|---:|
| w/o Target lifting | 0.3017 |
| w/o iterations | 0.2356 |
| w/o auxiliary `C3d` & `M3d` priors | 0.2201 |
| Full | 0.2056 |

target lifting 的影响最大，说明把 2D match 映射回 3D residual 是核心；迭代更新和 3D priors 则进一步提升精细程度和时序稳定性。

**hybrid formulation 比纯 2D 或纯 3D 都好。**

| Formulation | Abs Rel ↓ | delta < 1.25 ↑ | EPE3D ↓ |
|---|---:|---:|---:|
| 2D flow + depth | 0.1304 | 0.8911 | 0.8210 |
| Pure 3D flow | 0.0552 | 0.9570 | 0.2815 |
| Full hybrid | 0.0474 | 0.9607 | 0.2056 |

简单用 2D flow 加 depth lifting 不够，纯 3D 回归也不如 full hybrid。真正有效的是：用 2D correlation 提供可靠 image-plane match，再用 3D geometry 修正空间运动。

**效率：dense tracking 下传统 3D correlation 会 OOM。**

| Method | Time (s) ↓ | Mem. (GB) ↓ | Params (M) ↓ |
|---|---:|---:|---:|
| POMATO Dense | 4.8 | 16 | 133.64 |
| ZeroMSF Dense | 8.2 | 10 | 153.84 |
| STV2 Sparse | 5.8 | 19 | 65.99 |
| STV2 Dense | OOM | OOM | 65.99 |
| Ours w/o 2D-to-3D Dense | OOM | OOM | 56.90 |
| Track4World Dense | 3.4 | 14 | 26.06 |

这个表直接证明 bottleneck 在传统 3D spatial correlation。Track4World 不只是更准，也更能扩展到 dense all-pixel setting。

## 五、局限性与展望

作者明确指出，Track4World 仍依赖 captured 4D motion datasets，这些数据采集成本高、规模有限。因此模型在训练集中没覆盖的 extreme poses 和 complex topological changes 上可能泛化不足。未来方向包括用 generative diffusion models 或 physics engines 生成更大规模、更丰富的训练样本，以及研究无监督/半监督学习来降低标注依赖。

我的补充判断：

1. **还不是机器人闭环感知模块**：论文评估主要是离线 4D reconstruction / tracking，没有展示接入机器人 policy 后对成功率的影响。

2. **世界坐标系依赖 pose/geometry 稳定性**：world-centric tracking 的价值很大，但如果相机 pose 或 point map 在强遮挡、低纹理、反光物体上漂移，轨迹也会被污染。

3. **dense signal 很强，但动作语义仍缺失**：Track4World 能告诉你每个点如何运动，却不直接告诉你“这个运动对应抓、推、插、开门还是避障”。要服务 VLA，还需要和语言、接触、action label 或 latent action objective 结合。

4. **训练数据生态重**：两阶段训练混合大量 geometry、flow、tracking 数据集，普通实验室复现完整训练成本不低；更现实的用法可能是直接用开源 checkpoint 抽取 robot video features / pseudo labels。

## 六、灵魂三问

1. **它解决了什么问题？**

它解决的是单目视频中 dense 3D motion reconstruction 难以兼顾完整性、效率和世界坐标一致性的问题。相比稀疏 3D tracker，它覆盖全像素和后续新出现对象；相比 optimization-based dense pipeline，它是 feedforward，速度和可扩展性更好。

2. **为什么这么做？**

因为直接做 dense 3D correlation 太贵，而只做 2D flow 又缺少空间运动。Track4World 先用 2D correlation 找稳定图像匹配，再把匹配 lift 到 3D point map 中估计 scene flow，相当于用便宜且数据充足的 2D 监督托住稀缺的 3D motion learning。

3. **什么证据最有说服力？**

最有说服力的是消融和效率表：去掉 2D supervision 后 EPE3D 从 0.2056 坏到 0.6511；把 2D-to-3D correlation 换成传统 dense 3D 机制直接 OOM。这说明论文的核心机制同时解决了精度和规模问题。

## 七、个人总结

1. Track4World 的核心 idea 是把“全像素 3D 轨迹”分解成任意帧对 2D/3D scene flow，再用 world-centric fusion 得到全局 motion field。

2. 最大优势是 2D-to-3D joint design 很干净：2D 数据、3D 几何、scene flow、tracking 监督能合在一个 feedforward 框架里；最大弱点是它仍是感知/重建模型，和机器人 action learning 之间还差一层任务语义和控制接口。

3. 对机器人视频学习来说，它提供了一种很有价值的 pseudo-supervision 来源：与其只从 RGB 预测 action，不如先把视频变成 dense 3D motion、world-centric trajectories 和 visibility/confidence，再让 VLA 的预训练去吸收这些结构化物理线索。

## 八、可借用的预训练信号

Track4World 可以和 VLA action pretraining 接上几类信号：

| 信号 | 可能用途 | 风险 |
|---|---|---|
| Dense 3D scene flow | 给视频轨迹提供物体/手/末端执行器的空间运动监督 | 对透明、反光、低纹理物体可能不稳 |
| World-centric trajectory | 把移动相机自运动和物体真实运动分开，适合 mobile manipulation 视频 | 依赖 pose refinement 和静态区域分割 |
| Visibility / confidence | 训练 policy 区分可靠 motion cue 和遮挡/不确定区域 | confidence 不等于任务相关性 |
| 2D-3D paired flow | 让便宜 2D data 参与 3D motion prior 学习 | 2D 数据域和机器人场景差异仍需处理 |

如果和 ABot-M0.5、MiVLA、ACE-Ego-0 这类路线放在一起看，Track4World 更像底层视觉运动表征供应器：它不解决 action space bridge，但能让“无动作视频”不再只是 RGB 序列，而是带 dense 3D dynamics 的预训练材料。

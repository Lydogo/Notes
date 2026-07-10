# Spatial Forcing：不用显式深度输入，也让 VLA 学到空间感

> 原标题：Spatial Forcing: Implicit Spatial Representation Alignment for Vision-language-action Model  
> 作者：Fuhao Li, Wenxuan Song, Han Zhao, Jingbo Wang, Pengxiang Ding, Donglin Wang, Long Zeng, Haoang Li  
> 机构：HKUST(GZ), Tsinghua University, Westlake University, Zhejiang University, South China University of Technology  
> 发表：arXiv:2510.12276v2, 2025-10-17；项目页标注 ICLR 2026  
> 链接：https://arxiv.org/abs/2510.12276  
> 项目页：https://spatial-forcing.github.io/  
> 开源：项目页提供 Code / Models 入口

---

## 一、研究背景与动机

当前主流视觉-语言-动作模型（Vision-Language-Action, VLA）通常继承视觉语言模型（Vision-Language Model, VLM）的视觉骨干。问题是，这些 VLM 多数只在 2D 图像上预训练，语义能力不错，但对机器人真正需要的 3D 空间关系、物体高度、相对位姿、遮挡后的几何结构并不天然敏感。

已有 3D VLA 的常见路线是把深度图、点云或由深度估计器生成的 3D cues 显式喂给模型。但这条路线有几个现实阻力：

1. 深度传感器数据容易受噪声、反光、透明物体和标定误差影响。
2. 不同机器人传感器类型、安装位置和标定状态差异很大，跨 embodiment 扩展麻烦。
3. 大规模机器人数据集并不总是有深度或点云，显式 3D 输入会限制数据规模。
4. 用 2D 图像估计深度再喂给 VLA，又受限于深度估计器本身的误差。

这篇论文问的问题很直接：能不能不在推理时增加深度/点云输入，而是在训练时“逼”VLA 的中间视觉表征吸收 3D 空间信息？

作者先做了一个轻量 depth probing：冻结 VLA，只训练一个 DPT head 从 VLA 视觉 embedding 预测深度。结果显示，未经空间对齐的视觉 embedding 很难恢复有意义的空间结构。这说明问题可能不只是动作 head 不够强，而是中间视觉 token 本身缺空间表征。

## 二、核心贡献

1. 提出一个诊断观察：现有 2D VLA 的中间视觉 embedding 缺少足够 3D 空间信息。作者通过 depth probing 可视化和定量分析说明，单靠 2D 预训练语义特征不足以支撑精细机器人动作。

2. 提出 Spatial Forcing（SF）：在训练阶段把 VLA 的中间视觉 token 与预训练 3D foundation model 的几何表征对齐，从而隐式注入空间理解能力。

3. 保持推理零额外开销：SF 只作为训练监督存在，推理时不需要 VGGT、深度图、点云或额外传感器。

4. 在仿真和真实机器人上验证性能、收敛速度和数据效率：LIBERO 平均成功率达到 98.5%，训练效率最高提升 3.8×；真实机器人任务在少量 demo 下也有明显增益。

## 三、方法原理

### 3.1 整体框架

SF 的整体思想是“训练时借 3D 老师，推理时学生自己走”。

输入仍然是机器人多视角 RGB 图像和语言指令。VLA 按正常流程把图像编码成视觉 token，把语言编码成文本 token，再通过 action expert 或 flow-matching head 输出动作。

SF 额外引入一个冻结的 3D foundation model：VGGT。VGGT 接收同一组多视角 2D 图像，输出包含几何信息的像素级空间表征。训练时，作者把 VLA 某一中间层的视觉 token 映射到与 VGGT 表征兼容的空间，然后用 cosine similarity 做对齐监督。

推理时，VGGT 和对齐 loss 都被拿掉，模型结构与普通 VLA 一样。

### 3.2 关键技术细节

#### 对齐目标：VGGT 的几何 latent，而不是深度图本身

作者没有让 VLA 直接预测深度图，而是对齐 VGGT transformer backbone 的空间 latent representation。这样做的好处是：

- latent 里包含多视角一致性、点图、深度、相机等 3D 线索，比单一深度图更丰富；
- 避免把深度估计结果作为推理输入，减少 sensor 和 estimator 依赖；
- 对齐的是中间视觉表征，更贴近 action token 在自回归生成时真正会用到的信息。

#### 对齐位置：较深但不是最深的 VLA 层

VLA backbone 有多层 causal attention。作者发现，在比较深但非最后的层做监督最有效。论文的解释是：浅层视觉特征还不够全局，最后几层又更接近跨模态、语义化、模态无关的空间，视觉专属性下降；第 24 层在 32 层 backbone 中表现最好。

这个观察挺有意思：它说明“给空间监督”不是越靠输出越好，而要找视觉几何信息还没有被语言/action 语义完全混合掉的位置。

#### 位置编码很关键

VGGT 输出的空间表征会加额外 positional embedding（PE）。消融显示，VGGT w/o PE 在 LIBERO-Long 上明显掉点：84.4 vs 94.2。直觉上，这是因为 VLA 自回归生成 action 时，视觉 token 的相对位置顺序会影响动作生成；如果只学几何语义而不保留 token 的位置结构，长程任务更容易出错。

#### 损失函数

标准动作损失为：

```text
L_action = L[G(action tokens), ground-truth action]
```

SF 增加视觉表征对齐损失：

```text
L_align = - mean cosine_similarity(MLP(BN(VLA visual token)), VGGT spatial feature + PE)
```

最终目标：

```text
L_SF = L_action + α L_align
```

直观理解：动作监督告诉模型“怎么动”，VGGT 对齐告诉模型“你看到的空间结构应该长什么样”。

### 3.3 训练与优化

论文在两个基础模型上验证：

- LIBERO：基于 OpenVLA-OFT，Prismatic VLM backbone，融合 SigLIP + DINOv2 视觉 backbone；8×H100 训练 150k iterations 用于主对比。
- RoboTwin：基于 π0，PaliGemma VLM backbone，LoRA 训练，1×H100 训练 30k iterations。

在组件分析中，作者为了节省算力主要使用 1×H100，并以 OpenVLA-OFT 作为 base model 在 LIBERO 上做消融。

## 四、实验与结果

### 4.1 实验设置

仿真实验包含两个 benchmark：

- LIBERO：包含 Spatial、Object、Goal、Long 四个 suite，每个 suite 10 个任务，每个任务 500 条专家 demonstration；评估指标为 success rate（SR）。
- RoboTwin 2.0：real-to-sim 双臂 benchmark，包含 easy setting 和 hard setting；hard setting 有场景杂乱、背景纹理、光照和桌面高度随机化。

真实机器人实验使用 AgileX 双臂平台：每个 arm 是 6-DoF Piper manipulator + 1-DoF gripper，配一个 primary camera 和两个 wrist cameras。单臂任务每个只用 40 demonstrations，双臂任务只用 20 demonstrations。

### 4.2 主要结果

#### LIBERO 主结果

| 方法 | 类型 | Spatial | Object | Goal | Long | Average |
|---|---:|---:|---:|---:|---:|---:|
| π0 | 2D VLA | 96.8 | 98.8 | 95.8 | 85.2 | 94.2 |
| UniVLA | 2D VLA | 96.5 | 96.8 | 95.6 | 92.0 | 95.2 |
| OpenVLA-OFT | 2D VLA | 97.6 | 98.4 | 97.9 | 94.5 | 97.1 |
| SpatialVLA | explicit 3D VLA | 88.2 | 89.9 | 78.6 | 55.5 | 78.1 |
| GeoVLA | explicit 3D VLA | 98.4 | 99.0 | 96.6 | 96.6 | 97.7 |
| 3D-CAVLA | explicit 3D VLA | 98.2 | 99.8 | 98.2 | 96.1 | 98.1 |
| Spatial Forcing | implicit 3D VLA | **99.4** | **99.6** | **98.8** | 96.0 | **98.5** |

要点：

- SF 在不使用额外深度/点云输入的情况下，平均 SR 达到 98.5。
- 它超过了 OpenVLA-OFT 的 97.1，也接近或超过显式 3D VLA。
- Long 任务上不是最高，但仍从 OpenVLA-OFT 的 94.5 提到 96.0。

#### RoboTwin 2.0

论文图 4 显示，SF 在 easy 和 hard 任务上都比 π0 base model 有明显提升，并取得最高平均成功率。更关键的是 hard setting 的提升：这说明 SF 不只是拟合标准布局，而是更能抓住物体相对空间关系，减少对背景、光照、桌面高度这类 shortcut correlation 的依赖。

#### 真实机器人结果

| 任务 | 变化类型 | w/o SF | w/ SF | 主要观察 |
|---|---:|---:|---:|---|
| Stack Glass Cups | 光照变化、透明杯 | 15.0 | 62.5 | 透明/反光目标下，SF 明显减少视觉 shortcut 依赖 |
| Grasp Right-side Vegetable | 目标物体变化 | 10.0 | 47.5 | 需要理解不同物体的 3D 外形与夹爪姿态 |
| Place Green Block | 高度变化 | 67.5 | 85.0 | 对高度估计更敏感 |
| Lift Pot | 双臂水平平衡 | 30.0 | 42.5 | 对新配置和水平空间关系有帮助 |

单臂每任务仅 40 demos，双臂任务仅 20 demos。这里最有价值的不是绝对成功率，而是低数据下仍能显著提升。

### 4.3 消融实验

#### 目标表征与对齐层

| Target Representation | Aligned Layer | Spatial | Object | Goal | Long | Average |
|---|---:|---:|---:|---:|---:|---:|
| 无 SF baseline | - | 96.8 | 94.8 | 92.8 | 86.2 | 92.7 |
| SigLIP | 24 | 95.2 | 94.8 | 94.0 | 91.8 | 94.0 |
| DINOv2 | 24 | 93.4 | 95.2 | 93.8 | 93.8 | 94.1 |
| VGGT w/o PE | 24 | 97.8 | 100.0 | 96.6 | 84.4 | 94.7 |
| VGGT + PE | 24 | 97.2 | 99.2 | 96.8 | 94.2 | **96.9** |
| VGGT + PE | 1 | 96.8 | 99.4 | 99.0 | 83.0 | 94.6 |
| VGGT + PE | 8 | 96.2 | 98.4 | 95.6 | 92.4 | 95.7 |
| VGGT + PE | 16 | 97.4 | 98.8 | 95.8 | 83.2 | 93.8 |
| VGGT + PE | 32 | 98.8 | 99.4 | 96.2 | 84.8 | 94.8 |

要点：

- 对齐任意强视觉表征都有帮助，但 VGGT 最强，说明缺的主要是 3D 几何能力。
- PE 对长程任务非常重要。
- 第 24 层效果最好，支持“较深但不是最终层”的表征监督策略。

#### 训练效率

SF 达到同等成功率所需训练迭代更少，作者报告最高 3.8× training efficiency。这个结果的意义是：空间监督不只是提升最终性能，还给 VLA 提供了一条更短的表示学习路径。

#### 数据效率

| 训练数据比例 | Spatial | Object | Goal | Long | Average |
|---:|---:|---:|---:|---:|---:|
| 1% | 32.8 | 67.8 | 44.8 | 23.6 | 42.3 |
| 5% | 73.2 | 83.4 | 80.6 | 66.0 | 75.8 |
| 100% | 97.2 | 99.2 | 96.8 | 94.2 | 96.9 |

作者还报告，在相同数据量下 SF 可带来 25.8% 成功率提升，并在达到相同成功率时有 5.9× 数据效率提升。对机器人来说，这个点很现实：真实机器人 demo 贵，少量数据下的 inductive bias 比“再收一万条”更可用。

## 五、局限性与展望

作者没有单独给出很强的 limitation section，但从方法和实验可以看出几个边界：

1. 训练依赖 3D foundation model 的质量。SF 不在推理时依赖 VGGT，但训练监督来自 VGGT；如果 VGGT 对某些场景、材质或相机分布不稳，监督信号也会偏。

2. 主要解决空间表征，不直接解决长程规划。LIBERO-Long 有提升，但不是最强项；更复杂的任务分解、记忆、纠错仍需要其他机制。

3. 真实实验规模仍偏小。真实机器人任务覆盖透明物体、高度变化和双臂平衡，但任务数量和平台类型有限，还不能说明跨大量 embodiment 的通用性。

4. 对多模态干预的解释还可以更深入。t-SNE 和 depth probing 说明对齐有效，但还不足以完全解释哪些几何因素最终影响动作。

后续可以考虑：

- 把 SF 与更大规模多机器人预训练结合；
- 对不同 3D teacher 做系统比较，比如 VGGT、depth foundation model、scene flow/video 3D model；
- 在移动操作、动态场景和遮挡恢复中验证空间监督是否仍稳定；
- 研究是否能把 SF 扩展到 action token 或 world-model latent，而不只是 visual token。

## 六、灵魂三问

1. **它解决了什么问题？**

它解决的是 2D VLA 中间视觉表征缺乏 3D 空间感的问题。以往要么显式输入深度/点云，要么依赖深度估计器；SF 改成训练时对齐 3D foundation model 的几何 latent，让 VLA 自己把空间结构吸收到视觉 token 里。

2. **为什么这么做？**

因为动作生成真正依赖的是 VLA 内部 token，而不只是输入图像。如果视觉 token 本身没有空间结构，action expert 再强也会受限。训练时用 VGGT 做几何老师，推理时不引入额外传感器，是一个工程上很干净的折中：拿到 3D inductive bias，又不增加部署复杂度。

3. **什么证据最有说服力？**

最有说服力的是 LIBERO + 消融的组合证据：SF 平均 SR 从 OpenVLA-OFT 的 97.1 提到 98.5，并且 VGGT+PE、第 24 层对齐明显优于其他 target/layer 设置。真实机器人少样本实验也补了一层部署证据，尤其透明杯和高度变化任务能直接对应空间理解。

## 七、个人总结

1. 核心想法一句话：SF 是一种“训练时用 3D teacher 改造 VLA 视觉 token，推理时不带 teacher 上场”的空间表征蒸馏方法。

2. 最大优势是轻量、可插拔、推理零开销；最大弱点是它仍依赖 teacher 表征质量，而且主要补空间感，不等于解决了长程任务的规划和恢复。

3. 对 VLA 研究很有启发：与其一直往输入里堆 depth/point cloud，不如认真看 VLA 中间表示是不是学到了动作需要的几何结构。这个方向和 representation supervision、world model、future prediction 都能自然接上。

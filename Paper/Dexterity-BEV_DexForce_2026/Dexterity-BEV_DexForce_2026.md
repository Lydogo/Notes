# Dexterity-BEV：对齐 3D 世界与动作以实现可泛化的机器人策略学习

> 原标题：Dexterity-BEV: Aligning 3D World and Actions for Generalizable Robot Policies Learning
> 作者：Huayi Zhou, Wei Gao, Dekun Lu, Ruiji Liu, Zhanqi Zhang, Ziyang Zhang, Jian Chen, Wenlve Zhou, Sheng Xu, Shumin Li, Kangyi Guo, Shichen Xu, Zixin Huang, Yongyi Su, Kui Jia
> 发表：arXiv, 2026
> 链接：https://arxiv.org/abs/2606.02274
> 项目页：https://hnuzhy.github.io/projects/Dex-BEV

---

## 一、研究背景与动机

- **核心问题**：当前端到端操控策略（VLA）大多继承自 2D 预训练 VLM，但机器人操控本质上是 3D 问题。现有方法面临两类错位：
  1. **输入侧**：只用 RGB，忽略相机标定、深度等 3D 几何信息；
  2. **输出侧**：关节角或末端位姿在不同机器人、数据集、坐标系约定下不可比，造成跨本体泛化困难。
- **现有方法不足**：
  - 纯 2D VLA（π₀、X-VLA、OpenVLA 等）对相机视角变化、场景布局扰动极其敏感；
  - 纯 3D 表示（点云、体素、3DGS）难以复用 web-scale 2D VLM 预训练；
  - 已有 3D-VLA（SpatialVLA、DepthVLA 等）多在**各相机坐标系内**独立处理，缺乏多视角几何关联，且动作空间未与观测统一到同一 3D 坐标系。
- **本文目标**：在保留 2D VLM 泛化能力的前提下，把观测和动作都提升到**统一的 3D BEV 对齐空间**，并配套大规模时空对齐数据处理管线，实现跨本体、跨相机、跨数据集的灵巧操控泛化。

---

## 二、核心贡献

1. **Aligned Vertex Map & Vertex Spectrum**：像素级 3D 表示，将 2D 视觉输入"抬升"到 3D，可与预训练 2D VLM 无缝融合；无深度时用 Vertex Spectrum（多深度假设）替代。
2. **BEV 对齐框架**：选定规范鸟瞰图（Bird's-Eye-View）坐标系，构造合成 BEV 图像及对应 vertex map，实现视角不变的几何输入；同时将本体感知和动作统一表达为 BEV 坐标系下的 $SE(3)$ 位姿。
3. **时空对齐数据处理管线**：GUI 辅助 + ICP + 视觉基础模型（DepthAnything V3、FoundationStereo 等）完成多数据集 3D 空间对齐；提出基于末端速度归一化的**跨轨迹时间对齐**。
4. **系统验证**：仿真（LIBERO、RoboTwin 2.0）+ 四套真实双臂平台、五项长时程灵巧任务，证明在相机扰动、场景布局变化、跨本体部署上的显著优势。

---

## 三、方法原理

### 3.1 整体框架

```
多视角 RGB (+ 可选深度)
        ↓
  Aligned Vertex Map / Vertex Spectrum  ← 统一到 BEV 坐标系
        ↓
  合成 BEV 图像 + BEV vertex map       ← 视角不变几何输入
        ↓
  预训练 VLM backbone + 语言指令
        ↓
  Flow Matching Action Expert
        ↓
  BEV 坐标系下的 SE(3) 动作 chunk
```

**输入** $\mathcal{X}_t = \{(\mathbf{O}_{t,i}, \mathbf{K}_i, \mathbf{T}_{t,i})_{i=1}^N,\, \mathcal{L},\, \mathbf{s}_t\}$：多相机 RGB/深度、内外参、语言指令、本体感知。

**输出**：未来 $M$ 步动作 chunk $\{\mathbf{A}_{t+m}\}_{m=1}^M$，全部为 BEV 坐标系下的 $SE(3)$ 位姿（含统一 TCP 约定）。

### 3.2 关键技术细节

#### （1）Aligned Vertex Map：像素级 3D 对齐

给定深度图 $\mathbf{D}_{t,i}$ 和内参 $\mathbf{K}_i$，像素 $(u,v)$ 反投影到相机坐标系：

$$
\mathbf{P}_{\text{camera}_i}(u,v) = \mathbf{K}_i^{-1}[u,v,1]^T \cdot \mathbf{D}_{t,i}(u,v)
$$

**直觉**：每个像素不再只是颜色，而是携带一个 3D 坐标，且保持与 RGB 的像素对齐，可直接喂给 2D VLM。

SpatialVLA 等先前工作只在**各相机本地坐标系**使用 vertex map，同一物理点在不同视角下数值差异巨大。Dex-BEV 将所有视角变换到共享 BEV 对齐帧：

$$
\mathbf{F}_{3d,i} = \text{Enc}_{3d}(\mathbf{P}_{\text{aligned}_i}) = \text{Enc}_{3d}\!\left(\mathbf{T}_{\text{align}}^{-1}\,\mathbf{T}_{t,i}\,\mathbf{P}_{\text{camera}_i}\right)
$$

视觉特征融合：$\mathbf{F}_{\text{combined},i} = \text{Enc}_{\text{vis}}(\mathbf{I}_{t,i}) + \text{Enc}_{3d}(\mathbf{P}_{\text{aligned}_i})$。

**与已有方法的关键区别**：不是各视角独立编码 3D，而是**先对齐再编码**，多视角几何关联直接可见；动作和本体感知也在同一 $\mathbf{T}_{\text{align}}$ 下表达。

#### （2）BEV 图像构造：视角不变输入

$\mathbf{T}_{\text{align}}$ 实例化为规范 BEV 帧（机器人基座帧，或桌面工作区 ROI 底面中心）。

- 聚合所有相机的彩色点云；
- 沿 BEV 帧 $z$ 轴做正交俯投影，光栅化为 $224 \times 224$ RGB 图像；
- 同步生成像素对齐的高度图 → 转为 BEV vertex map。

**直觉**：即使第三人称相机位姿大幅变化，合成 BEV 图像中物体仍落在几乎相同的像素位置——这是对抗相机扰动的核心。

#### （3）Vertex Spectrum：无深度时的替代方案

受 PETR（自动驾驶）启发，对无深度相机，对每个像素采样 $M$ 个离散深度假设（LID 线性递增离散化）：

$$
d_j = d_{\min} + (d_{\max} - d_{\min}) \cdot \frac{j(j+1)}{M(M+1)}
$$

每个像素-深度对反投影并变换到 BEV 帧，得到体积坐标网格 $\mathcal{G}_{u,v} \in \mathbb{R}^{M \times 3}$，经轻量编码器生成位置嵌入，逐元素加到 RGB 特征上。

**直觉**：没有深度传感器时，用"深度光谱"让网络自己从中选合适深度，而非硬依赖单点深度。

#### （4）动作与本体感知的 3D 对齐

- 统一 **TCP（Tool Center Point）约定**：平行夹爪锚在爪尖，多指手锚在腕部；
- 各机器人 URDF 注册到共享 3D 观测空间，经正运动学计算统一 $SE(3)$ 位姿；
- 双臂机器人不再分左右臂基座帧，全部表达在 BEV 帧下。

#### （5）跨轨迹时间对齐

遥操作轨迹速度因操作者和平台而异，但大多数操控任务是**准静态**的——快慢不影响任务完成。Dex-BEV 对轨迹做速度归一化：

$$
\Delta\tau_t = \max\!\left(\frac{\Delta L_t}{v_{\text{std}}},\; \frac{\Delta \theta_t}{\omega_{\text{std}}}\right)
$$

其中 $\Delta L_t$、$\Delta \theta_t$ 分别为末端平移和旋转位移，$v_{\text{std}}$、$\omega_{\text{std}}$ 为预定义标准速度。双臂取两臂最大值；近静态帧可丢弃或保留原时长；训练时用三次样条插值得到 action chunk。

**直觉**：把"同任务不同速度"的轨迹拉齐到统一时间尺度，减少模型需要消化的无意义变化。

### 3.3 训练与优化

- **骨干**：预训练 2D VLM（架构与 π₀ / X-VLA 同类范式）+ Flow Matching action expert。
- **损失函数**（Flow Matching）：

$$
\mathcal{L}_{\text{FM}} = \mathbb{E}_{\sigma \sim \mathcal{U}[0,1],\,\mathbf{a}_1 \sim p_{\text{data}},\,\mathbf{a}_0 \sim \mathcal{N}(0,I)}\!\left[\left\|\mathbf{v}_\theta(\sigma\mathbf{a}_1 + (1-\sigma)\mathbf{a}_0,\,\sigma,\,\mathbf{c}_t) - (\mathbf{a}_1 - \mathbf{a}_0)\right\|^2\right]
$$

其中 $\mathbf{c}_t$ 为 VLM 多模态上下文嵌入。推理时通过 ODE 求解器从噪声采样动作序列。

- **数据处理管线**（训练前离线完成）：
  - 空间对齐：GUI 手动匹配 + ICP + DepthAnything V3 估计深度/外参；缺失深度用仿真回放或 FoundationStereo 合成；
  - 时间对齐：上述速度归一化 + 样条插值；
  - 覆盖数据集：LIBERO、RoboTwin、RoboCasa、RoboMind、Droid、Agibot、内部数据等。

- **2D 消融**：移除全部 3D 输入，且动作表达回退到 X-VLA 约定（不做 3D 对齐），用于隔离 3D 对齐的贡献。

---

## 四、实验与结果

### 4.1 实验设置

| 维度 | 内容 |
|---|---|
| **仿真基准** | LIBERO（单臂 Franka 7-DoF）、RoboTwin 2.0（双臂 Agile-X 12-DoF） |
| **扰动评测** | Modified LIBERO：第三人称相机随机旋转（各轴 60°/140°/60°）、距离 ±0.5m；机器人+场景基座位姿随机平移 ±10cm、旋转 ±5° |
| **真实平台** | Agilex 双臂、DexForce W1（灵巧手/夹爪）、DexForce A1 半人形 |
| **真实任务** | 折纸盒、折衣服（Agilex/A1）、舀爆米花（W1 灵巧手）、递书（W1 夹爪） |
| **基线** | π₀、X-VLA；仿真还对比 OpenVLA、SpatialVLA、4D-VLA、GeoVLA 等 15+ 方法 |
| **指标** | 任务成功率（每任务 30 次 rollout） |
| **跨本体** | **单一 checkpoint** 同时部署 LIBERO + RoboTwin |

### 4.2 主要结果

**仿真官方设置（Tab. 1）**

| 方法 | 跨本体 | LIBERO Avg | RoboTwin Clean | RoboTwin Randomized |
|---|---|---|---|---|
| π₀ | ✗ | 94.2% | 46.4% | 16.4% |
| X-VLA | ✗ | 98.1% | 70.0% | 39.0% |
| 2D Ablation | ✓ | 92.8% | 64.8% | 35.2% |
| **Dex-BEV** | **✓** | **97.8%** | **76.0%** | **42.0%** |

官方设置下 Dex-BEV 与 SOTA 持平或更优，且**一个权重覆盖两种截然不同的机器人**。

**Modified LIBERO（相机+场景强扰动）**

| 方法 | Average |
|---|---|
| X-VLA（官方 ckpt） | <10% |
| 2D Ablation | <10% |
| **Dex-BEV** | **89.9%** |

这是全文最有冲击力的数字：基线几乎归零，Dex-BEV 仍维持近 90%。

**真实机器人（5 任务 × 30 次）**

| 任务 | π₀ | X-VLA | Dex-BEV |
|---|---|---|---|
| Fold Mailer Box | 43.3% | 56.7% | **76.7%** |
| Fold Cloth (Agilex) | 66.7% | 80.0% | **93.3%** |
| Scoop Popcorn | 60.0% | 70.0% | **86.7%** |
| Handover Book | 40.0% | 70.0% | **93.3%** |
| Fold Cloth (A1) | 63.3% | 76.7% | **96.7%** |

五项任务全面领先，平均提升约 15–20 个百分点。

### 4.3 消融实验

| 消融 | 结论 |
|---|---|
| **2D Ablation**（去 3D 输入 + 去 3D 对齐） | 官方 LIBERO 降 ~5pp；RoboTwin 降 ~11pp；Modified LIBERO 从 89.9% → <10% |
| **训练 loss 曲线** | 2D 基线无法收敛吸收位姿变化，Dex-BEV loss 稳定下降 |
| **数据效率** | Fold Cloth 仅 400 条 demo 即超 X-VLA（~1500 条）；折纸盒 1500 条 ≈ 17 小时遥操作 |
| **OOD 定性** | 未见颜色/尺寸衣物零样本泛化；人工移动杯子后实时重规划；动态人手递接 |

---

## 五、局限性与展望

**作者自述局限：**

- 强依赖相机标定（外参 $\mathbf{T}_{t,i}$），非结构化环境中外参未知时部署困难；
- Vertex Spectrum 的 LID 离散化在亚厘米级精细操作中引入量化误差；
- 当前时间窗口较短，长时程任务语义记忆有限。

**附录补充的失败模式：**

- 硬件磨损导致抓取微滑；
- 固定基座下的运动学奇异/工作空间边界；
- 极端材质（丝绸、高反光）和剧烈光照变化；
- 人类遮挡导致感知退化。

**未来方向：**

- 扩展为 WAM（预测未来 3D BEV 状态）；
- 无标定 BEV lifting、3D 视觉基础模型在线标定；
- 轻量化边缘 VLA、长程记忆（Mamba 等）；
- 移动操控、多机器人协作、力触觉融合。

---

## 六、灵魂三问

1. **它解决了什么问题？** 之前 2D VLA 换个相机角度或桌子位置就几乎失效；Dex-BEV 把"看"和"动"都统一到 BEV 3D 空间后，在强相机/场景扰动下仍能保持约 90% 成功率，且一个模型跨两种机器人部署。

2. **为什么这么做？** 最关键的设计是选 **BEV 作为对齐帧** 而非各相机本地坐标系——BEV 俯视图对相机位姿变化天然鲁棒，且能同时承载多视角几何关联和动作输出；纯加点云/深度而不对齐动作坐标系，无法解决跨本体动作分布不一致的问题。

3. **什么证据最有说服力？** Modified LIBERO 实验：X-VLA 和 2D 消融成功率均 <10%，Dex-BEV 达 89.9%——这不是刷榜上的微小提升，而是直接证明了"3D 对齐"对视角/布局泛化的决定性作用；真实五项灵巧任务全面 +15~20pp 则证明不只仿真有效。

---

## 七、个人总结

1. **最核心 idea**：不要把 3D 当独立模态硬塞给 VLA，而是把观测（vertex map/spectrum + BEV 图）和动作（$SE(3)$ 位姿）都表达在**同一个 BEV 对齐坐标系**里，让 2D VLM 在"抬升后的 3D 像素"上继续工作。
2. **最大优势**：工程完整度高——不只有网络结构，还有可复现的时空对齐数据管线；跨本体单 checkpoint + 相机扰动鲁棒性在现有 VLA 中非常突出。**最大不足**：标定依赖重，Vertex Spectrum 精度有限，论文本质是"表征+数据对齐"框架而非全新学习范式。
3. **启发**：对你正在写的世界模型笔记（VAE 编码 → 动力学建模）有呼应——Dex-BEV 走的是另一条路：不做显式世界模型 roll-out，而是用 BEV 几何对齐让 VLA 隐式具备 3D 空间推理。若后续做 ego-centric 数据或跨平台部署，BEV 对齐 + 统一 TCP/时间归一化是值得借鉴的数据侧基础设施；与 π₀ 冻 action expert 训 VLM 的思路也可结合——Dex-BEV 的 3D 输入对齐本质上是在帮 VLM "看清空间"。

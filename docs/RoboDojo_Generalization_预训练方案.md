# RoboDojo Generalization预训练方案

本文整理RoboDojo Generalization的预训练思路，重点覆盖数据配方、动作接口和泛化训练方法。文中“目标”“建议”和“待验证”分别表示评测目标、可执行方案和仍需实验确认的假设。

## 1. 任务与主要问题

### 1.1 Benchmark设置

RoboDojo Generalization包含12个任务，测试集同时包含standard和random两类场景。random setting会改变背景、光照、桌面干扰物、物体布局和任务相关物体，而训练数据主要来自normal setting。

因此，预训练需要重点提升两种能力：

- **视觉泛化**：识别任务相关信息，忽略背景、光照和无关物体的变化。
- **空间理解**：理解物体之间、物体与末端执行器之间的相对位置和姿态。

目标是建立稳定的预训练基线，再通过受控消融验证各方法对Generalization的贡献。当前文档不直接记录SOTA结论，最终结果以统一评测脚本为准。

### 1.2 相关工作提供的启发

| 工作 | 可借鉴的思路 | 对本方案的启发 |
|---|---|---|
| X-VLA | 统一多来源、多机器人的动作表示，并平衡不同数据源 | 先统一动作接口，再按目标能力混合采样 |
| Spatial Forcing | 用3D teacher约束VLA的中间视觉表征 | 在精选数据上增加空间表征监督 |
| HyVLA | 用高质量操作轨迹形成通用action prior，并预测连续action chunk | 优先保证轨迹精度和动作连续性 |

## 2. 预训练数据方案

### 2.1 数据选择

优先选择与RoboDojo动作结构和任务形态接近的数据：

- 双臂操作，以及左右末端需要协同的任务；
- grasp、pick-place、push、stack、sorting；
- pouring、hanging、tool-use等对末端姿态要求高的技能；
- cloth等可变形物体操作。

数据质量比数据规模更重要。低质量轨迹、动作标注不准或控制空间差异过大的数据，可能稀释模型对RoboDojo精细操作的学习。egocentric数据可以保留，用于补充视觉和人手先验，但不宜在第一阶段占据主要比例。

数据字段、坐标系、轨迹清洗和Ego数据处理见[统一数据处理笔记](../Note_DataPipeline.md)；动作、旋转和归一化概念见[基础知识](../Note_Basics.md)。

### 2.2 统一动作表示

不同数据源尽量转换为相对末端动作：

~~~text
left EEF:  translation(3) + rotation-6D(6) + gripper(1)
right EEF: translation(3) + rotation-6D(6) + gripper(1)
~~~

如果模型同时使用joint和EEF动作，应在数据清单中明确每个维度的含义、有效mask和缺失处理方式。joint、位置和旋转通常使用相对当前状态的delta；gripper通常使用absolute值；旋转先转换为旋转矩阵，再计算相对旋转，不能直接逐元素相减。

32D与34D属于两个独立版本。两者的动作索引、归一化统计、dimension mask、动作头和部署脚本分别维护，不能混用。

### 2.3 轨迹清洗与重切分

建议沿用现有数据处理流程，并统一检查以下内容：

1. 删除图像缺失、时间戳不连续、动作异常跳变的片段；
2. 按固定观测频率重采样，保证图像、state和action时间对齐；
3. 按任务边界或固定长度切分轨迹，避免跨任务拼接；
4. 检查chunk末端是否超出轨迹，明确采用截断、padding或丢弃；
5. 保存数据源、机器人、视角、动作版本和归一化统计，便于复现实验。

### 2.4 保持语义不变的视觉增强

对同一物理状态生成多个视觉版本，只改变不影响正确动作的因素：

- 背景、桌面材质和光照；
- 适度模糊、噪声和遮挡；
- 物体纹理或颜色的小幅变化。

增强不能改变目标物位置、可达性、遮挡关系或动作标签。几何敏感任务应限制增强强度，并保留一部分未增强样本作为稳定基线。

### 2.5 异构数据混合采样

为每类数据配置采样权重，使训练分布贴近目标能力。第一阶段可以提高双臂、EEF精度高和任务结构相近的数据权重，降低弱相关或视角差异过大的数据权重。

当batch size为128、双臂操作与egocentric的权重为0.8和0.2时，每个batch约采样102至103条双臂数据和25至26条egocentric数据。实际实现应使用统一的取整和补差规则，避免不同worker产生不一致的比例。

采样权重需要作为实验变量记录，并至少比较：

| 配方 | 双臂操作 | egocentric | 目的 |
|---|---:|---:|---|
| A | 1.0 | 0.0 | 检查目标域动作能力 |
| B | 0.8 | 0.2 | 兼顾动作和视觉先验 |
| C | 0.6 | 0.4 | 检查更多Ego数据是否改善视觉泛化 |

## 3. 多视角输入

主视角适合承担场景和语言理解；手腕视角视野小、变化大且容易被遮挡。一个可行方案是主视角完整经过VLM，手腕视角先经过视觉编码器，再将视觉特征提供给动作分支。

该视角分工需要通过消融确认，不能直接假设适用于所有配置。至少比较主视角、主视角加手腕视角、两路都经过VLM和手腕视角只经过视觉编码器四种设置。X-VLA中的具体实现细节仍需结合代码和论文进一步核对。

## 4. RoboDojo数据与训练边界

RoboDojo任务数据按官方配置准备，在微调和测评阶段使用。预训练阶段可以使用同分布数据学习动作结构，但需要区分训练数据和最终测试任务，避免测试场景泄漏。

评测时固定以下条件：

- standard和random任务列表；
- 每个任务的测试次数和成功标准；
- checkpoint、动作版本和归一化统计；
- 是否启用多视角、Horizon normalization和World Expert。

评测指标、归一化和动作接口的基础定义见[基础知识](../Note_Basics.md#模型评测从训练loss到闭环任务)。

## 5. 泛化训练方法

### 5.1 Spatial Forcing：增强3D空间表征

Spatial Forcing使用冻结的3D foundation model作为teacher，将VLA中后层的visual tokens与teacher的空间表征对齐。对于同一图像patch，学生表征被约束为更接近带有几何信息的teacher表征：

~~~text
L = L_action + alpha * L_spatial
~~~

建议在backbone深度约70%至80%的位置对齐，并让alpha从小到大warmup。teacher只在训练时生成目标，推理时不参与计算，因此不会增加推理输入。

预训练阶段不建议让所有视频帧都经过VGGT。可以抽取高质量、包含精细操作的子集，优先保留主视角，在预训练后段进行空间对齐。这样可以控制计算成本，同时让模型获得稳定的3D视觉先验。子集大小、采样方式和对齐层需要通过消融确定。Dynamic/SGM中World Expert和teacher监督的实现见[Dynamic/SGM原理](../MagicAtom/04_Dynamic模型/01_MagicVLA_Dynamic_SGM.md)。

### 5.2 一致性学习：忽略无关变化

对物理状态和正确动作不变的normal/randomized视图对增加一致性约束：

~~~text
L_consistency = d(mid_normal, mid_random)
                 + lambda * D(action_normal, action_random)
~~~

其中，d可以使用中间visual-language tokens的L2距离或cosine distance，D约束两种视图下的动作预测接近。该损失只适用于目标物位置、可达性和动作标签不变的视图对，不能用于改变物理状态的增强样本。

### 5.3 Grounding辅助监督：找对目标

利用仿真metadata对目标物增加轻量辅助监督，减少模型选错对象的情况。可选目标包括：

- 目标中心或目标区域；
- 目标相对末端的3D位置；
- left/right、inside/on、near等语义关系。

辅助任务只提供视觉定位约束，动作loss仍是主目标。metadata不完整时应关闭对应mask，不能用默认值伪造标签。

### 5.4 联合损失

~~~text
L_total = L_action
        + alpha * L_spatial
        + beta  * L_consistency
        + gamma * L_grounding
~~~

L_spatial只在精选子集上启用，其余样本权重为0。alpha、beta和gamma需要warmup或从较小值开始，避免辅助任务压过动作学习。每项loss都应记录有效样本数和mask比例。

## 6. 训练顺序与实验计划

建议分三步推进：

1. **动作基线**：固定32D或34D中的一个版本，只使用清洗后的目标域数据，确认动作接口、归一化和评测链路。
2. **数据消融**：比较不同数据源比例、Ego数据比例、视角组合和视觉增强强度。
3. **泛化方法**：在最佳数据配方上依次加入Spatial Forcing、一致性学习和grounding监督，单独记录计算成本和收益。

每次实验至少记录代码commit、动作版本、数据清单、采样权重、视角配置、chunk设置、归一化方式、loss权重、训练步数和评测结果。结果需要区分“代码支持”“配置启用”“训练完成”和“效果验证”。

## 7. 风险与待确认项

- 32D和34D的动作契约、统计量和checkpoint必须隔离；
- Horizon normalization、delta/rotation变换必须与动作版本绑定；
- Spatial Forcing的teacher帧数和显存成本需要先做小规模估算；
- 一致性增强可能误伤遮挡、接触和可变形物体等几何信息；
- Ego数据的手人映射和控制空间差异需要单独评估；
- 多视角分工、采样比例和loss权重都属于实验假设，不能直接当作最终配置。

## 8. 参考资料

- [Spatial Forcing](https://arxiv.org/abs/2510.12276)
- [VGGT](https://arxiv.org/abs/2503.11651)
- [DPT头](https://arxiv.org/abs/2103.13413)
- [Hy-Embodied-0.5-VLA](https://arxiv.org/abs/2606.14409)
- [X-VLA](https://arxiv.org/abs/2510.10274)
- [RoboDojo](https://arxiv.org/abs/2607.04434)

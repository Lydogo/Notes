# 具身数据处理与评估

围绕时间对齐、动作表示、几何变换、质量控制与规模化训练，整理Pico和魔法原子数据管线中的关键方法。通用原理见[基础知识](Note_Basics.md)，项目背景见[项目笔记](Note_Project.md)。

- [Pico：第一视角数据处理](#data-chaowei)
- [魔法原子：异构数据统一](#data-magicatom)
- [两条管线对比](#data-comparison)
- [数据验收与训练接口](#data-validation)
- [质量控制与语义标注](#data-quality)
- [评估集与模型评测](#data-evaluation)
- [规模化处理与Dyna经验](#data-dyna)

<a id="data-chaowei"></a>

## 1. Pico：第一视角数据处理

### 1.1 流程与输出

输入为视频、手部tracking、相机参数、片段标注和可选质检报告，输出LeRobot v2.1：视频、20D state/action、任务及episode元数据。

```text
原始会话 → 质检报告筛选 → tracking时间补偿
→ tracking转HDF5 / 视频去畸变 → 按任务切片与时间匹配
→ 双手20D表示 → Parquet、视频和meta → 数据集合并
```

此管线保留人手运动监督，不执行人手到机器人joint的IK映射。

### 1.2 时间对齐

视频与tracking来自不同采集链路。入口默认补偿0.14秒，这是特定批次的配方。切片后，每个视频帧通过二分查找匹配最近的tracking记录，边界取首尾记录；最近邻匹配不包含轨迹插值。

| 操作 | 时间语义 |
|---|---|
| 按真实时间重采样 | 保留物理时长，调整采样点 |
| 保留帧数、重设FPS | 重写播放时间，可能改变物理时长 |

视频切片使用`setpts=N/fps/TB`，表格时间戳可由`arange(N)/fps`生成。改变FPS时必须核对帧数、时长、PTS及抓取/释放事件的图像与轨迹叠加结果。

同步误差分为固定偏移、时钟漂移、抖动和丢帧。可用独立运动事件验证补偿；静止或周期运动下，相关性峰值可能有歧义。可接受误差取决于动作速度和接触精度，不宜统一设定毫秒阈值。

### 1.3 20D表示与开合标签

每只手为`xyz(3) + rotation-6D(6) + gripper(1)`，双手合计20D。Rotation-6D取旋转矩阵前两列并按列展开，编码和解码顺序须一致；delta变换由训练侧决定。

默认曲率开合映射：

```text
有效关节帧 → Butterworth低通 + Savitzky–Golay平滑
→ 手指链平均曲率 → 高分位裁剪 → log1p → Otsu阈值
→ 高曲率闭合0，低曲率张开1
```

单一手形序列可能无法形成可靠阈值。有效帧不足10帧及无效帧的默认张开值属于兜底，不能视为有效测量；离线平滑也不等同于实时因果滤波。

### 1.4 质检与标签时间

管线读取已有`quality_inspection.json`，进行会话硬过滤和0～5质量分级，并将quality、skill、hand、target_object写入任务信息。报告缺失时跳过该检查；质量标签是否影响采样或loss，由训练侧决定。

`_assemble_training_data()`实际执行`actions = obs.copy()`，并未在此生成下一帧目标。最终监督时间需结合训练reader的偏移确认。

数据合并时同步重编号episode、frame和task。视频采用软链接时，合并结果仍依赖源路径。

代码：[调度入口](../kai_work/pico_egadata_pipline/pico_egadata_pipline-master-e019270fb57841410c22e780fae78604cb113e0c/run_pipeline.py)、[时间匹配](../kai_work/pico_egadata_pipline/pico_egadata_pipline-master-e019270fb57841410c22e780fae78604cb113e0c/scripts/02_video_hdf5_segment.py)、[20D与开合转换](../kai_work/pico_egadata_pipline/pico_egadata_pipline-master-e019270fb57841410c22e780fae78604cb113e0c/scripts/03_hdf5_to_parquet.py)、[质量筛选](../kai_work/pico_egadata_pipline/pico_egadata_pipline-master-e019270fb57841410c22e780fae78604cb113e0c/scripts/quality_filter.py)。

<a id="data-magicatom"></a>

## 2. 魔法原子：异构数据统一

### 2.1 框架与动作契约

```text
Source Reader → 统一episode → Checks → Transforms → Writer
```

Reader解释源字段，Checks检测异常，Transforms负责修复、几何和动作变换，Writer维护媒体、索引及元数据。具体步骤由配置启用。

| 来源 | 监督与处理 |
|---|---|
| EgoDex | 手部关键点构造virtual-hand EEF与开合量，无真实arm joint |
| Hy-UMI | EEF、夹爪状态/命令、相机估计及IK派生joint |
| 真机 | 区分实测state和控制command，核对运动学及相机参数 |
| 仿真 | 读取状态、动作和相机信息，核对坐标及控制约定 |

统一槽位时，缺失维度补零并关闭mask；真实零值仍有效。state、action、相机和EEF可见性分别定义有效性，并保留实测、命令、IK推导或模型预测等标签来源。

- **32D：** 双臂各6 joint + gripper，共14D；再加双手18D EEF。
- **34D：** 双臂各预留7 joint + gripper，共16D；再加双手18D EEF。

落盘格式与训练接口可以不同：既有32D→34D映射，也有直接生产34D的配置；Hy-UMI配方可将14D joint/gripper与18D EEF分列保存。

### 2.2 坐标、TCP与retarget

| 操作 | 含义 |
|---|---|
| 参考系变换 | 同一物理位姿从world表达转到camera表达 |
| 局部轴/TCP对齐 | 改变末端工具坐标定义，通常右乘固定变换 |
| Retarget | 将示教运动映射为目标机器人的动作表示 |

令`T_W_C`表示camera到world的变换，`T_W_E`表示EEF到world的变换，则：

```text
当前状态：T_C(t)_E(t)   = inverse(T_W_C(t)) @ T_W_E(t)
下一步目标：T_C(t)_E(t+1) = inverse(T_W_C(t)) @ T_W_E(t+1)
```

移动相机下，state和未来target应共用当前相机参考系。目标有效性随时间偏移同步更新。相机外参用于视觉几何，虚拟机器人base用于可达性求解，两者独立。

### 2.3 EgoDex：virtual-hand与连续开合

从手掌、指根和指尖构造EEF位置及正交轴；拇指尖与食指尖距离映射为连续开合量：

```text
gripper = clip((distance - d_closed) / (d_open - d_closed), 0, 1)
```

代表配置使用1 cm和12 cm作为闭合/张开端点。该值是开合代理信号，不是夹爪电机角度，公式本身也不包含滤波或滞回。

ActionPostprocess生成next-step、转换到当前相机系、编码rotation-6D并pack。普通生产配置可裁剪首尾低信息段；deferred-media配置保留源帧区间，先写表格和媒体清单，再生成视频。

### 2.4 Hy-UMI：相机参数估计

每帧仅有左右设备的3D位姿，图像对应点由设备区域估计，焦距、相机位姿和设备偏置存在耦合。

1. 用多帧灰度中值估计背景，在桌面区域提取运动/暗色候选，经形态学处理得到blob。
2. 尝试左右设备的两种匹配，选择重投影误差较小者。
3. 联合优化15个变量：相机位姿6D、共享焦距1D、主点2D、左右设备偏置6D。
4. 以最小60%残差的trimmed mean为目标，从已验证seed进行多次Nelder–Mead优化。
5. 使用table级共享先验，再按session/batch约束细化，并用留出数据与跨任务叠加图检查。

Device offset补偿跟踪原点与blob质心的差异，仅用于拟合，不写入EEF标签。项目记录中重投影中位误差约41 px、held-out约37 px；这是近似几何估计，验证时还需看误差尾部、遮挡及跨session稳定性。

### 2.5 Hy-UMI：EEF到ARX5

```text
UMI world位姿 → 局部轴对齐TCP → 固定虚拟base
→ 双臂DLS IK → 关节限位、连续性与碰撞诊断
```

同一校准组固定base，避免joint语义随episode漂移。EEF保留送入IK的目标，joint由IK生成，FK残差用于检查二者一致性，不反写覆盖目标EEF。

本地`Magiclab_0917`生产配置的关键约定：

| 项目 | 配置语义 |
|---|---|
| 媒体 | 30 FPS、424×240、无底部补边；旧方案中的25 FPS、424×256不适用此配方 |
| 输出 | 14D `action`与18D `camera_coordinate.state/action`分列 |
| 动作时间 | joint取next-step；gripper从`observation.gripper_command`打包 |
| IK约束 | 位置/姿态容差0.03 m/20°，单步joint变化上限0.12 rad |
| 过滤 | 未声明顶层`checks`；碰撞检测开启，但IK/碰撞失败未设为硬过滤 |

因此，成功写出、IK求解成功、实机任务成功是三个不同条件。

实现：[调度](../Magiclab/Magiclab_0917/pretrain_data_pipeline/src/pretrain_data_pipeline/pipeline/runner.py)、[人手EEF](../Magiclab/Magiclab_0917/pretrain_data_pipeline/src/pretrain_data_pipeline/data_io/eef_converter.py)、[ActionPostprocess](../Magiclab/Magiclab_0917/pretrain_data_pipeline/src/pretrain_data_pipeline/transforms/action_postprocess/pipeline.py)、[Hy-UMI方案](../Magiclab/Magiclab_0917/pretrain_data_pipeline/docs/HY_UMI_CAMERA_ARX5_LEROBOT_V21_PIPELINE.md)、[生产配置](../Magiclab/Magiclab_0917/pretrain_data_pipeline/configs/run/production/hy_umi_table000_arx5_v21.yaml)。

<a id="data-comparison"></a>

## 3. 两条管线对比

| 维度 | Pico | 魔法原子 |
|---|---|---|
| 重点 | 采集时序、任务切片、人手标签 | 多源语义统一、几何适配与生产 |
| 动作表示 | 双手20D | 32D/34D契约，按源配置落盘 |
| 开合 | 曲率阈值，二值 | virtual-hand使用指尖距离，连续 |
| 质量控制 | 会话过滤、评分、任务标签 | episode/frame/dimension检查与mask |
| 目标时间 | 转换函数复制state，需结合reader | 支持显式next-step，部分字段单独处理 |
| 组织方式 | 分阶段脚本 | 配置驱动的Reader/Checks/Transforms/Writer |

<a id="data-validation"></a>

## 4. 数据验收与训练接口

```text
落盘数据 → 有效anchor与未来chunk → 槽位映射
→ delta/旋转变换 → normalization → 图文与mask → loss
```

验收包括五项：

1. **媒体与索引：** 视频可解码，帧数、时间戳、Parquet行和episode/task对应。
2. **物理语义：** EEF投影、局部轴、单位、开合方向与关键交互事件一致。
3. **监督时间：** 明确state、command、next-step和chunk偏移，避免重复shift或delta。
4. **有效性：** 缺失维度不参与相应统计/loss；裁剪同步更新所有字段，chunk不跨episode或删帧断点。
5. **归一化：** 统计量匹配source、动作定义、维度和horizon，只由训练集计算。

Horizon normalization按`[H,D]`统计各预测位置的尺度，属于训练目标处理。见[逐时刻归一化](MagicAtom/03_Action表示/01_ActionChunk_逐时刻归一化.md)。

排查训练异常时，先将一个batch逆变换回物理量并叠加到视频，再检查模型与部署配置。仿真到真机的差异按观测、执行、接触分解：相机/时序，控制频率/延迟/TCP，以及摩擦/柔顺性/碰撞几何。

<a id="data-quality"></a>

## 5. 质量控制与语义标注

### 5.1 检测、修复与筛选

质量应按模态和用途记录，单一总分不能代替有效性判断。

| 异常 | 处理原则 |
|---|---|
| 缺失joint或相机 | 对应mask关闭，保留其他有效监督 |
| 局部尖峰 | 在可恢复条件下修复，记录范围与方法 |
| 首尾低信息段 | 可裁剪，更新媒体和索引 |
| 中间静止、回退、重复 | 区分接触保持、无效等待与失败恢复，保留连续上下文 |
| 图像正常、动作异常 | 停用异常动作监督；图文单独验收后再决定用途 |
| 图文动作冲突 | 排查错绑、错标和执行失败，保留原始指令与结果 |
| 严重损坏或证据不足 | 隔离或标记unknown，不补造标签 |

删除中间片段后不能直接拼接两侧轨迹，否则会制造动作跳变。失败恢复片段可保留，但需标明过程与结果，不能自动视为成功示教。

### 5.2 标注与抽检

自动标注可由视频帧、片段描述和动作事件生成skill、物体、左右手及交互关系候选，再做结构校验和人工复核。调用模型生成的标签属于候选监督，不能直接充当评估金标。

阶段边界应绑定可观察事件，例如接触、抓稳、释放；闭合命令与实际抓稳须分开。评估阶段类别、对应边界时间偏差、片段时域交并比及漏段/多切，预先固定匹配规则和容忍窗口。

抽检分为两类：

- **概率抽检：** 覆盖主要数据层，包含未告警样本，用于估计总体质量和漏检。
- **风险抽检：** 优先新来源、低置信度、模型/规则分歧和异常批次，用于定位问题；错误率不可直接外推总体。

以异常为正类，报告precision、recall；误删率为错误删除的合格样本数/全部合格样本数。标注分歧通过原始证据仲裁，并检查规范、工具时间轴和可见性；多数票不一定是真值。

参考：[Data Check](../Magiclab/Magiclab_0917/pretrain_data_pipeline/docs/DATA_CHECK.md)、[Video Skill Tagger](../kai_work/dataset_video_skill_tagger/dataset_video_skill_tagger-master-18b29097b9faabd663efa09c36aff424c049b18f/README.md)。

<a id="data-evaluation"></a>

## 6. 评估集与模型评测

### 6.1 检索、覆盖与数据划分

筛选目标数据时，先定义完整轨迹/片段及有效条件，再按元数据和视觉信息召回、交互证据精筛、去重、分层抽样。物体出现在背景中不等于发生目标操作；有效独立轨迹不足时不能靠重复切片凑数。

按原始session/轨迹划分训练与测试，派生切片不可跨集合。覆盖维度包括任务、物体实例/材质、姿态、遮挡、光照、操作者和机器人；多技能任务还需覆盖阶段衔接、长程依赖及恢复。

业务集保持实际分布，挑战集增加困难条件并单独报告。分层过采样后的业务总分可用`S = Σ p_k × S_k`汇总，`p_k`为真实业务占比；权重无法补救缺失层或层内选样偏差。

保留固定核心集、新失败回归集和未参与调参的盲测集。更新版本时用相同模型同时评估新旧集合；已用于训练的失败样本不再作为独立泛化证据。

### 6.2 指标与成功定义

| 层次 | 主要指标 |
|---|---|
| 完整性 | 解码通过率、帧/行不一致、模态缺失率 |
| 时序与几何 | 时间残差、事件偏移、重投影误差分位数、跨session稳定性 |
| 动作 | 有效维度、跳变/冻结率、开合事件错误、IK/FK残差与碰撞率 |
| 语义 | 逐类precision/recall、macro-F1、边界误差 |
| 分布 | 任务覆盖、重复率、过滤前后长尾占比 |
| 闭环 | 任务/阶段成功率、首次与最终成功率、干预、重试、超时及耗时 |

成功需预先定义终态和硬约束。例如抓取搬运要求正确物体到达目标区域并稳定，夹爪闭合本身不证明成功。目标达成、约束违反、人工介入和重试分开记录；违反硬约束时，完整任务判失败。

多次自主重试仍属于同一episode，不能拆开改变分母。多条合理路径按终态、约束与成本评估，不要求逐点逼近唯一示教。报告成功次数/尝试次数、任务聚合方式及不确定性；耗时同时给出成功率和超时率。

### 6.3 数据策略验证

```text
失败现象 → 数据问题假设 → 检测/修复/筛选
→ 人工审计 → 同预算训练对照 → 闭环评估
```

| 对照 | 主要控制条件 |
|---|---|
| 时间修正前后 | 相同轨迹、表示、模型和预算 |
| 随机子集与质量筛选 | 可比样本量、任务分布和训练步数 |
| 硬过滤与质量加权 | 相同候选池，记录实际采样分布 |
| EEF-only、joint-only、联合监督 | 相同划分，明确loss权重与执行解码 |

离线动作误差按source及物理单位报告，不直接横比不同归一化尺度的loss。离线提升而闭环无提升时，检查数据泄漏、简单任务占比、指标与任务目标的关联，以及训练—部署差异。

<a id="data-dyna"></a>

## 7. 规模化处理与Dyna经验

### 7.1 Dyna百万小时管线

Dyna的[基础设施报告](https://www.dyna.co/research/dyna-2-infrastructure)围绕百万小时数据，采用以下设计：

| 环节 | 方法 |
|---|---|
| 存储 | MCAP与视频压缩，按读取模式分组相机、状态/动作topic |
| 处理 | Airflow DAG、持久化状态、版本追溯和选择性重处理 |
| 调度 | 错峰启动、按字节均衡分批 |
| 筛选 | 生产库与分析仓库分工，以SQL生成manifest |
| 加载 | 本地下载、内存映射、按rank切分 |
| 训练供给 | NVMe缓存与预热、节点内优化器分片、故障恢复 |

报告自报吞吐由每周1.4万增至44万episode-hours，属于整体管线优化结果。[配套模型报告](https://www.dyna.co/dyna-2)使用手部轨迹生成伪动作监督，并通过保持来源比例的嵌套子集和固定验证集研究规模效应；动作监督量与额外视频监督的作用分别考察。

### 7.2 存储、调度与版本

通用设计应从训练样本的访问方式出发：相机数、历史/未来帧、action horizon和采样频率决定读取需求。存储chunk、视频GOP与action chunk分别属于存储、编码和监督组织。

- **访问效率：** 测量读取字节/有效字节、解码帧/使用帧，兼顾随机访问延迟、画质和存储量。
- **资源调度：** 按字节、时长、模态数或历史成本分批，有界并行；同时检查排队、长尾任务和存储限流。
- **版本依赖：** 以源身份、输入版本、配置和代码版本标识产物。caption更新通常不需重编码视频；标定更新可能使EEF、IK和统计量失效。
- **训练清单：** manifest固定源session、区间、步数、任务、质量、模态和版本。deferred media需单独标记视频就绪状态。
- **恢复：** 保留阶段状态和失败记录，续跑时保证无重复、无遗漏、不混用旧版本。

具体格式与工具应通过现有访问模式测试，不能只凭数据规模决定替换LeRobot或引入大型调度系统。

### 7.3 性能验证

分别测量处理吞吐、manifest构建、首batch时间、稳态等待、解码、传输和训练计算；多rank任务关注最慢rank。缓存需报告冷/热状态、预热成本和命中率。

优化前后固定数据及训练条件，检查样本ID、时间、动作、mask和采样分布；媒体重编码另验画质。最终同时报告有效产出、耗时、硬件、并发、成本与恢复时间，避免通过跳过质检或减少监督获得表面加速。

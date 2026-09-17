# 数据处理：超维动力与魔法原子

本文说明两条pipeline如何把原始数据变成训练样本，重点是时间、坐标、动作和质量的处理。项目概述见[项目笔记](Note_Project.md)，通用原理见[基础知识](Note_Basics.md)。

- [超维：Pico采集数据到20D训练样本](#data-chaowei)
- [魔法原子：多源数据统一与生产](#data-magicatom)
- [两条pipeline的区别](#data-comparison)
- [数据验收与训练衔接](#data-validation)

<a id="data-chaowei"></a>

## 1. 超维：Pico采集数据到20D训练样本

### 1.1 输入与目标

仓库：[ego-centric_data_pipeline](../ego-centric_data_pipeline)。输入按采集会话组织，主要包含视频、手部tracking、相机参数、片段标注，以及可选的质检报告。

目标是生成LeRobot v2.1数据：视频保留第一视角，Parquet保存双手20D状态与动作，meta记录任务、episode和帧率等信息。该仓库处理Pico人手数据，不负责把人手轨迹通过IK转换为机器人joint。

```text
原始会话
→ 读取质检报告，硬过滤与质量分级
→ tracking时间补偿
→ tracking转HDF5 / 视频去畸变
→ 按任务标注切片，视频帧匹配tracking
→ 手腕位姿与gripper转换为20D
→ Parquet、视频、任务和episode元数据
→ 可选合并
```

### 1.2 时间对齐：先匹配，再组织样本

视频和tracking来自不同链路，即使描述同一次运动，时间戳也可能存在偏移。入口默认使用0.14秒补偿参数；这是处理配方，不是所有设备通用的延迟常数。

切片根据标注的起止时间选择视频帧，再为每帧寻找时间最近的tracking记录。代码通过二分查找比较左右邻居，边界取首尾记录。这是最近邻匹配，不是轨迹插值；匹配误差过大时仍可能出现图像与动作错位。

输出阶段还会统一时间戳。这里需要区分两件事：

| 操作 | 含义 |
|---|---|
| 按真实时间重采样 | 保留物理时长，改变或插值采样点 |
| 保留帧数、重设FPS | 重写各帧播放时间，可能改变整段时长 |

当前视频切片使用`setpts=N/fps/TB`，Parquet可使用`arange(N)/fps`。若目标FPS与源FPS不同，不应仅理解为格式变化，还要检查动作时间尺度。验收时同时看帧数、时长、PTS和动作转折点的overlay。

基础：[标定与时间同步](Note_Basics.md#basic-calibration)。

### 1.3 从tracking得到20D表示

每帧左右手tracking保留关节位置与姿态。转换时提取指定手腕位姿，将四元数转换为rotation-6D，再加入开合标签：

| 每只手 | 维度 | 含义 |
|---|---:|---|
| xyz | 3 | 手腕位置 |
| rotation-6D | 6 | 旋转矩阵前两列，按列展开 |
| gripper | 1 | 根据手指形态推断的开合标签 |
| 双手合计 | 20 | 左手10D + 右手10D |

Rotation-6D的编码与解码必须使用相同展开顺序。这里生成的是姿态表示，训练中的xyz delta、旋转absolute等规则属于后续模型输入变换。

基础：[旋转表示](Note_Basics.md#basic-rotation)、[动作表示与归一化](Note_Basics.md#basic-normalization)。

### 1.4 曲率gripper：从手指弯曲判断开合

默认转换路径使用`make_curvature_gripper_mapper()`，按手指链的弯曲程度判断抓握：

```text
手部关节位置
→ 筛选非零有效帧
→ Butterworth低通与Savitzky–Golay平滑
→ 计算手指链平均曲率
→ 按高分位裁剪异常尖峰
→ 在log1p曲率空间用Otsu求阈值
→ 高曲率闭合0，低曲率张开1
```

这条路径输出二值标签。它利用多关节形态，和“拇指尖—食指尖距离映射连续开合量”是不同方案。平滑用于降低tracking抖动，Otsu用于按输入序列的曲率分布自动分组，也支持显式阈值。

局限是：序列若几乎只有一种手形，自动阈值未必对应真实开合边界。代码在有效帧不足10帧时返回全张开，无效帧也默认张开；这是兜底值，不能当成可靠测量。离线滤波可以利用前后帧，也不能直接等同于实时因果滤波。

### 1.5 质量控制：先排除不可用数据，再保留质量差异

pipeline读取已有`quality_inspection.json`进行判定，不应把它描述为重新执行了所有原始检测器。报告不存在时，入口会跳过这一步质量检查。

- **硬过滤：** 排除严重缺失、标注异常等不适合继续处理的会话。
- **软评分：** 从视觉、动作、时序和内容等维度汇总质量，映射为0～5。
- **语义记录：** 将quality、skill、hand、target_object等信息写入task prompt。

质量标签提供筛选或训练条件，但写入标签不等于已经按质量加权采样。是否过滤某一级、是否改变采样概率，需要在下游训练中另行决定。

### 1.6 action语义与数据组织

读取代码时要区分注释和执行逻辑：`_assemble_training_data()`注释写“下一帧action”，但函数内实际使用`actions = obs.copy()`。因此不能直接把此处描述为已生成next-step target；应结合训练reader的取样偏移解释最终监督时间。

Parquet、视频和meta通过episode、frame和task索引对应。合并数据集时需要同步重编号，不能只把文件拼到同一目录。当前合并脚本对视频使用软链接，节省空间，但合并结果仍依赖源视频路径。

### 1.7 代码入口

| 模块 | 代码 |
|---|---|
| 调度与质检入口 | [run_pipeline.py](../ego-centric_data_pipeline/run_pipeline.py) |
| 时间补偿 | [00_correct_tracking_time.py](../ego-centric_data_pipeline/scripts/00_correct_tracking_time.py) |
| episode切片与时间匹配 | [02_video_hdf5_segment.py](../ego-centric_data_pipeline/scripts/02_video_hdf5_segment.py) |
| 20D表示、曲率gripper与action组装 | [03_hdf5_to_parquet.py](../ego-centric_data_pipeline/scripts/03_hdf5_to_parquet.py) |
| 硬过滤与软评分 | [quality_filter.py](../ego-centric_data_pipeline/scripts/quality_filter.py) |

<a id="data-magicatom"></a>

## 2. 魔法原子：多源数据统一与生产

### 2.1 统一框架，而不是为每个数据源重写流水线

仓库：[pretrain_data_pipeline](../pretrain_data_pipeline)。输入覆盖人手、UMI、机器人和仿真数据；不同来源的格式、相机、动作定义各不相同。

```text
Source Reader → 统一episode结构
              → Checks / Transforms
              → Writer → LeRobot数据
              → 验收、合并、统计与训练接入
```

Reader负责读取源数据并解释字段；中间episode结构携带帧、任务、媒体和状态动作；Transform负责公共变换；Writer负责索引、元数据和媒体写出。执行顺序由配置决定，不能假定所有数据源都经过相同的retarget或修复步骤。

这种分工让新增数据源主要集中在Reader和源配置，坐标处理、质检、写盘等能力可以复用。

### 2.2 统一动作接口，不强行补造测量

不同源可以提供不同监督：

| 来源 | 主要处理 |
|---|---|
| EgoDex | 关键点构造virtual-hand EEF与开合量，无真实arm joint |
| Hy-UMI | EEF轨迹、相机参数估计，以及按配方进行机器人IK retarget |
| 真机 | 保留joint命令/状态；根据已有位姿或运动学获得EEF |
| 仿真 | 读取joint、EEF及相机信息，核对坐标约定 |

统一接口依靠槽位映射与mask。不存在的维度补零，但必须标为无效；真实零值仍可以有效。state、action、EEF可见性和相机有效位各有自己的语义。

32D与34D分别记录：32D每臂6个joint加gripper，34D每臂预留7个joint加gripper，两版再各加双手18D EEF。数据可以按32D落盘、在训练时映射到34D，也存在直接生产34D的源配置；不能把一种路径套到所有数据源。

### 2.3 坐标变换、局部轴与retarget是三件事

| 操作 | 改变什么 | 例子 |
|---|---|---|
| 参考坐标系变换 | 同一物理位姿的表达坐标 | world EEF转到cam_high |
| EEF局部轴/TCP对齐 | 位姿代表的工具坐标定义 | 手部approach轴对齐机器人工具轴 |
| Retarget | 示教运动到目标机器人的可执行表示 | UMI EEF经IK生成ARX5 joint |

相机系表达使用`T_C_E = inverse(T_W_C) @ T_W_E`；局部轴调整通常右乘固定变换。两者作用对象不同，不能用一个含义不明的“旋转补偿”概括。

Retarget还要检查工作空间、关节限位、碰撞和轨迹连续性。相机外参与虚拟机器人base位置也是不同参数：前者用于视觉几何，后者用于求解可达动作。

Hy-UMI的具体标定与ARX5设计见[项目数据章节](Note_Project.md#magic-data)。基础见[坐标与相对位姿](Note_Basics.md#basic-rotation)、[IK](Note_Basics.md#basic-ik)。

### 2.4 EgoDex：连续开合量与virtual-hand

virtual-hand根据手掌、指根和指尖构造EEF位置与正交轴，开合量则单独使用拇指尖与食指尖距离：

```text
gripper = clip((distance - d_closed) / (d_open - d_closed), 0, 1)
```

代表生产配置使用1 cm和12 cm作为闭合、张开端点，中间线性映射。它输出连续标量，公式本身没有时序滤波或滞回；该标量也不是直接的夹爪电机角度。

World-frame统一与局部轴对齐后，ActionPostprocess依次生成next-step target、转换到当前相机系、编码rotation-6D并pack。对移动相机，下一时刻EEF应表示为：

```text
T_C(t)_E(t+1) = inverse(T_W_C(t)) @ T_W_E(t+1)
```

这样state和target共用当前相机参考系。可见性也随next-step一起移动，不能把当前帧mask直接当作下一步目标mask。

### 2.5 质量检查、修复与筛选

数据异常不应统一处理：

| 情况 | 合理处理方向 |
|---|---|
| 不存在某个joint或相机 | 保留样本，关闭对应mask |
| 局部尖峰或短暂不连续 | 在条件允许时修复，并保留质量记录 |
| 首尾低信息片段 | 按配置裁剪，同时更新视频与索引 |
| 中间静止或遮挡 | 判断任务语义，不能一概删帧 |
| 媒体损坏、严重无效episode | 拒绝写出或过滤 |

检查覆盖图像可用性、EEF可见性、运动连续性、静止/冻结和场景运动等。处理前后的overlay用于检查修复是否合理；仅看统计指标容易漏掉轴方向或视频错位问题。

EgoDex普通生产与deferred-media版本还存在差异：前者可裁剪边界段，后者的当前配置保留源帧区间，以便后续媒体物化。它们是两种生产配方，不应混写为固定流程。

### 2.6 大规模生产与追溯

- **流式处理：** 以episode为单位读取、变换和写出，避免一次载入全量数据。
- **有界并行：** 限制在途任务和媒体工作量，控制内存并保持输出组织稳定。
- **恢复与追溯：** 保存处理进度、源episode身份和配置，避免中断后重复写入或混用版本。
- **Deferred media：** 先写表格、元数据和媒体清单，再物化视频；清单完成不等于数据已可直接训练。
- **可视化验收：** 展示轨迹、坐标轴和投影，辅助判断标定与动作处理是否符合物理含义。

当前统一工具入口还按run、ops、fleet、calib等任务组织配置与产物记录。笔记理解这些职责即可，具体命令以仓库文档为准。

### 2.7 代码入口

| 模块 | 代码 |
|---|---|
| 流水线调度 | [runner.py](../pretrain_data_pipeline/src/pretrain_data_pipeline/pipeline/runner.py) |
| 人手EEF与开合量 | [eef_converter.py](../pretrain_data_pipeline/src/pretrain_data_pipeline/data_io/eef_converter.py) |
| next-step、相机系与pack | [ActionPostprocess](../pretrain_data_pipeline/src/pretrain_data_pipeline/transforms/action_postprocess/pipeline.py) |
| EgoDex生产配置 | [普通版本](../pretrain_data_pipeline/configs/run/production/egodex_virtual_hand_quality_production.yaml)、[deferred版本](../pretrain_data_pipeline/configs/run/production/egodex_virtual_hand_quality_production_120core.yaml) |
| 操作与配置导航 | [仓库README](../pretrain_data_pipeline/README.md) |

<a id="data-comparison"></a>

## 3. 两条pipeline的区别

| 维度 | 超维Pico | 魔法原子多源预训练 |
|---|---|---|
| 主要问题 | 原始采集数据的时间、片段和标签整理 | 异构来源的表示统一与规模化生产 |
| 动作接口 | 双手20D | 按配方使用32D或34D |
| 人手gripper | 默认曲率阈值，二值开合 | virtual-hand分支按指尖距离，连续开合 |
| 时间处理 | 延迟补偿、最近邻匹配、切片与时间戳重建 | 按源时间语义处理，再与训练chunk对接 |
| 质量组织 | 会话硬过滤、评分与prompt标签 | episode/frame/dimension等层次的检查与mask |
| 标签时间 | 当前组装函数复制state；需结合训练reader理解 | ActionPostprocess显式支持next-step target |
| 生产方式 | 分阶段脚本与会话级输出 | 配置驱动的Reader/Transform/Writer |

两者可以复用数据组织和验收思路，但gripper、坐标与action时间语义不能直接互换。

<a id="data-validation"></a>

## 4. 数据验收与训练衔接

### 最少要检查什么

1. 视频能否解码，帧数、时间戳和Parquet行数是否对应。
2. episode与task索引是否正确，切片是否包含合理的动作上下文。
3. EEF投影、局部轴和开合方向是否正确，明显抓取/释放时刻是否对齐。
4. 缺失维度是否有mask，修复或裁剪后是否同步更新所有相关字段。
5. 归一化统计是否对应实际source、动作表示、维度和chunk设置。

### 数据处理结束后，训练还要做什么

```text
落盘的state/action与质量信息
→ 选择有效anchor与未来chunk
→ 按需要上采样或映射动作槽位
→ delta / rotation变换
→ normalization
→ 组装图像、文本、mask并计算loss
```

这些步骤由具体训练实现决定，不应全部归到数据转换脚本。尤其要避免重复做next-step、重复delta、错误复用统计量，或把相邻帧分进训练和验证两侧。

Horizon normalization是在训练目标上按`[H,D]`统计每个预测位置的尺度，不是把原始视频重新采样。原理见[逐时刻归一化](MagicAtom/03_Action表示/01_ActionChunk_逐时刻归一化.md)。

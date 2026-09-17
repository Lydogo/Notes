# MagicVLA预训练实验记录

本文按版本记录MagicVLA预训练的模型结构、数据配方和实验状态。32D base、Dynamic/SGM以及后续W0.5分别记录，不能混用动作索引、checkpoint或归一化统计。数据处理见[统一数据处理笔记](../Note_DataPipeline.md)，基础概念见[基础知识](../Note_Basics.md)。

## 1. 实验版本

| 版本 | 主要改动 | 动作接口 | 状态 |
|---|---|---|---|
| 0809 MagicVLA-base | VLM与Action Expert双主干、32D多源预训练 | 32D | 已有基础实验记录 |
| 0816 MagicVLA-Dynamic-V0 | 增加2D/3D World Expert和teacher监督 | 32D配置 | 已完成模型方案与训练配置 |
| 0830 Magic-W0.5 | 后续结构和训练优化 | 以实际配置为准 | 原记录待补充 |

日期用于区分代码和配置快照，不代表模型发布日期。

## 2. 0809 MagicVLA-base

### 2.1 模型结构

MagicVLA-base采用双主干MoT结构：

~~~text
图像、语言、state/mask → Qwen VLM prefix
                              ↓ Full Attention层提供K/V
带噪action chunk、flow time、state → Action Expert
                              ↓
                         velocity[H,D]
~~~

| 分支 | 结构 |
|---|---|
| VLM | Qwen3.5-2B，hidden size 2048，24层 |
| Action Expert | Qwen3.5-Text风格，hidden size 1024，24层，SwiGLU intermediate size 3072，约460.36M参数 |
| 混合层型 | 3×Gated DeltaNet + 1×Full Attention，重复6次 |

DeltaNet层中两条主干独立更新；6个Full Attention层用于跨分支交互。Action token读取VLM prefix和整个action chunk，VLM只在自己的prefix内更新，不读取action suffix，避免连续动作标签直接泄漏。

基础查阅：[Attention](../Note_Basics.md#basic-attention)、[Masked Attention](../Note_Basics.md#basic-masked-attention)、[旋转表示](../Note_Basics.md#basic-rotation)。

### 2.2 32D动作契约

| 索引 | 含义 |
|---|---|
| 0:6、6 | 左臂6个joint、左gripper |
| 7:13、13 | 右臂6个joint、右gripper |
| 14:17、17:23 | 左EEF xyz、rotation-6D |
| 23:26、26:32 | 右EEF xyz、rotation-6D |

不同机器人只监督自己具备的维度。缺失槽位补零并关闭dimension mask，不能把补零当成真实状态。joint和xyz通常相对当前状态构造chunk-delta；gripper保持absolute；rotation按旋转矩阵组合计算相对姿态，不能直接逐元素相减。

32D是本版本的独立接口。后续34D会重新安排joint和EEF槽位，不能直接加载32D的动作头、统计量或部署脚本。

### 2.3 知识隔离与训练目标

机器人batch主要使用Flow Matching loss。启用Knowledge Insulation时，Flow loss不更新VLM；VLM由FAST动作token或独立VLM-SFT数据更新。该设置保护VLM视觉语言先验，但不等于冻结整个模型；Action Expert及其投影层是否更新由具体配置决定。

基础查阅：[Flow Matching](../Note_Basics.md#basic-flow)、[动作表示与归一化](../Note_Basics.md#basic-normalization)。

### 2.4 数据组成

base预训练数据包括EgoDex、Hy-UMI、RoboDojo仿真与真机、RoboTwin2.0、AgiBot World、Galaxea以及EO VLM-SFT。早期统计为8个机器人数据源、约1067小时；具体数量以对应配置和数据清单为准。

| 数据源 | 数据特点 | 处理后或训练路径 |
|---|---|---|
| EgoDex | 单头部相机、人手tracking | /pfs/public/opensource/dataset/EgoDex-LeRobot-v2.1-virtual-hand-quality-full-120core-deferred |
| RoboDojo仿真 | ARX X5，三相机 | /pfs/public/opensource/dataset/RoboDojo-LeRobot-v2.1-action32-3cam-quality-full |
| RoboDojo真机 | ARX X5、Piper、Piper X | /pfs/public/opensource/dataset/RoboDojo_real_camframe_v21/{arx_x5,piper,piper_x} |
| RoboTwin2.0 | ARX X5、Piper、Aloha仿真 | /pfs/public/opensource/dataset/RoboTwin2.0-LeRobot-v2.1-action32-3cam-full |
| Hy-UMI | 双手EEF示教，可retarget到ARX5 | /pfs/public/opensource/dataset/Hy-Embodied-0.5-VLA-Data/table_000_lerobot_v21_arx5 |
| AgiBot World | 真机数据 | /pfs/public/opensource/dataset/agibot-world/AgiBotWorld-Alpha |
| Galaxea R1 Lite | 真机数据，静态相机 | /pfs/public/opensource/dataset/Galaxea_static_camframe_v21/r1lite |
| EO Robo2VLM | 视觉语言监督 | /pfs/public/pretrain/dataset/EO-Robo2VLM-Qwen3VL-SFT/eo |

完整输入格式、质量检查和坐标处理见[数据处理笔记](../Note_DataPipeline.md)。

### 2.5 Base动作实验

原记录列出了三类32D动作消融，但没有填入完整配置、测试次数和成功率。因此这里只保留实验变量：

| 实验 | 有效维度 | 目的 |
|---|---|---|
| only joint | 前14维 | 检查joint控制路径 |
| only EEF | gripper索引6/13和后18维 | 检查EEF控制路径 |
| joint-EEF union | 全32维 | 检查联合动作接口 |

代码快照：[3394d161](https://gitlab.magiclab.top/vla/magicvla/-/commit/3394d161bf24b8c17371e2454e04cdeea47dde08)。

## 3. 0816 MagicVLA-Dynamic-V0

### 3.1 四路同步结构

Dynamic在base的VLM和Action Expert之外增加2D、3D World Expert：

~~~text
VLM prefix ───────┐
2D World tokens ──┼→ Full Attention层的可见K/V → Action tokens
3D World tokens ──┘
~~~

| 分支 | 作用 |
|---|---|
| Action Expert | 预测连续action velocity |
| 2D World Expert | 用空间queries预测未来二维视觉特征 |
| 3D World Expert | 预测geometry和motion latent |

四路分支仍按3×Gated DeltaNet + 1×Full Attention重复6次。DeltaNet层分别运行；Full Attention层按配置允许分支读取VLM、2D、3D和Action的K/V。VLM不读取World或Action token。

这不是直接生成未来RGB或未来机器人轨迹，而是用未来相关teacher latent训练World Expert，再让Action读取这些表征。完整说明见[Dynamic/SGM原理](../MagicAtom/04_Dynamic模型/01_MagicVLA_Dynamic_SGM.md)和[SGM模块细节](../MagicAtom/04_Dynamic模型/02_SGM_模块与训练细节.md)。

### 3.2 Teacher与loss

| 目标 | Teacher | 预测分支 |
|---|---|---|
| 2D视觉latent | 冻结DINOv3，从未来图像提取patch特征 | 2D World Expert |
| 3D geometry | 冻结Track4World | 3D geometry head |
| 3D motion | 冻结Track4World | 3D motion head |
| 动作velocity | Flow Matching目标 | Action Expert |

代表性配置的联合目标为：

~~~text
L = L_FM + 0.05 L_FAST + 0.2 L_2D
    + 0.3 × (0.5 L_geometry + L_motion)
~~~

2D使用有效patch上的cosine loss；3D分别对预测和目标做LayerNorm后计算MSE；各项分母使用有效mask。Teacher只生成目标，不参与梯度更新。

Knowledge Insulation会截断机器人loss到VLM的梯度，但不妨碍Expert读取VLM条件。开启和关闭KI是两种不同训练语义，不能只根据是否冻结某个模块判断。

### 3.3 推理与实验边界

训练时在线构造DINOv3和Track4World目标；推理时不构建teacher，保留训练好的2D/3D World Expert。固定观测下，VLM prefix的K/V可以缓存；World和Action分支仍需参与专家前向。

动作从masked noise开始，经过多步Euler更新，再按对应的归一化、delta和mask规则恢复成控制命令。teacher-free不等于没有World Expert，也不等于推理成本与base完全相同。

Dynamic复用base的多源数据和32D动作接口；未来World target的时间位置需要和chunk实际物理跨度对齐。原记录没有完整的Dynamic闭环结果、数据规模对比和消融表，因此当前只记录已完成的方案和训练链路。

代码快照：[v0.2.1](https://gitlab.magiclab.top/vla/magicvla/-/tree/v0.2.1)。

## 4. 0830 Magic-W0.5

原记录只保留了“核心升级（优化）”标题，尚未给出可核对的模型改动、数据配方、配置、commit或评测结果，暂不补写具体结论。

| 项目 | 后续需要记录 |
|---|---|
| 改动目标 | 解决什么训练或部署问题 |
| 模型变化 | 新增或删除哪些分支，是否改变动作维度 |
| 数据变化 | 数据源、采样权重、FPS、chunk和norm |
| 训练设置 | checkpoint、batch、steps、学习率、loss权重 |
| 验证结果 | 任务、次数、成功标准、对照和失败类型 |

## 5. 预训练实验的统一记录方式

每次实验至少记录版本与代码、数据范围、模型开关、训练预算、结果定义和成本限制。结果必须说明任务集合、测试次数和成功标准，区分“代码支持”“配置启用”“训练完成”和“效果验证”。

32D、34D或不同base checkpoint之间的结果不能互相覆盖。Horizon normalization、delta/rotation变换及归一化统计也必须和动作版本一起记录。

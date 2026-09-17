# OpenEgo到Qwen JSONL的数据转换

本文说明如何将OpenEgo的第一视角视频和时间段动作标注转换为Qwen视觉语言训练使用的JSONL。当前有两种输出：视频多轮问答，以及动作片段的三帧图像问答。

## 1. 输入数据

每个demo目录通常包含：

~~~text
demo_xxxx/
├── annotation.json          任务、动作片段、对象和执行者
├── joints.hdf5              双手21关节点、可见性和相机内参
├── metadata.hdf5            视频参数和相机内参
├── original_metadata.hdf5   原始任务文本
└── video.mp4
~~~

annotation.json的核心字段包括task、actions和video_info。每个action包含起止时间、objects、actors和自然语言label。视频信息包括帧数、FPS、分辨率和时长。

当前转换重点是视频理解和动作语义，joints.hdf5中的手部坐标暂不写入Qwen JSONL的动作字段。

## 2. Version 1：一条视频对应一条多轮conversation

### 2.1 构造规则

每个demo生成一条VLMSample。视频标记只放在第一轮user message，后续问题共享同一视频上下文。

第一轮询问整体任务，之后每个动作片段依次询问：

1. 时间段内发生了什么；
2. 操作了什么对象；
3. 哪只手或哪些手执行了动作。

示意：

~~~json
{
  "messages": [
    {"role": "user", "content": "<video>\nWhat task is being performed?"},
    {"role": "assistant", "content": "Moving a small red plastic chair."},
    {"role": "user", "content": "What happens between 4.30s and 9.20s?"},
    {"role": "assistant", "content": "grasp, lift, and slightly move the chair"}
  ],
  "is_video": true,
  "video": "/path/to/video.mp4"
}
~~~

样本中还记录id、dataset、subset、split、source和能力标签等metadata。由于一条样本可能包含多个动作片段，conversation按时间顺序组织，避免打乱动作关系。

### 2.2 适用场景

这种格式保留完整视频上下文，适合task recognition、长时视频理解和按时间段回答问题。代价是单条样本较长，问题之间共享视频，训练时token和视频解码成本较高。

## 3. Version 2：视频问答与三帧动作问答

Version 2拆成两个subset：

| subset | 输入 | 主要问题 |
|---|---|---|
| egocentric_video_qa | 一段视频 | task、action、object、actor和temporal understanding |
| egocentric_action_multiframe_qa | 每个动作的start/middle/end三帧 | action、object、actor和时间顺序 |

### 3.1 视频样本

视频样本保留任务级问题，并为动作片段生成时间段问题。metadata增加qa_types，用于下游按能力筛选。与Version 1相比，Version 2明确记录token_len等长度信息，便于数据配额和训练采样。

### 3.2 三帧图像样本

每个action抽取三个时间点：

~~~text
target_timestamps = [start, (start+end)/2, end]
→ 映射到最近视频帧
→ 记录目标时间、实际帧时间和帧索引
→ 保存三张JPEG
~~~

三张图片按开始、中间、结束顺序放入同一个user message。问题不再重复时间段，而是直接询问这段交互的action、object和actor。

样本metadata至少保存action_index、action_start_timestamp、action_end_timestamp、target_timestamps、frame_indices、frame_timestamps、original_frame_wh、output_frame_wh和images路径。目标时间与实际帧时间可能不同，二者不能混为一谈。

### 3.3 图像处理

使用PyAV解码视频；图片等比例缩放，默认长边不超过768，JPEG quality为90。缩放不改变帧的时间戳，图片路径用稳定的内容hash或约定命名，避免重复保存同一帧。

## 4. Qwen JSONL字段约定

常用字段如下：

| 字段 | 含义 |
|---|---|
| id | 样本唯一标识 |
| dataset / source / source_cat | 数据集、原始路径和子集 |
| subset | 视频或三帧任务类型 |
| capability | 训练能力标签 |
| split | train/validation/test |
| has_action / has_coords | 是否包含可执行动作或坐标监督 |
| n_images / is_video | 图像数量和媒体类型 |
| qa_types | 样本包含的问答能力 |
| messages | Qwen多轮对话 |
| video / images | 视频路径或图片路径 |

OpenEgo当前是VLM监督，has_action和has_coords为false。不能因为元数据中存在joints.hdf5，就把它描述成已经接入动作训练。

## 5. 数据质量与异常处理

转换器对以下情况告警并跳过样本：

- JSON无法解析；
- 视频缺失或无法解码；
- action时间非法或超出视频范围；
- 三帧无法生成；
- 任务、对象或执行者字段格式不符合预期。

跳过原因应写入日志或汇总报告，避免只看到最终样本数量而无法解释数据损失。时间段边界要做裁剪或明确跳过规则，不能静默生成负时间或空图片。

## 6. 规模与工程限制

OpenEgo的双手关节数据形状约为300×21×3，完整写入JSONL会显著增加存储和读取成本；转成文本又会损失精确的运动信息。因此当前先把它作为视频理解数据，坐标是否另存为紧凑二进制或离线特征，作为后续方案。

视频数据还带来编解码和CPU解码成本。批量转换时应控制并发、复用帧读取、记录缓存路径，并在输出后检查图片尺寸、视频可读性和JSONL行数。

## 7. 两个版本的选择

| 需求 | 推荐格式 |
|---|---|
| 长视频、任务整体和多个动作片段 | Version 1或Version 2视频subset |
| 训练动作识别、对象和执行者识别 | Version 2三帧subset |
| 控制样本长度和图像token数量 | Version 2三帧subset |
| 保留动作发生前后的完整上下文 | 视频subset |

两个版本可以并行生成，但需要通过id和subset区分，避免同一视频问答重复计入同一训练配额。

## 8. 代码与后续扩展

Version 1已完成OpenEgo reader、task/action/object/actor问答、按时间均匀抽取max_actions_per_demo、source.type=openego注册、示例配置、单测和pipeline集成测试。无效JSON、缺失视频或非法action会告警并跳过。

后续如果接入手部动作监督，建议新增独立字段或sidecar：

~~~text
JSONL：任务与问答
sidecar：帧索引、坐标、可见性、相机参数
训练reader：按时间对齐后决定是否使用坐标监督
~~~

这样可以保持VLM JSONL简洁，也不会丢失高维运动数据。

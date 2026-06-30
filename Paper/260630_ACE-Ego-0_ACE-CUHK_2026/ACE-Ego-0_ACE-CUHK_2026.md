# ACE-Ego-0：把第一视角人类视频变成可用于 VLA 预训练的动作监督

> 原标题：ACE-Ego-0: Unifying Egocentric Human and Robotic Data for VLA Pretraining  
> 作者：Hao Li, Ganlong Zhao, Yufei Liu, Haotian Hou, Guoquan Ye, Tongyan Fang, Chunxiao Liu, Siyuan Huang, Jianbo Liu, Xiaogang Wang, Hongsheng Li  
> 机构：ACE Robotics, CUHK MMLab, CUHK Shenzhen, SJTU, THU  
> 发表：arXiv:2606.17200v1, 2026-06-15  
> 链接：https://arxiv.org/abs/2606.17200  
> 项目：https://acerobotics-vla.github.io/ACE-Ego/  
> 开源：https://github.com/ACERobotics-VLA/ACE-Ego  

---

## 一、研究背景与动机

VLA 模型要泛化，核心燃料还是大规模、多样化的具身数据。但机器人遥操作数据贵、慢、平台差异大，单靠 robot demonstrations 很难覆盖真实世界里长尾的物体交互。第一视角人类视频（egocentric human video）便宜且丰富，看起来是天然补充。

问题在于，人类视频和机器人轨迹不是同一种监督：动作空间不同、身体结构不同、控制频率不同，而且人类动作标签通常来自 3D 手部重建，噪声远高于机器人传感器记录。ACE-Ego-0 的目标就是把这些异构来源对齐到一个可共同训练的 VLA 预训练框架里，同时避免 noisy pseudo-actions 把机器人控制能力带偏。

## 二、核心贡献

1. 提出统一的 VLA 预训练框架 ACE-Ego-0，同时处理表示异构和监督质量不匹配两个问题。
2. 构建五阶段 egocentric video-to-action pipeline，把 6 个第一视角视频源转换成 1,478.9 小时 pseudo-action-labeled human data。
3. 设计统一动作表示：camera-space actions、cross-embodiment morphology conditioning、time-aligned action chunking，把机器人和人类手部轨迹放入共享接口。
4. 提出 reliability-aware training objective，让高保真机器人轨迹承担主 loss，人类伪动作只以加权辅助监督的方式进入训练。

## 三、方法原理

### 3.1 整体框架

ACE-Ego-0 使用 Qwen3-VL-4B-Instruct 作为视觉语言骨干，后接约 600M 参数的 flow-matching DiT action expert。输入是多视角图像和语言指令，输出是时间对齐的 camera-space action chunks。动作包含双臂末端位置、6D rotation、夹爪和 activity flag。

训练数据来自三类来源：机器人真机 demonstrations、仿真 rollouts、人类第一视角视频。机器人数据监督主 action loss；人类视频经过手部重建、动作参数化和质量过滤后，只通过 human auxiliary loss 提供可靠通道上的补充监督。

### 3.2 关键技术细节

**Camera-space action representation。** 所有机器人末端执行器位姿和人类手部伪末端轨迹都转换到头部相机坐标系。这样模型不需要在策略内部学习每个数据源的 world/base frame 到相机 frame 的复杂变换。部署到新机器人时，主要依赖相机外参把预测动作转回机器人执行坐标系。

**Human end-effector equivalents。** 人手没有机器人式末端执行器，作者用 wrist joint 作为末端原点，用掌平面和 wrist-to-finger 向量构造稳定的手部朝向，用 thumb-to-palm distance 近似夹爪开合。直觉上，这是把人手轨迹压成机器人能理解的“腕部末端 + 开合”格式。

**Morphology conditioning。** camera-space 解决空间坐标问题，但不同机器人和人类手的运动学结构仍不一样。ACE-Ego-0 给 action expert 注入 morphology token：机器人 token 来自 URDF graph encoder，人类 token 是按数据源学习的 surrogate embedding。这个 token 只进入 action decoder，不污染 VLM backbone。

**Time-aligned action chunking。** 不同数据源控制频率不同，固定步数 chunk 会对应不同物理时长。ACE-Ego-0 用目标物理时间窗 `T* = 2s` 定义 horizon：`Hd = round(fd T*)`。这样每个数据源监督的未来动作都覆盖相近的真实时间长度。

**Reliability-aware human auxiliary loss。** 人类伪动作的 position 比 rotation、gripper 更可靠，因此作者定义通道级先验 `rho_j` 和 step-level smoothness 权重 `w_t,j`，组合成可靠性权重 `W_t,j`。人类数据不直接进入主 flow-matching loss，而是用加权 Huber loss 辅助训练，重点监督可靠的位置通道。

### 3.3 训练与优化

预训练数据池超过 6,013.7 小时，其中 human video 1,478.9 小时，robot/simulation 4,534.8+ 小时。模型预训练使用 128 张 A800 80GB，任务微调用 16 张 A800。图像分辨率为 256x256，推理时 flow matching 解码 4 步。

人类视频处理 pipeline 包括：数据源筛选、交互视频选择、3D hand reconstruction、动作参数化、质量控制。质量控制会过滤静态片段、轨迹 spike、不完整姿态和不合理双手运动。

## 四、实验与结果

### 4.1 实验设置

评估覆盖 RoboCasa GR1 TableTop、RoboTwin 2.0 和真实 ARX 双臂平台。RoboCasa 每任务 50 次 rollout，共 24 个任务；RoboTwin 2.0 共 50 个任务，每任务 100 次；真实机器人每任务 30 次。

### 4.2 主要结果

| Benchmark | 指标 | 最强对比基线 | ACE-Ego-0 | 结论 |
|---|---:|---:|---:|---|
| RoboCasa GR1 TableTop | 平均成功率 | DIAL 70.2 | 72.8 | 超过已有 VLA/robot baselines |
| RoboTwin 2.0 Easy | 平均成功率 | Hy-VLA 90.9 | 91.12 | 小幅领先 |
| RoboTwin 2.0 Hard | 平均成功率 | Hy-VLA 90.1 | 90.62 | 随机化设置下仍领先 |
| ARX real robot | 6 任务平均成功率 | pi0.5 71.7 | 78.3 | 真实双臂任务提升明显 |

最有说服力的是真实 ARX 双臂评估：ACE-Ego-0 平均 78.3%，高于 pi0.5 的 71.7%，也显著高于 GR00T-N1.7 的 35.6%。其中 Scoop Coffee 这种接触丰富、双臂时序要求高的任务，ACE-Ego-0 达到 86.7%，比 pi0.5 高 16.7 个点。

### 4.3 消融实验

| 设置 | RoboCasa 成功率 | 变化 |
|---|---:|---:|
| Full ACE-Ego-0 | 72.8 | - |
| w/o Dynamic Chunking | 71.7 | -1.1 |
| w/o URDF / morphology token | 70.9 | -1.9 |
| w/o Human Loss | 69.2 | -3.6 |

| 预训练数据 | RoboCasa 成功率 |
|---|---:|
| From Qwen, no embodied pretrain | 65.4 |
| Robot Only | 68.3 |
| Robot + Human | 72.8 |

消融说明，人类视频不是装饰性数据：在 RoboCasa 上从 Robot Only 到 Robot + Human 提升 4.5 个点，是数据源消融里最大的一段增益。作者还在 Sweep Cubes 低数据微调中展示，34 条机器人 demo 只有 10% 成功率，加入 419 条任务匹配人类视频后提升到 40%。

## 五、局限性与展望

作者明确指出，当前评估主要集中在 tabletop manipulation，尚未覆盖移动操作、全身 humanoid control 或 deformable-object tasks。预训练池也没有 dexterous hand data 和 force/torque sensing，因此对细粒度接触和灵巧手操作的支持还有限。

我的额外判断是，ACE-Ego-0 的关键瓶颈仍在 pseudo-action fidelity。position channel 相对可靠，所以人类监督主要贡献轨迹覆盖；但 rotation、gripper、finger-level contact 仍被降权，说明它还没有真正从人类视频中稳定学习复杂接触策略。另一个问题是 pipeline 很重，依赖 hand reconstruction、camera pose、过滤阈值和人工/工程化数据清洗，复现成本不低。

## 六、灵魂三问

1. **它解决了什么问题？**

   它解决的是 VLA 预训练数据扩展中的异构数据融合问题：机器人数据可靠但贵，人类第一视角视频便宜但动作标签噪声大。ACE-Ego-0 试图让两者在统一 camera-space action interface 下共同训练，同时避免把人类伪标签噪声当作等价机器人动作。

2. **为什么这么做？**

   因为单靠 robot demonstrations 很难覆盖真实操作的长尾行为，而直接混入人类 pseudo-actions 又会污染 action expert。camera-space、morphology token、time-aligned chunking 负责对齐表示；reliability-aware loss 负责控制噪声注入。

3. **什么证据最有说服力？**

   数据源消融最直接：From Qwen 65.4，Robot Only 68.3，Robot + Human 72.8，说明人类视频在机器人预训练之外提供了额外行为覆盖。真实 ARX 双臂平均 78.3% 也很关键，因为它证明这种预训练不是只在仿真 benchmark 上刷分。

## 七、个人总结

1. ACE-Ego-0 的核心想法是：把人类第一视角视频转换成 camera-space 的机器人兼容伪动作，再用可靠性加权把它变成 VLA 预训练的补充监督。
2. 最大优势是数据扩展思路务实，知道人类伪动作有噪声，所以没有天真地和机器人动作一视同仁；最大弱点是对手部重建和数据过滤 pipeline 的依赖很强。
3. 对 VLA 研究的启发是，人类视频可能最先贡献的不是精确控制，而是更宽的动作空间覆盖和行为先验；要让它贡献 rotation、gripper、contact，还需要更强的动作恢复和可靠性估计。

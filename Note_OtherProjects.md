# 辅助项目记录

保留早期机械臂与Isaac Lab/RL实践，便于查阅实现和排障过程。核心项目与全部面试复盘见[项目笔记](Note_Project.md)，通用原理见[基础知识](Note_Basics.md)。

## 机械臂项目

### 1. ROS2 机械臂夹取项目

基础查阅：[ROS2](Note_Basics.md#basic-ros2)、[IK](Note_Basics.md#basic-ik)、[规划](Note_Basics.md#basic-planning)。

#### 用到的库

| 库 | 用途 |
|---|---|
| `rclpy` | ROS2 Python客户端库，创建节点 / 发布者 / 订阅者 |
| `gazebo_ros2_control` | 连接Gazebo仿真器与ROS2控制系统 |
| `MoveIt2` | 运动规划与操作 |
| `hardware_interface` | 硬件抽象层，连接控制器与机器人硬件 |
| `robot_state_publisher` | 发布机器人TF变换 |
| `joint_state_publisher` | 发布关节状态信息 |
| `rviz2` | 可视化机器人状态与轨迹 |

#### Pick-and-Place 实现

##### 状态机枚举

把整个pick-place流程拆成5个离散状态：

```
class StateMachineAction(enum.Enum):
    GRAB     # 激活夹爪，抓取物体
    DELIVER  # 运送物体
    RELEASE  # 释放物体
    HOVER    # 悬停在物体上方
    HOME     # 返回初始位置
```

##### 话题与通信

| 方向 | 接口 / 话题 | 用途 |
|---|---|---|
| Pub | `Float64MultiArray` → `joint_control_topic` | 把目标关节角发给ros2_control的 `JointGroupPositionController` |
| Sub | `JointState` ← `/joint_states` | 拿到实测关节角 / 角速度 / 力矩，用于FK反馈与速度判稳 |
| Sub | `Odometry` ← `/odom` | 拿到机器人基座在世界系的位姿；AGV场景下需 `T_world_base` 做全局定位 |
| Sub | `PoseStamped` ← `/goal_pose` | 接收外部点击的目标，RViz 2D Nav Goal可直接发到此话题 |

##### 运动学核心调用链

| 调用 | 功能 |
|---|---|
| `self._panda.solve_fk(joint_states)` | FK：关节角 → 末端4×4齐次矩阵 → 填到Odometry |
| `self._panda.solve_ik(end_effector_target)` | IK：末端Pose → 关节角，MoveIt2内部默认用KDL求解 |
| `move_fingers(..., FingersAction.OPEN/CLOSE)` | 返回平行夹爪最后两个关节角度，0 = 全开，1 = 全闭 |

#### 遇到的问题

##### 奇异点问题

- **现象**：某关节速度暴涨。
- **原因**：6DOF逆解时无解（雅可比向量线性相关出现奇异值）。
- **解决**：增加一个自由度，或在yaml配置里直接lock掉易出现奇异值的角度。

#### 关于动力学的说明

基础查阅：[运动学与动力学](Note_Basics.md#basic-dynamics)。

项目主要关注**运动学规划 + 基于位置的控制**：通过IK求解器算出目标关节角，再交给ROS2 Control的关节位置控制器执行。底层动力学由Gazebo根据URDF中的质量 / 惯量参数和控制器输出力矩自动仿真，本项目没有直接实现RNE等动力学推导。

### 2. 海恒智能国科大机械臂项目

#### task1 通信桥代码

##### 背景

AGX上位机通过HTTP把数据传给Docker内部，需要一个通信桥脚本把数据转到ROS内，再调用MoveIt规划功能控制电机执行。

##### 主要接口

| 接口 | 功能 |
|---|---|
| `leg_move` | 腿部电机控制服务（服务端 + 客户端） |
| `get_current_pose_http` | 获取当前末端执行器位姿（`position.x/y/z` + `orientation.x/y/z/w`，可直接给move_group） |
| `calculate_pre_position` | 计算从目标位置远离书架一定距离的中间位置，作为夹取前的预位 |
| `calculate_target_position_from_pixel` | 像素坐标 → base坐标 |
| `plan_to_position` | 调用move_group，把机械臂移动到目标位置 |

##### 难点与细节

**A. `plan_to_position` 运动规划演进**

1. **方案1（失败）**：直接用末端位置规划，频繁解算超时、候选路径过多。
2. **方案2（次优）**：观察RViz拖动末端的行为发现move_group内部倾向于关节空间规划；改成手动调IK解出关节角，再做关节空间规划。
3. **新问题**：路径中间缺少约束，机械臂出现大幅"甩动"，路径品质差。
4. **方案3（最终）**：直接用move_group高级接口 `go()` 并设置位姿目标，让move_group内部自动完成IK + 关节空间规划，并优选关节移动量最小的路径。

**B. 像素 + 抓取角度 → 基座坐标系下的目标位姿**

输入是2D像素 + 深度 + 期望抓取角度，输出是末端6D位姿（position + orientation）。

1. **像素 → 相机坐标**：用相机内参做反投影得到三维点。
2. **相机坐标 → 基座坐标**：用 `tf2_ros.Buffer.lookup_transform` 查 `camera_frame → base_frame` 的TF，`ros_numpy.numpify` 转成 `T_base_cam`，矩阵乘法完成变换。
3. **构建目标姿态**：
   - 相机检测书本得到一个向量，通过TF旋转部分映射到基座系，作为夹爪Z轴。
   - 与基座系X轴 `[1, 0, 0]`（保证夹爪垂直于书架）叉乘构建正交坐标系，得到旋转矩阵。
   - 解算前先把夹爪坐标系对齐到基座系方向，简化"Z对准向量、X朝前"的逻辑；最后再做一次旋转补偿，回到夹爪URDF的实际坐标系。

**C. 手眼标定流程**

1. `realsense2`：相机驱动
2. `aruco_ros`：marker标定
3. `easy_handeye`：坐标解算

> 标定原则：**深度尽量保持不变，角度尽量多变**。

#### task2 multi_action_server.py

##### 背景

ROS节点，作为上层MoveIt与底层CAN总线电机之间的桥梁，确保多组关节（手臂、腿部）在运动时不会冲突。

##### 难点与细节

- **问题复盘**：状态读取指令无序插入动作指令序列（如GOTO执行中多次插入STATE读取），导致部分电机动作延迟、轨迹与预设不符。
- **原因**：原控制机制中动作服务器高频读取电机状态，与动作指令无约束并发——动作指令未发完就被状态读取打断；同时存在非必要的状态读取浪费资源。
- **解决**：
  - 取消无约束高频定频读取；
  - 增设约束：动作指令发送完成前禁止状态获取；
  - 优化控制锁范围，删除无效长时间锁占用，避免锁竞争。

<a id="rl-project"></a>

## RL项目实践复盘与Isaac Lab

### 25.12.27 | 环境定义架构

#### 1. 配置类（@configclass）

- **本质**：纯数据容器（Python Decorator），不含运行逻辑。
- **作用**：实现参数与逻辑解耦。通过修改配置类即可切换物理属性，无需改动环境核心代码。

#### 2. Spawn 属性

- **机制**：支持配置对象的继承与复用。
- **随机化**：通过 `spawn` 实现资产的参数化定义，是实现大规模并行环境随机化的核心入口。

#### 3. 架构解耦

- **物理资产**（Asset）与**控制逻辑**（Manager）彻底分离。
- 资产层只定义“物体是什么”，逻辑层（Reward/Obs/Action Managers）定义“怎么做”。

---

### 26.1.4 | Docker 部署与项目跑通

#### 1. Docker Build 网络故障

- **问题**：`build` 过程中 `git clone` 失败。主机全局代理无效，因Docker编译环境与宿主机网络默认不互通。
- **解决方法**：
    1.  `docker-compose.yaml`：在 `build` 标签下添加 `network: host` 强制共享宿主机网络。
    2.  `Dockerfile.base`：显式设置环境变量 `ENV http_proxy` 和 `ENV https_proxy`。

#### 2. 容器操作流程

- **标准步骤**：`container.py start` -> `container.py enter`。
- **注意**：必须通过 `enter` 脚本进入容器，系统会自动挂载路径并配置 `PYTHONPATH` 等环境变量，手动 `docker exec` 会导致路径报错。

#### 3. 项目运行与迁移

- **基础链路**：`train.py` 训练模型 -> `play.py` 加载模型演示。
- **API兼容性**：老旧项目需对比官方最新Demo检查 `ManagerTermBase` 等API的函数签名，重点关注参数名的更新。

#### 4. 数据可视化（Tensorboard）

- **避坑**：Docker内端口转发不稳定，且占用容器资源。
- **最佳实践**：在宿主机终端直接运行，通过挂载的 `logs` 目录实时读取：

    ```bash
    tensorboard --logdir .
    ```
---

### 26.1.5 | Lift 项目跑通与核心逻辑

#### 1. 环境注册机制（Registration）

- **流程**：Isaac Lab通过 `gym.register` 将环境加入注册表。`train.py` 或 `play.py` 通过 `--task` 参数从注册表中检索配置。
- **入口**：注册信息通常集中在模块的 `__init__.py` 中。
- **链式导入**：通过 `from . import config` 等语句实现层层递进式加载，确保在运行脚本前，所有自定义环境配置已注入Gym注册表。

#### 2. 观测空间设计（Observations）

- **泛化性原则**：优先使用**相对坐标**。相比绝对坐标，相对坐标（如物体相对于机器人基座）能让策略更易学习空间几何关系，提高在不同初始位姿下的泛化能力。
- **坐标转换**：利用 `subtract_frame_transforms` 将物体从世界坐标系（World Frame）转换至机器人局部坐标系（Local/Root Frame）。

    ```python
    # 实现世界系到局部系的转换：(物体世界位姿 - 机器人世界位姿)
    object_pos_b, _ = subtract_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], object_pos_w
    )
    ```

#### 3. 奖励函数结构（Rewards）

Lift示例通常包含三个关键引导项：

- `object_is_lifted`：物体是否离开台面的离散/连续奖励。
- `object_ee_distance`：末端执行器（EE）与物体的接近诱导奖励。
- `object_goal_distance`：物体与目标位置的距离惩罚。

---

### 26.1.6 | 源码追踪与任务迁移

#### 1. 开发技巧：函数跳转

- **痛点**：由于Isaac Lab路径复杂，IDE默认无法直接跳转到外部库定义。
- **解决方法**：`Ctrl+Shift+P` -> `Tasks: Run Task` -> 运行一次Isaac Lab提供的Python环境配置脚本，使IDE索引生效。

#### 2. 源码阅读注意点

- **版本差异**：Isaac Lab迭代快，不同分支的代码实现（如库的调用路径）可能存在偏差。务必以当前本地库的源码定义为准进行修改。

#### 3. 任务拓展：从 Cube 迁移到长方体（模拟书本）抓取

若要训练机械臂从薄面抓取长方体，需从以下维度调整：

- **观测（Obs）**：必须引入**物体旋转角（Orientation）**，否则策略无法感知长方体的长短边，无法精准定位抓取面。
- **奖励（Rewards）**：
    * 增加姿态对齐奖励（如EE X轴与物体法线的夹角）。
    * 增加抓取稳定性奖励。
- **算法配置（RSL_RL）**：算法逻辑通常无需改动，但针对更精细的任务，可能需要微调学习率（Learning Rate）或增加训练迭代次数（Max Iterations）。
- **配置注册**：需新建对应的配置文件并在 `__init__.py` 中更新注册信息。

### 26.1.7 | 长方体抓取：奖励破解与物理约束

#### 1. 奖励破解（Reward Hacking）现象

- **问题**：改成长方体后，机械臂学会了通过“侧蹭”使物体竖立来骗取 `lift_object` 分数，而非真正夹取。
- **成因**：`lift_object` 权重过高且目标高度阈值设定过低，导致“竖立”动作产生的位移足以触发奖励。

#### 2. 引导奖励与物理极限

- **优化**：引入 `EE_to_object_distance` 奖励，强制末端靠近重心中点，抑制“蹭”的行为。
- **失败分析（紫色曲线）**：增加抬升高度后任务失败。对比Tensorboard曲线发现，抬升奖励归零是因为设置的高度**超出了机械臂的物理可达范围（Workspace Limit）**。
- **反思**：奖励目标必须设定在机器人运动学范围内，否则会引导策略进入死胡同。

![RL_26.1.7](Picture/RL_260107_01.png "RL_26.1.7")

---

### 26.1.8 | 姿态对齐与 Sim2Real 预演

#### 1. 随机化与观测增强

- **Domain Randomization**：在 `EventCfg` 中增加初始偏航角（Yaw）随机化，模拟物体在书架上的不同摆放姿态。
- **Sim2Real衔接**：模拟相机检测逻辑，将“物体中心指向倾斜方向的向量”注入观测空间（Observations），为后续实机部署对齐数据流。

#### 2. 引导对齐奖励（Orientation Guidance）

为引导夹爪从薄面夹取，新增两项奖励：

- **平行奖励**：EE的X轴与物体向量平行。
- **垂直奖励**：EE的Z轴与物体向量垂直。

#### 3. 负面现象：任务后过度调整（Over-optimization）

- **现象**：物体举起后，夹爪为追求姿态分持续扭动，导致机械臂高频抖动或姿态扭曲。
- **根源**：
    1.  `joint_vel` 和 `action_rate` 惩罚项过小，不足以抑制高频震荡。
    2.  奖励函数在任务完成后未失效，导致AI在高处“刷分”。

#### 4. 改进思路：奖励消隐与参考系切换

- **线性消隐（Linear Decay）**：引入线性插值，随着物体高度增加（任务接近完成），逐渐降低姿态奖励的权重，使机器人后期专注于稳定维持。
- **坐标系重构**：考虑将夹爪对齐目标由“物体局部向量”改为“世界坐标系轴向”。
    * **优点**：物体的局部向量在被抓起旋转时会剧烈变动，导致奖励不稳定；对齐世界坐标系（如垂直于地面）通常能提供更稳定的梯度。

<a id="ppo-practice"></a>

### 26.1.13 | PPO 算法原理（基于 rsl_rl 源码）

#### 1. 核心损失函数

PPO通过限制策略更新幅度来确保训练稳定性。其核心公式为：

- **策略裁剪（Clip Surrogate Object）**: 
    $$L^{CLIP}(\theta) = \hat{\mathbb{E}}_t \left[ \min \left( r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right]$$
- **总损失函数（Total Loss）**: 
    $$L_t^{PPO}(\theta) = \hat{\mathbb{E}}_t \left[ L_t^{CLIP}(\theta) - c_1 L_t^{VF}(\theta) + c_2 S[\pi_{\theta}](s_t) \right]$$

#### 2. Actor-Critic 模型结构

- **Actor（策略网络）**：输出动作的均值 $\mu$。通过 `std`（标准差）参数构建正态分布进行采样，维持探索性。
- **Critic（价值网络）**：输出状态价值 $V(s)$，用于评估当前局面的好坏。

#### 3. 计算比例和Clip损失

```python
# Actor-Critic简化逻辑示意
class ActorCritic(nn.Module):
    def __init__(self, ...):
        # 定义策略网络 (Actor)
        self.actor = nn.Sequential(...) # 输出动作均值 mu
        # 定义价值网络 (Critic)
        self.critic = nn.Sequential(...) # 输出状态价值 V(s)
        # 动作标准差 (Action Standard Deviation)，代表探索的随机性
        self.std = nn.Parameter(torch.ones(num_actions))

    def act(self, observations):
        # 采样动作：根据正态分布 N(mu, std)
        mu = self.actor(observations)
        dist = Normal(mu, self.std)
        action = dist.sample()
        return action, dist.log_prob(action), self.critic(observations)
```

```python
# 获取新旧策略的动作概率比
actions_log_prob_batch = self.actor_critic.get_actions_log_prob(obs_batch, actions_batch)
ratio = torch.exp(actions_log_prob_batch - old_actions_log_prob_batch)

# PPO Clipped Objective
surrogate = -advantages_batch * ratio
surrogate_clipped = -advantages_batch * torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
# 取两者中的最大值（因为这里带了负号，等同于论文里的 min）
action_loss = torch.max(surrogate, surrogate_clipped).mean()
```

```python
# Critic 的损失：预测值与目标值 (Returns) 的均方误差
value_loss = (return_batch - value_batch).pow(2).mean()
```

```python
# 鼓励探索
entropy_loss = dist.entropy().mean()
```

```python
# 总损失 = 策略损失 + 价值损失权重 * 价值损失 - 熵权重 * 熵损失
loss = action_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_loss

# 反向传播
self.optimizer.zero_grad()
loss.backward()
self.optimizer.step()
```

#### 4. 优势函数计算：GAE (Generalized Advantage Estimation)

GAE通过权衡偏差（Bias）和方差（Variance）来计算优势函数 $\hat{A}_t$。

- **TD误差（$\delta$）**: $r_t + \gamma V(s_{t+1}) - V(s_t)$。
- **递归计算**: 结合 $\gamma$（折扣因子）和 $\lambda$（平滑参数）进行逆序计算，平滑优势估计。

```python
def compute_returns(self, last_values, gamma, lam):
    advantage = 0
    for step in reversed(range(self.num_transitions_per_env)):
        next_values = last_values if step == self.num_transitions_per_env - 1 else self.values[step + 1]
        # TD 误差 delta
        delta = self.rewards[step] + gamma * next_values * self.not_done[step] - self.values[step]
        # GAE 递归计算
        advantage = delta + gamma * lam * self.not_done[step] * advantage
        self.advantages[step] = advantage
        self.returns[step] = self.advantages[step] + self.values[step]
```

---

### 26.1.14 | 任务进阶：从 Lift 到 Grasp & Pull

#### 1. 任务迁移

目标由单纯的垂直抬升（Lift）转变为从书架中夹取并向外拉出（Pull）。

#### 2. 坐标系陷阱：全局 vs 局部

基础查阅：[坐标变换](Note_Basics.md#basic-transforms)。
**痛点**：在Isaac Lab大规模并行仿真中，环境按 `env_spacing` 平铺分布。

- **全局坐标（root_pos_w）**：每个环境的坐标系原点在世界空间中是不同的。
- **风险**：若直接用全局 $X$ 坐标设定奖励阈值，除0号环境外，其他环境可能在起始点就已触发奖励（刷分），导致梯度爆炸或模型无法收敛。

**解决方案**：在这类并行环境中，用相对目标或环境原点的坐标计算奖励，避免环境平移改变奖励含义。

```python
# 将物体的世界 X 坐标减去该环境在世界系中的原点 X 坐标
relative_x = object.data.root_pos_w[:, 0] - env.scene.env_origins[:, 0]

# 基于相对位移判断拉出状态
is_pulled = relative_x < (target_x_offset - minimal_distance)
```

---

### 26.1.15 | 权重分配与训练稳定性

#### 1. 任务阶段权重失衡：Reach vs. Pull

- **现象**：末端执行器（EE）夹住书本后停止动作。
- **根源**：任务被分为“接近（Reach）”与“拉出（Pull）”两个阶段。若Reach阶段的引导奖励（距离、对齐等）权重过大，而Pull阶段（目标追踪、位移）权重过小，Agent会倾向于停留在接近状态以稳拿高分，失去后续冒险拉出的动力。
- **对策**：显著提升Pull相关项（如 `pulling_object`）的权重，确保后期奖励远高于前期引导奖励。

#### 2. 终止条件导致的 Value Loss 爆炸

- **现象**：增加“书本倒下”的终止条件（Termination）后，Value Loss飙升至 `inf`。
- **原理**：在PPO中，Critic网络负责预测长期回报。如果环境突然终止（书本倒下）却没有任何对应的负反馈（惩罚），Critic会无法理解为什么高分奖励流会瞬间中断，导致预测偏差剧烈震荡。
- **对策**：**保持奖励连续性**。对失败终止可设置与任务尺度匹配的惩罚，并检查终止后的价值估计；超时截断与真正失败应分别处理。

#### 3. 任务后期摆动问题

- **现象**：成功抓取并取出后，EE大幅度偏转或乱动。
- **原因**：Curriculum中的 `joint_vel` 和 `action_rate` 惩罚介入过晚或权重过小，导致Agent在完成核心任务后完全无视运动的平滑性。

```python
# Curriculum本身是为了初期训练的流畅，在一定步数之后在增大惩罚项的权重
@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""
    action_rate = CurrTerm(
        func=general_mdp.modify_reward_weight, params={"term_name": "action_rate", "weight": -1e-1, "num_steps": 10000}
    )

    joint_vel = CurrTerm(
        func=general_mdp.modify_reward_weight, params={"term_name": "joint_vel", "weight": -1e-1, "num_steps": 10000}
    )

```

---

### 26.1.16 | curriculum优化

#### 1. 惩罚项介入时机的量化参考

- **策略**：参考核心任务奖励（如 `pulling_object`）的曲线。
- **逻辑**：当前期奖励达到稳定阈值（说明Agent已掌握抓取基本功）时，即为施加运动限制的最佳时机。
- **计算示例**：若 `pulling_object` 在第400次迭代左右达标，则设置 `num_steps = 400 * num_steps_per_env`（如 $400 \times 24$）。

#### 2. 从跳变到线性插值（Smoothing）

- **现状**：原生的 `modify_reward_weight` 函数执行权重突变（Step Change），容易造成策略抖动。
- **改进**：自研 `modify_reward_weight_linear` 函数。
- **优势**：
    * **平滑过渡**：在设定的步数区间内（如从6000到20000步）线性增加惩罚。
    * **学习稳定性**：给Agent留出适应“运动限制”的时间缓冲区，避免因突然增加的惩罚导致已学到的抓取策略崩溃。

![RL_26.1.16](Picture/RL_260116_01.png "RL_26.1.16")

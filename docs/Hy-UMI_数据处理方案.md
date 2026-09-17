# Hy-UMI到ARX X5的数据处理方案

本文说明如何将Hy-UMI的双手EEF轨迹转换为ARX X5双臂的14D joint/action数据。核心流程是：统一坐标系、搜索机器人base、逐帧IK、检查碰撞和输出数据。相关基础见[坐标变换](../Note_Basics.md#basic-transforms)、[IK](../Note_Basics.md#basic-ik)和[标定](../Note_Basics.md#basic-calibration)。

## 1. 输入与输出

输入字段为observation.state，每帧16D：

| 手 | 内容 |
|---|---|
| 左手 | xyz(3)+quaternion xyzw(4)+gripper(1) |
| 右手 | xyz(3)+quaternion xyzw(4)+gripper(1) |

Hy-UMI夹爪默认约定为0=open、90=closed。输出为ARX X5双臂14D：

~~~text
[left_joint1..6, left_gripper,
 right_joint1..6, right_gripper]
~~~

同时可写入observation.eef_pose，保存由IK结果经FK得到的机器人实际末端姿态。输入手部pose与输出机器人EEF pose不是同一个坐标或物理定义。

## 2. 坐标系与位姿解析

### 2.1 Hy-UMI task frame

默认input_to_task为Identity，任务坐标约定为+X forward、+Y left、+Z up：

~~~text
p_task = p_hy
R_task = R_hy
~~~

如果数据和任务坐标存在固定轴向差异，通过配置提供input_to_task；position_scale默认1.0，位置单位不缩放。

### 2.2 齐次位姿

每帧将左右手的xyz和quaternion解析为：

~~~text
T_task_hand = [ R_task_hand  p_task_hand ]
               [     0           1       ]
~~~

这里的手部pose被解释为夹爪根部或手部根部，不直接假设它已经是ARX X5的TCP。默认left_ee_frame和right_ee_frame为对应的gripper_base，hand_to_ee为Identity；如果需要工具轴或位置补偿，再右乘固定T_hand_to_ee。

## 3. 自动搜索机器人root

Hy task frame原点通常与ARX X5 robot root不重合。配置中的task_from_root表示：

~~~text
T_task_from_root：robot root → Hy task
~~~

因此目标从task系转换到root系时使用：

~~~text
T_root_target = inverse(T_task_from_root) × T_task_target
~~~

搜索不是只看单帧，而是对整条轨迹采样，依次：

1. 统计左右手目标位置和中心；
2. 读取ARX neutral pose下左右末端位置；
3. 枚举候选旋转和平移；
4. 在均匀采样帧上执行IK和碰撞检查；
5. 选择误差小且整条轨迹可行的候选；
6. 用完整轨迹再次验证。

默认sample_count为16，require_feasible和validate_full_trajectory开启。候选角度通常包含-30°、0°、30°，平移测试约±0.15m及竖直方向约±0.10m。最终变换写入episode.source_meta，供复现和审计。

## 4. root到左右arm base

ARX X5左右臂有独立base frame。若T_root_base表示base到root，则：

~~~text
T_base_target =
    inverse(T_root_base)
    × inverse(T_task_from_root)
    × T_task_hand
    × T_hand_to_ee
~~~

默认T_hand_to_ee为Identity。该结果才是传给对应IK solver的base-frame目标。

## 5. 逐帧IK与约束

左右臂分别使用6DoF URDF chain和Pinocchio模型，求解器采用damped least-squares。初始值优先使用neutral pose，后续帧使用上一帧结果warm start，以减少跳解。

每帧约束包括：

- URDF joint limits；
- 最大单帧joint变化，默认0.35rad；
- 位置误差不超过0.03m；
- 姿态误差不超过20°；
- 左右臂与夹爪的全模型碰撞检查。

若fail_on_ik_error开启，任一关键帧无法满足约束则episode失败。最大步长限制用于抑制关节跳变，但不能修复错误的坐标系或不可达目标。

## 6. 夹爪值转换

先将Hy夹爪值转换为统一openness：

~~~text
openness = (gripper - closed_value) / (open_value - closed_value)
          = (90 - gripper) / 90
~~~

再按ARX preset映射为closed=0、open=1，因此：

| Hy值 | openness / ARX command |
|---:|---:|
| 90 | 0，闭合 |
| 45 | 0.5 |
| 0 | 1，张开 |

同一openness还用于驱动碰撞模型中的夹爪内部关节，保证碰撞状态和输出command一致。

## 7. 碰撞检查

两臂IK完成后，将左右臂joint和夹爪内部joint合成完整URDF configuration，检查：

- 左臂自身碰撞；
- 右臂自身碰撞；
- 左右臂之间碰撞；
- 夹爪与机器人其他部件碰撞。

fail_on_collision开启时，检测到碰撞的episode直接失败。碰撞检查应在IK之后进行；仅检查单臂关节限位不能替代全模型碰撞检查。

## 8. 最终数据流

~~~text
Hy observation.state
→ 解析左右手xyz + quaternion
→ Hy frame → task frame
→ 搜索task_from_root
→ task frame → robot root → arm base
→ 左右臂IK
→ joint limit / step limit / pose error
→ gripper映射与内部关节更新
→ 全模型碰撞检查
→ 输出14D joint/action与FK pose
~~~

默认情况下，T_task_hand=T_hy_hand、T_hand_to_ee=Identity。完整处理和多源统一见[数据处理笔记](../Note_DataPipeline.md)。

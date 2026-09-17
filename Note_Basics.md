# 基础知识

整理机器人、RL与VLA的常用原理。核心项目和面试复盘见[项目笔记](Note_Project.md)，早期实现过程见[辅助项目记录](Note_OtherProjects.md)。

- [运动学与动力学](#运动学与动力学)
- [坐标变换](#坐标变换)
- [IK 逆运动学](#ik-逆运动学)
- [规划算法](#规划算法)
- [ROS2 基本概念](#ros2-基本概念)
- [VLA / Transformer 基础模块](#basic-vla)
- [动作表示与归一化](#basic-normalization)
- [强化学习：PPO与GAE](#basic-rl)
- [相机、手眼标定与时间同步](#basic-calibration)
- [路径、轨迹与反馈控制](#basic-control)
- [模型评测](#basic-evaluation)

<a id="basic-dynamics"></a>

## 运动学与动力学

运动学描述关节与末端位姿的关系；动力学描述力矩与运动的关系。

### 基本动力学模型

τ = D(q)q̈ + C(q, q̇) + G(q)

| 符号             | 维度           | 含义            |
| -------------- | ------------ | ------------- |
| $\tau$         | $n \times 1$ | 关节驱动力矩向量      |
| $D(q)$         | $n \times n$ | 质量/惯性矩阵（正定对称） |
| $\ddot{q}$     | $n \times 1$ | 关节加速度向量       |
| $C(q,\dot{q})$ | $n \times 1$ | 科氏力 + 离心力向量   |
| $G(q)$         | $n \times 1$ | 重力向量          |

### 动力学分析方法 牛顿欧拉法

目标是给定机器人的关节位置q、速度q̇ 和加速度q̈，计算出为了产生这个运动状态，每个关节需要施加多大的驱动力矩 τ

<a id="basic-transforms"></a>

## 坐标变换

### 欧拉角

#### 1. 定义

**Z-Y-X欧拉角**（又称 **航向-俯仰-横滚**，Yaw-Pitch-Roll）：

- 先绕 **Z** 轴旋转 α（Yaw）  
- 再绕 **新Y** 轴旋转 β（Pitch）  
- 最后绕 **新X** 轴旋转 γ（Roll）

角度向量：

$$
[\alpha,\ \beta,\ \gamma]
$$

---

#### 2. 单轴基础旋转矩阵

绕当前坐标系 `i` 的各轴旋转角度 `θ` 时，对应的 **3×3旋转矩阵** 如下：

**绕Z轴旋转 α（Yaw）：**

$$
R_z(\alpha)=
\begin{bmatrix}
\cos\alpha & -\sin\alpha & 0 \\
\sin\alpha &  \cos\alpha & 0 \\
0          &  0          & 1
\end{bmatrix}
$$

**绕Y轴旋转 β（Pitch）：**

$$
R_y(\beta)=
\begin{bmatrix}
\cos\beta  & 0 & \sin\beta \\
0          & 1 & 0         \\
-\sin\beta & 0 & \cos\beta
\end{bmatrix}
$$

**绕X轴旋转 γ（Roll）：**

$$
R_x(\gamma)=
\begin{bmatrix}
1 & 0          & 0           \\
0 & \cos\gamma & -\sin\gamma \\
0 & \sin\gamma &  \cos\gamma
\end{bmatrix}
$$

---

#### 3. 合成旋转矩阵（Z-Y-X 欧拉角 → 旋转矩阵）

按**当前轴顺序**右乘：

$$
R = R_z(\alpha)\cdot R_y(\beta)\cdot R_x(\gamma)
$$

展开结果：

$$
R =
\begin{bmatrix}
\cos\alpha\cos\beta &
\cos\alpha\sin\beta\sin\gamma - \sin\alpha\cos\gamma &
\cos\alpha\sin\beta\cos\gamma + \sin\alpha\sin\gamma \\[6pt]
\sin\alpha\cos\beta &
\sin\alpha\sin\beta\sin\gamma + \cos\alpha\cos\gamma &
\sin\alpha\sin\beta\cos\gamma - \cos\alpha\sin\gamma \\[6pt]
-\sin\beta &
\cos\beta\sin\gamma &
\cos\beta\cos\gamma
\end{bmatrix}
$$

---

### 坐标变换的两种解释：固定系左乘 vs 运动系右乘

> 一句话区别：
> **固定轴（外部视角）→ 左乘（Premultiply）**
> **运动轴（本体视角）→ 右乘（Postmultiply）**

---

#### 1 固定坐标系解释（Fixed-Frame / 左乘）

| 说明 | 操作顺序 | 矩阵乘法顺序 |
|---|---|---|
| 所有变换都**绕最开始的固定系**进行 | 先旋转 → 再平移 | `T = Trans(d) · Rot(R)` |
| 几何意义：把物体在“世界”里依次摆位 | 步骤1：绕 `[0]` 旋转 → 得到 `[0']` <br> 步骤2：沿 `[0]` 平移 → 得到 `[1]` | 左乘 |

---

#### 2 运动坐标系解释（Moving-Frame / 右乘）

| 说明 | 操作顺序 | 矩阵乘法顺序 |
|---|---|---|
| 每一步变换都**绕最新建立的动系**进行 | 先平移 → 再旋转 | `T = Trans(d) · Rot(R)` |
| 几何意义：把物体“装”在动系上，再让动系自己动 | 步骤1：沿 `[0]` 平移 → 得到 `[0']` <br> 步骤2：绕**新轴 `[0']`** 旋转 → 得到 `[1]` | 右乘 |

<a id="basic-rotation"></a>

### 旋转表示与相对位姿

位置差可以直接相减；姿态差需要先明确坐标系和组合顺序。以列向量、主动旋转为例：

- 世界系增量：`R_delta = R_target @ R_current.T`，恢复为 `R_target = R_delta @ R_current`。
- 本体系增量：`R_delta = R_current.T @ R_target`，恢复为 `R_target = R_current @ R_delta`。

四元数也可用乘法和逆表示旋转增量；直接相减四个分量不是同一回事。`q` 与 `-q` 表示同一姿态，回归或插值时需处理符号一致性。

Rotation-6D用旋转矩阵的两个列向量表示姿态，解码时先正交化，再用叉积补齐第三列。它便于连续回归，但任意网络输出仍需投影到合法旋转；展平顺序也必须在编码与解码间一致。

齐次位姿 `T_A_B` 表示B到A的变换，组合规则是 `T_A_C = T_A_B @ T_B_C`。把世界系EEF转到相机系时，使用 `T_C_E = inverse(T_W_C) @ T_W_E`。移动相机下，还要标清每个位姿对应的时间。

<a id="basic-ik"></a>

## IK 逆运动学

正：已知 θ→T，用DH/MDH递推
逆：已知T→θ，解析法（Pieper准则，三轴交于一点）或数值法（Newton-Raphson、Jacobian伪逆、LM）。

### 解析法

解析法适用于结构较为简单的机械臂，可以通过几何或者代数方法进行求解。
对于特定六自由度机械臂，三个相邻关节轴交于一点或相互平行，是可利用的解析求解结构条件；不能反过来认为其他结构都没有解析解。

### 数值法

常见数值IK通过雅可比线性化反复修正关节角。下面介绍雅可比逆、伪逆和阻尼伪逆；也可将IK写成带约束的优化问题。

#### 1. 背景与目标

> 给定末端目标位姿 $\mathbf{x}_d \in \mathbb{R}^m$，求关节角度 $\mathbf{q} \in \mathbb{R}^n$，使 $\mathbf{x}_d = f(\mathbf{q})$。

$f$ 通常非线性，所以用「**线性化 + 迭代**」逼近解。

#### 2. 核心关系：雅可比把关节速度映射到末端速度

$$
\dot{\mathbf{x}} = \mathbf{J}(\mathbf{q}) \dot{\mathbf{q}}
$$

- $\dot{\mathbf{x}} \in \mathbb{R}^m$：末端任务空间速度，常用6×1 = 线速度v(3) + 角速度 ω(3)。
- $\dot{\mathbf{q}} \in \mathbb{R}^n$：n个关节各自的转动速度。
- $\mathbf{J}(\mathbf{q}) \in \mathbb{R}^{m \times n}$：雅可比矩阵，依赖当前关节角。
- 直觉：**关节空间的小速度 → 任务空间的小速度**。"如果我把每个关节拧快一点点，末端会往哪边跑、跑多快？"

#### 3. 一阶雅可比迭代法

**初始化**：给定 $\mathbf{q}_0$、目标 $\mathbf{x}_d$、误差容限 $\epsilon$、最大迭代次数 $N$。

**第k次迭代**：

1. 当前位姿：$\mathbf{x}_k = f(\mathbf{q}_k)$
2. 误差：$\Delta \mathbf{x}_k = \mathbf{x}_d - \mathbf{x}_k$
3. 计算雅可比 $\mathbf{J}(\mathbf{q}_k)$
4. 求关节速度：
   - 非冗余（$m = n$）：$\dot{\mathbf{q}}_k = \mathbf{J}^{-1}(\mathbf{q}_k) \Delta \mathbf{x}_k$
   - 冗余（$m < n$）：$\dot{\mathbf{q}}_k = \mathbf{J}^+(\mathbf{q}_k) \Delta \mathbf{x}_k$，其中 $\mathbf{J}^+ = \mathbf{J}^T (\mathbf{J} \mathbf{J}^T)^{-1}$ 是 **Moore–Penrose伪逆**
5. 更新关节：$\mathbf{q}_{k+1} = \mathbf{q}_k + \alpha \dot{\mathbf{q}}_k$（$\alpha$ 是小步长）
6. 收敛判断：$\|\Delta \mathbf{x}_k\| < \epsilon$ 则停止；否则继续

> 步骤4直觉：把"末端还差的速度"转换成"关节要补的速度"，就像齿轮比——末端差1 mm/s，关节需要转多少rad/s才能补上。

#### 4. 伪逆法（冗余机械臂）

当 $n > m$ 系统**欠定**，有无穷多解。伪逆法求其中的**最小范数解**（能量最小）：

$$
\dot{\mathbf{q}} = \mathbf{J}^+ \dot{\mathbf{x}}
$$

- 在所有能满足末端速度 $\Delta \mathbf{x}$ 的关节速度里，挑一个"总转速最小"的；
- 配合零空间投影还可以加避障 / 避奇异 / 关节限位等次级目标。

#### 5. 阻尼伪逆（奇异问题）

当 $\mathbf{J}$ 接近奇异（$\det \mathbf{J} \to 0$）时伪逆法数值不稳定。

$$
\mathbf{J}^* = \mathbf{J}^T (\mathbf{J} \mathbf{J}^T + \lambda^2 \mathbf{I})^{-1}
$$

- $\lambda$：阻尼因子，通常 $0.01 \sim 0.1$
- 牺牲少量精度，换取数值稳定性

#### 6. 三种方法对比

| 方法 | 适用场景 | 优点 | 缺点 |
|------|----------|------|------|
| 雅可比逆法 | 非冗余机械臂（n = m） | 简单、直接 | 奇异时失效 |
| 伪逆法 | 冗余机械臂（n > m） | 最小范数解 | 奇异时不稳定 |
| 阻尼伪逆法 | 所有构型 | 鲁棒性强 | 精度略降 |

---

<a id="basic-planning"></a>

## 规划算法

### RRT*算法

#### 1 算法定位

| 维度        | 内容                                      |
|-------------|-------------------------------------------|
| 类型        | 基于采样的运动规划 |
| 解决什么问题 | 高维连续C-space中找**可行**→**最优**路径 |
| 对比RRT    | RRT只保证概率完备，RRT\* 额外持续优化，成本→c\* |

#### 2 核心思想一句话

> 在RRT随机扩张的基础上，**新增两步**：选父节点时选最小代价（不只是最近）；并对邻居做 **rewire**，让新采样点反过来优化已有路径。

#### 3 算法流程

RRT基础流程：

1. 初始化环境参数
2. 随机采样x_rand
3. 寻找树中最近点x_nearest
4. 沿x_nearest → x_rand方向生长得到x_new
5. 碰撞检测
6. 无碰撞则将x_new加入树

RRT\* 在第6步前后多两步：

- **(a) 重选父节点（Choose Parent）**：在x_new半径r内找到所有候选父节点，计算"经过候选父节点到达x_new的总代价"，选代价最小的作为真正父节点。
- **(b) 重写邻居（Rewire）**：再次遍历半径r内的邻居，判断"经过x_new到达邻居"是否比邻居当前路径更短，是则把邻居的父节点改成x_new。

### A*算法

#### 1 算法定位

| 维度 | 内容 |
|------|------|
| 类别 | 图搜索 + 启发式（Informed Search） |
| 完备性 | 有限图等常见条件下完备 |
| 最优性 | 与启发式及节点重开策略有关；一致启发式可支持常见的闭集图搜索 |
| 时间复杂度 | O(b^d)（最坏，b分支因子，d解深度） |
| 空间复杂度 | O(b^d)（OPEN与CLOSED表） |
| 典型场景 | 2D栅格导航、3D UAV体素、任务级图、多关节离散网格 |

#### 2 核心思想一句话

> 每次展开 `f(n)=g(n)+h(n)` 最小的节点。`g` 是已走代价，`h` 是剩余代价估计；最优性还需满足对应的启发式和搜索条件。

#### 3 启发式函数

欧几里得距离：直线距离，开根号
曼哈顿距离：网格地图，只允许四向移动，不用开根号 |x1-x2| + |y1-y2|

<a id="basic-ros2"></a>

## ROS2 基本概念

### 节点与通信

- **节点（Node）**：ROS2的基本计算单元，单一职责，通过话题 / 服务 / 参数与其他节点通信。
- **话题通信**：发布/订阅（Pub/Sub）模式，异步、多对多，发布者和订阅者通过话题名、消息类型与QoS匹配；实际能否接收取决于匹配、网络和可靠性设置。

### Launch

Launch文件统一组织节点启动、参数和名称映射。以下是项目中的两个例子：

| Launch文件 | 职责 | 包含 |
|---|---|---|
| `robot_gazebo.launch.py` | 仿真环境 + 机器人模型 | `robot_state_publisher`（发模型） + Gazebo仿真环境 + 在Gazebo中生成机器人实例 + 加载并激活控制器 |
| `navi_launch.py` | 导航功能 | Nav2导航框架 + 自定义A* 规划器 + RViz2可视化 |

### Gazebo 仿真集成

| 组件 | 作用 |
|---|---|
| `gazebo_ros` 包 | ROS2与Gazebo之间的桥接 |
| URDF模型 | 定义机器人和环境模型 |
| Gazebo插件 | 连接Gazebo物理引擎与ROS2接口 |
| `ros2_control` | 硬件抽象层，连接控制器与仿真 |

<a id="basic-vla"></a>

## VLA / Transformer 基础模块

<a id="basic-attention"></a>

### 1. Self-Attention

Self-Attention让序列中的每个token根据其他token的信息更新自身表示。输入为 `X ∈ R^{B×L×D}`，先通过三个线性层得到Query、Key和Value：

$$
Q=XW_Q,\quad K=XW_K,\quad V=XW_V
$$

注意力计算为：

$$
S=\frac{QK^T}{\sqrt{d_k}},\quad A=\operatorname{softmax}(S+M),\quad Y=AV
$$

其中 `M` 是attention mask，最后通常再经过一个输出投影 `W_O`。

```python
def self_attention(x, w_q, w_k, w_v, w_o, mask=None):
    q = x @ w_q
    k = x @ w_k
    v = x @ w_v
    score = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
    if mask is not None:
        score = score.masked_fill(~mask, float("-inf"))
    weight = torch.softmax(score, dim=-1)
    return (weight @ v) @ w_o
```

在MagicVLA中，Qwen的full-attention层以及Action Expert的 `QwenJointFullAttention` 都建立在这个公式上。

<a id="basic-masked-attention"></a>

### 2. Masked Attention

Mask的作用是限制某个Query可以读取哪些Key。常见类型有：

- **Causal mask**：当前位置不能读取未来token，用于语言模型；
- **Padding mask**：忽略补齐位置；
- **非对称mask**：不同模态之间采用不同的可见性。

MagicVLA的full-attention逻辑可以概括为：

```text
VLM query     -> 只能读取 VLM prefix
Action query  -> 可以读取 VLM prefix 和整个 action chunk
```

因此VLM不会读取noisy action，避免动作噪声污染视觉语言表示；Action Expert可以使用完整的视觉语言条件和action chunk内部信息。

最小的masked attention写法如下：

```python
score = q @ k.transpose(-2, -1) / math.sqrt(d)
score = score.masked_fill(~allowed, float("-inf"))
attn = torch.softmax(score, dim=-1)
out = attn @ v
```

<a id="basic-rmsnorm"></a>

### 3. RMSNorm

RMSNorm只根据均方根缩放特征，不计算均值：

$$
\operatorname{RMS}(x)=\sqrt{\frac{1}{D}\sum_{i=1}^{D}x_i^2+\epsilon}
$$

$$
\operatorname{RMSNorm}(x)=\frac{x}{\operatorname{RMS}(x)}\odot\gamma
$$

最小实现：

```python
def rms_norm(x, weight, eps=1e-6):
    rms = torch.sqrt(x.float().square().mean(-1, keepdim=True) + eps)
    return (x.float() / rms * weight.float()).to(x.dtype)
```

MagicVLA的 `QwenRMSNorm` 使用 `1 + weight` 作为缩放因子，使参数初始化为0时接近恒等映射。

<a id="basic-swiglu"></a>

### 4. SwiGLU

SwiGLU是Qwen使用的MLP结构，由gate分支、up分支和down分支组成：

$$
\operatorname{SwiGLU}(x)=\left[\operatorname{SiLU}(xW_g)\odot(xW_u)\right]W_d
$$

```python
def swiglu(x, gate_proj, up_proj, down_proj):
    gate = torch.nn.functional.silu(gate_proj(x))
    up = up_proj(x)
    return down_proj(gate * up)
```

在Action Expert中，当前维度大致是：

```text
1024 -> 3072 -> 1024
```

它替代普通的 `Linear -> GELU -> Linear`，通过gate控制不同特征的保留程度。

<a id="basic-rope"></a>

### 5. RoPE

RoPE通过旋转Query和Key来编码位置信息。二维形式为：

$$
\begin{bmatrix}x_1'\\x_2'\end{bmatrix}
=
\begin{bmatrix}\cos\theta & -\sin\theta\\
\sin\theta & \cos\theta\end{bmatrix}
\begin{bmatrix}x_1\\x_2\end{bmatrix}
$$

其中 `θ` 由token的位置决定。对Q、K同时施加旋转后，内积自然包含相对位置信息。

```python
def apply_rope(x, cos, sin):
    x1, x2 = x.chunk(2, dim=-1)
    rotated = torch.cat([-x2, x1], dim=-1)
    return x * cos + rotated * sin
```

MagicVLA使用Qwen的RoPE。图像侧使用多模态的3D position id，动作token则使用连续的action position；这样模型可以区分不同时间步的动作以及图像中的空间位置。

<a id="basic-flow"></a>

### 6. Flow Matching

Flow Matching让模型学习从噪声动作到真实动作的连续变化方向。设真实动作为 `x_0`，随机噪声为 `ε`，随机时间为 `t∈[0,1]`：

$$
x_t=(1-t)x_0+t\epsilon
$$

对于线性路径，目标速度为：

$$
u_t=\frac{dx_t}{dt}=\epsilon-x_0
$$

模型输入带噪动作 `x_t`、时间 `t`、state和VLM条件，输出预测速度：

```python
noisy_action = (1 - t) * action + t * noise
target_velocity = noise - action
pred_velocity = action_expert(noisy_action, t, state, vlm_context)
```

推理时从高斯噪声开始，沿反方向用Euler方法逐步更新：

```python
action = torch.randn_like(action)
for _ in range(num_steps):
    velocity = model(action, time, condition)
    action = action - velocity / num_steps
```

<a id="basic-masked-mse"></a>

### 7. Masked MSE

MagicVLA的主要动作损失是预测速度和目标速度之间的均方误差：

$$
L_{MSE}=\frac{1}{N}\sum_i m_i(\hat{v}_i-v_i)^2
$$

其中 `m_i` 表示该动作元素是否有效。项目中需要同时考虑action padding和动作维度mask：

```python
valid = (
    ~action_is_pad.unsqueeze(-1)
) & action_dim_mask

error = (pred_velocity - target_velocity).square()
loss = (error * valid).sum() / valid.sum().clamp_min(1)
```

这样可以避免：

- 被标记为无效的补齐动作参与训练；
- 不存在的机器人维度参与训练；
- 缺失动作被错误当成真实的0。

尾部重复动作也可以被定义为有效的终点保持监督。是否屏蔽取决于数据语义；Masked MSE只负责执行给定的有效性规则。

<a id="basic-cross-attention"></a>

### 8. Cross-Attention

Cross-Attention和Self-Attention的区别是：Query和Key/Value来自不同序列。

```text
Query：当前 action token
Key/Value：历史 memory token
```

公式为：

$$
Q=X_{current}W_Q,\quad K=X_{memory}W_K,\quad V=X_{memory}W_V
$$

$$
Y=\operatorname{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

最小实现：

```python
def cross_attention(query, memory, w_q, w_k, w_v):
    q = query @ w_q
    k = memory @ w_k
    v = memory @ w_v
    weight = torch.softmax(
        q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1]),
        dim=-1,
    )
    return weight @ v
```

在RoboMME-style memory中，Action Expert用当前动作特征作为Query，历史视觉特征作为Key/Value；在MagicVLA Base中，类似的条件读取发生在full-attention层，只是Action Expert同时读取VLM prefix和action chunk。

<a id="basic-normalization"></a>

## 动作表示与归一化

动作表示决定预测什么；归一化只改变数值尺度。两者按顺序应用，并在推理时逆序恢复。

```text
绝对目标 → 按语义组构造 delta → normalize → 模型
模型输出 → inverse normalize → 恢复绝对目标 → 控制接口
```

常见归一化方式：

- mean/std：`z = (x - mean) / max(std, eps)`。
- quantile：`z = 2 * (x - q01) / max(q99 - q01, eps) - 1`，是否裁剪由配置决定。

裁剪会丢失区间外的信息，因此带裁剪的变换不是严格可逆。缺失维度需要独立mask，不能只靠补零识别。

`[D]` 统计量按动作维度归一化；`[H,D]` 统计量还区分chunk内位置，适合不同预测位置尺度差异明显的情况。统计量必须与动作表示、维度、chunk长度和source配套。

<a id="basic-rl"></a>

## 强化学习：PPO、GAE与任务设计

强化学习通过与环境交互优化累计回报。策略根据观测输出动作，环境返回奖励和下一观测；Actor学习动作分布，Critic估计未来回报。

### PPO与GAE

PPO限制新旧策略之间的更新幅度，避免一次优化使行为变化过大。令`r_t = π_new(a_t|s_t) / π_old(a_t|s_t)`，最大化：

```text
L_clip = E[min(r_t × A_t, clip(r_t, 1-ε, 1+ε) × A_t)]
```

`A_t`是优势，表示该动作相对当前平均水平好多少。实现为最小化loss时，需要对上述目标取负号；价值损失和熵项另行加权。

GAE将不同跨度的时序差分误差组合，平衡估计偏差与方差：

```text
δ_t = reward_t + γ × V(s_next) - V(s_t)
A_t = δ_t + γλ × δ_(t+1) + (γλ)² × δ_(t+2) + …
```

真正终止时不再从终点后bootstrap；由时间上限造成的截断通常仍需使用截断时的最终观测估计价值，不能误用重置后的观测。

### 任务设计与排查

- 观测提供完成任务所需的信息，尽量减少与目标无关的坐标变化。
- 动作范围、控制频率与机器人能力匹配。
- 成功奖励确定任务目标，接近、对齐等稠密奖励帮助探索；检查是否出现只刷奖励而不完成任务。
- Curriculum逐步增加难度；Domain Randomization改变初始状态、物理或视觉条件，检验泛化。
- 同时检查成功率、失败轨迹、奖励分项、价值误差、熵和策略变化，不能只看总奖励。

项目应用见[RL实践](Note_OtherProjects.md#rl-project)。

<a id="basic-calibration"></a>

## 相机、手眼标定与时间同步

| 概念 | 解决的问题 |
|---|---|
| 相机内参 | 三维相机坐标如何投影成像素，包括焦距、主点与畸变 |
| 相机外参 | 相机相对世界、机器人或其他传感器的位置和方向 |
| 手眼标定 | 求相机与末端或基座之间的固定变换 |
| 时间同步 | 图像与位姿是否描述同一时刻 |

针孔投影可写成`λ[u,v,1]ᵀ = K × p_camera`。投影前先把三维点变到相机系，实际图像还需考虑畸变。外参方向应明确为`T_A_B`，不要只写“相机矩阵”。

手眼标定常用多组运动关系构造`AX=XB`，其中`X`是待求固定变换；`A/B`如何构造取决于eye-in-hand或eye-to-hand布局及坐标约定。采样应覆盖不同位置与旋转方向，避免单一运动造成退化。

检查标定时使用重投影、已知点位和独立样本。像素残差小不自动代表三维误差小，尤其当对应点有偏差或视角覆盖不足时。

时间同步是独立问题：即使外参正确，图像与tracking错时也会导致overlay偏移。补偿前先检查时间戳单位、起点、帧率和延迟，再观察是否存在随时间漂移。固定offset适用于近似恒定延迟，不代表能处理所有抖动。

基础：[坐标变换](#basic-transforms)、[相对位姿](#basic-rotation)。

<a id="basic-control"></a>

## 路径、轨迹与反馈控制

路径只规定经过哪些位置；轨迹还规定何时到达，以及速度、加速度等时间约束。几何路径无碰撞，不等于按任意速度执行都安全。

| 内容 | 重点 |
|---|---|
| 三次多项式 | 可约束两端位置与速度 |
| 五次多项式 | 可增加两端加速度约束；段间jerk连续还需额外条件 |
| 时间参数化 | 给路径分配时间，满足速度、加速度等限制 |
| 碰撞检查 | 同时考虑环境碰撞、自碰撞和采样点之间的路径 |

PID根据误差生成控制量：`u = Kp·e + Ki·∫e dt + Kd·de/dt`。P响应当前误差，I累积误差，D响应误差变化。整定时关注执行器饱和、积分累积、测量噪声和采样延迟。

轨迹抖动可按顺序排查：目标是否连续 → IK是否换解 → 时间参数是否合理 → 控制增益与限幅 → 通信频率和延迟。不要把所有抖动都归因于模型预测。

<a id="basic-evaluation"></a>

## 模型评测：从训练loss到闭环任务

| 指标 | 能说明什么 | 不能直接说明什么 |
|---|---|---|
| 训练/验证loss | 拟合目标的程度及分布差异 | 机器人一定能完成任务 |
| Open-loop动作误差 | 固定观测下预测与标签的差异 | 执行动作后的状态变化与误差累积 |
| 闭环成功率 | 重复观测、执行和重规划后的任务表现 | 收益一定来自某个新增模块 |
| 延迟、显存、控制频率 | 部署成本与响应能力 | 策略本身的任务能力 |

对比模型时固定任务、数据划分、训练预算和评估协议，同时记录测试次数、成功标准与失败类型。相邻帧高度相关，数据划分优先按episode、场景或采集批次组织，避免训练与验证泄漏。

验证模块作用时加入消融：例如正确历史、无历史和打乱历史；或保留、屏蔽World信息。Attention热图和latent loss用于解释与诊断，闭环结果用于判断任务收益。

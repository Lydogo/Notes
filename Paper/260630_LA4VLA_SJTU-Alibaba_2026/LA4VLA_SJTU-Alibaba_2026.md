# LA4VLA：先不看图像，显式学习语言到动作的 VLA 预训练

> 原标题：LA4VLA: Learning to Act without Seeing via Language-Action Pretraining  
> 作者：Tao Lin, Yuxin Du, Yiran Mao, Zewei Ye, Yilei Zhong, Bing Cheng, Yiming Wang, Jiting Liu, Yang Tian, Junchi Yan, Feiran Wu, Zenan Meng, Hu Wei, Yuqian Fu, Gen Li, Bo Zhao  
> 机构：Shanghai Jiao Tong University, Alibaba Group, Nanyang Technological University, KAUST  
> 发表：arXiv:2606.27295v2, 2026-06-26  
> 链接：https://arxiv.org/abs/2606.27295  
> 项目/开源：https://github.com/MINT-SJTU/LA4VLA  

---

## 一、研究背景与动机

标准 VLA 预训练把图像、语言指令和动作轨迹一起输入模型。这个范式有效，但有一个结构性偏置：视觉-动作监督非常密集，一条轨迹有大量 image-action pairs；语言通常只有一个高层任务指令，语义变化稀疏。模型很容易学到视觉 shortcut，看起来在跟随指令，实际动作预测却主要被图像状态牵引。

LA4VLA 的出发点很直接：能不能先把视觉拿掉，只让模型从语言和 proprioceptive state 预测动作，从而显式学习语言如何约束动作？这不是要替代视觉 grounding，而是让模型先获得 reusable language-conditioned action priors，再回到完整 VLA 微调。

## 二、核心贡献

1. 提出 LA4VLA，一种 vision-agnostic language-action pretraining 框架，用来解耦语言-动作学习和视觉 grounding。
2. 从已有 DROID 示范中自动分割原子动作片段，并配低层动作描述，构建 LA-33K：33,116 条 language-action episodes，不需要额外采集机器人数据。
3. 系统比较 LA-only、VLA-only、LA-to-VLA sequential pretraining 和 mixed LA-VLA pretraining，证明 LA supervision 对 VLA 下游性能有稳定增益。
4. 通过视觉移除、视觉冲突和表示分析说明，标准 VLA 容易被视觉输入主导，而 LA pretraining 能让动作表示更按指令聚类。

## 三、方法原理

### 3.1 整体框架

LA4VLA 先把完整 demonstration trajectory 分解成短的 atomic action segments，每个 segment 配一个低层、去视觉化的动作指令。LA 预训练时，模型看不到图像，只接收语言指令和机器人状态，然后预测动作轨迹。之后再进入下游 VLA fine-tuning 或与 VLA-format 数据混合预训练。

作者实现了 LA4VLA-1B：基于 InternVL3-1B，接 flow matching action head。核心实验保持模型结构、训练预算、优化超参和下游数据一致，只改变预训练范式。

### 3.2 关键技术细节

**视觉依赖诊断。** 作者先训练一个标准 VLA policy，然后固定语言指令和机器人状态，改变视觉输入：标准配对图像、移除图像、同场景不对齐图像、相反方向冲突图像。若模型真的理解语言，动作方向应主要跟随指令；若动作随图像改变，则说明语言 grounding 脆弱。

**LA-33K 构建。** 从 DROID 的 9,560 条原始 VLA episodes 出发，先用 robot state 中的静止段和夹爪变化检测 keyframe hints，再定义原子动作词表，包括 grasp/place/lift/lower/transport/push/pull/press/move/rotate/reorient 等。然后用 Qwen-3-VL-Plus 对视频做 temporal segmentation，输出 primitive label、vision-agnostic instruction 和起止时间。最后人工验证，保留质量分不低于 2 的片段。

**Atomic instruction 的意义。** 标准 VLA 一整条轨迹可能只有一句“把杯子放到盘子上”，语言监督很稀；LA4VLA 把它拆成“移动到物体附近”“闭合夹爪抓取”“向上抬起”“水平搬运”“打开夹爪放置”等短片段，使语言和局部动作更密集对齐。

**三种结合方式。**

| 范式 | 做法 |
|---|---|
| LA | 只用 LA-33K 做无视觉预训练，再下游微调 |
| VLA | 用恢复图像的 LA-33K-V 做标准 VLA 预训练 |
| LA-VLA | 先 LA，再 VLA，最后下游微调 |
| MixPT | 同一预训练阶段混合 LA-33K 和 LA-33K-V |

### 3.3 训练与优化

LA-format batch 中视觉被 mask，模型从语言和 robot state 预测 action trajectory。VLA-format batch 和下游 fine-tuning 中启用视觉输入。真实机器人实验使用 xArm6、平行夹爪、腕部相机和第三视角相机；每个真实任务 100 条遥操作示范，共 300 条，多任务微调一个 checkpoint。

## 四、实验与结果

### 4.1 实验设置

实验覆盖 MetaWorld、LIBERO、StarVLA 架构迁移、真实 xArm6 任务、视觉噪声鲁棒性和方向跟随诊断。真实任务包括 Press Button、Place Book、Place Drink，目标位置由语言指令指定；同一任务内视觉初始场景相近，因此语言必须决定具体目标。

### 4.2 主要结果

| 模型/设置 | MetaWorld Avg. | LIBERO Avg. | 结论 |
|---|---:|---:|---|
| LA4VLA-1B No pretrain | 69.73 | 92.85 | 直接微调基线 |
| LA4VLA-1B VLA | 79.78 | 94.40 | 标准视觉-语言-动作预训练有效 |
| LA4VLA-1B LA | 83.00 | 95.30 | LA-only 优于 VLA-only |
| LA4VLA-1B LA-VLA | 86.75 | 96.28 | LIBERO 最好 |
| LA4VLA-1B MixPT | 87.53 | 95.75 | MetaWorld 最好 |

在真实 xArm6 任务上，LA pretraining 的增益更大：

| Pretrain | Press Button | Place Book | Place Drink | Avg. |
|---|---:|---:|---:|---:|
| No | 60.0 | 15.0 | 40.0 | 38.3 |
| VLA | 50.0 | 40.0 | 55.0 | 48.3 |
| LA | 85.0 | 65.0 | 95.0 | 81.7 |
| MixPT | 75.0 | 85.0 | 90.0 | 83.3 |

这组结果很强，因为真实任务中的目标由语言指定，视觉本身无法完全 disambiguate。LA 和 MixPT 远高于 VLA-only，支持“语言-动作先验”确实有用。

### 4.3 消融实验

**标准 VLA 的视觉依赖诊断：**

| 输入条件 | DAR | DCS | SR | SS |
|---|---:|---:|---:|---:|
| Standard paired input | 0.98 | 0.95 | 2.35 | 0.55 |
| Visual-removed input | 0.63 | 0.16 | 1.03 | 0.04 |
| Visual-unaligned input | 0.66 | 0.37 | 1.13 | 0.05 |
| Visual-conflict input | 0.35 | 0.03 | 1.03 | -0.04 |

最关键的是 Visual-conflict：DAR 掉到 0.35，低于随机方向选择的 0.5，说明模型不只是变得不确定，而是被冲突视觉输入带向了与语言相反的方向。

**LA-pretrained policy 的方向跟随：**

| 条件 | DAR | DCS | SR | SS |
|---|---:|---:|---:|---:|
| LA-pretrained | 0.99 | 0.85 | 1.46 | 0.20 |
| Conflicting state | 0.95 | 0.80 | 1.41 | 0.16 |

在无视觉输入下，LA-pretrained policy 仍能按语言预测方向；即便 robot state 来自相反方向样本，DCS 也保持 0.80。这是论文里最能支撑机制解释的证据。

## 五、局限性与展望

论文没有单独展开 limitations，但从实验和方法可以看出几个边界。第一，LA-33K 来自 DROID，动作原子词表和机器人形态都偏向单臂夹爪操作，对双臂、灵巧手、移动操作是否同样有效还没有证明。第二，atomic segmentation 依赖 VLM proposal 和人工验证，数据构建成本并非为零。第三，LA pretraining 学到的是视觉无关动作先验，但真实操作仍需要视觉定位、目标选择和闭环修正；如果视觉 grounding 本身弱，LA 只能补强语言-动作关系，不能单独解决 perception failure。

后续值得看的是：LA 数据能否从更多机器人数据集自动构建；是否能和更大 VLA backbone、跨 embodiment action space 结合；以及 LA supervision 是否能扩展到 force/contact、双臂协同和长时序技能。

## 六、灵魂三问

1. **它解决了什么问题？**

   它解决的是标准 VLA 中语言监督被视觉-动作密集信号稀释的问题。模型看似听指令，实际可能主要依赖视觉 shortcut；LA4VLA 通过移除视觉预训练，强迫模型先学习语言如何约束动作。

2. **为什么这么做？**

   因为语言和局部动作的对应关系在完整 trajectory 中太稀疏。把示范拆成 atomic language-action episodes，可以让每个短动作都有明确低层语言描述，并避免模型在预训练阶段靠图像偷懒。

3. **什么证据最有说服力？**

   视觉冲突诊断和真实机器人结果最有说服力。标准 VLA 在 Visual-conflict 下 DAR 只有 0.35，说明图像能覆盖语言；而 LA/MixPT 在真实任务上从 38.3/48.3 提到 81.7/83.3，说明显式 LA pretraining 对语言指定目标的操作确实有帮助。

## 七、个人总结

1. LA4VLA 的核心想法是：从已有机器人示范中挖出短粒度 language-action pairs，先不看图像学动作先验，再回到 VLA。
2. 最大优势是问题定义锋利，诊断实验直接证明标准 VLA 的语言 grounding 脆弱；最大弱点是 LA 数据构建仍依赖分割质量和人工验证，规模化程度有待观察。
3. 对 VLA 研究的启发是，更多图像和更多轨迹不一定自动带来更强 instruction following；有时需要专门设计监督，让语言在动作解码前真正“说得上话”。

# Attention 诊断代码附录

[返回整理报告](../02_Attention诊断/01_Action-to-Vision_Attention.md)

原始记录中的 `Attention` 实现片段，保留诊断和干预路径，供对照取数位置使用；不是独立可运行脚本。指标定义见主文档第 3 节。

```python
@at.typecheck
class Attention(nn.Module):
    """Attention module."""

    configs: Sequence[Config]

    @nn.compact
    # ===== Attention 可视化需求改造：开始 =====
    # 目的：为既有 Attention 前向增加静态诊断选择与可选因果干预；
    # 普通路径不传 intervention，q/k/v 与 KV cache 计算保持不变。
    def __call__(
        self,
        xs,
        positions,
        attn_mask,
        kv_cache,
        diagnostics_spec: GemmaAttentionSpec | None = None,
        attention_intervention: AttentionIntervention | None = None,
        layer_index=None,
    ):
        """Run model attention and optionally summarize/intervene in its real probabilities."""
        # ===== Attention 可视化需求改造：结束 =====
        # all experts must share the same head dim, num heads, and num kv heads for self-attention to work
        assert all(config.head_dim == self.configs[0].head_dim for config in self.configs)
        assert all(config.num_heads == self.configs[0].num_heads for config in self.configs)
        assert all(config.num_kv_heads == self.configs[0].num_kv_heads for config in self.configs)

        dtype = next(x.dtype for x in xs if x is not None)  # original dtype, could be half-precision

        qkvs = []
        ## i就是0 或者1 ，0的时候计算prefix hidden 1 的时候计算action hidden，网络长度不一样
        for i, (x, config) in enumerate(zip(xs, self.configs, strict=True)):    #xs = [prefix_hidden [B,P,2048] , action_hidden [B,50,1024] ,  ]
            ## 各 expert 用自己的 config 和 Q/K/V 参数
            if x is None:
                continue
            
            if config.num_kv_heads == config.num_heads:
                qkv_einsum = lora.Einsum(
                    shape=(3, config.num_heads, config.width, config.head_dim),
                    name=_name("qkv_einsum", i),
                    init_fn=nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0, 1)),
                    lora_config=config.lora_configs.get("attn"),
                )
                qkvs.append(qkv_einsum("BSD,3KDH->3BSKH", x))
            else:           #pi0.5
                #lora.Einsum(...)          创建一个带可训练权重的线性投影层
                #q_einsum("公式", x)       把输入 x  /送进这个投影层，得到 Q
                #jnp.einsum("公式", q, k)  纯张量计算    ，没有可训练参数
                q_einsum = lora.Einsum( 
                    shape=(config.num_heads, config.width, config.head_dim),
                    name=_name("q_einsum", i),
                    init_fn=nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0,)),
                    lora_config=config.lora_configs.get("attn"),
                ) # p[B,8,1024/2048,256]
                q = q_einsum("BTD,NDH->BTNH", x)# [B,T,1024/2048],[8,1024/2048,256]->[B,T,8,256] Tquery token 数
                kv_einsum = lora.Einsum(
                    shape=(2, config.num_kv_heads, config.width, config.head_dim),
                    name=_name("kv_einsum", i),
                    init_fn=nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0, 1)),
                    lora_config=config.lora_configs.get("attn"),
                )
                k, v = kv_einsum("BSD,2KDH->2BSKH", x) # [B,1018,256] [2,1,1024/2048,256] ->[2,B,1018,1,1024/2048]
                qkvs.append((q, k, v))

        # 沿 token 维拼接两个 expert 的 Q/K/V
        q, k, v = (jnp.concatenate(y, axis=1) for y in zip(*qkvs, strict=True))
        #  加位置编码并缩放 Q 这个是标准RoPE
        q = _apply_rope(q, positions=positions)
        q *= self.configs[0].head_dim ** -0.5

        k = _apply_rope(k, positions=positions)

        # should still be half-precision here (if input was half-precision)
        assert q.dtype == k.dtype == v.dtype == dtype

        # 推理时拼接预先缓存的 prefix K/V
        if kv_cache is not None:
            cache_k, cache_v = kv_cache
            k = jnp.concatenate([cache_k, k], axis=1)
            v = jnp.concatenate([cache_v, v], axis=1)

        q = einops.rearrange(q, "B T (K G) H -> B T K G H", K=self.configs[0].num_kv_heads)
        # 计算qk 基于h的点积 logits: [B,1,8,1018,1018]
        logits = jnp.einsum("BTKGH,BSKH->BKGTS", q, k, preferred_element_type=jnp.float32) 

        if attn_mask.shape != (q.shape[0], 1, q.shape[1], k.shape[1]):
            raise ValueError(
                f"Attention mask with shape {attn_mask.shape} but shapes for q and k are: {q.shape} and {k.shape}"
            )
        # big_neg = jnp.finfo(logits.dtype).min
        big_neg = -2.3819763e38  # See gemma/modules.py
        masked_logits = jnp.where(attn_mask[:, :, None, :, :], logits, big_neg)

        probs = jax.nn.softmax(masked_logits, axis=-1).astype(dtype)

        # ===== Attention 可视化需求改造：开始 =====
        # 目的：在真实 action attention 路径上执行两种可区分的干预。
        # mask-renorm(mode=2) 在 softmax 语义上删除连边并重分配读取权重；
        # zero-AV(mode=1) 保留原始 probs，只从 encoded 中减去目标 key 的 A*V。
        # enabled 和 layer_mask 都是运行时 JAX 值，同一个 JIT 可复用于不同组/层/步。
        intervention_edge_mask = None
        original_probs = probs
        if attention_intervention is not None:
            if layer_index is None:
                raise ValueError("layer_index is required when attention_intervention is provided")
            layer_enabled = jnp.take(attention_intervention.layer_mask, layer_index)
            intervention_enabled = jnp.logical_and(attention_intervention.enabled, layer_enabled)
            intervention_edge_mask = (
                attention_intervention.query_mask[:, :, None]
                & attention_intervention.key_mask[:, None, :]
                & intervention_enabled
            )
            renorm_logits = jnp.where(
                intervention_edge_mask[:, None, None, :, :], big_neg, masked_logits
            )
            renorm_probs = jax.nn.softmax(renorm_logits, axis=-1).astype(dtype)
            use_renorm = jnp.equal(attention_intervention.mode, 2)
            probs = jnp.where(use_renorm, renorm_probs, probs)
        # ===== Attention 可视化需求改造：结束 =====
        # ===== Attention 可视化需求改造：开始 =====
        # 目的：直接旁接本次正常前向实际使用的 probs，而不是用另一个模型或第二次 forward 复算 attention。
        attention_summary = (
            reduce_attention(probs, diagnostics_spec, attn_mask, v) if diagnostics_spec is not None else None
        )
        # ===== Attention 可视化需求改造：结束 =====

        # attention 权重乘 V 
        encoded = jnp.einsum("BKGTS,BSKH->BTKGH", probs, v)
        # ===== Attention 可视化需求改造：开始 =====
        # 目的：zero-AV 仅删除目标 action-query <- key 边的 Value 消息，
        # 不重分配其他 key 的 attention；mode=0/2 时该减项严格为零。
        if attention_intervention is not None:
            removed_probs = jnp.where(
                intervention_edge_mask[:, None, None, :, :], original_probs, 0
            )
            removed_encoded = jnp.einsum("BKGTS,BSKH->BTKGH", removed_probs, v)
            encoded -= removed_encoded * jnp.equal(attention_intervention.mode, 1).astype(encoded.dtype)
        # ===== Attention 可视化需求改造：结束 =====
        
        # encoded: [B,P+50,8,256]
        encoded = einops.rearrange(encoded, "B T K G H -> B T (K G) H")
        

        # 按 query token 区间拆回两个 expert ｜ encoded 已经完成联合 A@V，query顺序仍为 [prefix, action]。
        # 下面按照 query所属 expert 的token区间切开，
        # 再使用各自的 W_O 投影回各自 residual width。
        #
        # 这里只做 W_O，不做 residual，不调用 timestep gate；
        # residual/gate 位于外层 Block。
        out = []
        start = 0
        for i, (x, config) in enumerate(zip(xs, self.configs, strict=True)):
            if x is not None:
                end = start + x.shape[1] 
                # 这里就是W 0的计算 
                # 创建的是当前 expert 的 attention output projection：
                out_einsum = lora.Einsum(
                    shape=(config.num_heads, config.head_dim, config.width),
                    name=_name("attn_vec_einsum", i),
                    init_fn=nn.initializers.lecun_normal(in_axis=(-3, -2), out_axis=-1),
                    lora_config=config.lora_configs.get("attn"),
                )
                expert_encoded = encoded[:, start:end]
                # W_o 一个是吧256变成248/1024 另一个是要合并8个head
                expert_projected = out_einsum("BTNH,NHD->BTD", expert_encoded)# [B,T,8,256],[8,256,2048/1024]->[B,T,2048/1024]
                out.append(expert_projected)

                # post_wo_image_indices is deliberately zero-length when this
                # optional metric is disabled. Its static shape avoids traced
                # boolean branches and keeps the normal diagnostics path cheap.
                if (
                    diagnostics_spec is not None
                    and attention_summary is not None
                    and diagnostics_spec.post_wo_image_indices.shape[0]
                ):
                    selected_encoded = jnp.take(expert_encoded, diagnostics_spec.query_indices, axis=1)
                    lora_config = config.lora_configs.get("attn")
                    if lora_config is None:
                        # Re-evaluate only the compact diagnostic projection in
                        # FP32. The action path above remains completely
                        # unchanged and continues to use its configured dtype.
                        output_weight = out_einsum.w.astype(jnp.float32)
                        diagnostic_projected = jnp.einsum(
                            "bqnh,nhd->bqd", selected_encoded.astype(jnp.float32), output_weight
                        )
                    else:
                        diagnostic_projected = jnp.take(
                            expert_projected, diagnostics_spec.query_indices, axis=1
                        ).astype(jnp.float32)
                    direction = diagnostic_projected
                    direction /= jnp.maximum(
                        jnp.linalg.norm(direction, axis=-1, keepdims=True),
                        jnp.finfo(jnp.float32).tiny,
                    )
                    if lora_config is None:
                        backprojected = jnp.einsum("bqd,nhd->bqnh", direction, output_weight)
                    else:
                        # The pullback includes LoRA for LoRA-backed variants.
                        backprojected = jax.linear_transpose(
                            lambda value: out_einsum("BTNH,NHD->BTD", value),
                            selected_encoded,
                        )(direction.astype(expert_projected.dtype))[0].astype(jnp.float32)
                    backprojected = jnp.take(backprojected, diagnostics_spec.head_indices, axis=2)

                    flat_probs = einops.rearrange(probs, "b k g t s -> b (k g) t s")
                    selected_probs = jnp.take(flat_probs, diagnostics_spec.head_indices, axis=1)
                    selected_probs = jnp.take(selected_probs, diagnostics_spec.query_indices, axis=2)
                    kv_indices = diagnostics_spec.head_indices // probs.shape[2]
                    selected_values = jnp.take(v, kv_indices, axis=2)
                    # ===== Attention 可视化需求改造：开始 =====
                    # 目的：用真实 attention path 的 A、V 和本 expert 的 W_O，
                    # 一次计算每个 key 沿实际 attention 输出方向的 signed contribution：
                    # s_j=<W_O concat_h(A_hqj V_hj), y_q/||y_q||>。这比纯 A 多包含
                    # V/W_O 的强度、方向和 head 合成，但仍不是对最终 action 的因果影响。
                    signed_all = jnp.einsum(
                        "bnqs,bsnh,bqnh->bqs",
                        selected_probs.astype(jnp.float32),
                        selected_values.astype(jnp.float32),
                        backprojected,
                    )
                    image_signed = jnp.take(signed_all, diagnostics_spec.image_indices, axis=-1)
                    language_signed = jnp.take(
                        signed_all, diagnostics_spec.language_indices, axis=-1
                    )
                    suffix_signed = jnp.take(signed_all, diagnostics_spec.suffix_indices, axis=-1)
                    attention_summary = attention_summary._replace(
                        image_post_wo_magnitude=jnp.abs(image_signed),
                        image_post_wo_signed=image_signed,
                        language_post_wo_magnitude=jnp.abs(language_signed),
                        language_post_wo_signed=language_signed,
                        suffix_post_wo_magnitude_mass=jnp.sum(jnp.abs(suffix_signed), axis=-1),
                        suffix_post_wo_signed_mass=jnp.sum(suffix_signed, axis=-1),
                    )
                    # ===== Attention 可视化需求改造：结束 =====
                start = end
            else:
                out.append(None)

        # ===== Attention 可视化需求改造：开始 =====
        # 目的：仅诊断模式附加小型 summary；普通推理继续保持原有 (out, kv_cache) 返回结构。
        if diagnostics_spec is not None:
            return out, (k, v), attention_summary
        # ===== Attention 可视化需求改造：结束 =====
        return out, (k, v)
```

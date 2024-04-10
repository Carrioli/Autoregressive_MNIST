from typing import Callable

import jax.numpy as jnp
from jax import jit, lax, nn, profiler, random, vmap
from jax.random import PRNGKey
from jaxtyping import Array


def create_attention_mask(size: int, n_unmasked: int) -> Array:
    assert n_unmasked <= size, "Unmasked elements must be less than or equal to the sequence length"
    mask = jnp.triu(jnp.full((size, size), fill_value=-jnp.inf), k=1)
    # return mask
    return mask.at[:, :n_unmasked].set(0)


def init_params(initializer: Callable,
                n_outer_blocks: int,
                n_transformers: int,
                n_blocks: int,
                n_heads: int,
                num_classes: int,
                d_model: int,
                seq_len: int,
                d_qk: int,
                d_v: int,
                shrink_factor: int,
                key: PRNGKey) -> Array:
    params_keys = random.split(key, 10)
    embeddings  = initializer(params_keys[0], (num_classes, d_model))
    wq          = initializer(params_keys[1], (n_outer_blocks, n_transformers, n_blocks, n_heads, d_model, d_qk))
    e           = initializer(params_keys[2], (n_outer_blocks, n_transformers, n_blocks, n_heads, seq_len // shrink_factor, 1, shrink_factor))
    wk          = initializer(params_keys[3], (n_outer_blocks, n_transformers, n_blocks, n_heads, d_model, d_qk))
    f           = initializer(params_keys[4], (n_outer_blocks, n_transformers, n_blocks, n_heads, seq_len // shrink_factor, shrink_factor, 1))
    wv          = initializer(params_keys[5], (n_outer_blocks, n_transformers, n_blocks, n_heads, d_model, d_v))
    projection  = initializer(params_keys[6], (n_outer_blocks, n_transformers, n_blocks, n_heads * d_v, d_model))
    out_proj    = initializer(params_keys[7], (n_outer_blocks, n_transformers * d_model, d_model))
    final_proj  = initializer(params_keys[8], (d_model, num_classes))
    return embeddings, wq, e, wk, f, wv, projection, out_proj, final_proj


def concat_heads(x: Array) -> Array:
    x = jnp.transpose(x, (1, 0, 2))
    return x.reshape(x.shape[0], -1)


# @profile
def single_head_self_attention(x: Array,
                               params: Array,
                               mask: Array) -> Array:
    wq, e, wk, f, wv = params
    
    # constants
    seq_len, d_model = x.shape
    shrink_factor = e.shape[-1]
    d_qk = wq.shape[-1]
    dv   = wv.shape[-1]

    # for inference
    e = e[:seq_len // shrink_factor]
    f = f[:seq_len // shrink_factor]

    # shrink
    x = x.reshape(-1, shrink_factor, d_model)
    x = jnp.matmul(e, x)
    x = jnp.squeeze(x, axis=1)
    
    Q = x @ wq
    K = x @ wk
    scores = Q @ K.T / jnp.sqrt(d_qk)
    scores += mask
    attention_weights = nn.softmax(scores, axis=-1)
    V = x @ wv
    out = attention_weights @ V
    
    # expand
    out = jnp.expand_dims(out, axis=1)
    out = jnp.matmul(f, out)
    out = out.reshape(-1, dv)

    return out


def transformer_block(x: Array,
                      params: Array,
                      mask: Array) -> Array:
    *params, projection = params
    attention = vmap(single_head_self_attention, in_axes=(None, 0, None))(x, params, mask)
    attention = concat_heads(attention)
    attention = jnp.dot(attention, projection)
    attention += x
    mean = jnp.mean(attention, axis=-1, keepdims=True)
    var = jnp.var(attention, axis=-1, keepdims=True)
    attention = (attention - mean) / jnp.sqrt(var + 1e-5)
    return attention


def transformer(x: Array,
                params: Array,
                n_blocks: int,
                mask: Array) -> Array:
    wq, e, wk, f, wv, projection = params
    return lax.fori_loop(0,
                         n_blocks,
                         lambda i, x: transformer_block(x,
                                                        (wq[i], e[i], wk[i], f[i], wv[i], projection[i]),
                                                        mask),
                         x)


def super_block(x: Array,
                params: Array,
                n_blocks: int,
                mask: Array) -> Array:
    *params, out_proj = params
    super_attention = vmap(transformer, in_axes=(None, 0, None, None))(x, params, n_blocks, mask)
    super_attention = concat_heads(super_attention)
    super_attention = jnp.dot(super_attention, out_proj)
    x += super_attention
    mean = jnp.mean(x, axis=1, keepdims=True)
    var = jnp.var(x, axis=1, keepdims=True)
    x = (x - mean) / jnp.sqrt(var + 1e-5)
    return x


def super_transformer(x: Array,
                      params: Array,
                      n_outer_blocks: int,
                      n_blocks: int,
                      mask: Array) -> Array:
    embeddings, wq, e, wk, f, wv, projection, out_proj, final_proj = params
    x = jnp.take(embeddings, x, axis=0)
    out = lax.fori_loop(0,
                        n_outer_blocks,
                        lambda i, x: super_block(x,
                                                (wq[i], e[i], wk[i], f[i], wv[i], projection[i], out_proj[i]),
                                                n_blocks,
                                                mask),
                        x)
    return jnp.dot(out, final_proj)


# adding batch dimension
def batched_forward(x: Array,
                    params: Array,
                    n_outer_blocks: int,
                    n_blocks: int,
                    mask: Array) -> Array:
    return vmap(super_transformer, in_axes=(0, None, None, None, None))(x, params, n_outer_blocks, n_blocks, mask)



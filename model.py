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
                n_level_2_blocks: int,
                n_level_1_transformers: int,
                n_level_1_blocks: int,
                n_level_0_transformers: int,
                n_level_0_blocks: int,
                n_heads: int,
                num_classes: int,
                d_model: int,
                seq_len: int,
                d_qk: int,
                d_v: int,
                shrink_factor: int,
                key: PRNGKey) -> Array:
    params_keys  = random.split(key, 20)
    embeddings   = initializer(params_keys[0], (num_classes, d_model))
    wq           = initializer(params_keys[1], (n_level_2_blocks, n_level_1_transformers, n_level_1_blocks, n_level_0_transformers, n_level_0_blocks, n_heads, d_model, d_qk))
    e            = initializer(params_keys[2], (n_level_2_blocks, n_level_1_transformers, n_level_1_blocks, n_level_0_transformers, n_level_0_blocks, n_heads, seq_len // shrink_factor, 1, shrink_factor))
    wk           = initializer(params_keys[3], (n_level_2_blocks, n_level_1_transformers, n_level_1_blocks, n_level_0_transformers, n_level_0_blocks, n_heads, d_model, d_qk))
    f            = initializer(params_keys[4], (n_level_2_blocks, n_level_1_transformers, n_level_1_blocks, n_level_0_transformers, n_level_0_blocks, n_heads, seq_len // shrink_factor, shrink_factor, 1))
    wv           = initializer(params_keys[5], (n_level_2_blocks, n_level_1_transformers, n_level_1_blocks, n_level_0_transformers, n_level_0_blocks, n_heads, d_model, d_v))
    level_0_proj = initializer(params_keys[6], (n_level_2_blocks, n_level_1_transformers, n_level_1_blocks, n_level_0_transformers, n_level_0_blocks, n_heads * d_v, d_model))
    level_1_proj = initializer(params_keys[7], (n_level_2_blocks, n_level_1_transformers, n_level_1_blocks, n_level_0_transformers * d_model, d_model))
    level_2_proj = initializer(params_keys[8], (n_level_2_blocks, n_level_1_transformers * d_model, d_model))
    final_proj   = initializer(params_keys[9], (d_model, num_classes))
    return embeddings, wq, e, wk, f, wv, level_0_proj, level_1_proj, level_2_proj, final_proj


def normalize(x: Array) -> Array:
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.var(x, axis=-1, keepdims=True)
    return (x - mean) / jnp.sqrt(var + 1e-5)


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


def level_0_block(x: Array,
                  params: Array,
                  mask: Array) -> Array:
    *params, level_0_proj = params
    attention = vmap(single_head_self_attention, in_axes=(None, 0, None))(x, params, mask)
    attention = concat_heads(attention)
    attention = jnp.dot(attention, level_0_proj)
    return normalize(attention + x)


def level_0_transformer(x: Array,
                        params: Array,
                        n_level_0_blocks: int,
                        mask: Array) -> Array:
    wq, e, wk, f, wv, level_0_proj = params
    return lax.fori_loop(0,
                         n_level_0_blocks,
                         lambda i, x: level_0_block(x,
                                                    (wq[i], e[i], wk[i], f[i], wv[i], level_0_proj[i]),
                                                    mask),
                         x)


def level_1_block(x: Array,
                  params: Array,
                  n_level_0_blocks: int,
                  mask: Array) -> Array:
    *params, level_1_proj = params
    attention = vmap(level_0_transformer, in_axes=(None, 0, None, None))(x, params, n_level_0_blocks, mask)
    attention = concat_heads(attention)
    attention = jnp.dot(attention, level_1_proj)
    return normalize(attention + x)


def level_1_transformer(x: Array,
                        params: Array,
                        n_level_1_blocks: int,
                        n_level_0_blocks: int,
                        mask: Array) -> Array:
    wq, e, wk, f, wv, level_0_proj, level_1_proj = params
    return lax.fori_loop(0,
                         n_level_1_blocks,
                         lambda i, x: level_1_block(x,
                                                    (wq[i], e[i], wk[i], f[i], wv[i], level_0_proj[i], level_1_proj[i]),
                                                    n_level_0_blocks,
                                                    mask),
                         x)


def level_2_block(x: Array,
                params: Array,
                n_level_1_blocks: int,
                n_level_0_blocks: int,
                mask: Array) -> Array:
    *params, level_2_proj = params
    attention = vmap(level_1_transformer, in_axes=(None, 0, None, None, None))(x, params, n_level_1_blocks, n_level_0_blocks, mask)
    attention = concat_heads(attention)
    attention = jnp.dot(attention, level_2_proj)
    return normalize(attention + x)


def level_2_transformer(x: Array,
                      params: Array,
                      n_level_2_blocks: int,
                      n_level_1_blocks: int,
                      n_level_0_blocks: int,
                      mask: Array) -> Array:
    embeddings, wq, e, wk, f, wv, level_0_proj, level_1_proj, level_2_proj, final_proj = params
    x = jnp.take(embeddings, x, axis=0)
    out = lax.fori_loop(0,
                        n_level_2_blocks,
                        lambda i, x: level_2_block(x,
                                                (wq[i], e[i], wk[i], f[i], wv[i], level_0_proj[i], level_1_proj[i], level_2_proj[i]),
                                                n_level_1_blocks,
                                                n_level_0_blocks,
                                                mask),
                        x)
    return jnp.dot(out, final_proj)


# adding batch dimension
def batched_forward(x: Array,
                    params: Array,
                    n_level_2_blocks: int,
                    n_level_1_blocks: int,
                    n_level_0_blocks: int,
                    mask: Array) -> Array:
    return vmap(level_2_transformer, in_axes=(0, None, None, None, None, None))(x, params, n_level_2_blocks, n_level_1_blocks, n_level_0_blocks, mask)



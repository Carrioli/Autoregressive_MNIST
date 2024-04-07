from typing import Callable

import jax.numpy as jnp
from jax import jit, lax, nn, random, vmap, profiler
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


# @profile
def single_head_self_attention(x: Array,
                               wq: Array,
                               e: Array,
                               wk: Array,
                               f: Array,
                               wv: Array,
                               mask: Array,
                               start_index_for_x_wq: int) -> Array:
    
    # shrink
    seq_len, d_model = x.shape
    shrink_factor = e.shape[-1]
    x = x.reshape(-1, shrink_factor, d_model)
    e = e[:x.shape[0]]
    x = jnp.matmul(e, x)
    x = jnp.squeeze(x, axis=1)
    
    # attention
    x_tilda = x[start_index_for_x_wq:]
    Q = x_tilda @ wq
    K = x @ wk
    V = x @ wv
    d_qk = Q.shape[-1]
    scores = jnp.dot(Q, K.T) / jnp.sqrt(d_qk)
    mask = mask[-scores.shape[0]:]
    scores += mask
    attention_weights = nn.softmax(scores, axis=-1)
    out = attention_weights @ V
    
    # expand
    out = jnp.expand_dims(out, axis=1)
    # f   = f[:out.shape[0]] # this is no longer correct
    f   = f[start_index_for_x_wq:start_index_for_x_wq + out.shape[0]]
    out = jnp.matmul(f, out)
    out = out.reshape(-1, wv.shape[-1])
    
    return out


def multi_head_self_attention(x: Array,
                              params: Array,
                              mask: Array,
                              start_index_for_x_wq: int) -> Array:
    wq, e, wk, f, wv = params
    return vmap(single_head_self_attention, in_axes=(None, 0, 0, 0, 0, 0, None, None))(x, wq, e, wk, f, wv, mask, start_index_for_x_wq)


def concat_heads(x: Array) -> Array:
    x = jnp.transpose(x, (1, 0, 2))
    return x.reshape(x.shape[0], -1)


def transformer_block(x: Array,
                      params: Array,
                      mask: Array,
                      start_index_for_x_wq: int) -> Array:
    *in_params, projection = params
    attention = multi_head_self_attention(x, in_params, mask, start_index_for_x_wq)
    attention = concat_heads(attention)
    attention = jnp.dot(attention, projection)
    x += attention
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.var(x, axis=-1, keepdims=True)
    x = (x - mean) / jnp.sqrt(var + 1e-5)
    return x


def transformer(x: Array,
                wq: Array,
                e: Array,
                wk: Array,
                f: Array,
                wv: Array,
                projection: Array,
                n_blocks: int,
                mask: Array,
                start_index_for_x_wq: int) -> Array:
    return lax.fori_loop(0,
                         n_blocks,
                         lambda i, x: transformer_block(x,
                                                        (wq[i], e[i], wk[i], f[i], wv[i], projection[i]),
                                                        mask,
                                                        start_index_for_x_wq),
                         x)


def super_block(x: Array,
                params: Array,
                n_blocks: int,
                mask: Array,
                start_index_for_x_wq: int) -> Array:
    wq, e, wk, f, wv, projection, out_proj = params
    b_out = vmap(transformer, in_axes=(None, 0, 0, 0, 0, 0, 0, None, None))(x, wq, e, wk, f, wv, projection, n_blocks, mask, start_index_for_x_wq)
    b_out = concat_heads(b_out)
    b_out = jnp.dot(b_out, out_proj)
    x += b_out
    mean = jnp.mean(x, axis=1, keepdims=True)
    var = jnp.var(x, axis=1, keepdims=True)
    x = (x - mean) / jnp.sqrt(var + 1e-5)
    return x


def super_transformer(x: Array,
                      params: Array, # NOTE: this will need to be split up into individual params to vectorize
                      n_outer_blocks: int,
                      n_blocks: int,
                      mask: Array,
                      start_index_for_x_wq: int) -> Array:
    embeddings, wq, e, wk, f, wv, projection, out_proj, final_proj = params
    x = jnp.take(embeddings, x, axis=0)
    out = lax.fori_loop(0,
                        n_outer_blocks,
                        lambda i, x: super_block(x,
                                                (wq[i], e[i], wk[i], f[i], wv[i], projection[i], out_proj[i]),
                                                n_blocks,
                                                mask,
                                                start_index_for_x_wq),
                        x)
    return jnp.dot(out, final_proj)


# adding batch dimension
def batched_forward(x: Array,
                    params: Array,
                    n_outer_blocks: int,
                    n_blocks: int,
                    mask: Array,
                    start_index_for_x_wq: int) -> Array:
    return vmap(super_transformer, in_axes=(0, None, None, None, None, None))(x, params, n_outer_blocks, n_blocks, mask, start_index_for_x_wq)



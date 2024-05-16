from typing import Callable

import jax.numpy as jnp
from jax import jit, lax, nn, profiler, random, vmap
from jax.random import PRNGKey
from jaxtyping import Array


@jit
def cubic_spline(x, knots, coeffs):
    # Custom cubic spline interpolation
    idx = jnp.searchsorted(knots, x) - 1
    idx = jnp.clip(idx, 0, len(knots) - 2)
    
    t = (x - knots[idx]) / (knots[idx + 1] - knots[idx])
    a = coeffs[idx]
    b = (coeffs[idx + 1] - coeffs[idx])
    return (1 - t) * a + t * b


def create_attention_mask(size: int, n_unmasked: int) -> Array:
    assert n_unmasked <= size, "Unmasked elements must be less than or equal to the sequence length"
    mask = jnp.triu(jnp.full((size, size), fill_value=-jnp.inf), k=1)
    # return mask
    return mask.at[:, :n_unmasked].set(0)


def init_params(initializer: Callable,
                l_2_blocks: int,
                l_1_tfms: int,
                l_1_blocks: int,
                l_0_tfms: int,
                l_0_blocks: int,
                n_heads: int,
                num_classes: int,
                d_model: int,
                seq_len: int,
                d_qk: int,
                d_v: int,
                shrink_factor: int,
                l_0_degree: int,
                l_0_num_knots: int,
                key: PRNGKey) -> Array:
    keys  = random.split(key, 20)
    embeddings   = initializer(keys[0], (num_classes, d_model))
    wq           = initializer(keys[1], (l_2_blocks, l_1_tfms, l_1_blocks, l_0_tfms, l_0_blocks, n_heads, d_model, d_qk))
    e            = initializer(keys[2], (l_2_blocks, l_1_tfms, l_1_blocks, l_0_tfms, l_0_blocks, n_heads, seq_len // shrink_factor, 1, shrink_factor))
    wk           = initializer(keys[3], (l_2_blocks, l_1_tfms, l_1_blocks, l_0_tfms, l_0_blocks, n_heads, d_model, d_qk))
    f            = initializer(keys[4], (l_2_blocks, l_1_tfms, l_1_blocks, l_0_tfms, l_0_blocks, n_heads, seq_len // shrink_factor, shrink_factor, 1))
    wv           = initializer(keys[5], (l_2_blocks, l_1_tfms, l_1_blocks, l_0_tfms, l_0_blocks, n_heads, d_model, d_v))
    l_0_proj     = initializer(keys[6], (l_2_blocks, l_1_tfms, l_1_blocks, l_0_tfms, l_0_blocks, n_heads * d_v, d_model))
    l_0_knots    = jnp.sort(initializer(keys[7], (l_2_blocks, l_1_tfms, l_1_blocks, l_0_tfms, l_0_blocks, l_0_num_knots)), axis=-1)
    l_0_coeff    = initializer(keys[8], (l_2_blocks, l_1_tfms, l_1_blocks, l_0_tfms, l_0_blocks, l_0_num_knots + l_0_degree - 1))
    l_1_proj     = initializer(keys[9], (l_2_blocks, l_1_tfms, l_1_blocks, l_0_tfms * d_model, d_model))
    l_2_proj     = initializer(keys[10], (l_2_blocks, l_1_tfms * d_model, d_model))
    final_proj   = initializer(keys[11], (d_model, num_classes))
    return embeddings, wq, e, wk, f, wv, l_0_proj, l_0_knots, l_0_coeff, l_1_proj, l_2_proj, final_proj


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
    *params, l_0_proj, l_0_knots, l_0_coeff = params
    attention = vmap(single_head_self_attention, in_axes=(None, 0, None))(x, params, mask)
    attention = concat_heads(attention)
    attention = jnp.dot(attention, l_0_proj)
    attention = normalize(attention)
    attention = cubic_spline(attention, l_0_knots, l_0_coeff)
    # attention = nn.relu(attention)
    return attention + x


def level_0_transformer(x: Array,
                        params: Array,
                        l_0_blocks: int,
                        mask: Array) -> Array:
    wq, e, wk, f, wv, l_0_proj, l_0_knots, l_0_coeff = params
    return lax.fori_loop(0,
                         l_0_blocks,
                         lambda i, x: level_0_block(x,
                                                    (wq[i], e[i], wk[i], f[i], wv[i], l_0_proj[i], l_0_knots[i], l_0_coeff[i]),
                                                    mask),
                         x)


def level_1_block(x: Array,
                  params: Array,
                  l_0_blocks: int,
                  mask: Array) -> Array:
    *params, l_1_proj = params
    attention = vmap(level_0_transformer, in_axes=(None, 0, None, None))(x, params, l_0_blocks, mask)
    attention = concat_heads(attention)
    attention = jnp.dot(attention, l_1_proj)
    attention = normalize(attention)
    attention = nn.relu(attention)
    return attention + x


def level_1_transformer(x: Array,
                        params: Array,
                        l_1_blocks: int,
                        l_0_blocks: int,
                        mask: Array) -> Array:
    wq, e, wk, f, wv, l_0_proj, l_0_knots, l_0_coeff, l_1_proj = params
    return lax.fori_loop(0,
                         l_1_blocks,
                         lambda i, x: level_1_block(x,
                                                    (wq[i], e[i], wk[i], f[i], wv[i], l_0_proj[i], l_0_knots[i], l_0_coeff[i], l_1_proj[i]),
                                                    l_0_blocks,
                                                    mask),
                         x)


def level_2_block(x: Array,
                params: Array,
                l_1_blocks: int,
                l_0_blocks: int,
                mask: Array) -> Array:
    *params, l_2_proj = params
    attention = vmap(level_1_transformer, in_axes=(None, 0, None, None, None))(x, params, l_1_blocks, l_0_blocks, mask)
    attention = concat_heads(attention)
    attention = jnp.dot(attention, l_2_proj)
    attention = normalize(attention)
    attention = nn.relu(attention)
    return attention + x


def level_2_transformer(x: Array,
                      params: Array,
                      l_2_blocks: int,
                      l_1_blocks: int,
                      l_0_blocks: int,
                      mask: Array) -> Array:
    embeddings, wq, e, wk, f, wv, l_0_proj, l_0_knots, l_0_coeff, l_1_proj, l_2_proj, final_proj = params
    x = jnp.take(embeddings, x, axis=0)
    out = lax.fori_loop(0,
                        l_2_blocks,
                        lambda i, x: level_2_block(x,
                                                (wq[i], e[i], wk[i], f[i], wv[i], l_0_proj[i], l_0_knots[i], l_0_coeff[i], l_1_proj[i], l_2_proj[i]),
                                                l_1_blocks,
                                                l_0_blocks,
                                                mask),
                        x)
    return jnp.dot(out, final_proj) # normalization and activation here?


# adding batch dimension
def batched_forward(x: Array,
                    params: Array,
                    l_2_blocks: int,
                    l_1_blocks: int,
                    l_0_blocks: int,
                    mask: Array) -> Array:
    return vmap(level_2_transformer, in_axes=(0, None, None, None, None, None))(x, params, l_2_blocks, l_1_blocks, l_0_blocks, mask)



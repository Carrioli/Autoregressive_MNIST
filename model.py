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
                l3_blocks: int,
                l2_tfms: int,
                l2_blocks: int,
                l1_tfms: int,
                l1_blocks: int,
                l0_tfms: int,
                l0_blocks: int,
                n_heads: int,
                n_classes: int,
                d_model: int,
                seq_len: int,
                d_qk: int,
                d_v: int,
                shrink_factor: int,
                key: PRNGKey) -> Array:
    keys  = random.split(key, 20)
    embeddings = initializer(keys[0], (n_classes, d_model))
    wq         = initializer(keys[1], (l3_blocks, l2_tfms, l2_blocks, l1_tfms, l1_blocks, l0_tfms, l0_blocks, n_heads, d_model, d_qk))
    e          = initializer(keys[2], (l3_blocks, l2_tfms, l2_blocks, l1_tfms, l1_blocks, l0_tfms, l0_blocks, n_heads, seq_len // shrink_factor, 1, shrink_factor))
    wk         = initializer(keys[3], (l3_blocks, l2_tfms, l2_blocks, l1_tfms, l1_blocks, l0_tfms, l0_blocks, n_heads, d_model, d_qk))
    f          = initializer(keys[4], (l3_blocks, l2_tfms, l2_blocks, l1_tfms, l1_blocks, l0_tfms, l0_blocks, n_heads, seq_len // shrink_factor, shrink_factor, 1))
    wv         = initializer(keys[5], (l3_blocks, l2_tfms, l2_blocks, l1_tfms, l1_blocks, l0_tfms, l0_blocks, n_heads, d_model, d_v))
    l0_proj    = initializer(keys[6], (l3_blocks, l2_tfms, l2_blocks, l1_tfms, l1_blocks, l0_tfms, l0_blocks, n_heads * d_v, d_model))
    l1_proj    = initializer(keys[7], (l3_blocks, l2_tfms, l2_blocks, l1_tfms, l1_blocks, l0_tfms * d_model, d_model))
    l2_proj    = initializer(keys[8], (l3_blocks, l2_tfms, l2_blocks, l1_tfms * d_model, d_model))
    l3_proj    = initializer(keys[9], (l3_blocks, l2_tfms * d_model, d_model))
    final_proj = initializer(keys[10], (d_model, n_classes))
    return embeddings, wq, e, wk, f, wv, l0_proj, l1_proj, l2_proj, l3_proj, final_proj


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
    *params, l0_proj = params
    attention = vmap(single_head_self_attention, in_axes=(None, 0, None))(x, params, mask)
    attention = concat_heads(attention)
    attention = jnp.dot(attention, l0_proj)
    attention = nn.standardize(attention)
    attention = nn.relu(attention)
    return attention + x


def level_0_transformer(x: Array,
                        params: Array,
                        l0_blocks: int,
                        mask: Array) -> Array:
    wq, e, wk, f, wv, l0_proj = params
    return lax.fori_loop(0,
                         l0_blocks,
                         lambda i, x: level_0_block(x,
                                                    (wq[i], e[i], wk[i], f[i], wv[i], l0_proj[i]),
                                                    mask),
                         x)


def level_1_block(x: Array,
                  params: Array,
                  l0_blocks: int,
                  mask: Array) -> Array:
    *params, l1_proj = params
    attention = vmap(level_0_transformer, in_axes=(None, 0, None, None))(x, params, l0_blocks, mask)
    attention = concat_heads(attention)
    attention = jnp.dot(attention, l1_proj)
    attention = nn.standardize(attention)
    attention = nn.relu(attention)
    return attention + x


def level_1_transformer(x: Array,
                        params: Array,
                        l1_blocks: int,
                        l0_blocks: int,
                        mask: Array) -> Array:
    wq, e, wk, f, wv, l0_proj, l1_proj = params
    return lax.fori_loop(0,
                         l1_blocks,
                         lambda i, x: level_1_block(x,
                                                    (wq[i], e[i], wk[i], f[i], wv[i], l0_proj[i], l1_proj[i]),
                                                    l0_blocks,
                                                    mask),
                         x)


def level_2_block(x: Array,
                params: Array,
                l1_blocks: int,
                l0_blocks: int,
                mask: Array) -> Array:
    *params, l2_proj = params
    attention = vmap(level_1_transformer, in_axes=(None, 0, None, None, None))(x, params, l1_blocks, l0_blocks, mask)
    attention = concat_heads(attention)
    attention = jnp.dot(attention, l2_proj)
    attention = nn.standardize(attention)
    attention = nn.relu(attention)
    return attention + x


def level_2_transformer(x: Array,
                      params: Array,
                      l2_blocks: int,
                      l1_blocks: int,
                      l0_blocks: int,
                      mask: Array) -> Array:
    wq, e, wk, f, wv, l0_proj, l1_proj, l2_proj = params
    return lax.fori_loop(0,
                         l2_blocks,
                         lambda i, x: level_2_block(x,
                                                    (wq[i], e[i], wk[i], f[i], wv[i], l0_proj[i], l1_proj[i], l2_proj[i]),
                                                    l1_blocks,
                                                    l0_blocks,
                                                    mask),
                         x)



def level_3_block(x: Array,
                params: Array,
                l2_blocks: int,
                l1_blocks: int,
                l0_blocks: int,
                mask: Array) -> Array:
    *params, l3_proj = params
    attention = vmap(level_2_transformer, in_axes=(None, 0, None, None, None, None))(x, params, l2_blocks, l1_blocks, l0_blocks, mask)
    attention = concat_heads(attention)
    attention = jnp.dot(attention, l3_proj)
    attention = nn.standardize(attention)
    attention = nn.relu(attention)
    return attention + x


def level_3_transformer(x: Array,
                      params: Array,
                      l3_blocks: int,
                      l2_blocks: int,
                      l1_blocks: int,
                      l0_blocks: int,
                      mask: Array) -> Array:
    embeddings, wq, e, wk, f, wv, l0_proj, l1_proj, l2_proj, l3_proj, final_proj = params
    x = jnp.take(embeddings, x, axis=0)
    out = lax.fori_loop(0,
                        l3_blocks,
                        lambda i, x: level_3_block(x,
                                                (wq[i], e[i], wk[i], f[i], wv[i], l0_proj[i], l1_proj[i], l2_proj[i], l3_proj[i]),
                                                l2_blocks,
                                                l1_blocks,
                                                l0_blocks,
                                                mask),
                        x)
    return jnp.dot(out, final_proj) # normalization and activation here?




# adding batch dimension
def batched_forward(x: Array,
                    params: Array,
                    l3_blocks: int,
                    l2_blocks: int,
                    l1_blocks: int,
                    l0_blocks: int,
                    mask: Array) -> Array:
    return vmap(level_3_transformer, in_axes=(0, None, None, None, None, None, None))(x, params, l3_blocks, l2_blocks, l1_blocks, l0_blocks, mask)



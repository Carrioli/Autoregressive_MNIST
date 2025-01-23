import os
import pickle
from functools import partial

import jax.numpy as jnp
import numpy as np
import optax
import jax
from jax.tree_util import tree_flatten
from tqdm import tqdm
from PIL import Image

from data import create_mnist_dataset
from model import batched_forward, create_attention_mask, init_params

# shrink_factors = [1,2,4,7,8,14,16,28,49,56,98,112,196,392]

def colorize_batch(batch, index):
    # Reshape the batch for vectorized operations
    images_vector = batch.reshape(batch.shape[0], 784)
    
    # Create a new RGB batch where all pixels are initially set to black
    rgb_batch = np.zeros((batch.shape[0], 784, 3))
    
    # Copy the grayscale values to all channels before the index for all images
    rgb_batch[:, :index, 0] = images_vector[:, :index]
    rgb_batch[:, :index, 1] = images_vector[:, :index]
    rgb_batch[:, :index, 2] = images_vector[:, :index]
    
    # Colorize the pixels after the specified index in red for all images
    rgb_batch[:, index:, 0] = images_vector[:, index:]
    
    # Reshape the rgb batch back to the original shape but with 3 channels for all images
    rgb_batch_reshaped = rgb_batch.reshape(batch.shape[0], 28, 28, 3)
    return rgb_batch_reshaped


def save_batch(batch, predicted_batch, epoch_index, save_dir='saved_images'):
    os.makedirs(save_dir, exist_ok=True)
    colorized_batch = colorize_batch(predicted_batch, original_n_unmasked)
    batch = jnp.stack([batch, batch, batch], axis=-1)
    combined_batch = jnp.concatenate([batch, colorized_batch], axis=-2)

    for i, img in enumerate(combined_batch):
        im = Image.fromarray(np.array(img).astype("uint8"))
        im.save(f"{save_dir}/epoch_{epoch_index}_item_{i}.png")


def inverse_transform(matrix, original_shape, patch_shape):
    patches_per_dim = original_shape[0] // patch_shape[0], original_shape[1] // patch_shape[1]
    reshaped_matrix = matrix.reshape(patches_per_dim[0], patches_per_dim[1], patch_shape[0], patch_shape[1])
    transposed_matrix = reshaped_matrix.transpose(0, 2, 1, 3)
    original_matrix = transposed_matrix.reshape(original_shape)
    return original_matrix


def count_params(params):
    params_flat, _ = tree_flatten(params)
    num_params = sum([p.size for p in params_flat])
    print(f'Number of parameters: {num_params:_}')


def loss_fn(params, x, y):
    pred_y = batched_forward(x, params, l3_blocks, l2_blocks, l1_blocks, l0_blocks, mask)
    return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(pred_y, y))


# @jit
# TODO rewrite this so that I can jit compile it and not throw nan
def batch_inference(batch, params):
    prediction = batch[:, :original_n_unmasked]
    logits = jnp.empty((batch_size, 0, n_classes)) 

    while prediction.shape[-1] != (seq_len + shrink_factor):
        out = batched_forward(prediction, params, l3_blocks, l2_blocks, l1_blocks, l0_blocks, mask=0)
        out = out[:, -shrink_factor:, :]
        logits = jnp.concatenate([logits, out], axis=1)
        out = jnp.argmax(out, axis=-1)
        prediction = jnp.concatenate([prediction, out], axis=-1)

    labels = batch[:, original_n_unmasked:]
    average_softmax_cross_entropy = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, labels))
    return prediction, average_softmax_cross_entropy


def inference_and_save(test_loader, params, epoch):
    batch = jnp.array(next(iter(test_loader))[0])
    predicted_batch, average_softmax_cross_entropy = batch_inference(batch, params)
    print('Average inference loss:', average_softmax_cross_entropy)
    print('Average inference L2 loss:', jnp.mean((batch - predicted_batch) ** 2))
    print('Perplexity:', jnp.exp(average_softmax_cross_entropy))
    predicted_batch = jax.vmap(inverse_transform, in_axes=(0, None, None))(predicted_batch, (28, 28), patch_shape)
    batch           = jax.vmap(inverse_transform, in_axes=(0, None, None))(batch, (28, 28), patch_shape)
    save_batch(batch, predicted_batch, epoch)


@partial(jax.pmap, axis_name='batch')
def train_step(params, opt_state, batch):
    x, y = batch[:, :-shrink_factor], batch[:, shrink_factor:]
    loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
    grads = jax.lax.pmean(grads, axis_name='batch')
    updates, opt_state = optimizer.update(grads, opt_state, params = params)
    params = optax.apply_updates(params, updates)
    loss = jax.lax.pmean(loss, axis_name='batch')
    return loss, params, opt_state


@partial(jax.pmap, axis_name='batch')
def test_step(params, batch):
    x, y = batch[:, :-shrink_factor], batch[:, shrink_factor:]
    loss, _ = jax.value_and_grad(loss_fn)(params, x, y)
    return loss


def train(train_loader, params, opt_state):   
    params = jax.device_put_replicated(params, devices)
    opt_state = jax.device_put_replicated(opt_state, devices)
    
    train_loss = 0.0
    num_iter = len(train_loader)
    for batch in tqdm(train_loader, total=num_iter):
        input_batch = jnp.array(batch[0]).reshape(n_devices, batch_size, -1)
        loss, params, opt_state = train_step(params, opt_state, input_batch)
        train_loss += loss[0]
        
    params = jax.tree.map(lambda x: x[0], params)
    opt_state = jax.tree.map(lambda x: x[0], opt_state)

    avg_train_loss = train_loss / num_iter
    print(f'Average train loss: {avg_train_loss}')
    print(f'Perplexity: {jnp.exp(avg_train_loss)}')
    return params, opt_state


def test(test_loader, params):    
    params = jax.device_put_replicated(params, devices)

    test_loss = 0.0
    for batch in test_loader:
        input_batch = jnp.array(batch[0]).reshape(n_devices, batch_size, -1)
        test_loss += test_step(params, input_batch)[0]

    avg_test_loss = test_loss / len(test_loader)
    print(f'Average test loss: {avg_test_loss}')
    print(f'Perplexity: {jnp.exp(avg_test_loss)}')


def save_params(params, path):
    with open(path, 'wb') as f:
        pickle.dump(params, f)


def train_and_test(train_loader, test_loader, params, opt_state):
    for epoch in range(1, 30):
        print('Epoch: ' + str(epoch))
        
        params, opt_state = train(train_loader, params, opt_state)
        
        test(test_loader, params)
        
        # save_params(params, f'params.pkl')
        
        # if (epoch) % 10 == 0:
        inference_and_save(test_loader, params, epoch)


if __name__ == '__main__':
    l3_blocks = 1
    l2_tfms   = 2
    l2_blocks = 1
    l1_tfms   = 3
    l1_blocks = 1
    l0_tfms   = 8
    l0_blocks = 1
    n_heads   = 16
    n_classes = 256 # same as d_out
    d_model   = 96 # same as feature size
    patch_shape = (4, 4)
    shrink_factor = patch_shape[0] * patch_shape[1]
    seq_len = 784 - shrink_factor
    d_qk = 16
    d_v  = 16

    n_devices = jax.device_count()
    devices = jax.devices()
    
    print('Number of devices:', n_devices)
    print('Available devices:', devices)
    
    original_n_unmasked = 320

    assert seq_len % shrink_factor == 0, 'Sequence length must be divisible by the shrink factor'
    assert original_n_unmasked % shrink_factor == 0, 'Unmasked elements (should) be divisible by the shrink factor'

    params_key = jax.random.PRNGKey(42)
    initializer = jax.nn.initializers.lecun_normal()

    # params
    try:
        with open('params.pkl', 'rb') as f:
            print('Loading params from params.pkl')
            params = pickle.load(f)
    except FileNotFoundError:
        print('No params.pkl file found, initializing new parameters')
        params = init_params(initializer, 
                            l3_blocks,
                            l2_tfms,
                            l2_blocks,
                            l1_tfms, 
                            l1_blocks, 
                            l0_tfms,
                            l0_blocks,
                            n_heads, 
                            n_classes, 
                            d_model, 
                            seq_len, 
                            d_qk, 
                            d_v, 
                            shrink_factor, 
                            params_key)

    optimizer = optax.lion(2e-4)
    opt_state = optimizer.init(params)

    mask = create_attention_mask(seq_len // shrink_factor, original_n_unmasked // shrink_factor)

    batch_size = 128
    train_loader, test_loader = create_mnist_dataset(bsz=n_devices*batch_size, patch_shape=patch_shape)

    count_params(params)

    train_and_test(train_loader, test_loader, params, opt_state)


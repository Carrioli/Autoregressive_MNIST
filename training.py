import os
import pickle

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from jax import config, devices, jit, nn, random, value_and_grad, vmap
from jax.tree_util import tree_flatten
from tqdm import tqdm

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
    
    combined_batch = jnp.concatenate([batch, colorized_batch], axis=-2) / 255

    for i, img in enumerate(combined_batch):
        plt.imsave(f'{save_dir}/epoch_{epoch_index}_item_{i}.png', img)


def inverse_transform(matrix, original_shape, patch_shape):
    patches_per_dim = original_shape[0] // patch_shape[0], original_shape[1] // patch_shape[1]
    reshaped_matrix = matrix.reshape(patches_per_dim[0], patches_per_dim[1], patch_shape[0], patch_shape[1])
    transposed_matrix = reshaped_matrix.transpose(0, 2, 1, 3)
    original_matrix = transposed_matrix.reshape(original_shape)
    return original_matrix


def count_params(params):
    params_flat, _ = tree_flatten(params)
    num_params = sum([p.size for p in params_flat])
    print(f"Number of parameters: {num_params:_}")


def loss_fn(params, x, y):
    pred_y = batched_forward(x, params, n_outer_blocks, n_blocks, mask)
    return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(pred_y, y))


@jit
def train_step(params, opt_state, x, y):
    loss, grads = value_and_grad(loss_fn)(params, x, y)
    updates, opt_state = optimizer.update(grads, opt_state, params = params)
    params = optax.apply_updates(params, updates)    
    return loss, params, opt_state


def nll(probabilities, labels):
    return jnp.mean(-jnp.log(probabilities[jnp.arange(labels.size), labels] + 1e-9))


@jit
def test_a_batch(batch, params):
    prediction = batch[:, :original_n_unmasked]
    
    out_probabilities = jnp.empty((128, 0, 256)) 

    while prediction.shape[-1] != (seq_len + shrink_factor):
        out = batched_forward(prediction, params, n_outer_blocks, n_blocks, mask = 0)
        out = out[:, -shrink_factor:, :]
        out_probabilities = jnp.concatenate([out_probabilities, nn.softmax(out, axis=-1)], axis=1)
        out = jnp.argmax(out, axis=-1)
        prediction = jnp.concatenate([prediction, out], axis=-1)

    labels = batch[:, original_n_unmasked:]
    out_probabilities = out_probabilities.reshape(-1, out_probabilities.shape[-1])
    average_batch_nll = nll(out_probabilities, labels.flatten())
    return prediction, average_batch_nll


def test_and_save(test_loader, params, epoch):
    batch = jnp.array(next(iter(test_loader))[0])
    predicted_batch, average_batch_nll = test_a_batch(batch, params)
    predicted_batch = vmap(inverse_transform, in_axes=(0, None, None))(predicted_batch, (28, 28), patch_shape)
    batch           = vmap(inverse_transform, in_axes=(0, None, None))(batch, (28, 28), patch_shape)
    print("Average test NLL over batch:", average_batch_nll)
    print("Average test L2 loss:", jnp.mean((batch - predicted_batch) ** 2))
    save_batch(batch, predicted_batch, epoch_index=epoch)


def train_and_test(train_loader, test_loader, params, opt_state):
    for epoch in range(66):
        total_loss = 0 
        print('Epoch: ' + str(epoch + 1))
        for batch in tqdm(train_loader):
            batch = jnp.array(batch[0])
            x, y = batch[:, :-shrink_factor], batch[:, shrink_factor:]
            loss, params, opt_state = train_step(params, opt_state, x, y)
            total_loss += loss
        print(f"Average train epoch loss: {total_loss / len(train_loader)}")
        
        # save pickle
        # with open(f"params.pkl", "wb") as f:
        #     pickle.dump(params, f)
        
        if (epoch) % 1 == 0:
            test_and_save(test_loader, params, epoch)


n_outer_blocks = 2
n_transformers = 2
n_blocks       = 4
n_heads        = 2
num_classes    = 256 # same as d_out
d_model        = 64 # same as feature size
d_qk           = 8
d_v            = 8
patch_shape    = (4, 4)
shrink_factor  = patch_shape[0] * patch_shape[1]
seq_len        = 784 - shrink_factor
original_n_unmasked = 448

assert seq_len % shrink_factor == 0, "Sequence length must be divisible by the shrink factor"
assert original_n_unmasked % shrink_factor == 0, "Unmasked elements (should) be divisible by the shrink factor"

params_key = random.PRNGKey(48)
initializer = nn.initializers.lecun_normal()

# params
try:
    with open("params.pkl", "rb") as f:
        print("Loading params from params.pkl")
        params = pickle.load(f)
except FileNotFoundError:
    print("No params.pkl file found, initializing new parameters")
    params = init_params(initializer, n_outer_blocks, n_transformers, n_blocks, n_heads, num_classes, d_model, seq_len, d_qk, d_v, shrink_factor, params_key)

optimizer = optax.lion(2e-4)
opt_state = optimizer.init(params)

mask = create_attention_mask(seq_len // shrink_factor, original_n_unmasked // shrink_factor)

batch_size = 128
train_loader, test_loader = create_mnist_dataset(bsz=batch_size, patch_shape=patch_shape)

count_params(params)
print("Available devices:", devices())

train_and_test(train_loader, test_loader, params, opt_state)


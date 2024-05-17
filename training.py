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
    pred_y = batched_forward(x, params, n_level_2_blocks, n_level_1_blocks, n_level_0_blocks, mask)
    return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(pred_y, y))


@jit
def train_step(params, opt_state, batch):
    x, y = batch[:, :-shrink_factor], batch[:, shrink_factor:]
    loss, grads = value_and_grad(loss_fn)(params, x, y)
    updates, opt_state = optimizer.update(grads, opt_state, params = params)
    params = optax.apply_updates(params, updates)    
    return loss, params, opt_state


@jit
def test_step(params, batch):
    x, y = batch[:, :-shrink_factor], batch[:, shrink_factor:]
    loss, _ = value_and_grad(loss_fn)(params, x, y)
    return loss


@jit
def batch_inference(batch, params):
    prediction = batch[:, :original_n_unmasked]
    logits = jnp.empty((batch_size, 0, num_classes)) 

    while prediction.shape[-1] != (seq_len + shrink_factor):
        out = batched_forward(prediction, params, n_level_2_blocks, n_level_1_blocks, n_level_0_blocks, mask=0)
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
    print("Average inference loss:", average_softmax_cross_entropy)
    print("Average inference L2 loss:", jnp.mean((batch - predicted_batch) ** 2))
    predicted_batch = vmap(inverse_transform, in_axes=(0, None, None))(predicted_batch, (28, 28), patch_shape)
    batch           = vmap(inverse_transform, in_axes=(0, None, None))(batch, (28, 28), patch_shape)
    save_batch(batch, predicted_batch, epoch)


def train(train_loader, params, opt_state):
    train_loss = 0 
    for batch in tqdm(train_loader):
        loss, params, opt_state = train_step(params, opt_state, jnp.array(batch[0]))
        train_loss += loss
    print(f"Average train loss: {train_loss / len(train_loader)}")
    return params, opt_state


def test(test_loader, params):
    test_loss = 0
    for batch in test_loader:
        test_loss += test_step(params, jnp.array(batch[0]))
    print(f"Average test loss: {test_loss / len(test_loader)}")


def save_params(params, path):
    with open(path, "wb") as f:
        pickle.dump(params, f)

def main(train_loader, test_loader, params, opt_state):
    for epoch in range(1, 100):
        print('Epoch: ' + str(epoch))
        params, opt_state = train(train_loader, params, opt_state)
        
        test(test_loader, params)
        
        # save_params(params, f"params.pkl")
        
        if (epoch) % 1 == 0:
            inference_and_save(test_loader, params, epoch)


n_level_2_blocks = 1
n_level_1_transformers = 4
n_level_1_blocks = 1
n_level_0_transformers = 8
n_level_0_blocks = 1
n_heads = 32
num_classes = 256  # same as d_out
d_model = 64  # same as feature size
d_qk = 8
d_v = 8
patch_shape = (4, 4)
shrink_factor = patch_shape[0] * patch_shape[1]
seq_len = 784 - shrink_factor
original_n_unmasked = 320

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
    params = init_params(initializer, 
                         n_level_2_blocks,
                         n_level_1_transformers,
                         n_level_1_blocks, 
                         n_level_0_transformers, 
                         n_level_0_blocks, 
                         n_heads, 
                         num_classes, 
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
train_loader, test_loader = create_mnist_dataset(bsz=batch_size, patch_shape=patch_shape)

count_params(params)
print("Available devices:", devices())

main(train_loader, test_loader, params, opt_state)


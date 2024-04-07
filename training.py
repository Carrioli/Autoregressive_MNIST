import os
import pickle

import jax.numpy as jnp
import numpy as np
import optax
from jax import devices, jit, nn, random, value_and_grad, config
from jax.tree_util import tree_flatten
from PIL import Image
from tqdm import tqdm

from data import create_mnist_dataset
from model import batched_forward, create_attention_mask, init_params

# shrink_factors = [1,2,4,7,8,14,16,28,49,56,98,112,196,392]

def save_batch_images(batch, batch_index, epoch_index, save_dir='saved_images'):
    os.makedirs(save_dir, exist_ok=True)
    
    # Convert the JAX array to a NumPy array if it's not already
    batch_np = np.array(batch)  # Assuming batch is a jnp.array
    batch_np = batch_np.reshape((-1, 28, 28))

    for i, img in enumerate(batch_np):
        # Convert the 2D numpy array to a PIL image (assuming grayscale)
        img_pil = Image.fromarray(img)
        if img_pil.mode != 'L':
            img_pil = img_pil.convert('L')  # Convert to grayscale if not already
        
        img_pil.save(os.path.join(save_dir, f'epoch_{epoch_index}_batch_{batch_index}_image_{i}.png'))


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
    updates, opt_state = optimizer.update(grads, opt_state, params=params)
    params = optax.apply_updates(params, updates)    
    return loss, params, opt_state


def predict(x, params, mask):
    out = batched_forward(x, params, n_outer_blocks, n_blocks, mask)
    return jnp.argmax(out, axis=-1)


@jit
def test_a_batch(batch, params):
    batch = batch[:, :original_n_unmasked]

    while batch.shape[-1] != (seq_len + shrink_factor):
        out = batched_forward(batch, params, n_outer_blocks, n_blocks, mask = jnp.zeros(batch.shape[-1] // shrink_factor))
        out = jnp.argmax(out, axis=-1)
        out = out[:, -shrink_factor:]
        batch = jnp.concatenate([batch, out], axis=-1)

    return batch


def test_and_save(test_loader, params, epoch):
    batch = jnp.array(next(iter(test_loader))[0])
    batch = test_a_batch(batch, params)
    save_batch_images(batch, batch_index=0, epoch_index=epoch)


def train_and_test(train_loader, test_loader, params, opt_state):
    for epoch in range(66):
        total_loss = 0 
        print('Epoch: ' + str(epoch + 1))
        for batch in tqdm(train_loader):
            batch = batch[0]
            x, y = jnp.array(batch[:, :-shrink_factor]), jnp.array(batch[:, shrink_factor:])
            loss, params, opt_state = train_step(params, opt_state, x, y)
            total_loss += loss
        print(f"Average epoch loss: {total_loss / len(train_loader)}")
        
        # save pickle
        with open(f"params.pkl", "wb") as f:
            pickle.dump(params, f)
        
        if (epoch) % 1 == 0:
            test_and_save(test_loader, params, epoch)


n_outer_blocks = 4
n_transformers = 5
n_blocks       = 6
n_heads        = 6
num_classes    = 256 # same as d_out
d_model        = 64 # same as feature size
d_qk           = 8
d_v            = 8
shrink_factor  = 28
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
train_loader, test_loader = create_mnist_dataset(bsz=batch_size)

count_params(params)
print("Available devices:", devices())

train_and_test(train_loader, test_loader, params, opt_state)


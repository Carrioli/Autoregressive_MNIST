import numpy as np

# Example dimensions
seq_len, d_model, shrink_factor = 9, 2, 3

# set random seed for reproducibility
np.random.seed(4)


def shrink(x, e):
    x = x.reshape(-1, shrink_factor, d_model)
    e = e[:x.shape[0]]
    x = np.matmul(e, x)
    x = np.squeeze(x, axis=1)
    return x


def expand(x, f):
    x = np.expand_dims(x, axis=1)
    f   = f[:x.shape[0]]
    x = np.matmul(f, x)
    x = x.reshape(-1, d_model)
    return x


x = np.random.randint(0, 10, (seq_len, d_model))

shrink_params = np.random.randint(0, 10, (seq_len // shrink_factor, 1, shrink_factor))
expand_params = np.random.randint(0, 10, (seq_len // shrink_factor, shrink_factor, 1))


shrunken = shrink(x, shrink_params)
expanded = expand(shrunken, expand_params)

y = x[:6, :]

shrunken2 = shrink(y, shrink_params)
expanded2 = expand(shrunken2, expand_params)

print()

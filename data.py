
import torch
from torchvision import datasets, transforms


def transform(matrix, patch_shape):
    
    if matrix.shape[0] % patch_shape[0] != 0 or matrix.shape[1] % patch_shape[1] != 0:
        raise ValueError("patch_shape should divide original_size along each dimension.")
    
    patches_per_dim = matrix.shape[0] // patch_shape[0], matrix.shape[1] // patch_shape[1]
    reshaped_matrix = matrix.reshape(patches_per_dim[0], patch_shape[0], patches_per_dim[1], patch_shape[1])
    transposed_matrix = reshaped_matrix.permute(0, 2, 1, 3)
    
    return transposed_matrix.flatten()



def create_mnist_dataset(bsz, patch_shape):
    print("[*] Generating MNIST Sequence Modeling Dataset...")

    tf = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(
                lambda x: transform((x * 255).int().squeeze(), patch_shape)
            )
        ]
    )

    train = datasets.MNIST(
        "./data", train=True, download=True, transform=tf
    )
    test = datasets.MNIST(
        "./data", train=False, download=True, transform=tf
    )

    # Return data loaders, with the provided batch size
    trainloader = torch.utils.data.DataLoader(
        train,
        batch_size=bsz,
        shuffle=True,
    )
    testloader = torch.utils.data.DataLoader(
        test,
        batch_size=bsz,
        shuffle=False,
    )

    return trainloader, testloader

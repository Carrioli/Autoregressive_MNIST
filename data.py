
import torch
from torchvision import datasets, transforms


def create_mnist_dataset(bsz=128):
    print("[*] Generating MNIST Sequence Modeling Dataset...")

    tf = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(
                lambda x: (torch.flatten(x * 255)).int().contiguous()
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

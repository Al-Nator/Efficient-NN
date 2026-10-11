from torchvision import datasets, transforms


def load_data(root="data"):
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.24703223, 0.24348512, 0.26158784)
    normalize = [transforms.ToTensor(), transforms.Normalize(mean, std)]
    augment = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
    transform = transforms.Compose(augment + normalize)
    train = datasets.CIFAR10(root, train=True, download=True, transform=transform)
    pool = datasets.CIFAR10(root, train=True, transform=transforms.Compose(normalize))
    test = datasets.CIFAR10(root, train=False, transform=transforms.Compose(normalize))
    return train, pool, test

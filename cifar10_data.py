"""
cifar-10 dataset, with support for random labels and transformations.
"""
import numpy as np
import torch
import torchvision.datasets as datasets


class CIFAR10RandomLabels(datasets.CIFAR10):
    """CIFAR10 dataset, with support for randomly corrupt labels.

    Params
    ------
    corrupt_prob: float
        Default 0.0. The probability of a label being replaced with
        random label.
    num_classes: int
        Default 10. The number of classes in the dataset.
    transform_mode: str
        One of {"none", "shuffled_pixels", "random_pixels", "gaussian"}.
        Applies corresponding transformation to the images.
    """
    def __init__(self, corrupt_prob=0.0, num_classes=10, transform_mode="none", **kwargs):
        super(CIFAR10RandomLabels, self).__init__(**kwargs)
        self.n_classes = num_classes
        self.transform_mode = transform_mode
        if corrupt_prob > 0:
            self.corrupt_labels(corrupt_prob)

        # Use predefined means and stds from get_data_loaders
        self.dataset_mean = [125.3 / 255.0, 123.0 / 255.0, 113.9 / 255.0]
        self.dataset_std = [63.0 / 255.0, 62.1 / 255.0, 66.7 / 255.0]

        if self.transform_mode == "shuffled_pixels":
            self.pixel_permutation = np.random.permutation(32 * 32 * 3)

    def corrupt_labels(self, corrupt_prob):
        labels = np.array(self.targets)
        np.random.seed(12345)
        mask = np.random.rand(len(labels)) <= corrupt_prob
        rnd_labels = np.random.choice(self.n_classes, mask.sum())
        labels[mask] = rnd_labels
        labels = [int(x) for x in labels]  # Cast to int for PyTorch compatibility
        self.targets = labels

    def __getitem__(self, index):
        img, target = super(CIFAR10RandomLabels, self).__getitem__(index)

        if self.transform_mode == "shuffled_pixels":
            img = self.apply_shuffled_pixels(img)
        elif self.transform_mode == "random_pixels":
            img = self.apply_random_pixels(img)
        elif self.transform_mode == "gaussian":
            img = self.apply_gaussian_noise(img)

        return img, target

    def apply_shuffled_pixels(self, img):
        img_np = np.array(img).reshape(-1)  # Flatten image
        img_np = img_np[self.pixel_permutation]  # Apply fixed permutation
        return torch.tensor(img_np, dtype=torch.float32).view(3, 32, 32)

    def apply_random_pixels(self, img):
        img_np = np.array(img).reshape(-1)  # Flatten image
        random_permutation = np.random.permutation(len(img_np))
        img_np = img_np[random_permutation]  # Apply random permutation
        return torch.tensor(img_np, dtype=torch.float32).view(3, 32, 32)

    def apply_gaussian_noise(self, img):
        # Generate Gaussian noise based on predefined dataset mean and std
        noise = np.random.normal(
            loc=self.dataset_mean, scale=self.dataset_std, size=np.array(img).shape
        )
        return torch.tensor(noise, dtype=torch.float32).view(3, 32, 32)
import numpy as np
import os
import shutil
from neural_network.utils import one_hot_vector
from data_processing.utils import PCA
from PIL import Image


class Dataset:
    def __init__(
        self,
        path: str = None,
    ):
        if path:
            self.data, self.label, self.keys = self._read_data(path)

    def __getitem__(self, idx):
        return (
            self.data[idx].reshape(self.data[idx].shape[-2] * self.data[idx].shape[-1]),
            one_hot_vector(self.label[idx], length=len(self.keys)),
        )

    def __len__(self):
        return len(self.data)

    def _read_data(self, path: str):
        # Create dictionary based on face orientation
        data_dict = {"left": [], "right": [], "straight": [], "up": []}

        for human in os.listdir(path):
            if not human.startswith("."):
                for image in os.listdir(os.path.join(path, human)):
                    if (
                        image.endswith(".pgm")
                        and not image.endswith("4.pgm")
                        and not image.endswith("2.pgm")
                    ):
                        key = image.split("_")[1]
                        data_dict[key].append(os.path.join(path, human, image))

        # Create folders with respective labels

        orient_folder = "face_orientation"

        location = os.path.join(path, "..", "..")
        if not os.path.exists(os.path.join(location, orient_folder)):
            os.mkdir(os.path.join(location, orient_folder))
        face_orientation = os.path.join(location, orient_folder)

        for key in data_dict.keys():
            if not os.path.exists(os.path.join(face_orientation, key)):
                os.mkdir(os.path.join(face_orientation, key))
            path = os.path.join(face_orientation, key)
            for image in data_dict[key]:
                shutil.copy(image, os.path.join(path, os.path.split(image)[-1]))

        # Gather all data
        keys = ["left", "right", "straight", "up"]

        data = []
        label = []

        for i, key in enumerate(keys):
            for image in os.listdir(os.path.join(face_orientation, key)):
                data.append(
                    Image.open(os.path.join(face_orientation, key, image)).convert(
                        mode="L"
                    )
                )
                label.append(i)

        data = np.array(data)
        data = data.reshape(-1, data.shape[-2], data.shape[-1])

        label = np.array(label)

        return data, label, np.array(keys)

    def normalize(self, mean=None, std=None):

        if (mean is None) or (std is None):
            self.mean = np.mean(self.data, axis=0)
            self.std = np.std(self.data, axis=0)
        else:
            self.mean, self.std = mean, std

        self.data = (self.data - self.mean) / self.std


class PCADataset(Dataset):
    def __init__(self, path: str = None, k: int = None):
        super().__init__(path)
        self.orig_shape = self.data[0].shape
        self._pca(k)

    def _pca(self, k: int = None):
        self.data = self.data.reshape(self.data.shape[0], -1)
        self.center, self.lambdas, self.vt, self.eig_idx = PCA(
            self.data, k, self.data[0].shape
        )

        if k:
            self.eig_idx = k

        coeffs = self.data @ self.vt[: self.eig_idx].T
        self.data = coeffs.reshape(-1, self.eig_idx, 1)

        # Compute coefficients and reconstruct image
        # coeffs = X[0].reshape((1, -1)) @ vt[:eig_idx].T
        # recon = coeffs @ vt[:eig_idx]

    def __getitem__(self, idx):
        return (
            self.data[idx],
            one_hot_vector(self.label[idx], length=len(self.keys)),
        )

    def get_reconstructed_image(self, idx):
        print(self.data[idx].shape)
        print(self.vt[: self.eig_idx].shape)
        print(self.orig_shape)
        return (self.data[idx].T @ self.vt[: self.eig_idx]).reshape(
            self.orig_shape
        ) + self.center.reshape(self.orig_shape)


def train_test_split(dataset: Dataset, ratios=(0.8, 0.0, 0.2)):

    shuffle_order = np.random.permutation(np.arange(len(dataset)))
    dataset.data = dataset.data[shuffle_order].reshape(
        -1, dataset.data.shape[-2], dataset.data.shape[-1]
    )
    dataset.label = dataset.label[shuffle_order].reshape(dataset.label.shape[-1])

    # train_idxs = np.random.choice(range(len(dataset.data)))
    train_idx = int(ratios[0] * len(dataset))
    validation_idx = int((ratios[0] + ratios[1]) * len(dataset))
    train_dataset = Dataset()
    train_dataset.data = dataset.data[:train_idx]
    train_dataset.label = dataset.label[:train_idx]
    train_dataset.keys = dataset.keys
    train_dataset.normalize()

    validation_dataset = Dataset()
    validation_dataset.data = dataset.data[train_idx:validation_idx]
    validation_dataset.label = dataset.label[train_idx:validation_idx]
    validation_dataset.keys = dataset.keys
    validation_dataset.normalize(train_dataset.mean, train_dataset.std)

    test_dataset = Dataset()
    test_dataset.data = dataset.data[validation_idx:]
    test_dataset.label = dataset.label[validation_idx:]
    test_dataset.keys = dataset.keys
    test_dataset.normalize(train_dataset.mean, train_dataset.std)

    return train_dataset, validation_dataset, test_dataset

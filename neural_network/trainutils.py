import numpy as np
import os
import shutil
from neural_network.utils import one_hot_vector
from data_processing.utils import PCA
from PIL import Image
from neural_network.net import NeuralNetwork
from neural_network.layer import *
from tqdm import tqdm


class Dataset:
    def __init__(
        self,
        path: str = None,
        flatten: bool = True
    ):
        if path:
            self.data, self.label, self.keys = self._read_data(path)

        self.flatten = flatten

    def __getitem__(self, idx):
        if self.flatten:
            return (
                self.data[idx].reshape(self.data[idx].shape[-2] * self.data[idx].shape[-1]),
                one_hot_vector(self.label[idx], length=len(self.keys)),
            )
        else:
            return (
                self.data[idx].reshape(-1, 1, self.data[idx].shape[-2], self.data[idx].shape[-1]),
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


class NMFDataset(Dataset):
    def __init__(self, path: str = None, flatten=True):
        super().__init__(path)
        self.flatten = flatten
        self.selected_indices = np.arange(64)
    
    def update(self):
        W_numerator = np.matmul(self.data, self.H.T)
        W_denominator = np.matmul(np.matmul(self.W, self.H), self.H.T)
        W_alpha = np.divide(W_numerator, W_denominator)
        
        self.W = self.W*W_alpha
        
        H_numerator = np.matmul(self.W.T, self.data)
        H_denominator = np.matmul(np.matmul(self.W.T, self.W), self.H)
        H_alpha = np.divide(H_numerator, H_denominator)
        
        self.H = self.H*H_alpha

    def nmf(self, k: int = None, convergence: float = 0.9999):
        
        self.data = self.data.reshape(self.data.shape[0], -1)
        self.nnmf_scaler = np.sqrt(np.mean(self.data)/self.data.shape[0])
        self.nnmf_loss = []

        # Initialize non-negative matrices
        self.W = abs(np.random.standard_normal((self.data.shape[0], k)))*self.nnmf_scaler
        self.H = abs(np.random.standard_normal((k, self.data.shape[-1])))*self.nnmf_scaler

        self.nnmf_loss.append(np.linalg.norm(self.data - np.matmul(self.W, self.H), "fro"))

        self.update()

        self.nnmf_loss.append(np.linalg.norm(self.data - np.matmul(self.W, self.H), "fro"))

        while self.nnmf_loss[-1]/self.nnmf_loss[-2] < convergence:
            
            self.update()
            self.nnmf_loss.append(np.linalg.norm(self.data - np.matmul(self.W, self.H), "fro"))

        self.data = self.W.reshape(-1, k, 1)
        self.selected_indices = np.arange(k)

    def __getitem__(self, idx):
        return (
            self.data[idx, self.selected_indices].reshape(len(self.selected_indices), 1),
            one_hot_vector(self.label[idx], length=len(self.keys)),
        )

    def get_reconstructed_image(self, idx):
        coeffs = self.W[idx, :].reshape(1, -1)
        recon = np.matmul(coeffs, self.H)
        return recon


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


def train_test_split(dataset: Dataset, ratios=(0.8, 0.0, 0.2), mode: str ="shuffle", shift: int = 0, dataset_type="default"):

    if mode == "shuffle":
        shuffle_order = np.random.permutation(np.arange(len(dataset)))
        dataset.data = dataset.data[shuffle_order].reshape(
            -1, dataset.data.shape[-2], dataset.data.shape[-1]
        )
        dataset.label = dataset.label[shuffle_order].reshape(dataset.label.shape[-1])
    elif mode == "shift":
        shuffle_order = np.roll(np.arange(len(dataset)), shift)
        dataset.data = dataset.data[shuffle_order].reshape(
            -1, dataset.data.shape[-2], dataset.data.shape[-1]
        )
        dataset.label = dataset.label[shuffle_order].reshape(dataset.label.shape[-1])

    # train_idxs = np.random.choice(range(len(dataset.data)))
    train_idx = int(ratios[0] * len(dataset))
    validation_idx = int((ratios[0] + ratios[1]) * len(dataset))
    if dataset_type == "nmf":
        train_dataset = NMFDataset(flatten=dataset.flatten)
        train_dataset.flatten = dataset.flatten
    else:
        train_dataset = Dataset(flatten=dataset.flatten)
    train_dataset.data = dataset.data[:train_idx]
    train_dataset.label = dataset.label[:train_idx]
    train_dataset.keys = dataset.keys
    if dataset_type != "nmf":
        train_dataset.normalize()

    if dataset_type == "nmf":
        validation_dataset = NMFDataset(flatten=dataset.flatten)
    else:
        validation_dataset = Dataset(flatten=dataset.flatten)
        
    validation_dataset.data = dataset.data[train_idx:validation_idx]
    validation_dataset.label = dataset.label[train_idx:validation_idx]
    validation_dataset.keys = dataset.keys
    if dataset_type != "nmf":
        validation_dataset.normalize(train_dataset.mean, train_dataset.std)

    if dataset_type == "nmf":
        test_dataset = NMFDataset(flatten=dataset.flatten)
    else:
        test_dataset = Dataset(flatten=dataset.flatten)
    test_dataset.data = dataset.data[validation_idx:]
    test_dataset.label = dataset.label[validation_idx:]
    test_dataset.keys = dataset.keys
    if dataset_type != "nmf":
        test_dataset.normalize(train_dataset.mean, train_dataset.std)

    return train_dataset, validation_dataset, test_dataset

def train(model: NeuralNetwork, train_dataset: Dataset, validation_dataset: Dataset = None, epochs: int = 100, lr: float = 1e-3, validation_period: int = 5, seed: int = None, verbose=False):
    train_losses = []
    validation_losses = []
    train_confmats = []
    validation_confmats = []

    if seed is not None:
        np.random.seed(seed)

    pbar = tqdm(range(epochs)) if verbose else range(epochs)
    for epoch in pbar:
        train_loss = 0
        train_confmat = np.zeros((len(train_dataset.keys), len(train_dataset.keys)))
        for data, label in train_dataset:
            #data = data.reshape(-1, 1)
            out = model.forward(data)
            loss = model.loss_layer.forward(out, label)
            train_loss += loss
            train_confmat[np.argmax(label), np.argmax(out)] += 1
            model.backward()
            model.step(lr=lr)
        
        if validation_dataset is not None and epoch % validation_period == 0:
            validation_loss = 0
            validation_confmat = np.zeros((len(validation_dataset.keys), len(validation_dataset.keys)))
            for data, label in validation_dataset:
                #data = data.reshape(-1, 1)
                out = model.forward(data)
                loss = model.loss_layer.forward(out, label)
                validation_loss += loss
                validation_confmat[np.argmax(label), np.argmax(out)] += 1
                
        train_loss = train_loss / len(train_dataset)
        validation_loss = validation_loss / len(validation_dataset)
        
        train_losses.append(train_loss)
        validation_losses.append(validation_loss)
        train_confmats.append(train_confmat)
        validation_confmats.append(validation_confmat)

        if verbose:
            pbar.set_description(str(train_loss))

    if validation_dataset is not None:
        validation_loss = 0
        validation_confmat = np.zeros((len(validation_dataset.keys), len(validation_dataset.keys)))
        for data, label in validation_dataset:
            #data = data.reshape(-1, 1)
            out = model.forward(data)
            loss = model.loss_layer.forward(out, label)
            validation_loss += loss
            validation_confmat[np.argmax(label), np.argmax(out)] += 1
    
    return {"model": model, "train_losses": train_losses, "validation_losses": validation_losses, "train_confmats": train_confmats, "validation_confmats": validation_confmats}

def k_fold_cross_validation(k: int, model: NeuralNetwork, dataset, epochs: int = 100, lr: float = 1e-3, validation_period: int = 5, seed: int = None, dataset_type="default"):
    results = []

    # shuffle once
    shuffle_order = np.random.permutation(np.arange(len(dataset)))
    dataset.data = dataset.data[shuffle_order].reshape(
        -1, dataset.data.shape[-2], dataset.data.shape[-1]
    )
    dataset.label = dataset.label[shuffle_order].reshape(dataset.label.shape[-1])
    #pbar = tqdm(range(k), desc="Folds")
    for fold in range(k):
        print(f"Fold {fold}")
        model.init_weights()
        train_dataset, validation_dataset, _ = train_test_split(dataset, ratios=((k - 1) / k, 1 / k, 0), mode="shift", shift=len(dataset) // k, dataset_type=dataset_type)
        result = train(model, train_dataset, validation_dataset, epochs, lr, validation_period, seed+fold if seed is not None else seed, verbose=True)
        results.append(result)
    
    return results

def create_model(input_shape: int, output_shape: int, min_size=4):
    linear1 = Linear(input_shape, max(input_shape//2, min_size))
    relu1 = ReLU(linear1)

    linear2 = Linear(max(input_shape//2, min_size), max(input_shape//4, min_size), relu1)
    relu2 = ReLU(linear2)

    linear3 = Linear(max(input_shape//4, min_size), max(input_shape//8, min_size), relu2)
    relu3 = ReLU(linear3)

    linear4 = Linear(max(input_shape//8, min_size), output_shape, relu3)
    softmaxlayer = Softmax(linear4)

    loss_layer = CrossEntropy(softmaxlayer)

    model = NeuralNetwork(linear1, softmaxlayer, loss_layer)
    return model


def backward_feature_selection(dataset, k_fold=5, idx_amount=4):

    model = create_model(dataset.data.shape[-2], 4)
    init_results = k_fold_cross_validation(k_fold, model, dataset, epochs=1)
    init_val_loss = sum([d["validation_losses"][-1] for d in init_results])/len(init_results)
    remaining_indices = np.arange(dataset.data.shape[-2])
    model = create_model(dataset.data.shape[-2], 4)

    prev_fold_loss = init_val_loss
    min_fold_loss, worst_idx = backward_feature_selection_step(model, dataset, remaining_indices)
    print(len(remaining_indices), "min:", min_fold_loss, "prev", prev_fold_loss, worst_idx)
    

    while (min_fold_loss < prev_fold_loss) and (len(remaining_indices) > idx_amount):
        
        remaining_indices = np.delete(remaining_indices, worst_idx)
        
        model = create_model(len(remaining_indices), 4)
        prev_fold_loss = min_fold_loss

        min_fold_loss, worst_idx = backward_feature_selection_step(model, dataset, remaining_indices)
        print(len(remaining_indices), "min:", min_fold_loss, "prev", prev_fold_loss, worst_idx)
    
    return remaining_indices


def backward_feature_selection_step(model, dataset, remaining_indices, k_fold=5, idx_amount=4):
    
    np.random.shuffle(remaining_indices)
    cur_iter_remaining_indices = np.copy(remaining_indices)
    max_fold_loss, min_fold_loss = -np.inf, np.inf
    worst_idx = None
    print(int(np.ceil(len(cur_iter_remaining_indices)/idx_amount)))
    for idx in range(int(np.ceil(len(cur_iter_remaining_indices)/idx_amount))):
        # random_indices = np.random.choice(cur_iter_remaining_indices, idx_amount)
        random_indices = cur_iter_remaining_indices[idx*idx_amount:min((idx+1)*idx_amount, len(cur_iter_remaining_indices))]
        dataset.selected_indices = np.delete(remaining_indices, random_indices)
        fold_results = k_fold_cross_validation(k_fold, model, dataset, epochs=1)
        fold_val_loss = sum([d["validation_losses"][-1] for d in fold_results])/len(fold_results)
        
        if fold_val_loss > max_fold_loss:
            max_fold_loss = fold_val_loss
            worst_idx = random_indices
        
        min_fold_loss = min(min_fold_loss, fold_val_loss)
        print(f"current min: {min_fold_loss}, new loss: {fold_val_loss}")
        
    
    return min_fold_loss, worst_idx
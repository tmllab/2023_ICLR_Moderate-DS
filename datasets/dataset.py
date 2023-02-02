# Code adapted from pytorch implementation of CIFAR (https://github.com/pytorch/vision/blob/main/torchvision/datasets/cifar.py)

import os.path
import pickle
from typing import Any, Callable, Optional, Tuple
import numpy as np
from PIL import Image
from .folder import ImageFolder

from .utils import check_integrity, download_and_extract_archive
from .vision import VisionDataset


class CIFAR10(VisionDataset):
    """
    Adapted version of CIFAR10 dataset
    """
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """

    base_folder = "cifar-10-batches-py"
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = "c58f30108f718f92721af3b95e74349a"
    train_list = [
        ["data_batch_1", "c99cafc152244af753f735de768cd75f"],
        ["data_batch_2", "d4bba439e000b95fd0a9bffe97cbabec"],
        ["data_batch_3", "54ebc095f3ab1f0389bbae665268c751"],
        ["data_batch_4", "634d18415352ddfa80567beed471001a"],
        ["data_batch_5", "482c414d41f54cd18b22e5b47cb7c3cb"],
    ]

    test_list = [
        ["test_batch", "40351d587109b95175f43aff81a1287e"],
    ]
    meta = {
        "filename": "batches.meta",
        "key": "label_names",
        "md5": "5ff9c542aee3614f3951f8cda6e48888",
    }

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:

        super().__init__(root, transform=transform, target_transform=target_transform)

        self.train = train  # training set or test set

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. You can use download=True to download it")

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data: Any = []
        self.targets = []

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, "rb") as f:
                entry = pickle.load(f, encoding="latin1")
                self.data.append(entry["data"])
                if "labels" in entry:
                    self.targets.extend(entry["labels"])
                else:
                    self.targets.extend(entry["fine_labels"])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC
        
        self._load_meta()

    def _load_meta(self) -> None:
        path = os.path.join(self.root, self.base_folder, self.meta["filename"])
        if not check_integrity(path, self.meta["md5"]):
            raise RuntimeError("Dataset metadata file not found or corrupted. You can use download=True to download it")
        with open(path, "rb") as infile:
            data = pickle.load(infile, encoding="latin1")
            self.classes = data[self.meta["key"]]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return index, img, target

    def __len__(self) -> int:
        return len(self.data)

    def _check_integrity(self) -> bool:
        root = self.root
        for fentry in self.train_list + self.test_list:
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self) -> None:
        if self._check_integrity():
            print("Files already downloaded and verified")
            return
        download_and_extract_archive(self.url, self.root, filename=self.filename, md5=self.tgz_md5)

    def extra_repr(self) -> str:
        split = "Train" if self.train is True else "Test"
        
        return f"Split: {split}"


class CIFAR100(CIFAR10):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    This is a subclass of the `CIFAR10` Dataset.
    """

    base_folder = "cifar-100-python"
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = "eb9058c3a382ffc7106e4002c42a8d85"
    train_list = [
        ["train", "16019d7e3df5f24257cddd939b257f8d"],
    ]

    test_list = [
        ["test", "f0ef6b0ae62326f3e7ffdfab6717acfc"],
    ]
    meta = {
        "filename": "meta",
        "key": "fine_label_names",
        "md5": "7973b15100ade9c7d40fb424638fde48",
    }


class CIFAR100Core(CIFAR100):
    """
        Coreset constructed from CIFAR-100
    """
    
    def __init__(self, drop_id, **kwargs):
        # drop_id must be a list containing index to drop
        super().__init__(**kwargs)
        self.data = np.delete(self.data, drop_id, axis=0)
        self.targets = np.array(self.targets)
        self.targets = np.delete(self.targets, drop_id, axis=0)
        

class CIFAR100Corrupt(VisionDataset):
    
    base_folder = "cifar-100-corrupt"
    
    def __init__(self, root, transform=None):
        super().__init__(root, transform=transform)
        
        img_path = os.path.join(root, self.base_folder, "img.bin")
        target_path = os.path.join(root, self.base_folder, "targets.bin")
        
        with open(img_path, "rb") as f1:
            self.data = pickle.load(f1)
        with open(target_path, "rb") as f2:
            self.targets = pickle.load(f2)
    
    def __getitem__(self, index: int):
        img, target = self.data[index], self.targets[index]
        
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)
            
        return index, img, target
    
    def __len__(self):
        return len(self.data)
        
        
class CIFAR100CorruptCore(CIFAR100Corrupt):
    
    def __init__(self, drop_id, **kwargs):
        super().__init__(**kwargs)
        self.data = np.delete(self.data, drop_id, axis=0)
        self.targets = np.array(self.targets)
        self.targets = np.delete(self.targets, drop_id, axis=0)


class CIFAR100Noisy(CIFAR100):
    def __init__(self, root, **kwargs):
        super().__init__(root=root, **kwargs)
        label_path = os.path.join(root, "noisy/cifar.npy")
        self.targets = np.load(label_path)


class CIFAR100NoisyCore(CIFAR100Noisy):
    def __init__(self, drop_id, **kwargs):
        super().__init__(**kwargs)
        self.data = np.delete(self.data, drop_id, axis=0)
        self.targets = np.delete(self.targets, drop_id, axis=0)
        

class CIFAR100Attack(CIFAR100):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.data = np.load("data/attack/cifar_train.npy")
        
    def __getitem__(self, index: int):
        img = self.data[index]
        img = Image.fromarray(img)
        
        label = self.targets[index]
        if self.transform is not None:
            img = self.transform(img)
        return index, img, label
    

class CIFAR100AttackCore(CIFAR100Attack):
    def __init__(self, drop_id, **kwargs):
        super().__init__(**kwargs)
        self.data = np.delete(self.data, drop_id, axis=0)
        self.targets = np.array(self.targets)
        self.targets = np.delete(self.targets, drop_id, axis=0)


def pil_loader(path: str):
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


class TinyNoisy(ImageFolder):
    def __init__(self, **kwargs):
        super().__init__(loader=pil_loader, **kwargs)
        self.data = np.load("data/noisy/tiny_img.npy")
        self.target = np.load("data/noisy/tiny_target.npy")
    
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, index):
        sample = self.data[index]
        target = self.target[index]
        sample = Image.fromarray(sample)
        
        if self.transform is not None:
            sample = self.transform(sample)
        return index, sample, target


class TinyAttack(ImageFolder):
    def __init__(self, **kwargs):
        super().__init__(loader=pil_loader, **kwargs)
        
        self.data = np.load("data/attack/tiny_img.npy")
        self.target = np.load("data/attack/tiny_target.npy")
    
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, index):
        sample = self.data[index]
        target = self.target[index]
        sample = Image.fromarray(sample)
        
        if self.transform is not None:
            sample = self.transform(sample)
        return index, sample, target


class TinyAttackCore(TinyAttack):
    def __init__(self, drop_id, **kwargs):
        super().__init__(**kwargs)
        drop_id = np.sort(drop_id)
        self.data = np.delete(self.data, drop_id, axis=0)
        self.target = np.array(self.target)
        self.target = np.delete(self.target, drop_id, axis=0)


class TinyNoisyCore(TinyNoisy):
    def __init__(self, drop_id, **kwargs):
        super().__init__(**kwargs)
        drop_id = np.sort(drop_id)
        self.data = np.delete(self.data, drop_id, axis=0)
        self.target = np.array(self.target)
        self.target = np.delete(self.target, drop_id, axis=0)
import os

import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from .data_utils import make_data_matrix, train_test_split


class SemanticDataset(Dataset):

    def __init__(self, samples, root_dir, img_transform=None):
        """
        Args:
            samples (list): each element (item, img path, rel, attr).
            root_dir (string): Directory with all the images.
            img_transform (callable, optional): Optional transform to be applied on an image sample.
        """
        self.samples = samples
        self.root_dir = root_dir
        self.img_transform = img_transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item_name, img_path, rel, attr = self.samples[idx]

        img_path = os.path.join(self.root_dir, img_path)
        image = Image.open(img_path)

        if self.img_transform:
            image = self.img_transform(image)

        return (item_name, image, rel, attr)


class SemanticDataModule(pl.LightningDataModule):

    def __init__(self,
                 root_dir,
                 imgs_per_item,
                 batch_size=16,
                 img_transform=None,
                 test_pcnt=0.2,
                 seed=1945,
                 num_workers=4):
        super().__init__()
        self.root_dir = root_dir
        self.imgs_per_item = imgs_per_item
        self.batch_size = batch_size
        self.test_pcnt = test_pcnt
        self.seed = seed
        self.num_workers = num_workers

        if img_transform:
            self.img_transform = img_transform
        else:
            self.img_transform = transforms.Compose([transforms.CenterCrop(5),
                                                     transforms.ToTensor(),
                                                     ])

    def prepare_data(self):
        samples = make_data_matrix(self.root_dir, self.imgs_per_item)
        self.train_samples, self.test_samples = train_test_split(samples, self.test_pcnt, self.seed)

    def train_dataloader(self):
        train_ds = SemanticDataset(self.train_samples, self.root_dir, self.img_transform)
        train_dl = DataLoader(train_ds, batch_size=self.batch_size, num_workers=self.num_workers)
        return train_dl

    def test_dataloader(self):
        test_ds = SemanticDataset(self.train_samples, self.root_dir, self.img_transform)
        test_dl = DataLoader(test_ds, batch_size=self.batch_size, num_workers=)
        return test_dl
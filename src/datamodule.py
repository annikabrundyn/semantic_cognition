import os

import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from data_utils import make_data_matrix, train_test_split


class SemanticDataset(Dataset):

    def __init__(self, samples, root_dir, img_transform=None, fix_item_class=None):
        """
        Args:
            samples (list): each element (item, img path, rel, attr).
            root_dir (string): Directory with all the images.
            img_transform (callable, optional): Optional transform to be applied on an image sample.
        """
        self.samples = samples
        self.root_dir = root_dir
        self.img_transform = img_transform
        self.fix_item_class = fix_item_class

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item_name, img_path, rel, attr = self.samples[idx]

        img_path = os.path.join(self.root_dir, img_path)
        image = Image.open(img_path).convert("RGB")

        if self.img_transform:
            image = self.img_transform(image)

        return {'item_name':item_name, 'img':image, 'rel':rel, 'attr':attr}




class SemanticDataModule(pl.LightningDataModule):

    def __init__(self,
                 root_dir,
                 imgs_per_item,
                 crop_size,
                 seed,
                 batch_size,
                 num_workers,
                 **kwargs,
                 ):
        super().__init__()
        self.root_dir = root_dir
        self.imgs_per_item = imgs_per_item
        self.crop_size = crop_size
        self.batch_size = batch_size
        #self.test_pcnt = test_pcnt
        self.seed = seed
        self.num_workers = num_workers

        self.img_transform = transforms.Compose([transforms.Resize(self.crop_size),
                                                 transforms.CenterCrop(self.crop_size),
                                                 transforms.ToTensor(),
                                                 ])

        self.item_names = ["canary", "daisy", "oak", "pine", "robin", "rose", "salmon", "sunfish"]

    def prepare_data(self):
        self.samples = make_data_matrix(self.root_dir, self.imgs_per_item)
        self.class_samples = {}

        for item_name in self.item_names:
            self.class_samples[item_name] = [s for s in self.samples if s[0] == item_name]

    def train_dataloader(self):
        train_ds = SemanticDataset(self.samples, self.root_dir, self.img_transform)
        train_dl = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        return train_dl





    # def test_dataloader(self):
    #     test_ds = SemanticDataset(self.train_samples, self.root_dir, self.img_transform)
    #     test_dl = DataLoader(test_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
    #     return test_dl


# dm = SemanticDataModule(root_dir='../data',
#                         imgs_per_item=200,
#                         crop_size=64,
#                         batch_size=4,
#                         num_workers=0)
# dm.prepare_data()
# dl = dm.train_dataloader()



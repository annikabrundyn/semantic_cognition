import os
import numpy as np

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from PIL import Image


def make_data_matrix(root_dir, imgs_per_item):

    with open(os.path.join(root_dir, 'sem_items.txt'), 'r') as fid:
        names_items = np.array([l.strip().lower() for l in fid.readlines()])
    with open(os.path.join(root_dir, 'sem_relations.txt'), 'r') as fid:
        names_relations = np.array([l.strip().lower() for l in fid.readlines()])
    with open(os.path.join(root_dir, 'sem_attributes.txt'), 'r') as fid:
        names_attributes = np.array([l.strip().lower() for l in fid.readlines()])

    nobj = len(names_items)
    nrel = len(names_relations)
    nattr = len(names_attributes)

    data_matrix = np.loadtxt(os.path.join(root_dir, 'sem_data.txt'))
    item_strings = np.apply_along_axis(lambda v: names_items[v.astype('bool')], 1, data_matrix[:, :nobj])
    data2 = np.concatenate((item_strings, data_matrix[:, nobj:]), axis=1, dtype='object')

    img_matrix = []

    for sample in data2:
        item_name = sample[0]
        rel = list(sample[1:1 + nrel])
        attr = list(sample[1 + nrel:])

        for idx in range(1, imgs_per_item + 1):
            item_img = os.path.join(item_name, f"Image_{idx}.jpg")
            matrix_item = [item_name, item_img, rel, attr]
            img_matrix.append(matrix_item)

    return img_matrix


class SemanticDataset(Dataset):

    def __init__(self, samples, root_dir, img_transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
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
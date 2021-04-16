
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


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
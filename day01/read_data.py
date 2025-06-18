from torch.utils.data import Dataset

from PIL import Image

import os


class MyData(Dataset):
    def __init__(self, root_dir, label_dir):
        self.root_dir = root_dir;
        self.label_dir = label_dir;
        self.path = os.path.join(self.root_dir, self.label_dir)
        self.img_path = os.listdir(self.path)

    def __getitem__(self, idx):
        img_name = self.img_path[idx]
        img_item_path = os.path.join(self.root_dir, self.label_dir, img_name)
        print(img_item_path)
        img = Image.open(img_item_path)
        label = self.label_dir
        return img, label

    def __len__self():
        return len(self.img_path)


root_dir = "/Users/hechenyi/ML-learning/dataset/raw-img"
elefante_label_dir = "elefante"
elefante_dataset = MyData(root_dir, elefante_label_dir)

img, label = elefante_dataset[1]

img.show()



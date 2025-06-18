import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils  # 用于拼接图像展示

class MyData(Dataset):
    def __init__(self, root_dir, label_dir, transform=None):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir, self.label_dir)
        self.img_path = os.listdir(self.path)
        self.transform = transform

    def __getitem__(self, idx):
        img_name = self.img_path[idx]
        img_item_path = os.path.join(self.root_dir, self.label_dir, img_name)
        img = Image.open(img_item_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = self.label_dir
        return img, label

    def __len__(self):
        return len(self.img_path)

# ✅ 定义 transform：统一尺寸 + 转为 Tensor
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# 数据集和 DataLoader
root_dir = "/Users/hechenyi/ML-learning/dataset/raw-img"
elefante_label_dir = "elefante"
elefante_dataset = MyData(root_dir, elefante_label_dir, transform=transform)
elefante_dataloader = DataLoader(elefante_dataset, batch_size=4, shuffle=True)
print(elefante_dataset.__len__())
# ✅ 初始化 TensorBoard writer
writer = SummaryWriter("logs")
step = 0
# 从 dataloader 取一个 batch
for imgs, labels in elefante_dataloader:
    step += 1
    print(imgs.shape)  # [4, 3, 224, 224]
    print(labels)

    # 拼成网格图像写入 TensorBoard
    img_grid = vutils.make_grid(imgs, nrow=2, normalize=True)
    writer.add_image("Elefante_Batch", img_grid, step)


writer.close()

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import os

# 定义统一的图片预处理transform
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# 自定义Dataset，支持多类别目录，返回图片和对应标签（0/1）
class MyData(Dataset):
    def __init__(self, root_dir, classes, transform=None):
        self.root_dir = root_dir
        self.classes = classes
        self.transform = transform

        self.img_paths = []
        self.labels = []

        for idx, cls in enumerate(classes):
            cls_path = os.path.join(root_dir, cls)
            for img_name in os.listdir(cls_path):
                self.img_paths.append(os.path.join(cls_path, img_name))
                self.labels.append(idx)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        label = self.labels[idx]
        return img, label

    def __len__(self):
        return len(self.img_paths)

# 定义简单CNN模型
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 64 * 64, 64),  # 输入尺寸为32通道*64*64特征图
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x.squeeze(1)

# 准备设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 路径和类别
root_dir = "/Users/hechenyi/ML-learning/dataset/raw-img"
classes = ['elefante', 'cane']

# 创建完整数据集
dataset = MyData(root_dir, classes, transform=transform)

# 按比例划分训练、验证、测试集 70%, 15%, 15%
total_size = len(dataset)
train_size = int(0.5 * total_size)
val_size = int(0.15 * total_size)
test_size = total_size - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# 创建DataLoader
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 初始化模型、损失函数、优化器
model = SimpleCNN().to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练+验证循环
n_epochs = 10
for epoch in range(n_epochs):
    # 训练
    model.train()
    running_loss = 0.0
    for x, y in train_loader:
        x, y = x.to(device), y.float().to(device)
        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * x.size(0)
    train_loss = running_loss / len(train_dataset)

    # 验证
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for x_val, y_val in val_loader:
            x_val, y_val = x_val.to(device), y_val.float().to(device)
            pred_val = model(x_val)
            loss_val = criterion(pred_val, y_val)
            val_loss += loss_val.item() * x_val.size(0)
    val_loss /= len(val_dataset)

    print(f"Epoch {epoch+1}/{n_epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")

print("训练完成！")

# 保存模型参数
torch.save(model.state_dict(), 'simple_cnn.pth')

# 保存训练检查点
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': n_epochs,
}, 'checkpoint.pth')

# 测试示例（可选）
model.eval()
test_loss = 0.0
correct = 0
total = 0
with torch.no_grad():
    for x_test, y_test in test_loader:
        x_test, y_test = x_test.to(device), y_test.float().to(device)
        pred_test = model(x_test)
        loss_test = criterion(pred_test, y_test)
        test_loss += loss_test.item() * x_test.size(0)
        predicted = (pred_test > 0.5).float()
        correct += (predicted == y_test).sum().item()
        total += y_test.size(0)
test_loss /= len(test_dataset)
accuracy = correct / total
print(f"Test Loss: {test_loss:.4f} - Test Accuracy: {accuracy:.4f}")

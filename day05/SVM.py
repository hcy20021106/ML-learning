import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import os


# 定义特征提取模型，去掉最后两层fc，只输出64维特征
class FeatureExtractor(nn.Module):
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
        self.flatten_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32*64*64, 64),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.flatten_fc(x)
        return x  # 返回64维特征

# 加载训练好的模型参数
feature_extractor = FeatureExtractor().to(device)
feature_extractor.load_state_dict(torch.load('simple_cnn.pth'), strict=False)
feature_extractor.eval()

def extract_features(dataloader):
    features, labels = [], []
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            feat = feature_extractor(x).cpu().numpy()
            features.append(feat)
            labels.append(y.numpy())
    return np.concatenate(features), np.concatenate(labels)

# 提取训练和测试特征
X_train, y_train = extract_features(train_loader)
X_test, y_test = extract_features(test_loader)

# 用sklearn训练SVM
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

svm = SVC(kernel='linear')
svm.fit(X_train, y_train)

y_pred = svm.predict(X_test)
print("SVM Test Accuracy:", accuracy_score(y_test, y_pred))

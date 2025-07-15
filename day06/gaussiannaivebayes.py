import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

class IrisDataset(Dataset):
    def __init__(self, csv_file):
        df = pd.read_csv(csv_file)
        self.X = df.iloc[:, :-1].values.astype(float)
        self.y = pd.factorize(df.iloc[:, -1])[0].astype('int64')
        self.X = torch.tensor(self.X)
        self.y = torch.tensor(self.y)
    def __getitem__(self, index):
        return self.X[index], self.y[index]
    def __len__(self):
        return len(self.y)
dataset = IrisDataset('data.csv')
# 用 DataLoader
loader = DataLoader(dataset, batch_size=2, shuffle=True)

# 迭代示例
for batch_X, batch_y in loader:
    print("Batch X:", batch_X)
    print("Batch y:", batch_y)

X = dataset.X.numpy()
y = dataset.y.numpy()
# -----------------------------
# 4. 划分训练集和测试集
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
# -----------------------------
# 5. 用 GaussianNB 训练
# -----------------------------
model = GaussianNB()
model.fit(X_train, y_train)

# -----------------------------
# 6. 测试并输出准确率
# -----------------------------
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print("---- GaussianNB ----")
print(f"Test Accuracy: {acc:.4f}")

# 如果需要输出预测和真实标签对比：
print("真实标签:", y_test)
print("预测标签:", y_pred)

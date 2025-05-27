import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os

# ================== 模型定义 =====================
class DifficultyRegressor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.rnn = nn.GRU(input_dim, 64, batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        rnn_out, _ = self.rnn(x)  # x: (B, T, F)
        pooled = torch.mean(rnn_out, dim=1)  # (B, 128)
        return self.fc(pooled).squeeze(1)  # (B,)

# ================== 数据集定义 =====================
class ChartDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y.astype(np.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx])

def collate_fn(batch):
    inputs, targets = zip(*batch)
    lens = [b.shape[0] for b in inputs]
    padded = nn.utils.rnn.pad_sequence(inputs, batch_first=True)
    return padded, torch.tensor(targets)

# ================== 主训练流程 =====================
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备：{device}")

    # 加载数据集
    data = np.load("dataset/training_data.npz", allow_pickle=True)
    X, y = data['X'], data['y']

    dataset = ChartDataset(X, y)
    loader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

    input_dim = X[0].shape[1]
    model = DifficultyRegressor(input_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    epochs = 30
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch_x, batch_y in pbar:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            pred = model(batch_x)
            loss = loss_fn(pred, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1} 训练均方误差: {avg_loss:.4f}")

    # 保存模型
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), f"model/difficulty_model.pt")
    print("模型已保存至 modes/difficulty_model.pt")

if __name__ == "__main__":
    train()

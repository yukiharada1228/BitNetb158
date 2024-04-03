import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from bitnet158.nn import BitLinear158


import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# デバイスの設定（GPUが利用可能な場合はGPU、そうでない場合はCPUを使用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ハイパーパラメータの設定
input_size = 28 * 28  # 入力サイズ（MNIST画像のサイズ）
hidden_size = 128     # 隠れ層のサイズ
num_classes = 10      # クラス数
num_epochs = 10       # エポック数
batch_size = 100      # バッチサイズ
learning_rate = 0.001 # 学習率

# MNISTデータセットの読み込みと前処理
transform = transforms.Compose([
    transforms.ToTensor(),  # テンソルに変換
    transforms.Normalize((0.5,), (0.5,))  # 平均0.5、標準偏差0.5で正規化
])

train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# 多層パーセプトロン（MLP）モデルの定義
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MLP, self).__init__()
        self.fc1 = BitLinear158(input_size, hidden_size)
        self.fc2 = BitLinear158(hidden_size, num_classes)
    
    def forward(self, x):
        x = x.view(-1, input_size)  # 画像の形状をバッチサイズ×784に変更
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = MLP(input_size, hidden_size, num_classes).to(device)

# 損失関数と最適化手法の定義
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# モデルの学習
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        # 順伝播と損失の計算
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # 逆伝播とパラメータの更新
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

# モデルの評価
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f'Accuracy of the model on the 10000 test images: {100 * correct / total}%')

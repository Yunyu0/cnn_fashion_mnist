import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import time
import os

# 设置matplotlib中文字体支持
try:
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
except:
    print("使用英文显示")

# 设置随机种子确保结果可复现
torch.manual_seed(42)
np.random.seed(42)

# 检查GPU是否可用
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 定义数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 下载并加载数据集
train_set = torchvision.datasets.FashionMNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform)
test_set = torchvision.datasets.FashionMNIST(
    root='./data',
    train=False,
    download=True,
    transform=transform)
# 数据加载
train_loader = torch.utils.data.DataLoader(
    train_set,
    batch_size=64,
    shuffle=True)
test_loader = torch.utils.data.DataLoader(
    test_set,
    batch_size=64,
    shuffle=False)

# 定义类别标签（英文版避免字体问题）
classes = [
    'T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]


# 展示一些训练图像
def show_images(images, labels):
    plt.figure(figsize=(10, 5))
    for i in range(12):
        plt.subplot(3, 4, i + 1)
        img = images[i].numpy().squeeze()
        plt.imshow(img, cmap='gray')
        plt.title(classes[labels[i]], fontsize=8)
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('fashion_samples.png', dpi=150)
    plt.close()
    print("数据样本展示完成")


# 获取一批训练数据
dataiter = iter(train_loader)
images, labels = next(dataiter)
print(f"训练集大小: {len(train_set)}, 测试集大小: {len(test_set)}")
print(f"批量数据: {images.shape[0]}个样本, {images.shape[2]}x{images.shape[3]}像素")
show_images(images, labels)


# 定义完整模型
class FashionCNN(nn.Module):
    def __init__(self):
        super(FashionCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)  # 输入通道1，输出通道32，卷积核3x3，填充1
        self.pool = nn.MaxPool2d(2, 2)  # 池化层，2x2，步长2
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)  # 输入通道32，输出通道64，卷积核3x3，填充1
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  # 全连接层1，输入64*7*7，输出128
        self.fc2 = nn.Linear(128, 10)  # 全连接层2，输入128，输出10（10个类别）
        self.dropout = nn.Dropout(0.25)  # Dropout层，防止过拟合

    def forward(self, x):
        # 卷积层1 -> 激活函数 -> 池化
        x = self.pool(nn.functional.relu(self.conv1(x)))  # 输出尺寸: [batch, 32, 14, 14]
        # 卷积层2 -> 激活函数 -> 池化
        x = self.pool(nn.functional.relu(self.conv2(x)))  # 输出尺寸: [batch, 64, 7, 7]
        # 展平
        x = x.view(-1, 64 * 7 * 7)
        # Dropout
        x = self.dropout(x)
        # 全连接层1 -> 激活函数
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# 初始化模型
model = FashionCNN().to(device)
print("模型结构:")
print(model)
# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# 训练模型
def train_model(model, train_loader, criterion, optimizer, epochs=10):
    model.train()
    train_losses = []
    train_accuracies = []
    print("开始训练...")
    start_time = time.time()
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # 统计信息
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            # 每100个batch打印一次
            if (i + 1) % 100 == 0:
                print(f'Epoch [{epoch + 1}/{epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')
    elapsed = time.time() - start_time
    print(f'训练完成! 总耗时: {elapsed / 60:.2f}分钟')

    # 绘制训练过程
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='训练损失')
    plt.title('训练损失')
    plt.xlabel('Epoch')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='训练准确率')
    plt.title('训练准确率')
    plt.xlabel('Epoch')
    plt.ylabel('准确率 (%)')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150)
    plt.close()

    return train_losses, train_accuracies


# 训练模型（10个epoch）
train_losses, train_accuracies = train_model(model, train_loader, criterion, optimizer, epochs=10)


# 测试模型
def test_model(model, test_loader):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    print("开始测试...")
    start_time = time.time()

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    test_loss /= len(test_loader)
    test_acc = 100 * correct / total

    elapsed = time.time() - start_time
    print(f'测试完成! 耗时: {elapsed:.2f}秒')
    print(f'测试损失: {test_loss:.4f}, 测试准确率: {test_acc:.2f}%')

    # 生成分类报告
    print("\n分类报告:")
    print(classification_report(all_labels, all_preds, target_names=classes))

    # 生成混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title('混淆矩阵')
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.xticks(rotation=45)
    plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()

    return test_loss, test_acc, all_preds, all_labels


# 测试模型
test_loss, test_acc, all_preds, all_labels = test_model(model, test_loader)


# 可视化测试结果
def visualize_results(test_loader, model, classes, num_images=12):
    model.eval()
    images, labels = next(iter(test_loader))
    images, labels = images.to(device), labels.to(device)

    with torch.no_grad():
        outputs = model(images)
        _, preds = torch.max(outputs, 1)

    images = images.cpu().numpy()
    labels = labels.cpu().numpy()
    preds = preds.cpu().numpy()

    plt.figure(figsize=(15, 10))
    for i in range(num_images):
        plt.subplot(3, 4, i + 1)
        img = images[i].squeeze()
        plt.imshow(img, cmap='gray')
        plt.axis('off')

        title = f"True: {classes[labels[i]]}"
        if preds[i] == labels[i]:
            title += f"\nPred: {classes[preds[i]]}"
            color = 'green'
        else:
            title += f"\nPred: {classes[preds[i]]}"
            color = 'red'

        plt.title(title, color=color, fontsize=10)

    plt.tight_layout()
    plt.savefig('test_predictions.png', dpi=150)
    plt.close()
    print("测试结果可视化完成")


# 可视化测试结果
visualize_results(test_loader, model, classes)

# 保存模型
torch.save(model.state_dict(), 'fashion_mnist_cnn.pth')
print("模型已保存为 'fashion_mnist_cnn.pth'")

print("=" * 50)
print("完整训练和测试完成！")
print("=" * 50)
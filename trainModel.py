import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.optim as optim
from AudioDataProcess import MyData
from Vadmodel import CRNN_VAD, BCEFocalLoss
import glob
import os
from tqdm import tqdm
import matplotlib.pyplot as plt  # 导入Matplotlib用于绘图

# 超参数设置
lr = 0.0001  # 学习率
batch_size = 32  # 每个批次的样本数量
epochs = 50  # 训练的轮数
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 选择设备（GPU或CPU）

# 获取当前脚本所在目录
base_dir = os.path.dirname(os.path.abspath(__file__))

# 数据集路径：指向刚才 split_wav.py 生成数据的 savePath 
# 建议在 data 目录下统一管理
wavPath = os.path.join(base_dir, "data", "processed")  

# 模型保存路径：建议放在项目根目录下的 models 文件夹
modelPath = os.path.join(base_dir, "models")

# 获取所有.wav文件路径
allWav = glob.glob(os.path.join(wavPath, "*.wav"))

# 划分训练集和测试集，前1680个为测试集，其余为训练集
testList = allWav[:1680]
trainList = allWav[1680:]

# 创建训练集和测试集的DataLoader
trainData = MyData(trainList)
trainLoader = DataLoader(trainData, batch_size=batch_size, shuffle=True)  # shuffle=True: 数据集会在每个epoch前打乱顺序
testData = MyData(testList)
testLoader = DataLoader(testData, batch_size=batch_size)

# 创建模型实例
model = CRNN_VAD(257).to(device)  # 257 是特征的数量，传入模型

# 使用 Adam 优化器并调整学习率
optimizer = optim.Adam(model.parameters(), lr=lr)

# 使用自定义的损失函数：BCEFocalLoss
loss = BCEFocalLoss()

# 学习率调度器：根据训练损失自动调整学习率
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=2, verbose=True)

# 提前停止机制的参数
best_accuracy = 0  # 初始化最佳准确率
patience = 8  # 设置耐心值，即在多少轮没有提升后停止训练
counter = 0  # 计数器，用于记录连续多少轮没有提升

# 用于记录损失和准确率
train_losses = []
validation_accuracies = []

# 开始训练过程
for epoch in range(epochs):
    model.train()  # 设置模型为训练模式
    epoch_train_loss = 0  # 初始化训练损失

    # 训练过程：遍历训练集
    for data, labels in tqdm(trainLoader, desc=f"Epoch {epoch + 1}/{epochs}", dynamic_ncols=True, leave=True):
        data, labels = data.to(device), labels.to(device)  # 将数据和标签转移到设备（GPU或CPU）

        optimizer.zero_grad()  # 清除之前的梯度
        outputs = model(data)  # 模型前向传播，获得预测值
        outputs = torch.squeeze(outputs)  # 去掉多余的维度

        # 计算损失
        trainLoss = loss(labels, outputs)  # 使用自定义的损失函数计算训练损失
        trainLoss.backward()  # 反向传播
        optimizer.step()  # 更新模型参数

        epoch_train_loss += trainLoss.item()  # 累加每个批次的损失

    # 记录训练损失
    avg_train_loss = epoch_train_loss / len(trainLoader)  # 计算平均训练损失
    train_losses.append(avg_train_loss)

    # 调整学习率，根据训练损失情况动态调整
    scheduler.step(avg_train_loss)

    # 模型评估：测试集上的准确率
    model.eval()  # 设置模型为评估模式
    correct = 0  # 初始化正确的预测数
    total = 0  # 初始化总样本数

    with torch.no_grad():  # 在评估时不需要计算梯度
        for testData, testLabels in testLoader:
            testData, testLabels = testData.to(device), testLabels.to(device)

            predicts = model(testData)  # 模型前向传播，获得预测结果
            predicts = torch.squeeze(predicts)  # 去掉多余的维度

            # 将预测值大于0.5的设为1，小于0.5的设为0
            predicts = torch.where(predicts > 0.5, torch.tensor(1, device=device).to(device), torch.tensor(0).to(device))

            total += testLabels.numel()  # 计算样本总数
            correct += (predicts == testLabels).sum().item()  # 计算预测正确的样本数

    # 计算并记录当前epoch的准确率
    accuracy = 100 * correct / total
    validation_accuracies.append(accuracy)

    print(f'Epoch {epoch + 1}: Learning rate: {scheduler.get_last_lr()[0]}, Train Loss = {avg_train_loss:.4f}, Validation Accuracy = {accuracy:.2f}%')

    # 提前停止机制：判断是否更新最佳准确率
    if accuracy > best_accuracy:
        best_accuracy = accuracy  # 更新最佳准确率
        counter = 0  # 重置计数器
    else:
        counter += 1  # 增加计数器

    # 如果连续多个epoch准确率没有提升，则停止训练
    if counter >= patience:
        print("Early stopping triggered")
        break

# 保存模型
if not os.path.exists(modelPath):  # 如果模型保存路径不存在，则创建
    os.makedirs(modelPath)
torch.save(model, os.path.join(modelPath, "VADModel.pth"))  # 保存模型权重到指定路径

# 绘制训练损失图像
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss', color='blue')
plt.title('Train Loss vs Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# 绘制验证准确率图像
plt.subplot(1, 2, 2)
plt.plot(validation_accuracies, label='Validation Accuracy', color='green')
plt.title('Validation Accuracy vs Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.legend()

# 保存图像到文件
plt.tight_layout()
plt.savefig(os.path.join(modelPath, "training_results.png"))
plt.show()


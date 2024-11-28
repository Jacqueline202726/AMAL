import logging

from torch.nn.modules.pooling import MaxPool1d
logging.basicConfig(level=logging.INFO)

import heapq
from pathlib import Path
import gzip

from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import sentencepiece as spm

from tp8_preprocess import TextDataset

# Utiliser tp8_preprocess pour générer le vocabulaire BPE et
# le jeu de donnée dans un format compact

# --- Configuration

# Taille du vocabulaire
vocab_size = 1000
MAINDIR = Path(__file__).parent

# Chargement du tokenizer 

# 加载 SentencePiece 分词器
tokenizer = spm.SentencePieceProcessor()
tokenizer.Load(f"wp{vocab_size}.model")
ntokens = len(tokenizer)

# 定义 loaddata() 函数，用于从压缩的 .pth 文件中加载指定模式的数据（如 "train"）
def loaddata(mode):
    with gzip.open(f"{mode}-{vocab_size}.pth", "rb") as fp:
        return torch.load(fp)


train = loaddata("train")
TRAIN_BATCHSIZE=500
TEST_BATCHSIZE=500


# --- Chargements des jeux de données train, validation et test 分割数据集

val_size = 1000
test_size = 10000
train_size = len(train) - val_size -test_size
train, val, test = torch.utils.data.random_split(train, [train_size, val_size,test_size])

# 打印数据集的大小信息和词汇表大小
logging.info("Datasets: train=%d, val=%d, test=%d", train_size, val_size, len(test))
logging.info("Vocabulary size: %d", vocab_size)
# 使用 DataLoader 创建训练、验证和测试数据加载器
train_iter = torch.utils.data.DataLoader(train, batch_size=TRAIN_BATCHSIZE, collate_fn=TextDataset.collate)
val_iter = torch.utils.data.DataLoader(val, batch_size=TEST_BATCHSIZE, collate_fn=TextDataset.collate)
test_iter = torch.utils.data.DataLoader(test, batch_size=TEST_BATCHSIZE, collate_fn=TextDataset.collate)


#  TODO: 
from collections import Counter

# 基准模型：总是返回训练集中多数类别
def majority_class_baseline(data_loader):
    # 计算训练集中标签的出现频率
    all_labels = []
    for _, labels in data_loader:
        all_labels.extend(labels.numpy())
    most_common_label = Counter(all_labels).most_common(1)[0][0]
    
    correct_predictions = sum([1 for label in all_labels if label == most_common_label])
    accuracy = correct_predictions / len(all_labels)
    
    logging.info("Majority class baseline accuracy: %.4f", accuracy)
    return accuracy

# 计算基准模型在训练集上的表现
baseline_accuracy = majority_class_baseline(train_iter)

# 定义CNN
class SimpleCNN(nn.Module):
    def __init__(self, vocab_size, embed_size, num_classes):
        super(SimpleCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.conv1 = nn.Conv1d(in_channels=embed_size, out_channels=100, kernel_size=3)
        self.pool = MaxPool1d(kernel_size=2, stride=2)
        self.fc = nn.Linear(100, num_classes)
        
    def forward(self, x):
        x = self.embedding(x).permute(0, 2, 1)  # (batch_size, embed_size, seq_length)
        x = self.pool(torch.relu(self.conv1(x)))
        x = torch.max(x, dim=2).values  # Global max pooling
        x = self.fc(x)
        return x

import torch.optim as optim
device = "cuda"

# 定义训练模型的函数
def train_model(model, train_loader, criterion, optimizer, num_epochs=5):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = correct / total
        logging.info(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}, Training Accuracy: {epoch_accuracy:.4f}")
    return model

# 定义验证模型的函数
def evaluate_model(model, val_loader):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = correct / total
    logging.info(f"Validation Accuracy: {accuracy:.4f}")
    return accuracy

embed_size = 128  # 嵌入层维度
num_classes = 2   # 类别数量

# 初始化模型
model = SimpleCNN(vocab_size=ntokens, embed_size=embed_size, num_classes=num_classes)
model.to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
trained_model = train_model(model, train_iter, criterion, optimizer, num_epochs=5)

# 在验证集上检测效果
validation_accuracy = evaluate_model(trained_model, val_iter)
logging.info(f"Final Validation Accuracy: {validation_accuracy:.4f}")

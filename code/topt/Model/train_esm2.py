from gwo import GWO  # 导入GWO模块
import torch.nn as nn
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import EsmModel, EsmTokenizer
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

# 检查CUDA是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 读取CSV文件
csv_file = '../Data/train_data_all.csv'  # 替换为您的CSV文件名
data = pd.read_csv(csv_file)
print(data.isna().sum())  # 检查是否有缺失值


# 提取特征和标签
sequences = data['sequence'].tolist()  # 蛋白质序列
topt_values = data['topt'].tolist()  # Topt值

# 对EC值进行拆分和数值化
ec_columns = [col for col in data.columns if col.startswith('ec')]
ec_values = data[ec_columns].values
ec_values = np.array([list(map(int, row)) for row in ec_values])  # 将EC值转为整数类型



# 划分训练集和测试集
X_train, X_test, y_train, y_test, ec_train, ec_test = train_test_split(
    sequences, topt_values, ec_values, test_size=0.2, random_state=42)

# 定义Dataset类
class ProteinDataset(Dataset):
    def __init__(self, sequences, topt_values, ecs):
        self.sequences = sequences
        self.topt_values = topt_values
        self.ecs = ecs          # EC特征

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        topt_value = self.topt_values[idx]
        ec = self.ecs[idx]
        return sequence, torch.tensor(float(topt_value), dtype=torch.float), torch.tensor(ec, dtype=torch.long)

# 加载ESM2模型和分词器
model_name = "facebook/esm2_t30_159M_UR50D"  # 使用您的ESM2模型
tokenizer = EsmTokenizer.from_pretrained(model_name)
esm_model = EsmModel.from_pretrained(model_name).to(device)  # 将模型移到GPU

# 定义带有EC和Domain嵌入的模型
class ESM2ToptPredictor(torch.nn.Module):
    def __init__(self, esm_model, num_ecs, ec_embedding_dim=32, dropout_rate=0.2, num_transformer_layers=6):
        super(ESM2ToptPredictor, self).__init__()
        self.esm_model = esm_model
        
        # EC嵌入层
        self.ec_embedding = torch.nn.EmbeddingBag(num_embeddings=400, embedding_dim=ec_embedding_dim, mode='mean', sparse=False)

        # 增加多头注意力层的数量
        self.attention_layers = torch.nn.ModuleList(
            [torch.nn.MultiheadAttention(embed_dim=640, num_heads=8, dropout=dropout_rate) for _ in range(num_transformer_layers)]
        )

        # 添加线性层以将attn_output的维度从480降低到64
        self.attn_linear = torch.nn.Linear(640, 64)


        # EC特征处理分支
        self.ec_mlp = torch.nn.Sequential(
            torch.nn.Linear(ec_embedding_dim, 64),
            torch.nn.LeakyReLU(negative_slope=0.01),
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(64, 32),
            torch.nn.LeakyReLU(negative_slope=0.01),
            torch.nn.Dropout(dropout_rate)
        )

        # 序列特征处理分支
        self.seq_mlp = torch.nn.Sequential(
            torch.nn.Linear(64, 128),
            torch.nn.BatchNorm1d(128),
            torch.nn.LeakyReLU(negative_slope=0.01),
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(128, 64)
        )

        # 综合层
        self.combined_layer = torch.nn.Linear(64 + 32 + 32, 64)

        # 输出层
        self.output_layer = torch.nn.Linear(64, 1)

    def forward(self, input_ids, attention_mask, ec, domain):
        # 处理序列特征，使用ESM2模型
        outputs = self.esm_model(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # 获取[CLS] token的输出

        # 通过多个Transformer层
        for attention in self.attention_layers:
            cls_output, _ = attention(cls_output.unsqueeze(0), cls_output.unsqueeze(0), cls_output.unsqueeze(0))
            cls_output = cls_output.squeeze(0)

        # 通过线性层调整attn_output的维度
        attn_output = self.attn_linear(cls_output)  # 调整为64维

        # 通过序列特征分支
        x_seq = self.seq_mlp(attn_output) + attn_output  # 残差连接

        # 处理EC嵌入特征
        ec_embedding = self.ec_embedding(ec)
        x_ec = self.ec_mlp(ec_embedding)

        # 结合所有特征
        combined = torch.cat([x_seq, x_ec], dim=1)
        combined = F.leaky_relu(self.combined_layer(combined), negative_slope=0.01)

        # 输出层
        output = self.output_layer(combined)
        output = torch.clamp(output, min=-1e10, max=1e10)  # 避免数值不稳定

        return output
    
# 实例化模型
num_ecs = ec_values.shape[1]  # EC特征的数量
ec_embedding_dim = 32  # EC嵌入的维度
domain_embedding_dim = 8  # Domain嵌入的维度
dropout_rate = 0.2  # Dropout比率
num_transformer_layers = 6

model = ESM2ToptPredictor(esm_model, num_ecs, ec_embedding_dim, dropout_rate, num_transformer_layers).to(device)

# 训练和优化器配置
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)  # 学习率调度器
criterion = torch.nn.MSELoss()


# 定义训练函数
def train(model, train_loader, optimizer, criterion, epoch):
    model.train()
    total_loss = 0.0
    for sequences, targets, ecs in train_loader:
        inputs = tokenizer(sequences, return_tensors="pt", padding=True, truncation=True, max_length=512)
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
        targets = targets.to(device)
        ecs = ecs.to(device)
        # 修复域问题：确保域输入是long类型
        domains = domains.to(device).long()

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask, ecs, domains)
        loss = criterion(outputs.squeeze(), targets)
        loss.backward()
        
         # 添加梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
       
        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch + 1} | Average Loss: {average_loss:.4f}")


# 定义评估函数
def evaluate(model, test_loader):
    model.eval()
    predictions = []
    actuals = []

    with torch.no_grad():
        for sequences, targets, ecs in test_loader:
            inputs = tokenizer(sequences, return_tensors="pt", padding=True, truncation=True, max_length=512)
            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)
            outputs = model(input_ids, attention_mask, ecs.to(device))

            predictions.append(outputs.squeeze().cpu().numpy().flatten())
            actuals.append(targets.numpy())

    predictions = np.concatenate(predictions)
    actuals = np.concatenate(actuals)

    mse = mean_squared_error(actuals, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(actuals, predictions)

    return mse, rmse, r2

# 创建DataLoader
train_dataset = ProteinDataset(X_train, y_train, ec_train)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

test_dataset = ProteinDataset(X_test, y_test, ec_test)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# 定义GWO优化目标函数
def objective_function(params):
    try:
        ec_embedding_dim, learning_rate = params
        ec_embedding_dim = int(ec_embedding_dim)
        learning_rate = float(learning_rate)

        # 实例化模型
        model = ESM2ToptPredictor(esm_model, num_ecs, ec_embedding_dim,  dropout_rate, num_transformer_layers=6).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

        # 训练模型
        for epoch in range(5):  # 只训练少量epoch进行优化
            train(model, train_loader, optimizer, criterion, epoch)

        # 评估模型
        mse, _, _ = evaluate(model, test_loader)
        return mse  # GWO的目标是最小化MSE
    except Exception as e:
        print(f"Error during optimization: {e}")
        return float('inf')  # 遇到问题时返回一个极大值，防止崩溃

# 定义超参数的搜索范围
bounds = np.array([[32, 128], [8, 32], [1e-6, 1e-2]])  # 将lb和ub组合成bounds

# 初步搜索， max_iter = 10
gwo_initial = GWO(objective_function=objective_function, dim=3, bounds=bounds, num_wolves=5, max_iter=10)
best_params_initial, best_score_initial = gwo_initial.optimize()
print(f"Initial search best params found: {best_params_initial}")
print(f"Initial search best score: {best_score_initial}")

# 微调， max_iter = 20 或更高，使用初步搜索得到的最佳参数作为起点
gwo_fine_tune = GWO(objective_function=objective_function, dim=3, bounds=bounds, num_wolves=5, max_iter=20)
best_params_fine_tuned, best_score_fine_tuned = gwo_fine_tune.optimize()
print(f"Fine-tuned best params found: {best_params_fine_tuned}")
print(f"Fine-tuned best score: {best_score_fine_tuned}")

# 获取微调后的最佳超参数
best_ec_embedding_dim, best_domain_embedding_dim, best_learning_rate = best_params_fine_tuned

# 使用微调后的最佳超参数进行最终训练
model = ESM2ToptPredictor(
    esm_model, num_ecs,
    int(best_ec_embedding_dim),
    dropout_rate, num_transformer_layers=6
).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=best_learning_rate, weight_decay=1e-4)

# 正常进行训练
for epoch in range(1000):
    train(model, train_loader, optimizer, criterion, epoch)
    mse, rmse, r2 = evaluate(model, test_loader)
    print(f"Evaluation after Epoch {epoch + 1} | MSE: {mse:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")
    scheduler.step()

# 保存模型
torch.save(model.state_dict(), 'esm2_topt_predictor.pth')
print("模型训练完成，已保存！")

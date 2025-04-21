import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error  # 计算MSE损失


# 设备设置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 全局变量设置
LEARNING_RATE = 1e-4  # 学习率
WEIGHT_DECAY = 1e-5   # 权重衰减
BATCH_SIZE = 32       # 批量大小
EPOCHS = 50           # 训练轮数
SEED = 42             # 随机种子，保证实验可复现
NUMBER_EPOCHS = 50    # 训练轮数

# 设定随机种子
torch.manual_seed(SEED)
np.random.seed(SEED)

Result_Path = './Results/'
Feature_Path = './Features/'
Model_Path = './Model'

# 加载 ESM2 预训练模型
MODEL_NAME = "facebook/esm2_t33_650M_UR50D"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(device)
# 数据文件路径
csv_file_train = "train_data.csv"  # 训练集文件
csv_file_valid = "valid_data.csv"  # 验证集文件

# 交叉验证的 fold 设定（如果使用K折交叉验证）
fold = 5  # 例如5折交叉验证

class ProteinDataset(Dataset):
    def __init__(self, csv_file, tokenizer, Feature_Path):
        self.data = pd.read_csv(csv_file)
        self.tokenizer = tokenizer
        self.feature_path = Feature_Path

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        sequence = row['sequence']
        topt = row['topt']
        features = self.load_features(row['uniprot_id'])  # 加载额外特征

        # 使用 ESM2 的 tokenizer 对序列进行处理
        inputs = self.tokenizer(sequence, return_tensors="pt", padding="max_length", truncation=True, max_length=500)

        # 从 tokenizer 获取 input_ids 和 attention_mask
        input_ids = inputs['input_ids'].squeeze().to(device)
        attention_mask = inputs['attention_mask'].squeeze().to(device)

        # 填充特征矩阵，如果大小不一致
        max_feature_len = 500  # 假设最大特征矩阵的长度为500，可以根据你的实际数据调整
        if len(features.shape) == 1:  # 如果 features 是 1D（只有一个特征），则扩展为 2D
            features = features.unsqueeze(0)  # 将其转换为二维张量

        # 确保特征矩阵有 2 个维度
        assert len(features.shape) == 2, f"Invalid features shape: {features.shape}. Expected 2D tensor."

        if features.shape[0] < max_feature_len:
            padding = torch.zeros(max_feature_len - features.shape[0], features.shape[1])
            features = torch.cat([features, padding], dim=0)
        elif features.shape[0] > max_feature_len:
            features = features[:max_feature_len, :]  # 截断

        return input_ids, attention_mask, features, torch.tensor(topt, dtype=torch.float32)

    def load_features(self, uniprot_id):
        """
        根据 uniprot_id 加载额外的特征（如 BLOSUM, pp7等),
        返回的应该是一个张量。
        """
        feature_file = os.path.join(self.feature_path, f"{uniprot_id}.npy")
    
        if os.path.exists(feature_file):
            feature_matrix = np.load(feature_file)
            # 确保加载的 feature_matrix 是二维的
            if len(feature_matrix.shape) == 1:
                feature_matrix = feature_matrix.reshape(1, -1)  # 如果是 1D，则转换为 2D（1, feature_dim）
            return torch.tensor(feature_matrix, dtype=torch.float32)
        else:
            # 如果没有特征文件，返回一个默认的零向量（这里可以根据需求调整）
            return torch.zeros(1, dtype=torch.float32)  # 默认返回 1x1 的零矩阵


class Attention(nn.Module):
    def __init__(self, input_dim, dense_dim, n_heads):
        super(Attention, self).__init__()
        self.input_dim = input_dim
        self.dense_dim = dense_dim
        self.n_heads = n_heads
        self.fc1 = nn.Linear(self.input_dim, self.dense_dim)  # 映射到较小的维度
        self.fc2 = nn.Linear(self.dense_dim, self.n_heads)    # 输出多个注意力头的权重

    def softmax(self, input, axis=1):
        input_size = input.size()
        trans_input = input.transpose(axis, len(input_size) - 1)
        trans_size = trans_input.size()
        input_2d = trans_input.contiguous().view(-1, trans_size[-1])
        soft_max_2d = torch.softmax(input_2d, dim=1)
        soft_max_nd = soft_max_2d.view(*trans_size)
        return soft_max_nd.transpose(axis, len(input_size) - 1)

    def forward(self, input):  # input.shape = (batch_size, seq_len, input_dim)
        #print("Input to attention layer shape:", input.shape) 
        # 第一步：通过fc1将输入映射到较小的维度
        x = torch.tanh(self.fc1(input))  # x.shape = (batch_size, seq_len, dense_dim)
        #print("After fc1 (x) shape:", x.shape) 
        # 第二步：通过fc2将其映射到n_heads维度
        x = self.fc2(x)  # x.shape = (batch_size, seq_len, n_heads)
        #print("After fc2 (x) shape:", x.shape) 
        # 第三步：使用softmax计算注意力权重
        attention = self.softmax(x, 1)  # attention.shape = (batch_size, seq_len, n_heads)
        attention = x.transpose(1, 2)
        #print("Attention layer output shape:", attention.shape)
        return attention

class ESM2ToptPredictor(nn.Module):
    def __init__(self):
        super(ESM2ToptPredictor, self).__init__()

        # 加载 ESM2 模型和 tokenizer
        self.esm2_model_name = 'facebook/esm2_t12_35M_UR50D'  # ESM2模型名称
        self.tokenizer = AutoTokenizer.from_pretrained(self.esm2_model_name)
        self.esm2 = AutoModel.from_pretrained(self.esm2_model_name)
        
        # 定义注意力模块
        self.attention = Attention(input_dim=480, dense_dim=128, n_heads=4)
        
        # 定义全连接层
        # 输入是 ESM2 输出和其他特征的拼接，调整输入维度为总维度
        self.fc = nn.Linear(217480, 128)  # 640 是 ESM2 的输出维度，500*7 是其他特征维度
        self.dropout = nn.Dropout(p=0.5)  # 50% 概率丢弃神经元
        self.output_layer = nn.Linear(128, 1)  # 最终输出预测值，预测的是温度，因此是1维

        # 使用 Adam 优化器，处理权重和偏置的不同
        weight_p, bias_p = [], []
        for name, p in self.named_parameters():
            if 'bias' in name:
                bias_p += [p]
            else:
                weight_p += [p]

        self.optimizer = torch.optim.Adam([{'params': weight_p, 'weight_decay': WEIGHT_DECAY},
                                           {'params': bias_p, 'weight_decay': 0}], lr=LEARNING_RATE)

        # 损失函数
        self.criterion = nn.MSELoss()
        # 定义加权因子
        self.sequence_weight = 1.0  # 可以调整序列特征的权重
        self.feature_weight = 0.2   # 可以调整其他特征的权重
        
    def forward(self, input_ids, attention_mask, features):
        # 获取 ESM2 的输出
        esm_output = self.esm2(input_ids=input_ids, attention_mask=attention_mask)
        esm_embedding = esm_output.last_hidden_state[:, 0, :]  # 获取 [CLS] token 的输出作为序列特征
        
        # 使用注意力模块处理 ESM2 输出
        att = self.attention(esm_embedding.unsqueeze(0))  # 增加 batch 维度
        
        # 计算注意力输出
        esm_embedding_avg = torch.sum(att @ esm_embedding.unsqueeze(0), dim=1) / self.attention.n_heads
        
        # 计算平均后的 ESM2 输出
        esm_embedding_avg = esm_embedding_avg.squeeze(1)  # [batch_size, hidden_dim]
        
        # 确保 features 是二维张量
        if features.dim() == 3:
            features = features.view(features.size(0), -1)  # 展平为 [batch_size, seq_len * feature_dim]
        
        # 调整 esm_embedding_avg 的 batch size 使其和其他特征一致
        esm_embedding_avg = esm_embedding_avg.expand(features.size(0), -1)  # 将 esm_embedding_avg 扩展到 batch_size
        
        # 拼接 ESM2 输出和其他提取的特征
        combined_features = torch.cat((esm_embedding_avg, features), dim=1)
        
        # 通过全连接层处理拼接后的特征
        dense_output = F.relu(self.fc(combined_features))
        dense_output = self.dropout(dense_output)
        
        # 最终输出层
        output = self.output_layer(dense_output)
        
        # 输出预测值（去除多余维度）
        return output.squeeze(1)
    
def evaluate(model, data_loader, fold):
    model.eval()

    epoch_loss = 0.0
    n_batches = 0
    valid_pred = []
    valid_true = []
    valid_uniprot_ids = []

    # 加入进度条
    for data in data_loader:
        with torch.no_grad():
            input_ids, attention_mask, features, labels = data

            if torch.cuda.is_available():
                input_ids = input_ids.cuda()
                attention_mask = attention_mask.cuda()
                features = features.cuda()
                labels = labels.cuda()
                
            y_pred = model(input_ids, attention_mask, features)
            y_pred = torch.squeeze(y_pred)
            y_true = labels.float() / 120.0  # 如果需要归一化标签
            
            loss = model.criterion(y_pred, y_true)

            # 保存预测值和真实值
            y_pred = y_pred.cpu().detach().numpy().tolist()
            y_true = y_true.cpu().detach().numpy().tolist()

            # 处理 UniProt IDs（可选，根据实际需求调整）
            if len(data) > 4:  # 如果 data 中包含 uniprot_id，假设它是第5个元素
                sequence = data[4]  # 获取 uniprot_id
                valid_uniprot_ids.extend(sequence)
            else:
                # 如果没有 uniprot_id，尝试从 input_ids 提取
                valid_uniprot_ids.extend(input_ids.cpu().detach().numpy().tolist())

            valid_pred.extend(y_pred)
            valid_true.extend(y_true)

            epoch_loss += loss.item()
            n_batches += 1

    # 修正长度不一致的问题
    if len(valid_true) != len(valid_pred):
        print(f"Warning: 数据长度不一致：valid_true长度为{len(valid_true)}，valid_pred长度为{len(valid_pred)}")
        min_length = min(len(valid_true), len(valid_pred))
        valid_true = valid_true[:min_length]
        valid_pred = valid_pred[:min_length]
        
    # 绘制散点图
    if len(valid_true) == len(valid_pred):
        plot_scatter_with_stats(valid_true, valid_pred, fold)
    else:
        print(f"数据长度不一致：valid_true长度为{len(valid_true)}，valid_pred长度为{len(valid_pred)}")
    
    epoch_loss_avg = epoch_loss / n_batches

    # 返回损失和预测结果
    return epoch_loss_avg, valid_true, valid_pred, valid_uniprot_ids
def train_one_epoch(model, train_loader, epoch):
    model.train()
    
    epoch_loss = 0.0
    n_batches = 0
    
    for data in train_loader:
        input_ids, attention_mask, features, labels = data

        if torch.cuda.is_available():
            input_ids = input_ids.cuda()
            attention_mask = attention_mask.cuda()
            features = features.cuda()
            labels = labels.cuda()

        # 将 features（或者其他特征）传递给模型
        y_pred = model(input_ids, attention_mask, features)  # 这里传递了 features
        y_pred = torch.squeeze(y_pred)
        y_true = labels.float() / 120.0  # 如果需要归一化标签

        loss = model.criterion(y_pred, y_true)

        epoch_loss += loss.item()
        n_batches += 1

    epoch_loss_avg = epoch_loss / n_batches
    return epoch_loss_avg
def plot_scatter_with_stats(y_true, y_pred, fold):
    # 创建图形
    plt.figure(figsize=(8, 6))

    # 绘制散点图
    plt.scatter(y_true, y_pred, color='blue', alpha=0.6, label='Predicted vs True')

    # 计算Pearson相关系数
    pearson_corr, _ = pearsonr(y_true, y_pred)

    # 计算R²值
    r2 = r2_score(y_true, y_pred)

    # 绘制对角线 y = x
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label="y = x (Perfect Prediction)")

    # 添加标签和标题
    plt.xlabel("Topt Experimental Values (True)", fontsize=12)
    plt.ylabel("Topt Predicted Values", fontsize=12)
    plt.title(f"Topt Prediction vs Experimental Values - Fold {fold}", fontsize=14)

    # 显示网格和图例
    plt.grid(True)
    plt.legend()

    # 在图形中添加Pearson相关系数和R²值
    plt.text(min_val + 0.05 * (max_val - min_val), max_val - 0.05 * (max_val - min_val),
             f'Pearson Corr: {pearson_corr:.2f}\nR²: {r2:.2f}',
             fontsize=12, color='red', bbox=dict(facecolor='white', alpha=0.7))

    # 保存图像
    plt.savefig(f'../Result/Fold{fold}_scatter_plot_with_stats.png')  # 选择保存的路径
    plt.close()  # 关闭当前图形，防止图形堆积
def train(model, train_dataframe, valid_dataframe, fold, fold_results):
    # 获取训练数据的真实标签（即topt值）
    train_true = train_dataframe['topt'].values  # 真实标签

    # 训练过程中的批次处理
    train_pred = []  # 用于存储预测值

    for inputs, targets in train_loader:
        predictions = model(inputs)  # 进行模型预测
        train_pred.extend(predictions.cpu().detach().numpy())  # 预测值

    # 检查是否包含NaN或Inf值
    if train_dataframe.select_dtypes(include=[np.number]).isnull().any().any():
        print("NaN found in train_dataframe")
    if np.isinf(train_dataframe.select_dtypes(include=[np.number]).values).any():
        print("Inf found in train_dataframe")

    # 检查train_pred是否包含NaN或Inf
    if np.isnan(train_pred).any():
        print("NaN found in train_pred")
    if np.isinf(train_pred).any():
        print("Inf found in train_pred")
    
    # 计算MSE等评估指标
    train_mse_value = mean_squared_error(train_true, train_pred)  # 使用train_true和train_pred
    print("Train MSE:", train_mse_value)

def collate_fn(batch):
    input_ids = torch.stack([item[0] for item in batch])
    attention_mask = torch.stack([item[1] for item in batch])
    features = torch.stack([item[2] for item in batch])
    topt = torch.stack([item[3] for item in batch])  # 修改：原 item[4]，删除 ec_features

    return input_ids, attention_mask, features, topt  # 修改：删除 ec_features


# 创建 train_loader 和 valid_loader
train_loader = DataLoader(
    dataset=ProteinDataset(csv_file=csv_file_train, tokenizer=tokenizer, Feature_Path=Feature_Path),
    batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True, collate_fn=collate_fn  # 修改：删除 ec_data
)

valid_loader = DataLoader(
    dataset=ProteinDataset(csv_file=csv_file_valid, tokenizer=tokenizer, Feature_Path=Feature_Path),
    batch_size=BATCH_SIZE, shuffle=False, num_workers=0, drop_last=False, collate_fn=collate_fn  # 修改：删除 ec_data
)

# 初始化训练和验证损失、Pearson、R2等列表
train_losses = []
train_pearson = []
train_r2 = []
train_rmse = []
train_mse = []

valid_losses = []
valid_pearson = []
valid_r2 = []
valid_rmse = []
valid_mse = []

best_val_loss = 1000
best_epoch = 0

for epoch in range(NUMBER_EPOCHS):
    print("\n========== Train epoch " + str(epoch + 1) + " ==========")
    model.train()

    # 训练一个epoch
    epoch_loss_train_avg = train_one_epoch(model, train_loader, epoch + 1)
    print("========== Evaluate Train set ==========")
    _, train_true, train_pred, train_uniprot_ids = evaluate(model, train_loader, fold)  # 获取训练结果

    # 计算 RMSE 和 MSE
    train_mse_value = mean_squared_error(train_true, train_pred)
    train_rmse_value = np.sqrt(train_mse_value)
    result_train = analysis(train_true, train_pred)
    print("Train loss: ", np.sqrt(epoch_loss_train_avg))
    print("Train pearson:", result_train['pearson'])
    print("Train r2:", result_train['r2'])
    print("Train RMSE:", train_rmse_value)
    print("Train MSE:", train_mse_value)

    train_losses.append(np.sqrt(epoch_loss_train_avg))
    train_pearson.append(result_train['pearson'])
    train_r2.append(result_train['r2'])
    train_rmse.append(train_rmse_value)
    train_mse.append(train_mse_value)
    
    print("========== Evaluate Valid set ==========")
    epoch_loss_valid_avg, valid_true, valid_pred, valid_uniprot_ids = evaluate(model, valid_loader, fold)  # 获取验证结果
    
    # 计算 RMSE 和 MSE
    valid_mse_value = mean_squared_error(valid_true, valid_pred)
    valid_rmse_value = np.sqrt(valid_mse_value)
    
    result_valid = analysis(valid_true, valid_pred)
    print("Valid loss: ", np.sqrt(epoch_loss_valid_avg))
    print("Valid pearson:", result_valid['pearson'])
    print("Valid r2:", result_valid['r2'])
    print("Valid RMSE:", valid_rmse_value)
    print("Valid MSE:", valid_mse_value)
    
    # 确保长度一致
    if len(valid_true) == len(valid_pred) == len(valid_uniprot_ids):
        valid_detail_dataframe = pd.DataFrame({'uniprot_id': valid_uniprot_ids, 'y_true': valid_true, 'y_pred': valid_pred})
        valid_detail_dataframe.to_csv(Result_Path + 'Fold' + str(fold) + "_valid_detail.csv", header=True, sep=',')
    else:
        print("Error: Lists have different lengths!")
        
    valid_losses.append(np.sqrt(epoch_loss_valid_avg))
    valid_pearson.append(result_valid['pearson'])
    valid_r2.append(result_valid['r2'])
    valid_rmse.append(valid_rmse_value)
    valid_mse.append(valid_mse_value)

    # 如果当前验证损失较好，保存模型
    if best_val_loss > epoch_loss_valid_avg:
        best_val_loss = epoch_loss_valid_avg
        best_epoch = epoch + 1
        os.makedirs(Model_Path, exist_ok=True)  # 确保模型路径存在
        torch.save(model, os.path.join(Model_Path, 'Fold' + str(fold) + '_best_model.pth'))

        # 保存验证结果详细信息
        valid_detail_dataframe = pd.DataFrame({'uniprot_id': valid_uniprot_ids, 'y_true': valid_true, 'y_pred': valid_pred})
        valid_detail_dataframe.to_csv(Result_Path + 'Fold' + str(fold) + "_valid_detail.csv", header=True, sep=',')
        
        # 保存训练结果详细信息
        train_detail_dataframe = pd.DataFrame({'uniprot_id': train_uniprot_ids, 'y_true': train_true, 'y_pred': train_pred})
        train_detail_dataframe.to_csv(Result_Path + 'Fold' + str(fold) + "_train_detail.csv", header=True, sep=',')

# 绘制训练和验证的 RMSE, R2, 和 Pearson 曲线
#plot_metrics(train_rmse, valid_rmse, "RMSE", "RMSE", fold)
#plot_metrics(train_r2, valid_r2, "R2", "R² Score", fold)
#plot_metrics(train_pearson, valid_pearson, "Pearson", "Pearson Correlation", fold)

# 保存训练过程中的所有结果
result_all = {
    'Train_loss': train_losses,
    'Train_pearson': train_pearson,
    'Train_r2': train_r2,
    'Train_rmse': train_rmse,
    'Train_mse': train_mse,
    
    'Valid_loss': valid_losses,
    'Valid_pearson': valid_pearson,
    'Valid_r2': valid_r2,
    'Valid_rmse': valid_rmse,
    'Valid_mse': valid_mse,
    
    'Best_epoch': [best_epoch for _ in range(len(train_losses))],
}
result = pd.DataFrame(result_all)
print("Fold", str(fold), "Best epoch at", str(best_epoch))
result.to_csv(Result_Path + "Fold" + str(fold) + "_result.csv", sep=',')
# 将验证集的预测结果存储在 fold_results 中以便后续使用
fold_results['valid_predictions'] = valid_pred  # 将验证集的预测值赋值到 fold_results 中

def cross_validation(all_dataframe, fold_number=10):
    print("split_seed: ", SEED)
    sequence_names = all_dataframe['uniprot_id'].values
    sequence_labels = all_dataframe['topt'].values
    kfold = KFold(n_splits=fold_number, shuffle=True)
    fold = 0
    
    all_train_losses = []
    all_train_pearson = []
    all_train_r2 = []
    all_train_rmse = []
    all_train_mse = []
    
    all_valid_losses = []
    all_valid_pearson = []
    all_valid_r2 = []
    all_valid_rmse = []
    all_valid_mse = []
    
    for train_index, valid_index in kfold.split(sequence_names, sequence_labels):
        print("\n========== Fold " + str(fold + 1) + " ==========")
        train_dataframe = all_dataframe.iloc[train_index, :]
        valid_dataframe = all_dataframe.iloc[valid_index, :]
        print("Training on", str(train_dataframe.shape[0]), "examples, Validation on", str(valid_dataframe.shape[0]), "examples")

        model = ESM2ToptPredictor()
        model.to(device)

        
        
        # Create fold_results to store results for this fold
        fold_results = {
            'train_losses': [],
            'train_pearson': [],
            'train_r2': [],
            'train_rmse': [],
            'train_mse': [],
            'valid_losses': [],
            'valid_pearson': [],
            'valid_r2': [],
            'valid_rmse': [],
            'valid_mse': [],
            'valid_predictions': []
        }

        train(model, train_dataframe, valid_dataframe, fold + 1, fold_results)
        fold += 1
        
        # Get the predictions for this fold
        y_true = valid_dataframe['topt'].values
        y_pred = fold_results['valid_predictions']  # Assuming fold_results['valid_predictions'] holds predictions

        # 绘制残差图
        if len(y_true) == len(y_pred):
            plot_residuals(y_true, y_pred, fold)
        else:
            print(f"Error: Mismatch in lengths for fold {fold} - y_true: {len(y_true)}, y_pred: {len(y_pred)}")
        
        # 绘制误差直方图
        if len(y_true) == len(y_pred):
            plot_error_histogram(y_true, y_pred, fold)
        else:
            print(f"Error: Mismatch in lengths for fold {fold} - y_true: {len(y_true)}, y_pred: {len(y_pred)}")
        
        # 绘制KDE图
        if len(y_true) == len(y_pred):
            plot_kde(y_true, y_pred, fold)
        else:
            print(f"Error: Mismatch in lengths for fold {fold} - y_true: {len(y_true)}, y_pred: {len(y_pred)}")
        
        
        # 汇总每个fold的训练和验证结果
        all_train_losses.extend(fold_results['train_losses'])
        all_train_pearson.extend(fold_results['train_pearson'])
        all_train_r2.extend(fold_results['train_r2'])
        all_train_rmse.extend(fold_results['train_rmse'])
        all_train_mse.extend(fold_results['train_mse'])
        
        all_valid_losses.extend(fold_results['valid_losses'])
        all_valid_pearson.extend(fold_results['valid_pearson'])
        all_valid_r2.extend(fold_results['valid_r2'])
        all_valid_rmse.extend(fold_results['valid_rmse'])
        all_valid_mse.extend(fold_results['valid_mse'])

    # 绘制所有fold的训练和验证结果
    plot_metrics(all_train_rmse, all_valid_rmse, "RMSE", "RMSE", "All")
    plot_metrics(all_train_r2, all_valid_r2, "R2", "R² Score", "All")
    plot_metrics(all_train_pearson, all_valid_pearson, "Pearson", "Pearson Correlation", "All")

    # 汇总所有fold的结果并保存
    result_all = {
        'Train_loss': all_train_losses,
        'Train_pearson': all_train_pearson,
        'Train_r2': all_train_r2,
        'Train_rmse': all_train_rmse,
        'Train_mse': all_train_mse,
        
        'Valid_loss': all_valid_losses,
        'Valid_pearson': all_valid_pearson,
        'Valid_r2': all_valid_r2,
        'Valid_rmse': all_valid_rmse,
        'Valid_mse': all_valid_mse,
    }
    result = pd.DataFrame(result_all)
    result.to_csv(Result_Path + "cross_validation_result.csv", sep=',')
    # 训练历史记录
history = {
    "mse": [],
    "rmse": [],
    "r2": [],
    "pearson": []
}

# **绘制训练过程中损失函数与评估指标变化**
def plot_metrics(history):
    epochs = range(1, len(history["mse"]) + 1)
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["mse"], label="MSE")
    plt.plot(epochs, history["rmse"], label="RMSE")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss Metrics")
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["r2"], label="R² Score")
    plt.plot(epochs, history["pearson"], label="Pearson Correlation")
    plt.xlabel("Epochs")
    plt.ylabel("Score")
    plt.title("Evaluation Metrics")
    plt.legend()
    
    plt.show()

# **绘制 KDE 核密度分布**
def plot_kde(y_true, y_pred):
    plt.figure(figsize=(8, 6))
    sns.kdeplot(y_true, label="True Values", fill=True)
    sns.kdeplot(y_pred, label="Predicted Values", fill=True)
    plt.xlabel("Topt Value")
    plt.ylabel("Density")
    plt.title("KDE Distribution of True vs Predicted Values")
    plt.legend()
    plt.show()

# **绘制误差直方图**
def plot_error_histogram(y_true, y_pred):
    errors = y_true - y_pred
    plt.figure(figsize=(8, 6))
    plt.hist(errors, bins=30, alpha=0.7, color="b", edgecolor="k")
    plt.axvline(x=0, color="red", linestyle="--", label="Zero Error Line")
    plt.xlabel("Prediction Error (True - Predicted)")
    plt.ylabel("Frequency")
    plt.title("Error Histogram")
    plt.legend()
    plt.show()

# **绘制残差图**
def plot_residuals(y_true, y_pred):
    residuals = y_true - y_pred
    plt.figure(figsize=(8, 6))
    plt.scatter(y_pred, residuals, alpha=0.5, color="b", edgecolors="k")
    plt.axhline(y=0, color="red", linestyle="--", label="Zero Residual Line")
    plt.xlabel("Predicted Topt Value")
    plt.ylabel("Residuals")
    plt.title("Residuals Plot")
    plt.legend()
    plt.show()

# **分析模型结果**
def analysis(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    pearson_corr, _ = pearsonr(y_true, y_pred)

    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R² Score: {r2:.4f}")
    print(f"Pearson Correlation: {pearson_corr:.4f}")

    # 可视化
    plot_kde(y_true, y_pred)
    plot_error_histogram(y_true, y_pred)
    plot_residuals(y_true, y_pred)

# **主函数**
def main():
    # 读取训练集数据
    train_data = pd.read_csv(csv_file_train)
    valid_data = pd.read_csv(csv_file_valid)

    # 提取特征和标签
    X_train = train_data.drop(columns=["Topt"]).values
    y_train = train_data["Topt"].values

    X_valid = valid_data.drop(columns=["Topt"]).values
    y_valid = valid_data["Topt"].values

    # 定义简单的MLP模型
    class MLP(nn.Module):
        def __init__(self, input_dim):
            super(MLP, self).__init__()
            self.fc1 = nn.Linear(input_dim, 128)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(128, 64)
            self.fc3 = nn.Linear(64, 1)

        def forward(self, x):
            x = self.relu(self.fc1(x))
            x = self.relu(self.fc2(x))
            return self.fc3(x)

    model = MLP(input_dim=X_train.shape[1]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.MSELoss()

    # 训练循环
    for epoch in range(NUMBER_EPOCHS):
        model.train()
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)

        optimizer.zero_grad()
        y_pred_train = model(X_train_tensor).squeeze()
        loss = criterion(y_pred_train, y_train_tensor)
        loss.backward()
        optimizer.step()

        # 计算评估指标
        model.eval()
        X_valid_tensor = torch.tensor(X_valid, dtype=torch.float32).to(device)
        y_valid_tensor = torch.tensor(y_valid, dtype=torch.float32).to(device)

        y_pred_valid = model(X_valid_tensor).squeeze().detach().cpu().numpy()
        mse = mean_squared_error(y_valid, y_pred_valid)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_valid, y_pred_valid)
        pearson_corr, _ = pearsonr(y_valid, y_pred_valid)

        history["mse"].append(mse)
        history["rmse"].append(rmse)
        history["r2"].append(r2)
        history["pearson"].append(pearson_corr)

        print(f"Epoch {epoch + 1}/{NUMBER_EPOCHS} - MSE: {mse:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}, Pearson: {pearson_corr:.4f}")

    # 训练完成后进行评估
    analysis(y_valid, y_pred_valid)
    plot_metrics(history)

if __name__ == "__main__":
    main()
import os
import math
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import sys
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn import metrics
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
from sklearn.model_selection import KFold, train_test_split
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error
np.set_printoptions(threshold=sys.maxsize)

# path
Dataset_Path = '../Data/'
Model_Path = '../Model/'
Result_Path = '../Result/'
TrainData_Path = '../Data/train_data_all.csv'
TestData_Path = '../Data/test_data.csv'
Feature_Path = '../Data/features/'

# Hyperparameters
SEED = 2333
NUMBER_EPOCHS = 50
LEARNING_RATE = 1E-6
WEIGHT_DECAY = 1E-3
BATCH_SIZE = 4
NUM_CLASSES = 1
LENG_SIZE = 1028
DENSE_DIM = 16
ATTENTION_HEADS = 4
ESM2_MODEL_NAME = 'facebook/esm2_t6_8M_UR50D'

# Set random seeds
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

# Load ESM2 model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(ESM2_MODEL_NAME)
esm2_model = AutoModel.from_pretrained(ESM2_MODEL_NAME)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
esm2_model.to(device)

amino_acid = list("ACDEFGHIKLMNPQRSTVWY")
amino_dict = {aa: i for i, aa in enumerate(amino_acid)}
aalist = list('ACDEFGHIKLMNPQRSTVWY')
amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
polar_aa = 'AVLIFWMP'
nonpolar_aa = 'GSTCYNQHKRDE'


def load_features(uniprot_id, mean, std, df):
    # len(sequence) * 94
    feature_matrix1 = np.load(Dataset_Path + 'features/' + uniprot_id + '.npy')
    
    # 直接从 df 获取对应的 topt 值
    topt_value = df.loc[df['uniprot_id'] == uniprot_id, 'topt'].values[0]
    
    feature_matrix2 = np.ones(feature_matrix1.shape[0]) * topt_value
    feature_matrix = np.concatenate((feature_matrix1, feature_matrix2.reshape(-1, 1)), axis=1)
    feature_matrix = (feature_matrix - mean) / std
    part1 = feature_matrix[:, 0:20]
    part2 = feature_matrix[:, 23:]

    # len(sequence) * 91
    feature_matrix = np.concatenate([part1, part2], axis=1)
    return feature_matrix



def load_values():
    # (94,)
    mean1 = np.load(Dataset_Path + 'features/' + 'oneD_mean.npy')
    std1 = np.load(Dataset_Path + 'features/' + 'oneD_std.npy')
    # (1,)
    df = pd.read_csv(TrainData_Path,sep=',')
    topt = df['topt'].values
    mean2 = []
    mean2.append(np.mean(topt))
    std2 = []
    std2.append(np.std(topt))
    mean2 = np.array(mean2)
    std2 = np.array(std2)
    mean = np.concatenate([mean1, mean2])
    std = np.concatenate([std1, std2])
    return mean, std


class ProteinDataset(Dataset):
    def __init__(self, csv_file, tokenizer, Feature_Path, ec_data=None):
        self.data = pd.read_csv(csv_file)
        self.tokenizer = tokenizer
        self.feature_path = Feature_Path
        self.ec_data = ec_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        sequence = row['sequence']
        topt = row['topt']
        ec_features = self.get_ec_features(row['ec_number'])  # EC 编码
        features = self.load_features(row['uniprot_id'])  # 加载额外特征

        # 使用 ESM2 的 tokenizer 对序列进行处理
        inputs = self.tokenizer(sequence, return_tensors="pt", padding="max_length", truncation=True, max_length=512)

        # 从 tokenizer 获取 input_ids 和 attention_mask
        input_ids = inputs['input_ids'].squeeze().to(device)
        attention_mask = inputs['attention_mask'].squeeze().to(device)

        # 填充特征矩阵，如果大小不一致
        max_feature_len = 500  # 假设最大特征矩阵的长度为300，可以根据你的实际数据调整
        if features.shape[0] < max_feature_len:
            padding = torch.zeros(max_feature_len - features.shape[0], features.shape[1])
            features = torch.cat([features, padding], dim=0)
        elif features.shape[0] > max_feature_len:
            features = features[:max_feature_len, :]  # 截断

        return input_ids, attention_mask, features, ec_features, torch.tensor(topt, dtype=torch.float32)


    def get_ec_features(self, ec_number):
        """
        假设 ec_number 是一个字符串（例如 "1.1.1.1"),
        创建一个字典，将 EC 号映射到索引并生成 One-hot 编码。
        """
        if self.ec_data is not None:
            # 如果有 EC 数据，进行编码
            unique_ec_numbers = self.ec_data['ec_number'].unique()
            ec_dict = {ec: idx for idx, ec in enumerate(unique_ec_numbers)}
            
            one_hot_vector = np.zeros(len(ec_dict))
            if ec_number in ec_dict:
                one_hot_vector[ec_dict[ec_number]] = 1  # 设置对应位置为 1
            return torch.tensor(one_hot_vector, dtype=torch.float32)
        else:
            # 如果没有提供 EC 数据，返回空的向量或默认值
            return torch.zeros(1, dtype=torch.float32)

    def load_features(self, uniprot_id):
        """
        根据 uniprot_id 加载额外的特征（如 BLOSUM, pp7等),
        返回的应该是一个张量。
        """
        feature_file = os.path.join(self.feature_path, f"{uniprot_id}.npy")
        if os.path.exists(feature_file):
            feature_matrix = np.load(feature_file)
            return torch.tensor(feature_matrix, dtype=torch.float32)
        else:
            # 如果没有特征文件，返回一个默认的零向量（这里可以根据需求调整）
            return torch.zeros(1, dtype=torch.float32)


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
        # 输入是 ESM2 输出、EC 特征和其他特征的拼接，输出是128维
        self.fc = nn.Linear(217481, 128)  # 调整为 n_heads * input_dim + 其他特征维度
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

    def forward(self, input_ids, attention_mask, features, ec_features):
        # 获取 ESM2 的输出
        esm_output = self.esm2(input_ids=input_ids, attention_mask=attention_mask)
        esm_embedding = esm_output.last_hidden_state[:, 0, :]  # 获取 [CLS] token 的输出作为序列特征
        
        # 打印 ESM2 输出的形状
        #print("ESM2 embedding shape:", esm_embedding.shape)
        
        # 使用注意力模块处理 ESM2 输出
        att = self.attention(esm_embedding.unsqueeze(0))  # 增加 batch 维度
        #print("Attention output shape:", att.shape)
        
        # 计算注意力输出
        esm_embedding_avg = torch.sum(att @ esm_embedding.unsqueeze(0), dim=1) / self.attention.n_heads
        
        # 计算平均后的 ESM2 输出
        esm_embedding_avg = esm_embedding_avg.squeeze(1)  # [batch_size, hidden_dim]
        #print("Esm embedding avg shape after squeeze:", esm_embedding_avg.shape)
        
        # 确保 features 和 ec_features 是二维张量
        if features.dim() == 3:
            features = features.view(features.size(0), -1)  # 展平为 [batch_size, seq_len * feature_dim]
        if ec_features.dim() == 3:
            ec_features = ec_features.view(ec_features.size(0), -1)  # 展平为 [batch_size, seq_len * feature_dim]
        
        # 调整 esm_embedding_avg 的 batch size 使其和其他特征一致
        esm_embedding_avg = esm_embedding_avg.expand(features.size(0), -1)  # 将 esm_embedding_avg 扩展到 batch_size
        
        # 拼接 ESM2 输出、EC 特征和其他提取的特征
        combined_features = torch.cat((esm_embedding_avg, features, ec_features), dim=1)
        #print("Combined features shape:", combined_features.shape)
        
        # 通过全连接层处理拼接后的特征
        dense_output = F.relu(self.fc(combined_features))
        dense_output = self.dropout(dense_output)
        # 最终输出层
        output = self.output_layer(dense_output)
        
        # 输出预测值（去除多余维度）
        return output.squeeze(1)


def train_one_epoch(model, data_loader, epoch):
    epoch_loss_train = 0.0
    n_batches = 0
    
    # 加入进度条
    for data in tqdm(data_loader, desc=f"Training Epoch {epoch}", ncols=100):
        optimizer = model.module.optimizer if isinstance(model, nn.DataParallel) else model.optimizer
        optimizer.zero_grad()

        input_ids, attention_mask, features, ec_features, labels = data

        # 处理输入
        if torch.cuda.is_available():
            input_ids = input_ids.cuda()
            attention_mask = attention_mask.cuda()
            features = features.cuda()
            ec_features = ec_features.cuda()
            labels = labels.cuda()

        # 前向传播
        y_pred = model(input_ids, attention_mask, features, ec_features)  
        y_pred = torch.squeeze(y_pred)

        # 计算损失
        y_true = labels.float() / 120.0
        loss = model.criterion(y_pred, y_true)

        loss.backward()
        optimizer.step()

        epoch_loss_train += loss.item()
        n_batches += 1

    epoch_loss_train_avg = epoch_loss_train / n_batches
    return epoch_loss_train_avg


def evaluate(model, data_loader):
    model.eval()

    epoch_loss = 0.0
    n_batches = 0
    valid_pred = []
    valid_true = []
    valid_uniprot_ids = []

    # 加入进度条
    for data in tqdm(data_loader, desc="Evaluating", ncols=100):
        with torch.no_grad():
            input_ids, attention_mask, features, ec_features, labels = data  # 假设 sequence 是数据的一部分

            if torch.cuda.is_available():
                input_ids = input_ids.cuda()
                attention_mask = attention_mask.cuda()
                features = features.cuda()
                ec_features = ec_features.cuda()
                labels = labels.cuda()

            y_pred = model(input_ids, attention_mask, features, ec_features)
            y_pred = torch.squeeze(y_pred)
            y_true = labels.float() / 120.0  # 如果需要归一化标签
            

            loss = model.criterion(y_pred, y_true)

            y_pred = y_pred.cpu().detach().numpy().tolist()
            y_true = y_true.cpu().detach().numpy().tolist()
            # 确保 valid_uniprot_ids 被填充
            if len(data) > 5:  # 如果 data 中包含 uniprot_id，假设它是第6个元素
                sequence = data[5]  # 获取 uniprot_id
                valid_uniprot_ids.extend(sequence)
            else:
                # 如果没有 uniprot_id，尝试从 input_ids 提取
                valid_uniprot_ids.extend(input_ids.cpu().detach().numpy().tolist())

            valid_pred.extend(y_pred)
            valid_true.extend(y_true)

            epoch_loss += loss.item()
            n_batches += 1

    epoch_loss_avg = epoch_loss / n_batches
    # 调试：打印所有返回的列表长度
    #print(f"Valid uniprot ids length: {len(valid_uniprot_ids)}")
    #print(f"Valid true length: {len(valid_true)}")
    #print(f"Valid pred length: {len(valid_pred)}")
    return epoch_loss_avg, valid_true, valid_pred, valid_uniprot_ids



def train(model, train_dataframe, valid_dataframe, fold=0):
    # 定义csv_file、tokenizer和Feature_Path（假设这些是全局变量或在函数外定义的）
    csv_file_train = TrainData_Path  # 训练数据的CSV文件路径
    csv_file_valid = TestData_Path  # 验证数据的CSV文件路径
    tokenizer = AutoTokenizer.from_pretrained(ESM2_MODEL_NAME)  # 从 Hugging Face 加载的 tokenizer
    Feature_Path = '../Data/features/'  # 特征文件路径

    # 如果有 EC 数据，传递它；如果没有，可以将其设置为 None
    ec_data = None  # 如果有EC数据，可以传递实际的ec_data

    def collate_fn(batch):
        input_ids = torch.stack([item[0] for item in batch])
        attention_mask = torch.stack([item[1] for item in batch])
        features = torch.stack([item[2] for item in batch])
        ec_features = torch.stack([item[3] for item in batch])
        topt = torch.stack([item[4] for item in batch])

        return input_ids, attention_mask, features, ec_features, topt
    # 创建 train_loader 和 valid_loader
    train_loader = DataLoader(
        dataset=ProteinDataset(csv_file=csv_file_train, tokenizer=tokenizer, Feature_Path=Feature_Path, ec_data=ec_data),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True, collate_fn=collate_fn
    )

    valid_loader = DataLoader(
        dataset=ProteinDataset(csv_file=csv_file_valid, tokenizer=tokenizer, Feature_Path=Feature_Path, ec_data=ec_data),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0, drop_last=True, collate_fn=collate_fn
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
        _, train_true, train_pred, train_uniprot_ids = evaluate(model, train_loader)  # 获取训练结果
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
        epoch_loss_valid_avg, valid_true, valid_pred, valid_uniprot_ids = evaluate(model, valid_loader)  # 获取验证结果
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
            torch.save(model.state_dict(), os.path.join(Model_Path, 'Fold' + str(fold) + '_best_model.pkl'))

            # 保存验证结果详细信息
            valid_detail_dataframe = pd.DataFrame({'uniprot_id': valid_uniprot_ids, 'y_true': valid_true, 'y_pred': valid_pred})
            valid_detail_dataframe.to_csv(Result_Path + 'Fold' + str(fold) + "_valid_detail.csv", header=True, sep=',')
            
            # 保存训练结果详细信息
            train_detail_dataframe = pd.DataFrame({'uniprot_id': train_uniprot_ids, 'y_true': train_true, 'y_pred': train_pred})
            train_detail_dataframe.to_csv(Result_Path + 'Fold' + str(fold) + "_train_detail.csv", header=True, sep=',')

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



def analysis(y_true, y_pred):
    # Pearson correlation coefficient
    pearson = pearsonr(y_true, y_pred)[0]  # Get only the correlation coefficient value
    # R2 score
    r2 = metrics.r2_score(y_true, y_pred)

    result = {
        'pearson': pearson,
        'r2': r2,
    }
    return result


def cross_validation(all_dataframe, fold_number=10):
    print("split_seed: ", SEED)
    sequence_names = all_dataframe['uniprot_id'].values
    sequence_labels = all_dataframe['topt'].values
    kfold = KFold(n_splits=fold_number, shuffle=True)
    fold = 0

    for train_index, valid_index in kfold.split(sequence_names, sequence_labels):
        print("\n========== Fold " + str(fold + 1) + " ==========")
        train_dataframe = all_dataframe.iloc[train_index, :]
        valid_dataframe = all_dataframe.iloc[valid_index, :]
        print("Training on", str(train_dataframe.shape[0]), "examples, Validation on", str(valid_dataframe.shape[0]), "examples")

        model = ESM2ToptPredictor()
        model.to(device)

        train(model, train_dataframe, valid_dataframe, fold + 1)
        fold += 1


if __name__ == "__main__":
    # Ensure your dataset CSV file contains a `sequence` column with protein sequences
    train_dataframe = pd.read_csv(Dataset_Path + 'train_data_all.csv', sep=',')
    cross_validation(train_dataframe, fold_number=5)




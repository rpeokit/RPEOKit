import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import Counter

# 路径定义
Dataset_Path = '../Data/'
Fasta_Path = '../Data/fasta/'
Pccp_Path = '../Data/'
Feature_Path = '../Data/tnpb.csv' 

aalist = list('ACDEFGHIKLMNPQRSTVWY')
amino_acids = 'ACDEFGHIKLMNPQRSTVWY'

# 读取PCCP数据
with open(Pccp_Path + 'pp7.csv', 'r') as f:
    pccp = f.read().splitlines()
    pccp = [i.split() for i in pccp]
    pccp_dic = {i[0]: np.array(i[1:]).astype(float) for i in pccp}

# 读取特征数据
features_df = pd.read_csv(Feature_Path)

def read_pccp(seq):
    valid_seq = [i for i in seq if i in pccp_dic]
    return np.array([pccp_dic.get(i, np.zeros(7)) for i in seq])

def read_fasta(fname):
    sequences = {}
    with open(fname, 'r') as f:
        content = f.read().strip().split('>')[1:]  # Split by '>' and skip the first empty part
        for entry in content:
            lines = entry.splitlines()
            uniprot_id = lines[0].strip()  # First line is the protein ID
            sequence = ''.join(lines[1:]).strip()  # Join the rest as the sequence
            sequences[uniprot_id] = sequence
    return sequences

def get_AAfq(seq):
    AAfq_dic = dict()
    AAfq = np.array([seq.count(x) for x in aalist])/len(seq)
    for (key,value) in zip(aalist, AAfq):
         AAfq_dic[key] = value
    AAfq_dic['X'] = 0.0  # 假设未知氨基酸的频率为0
    seq_AAfq = np.array([AAfq_dic.get(x, 0.0) for x in seq])  # 使用get避免KeyError
    return seq_AAfq

def do_count(seq):
    result = {}
    for i in range(len(seq) - 1):
        dimer = seq[i] + seq[i + 1]
        if dimer in result:
            result[dimer] += 1
        else:
            result[dimer] = 1
    return result

def get_dipfq(seq):
    result = do_count(seq)
    dimers = sum(result.values())
    dimers_fq = {}
    
    for a1 in amino_acids:
        for a2 in amino_acids:
            dimers_fq[a1 + a2] = (result.get(a1 + a2, 0) * 1.0) / dimers
    
    for a1 in amino_acids:
        dimers_fq[a1 + 'X'] = dimers_fq.get(a1 + 'X', 0)
        dimers_fq['X' + a1] = dimers_fq.get('X' + a1, 0)

    seq_dipfq = np.zeros((len(seq), len(amino_acids) * len(amino_acids)))  # [序列长度, 40]
    
    for i in range(len(seq) - 1):  
        a1 = seq[i]
        a2 = seq[i + 1]
        dip_key = a1 + a2
        dipfq_values = dimers_fq.get(dip_key, np.zeros(len(amino_acids) * len(amino_acids)))
        seq_dipfq[i] = dipfq_values
    
    seq_dipfq[-1] = np.zeros(len(amino_acids) * len(amino_acids))

    return seq_dipfq

def load_blosum():
    with open(Dataset_Path + 'BLOSUM62.csv', 'r') as f:
        result = {}
        next(f)
        lines = f.readlines()
        for line in lines:
            line = line.strip().split()
            result[line[0]] = [int(i) for i in line[1:]]
    return result

def get_blosum(sequence):
    blosum_dic = load_blosum()
    return np.array([blosum_dic[i] for i in sequence])

def min_max_normalization(matrix):
    """
    对特征矩阵进行最小-最大归一化，使其值范围在 [0, 1] 之间。
    """
    min_vals = np.min(matrix, axis=0)
    max_vals = np.max(matrix, axis=0)
    range_vals = max_vals - min_vals
    range_vals = np.where(range_vals == 0, 1, range_vals)  # 防止除以零
    return (matrix - min_vals) / range_vals

def get_matrix():
    fasta_sequences = read_fasta(Fasta_Path + 'tnpb.fasta')  # 读取FASTA文件
    for uniprot_id, sequence in tqdm(fasta_sequences.items()):
        # 提取特征
        blosum = get_blosum(sequence)  # L * 23
        AAfq = get_AAfq(sequence)  # L * 1
        dipfq = get_dipfq(sequence)  # L * 40
        PP7 = read_pccp(sequence)  # L * 7
        
        # 合并特征矩阵
        matrix = np.concatenate([blosum, PP7, np.array(AAfq).reshape(-1, 1), np.array(dipfq)], axis=1)

        # 使用最小-最大归一化
        matrix = min_max_normalization(matrix)

        # 保存标准化后的特征矩阵
        print(f"Saving features for {uniprot_id} to ../Data/features/tnpb/{uniprot_id}.npy")
        np.save('../Data/features/tnpb/' + uniprot_id + '.npy', matrix)

def cal_mean_std():
    total_length = 0
    oneD_mean = None
    oneD_mean_square = None
    for name in tqdm(features_df['uniprot_id']):
        matrix = np.load('../Data/features/tnpb/' + name + '.npy')

        if oneD_mean is None:  # 第一次读取矩阵时初始化
            oneD_mean = np.zeros(matrix.shape[1])
            oneD_mean_square = np.zeros(matrix.shape[1])

        total_length += matrix.shape[0]
        oneD_mean += np.sum(matrix, axis=0)
        oneD_mean_square += np.sum(np.square(matrix), axis=0)

    oneD_mean /= total_length  # E(X)
    oneD_mean_square /= total_length  # E(X^2)
    oneD_std = np.sqrt(np.subtract(oneD_mean_square, np.square(oneD_mean)))  # sqrt(DX)

    np.save('../Data/features/tnpb/oneD_mean.npy', oneD_mean)
    np.save('../Data/features/tnpb/oneD_std.npy', oneD_std)

if __name__ == "__main__":
    get_matrix()  # 提取特征并保存
    cal_mean_std()  # 计算均值和标准差

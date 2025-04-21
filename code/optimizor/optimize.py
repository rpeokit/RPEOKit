from __future__ import print_function
import argparse
import random
from collections import defaultdict, deque
from sequence_env_gp import Seq_env, Mutate
from mcts_alphaZero_mutate import MCTSMutater
from p_v_net_2 import PolicyValueNet
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import torch.utils.data as data
from torch.utils.data import DataLoader
import time

from sklearn.cluster import KMeans
from sklearn.gaussian_process import GaussianProcessRegressor
import os
import pandas as pd

#酶的野生型氨基酸序列，用作优化的参考序列
wt_sequence = "MKKYWNRRKNRVEDFINKLTSQLSKLFPDAIFIFEDLDKFNMYDKNSNFNRNLDRTNWRKIAKKLEYKSVVLYVNPHYTSKTCPVCGSKMKSQEGQVVKCDKCGIFDRQFVRCYNIFKRGVELAKKLLGGVGVPVAGAEVDDLLSNEPRGELRLVKPNPNVEAKLPVRKSNRRFELQNPKDFVQIFDFPLMVYTVDLNGKYLKIYNCP"


#高斯回归模型，输入为蛋白质序列的特征和适应度，输出训练好的模型
#X_GP是输入的序列特征，Y_GP是对应的适应度分数
def train_regr_predictor(features, fitness, seed):
    X_GP = features
    Y_GP = fitness
    X_GP = np.asarray(X_GP)
    Y_GP = np.asarray(Y_GP)
    regr = GaussianProcessRegressor(random_state=seed)
    regr.fit(X_GP, Y_GP)
    return regr


#定义KMeans聚类，将序列的特征向量划分为n_clusters个类别，目的是在序列空间中形成子群，用于进一步优化分析
def run_Clustering(features, n_clusters, subclustering_index=np.zeros([0])):
    if len(subclustering_index) > 0:
        features_sub = features[subclustering_index, :]
    else:
        features_sub = features

    kmeans = KMeans(n_clusters=n_clusters).fit(features_sub)
    cluster_labels = kmeans.labels_

    Length = []
    Index = []

    if len(subclustering_index) > 0:
        for i in range(cluster_labels.max() + 1):
            index = subclustering_index[np.where(cluster_labels == i)[0]]
            l = len(index)
            Index.append(index)
            Length.append(l)
    else:
        for i in range(cluster_labels.max() + 1):
            index = np.where(cluster_labels == i)[0]
            l = len(index)
            Index.append(index)
            Length.append(l)

    return Index


#定义序列随机化函数，将聚类后得到的序列索引进行随机打乱，防止模型训练时出现顺序偏差
def shuffle_index(Index):
    for i in range(len(Index)):
        np.random.shuffle(Index[i])

    return Index

#将氨基酸序列转换为one-hot向量
def string_to_one_hot(sequence: str, alphabet: str) -> np.ndarray:

    out = np.zeros((len(sequence), len(alphabet)))
    for i in range(len(sequence)):
        out[i, alphabet.index(sequence[i])] = 1
    return out


seed = 100
AAS = "ILVAGMFYWEDQNHCRKSTP"

#定义氨基酸特征的处理函数，将单个氨基酸序列转换为特征矩阵，随后将其展平为一维向量，用于训练回归模型
def feature_single(variant):
    Feature = []
    aalist = list(AAS)
    for AA in variant:
        Feature.append([AA == aa for aa in aalist])
    Feature = np.asarray(Feature).astype(float)
    if len(Feature.shape) == 2:
        features = np.reshape(Feature, [Feature.shape[0] * Feature.shape[1]])

    return features

AAS = "ILVAGMFYWEDQNHCRKSTP"

#TrainPipeline 类实现了强化学习的训练流程，结合了 MCTS 和策略-价值网络，逐步优化蛋白质序列
class TrainPipeline():
    def __init__(self, start_seq_pool, alphabet, model, trust_radius, init_model=None): #init_model=None  feature_list, #, combo_feature_map, combo_index_map, first_round_index, round,
        # params of the board and the game
        self.seq_len = len(start_seq_pool[0])
        self.vocab_size = len(alphabet)
        self.n_in_row = 4

        self.round = round
        #定义了序列优化的环境
        self.seq_env = Seq_env(
            self.seq_len,
            alphabet,
            model,
            start_seq_pool,
            trust_radius) 
        self.mutate = Mutate(self.seq_env)
        # training params
        self.learn_rate = 2e-3
        self.lr_multiplier = 1.0  # adaptively adjust the learning rate based on KL
        self.temp = 1.0  # the temperature param
        self.n_playout = 40  # num of simulations for each move 400
        self.c_puct = 10  # 5
        self.buffer_size = 10000
        self.batch_size = 8  # mini-batch size for training  512
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.play_batch_size = 1
        self.epochs = 5  # num of train_steps for each update
        self.kl_targ = 0.02
        self.check_freq = 50
        self.game_batch_num = 10000#1500
        self.best_win_ratio = 0.0
        # num of simulations used for the pure mcts, which is used as
        # the opponent to evaluate the trained policy
        self.pure_mcts_playout_num = 1000
        self.buffer_no_extend = False
        self.collected_seqs_index_set = set()
        # self.combo_to_index = combo_index_map
        # self.first_round_index = first_round_index
        self.update_predictor = 0
        self.updata_predictor_index_set = set()
        self.collected_seqs = set()
        self.seqs_and_fitness = []
        self.last_tmp_set_len = 0
        if init_model:
            # start training from an initial policy-value net
            #策略-价值网络，用于评估生成的序列
            self.policy_value_net = PolicyValueNet(self.seq_len,
                                                   self.vocab_size,
                                                   model_file=init_model)
        else:
            # start training from a new policy-value net
            self.policy_value_net = PolicyValueNet(self.seq_len,
                                                   self.vocab_size)
        #通过 MCTS 搜索生成最优的序列突变
        self.mcts_player = MCTSMutater(self.policy_value_net.policy_value_fn,
                                      c_puct=self.c_puct,
                                      n_playout=self.n_playout,
                                      is_selfplay=1)

    #通过自对弈收集数据，使用 MCTS 和策略网络生成突变序列，并存储序列和对应的适应度
    def collect_selfplay_data(self, n_games=1):
        """collect self-play data for training"""
        #
        self.update_predictor = 0
        #
        self.buffer_no_extend = False
        for i in range(n_games):
            play_data, seqs_and_fitness = self.mutate.start_mutating(self.mcts_player,
                                                          temp=self.temp)    
            play_data = list(play_data)[:]
            self.episode_len = len(play_data)
            if self.episode_len == 0:
                self.buffer_no_extend = True
            else:
                self.data_buffer.extend(play_data)
                s_f_list = list(seqs_and_fitness)
                for seq, fitness ,frag_seq in s_f_list:
                    if seq not in ep_start_pool and seq not in self.collected_seqs:
                        self.collected_seqs.add(seq)
                        self.seqs_and_fitness.append({"sequence": seq, "fitness": fitness, "frag":frag_seq})

    def policy_update(self):
        """update the policy-value net"""
        mini_batch = random.sample(self.data_buffer, self.batch_size)
        state_batch = [data[0] for data in mini_batch]
        mcts_probs_batch = [data[1] for data in mini_batch]
        winner_batch = [data[2] for data in mini_batch]
        old_probs, old_v = self.policy_value_net.policy_value(state_batch)
        for i in range(self.epochs):
            loss, entropy = self.policy_value_net.train_step(
                    state_batch,
                    mcts_probs_batch,
                    winner_batch,
                    self.learn_rate*self.lr_multiplier)
            new_probs, new_v = self.policy_value_net.policy_value(state_batch)
            kl = np.mean(np.sum(old_probs * (
                    np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)),
                    axis=1)
            )
            if kl > self.kl_targ * 4:  # early stopping if D_KL diverges badly
                break
        # adaptively adjust the learning rate
        if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
            self.lr_multiplier /= 1.5
        elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
            self.lr_multiplier *= 1.5

        explained_var_old = (1 -
                             np.var(np.array(winner_batch) - old_v.flatten()) /
                             np.var(np.array(winner_batch)))
        explained_var_new = (1 -
                             np.var(np.array(winner_batch) - new_v.flatten()) /
                             np.var(np.array(winner_batch)))
        print(("kl:{:.5f},"
               "lr_multiplier:{:.3f},"
               "loss:{},"
               "entropy:{},"
               "explained_var_old:{:.3f},"
               "explained_var_new:{:.3f}"
               ).format(kl,
                        self.lr_multiplier,
                        loss,
                        entropy,
                        explained_var_old,
                        explained_var_new))
        return loss, entropy

    #主训练循环，通过多次迭代生成和优化序列，并在适应度达到一定数量时保存生成的序列
    def run(self):
        """run the training pipeline"""
        try:
            for i in range(self.game_batch_num):
                self.collect_selfplay_data(self.play_batch_size)
                print("batch i:{}, episode_len:{}".format(
                        i+1, self.episode_len))

                if len(self.seqs_and_fitness) >= 150:
                    saved_flag_4 = 1
                    #and saved_flag != 1: #100
                    df = pd.DataFrame(self.seqs_and_fitness)
                    df.to_csv(r"./output_design/Tnpb_generated_1.csv", index=False)
                    print("saving seqs.............")
                    break

                if len(self.data_buffer) > self.batch_size and self.buffer_no_extend == False:
                    loss, entropy = self.policy_update()
                
        except KeyboardInterrupt:
            print('\n\rquit')

if __name__ =='__main__':
    parser = argparse.ArgumentParser(description='Simple_cnn')
    
    parser.add_argument("--gpu", type=int, default=0,
                        help="which GPU to use. Set -1 to use CPU")
    parser.add_argument("--epochs", type=int, default=26,
                        help="number of training epochs")
    parser.add_argument("--seq_len", type=int, default=4,
                        help="protein len")
    parser.add_argument("--alphabet_len", type=int, default=0,
                        help="alphabet len")
    parser.add_argument("--batch_size", type=int, default=24,
                        help="batch size")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-3,
                        help="batch size")
    args = parser.parse_args()
    
    # 输入文件
    df_seq = pd.read_csv(r"./input_file/tnpb_model_input_data_new.csv")

    sequences = df_seq["seqs"].values
    Fitness = df_seq["fitness"].values
    Fitness = Fitness / Fitness.max()
    # max start
    seq_to_fit = dict(zip(sequences, Fitness))
    first_round_d = sorted(seq_to_fit.items(), key=lambda x: x[1], reverse=True)  # [:40]
    ep_start_pool = [k for k, v in first_round_d]
    # max start
    # random start
    rand_seqs = list(sequences)
    random.shuffle(rand_seqs)
    seq_list = list(sequences)
    Features = []
    for variant in seq_list:
        feature=feature_single(variant)
        Features.append(feature)
    Features = np.array(Features)
    model_gp = train_regr_predictor(Features, Fitness, seed)
        
    training_pipeline = TrainPipeline(
        rand_seqs, # rand start
        AAS,
        model_gp,
        trust_radius=15,

    )
    training_pipeline.run()






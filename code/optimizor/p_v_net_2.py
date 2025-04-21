# -*- coding: utf-8 -*-

# 实现了策略-价值网络（Policy-Value Network），这个网络结合了卷积神经网络（CNN）来评估当前状态下的可能行动，
# 并输出每个行动的概率和状态的价值。这种网络常用于强化学习中的决策任务，
# 特别是与蒙特卡洛树搜索（MCTS）相结合时，例如在 AlphaZero 中。
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


def set_learning_rate(optimizer, lr):
    """Sets the learning rate to the given value"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# 基于 PyTorch 的卷积神经网络
class Net(nn.Module):
    """policy-value network module"""
    def __init__(self, board_width, board_height):
        super(Net, self).__init__()

        self.board_width = board_width
        self.board_height = board_height
        # common layers
        #共享层（common layers）：3 层 1D 卷积层，分别有 32、64 和 128 个过滤器，负责提取输入状态的特征
        self.conv1 = nn.Conv1d(20, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        # action policy layers
        #动作策略层（action policy layers）：这部分网络输出每个可能动作的概率，最终通过 softmax 得到归一化的概率分布
        self.act_conv1 = nn.Conv1d(128, 80, kernel_size=3, padding=1)
        self.act_fc1 = nn.Linear(4*board_width*board_height,
                                 board_width*board_height)
        # state value layers
        #状态价值层（state value layers）：用于估算当前状态的价值（胜率），输出一个标量值，使用 tanh 激活函数来保证输出在 [-1, 1] 之间
        self.val_conv1 = nn.Conv1d(128, 20, kernel_size=3, padding=1)
        self.val_fc1 = nn.Linear(board_width*board_height, 128)
        self.val_fc2 = nn.Linear(128, 1)

    def forward(self, state_input):
        # common layers
        x = F.relu(self.conv1(state_input))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        # action policy layers
        x_act = F.relu(self.act_conv1(x))
        x_act = x_act.view(-1, 4*self.board_width*self.board_height)
        x_act = F.log_softmax(self.act_fc1(x_act), dim=1)  #
        # state value layers
        x_val = F.relu(self.val_conv1(x))
        x_val = x_val.view(-1, self.board_width*self.board_height)
        x_val = F.relu(self.val_fc1(x_val))
        #x_val = F.tanh(self.val_fc2(x_val))
        x_val = torch.tanh(self.val_fc2(x_val))
        return x_act, x_val

# 包含一个策略-价值网络的实例，用于在强化学习中进行决策
class PolicyValueNet():
    """policy-value network """
    def __init__(self, board_width, board_height,
                 model_file=None, use_gpu=False):
        self.use_gpu = use_gpu
        self.board_width = board_width
        self.board_height = board_height
        self.l2_const = 1e-4  # coef of l2 penalty
        # the policy value net module
        if self.use_gpu:
            self.policy_value_net = Net(board_width, board_height).cuda()
        else:
            self.policy_value_net = Net(board_width, board_height)
        self.optimizer = optim.Adam(self.policy_value_net.parameters(),
                                    weight_decay=self.l2_const)

        if model_file:
            net_params = torch.load(model_file)
            self.policy_value_net.load_state_dict(net_params)
    # 给定一批状态，输出对应的每个可能动作的概率和当前状态的价值
    def policy_value(self, state_batch):
        """
        input: a batch of states
        output: a batch of action probabilities and state values
        """
        if self.use_gpu:
            state_batch = Variable(torch.FloatTensor(state_batch).cuda())
            log_act_probs, value = self.policy_value_net(state_batch)
            act_probs = np.exp(log_act_probs.data.cpu().numpy())
            return act_probs, value.data.cpu().numpy()
        else:
            state_batch = Variable(torch.FloatTensor(state_batch))
            log_act_probs, value = self.policy_value_net(state_batch)
            act_probs = np.exp(log_act_probs.data.numpy())
            return act_probs, value.data.numpy()
    # 用于获取给定棋盘状态的动作概率和状态价值。这里的 board 表示游戏状态，可以是蛋白质序列优化的状态表示
    def policy_value_fn(self, board):
        """
        input: board
        output: a list of (action, probability) tuples for each available
        action and the score of the board state
        """
        legal_positions = board.availables
        # current_state = np.ascontiguousarray(board.current_state().reshape(
        #         -1, 2, self.board_width, self.board_height))  ##
        current_state_0 = np.expand_dims(board.current_state(), axis = 0)
        current_state = np.ascontiguousarray(current_state_0)  ##

        if self.use_gpu:
            log_act_probs, value = self.policy_value_net(
                    Variable(torch.from_numpy(current_state)).cuda().float())
            act_probs = np.exp(log_act_probs.data.cpu().numpy().flatten())
        else:
            log_act_probs, value = self.policy_value_net(
                    Variable(torch.from_numpy(current_state)).float())
            act_probs = np.exp(log_act_probs.data.numpy().flatten())
        act_probs = zip(legal_positions, act_probs[legal_positions])
        value = value.data[0][0]
        return act_probs, value
    # 执行一次训练步骤，计算损失并进行反向传播更新网络参数。损失函数结合了策略损失和价值损失
    def train_step(self, state_batch, mcts_probs, winner_batch, lr):
        """perform a training step"""
        # wrap in Variable
        if self.use_gpu:
            state_batch = Variable(torch.FloatTensor(state_batch).cuda())
            mcts_probs = Variable(torch.FloatTensor(mcts_probs).cuda())
            winner_batch = Variable(torch.FloatTensor(winner_batch).cuda())
        else:
            state_batch = Variable(torch.FloatTensor(state_batch))
            mcts_probs = Variable(torch.FloatTensor(mcts_probs))
            winner_batch = Variable(torch.FloatTensor(winner_batch))

        # zero the parameter gradients
        self.optimizer.zero_grad()
        # set learning rate
        set_learning_rate(self.optimizer, lr)

        # forward
        log_act_probs, value = self.policy_value_net(state_batch)
        # define the loss = (z - v)^2 - pi^T * log(p) + c||theta||^2
        # Note: the L2 penalty is incorporated in optimizer
        value_loss = F.mse_loss(value.view(-1), winner_batch)
        policy_loss = -torch.mean(torch.sum(mcts_probs*log_act_probs, 1))
        loss = value_loss + policy_loss
        # backward and optimize
        loss.backward()
        self.optimizer.step()
        # calc policy entropy, for monitoring only
        entropy = -torch.mean(
                torch.sum(torch.exp(log_act_probs) * log_act_probs, 1)
                )
        #return loss.data[0], entropy.data[0]
        #for pytorch version >= 0.5 please use the following line instead.
        return loss.item(), entropy.item()

    def get_policy_param(self):
        net_params = self.policy_value_net.state_dict()
        return net_params

    def save_model(self, model_file):
        """ save model params to file """
        net_params = self.get_policy_param()  # get model params
        torch.save(net_params, model_file)

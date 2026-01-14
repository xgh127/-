import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

"""
这是Q网络的强化学习算法，核心是批量更新。属于off-policy方法。

"""


# ===== 1. 定义Q网络（替代Q表，拟合Q函数）【对应单步版的 Q_table】=====
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
    def forward(self, x):
        return self.fc(x)  # 输入状态s，输出所有动作的Q值【对应单步版的 Q_table[discrete_s]】

# ===== 2. 定义经验池（存储单步样本）【对应单步版的 即时使用样本，无存储】=====
class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer = deque(maxlen=buffer_size)
    def add(self, exp):
        self.buffer.append(exp)  # 存储 (s,a,r,s',done)【对应单步版的 步骤2获取的单步样本】
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)  # 采样批量样本

# ===== 3. 初始化组件和超参数 =====
env = gym.make("CartPole-v1")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

# 超参数（批量更新专用超参数标注，单步版对应参数标注）
BUFFER_SIZE = 10000
BATCH_SIZE = 64  # 批量大小（一次处理64个单步样本）
GAMMA = 0.95     # 折扣因子【和单步版 gamma 完全一致】
LR = 1e-3        # 学习率（对应单步版 alpha，只是用于梯度下降）
epsilon = 0.1

# 初始化网络、优化器、损失函数、经验池
q_net = QNetwork(state_dim, action_dim)
optimizer = optim.Adam(q_net.parameters(), lr=LR)
criterion = nn.MSELoss()  # 均方误差损失（衡量批量TD误差的平均）
replay_buffer = ReplayBuffer(BUFFER_SIZE)

# ===== 批量更新核心训练循环 =====
for episode in range(100):
    state, _ = env.reset()
    done = False
    
    while not done:
        # 1. 选动作（和单步版一致，非核心计算）
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = q_net(state_tensor)  # 【对应单步版的 Q_table[discrete_s]】
            action = torch.argmax(q_values).item()
        
        # 2. 与环境交互，获取单步样本 (s,a,r,s',done)【和单步版 步骤2 完全一致】
        next_state, reward, done, _, _ = env.step(action)
        replay_buffer.add((state, action, reward, next_state, done))  # 存入经验池
        
        # 3. 满足条件后，执行批量更新（攒够样本才更新，对应单步版的 每步必更）
        if len(replay_buffer.buffer) > BATCH_SIZE:
            # ======================================
            # 核心计算部分（批量更新的核心5行，与单步版核心3行一一对应）
            # ======================================
            # 3.1 采样批量样本，转换为张量（适配神经网络计算）【对应单步版的 单个样本】
            batch = replay_buffer.sample(BATCH_SIZE)
            # 拆分批量样本：64个s、64个a、64个r、64个s'、64个done
            states = torch.FloatTensor(np.array([e[0] for e in batch]))
            actions = torch.LongTensor(np.array([e[1] for e in batch])).unsqueeze(1)
            rewards = torch.FloatTensor(np.array([e[2] for e in batch])).unsqueeze(1)
            next_states = torch.FloatTensor(np.array([e[3] for e in batch]))
            dones = torch.FloatTensor(np.array([e[4] for e in batch])).unsqueeze(1)
            
            # 3.2 计算批量TD目标：r + γ * maxQ(s', a') 【对应单步版 步骤3 td_target】
            # 解释：对64个s'，分别计算最大Q值，批量并行处理
            next_q_values = q_net(next_states)  # 计算64个s'的所有动作Q值【对应单步版 Q_table[discrete_next_s]】
            max_next_q = next_q_values.max(1)[0].unsqueeze(1)  # 取每个s'的最大Q值【对应单步版 np.max(Q_table[discrete_next_s])】
            td_target = rewards + GAMMA * max_next_q * (1 - dones)  # 批量TD目标【和单步版 td_target 公式完全一致】
            td_target = td_target.detach()  # 固定目标值，不参与梯度计算
            
            # 3.3 计算批量当前Q值：Q(s, a) 【对应单步版 步骤4 current_q】
            # 解释：对64个(s,a)，分别提取对应Q值，批量并行处理
            current_q_values = q_net(states)  # 计算64个s的所有动作Q值【对应单步版 Q_table[discrete_s]】
            current_q = current_q_values.gather(1, actions)  # 提取每个s对应动作a的Q值【对应单步版 Q_table[discrete_s + (action,)]】
            
            # 3.4 计算批量损失（TD误差的均方值）【对应单步版 TD误差 (td_target - current_q)】
            # 解释：单步版是单个TD误差，批量版是64个TD误差的平均，用于稳定更新
            loss = criterion(current_q, td_target)
            
            # 3.5 梯度下降更新网络参数（更新Q函数）【对应单步版 步骤5 更新Q表】
            # 解释：单步版是直接代数更新Q表，批量版是通过梯度下降更新网络权重（间接更新Q函数）
            optimizer.zero_grad()  # 清空梯度
            loss.backward()  # 计算梯度（基于批量损失）
            optimizer.step()  # 更新网络参数（对应单步版 Q表的数值更新）
            # ======================================
            # 核心计算结束
            # ======================================
        
        # 更新当前状态，进入下一步
        state = next_state

env.close()
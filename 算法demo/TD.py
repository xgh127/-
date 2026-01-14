import gymnasium as gym
import numpy as np
"""
这是Q表的强化学习算法，核心是单步更新。属于off-policy方法。

"""


# 1. 初始化环境（CartPole）
env = gym.make("CartPole-v1", render_mode="human")  # render_mode=human会显示动画
state_space = env.observation_space.shape[0]
action_space = env.action_space.n

# 2. 初始化Q表（状态离散化：CartPole的状态是连续的，先离散成有限区间）
def discretize_state(state):
    # 把小车的位置、速度、杆子角度、角速度离散成区间
    bins = [np.linspace(-4.8, 4.8, 20),  # 小车位置
            np.linspace(-4, 4, 20),      # 小车速度
            np.linspace(-0.418, 0.418, 20), # 杆子角度
            np.linspace(-4, 4, 20)]      # 杆子角速度
    discrete_state = tuple([np.digitize(s, b) for s, b in zip(state, bins)])
    return discrete_state

Q_table = np.zeros((20, 20, 20, 20, action_space))  # 离散后状态+动作的Q表，形状为(20,20,20,20,2)

# 3. 超参数
alpha = 0.1  # 学习率
gamma = 0.95 # 折扣因子
epsilon = 0.1 # ε-贪心（探索概率）
# ===== 新增：统计每集步数 =====
episode_rewards = []  # 存储每集步数（CartPole中每步奖励=1，步数=总奖励）
# 4. 训练循环
for episode in range(1000):
    state, _ = env.reset()
    # print(state)
    discrete_s = discretize_state(state) # 这里是index，映射过去的index
    # print(discrete_s)
    done = False
    step_count = 0  # 记录当前集的步数
    # print(Q_table)
    while not done:
        step_count += 1
        # ε-贪心选动作：有ε概率随机选，否则选Q最大的动作
        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            # 获取a
            action = np.argmax(Q_table[discrete_s]) # 输入当前状态，会得到不同动作和状态的Q值，即Q（s|a_i）(这里就两个动作，向左或向右对应不同Q值，所以argmax后就是1或0)
            # print(f" Action {action}, Q-value {Q_table[discrete_s + (action,)]}")
        
        # 执行动作，得到下一个状态、奖励，获取下一状态s',r
        next_state, reward, done, _, _ = env.step(action)

        # 离散化下一个状态
        discrete_next_s = discretize_state(next_state)
        
        # Q-Learning核心更新：Q(s,a) = Q(s,a) + α*(r + γ*maxQ(s',a') - Q(s,a))
        # 找到下一个状态的最大Q值，怎么找到的
        # 这是max_a' Q(s',a')
        
        # 1.Learning是off-policy的，因为它使用了下一个状态的最大Q值，而不考虑实际采取的动作
        max_next_Q = np.max(Q_table[discrete_next_s]) # 输入当前状态，找到下一个状态的最大Q值，比如[0.68598675 0.11219399]
        
        Q_table[discrete_s + (action,)] += alpha * (reward + gamma * max_next_Q - Q_table[discrete_s + (action,)]) # 找到大的那一个，更新对的Q值
        
        # 2. 如果是sarsa的话，这里采用epsilon-greedy选出的动作对应的Q值，而不是maxQ值
        # if np.random.uniform(0, 1) < epsilon:
        #     next_action = env.action_space.sample()
        # else:
        #     next_action = np.argmax(Q_table[discrete_next_s])
        # next_Q = Q_table[discrete_next_s + (next_action,)]
        # 更新Q值
        # Q_table[discrete_s + (action,)] += alpha * (reward + gamma * next_Q - Q_table[discrete_s + (action,)])

        discrete_s = discrete_next_s

        # 记录当前集步数
    episode_rewards.append(step_count)
     # 每100集打印一次平均步数，看学习趋势
    if (episode + 1) % 100 == 0:
        avg_step = np.mean(episode_rewards[-100:])
        print(f"第{episode+1}集，最近100集平均步数：{avg_step:.2f}")

# ===== 新增：训练完成后，验证学习成果（关闭探索，只选最优动作）=====
print("\n开始验证学习成果（关闭探索，只选最优动作）...")
for episode in range(5):
    state, _ = env.reset()
    discrete_s = discretize_state(state)
    done = False
    step_count = 0
    
    while not done:
        step_count += 1
        # 关闭ε探索，只选Q值最大的最优动作
        action = np.argmax(Q_table[discrete_s])
        next_state, reward, done, _, _ = env.step(action)
        discrete_next_s = discretize_state(next_state)
        discrete_s = discrete_next_s
    
    print(f"验证第{episode+1}集，步数：{step_count}")

env.close()

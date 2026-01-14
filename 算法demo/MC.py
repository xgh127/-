import gymnasium as gym
import numpy as np

env = gym.make("CartPole-v1", render_mode="human")
state_space = env.observation_space.shape[0]

# 状态离散化（不变）
def discretize_state(state):
    bins = [np.linspace(-4.8, 4.8, 20),
            np.linspace(-4, 4, 20),
            np.linspace(-0.418, 0.418, 20),
            np.linspace(-4, 4, 20)]
    return tuple([np.digitize(s, b) for s, b in zip(state, bins)])

V = np.zeros((20, 20, 20, 20))
N = np.zeros((20, 20, 20, 20))
gamma = 0.95

# 训练循环（不变）
for episode in range(1000):
    state, _ = env.reset()
    discrete_s = discretize_state(state)
    done = False
    trajectory = []
    
    while not done:
        action = env.action_space.sample()
        next_state, reward, done, _, _ = env.step(action)
        trajectory.append((discrete_s, reward))
        discrete_s = discretize_state(next_state)
    
    G = 0
    for (state, reward) in reversed(trajectory):
        G = reward + gamma * G
        N[state] += 1
        V[state] += (G - V[state]) / N[state]

# ===== 新增：打印典型状态的V值，体现学习成果 =====
print("===== 典型状态的V值对比 =====")
# 状态1：好状态（小车居中、杆子垂直）
good_state = (10, 10, 10, 10)
print(f"好状态（小车居中、杆子垂直）V值：{V[good_state]:.2f}")

# 状态2：坏状态1（小车靠边、杆子垂直）
bad_state1 = (1, 10, 10, 10)
print(f"坏状态1（小车靠左、杆子垂直）V值：{V[bad_state1]:.2f}")

# 状态3：坏状态2（小车居中、杆子严重倾斜）
bad_state2 = (10, 10, 1, 10)
print(f"坏状态2（小车居中、杆子严重左倾）V值：{V[bad_state2]:.2f}")

env.close()
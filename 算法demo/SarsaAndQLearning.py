import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
class CliffMaze:
    def __init__(self):
        # 迷宫尺寸：5行8列（对应你的示意图）
        self.rows = 5
        self.cols = 8
        # 起点(行,列)、终点(行,列)
        self.start = (4, 0)
        self.end = (4, 7)
        # 悬崖区域（黑色部分）
        self.cliff = [(4, i) for i in range(1, 7)]
        # 动作：上、下、左、右（对应0-3）
        self.actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 动作顺序：上、下、左、右
        self.action_names = ["上", "下", "左", "右"]

    def reset(self):
        # 重置到起点
        return self.start

    def step(self, state, action):
        # 计算下一个状态
        x, y = state
        dx, dy = self.actions[action]
        next_x = x + dx
        next_y = y + dy

        # 边界检查：不能走出迷宫
        next_x = max(0, min(next_x, self.rows - 1))
        next_y = max(0, min(next_y, self.cols - 1))
        next_state = (next_x, next_y)

        # 奖励设置
        if next_state in self.cliff:
            # 掉进悬崖：惩罚-100，回合结束
            reward = -100
            done = True
        elif next_state == self.end:
            # 到达终点：奖励100，回合结束
            reward = 100
            done = True
        else:
            # 其他位置：小惩罚，鼓励尽快到达
            reward = -1
            done = False

        return next_state, reward, done
def train_q_learning(maze, episodes=5000, alpha=0.1, gamma=0.9, epsilon=0.5):
    # 初始化Q表：状态(元组)→动作(列表)，初始值为0
    Q = {}
    for x in range(maze.rows):
        for y in range(maze.cols):
            Q[(x, y)] = [0.0 for _ in range(len(maze.actions))]

    for episode in range(episodes):
        state = maze.reset()
        done = False
        # 探索率衰减：后期减少探索
        current_epsilon = max(0.01, epsilon * (0.99 ** (episode // 100)))

        while not done:
            """
            1.行为策略:ε-贪心,
            以概率 1-ε 选择当前状态下 Q 值最大的动作。这有助于智能体利用已经学到的最佳策略来获得最大奖励。
            """
            # ε-贪心选动作，以概率 1−ε 选择当前状态下 Q 值最大的动作。这有助于智能体利用已经学到的最佳策略来获得最大奖励。
            if np.random.uniform(0, 1) < current_epsilon:
                action = np.random.choice(len(maze.actions))
            else:
                action = np.argmax(Q[state])

            # 执行动作
            next_state, reward, done = maze.step(state, action)
            """
            2.目标策略:贪心
            选择使得next_state下 Q 值最大的动作。
            然后用该动作去更新Q值
            """
            # Q-Learning核心更新：用next_state的最大Q值
            max_next_Q = np.max(Q[next_state])
            Q[state][action] += alpha * (reward + gamma * max_next_Q - Q[state][action])

            # 更新状态
            state = next_state

    return Q
def train_sarsa(maze, episodes=5000, alpha=0.1, gamma=0.9, epsilon=0.5):
    # 初始化Q表（和Q-Learning相同）
    Q = {}
    for x in range(maze.rows):
        for y in range(maze.cols):
            Q[(x, y)] = [0.0 for _ in range(len(maze.actions))]

    for episode in range(episodes):
        state = maze.reset()
        done = False
        current_epsilon = max(0.01, epsilon * (0.99 ** (episode // 100)))

        """
        1.行为策略:𝜖-贪心,
        以概率 1-𝜖 选择当前状态下 Q 值最大的动作。这有助于智能体利用已经学到的最佳策略来获得最大奖励。
        """
        # SARSA：先选初始动作，
        if np.random.uniform(0, 1) < current_epsilon:
            action = np.random.choice(len(maze.actions))
        else:
            action = np.argmax(Q[state])

        while not done:
            # 执行动作
            next_state, reward, done = maze.step(state, action)
            """
            2.目标策略:𝜖-贪心
            利用行为策略选择的action,与环境交互,获取下一个状态和奖励
            以ε选择随机动作
            以概率 1-ε 选择当前状态下 Q 值最大的动作。
            然后用实际动作去更新Q值
            """
            # SARSA：选next_state的实际动作
            if np.random.uniform(0, 1) < current_epsilon:
                next_action = np.random.choice(len(maze.actions))
            else:
                next_action = np.argmax(Q[next_state])

            # SARSA核心更新：用next_state的实际动作Q值
            Q[state][action] += alpha * (reward + gamma * Q[next_state][next_action] - Q[state][action])

            # 更新状态和动作
            state = next_state
            action = next_action

    return Q
def get_path(maze, Q):
    path = []
    state = maze.reset()
    path.append(state)
    done = False

    while not done:
        # 仅选最优动作（无探索）
        action = np.argmax(Q[state])
        next_state, reward, done = maze.step(state, action)
        path.append(next_state)
        state = next_state
        # 防止无限循环（如果Q表没学好）
        if len(path) > 100:
            break
    return path
def visualize(maze, q_learning_path, sarsa_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_xlim(0, maze.cols)
    ax.set_ylim(0, maze.rows)
    ax.set_xticks(np.arange(maze.cols + 1))
    ax.set_yticks(np.arange(maze.rows + 1))
    ax.invert_yaxis()  # 让(0,0)在左下角，对应迷宫的起点位置

    # 绘制迷宫元素
    # 起点（黄色）
    ax.add_patch(Rectangle((maze.start[1], maze.start[0]), 1, 1, color='yellow'))
    # 终点（绿色）
    ax.add_patch(Rectangle((maze.end[1], maze.end[0]), 1, 1, color='lightgreen'))
    # 悬崖（黑色）
    for (x, y) in maze.cliff:
        ax.add_patch(Rectangle((y, x), 1, 1, color='black'))

    # 绘制Q-Learning路径（绿色）
    q_x = [y + 0.5 for (x, y) in q_learning_path]
    q_y = [x + 0.5 for (x, y) in q_learning_path]
    ax.plot(q_x, q_y, color='green', linewidth=3, label='Q-Learning')

    # 绘制SARSA路径（橙色）
    s_x = [y + 0.5 for (x, y) in sarsa_path]
    s_y = [x + 0.5 for (x, y) in sarsa_path]
    ax.plot(s_x, s_y, color='orange', linewidth=3, label='SARSA')

    ax.legend()
    plt.grid(True)
    plt.title("Q-Learning vs SARSA in Cliff Maze")
    plt.show()
    # 初始化迷宫
maze = CliffMaze()

# 训练Q-Learning和SARSA
# 比较两种算法的收敛时间
start_time = time.time()
q_learning_Q = train_q_learning(maze)
q_time = time.time() - start_time
print(f"Q-Learning训练时间: {q_time:.2f}秒")

start_time = time.time()
sarsa_Q = train_sarsa(maze)
sarsa_time = time.time() - start_time
print(f"SARSA训练时间: {sarsa_time:.2f}秒")

if q_time < sarsa_time:
    print("Q-Learning算法收敛更快！")
else:
    print("SARSA算法收敛更快！")

# 获取路径
q_learning_path = get_path(maze, q_learning_Q)
sarsa_path = get_path(maze, sarsa_Q)

# 可视化
visualize(maze, q_learning_path, sarsa_path)
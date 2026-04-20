import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

# ==========================================
# 1. 遊戲環境 (Tic-Tac-Toe Environment)
# ==========================================
class TicTacToeEnv:
    def __init__(self):
        self.board = np.zeros(9, dtype=np.float32)
        self.done = False

    def reset(self):
        self.board = np.zeros(9, dtype=np.float32)
        self.done = False
        return self.board.copy()

    def check_winner(self):
        # 所有可能的贏法 (直線、橫線、斜線)
        win_conditions = [(0,1,2), (3,4,5), (6,7,8), (0,3,6), (1,4,7), (2,5,8), (0,4,8), (2,4,6)]
        for a, b, c in win_conditions:
            if self.board[a] == self.board[b] == self.board[c] and self.board[a] != 0:
                return self.board[a] # 回傳贏家 (1 或 -1)
        if 0 not in self.board:
            return 0 # 平手
        return None # 遊戲繼續

    def step(self, action, player):
        # 1. 違規移動懲罰
        if self.board[action] != 0:
            self.done = True
            return self.board.copy(), -10.0, self.done 

        # 2. 正常落子
        self.board[action] = player
        winner = self.check_winner()

        # 3. 給予獎勵
        if winner == player:
            reward = 1.0    # 贏了
            self.done = True
        elif winner is not None:
            reward = 0.0    # 平手 (給 0 分或 0.5 分皆可)
            self.done = True
        else:
            reward = 0.0    # 遊戲繼續，這步沒得分
        
        return self.board.copy(), reward, self.done

# ==========================================
# 2. 神經網路模型 (Deep Q-Network)
# ==========================================
class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        # 輸入 9 個格子，隱藏層 64，輸出 9 個動作的 Q 值
        self.net = nn.Sequential(
            nn.Linear(9, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 9)
        )

    def forward(self, x):
        return self.net(x)

# ==========================================
# 3. 訓練與記憶庫設定
# ==========================================
env = TicTacToeEnv()
model = DQN()
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

# 經驗回放池 (Replay Buffer)
memory = deque(maxlen=10000)

# 超參數
EPISODES = 5000
GAMMA = 0.99       # 折扣因子
epsilon = 1.0      # 初始探索率
epsilon_min = 0.1  # 最低探索率
epsilon_decay = 0.999 # 探索率衰減

print("開始訓練...")

for episode in range(EPISODES):
    state = env.reset()
    
    while True:
        # --- AI (玩家 1) 的回合 ---
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        # Epsilon-Greedy 策略：決定探索或利用
        if random.random() < epsilon:
            action = random.randint(0, 8) # 隨機亂走
        else:
            with torch.no_grad():
                q_values = model(state_tensor)
                # 為了避免選到已走過的位置，將非空格的 Q 值設為極小
                invalid_moves = (state != 0)
                q_values[0][invalid_moves] = -float('inf')
                action = torch.argmax(q_values).item()
        
        next_state, reward, done = env.step(action, player=1)
        
        # 儲存經驗 (state, action, reward, next_state, done)
        memory.append((state.copy(), action, reward, next_state.copy(), done))
        state = next_state

        # --- 隨機對手 (玩家 -1) 的回合 ---
        if not done:
            available_moves = np.where(state == 0)[0]
            opponent_action = random.choice(available_moves)
            state, _, done = env.step(opponent_action, player=-1)
            # 這裡我們簡化處理，對手走完後如果 AI 輸了，我們就在訓練時懲罰最後一步
            if done and env.check_winner() == -1:
                # 修改 memory 中最後一筆的 reward 為 -1 (因為 AI 輸了)
                last_exp = list(memory[-1])
                last_exp[2] = -1.0 
                memory[-1] = tuple(last_exp)

        # --- 神經網路學習 (Experience Replay) ---
        if len(memory) > 64:
            batch = random.sample(memory, 64)
            b_states = torch.FloatTensor(np.array([b[0] for b in batch]))
            b_actions = torch.LongTensor(np.array([b[1] for b in batch])).view(-1, 1)
            b_rewards = torch.FloatTensor(np.array([b[2] for b in batch])).view(-1, 1)
            b_next_states = torch.FloatTensor(np.array([b[3] for b in batch]))
            b_dones = torch.FloatTensor(np.array([b[4] for b in batch])).view(-1, 1)

            # 計算當前 Q 值
            curr_q = model(b_states).gather(1, b_actions)
            
            # 計算目標 Q 值 (Bellman Equation)
            with torch.no_grad():
                max_next_q = model(b_next_states).max(1)[0].view(-1, 1)
                target_q = b_rewards + GAMMA * max_next_q * (1 - b_dones)

            # 更新權重
            loss = loss_fn(curr_q, target_q)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if done:
            break

    # 降低探索率
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

    if (episode + 1) % 500 == 0:
        print(f"訓練回合: {episode + 1}/{EPISODES}, Epsilon: {epsilon:.3f}")

print("訓練完成！模型已具備基礎智力。")
# 將模型權重存成檔案
torch.save(model.state_dict(), "tictactoe_dqn.pth")
print("模型已儲存為 tictactoe_dqn.pth")
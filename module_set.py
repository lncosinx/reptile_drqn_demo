import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque


# 定义奖励计算类
class RewardSet:
    def __init__ (self, num_agents, device):
        self.num_agents = num_agents
        self.device = device
        self.num_get_obs_rewards_tensor = torch.zeros(num_agents, dtype=torch.int32, device=self.device)
        self.total_episode_reward = torch.zeros(num_agents, dtype=torch.float32, device=self.device)

    def _find_goal_reward(self, states_tensor, initial_rewards_tensor):
        """
        检查观察中是否存在目标，并为每个智能体返回单独的奖励。

        Args:
            states_tensor (torch.Tensor): 形状为 (B, C, H, W) 的观测张量，B 是智能体数量。
            num_get_obs_rewards_tensor (list[int]): 形状为 (B,) 的列表，跟踪每个智能体的计数值。
            device (torch.device): GPU 或 CPU。

        Returns:
            torch.Tensor: 形状为 (B,) 的奖励张量，例如 [1.0, 0.0, 1.0]
            list[int]: 更新后的计数值列表。
        """
        
        B, C, H, W = states_tensor.shape
        
        # 1. 获取目标通道 (B, H, W)
        target_channel_tensor = states_tensor[:, 2, :, :]  #

        # 2. 初始化 B 个奖励
        find_goal_reward_tensor = torch.zeros_like(initial_rewards_tensor)  #
        # 3. 逐个智能体检查
        for i in range(B):
            # 3a. 获取该智能体已获得的奖励次数
            num_rewards_agent_i = self.num_get_obs_rewards_tensor[i].item()
            
            # 3b. 计算该智能体的视野遮罩
            idx = min(num_rewards_agent_i, H // 2) #
            check_tensor_2d = torch.zeros((H, W), dtype=torch.float32, device=self.device) #
            check_tensor_2d[idx : H - idx, idx : W - idx] = 1.0 #
            
            # 3c. 只检查第 i 个智能体的视野
            agent_i_obs = target_channel_tensor[i, :, :] # 形状 (H, W)
            
            # 3d. 应用遮罩并求和
            goal_in_obs = agent_i_obs * check_tensor_2d #
            
            # 3e. 如果该智能体看到目标，给予奖励
            if goal_in_obs.sum() > 0: #  
                # 并且更新该智能体的计数值
                self.num_get_obs_rewards_tensor[i] += 1
                # 智能体每次发现目标时，获得与其计数值相等的奖励
                find_goal_reward_tensor[i] += self.num_get_obs_rewards_tensor[i] 

            # 3f. 将获得发现目标奖励的智能体的计数值置0
            self.num_get_obs_rewards_tensor *= (1 - initial_rewards_tensor.int()) 

        return find_goal_reward_tensor
    
    def _stop_punish(self, actions_tensor):
        """
        对于执行停止动作的智能体，给予惩罚奖励。
        """
        stop_action_value = 0  # 停止动作的值
        punish_reward_value = -0.1  # 惩罚奖励值
        # actions_tensor 是 (B,)
        # (actions_tensor == stop_action_value) 产生一个 bool 张量 [True, False, ...]
        # .float() 将其转换为 [1.0, 0.0, ...]
        # 乘以惩罚值
        return (actions_tensor == stop_action_value).float() * punish_reward_value
    
    def _step_punish(self):
        """
        对于智能体，给予时间惩罚奖励。
        """
        punish_reward_value = -0.01  # 惩罚奖励值
        return torch.full((self.num_agents,), punish_reward_value, dtype=torch.float32, device=self.device)
    
    def _goal_rewards(self, initial_rewards_tensor):
        """
        计算目标奖励，得到的奖励乘oal_reawrds_coeff。
        """
        goal_reawrds_coeff = 10.0
        
        return initial_rewards_tensor * goal_reawrds_coeff
    
    def calculate_total_reward(self, rewards, states_tensor, actions):
        """
        计算总奖励，结合所有自定义奖励，并更新内部状态（计数器）。

        Args:
            rewards (list[float]): [B,] 来自 pogema.step() 的原始奖励列表
            states_tensor (torch.Tensor): (B, C, H, W) 观测张量
            actions (np.ndarray): (B,) 智能体执行的动作

        Returns:
            torch.Tensor: (B,) 的总自定义奖励张量
        """
        
        # 5. 将输入转换为张量
        # 我们假设 B = num_agents
        initial_rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        actions_tensor = torch.tensor(actions, dtype=torch.int64, device=self.device)
        
        # 6. 调用私有方法
        find_goal_rewards_tensor = self._find_goal_reward(states_tensor, initial_rewards_tensor)
        stop_punish_tensor = self._stop_punish(actions_tensor)
        step_punish_tensor = self._step_punish()
        goal_rewards_tensor = self._goal_rewards(initial_rewards_tensor)
        
        total_rewards_tensor = find_goal_rewards_tensor + stop_punish_tensor + step_punish_tensor + goal_rewards_tensor

        # 7. 更新回合总奖励
        self.total_episode_reward += total_rewards_tensor
        
        return total_rewards_tensor
    
    def total_rewards(self):
        """
        计算总奖励，结合发现奖励、目标到达奖励、停止惩罚奖励、时间惩罚奖励。
        """
        # .cpu().numpy() 适用于当 B > 1 时
        # .item() 适用于 B=1
        if self.num_agents == 1:
            return self.total_episode_reward[0].item()
        else:
            return self.total_episode_reward.cpu().numpy()
    

# 定义DRQN网络
class CnnQnet(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(CnnQnet, self).__init__()
        input_channels = input_shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )

        with torch.no_grad():
            dummy_input = torch.zeros(1, input_channels, input_shape[1], input_shape[2])
            cnn_output_features = self.cnn(dummy_input).shape[1]

        self.lstm_hidden_size = 512
        self.lstm = nn.LSTM(
            input_size = cnn_output_features, 
            hidden_size = self.lstm_hidden_size, 
            batch_first = True
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(self.lstm_hidden_size, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def forward(self, x, hidden_state = None):
        if x.dim() == 5: # (B, T, C, H, W)
            B, T, C, H, W = x.shape
        else: # (B, C, H, W)
            B, C, H, W = x.shape
            T = 1
            x = x.unsqueeze(1) # (B, 1, C, H, W)

        cnn_in = x.view(B * T, C, H, W)
        cnn_out = self.cnn(cnn_in)

        lstm_in = cnn_out.view(B, T, -1)
        lstm_out, new_hidden_state = self.lstm(lstm_in, hidden_state)

        fc_in = lstm_out.contiguous().view(B * T, -1)
        fc_out = self.fc_layers(fc_in)
        qvalues = fc_out.view(B, T, -1)

        if T == 1:
            qvalues = qvalues.squeeze(1)
        return qvalues, new_hidden_state
    

#经验回放池
class ReplayBuffer:
    def __init__(self, capacity, seq_len, num_agents, obs_shape, device):
        self.capacity = capacity
        self.seq_len = seq_len
        self.num_agents = num_agents
        self.obs_shape = obs_shape
        self.device = device
        self.buffer = deque(maxlen=capacity) 

    def push(self, episode_data):

        self.buffer.append(episode_data)

    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            if not self.buffer:
                return None
            sampled_episodes = random.choices(self.buffer, k=batch_size)    #有放回采样
        else:
            sampled_episodes = random.sample(self.buffer, batch_size)   #无放回采样

        batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones = [], [], [], [], []

        for episode in sampled_episodes:
            start_idx = random.randint(0, len(episode['states']) - self.seq_len)
            end_idx = start_idx + self.seq_len

            batch_states.append(episode['states'][start_idx:end_idx])
            batch_actions.append(episode['actions'][start_idx:end_idx])
            batch_rewards.append(episode['rewards'][start_idx:end_idx])
            batch_next_states.append(episode['next_states'][start_idx:end_idx])
            batch_dones.append(episode['dones'][start_idx:end_idx])

        states_np = np.array(batch_states, dtype=np.float32)
        actions_np = np.array(batch_actions, dtype=np.int64)
        rewards_np = np.array(batch_rewards, dtype=np.float32)
        next_states_np = np.array(batch_next_states, dtype=np.float32)
        dones_np = np.array(batch_dones, dtype=np.float32)

        states_np = states_np.transpose(0, 2, 1, 3, 4, 5)
        next_states_np = next_states_np.transpose(0, 2, 1, 3, 4, 5)
        actions_np = actions_np.transpose(0, 2, 1)
        rewards_np = rewards_np.transpose(0, 2, 1)
        dones_np = dones_np.transpose(0, 2, 1)

        batch_states_tensor = torch.tensor(states_np.reshape(-1, self.seq_len, *self.obs_shape), device=self.device)
        batch_next_states_tensor = torch.tensor(next_states_np.reshape(-1, self.seq_len, *self.obs_shape), device=self.device)
        batch_actions_tensor = torch.tensor(actions_np.reshape(-1, self.seq_len), device=self.device)
        batch_rewards_tensor = torch.tensor(rewards_np.reshape(-1, self.seq_len), device=self.device)
        batch_dones_tensor = torch.tensor(dones_np.reshape(-1, self.seq_len), device=self.device)

        return {
            'states': batch_states_tensor,
            'actions': batch_actions_tensor,
            'rewards': batch_rewards_tensor,
            'next_states': batch_next_states_tensor,
            'dones': batch_dones_tensor
        }

    def __len__(self):
        return len(self.buffer)


# 定义DRQN智能体
class DRQNAgent:
    def __init__(self, num_agents, state_shape, num_actions, replay_buffer, lr):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # [优化] 仅当在 CUDA 上时才使用 GradScaler
        self.use_scaler = torch.cuda.is_available()
        
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_scaler) #由于cuda版本问题，cuda版本较新时，请注释此行，恢复下行
        # self.scaler = torch.amp.GradScaler('cuda', enabled=self.use_scaler) 
        
        self.num_agents = num_agents
        self.state_shape = state_shape
        self.num_actions = num_actions
        self.lr = lr
        self.q_net = CnnQnet(state_shape, num_actions).to(self.device)
        self.target_q_net = CnnQnet(state_shape, num_actions).to(self.device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.lr)
        self.replay_buffer = replay_buffer
        self.gamma = 0.99
        self.batch_size = 32
        self.epsilon = 1.0
        self.epsilon_decay = 0.999999955
        self.epsilon_min = 0.1
        self.update_target_steps = 100
        self.step_count = 0

    def select_actions(self, states, current_hidden_state):
        self.q_net.eval()
        with torch.no_grad():
            q_values, new_hidden_state = self.q_net(states, current_hidden_state)
        self.q_net.train()

        greedy_actions = q_values.argmax(dim=-1)
        random_actions = torch.randint(0, self.num_actions, (states.shape[0],), device=self.device)
        is_random = torch.rand(states.shape[0], device=self.device) < self.epsilon
        actions = torch.where(is_random, random_actions, greedy_actions)
        return actions.cpu().numpy(), new_hidden_state
    
    def train(self):
        batch = self.replay_buffer.sample(self.batch_size)
        if batch is None:
            return None
        
        states = batch['states']
        actions = batch['actions']
        rewards = batch['rewards']
        next_states = batch['next_states']
        dones = batch['dones']

        # [优化] 明确启用 autocast
        
        with torch.cuda.amp.autocast(enabled=self.use_scaler):   #由于cuda版本问题，cuda版本较新时，请注释此行，恢复下行
        #with torch.amp.autocast('cuda', enabled=self.use_scaler):
            q_values, _ = self.q_net(states)
            q_values = q_values.gather(-1, actions.unsqueeze(-1)).squeeze(-1)

            with torch.no_grad():
                q_values_main_net, _ = self.q_net(next_states)
                next_actions = q_values_main_net.argmax(dim=-1)
                next_q_values_target_net, _ = self.target_q_net(next_states)
                target_next_q_values = torch.gather(next_q_values_target_net, 2, next_actions.unsqueeze(-1)).squeeze(-1)
                target_q_values = rewards + self.gamma * target_next_q_values * (1 - dones)
            
            loss = nn.MSELoss()(q_values, target_q_values)

        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        if self.step_count % self.update_target_steps == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        self.step_count += 1
        return loss.item()
    
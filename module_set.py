import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

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
    
class SumTree:
    """
    SumTree 数据结构，用于高效地按优先级采样。
    树的叶节点存储优先级 (p)，内部节点存储其子节点的总和。
    """
    def __init__(self, capacity):
        self.capacity = capacity
        # 树的层级结构。
        # 0: [-------------Root-------------]
        # 1: [---Child 1---] [---Child 2---]
        # ...
        # N: [p1] [p2] [p3] ... [p_capacity]
        # 实际的树数组大小为 2 * capacity - 1
        self.tree = np.zeros(2 * capacity - 1)
        # self.data 存储实际的 (s, a, r, s') 转换或 (episode_data)
        self.data = np.zeros(capacity, dtype=object)
        
        self.data_pointer = 0 # 指向 self.data 中的下一个写入位置
        self.n_entries = 0    # 缓冲区中的实际条目数

    def _propagate(self, tree_index, change):
        """将优先级的变化向上（向根）传播"""
        parent_index = (tree_index - 1) // 2
        self.tree[parent_index] += change
        
        # 递归传播直到根节点 (index 0)
        if parent_index != 0:
            self._propagate(parent_index, change)

    def _retrieve(self, tree_index, s):
        """
        在树中查找与采样值 's' 对应的叶节点
        """
        left_child_index = 2 * tree_index + 1
        right_child_index = left_child_index + 1

        # 如果我们到达了叶节点，则返回它
        if left_child_index >= len(self.tree):
            return tree_index

        if s <= self.tree[left_child_index]:
            return self._retrieve(left_child_index, s)
        else:
            return self._retrieve(right_child_index, s - self.tree[left_child_index])

    def total(self):
        """返回总优先级（根节点的值）"""
        return self.tree[0]

    def add(self, priority, data):
        """在树中添加一个新条目 (data) 及其优先级 (priority)"""
        # 找到要写入的叶节点索引 (在树数组中的位置)
        # 它与 self.data_pointer 相关，但偏移了 capacity - 1
        tree_index = self.data_pointer + self.capacity - 1
        
        self.data[self.data_pointer] = data
        
        # 更新优先级
        self.update(tree_index, priority)

        # 移动指针（循环缓冲区）
        self.data_pointer += 1
        if self.data_pointer >= self.capacity:
            self.data_pointer = 0
            
        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, tree_index, priority):
        """更新一个叶节点的优先级并向上传播变化"""
        change = priority - self.tree[tree_index]
        self.tree[tree_index] = priority
        self._propagate(tree_index, change)

    def get_leaf(self, s):
        """根据采样值 's' 获取叶节点索引、优先级和数据"""
        leaf_index = self._retrieve(0, s)
        data_index = leaf_index - self.capacity + 1
        
        return leaf_index, self.tree[leaf_index], self.data[data_index]

    def __len__(self):
        return self.n_entries

# 定义 DRQN 网络
class CRnnQnet(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(CRnnQnet, self).__init__()
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
            self.cnn_output_features = self.cnn(dummy_input).shape[1]

        self.lstm_hidden_size = 512
        self.lstm = nn.LSTM(
            input_size = self.cnn_output_features, 
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

# 经验回放池
class PrioritizedReplayBuffer:
    def __init__(self, capacity, seq_len, num_agents, obs_shape, device,
                 alpha=0.6, beta_start=0.4, beta_frames=100000):
        
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.seq_len = seq_len
        self.num_agents = num_agents
        self.obs_shape = obs_shape
        self.device = device
        
        # PER 超参数
        self.alpha = alpha               # [0~1] 优先级 p_i = (TD_error + epsilon)^alpha
        self.beta_start = beta_start     # [0~1] IS 权重 w_i = (N * P(i))^(-beta)
        self.beta = beta_start
        self.beta_increment = (1.0 - beta_start) / beta_frames
        self.epsilon = 0.01              # 用于 p_i 计算，避免 0 优先级
        self.max_priority = 1.0          # 新条目的初始优先级，确保它们被采样
        
    def push(self, episode_data):
        """
        存储一个完整的回合数据，并赋予其最大优先级，
        以确保新数据至少被采样一次。
        """
        self.tree.add(self.max_priority, episode_data)

    def sample(self, batch_size):
        if len(self.tree) == 0:
            return None

        # 准备批次数据
        batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones = [], [], [], [], []
        
        # 存储用于更新优先级的树索引和用于 IS 权重的优先级
        tree_indices = np.empty(batch_size, dtype=np.int32)
        sample_priorities = np.empty(batch_size, dtype=np.float32)

        # 1. 计算总优先级和分段
        total_p = self.tree.total()
        segment = total_p / batch_size

        # 2. 采样
        for i in range(batch_size):
            # 从每个分段中均匀采样一个值
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)

            # 3. 获取数据
            tree_idx, priority, episode_data = self.tree.get_leaf(s)
            
            if not isinstance(episode_data, dict):
                # 缓冲区中可能存在尚未被覆盖的 0
                continue 

            # 4. 从采样的回合中随机选择一个序列（与原 ReplayBuffer 相同）
            start_idx = random.randint(0, len(episode_data['states']) - self.seq_len)
            end_idx = start_idx + self.seq_len
            
            batch_states.append(episode_data['states'][start_idx:end_idx])
            batch_actions.append(episode_data['actions'][start_idx:end_idx])
            batch_rewards.append(episode_data['rewards'][start_idx:end_idx])
            batch_next_states.append(episode_data['next_states'][start_idx:end_idx])
            batch_dones.append(episode_data['dones'][start_idx:end_idx])
            
            tree_indices[i] = tree_idx
            sample_priorities[i] = priority

        # 5. 计算重要性采样 (IS) 权重
        # P(i) = priority_i / total_p
        sampling_probabilities = sample_priorities / total_p
        
        # w_i = (N * P(i))^(-beta)
        # N = len(self.tree)
        weights = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        
        # 归一化权重（为了稳定性）
        weights /= weights.max()
        
        # 6. 退火 beta
        self.beta = min(1.0, self.beta + self.beta_increment)

        # --- 7. 格式化批次数据 (与原 ReplayBuffer 相同) ---
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
        
        is_weights_tensor = torch.tensor(weights, dtype=torch.float32, device=self.device)

        batch_data = {
            'states': batch_states_tensor,
            'actions': batch_actions_tensor,
            'rewards': batch_rewards_tensor,
            'next_states': batch_next_states_tensor,
            'dones': batch_dones_tensor
        }
        
        # 返回批次数据、对应的树索引和 IS 权重
        return (batch_data, tree_indices, is_weights_tensor)

    def update_priorities(self, tree_indices, td_errors):
        """
        在训练后，根据 TD-Errors 更新采样的回合的优先级。
        
        Args:
            tree_indices (np.ndarray): sample() 返回的索引
            td_errors (np.ndarray): 批次中每个回合的（平均）TD-Error
        """
        
        # p_i = (|TD_error| + epsilon)^alpha
        priorities = (np.abs(td_errors) + self.epsilon) ** self.alpha
        priorities = np.clip(priorities, 0.001, self.max_priority)
        
        for idx, p in zip(tree_indices, priorities):
            self.tree.update(idx, p)

    def __len__(self):
        return len(self.tree)

# 定义 DRQN 智能体
class DRQNAgent:
    def __init__(self, num_agents, state_shape, num_actions, replay_buffer, lr):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # [优化] 仅当在 CUDA 上时才使用 GradScaler
        self.use_scaler = torch.cuda.is_available()
        
        # self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_scaler) #由于cuda版本问题，cuda版本较新时，请注释此行，恢复下行
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.use_scaler) 
        
        self.num_agents = num_agents
        self.state_shape = state_shape
        self.num_actions = num_actions
        self.lr = lr
        self.q_net = CRnnQnet(state_shape, num_actions).to(self.device)
        self.target_q_net = CRnnQnet(state_shape, num_actions).to(self.device)
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
        # 1. 采样现在返回 (batch, indices, is_weights)
        sample_data = self.replay_buffer.sample(self.batch_size)
        if sample_data is None:
            return None
        
        batch, indices, is_weights_tensor = sample_data
        
        states = batch['states']
        actions = batch['actions']
        rewards = batch['rewards']
        next_states = batch['next_states']
        dones = batch['dones']

        # 明确启用 autocast
        # with torch.cuda.amp.autocast(enabled=self.use_scaler):   #由于cuda版本问题，cuda版本较新时，请注释此行，恢复下行
        with torch.amp.autocast('cuda', enabled=self.use_scaler):
            q_values, _ = self.q_net(states)
            q_values = q_values.gather(-1, actions.unsqueeze(-1)).squeeze(-1)

            with torch.no_grad():
                q_values_main_net, _ = self.q_net(next_states)
                next_actions = q_values_main_net.argmax(dim=-1)
                next_q_values_target_net, _ = self.target_q_net(next_states)
                target_next_q_values = torch.gather(next_q_values_target_net, 2, next_actions.unsqueeze(-1)).squeeze(-1)
                target_q_values = rewards + self.gamma * target_next_q_values * (1 - dones)
            
            # 2. 计算逐元素的损失 (reduction='none')
            loss_fn = nn.MSELoss(reduction='none')
            elementwise_loss = loss_fn(q_values, target_q_values)
            
            # 3. 应用重要性采样 (IS) 权重
            #    is_weights_tensor 形状为 (B,)
            #    elementwise_loss 形状为 (B * N_Agents, T_Seq)
            
            #    将 is_weights 扩展以匹配 (B, N_Agents, T_Seq) -> (B * N_Agents, T_Seq)
            is_weights_expanded = is_weights_tensor.repeat_interleave(self.num_agents).unsqueeze(1)
            is_weights_expanded = is_weights_expanded.expand_as(elementwise_loss)

            #    应用权重并计算平均损失
            loss = (elementwise_loss * is_weights_expanded).mean()

        # 4. 在反向传播 *之前*，计算 TD-Errors 以更新优先级
        with torch.no_grad():
            # (B * N_Agents, T_Seq)
            td_errors = (q_values - target_q_values).abs()
            
            # 我们需要每个采样的 *回合* (B 个) 的优先级，
            # 而不是每个转换 (B * N_Agents * T_Seq 个)
            
            # 1. 变回 (B, N_Agents, T_Seq)
            td_errors_reshaped = td_errors.view(self.batch_size, self.num_agents, -1)
            
            # 2. 计算每个回合的平均 TD-Error (在 N_Agents 和 T_Seq 维度上取平均)
            episode_priorities = td_errors_reshaped.mean(dim=(1, 2)).cpu().numpy()

        # 5. 更新缓冲区中的优先级
        self.replay_buffer.update_priorities(indices, episode_priorities)


        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        if self.step_count % self.update_target_steps == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())

        # Epsilon 衰减现在由 meta-agent 控制，

        self.step_count += 1
        return loss.item()

# 定义 DQN 网络
class CnnQnet(CRnnQnet):
    def __init__(self, input_shape, num_actions):
        super(CnnQnet, self).__init__(input_shape, num_actions)

        self.fc_layers = nn.Sequential(
            nn.Linear(self.cnn_output_features, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def forward(self, x):
        T = 1
        if x.dim() == 5: # (B, T, C, H, W)
            B, T, C, H, W = x.shape
        else: # (B, C, H, W)
            B, C, H, W = x.shape
            x = x.unsqueeze(1)

        cnn_in = x.view(B * T, C, H, W)
        cnn_out = self.cnn(cnn_in)
        
        fc_in = cnn_out.view(B*T, -1)
        fc_out = self.fc_layers(fc_in)
        qvalues = fc_out.view(B, T, -1)

        if T == 1:
            qvalues = qvalues.squeeze(1)
        return qvalues
    
# 定义 DQN 智能体
class DQNAgent(DRQNAgent):
    def __init__(self, num_agents, state_shape, num_actions, replay_buffer, lr):
        super(DRQNAgent,self).__init__()
        # --- 重新初始化 DRQNAgent 的所有属性 ---
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_scaler = torch.cuda.is_available()
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_scaler)
        self.num_agents = num_agents
        self.state_shape = state_shape
        self.num_actions = num_actions
        self.lr = lr
        
        # 使用 CnnQnet 而不是 CRnnQnet
        self.q_net = CnnQnet(state_shape, num_actions).to(self.device)
        self.target_q_net = CnnQnet(state_shape, num_actions).to(self.device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.lr)
        self.replay_buffer = replay_buffer
        self.gamma = 0.99
        self.batch_size = 32
        self.epsilon = 1.0
        self.epsilon_decay = 0.999999955 # 与 DRQN/Reptile 保持一致
        self.epsilon_min = 0.1
        self.update_target_steps = 100
        self.step_count = 0

    def select_actions(self, states):
        self.q_net.eval()
        with torch.no_grad():
            q_values = self.q_net(states)
        self.q_net.train()

        greedy_actions = q_values.argmax(dim=-1)
        random_actions = torch.randint(0, self.num_actions, (states.shape[0],), device=self.device)
        is_random = torch.rand(states.shape[0], device=self.device) < self.epsilon
        actions = torch.where(is_random, random_actions, greedy_actions)
        return actions.cpu().numpy()
    
    def train(self):
        # 1. 采样现在返回 (batch, indices, is_weights)
        sample_data = self.replay_buffer.sample(self.batch_size)
        if sample_data is None:
            return None
        
        batch, indices, is_weights_tensor = sample_data
        
        states = batch['states']
        actions = batch['actions']
        rewards = batch['rewards']
        next_states = batch['next_states']
        dones = batch['dones']

        # 明确启用 autocast
        with torch.cuda.amp.autocast(enabled=self.use_scaler):   #由于cuda版本问题，cuda版本较新时，请注释此行，恢复下行
        #with torch.amp.autocast('cuda', enabled=self.use_scaler):
            q_values = self.q_net(states)
            q_values = q_values.gather(-1, actions).squeeze(-1)

            with torch.no_grad():
                q_values_main_net = self.q_net(next_states)
                next_actions = q_values_main_net.argmax(dim=-1)
                next_q_values_target_net = self.target_q_net(next_states)
                target_next_q_values = torch.gather(next_q_values_target_net, 1, next_actions.unsqueeze(-1)).squeeze(-1)
                target_q_values = rewards + self.gamma * target_next_q_values * (1 - dones)
            
            # 2. 计算逐元素的损失 (reduction='none')
            loss_fn = nn.MSELoss(reduction='none')
            elementwise_loss = loss_fn(q_values, target_q_values)
            
            # 3. 应用重要性采样 (IS) 权重
            #    is_weights_tensor 形状为 (B,)
            #    elementwise_loss 形状为 (B * N_Agents, T_Seq)
            
            #    将 is_weights 扩展以匹配 (B, N_Agents, T_Seq) -> (B * N_Agents, T_Seq)
            is_weights_expanded = is_weights_tensor.repeat_interleave(self.num_agents).unsqueeze(1)
            is_weights_expanded = is_weights_expanded.expand_as(elementwise_loss)

            #    应用权重并计算平均损失
            loss = (elementwise_loss * is_weights_expanded).mean()

        # 4. 在反向传播 *之前*，计算 TD-Errors 以更新优先级
        with torch.no_grad():
            # (B * N_Agents, T_Seq)
            td_errors = (q_values - target_q_values).abs()
            
            # 我们需要每个采样的 *回合* (B 个) 的优先级，
            # 而不是每个转换 (B * N_Agents * T_Seq 个)
            
            # 1. 变回 (B, N_Agents, T_Seq)
            td_errors_reshaped = td_errors.view(self.batch_size, self.num_agents, -1)
            
            # 2. 计算每个回合的平均 TD-Error (在 N_Agents 和 T_Seq 维度上取平均)
            episode_priorities = td_errors_reshaped.mean(dim=(1, 2)).cpu().numpy()

        # 5. 更新缓冲区中的优先级
        self.replay_buffer.update_priorities(indices, episode_priorities)


        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        if self.step_count % self.update_target_steps == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())

        # Epsilon 衰减现在由 meta-agent 控制，

        self.step_count += 1
        return loss.item()

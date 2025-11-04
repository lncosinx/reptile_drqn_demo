import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import copy
import random
import pogema
from collections import deque
import time
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_tasks_per_set = 20000 # 每个任务集中的任务数量

# 定义任务集(以每个任务集的参数生成num_tasks_per_set个任务)
task_sets_config = {
    "T1":   {"num_targets": 2, "num_agents": 1, "density": 0.1, "width": 12, "height": 12, "obs_radius": 5},
    "T2":   {"num_targets": 4, "num_agents": 1, "density": 0.15, "width": 16, "height": 16, "obs_radius": 5},
    "T3":   {"num_targets": 6, "num_agents": 1, "density": 0.2, "width": 20, "height": 20, "obs_radius": 5},
    "T4":   {"num_targets": 8, "num_agents": 1, "density": 0.25, "width": 24, "height": 24, "obs_radius": 5},
    "T5":   {"num_targets": 4, "num_agents": 1, "density": 0.3, "width": 16, "height": 16, "obs_radius": 5},
}

# 定义任务环境类
class task_environment:
    def __init__(self, config, num_tasks, pogema):
        self.config = config
        self.pogema = pogema
        self.num_tasks = num_tasks
        self.num_targets = config['num_targets']
        self.num_agents = config['num_agents']
        self.density = config['density']
        self.width = config['width']
        self.height = config['height']
        self.obs_radius = config['obs_radius']
    
    # 方便记录生成的随机种子,以便复现
    def seed_generator(self):
        seeds = random.sample(range(1, 100000), self.num_tasks)
        return seeds
    
    # 生成任务环境的生成器
    def sample_task(self):
        envs = {}
        for seed in self.seed_generator():
            config = self.pogema.GridConfig(
                num_agents=1,
                width=self.width,
                height=self.height,
                density=self.density,
                seed=seed,
                max_episode_steps=self.width * self.height,
                obs_radius=self.obs_radius,
                on_target='finish',
            )
            env = self.pogema.pogema_v0(grid_config=config)
            envs[seed] = env
        return envs
    @classmethod
    def create_task_sets(cls, configs,num_tasks, pogema):
        task_set_dict = {}
        for task_name, params in configs.items():
            task_env = cls(params, num_tasks, pogema)
            task_set_dict[task_name] = task_env.sample_task()
        return task_set_dict

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

        # 计算CNN输出的特征数量
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_channels, input_shape[1], input_shape[2])
            cnn_output_features = self.cnn(dummy_input).shape[1]

        # 定义LSTM层
        self.lstm_hidden_size = 512
        self.lstm = nn.LSTM(
            input_size = cnn_output_features, 
            hidden_size = self.lstm_hidden_size, 
            batch_first = True  # 确保输入形状是 (B, T, Features)
        )

        # 定义全连接层
        self.fc_layers = nn.Sequential(
            nn.Linear(self.lstm_hidden_size, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def forward(self, x, hidden_state = None):
        # x = torch.Tensor of shape (B, T, C, H, W) or (B, C, H, W)
        # 输入 x 的形状: (B, T, C, H, W)or((B, C, H, W)
        # B = Batch Size(智能体数量), T = Sequence Length (时间步长，在训练阶段为1，经验回放阶段等于采样的序列长度),
        # C = Channels, H = Height, W = Width -> input_shape
        if x.dim() == 5: # 经验回放阶段: (B, T, C, H, W)
            B, T, C, H, W = x.shape
        else: # 训练阶段: (B, C, H, W)
            B, C, H, W = x.shape
            T = 1
            x = x.unsqueeze(1) # 增加时间维度 -> (B, 1, C, H, W)

        # 通过CNN层
        cnn_in = x.view(B * T, C, H, W)  # 将时间步长和批次合并以通过CNN
        cnn_out = self.cnn(cnn_in)  # 通过CNN层 cnn_out: (B*T, Features)

        # 通过LSTM层
        lstm_in = cnn_out.view(B, T, -1)  # 重塑为 (B, T, Features) 以通过LSTM
        lstm_out, new_hidden_state = self.lstm(lstm_in, hidden_state)  # lstm_out: (B, T, Hidden Size)

        # 通过全连接层
        fc_in = lstm_out.contiguous().view(B * T, -1)  # 展平为 (B*T, Hidden Size)
        fc_out = self.fc_layers(fc_in)  # fc_out: (B*T, num_actions)
        qvalues = fc_out.view(B, T, -1)  # 重塑为 (B, T, num_actions)

        if T == 1:
            qvalues = qvalues.squeeze(1)  # 如果时间步长为1，移除时间维度，输出形状为 (B, num_actions)  
        return qvalues, new_hidden_state
    
#经验回放池
class ReplayBuffer:
    def __init__(self, capacity, seq_len, num_agents, obs_shape, device):
        self.capacity = capacity
        self.seq_len = seq_len
        self.num_agents = num_agents
        self.obs_shape = obs_shape
        self.device = device
        # 使用 deque (双端队列) 作为缓冲区，它在达到容量时会自动丢弃旧数据
        self.buffer = deque(maxlen=capacity) 

    def push(self, episode_data):
        """
            episode_data (dict): 包含以下键的字典:
                'states': list of np.array, 每个元素形状为 (B, N, C, H, W)
                'actions': list of int, 每个元素形状为 (B, N,)
                'rewards': list of float, 每个元素形状为 (B, N,)
                'next_states': list of np.array, 每个元素形状为 (B, N, C, H, W)
                'dones': list of bool, 每个元素形状为 (B, N,)
        """
        # 确保 episode 至少有序列长度那么长（或处理更短的 episode）
        if len(episode_data['states']) < self.seq_len:
            return

        self.buffer.append(episode_data)

    def sample(self, batch_size):
        # 检查是否有足够的 episode 可供采样
        if len(self.buffer) < batch_size:
            if not self.buffer:
                return None
            sampled_episodes = random.choices(self.buffer, k=batch_size) # 有放回采样
        else:
            sampled_episodes = random.sample(self.buffer, batch_size)

        # 初始化列表来收集序列
        batch_states = []
        batch_actions = []
        batch_rewards = []
        batch_next_states = []
        batch_dones = []

        for episode in sampled_episodes:
            # 随机选择一个起始点
            start_idx = random.randint(0, len(episode['states']) - self.seq_len)
            end_idx = start_idx + self.seq_len

            # 截取序列 (注意：这里截取的是列表的列表)
            batch_states.append(episode['states'][start_idx:end_idx])
            batch_actions.append(episode['actions'][start_idx:end_idx])
            batch_rewards.append(episode['rewards'][start_idx:end_idx])
            batch_next_states.append(episode['next_states'][start_idx:end_idx])
            batch_dones.append(episode['dones'][start_idx:end_idx])

        # --- 关键：维度转换 ---
        # 1. 转换为 Numpy 数组，方便进行维度转换
        states_np = np.array(batch_states, dtype=np.float32)            # (B, T, N, C, H, W)
        actions_np = np.array(batch_actions, dtype=np.int64)            # (B, T, N)
        rewards_np = np.array(batch_rewards, dtype=np.float32)          # (B, T, N)
        next_states_np = np.array(batch_next_states, dtype=np.float32)  # (B, T, N, C, H, W)
        dones_np = np.array(batch_dones, dtype=np.float32)              # (B, T, N)

        # 2. 交换 B 和 N 维度
        states_np = states_np.transpose(0, 2, 1, 3, 4, 5)               # (B, N, T, C, H, W)
        next_states_np = next_states_np.transpose(0, 2, 1, 3, 4, 5)     # (B, N, T, C, H, W)
        actions_np = actions_np.transpose(0, 2, 1)                      # (B, N, T)
        rewards_np = rewards_np.transpose(0, 2, 1)                      # (B, N, T)
        dones_np = dones_np.transpose(0, 2, 1)                          # (B, N, T)

        # 3. 合并 B 和 N 维度
        #    形状: (B*N, T, ...)
        batch_states_tensor = torch.tensor(states_np.reshape(-1, self.seq_len, *self.obs_shape), device=self.device)            # (B*N, T, C, H, W)
        batch_next_states_tensor = torch.tensor(next_states_np.reshape(-1, self.seq_len, *self.obs_shape), device=self.device)  # (B*N, T, C, H, W)
        batch_actions_tensor = torch.tensor(actions_np.reshape(-1, self.seq_len), device=self.device)                           # (B*N, T)  
        batch_rewards_tensor = torch.tensor(rewards_np.reshape(-1, self.seq_len), device=self.device)                           # (B*N, T)
        batch_dones_tensor = torch.tensor(dones_np.reshape(-1, self.seq_len), device=self.device)                               # (B*N, T)

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
        self.num_agents = num_agents
        self.state_shape = state_shape
        self.num_actions = num_actions
        self.lr = lr
        self.q_net = CnnQnet(state_shape, num_actions).to(device)
        self.target_q_net = CnnQnet(state_shape, num_actions).to(device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.lr)
        self.replay_buffer = replay_buffer
        self.gamma = 0.99
        self.batch_size = 32
        self.epsilon = 1.0
        self.epsilon_decay = 0.99995
        self.epsilon_min = 0.1
        self.update_target_steps = 100
        self.step_count = 0

    def select_actions(self, states, current_hidden_state): #states: (N, C, H, W)
        self.q_net.eval()
        with torch.no_grad():
            q_values, new_hidden_state = self.q_net(states, current_hidden_state)
        self.q_net.train()

        greedy_actions = q_values.argmax(dim=-1)  # (N,)
        random_actions = torch.randint(0, self.num_actions, (states.shape[0],), device=device)  # (N,)

        is_random = torch.rand(states.shape[0], device=device) < self.epsilon
        actions = torch.where(is_random, random_actions, greedy_actions)  # (N,)
        return actions.cpu().numpy(), new_hidden_state
    
    def train(self):
        batch = self.replay_buffer.sample(self.batch_size)
        if batch is None:
            return None
        states = batch['states']          # (B*N, T, C, H, W)
        actions = batch['actions']        # (B*N, T)
        rewards = batch['rewards']        # (B*N, T)
        next_states = batch['next_states']# (B*N, T, C, H, W)
        dones = batch['dones']            # (B*N, T)

        # 计算当前 Q 值
        q_values, _ = self.q_net(states)                                   # (B*N, T, num_actions)
        q_values = q_values.gather(-1, actions.unsqueeze(-1)).squeeze(-1)  # (B*N, T)

        # 计算目标 Q 值
        with torch.no_grad():
            #使用主网络找到最佳动作
            q_values_main_net, _ = self.q_net(next_states)          # (B*N, T, num_actions)
            next_actions = q_values_main_net.argmax(dim=-1)         # (B*N, T)
            #用目标网络评估这些动作的价值
            next_q_values_target_net, _ = self.target_q_net(next_states)  # (B*N, T, num_actions)
            # 使用 gather 挑选出价值
            target_next_q_values = torch.gather(next_q_values_target_net, 2, next_actions.unsqueeze(-1)).squeeze(-1)  # (B*N, T)
            #贝尔曼方程计算目标 Q 值
            target_q_values = rewards + self.gamma * target_next_q_values * (1 - dones)  # (B*N, T)
        # 计算损失
        loss = nn.MSELoss()(q_values, target_q_values.detach())

        # 优化网络
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 更新目标网络
        if self.step_count % self.update_target_steps == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())

        # 衰减 epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        self.step_count += 1

        return loss.item()

# 定义 Reptile 元学习算法
class Reptile:
    def __init__(self, all_task_sets, state_shape, num_actions, num_agents, task_sets, model_path=None):
        self.all_task_sets = all_task_sets
        self.state_shape = state_shape # (C, H, W)
        self.num_actions = num_actions
        self.num_agents = num_agents
        self.task_sets = task_sets

        # --- 元学习超参数 ---
        self.meta_iterations = 20000
        self.meta_lr = 0.001       # 外循环学习率

        # --- 内循环超参数 ---
        self.inner_lr = 0.0001     # 内循环学习率 (DRQNAgent 的 lr)
        self.inner_steps = 32      # k: 内循环中的 *梯度下降步数*
        self.episodes_per_task = 10 # 每个任务收集多少个 episode 来填充 buffer
        
        # --- 回放池参数 ---
        self.replay_buffer_capacity = 1000 # 减小容量，使其特定于任务
        self.seq_len = 8           # 减小序列长度
        self.batch_size = 4        # 减小批次大小

        # 1. 创建一个“模板”回放池。这实际上不会被 meta_agent 使用。
        template_buffer = ReplayBuffer(self.replay_buffer_capacity, self.seq_len, num_agents, state_shape, device)
        
        # 2. 创建元智能体 (meta-agent)
        # 它持有“元权重” (phi)
        self.meta_agent = DRQNAgent(num_agents, state_shape, num_actions, template_buffer, lr=self.inner_lr)
        # 将 batch_size 同步到 meta_agent，以便 deepcopy
        self.meta_agent.batch_size = self.batch_size

         # [新增] 用于恢复训练的变量
        self.start_meta_iter = 0
        self.loaded_meta_agent_total_steps = 0

        # [新增] 检查点加载逻辑
        if model_path and os.path.exists(model_path):
            print(f"正在从 '{model_path}' 加载检查点...")
            try:
                checkpoint = torch.load(model_path, map_location=device)
                
                # 检查是新格式（字典）还是旧格式（仅 state_dict）
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    self.meta_agent.q_net.load_state_dict(checkpoint['model_state_dict'])
                    self.meta_agent.target_q_net.load_state_dict(checkpoint['model_state_dict'])
                    self.meta_agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    self.meta_agent.epsilon = checkpoint['epsilon']
                    self.loaded_meta_agent_total_steps = checkpoint.get('meta_agent_total_steps', 0)
                    self.start_meta_iter = checkpoint.get('current_meta_iter', 0) + 1
                    print(f"成功加载检查点。将从 Iter {self.start_meta_iter} 和 Epsilon {self.meta_agent.epsilon:.4f} 处恢复。")
                else:
                    # 兼容只存了权重的旧格式
                    self.meta_agent.q_net.load_state_dict(checkpoint)
                    self.meta_agent.target_q_net.load_state_dict(checkpoint)
                    print(f"警告：成功加载旧格式模型权重。Epsilon 和优化器将重置。")

            except Exception as e:
                print(f"加载模型 '{model_path}' 失败: {e}。将从头开始训练。")
        else:
            print("未找到模型路径或 model_path 为 None。将从头开始训练。")


        self.print_freq = 100 # 每多少次 meta iteration 打印一次日志
        self.save_freq = 1000
        # 模型保存路径
        self.model_save_path_interrupt = 'reptile_drqn_meta_agent_interrupt.pth'
        self.model_save_path_final = 'reptile_drqn.pth'


    def _save_checkpoint(self, path, meta_iter, meta_agent_total_steps):
        try:
            checkpoint = {
                'model_state_dict': self.meta_agent.q_net.state_dict(),
                'optimizer_state_dict': self.meta_agent.optimizer.state_dict(),
                'epsilon': self.meta_agent.epsilon,
                'meta_agent_total_steps': meta_agent_total_steps,
                'current_meta_iter': meta_iter 
            }
            torch.save(checkpoint, path)
            print(f"--- 检查点已保存到 {path} (Iter: {meta_iter + 1}) ---")
        except Exception as e:
            print(f"警告：保存检查点到 {path} 失败: {e}")

    def meta_train(self):
        print(f"开始 Reptile 元学习训练 (共 {self.meta_iterations} 次迭代)...")
        start_time = time.time()
        total_episodes_run = 0
        total_steps_collected = 0
        # 修正：记录所有内部训练步骤的损失，而不仅仅是每个 meta iter 的平均值
        all_inner_step_losses = []
        # 修正：跟踪 meta_agent 的总训练步数，用于 epsilon 衰减
        meta_agent_total_steps = self.loaded_meta_agent_total_steps

        try:
            # [修改] 循环从 start_meta_iter 开始
            for meta_iter in range(self.start_meta_iter, self.meta_iterations):
                task_set_name = random.choice(list(self.task_sets.keys()))
                task_set = self.all_task_sets[task_set_name]
                if not task_set:
                    print(f"警告：任务集 {task_set_name} 为空，跳过 Meta Iter {meta_iter + 1}。")
                    continue
                task_env = random.choice(list(task_set.values()))

                # 3. 创建 task_agent, 复制 meta_agent 权重和 *当前的* epsilon
                task_agent = copy.deepcopy(self.meta_agent)
                task_agent.q_net.lstm.flatten_parameters()
                task_agent.target_q_net.lstm.flatten_parameters()
                task_agent.replay_buffer = ReplayBuffer(self.replay_buffer_capacity, self.seq_len, self.num_agents, self.state_shape, device)
                # 修正：确保 task_agent 使用 meta_agent 当前的 epsilon 开始
                task_agent.epsilon = self.meta_agent.epsilon

                # --- 内循环 ---
                current_task_episodes = 0
                current_task_steps = 0
                current_task_rewards = []

                # 阶段 1: 收集经验 (保持不变)
                while current_task_episodes < self.episodes_per_task:
                    ep_states, ep_actions, ep_rewards, ep_next_states, ep_dones = [], [], [], [], []
                    obs, info = task_env.reset()
                    current_hidden_state = None
                    terminated = [False] * self.num_agents
                    truncated = [False] * self.num_agents
                    episode_reward = 0
                    episode_len = 0

                    while not (all(terminated) or all(truncated)):
                        obs_np = np.array(obs)
                        states_tensor = torch.tensor(obs_np, dtype=torch.float32, device=device)
                        # 使用 task_agent 的 epsilon 进行探索
                        actions_np, new_hidden_state = task_agent.select_actions(states_tensor, current_hidden_state)
                        current_hidden_state = new_hidden_state
                        next_obs, rewards, terminated, truncated, info = task_env.step(actions_np)
                        next_obs_np = np.array(next_obs)
                        ep_states.append(obs_np)
                        ep_actions.append(actions_np)
                        ep_rewards.append(rewards)
                        ep_next_states.append(next_obs_np)
                        ep_dones.append(terminated)
                        obs = next_obs
                        episode_reward += rewards[0] if rewards else 0
                        episode_len += 1
                        current_task_steps += 1

                    task_agent.replay_buffer.push({
                        'states': ep_states, 'actions': ep_actions, 'rewards': ep_rewards,
                        'next_states': ep_next_states, 'dones': ep_dones
                    })
                    current_task_episodes += 1
                    current_task_rewards.append(episode_reward)

                total_episodes_run += current_task_episodes
                total_steps_collected += current_task_steps

                # --- 阶段 2: 训练 ---
                inner_losses_current_iter = [] # 只记录当前 meta iter 的损失用于平均
                if len(task_agent.replay_buffer) >= task_agent.batch_size:
                    for i in range(self.inner_steps):
                        loss = task_agent.train() # task_agent 训练并更新自己的网络权重
                        if loss is not None:
                            inner_losses_current_iter.append(loss)
                            all_inner_step_losses.append(loss) # 记录每一步的损失
                            # 修正：同时衰减 meta_agent 的 epsilon
                            self.meta_agent.epsilon = max(self.meta_agent.epsilon_min, self.meta_agent.epsilon * self.meta_agent.epsilon_decay)
                            meta_agent_total_steps += 1 # 增加 meta_agent 的总步数计数

                else:
                    print(f"Meta-Iter {meta_iter+1}: 收集阶段后数据仍然不足 ({len(task_agent.replay_buffer)} episodes)，跳过训练。")


                # 修正：计算当前迭代的平均损失
                avg_inner_loss_iter = np.mean(inner_losses_current_iter) if inner_losses_current_iter else 0


                # --- 元更新 ---
                if inner_losses_current_iter: # 只有在内循环成功训练后才执行元更新
                    with torch.no_grad():
                        # 这里更新的是 meta_agent 的权重
                        for meta_param, task_param in zip(self.meta_agent.q_net.parameters(), task_agent.q_net.parameters()):
                            update_direction = task_param.data - meta_param.data
                            meta_param.data += self.meta_lr * update_direction

                # --- 日志和保存 ---
                if (meta_iter + 1) % self.print_freq == 0:
                    elapsed = time.time() - start_time
                    avg_task_reward = np.mean(current_task_rewards) if current_task_rewards else 0
                    print(f"Meta Iter {meta_iter + 1}/{self.meta_iterations} | "
                        f"TaskSet: {task_set_name} | "
                        f"Avg Task Reward (Collect): {avg_task_reward:.2f} | "
                        # 修正：打印当前迭代的平均损失
                        f"Avg Inner Loss (Iter): {avg_inner_loss_iter:.4f} | "
                        # 修正：打印 meta_agent 的 epsilon
                        f"Meta Epsilon: {self.meta_agent.epsilon:.4f} | "
                        f"Total Train Steps: {meta_agent_total_steps} | " # 打印总训练步数
                        f"Time/Iter: {elapsed / self.print_freq:.2f}s")
                    start_time = time.time() # 重置计时器

                if (meta_iter + 1) % self.save_freq == 0:
                    # [修改] 调用新的保存函数
                    self._save_checkpoint(self.model_save_path_interrupt, meta_iter, meta_agent_total_steps)

        # [新增] 捕获 KeyboardInterrupt
        except KeyboardInterrupt:
            print(f"\n训练被中断。正在保存当前检查点 (Iter: {meta_iter})...")
            # [修改] 调用新的保存函数
            self._save_checkpoint(self.model_save_path_interrupt, meta_iter, meta_agent_total_steps)
            print("检查点已保存。安全退出。")
            # 在这里绘制已有的损失
            if all_inner_step_losses:
                 plt.figure(figsize=(12, 6))
                 plt.plot(all_inner_step_losses)
                 plt.title('Inner Training Step Loss (Interrupted)')
                 plt.xlabel('Inner Training Step')
                 plt.ylabel('MSE Loss')
                 plt.savefig('reptile_inner_loss_curve_interrupt.png')
                 print("中断的损失曲线图已保存。")
            return # 干净地退出函数

        # ... (训练结束后的保存和绘图保持不变) ...
        # 训练结束后保存最终模型
        print("训练完成。正在保存最终模型...")
        # [修改] 调用新的保存函数
        self._save_checkpoint(self.model_save_path_final, self.meta_iterations - 1, meta_agent_total_steps)

        # 绘制所有内部训练步骤的损失曲线
        if all_inner_step_losses:
            plt.figure(figsize=(12, 6))
            plt.plot(all_inner_step_losses)
            plt.title('Inner Training Step Loss during Meta Training')
            plt.xlabel('Inner Training Step')
            plt.ylabel('MSE Loss')
            plt.grid(True)
            # 可以添加移动平均线使曲线更平滑
            if len(all_inner_step_losses) > 100:
                moving_avg = np.convolve(all_inner_step_losses, np.ones(100)/100, mode='valid')
                plt.plot(np.arange(99, len(all_inner_step_losses)), moving_avg, label='100-step Moving Average')
                plt.legend()

            plt.savefig('reptile_inner_loss_curve.png')
            print("内部损失曲线图已保存为 reptile_inner_loss_curve.png")

# 测试类，继承CnnQnet
class TestAgent(DRQNAgent):
    def __init__(self, state_shape, num_actions, state_dict_file_path, finetune_lr, finetune_buffer_capacity, seq_len, num_agents_to_test):
        
        # [新增] 确保测试 device 也被设置
        test_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        finetune_buffer = ReplayBuffer(finetune_buffer_capacity, seq_len, num_agents_to_test, state_shape, test_device)
        
        super().__init__(
            num_agents=num_agents_to_test,
            state_shape=state_shape,
            num_actions=num_actions,
            replay_buffer=finetune_buffer,
            lr=finetune_lr
        )
        
        # 3. 加载元学习到的权重
        try:
            checkpoint = torch.load(state_dict_file_path, map_location=self.device) # 使用 super() 中设置的 device
            
            load_state_dict = None
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                load_state_dict = checkpoint['model_state_dict']
            else:
                load_state_dict = checkpoint

            self.q_net.load_state_dict(load_state_dict)
            self.target_q_net.load_state_dict(load_state_dict)
            print(f"成功加载元学习权重 from '{state_dict_file_path}'")
        except Exception as e:
            print(f"加载模型权重 '{state_dict_file_path}' 失败: {e}")

    # ... fine_tune_on_task (保持不变) ...
    def fine_tune_on_task(self, task_env, num_episodes=20, num_steps_per_train=32):
        print(f"开始在新任务上微调 {num_episodes} 个回合...")
        self.epsilon = 0.5 # 微调初始 epsilon
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.99
        total_steps_trained = 0
        all_finetune_losses = []

        for ep in range(num_episodes):
            ep_states, ep_actions, ep_rewards, ep_next_states, ep_dones = [], [], [], [], []
            obs, info = task_env.reset()
            current_hidden_state = None
            terminated = [False] * self.num_agents
            truncated = [False] * self.num_agents
            ep_len = 0
            ep_reward = 0

            while not (all(terminated) or all(truncated)):
                obs_np = np.array(obs)
                states_tensor = torch.tensor(obs_np, dtype=torch.float32, device=self.device)
                actions_np, new_hidden_state = self.select_actions(states_tensor, current_hidden_state)
                current_hidden_state = new_hidden_state

                next_obs, rewards, terminated, truncated, info = task_env.step(actions_np)
                next_obs_np = np.array(next_obs)

                ep_states.append(obs_np)
                ep_actions.append(actions_np)
                ep_rewards.append(rewards)
                ep_next_states.append(next_obs_np)
                ep_dones.append(terminated)
                obs = next_obs
                ep_len += 1
                ep_reward += rewards[0] if rewards else 0

            self.replay_buffer.push({
                'states': ep_states, 'actions': ep_actions, 'rewards': ep_rewards,
                'next_states': ep_next_states, 'dones': ep_dones
            })
            print(f"  微调回合 {ep+1}/{num_episodes} 完成 (长度: {ep_len}, 奖励: {ep_reward:.2f})。Buffer: {len(self.replay_buffer)}")

            if len(self.replay_buffer) >= self.batch_size:
                for _ in range(num_steps_per_train): 
                    loss = self.train()
                    if loss is not None:
                         all_finetune_losses.append(loss)
                         total_steps_trained += 1

        avg_finetune_loss = np.mean(all_finetune_losses) if all_finetune_losses else 0
        print(f"微调完成。共训练 {total_steps_trained} 步。平均损失: {avg_finetune_loss:.4f}")

    # ... evaluate (保持不变) ...
    def evaluate(self, task_env, render_animation=False, animation_filename="task_evaluation.svg"):
        print(f"开始评估 (零探索)...")
        
        if render_animation:
            env = pogema.AnimationMonitor(task_env)
        else:
            env = task_env

        states, info = env.reset()
        terminated = [False] * self.num_agents
        truncated = [False] * self.num_agents
        total_rewards = np.zeros(self.num_agents)
        current_hidden_state = None
        step_count = 0

        original_epsilon = self.epsilon
        self.epsilon = 0.0 
        self.q_net.eval() 

        with torch.no_grad():
            while not (all(terminated) or all(truncated)):
                states_tensor = torch.tensor(np.array(states), dtype=torch.float32, device=self.device)
                
                actions_np, new_hidden_state = self.select_actions(states_tensor, current_hidden_state)
                current_hidden_state = new_hidden_state
                next_states, rewards, terminated, truncated, info = env.step(actions_np)
                total_rewards += np.array(rewards)
                states = next_states
                step_count += 1
        
        self.q_net.train() 
        self.epsilon = original_epsilon 
        
        print(f"评估完成! 总步数: {step_count}, 总奖励: {total_rewards}")

        if render_animation:
            try:
                env.save_animation(animation_filename)
                print(f"动画已保存至 {animation_filename}\n")
            except Exception as e:
                print(f"保存动画时出错: {e}")
        
        return total_rewards, step_count
    
    # ... task_env (保持不变) ...
    def task_env(self, task_config):
        config = pogema.GridConfig(
            num_agents=task_config["num_agents"],
            width=task_config["width"],
            height=task_config["height"],
            density=task_config["density"],
            seed=1031,
            max_episode_steps=task_config["width"] * task_config["height"],
            obs_radius=task_config["obs_radius"],
            on_target='finish',
        )
        env = pogema.pogema_v0(grid_config=config)
        return env
    
"""
        animation_filename = "task_test.svg"
        env.pogema.AnimationMonitor.save_animation(animation_filename)
        print(f"动画已保存至 {animation_filename}\n")
"""
if __name__ == '__main__':
    generated_task = task_environment.create_task_sets(task_sets_config,num_tasks_per_set, pogema)

    model_path = 'reptile_drqn_meta_agent_interrupt.pth'if os.path.exists('reptile_drqn_meta_agent_interrupt.pth') else None

    reptile = Reptile(
    all_task_sets=generated_task,
    state_shape=(3, 11, 11),
    num_actions=5,
    num_agents=1,
    task_sets=task_sets_config,
    model_path= model_path,
    )
    reptile.meta_train()

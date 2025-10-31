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
# [修改] 导入 torch.multiprocessing
import torch.multiprocessing as mp

# [修改] 不要在全局定义 device。
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 我们将在主进程和 worker 进程中分别定义它。

num_tasks_per_set = 20000 # 每个任务集中的任务数量

# 定义任务集(以每个任务集的参数生成num_tasks_per_set个任务)
task_sets_config = {
    "T1":   {"num_targets": 2, "num_agents": 1, "density": 0.1, "width": 12, "height": 12, "obs_radius": 5},
    "T2":   {"num_targets": 4, "num_agents": 1, "density": 0.15, "width": 16, "height": 16, "obs_radius": 5},
    "T3":   {"num_targets": 6, "num_agents": 1, "density": 0.2, "width": 20, "height": 20, "obs_radius": 5},
    "T4":   {"num_targets": 8, "num_agents": 1, "density": 0.25, "width": 24, "height": 24, "obs_radius": 5},
    "T5":   {"num_targets": 4, "num_agents": 1, "density": 0.3, "width": 16, "height": 16, "obs_radius": 5},
}

# -------------------------------------------------------------------
# [新增] 并行工作函数 (必须在全局范围)
# -------------------------------------------------------------------
def reptile_worker(args):
    """
    在单独的进程中执行完整的内部循环（数据收集 + 训练）。
    
    Args:
        args (tuple): 包含:
            task_env (pogema.env): 要运行的任务环境实例。
            meta_weights_cpu (dict): 元智能体q_net的 state_dict (在CPU上)。
            meta_epsilon (float): 当前的元 epsilon。
            config (dict): 包含所有超参数的字典。
    """
    try:
        task_env, meta_weights_cpu, meta_epsilon, config = args
        
        # 1. [关键] 在工作进程中定义 device
        worker_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 2. 重新创建 task_agent
        # (需要 ReplayBuffer, CnnQnet, DRQNAgent 类在全局范围内可用)
        task_buffer = ReplayBuffer(
            config['replay_buffer_capacity'],
            config['seq_len'],
            config['num_agents'],
            config['state_shape'],
            worker_device
        )
        task_agent = DRQNAgent(
            config['num_agents'],
            config['state_shape'],
            config['num_actions'],
            task_buffer,
            config['inner_lr']
        )
        task_agent.batch_size = config['batch_size']
        
        # 3. 加载元权重并移动到 worker 的 GPU
        task_agent.q_net.load_state_dict(meta_weights_cpu)
        task_agent.target_q_net.load_state_dict(meta_weights_cpu)
        task_agent.q_net.to(worker_device)
        task_agent.target_q_net.to(worker_device)
        task_agent.q_net.lstm.flatten_parameters()
        task_agent.target_q_net.lstm.flatten_parameters()

        # 4. 设置当前的 epsilon
        task_agent.epsilon = meta_epsilon
        task_agent.epsilon_min = config['epsilon_min']
        task_agent.epsilon_decay = config['epsilon_decay'] # worker 内部的探索会衰减

        # --- 阶段 1: 收集经验 ---
        current_task_episodes = 0
        current_task_rewards = []
        
        while current_task_episodes < config['episodes_per_task']:
            ep_states, ep_actions, ep_rewards, ep_next_states, ep_dones = [], [], [], [], []
            obs, info = task_env.reset()
            current_hidden_state = None
            terminated = [False] * config['num_agents']
            truncated = [False] * config['num_agents']
            episode_reward = 0

            while not (all(terminated) or all(truncated)):
                obs_np = np.array(obs)
                states_tensor = torch.tensor(obs_np, dtype=torch.float32, device=worker_device)
                
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

            task_agent.replay_buffer.push({
                'states': ep_states, 'actions': ep_actions, 'rewards': ep_rewards,
                'next_states': ep_next_states, 'dones': ep_dones
            })
            current_task_episodes += 1
            current_task_rewards.append(episode_reward)

        # --- 阶段 2: 训练 ---
        inner_losses = []
        steps_trained = 0
        
        if len(task_agent.replay_buffer) > 0:
            for i in range(config['inner_steps']):
                # 注意：task_agent.train() 会衰减它自己的 epsilon
                # 我们只关心它训练了多少步
                loss = task_agent.train() 
                if loss is not None:
                    inner_losses.append(loss)
                    steps_trained += 1
        
        avg_loss = np.mean(inner_losses) if inner_losses else 0
        avg_reward = np.mean(current_task_rewards) if current_task_rewards else 0

        # --- 阶段 3: 计算增量 (Deltas) 并移至 CPU ---
        with torch.no_grad():
            delta = {}
            # 重新加载 meta_weights 到 worker_device 以进行比较
            meta_weights_gpu = {k: v.to(worker_device) for k, v in meta_weights_cpu.items()}

            for name, task_param in task_agent.q_net.named_parameters():
                # 计算增量 (new - old)
                param_delta = task_param.data - meta_weights_gpu[name]
                # [关键] 将增量移回 CPU 以便返回
                delta[name] = param_delta.cpu()
        
        # 返回 (增量, 平均损失, 训练步数, 平均奖励)
        return (delta, avg_loss, steps_trained, avg_reward)

    except Exception as e:
        print(f"[Worker Error] 工作进程失败: {e}")
        import traceback
        traceback.print_exc()
        return (None, 0, 0, 0)

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
    
    # [修改] 并行化任务创建 (方向 1)
    # 我们也在这里应用并行化，因为 100,000 个任务串行创建会很慢
    
    # 辅助函数，必须是可 pickle 的 (顶层或静态)
    @staticmethod
    def _create_single_env(config_dict):
        try:
            config = pogema.GridConfig(
                num_agents=config_dict['num_agents'],
                width=config_dict['width'],
                height=config_dict['height'],
                density=config_dict['density'],
                seed=config_dict['seed'],
                max_episode_steps=config_dict['width'] * config_dict['height'],
                obs_radius=config_dict['obs_radius'],
                on_target='finish',
            )
            env = pogema.pogema_v0(grid_config=config)
            return (config_dict['seed'], env)
        except Exception as e:
            print(f"创建环境失败 (Seed: {config_dict.get('seed')}): {e}")
            return (config_dict.get('seed'), None)

    def sample_task(self):
        seeds = self.seed_generator()
        
        config_params_list = []
        for seed in seeds:
            params = self.config.copy()
            params['seed'] = seed
            config_params_list.append(params)

        envs = {}
        
        print(f"正在为 {self.width}x{self.height} (density {self.density}) 并行创建 {self.num_tasks} 个环境...")
        start_time = time.time()
        
        # [修改] 使用 multiprocessing Pool
        # 让 Python 自动决定进程数
        with mp.Pool(processes=os.cpu_count()) as pool:
            results = pool.map(task_environment._create_single_env, config_params_list)
        
        # 过滤掉创建失败的环境
        envs = {seed: env for seed, env in results if env is not None}
        
        elapsed = time.time() - start_time
        print(f"创建 {len(envs)} 个环境完成，耗时: {elapsed:.2f}s")
        return envs

    @classmethod
    def create_task_sets(cls, configs,num_tasks, pogema):
        task_set_dict = {}
        for task_name, params in configs.items():
            task_env = cls(params, num_tasks, pogema)
            # sample_task 现在是并行的了
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
        # [修改] device 在这里定义，从主进程或 worker 传入
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.scaler = torch.cuda.amp.GradScaler()
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
        self.epsilon_decay = 0.9999995
        self.epsilon_min = 0.1
        self.update_target_steps = 100
        self.step_count = 0

    def select_actions(self, states, current_hidden_state): #states: (N, C, H, W)
        self.q_net.eval()
        with torch.no_grad():
            q_values, new_hidden_state = self.q_net(states, current_hidden_state)
        self.q_net.train()

        greedy_actions = q_values.argmax(dim=-1)  # (N,)
        random_actions = torch.randint(0, self.num_actions, (states.shape[0],), device=self.device)  # (N,)

        is_random = torch.rand(states.shape[0], device=self.device) < self.epsilon
        actions = torch.where(is_random, random_actions, greedy_actions)  # (N,)
        return actions.cpu().numpy(), new_hidden_state
    
    def train(self):
        batch = self.replay_buffer.sample(self.batch_size)
        if batch is None:
            return None
        
        # batch 已经在创建时被放到了正确的 device 上
        states = batch['states']          # (B*N, T, C, H, W)
        actions = batch['actions']        # (B*N, T)
        rewards = batch['rewards']        # (B*N, T)
        next_states = batch['next_states']# (B*N, T, C, H, W)
        dones = batch['dones']            # (B*N, T)

        with torch.cuda.amp.autocast():
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
            loss = nn.MSELoss()(q_values, target_q_values)

        # 优化网络 (使用 scaler)
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        # 更新目标网络
        if self.step_count % self.update_target_steps == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())

        # [修改] worker 不应该衰减 meta-epsilon
        # 它只衰减自己的 epsilon，这对探索是必要的
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        self.step_count += 1

        return loss.item()

# 定义 Reptile 元学习算法
class Reptile:
    def __init__(self, all_task_sets, state_shape, num_actions, num_agents, task_sets, 
                 parallel_batch_size, total_meta_iterations, model_path=None):
        
        # [新增] 主进程 device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[Main] 主进程使用 device: {self.device}")
        
        self.all_task_sets = all_task_sets
        self.state_shape = state_shape # (C, H, W)
        self.num_actions = num_actions
        self.num_agents = num_agents
        self.task_sets = task_sets

        # --- [修改] 元学习超参数 ---
        self.parallel_batch_size = parallel_batch_size # B: 每次并行运行多少个任务
        self.total_meta_iterations = total_meta_iterations # 目标总任务数
        self.meta_batches = self.total_meta_iterations // self.parallel_batch_size # 主循环次数
        self.meta_lr = 0.001       # 外循环学习率

        # --- [修改] 内循环超参数 (将打包发送给 worker) ---
        self.inner_lr = 0.0001     # 内循环学习率 (DRQNAgent 的 lr)
        self.inner_steps = 32      # k: 内循环中的 *梯度下降步数*
        self.episodes_per_task = 10 # 每个任务收集多少个 episode 来填充 buffer
        
        # --- [修改] 回放池参数 (将打包发送给 worker) ---
        self.replay_buffer_capacity = 1000 # 减小容量，使其特定于任务
        self.seq_len = 24           # 减小序列长度
        self.batch_size = 128        # 减小批次大小

        # 1. 创建一个“模板”回放池。这实际上不会被 meta_agent 使用。
        template_buffer = ReplayBuffer(1, 1, num_agents, state_shape, self.device)
        
        # 2. 创建元智能体 (meta-agent)
        # 它持有“元权重” (phi)，并生活在主进程的 device 上
        self.meta_agent = DRQNAgent(num_agents, state_shape, num_actions, template_buffer, lr=self.inner_lr)
        # 将 batch_size 同步到 meta_agent (虽然它不训练，但 worker 会用到)
        self.meta_agent.batch_size = self.batch_size

         # [新增] 用于恢复训练的变量
        self.start_meta_iter = 0
        self.loaded_meta_agent_total_steps = 0

        # [新增] 检查点加载逻辑 (不变)
        if model_path and os.path.exists(model_path):
            print(f"正在从 '{model_path}' 加载检查点...")
            try:
                # [修改] 确保加载到主进程的 device
                checkpoint = torch.load(model_path, map_location=self.device)
                
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    self.meta_agent.q_net.load_state_dict(checkpoint['model_state_dict'])
                    self.meta_agent.target_q_net.load_state_dict(checkpoint['model_state_dict'])
                    self.meta_agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    self.meta_agent.epsilon = checkpoint['epsilon']
                    self.loaded_meta_agent_total_steps = checkpoint.get('meta_agent_total_steps', 0)
                    self.start_meta_iter = checkpoint.get('current_meta_iter', 0) + 1
                    print(f"成功加载检查点。将从 Iter {self.start_meta_iter} 和 Epsilon {self.meta_agent.epsilon:.4f} 处恢复。")
                else:
                    self.meta_agent.q_net.load_state_dict(checkpoint)
                    self.meta_agent.target_q_net.load_state_dict(checkpoint)
                    print(f"警告：成功加载旧格式模型权重。Epsilon 和优化器将重置。")

            except Exception as e:
                print(f"加载模型 '{model_path}' 失败: {e}。将从头开始训练。")
        else:
            print("未找到模型路径或 model_path 为 None。将从头开始训练。")

        # [修改] 基于 "批次" 的日志和保存
        self.print_freq_batch = 10 # 每 10 个 *批次* 打印一次
        self.save_freq_batch = 100 # 每 100 个 *批次* 保存一次
        
        # 模型保存路径 (不变)
        self.model_save_path_interrupt = 'reptile_drqn_meta_agent_interrupt.pth'
        self.model_save_path_final = 'reptile_drqn.pth'
        
        # [新增] 打包 worker 所需的超参数
        self.worker_config = {
            'replay_buffer_capacity': self.replay_buffer_capacity,
            'seq_len': self.seq_len,
            'num_agents': self.num_agents,
            'state_shape': self.state_shape,
            'num_actions': self.num_actions,
            'inner_lr': self.inner_lr,
            'batch_size': self.batch_size,
            'episodes_per_task': self.episodes_per_task,
            'inner_steps': self.inner_steps,
            'epsilon_min': self.meta_agent.epsilon_min,
            'epsilon_decay': self.meta_agent.epsilon_decay,
            'gamma': self.meta_agent.gamma,
            'update_target_steps': self.meta_agent.update_target_steps,
        }
        
        # [新增] 创建进程池
        print(f"正在创建 {self.parallel_batch_size} 个工作进程的进程池...")
        self.pool = mp.Pool(self.parallel_batch_size)
        print("进程池创建完毕。")


    def _save_checkpoint(self, path, meta_iter, meta_agent_total_steps):
        # [修改] 保存前确保模型在 CPU 上，或使用 get_state_dict
        try:
            checkpoint = {
                'model_state_dict': self.meta_agent.q_net.state_dict(),
                'optimizer_state_dict': self.meta_agent.optimizer.state_dict(),
                'epsilon': self.meta_agent.epsilon,
                'meta_agent_total_steps': meta_agent_total_steps,
                'current_meta_iter': meta_iter # 保存的是已完成的任务迭代次数
            }
            torch.save(checkpoint, path)
            print(f"--- 检查点已保存到 {path} (Iter: {meta_iter + 1}) ---")
        except Exception as e:
            print(f"警告：保存检查点到 {path} 失败: {e}")

    # [修改] meta_train 被完全重构
    def meta_train(self):
        print(f"开始 Reptile 元学习训练 (共 {self.meta_batches} 个批次, {self.parallel_batch_size} 个任务/批次)...")
        start_time = time.time()
        meta_agent_total_steps = self.loaded_meta_agent_total_steps
        
        # 记录所有批次的平均损失
        all_batch_losses = []
        all_batch_rewards = []
        
        # 计算起始批次
        start_batch = self.start_meta_iter // self.parallel_batch_size

        try:
            for meta_batch in range(start_batch, self.meta_batches):
                
                # --- 1. 准备参数 ---
                # [关键] 将元权重复制到 CPU，以便安全地跨进程共享
                meta_weights_cpu = {k: v.cpu() for k, v in self.meta_agent.q_net.state_dict().items()}
                current_meta_epsilon = self.meta_agent.epsilon
                
                # --- 2. 采样 B 个任务 ---
                args_list = []
                tasks_in_batch_names = [] # 用于日志
                for _ in range(self.parallel_batch_size):
                    task_set_name = random.choice(list(self.task_sets.keys()))
                    task_set = self.all_task_sets[task_set_name]
                    if not task_set:
                        print(f"警告：任务集 {task_set_name} 为空，跳过一个任务。")
                        continue
                    task_env = random.choice(list(task_set.values()))
                    
                    tasks_in_batch_names.append(task_set_name)
                    args_list.append((task_env, meta_weights_cpu, current_meta_epsilon, self.worker_config))

                if not args_list:
                    print(f"批次 {meta_batch + 1} 中没有有效的任务，跳过。")
                    continue

                # --- 3. 并行执行 ---
                batch_start_time = time.time()
                try:
                    # pool.map 会阻塞，直到所有 worker 都返回
                    results = self.pool.map(reptile_worker, args_list)
                except Exception as e:
                    print(f"批次 {meta_batch + 1} 执行失败: {e}")
                    # 尝试重建进程池
                    self.pool.close()
                    self.pool.join()
                    self.pool = mp.Pool(self.parallel_batch_size)
                    continue

                # --- 4. 聚合结果 ---
                all_deltas_cpu = []
                batch_losses = []
                batch_rewards = []
                batch_steps_trained = 0

                for (delta, avg_loss, steps_trained, avg_reward) in results:
                    if delta is not None:
                        all_deltas_cpu.append(delta)
                        batch_losses.append(avg_loss)
                        batch_rewards.append(avg_reward)
                        batch_steps_trained += steps_trained
                
                if not all_deltas_cpu:
                    print(f"批次 {meta_batch + 1} 中所有 worker 都失败了，跳过元更新。")
                    continue
                    
                avg_batch_loss = np.mean(batch_losses) if batch_losses else 0
                avg_batch_reward = np.mean(batch_rewards) if batch_rewards else 0
                all_batch_losses.append(avg_batch_loss)
                all_batch_rewards.append(avg_batch_reward)

                # --- 5. 元更新 (在主进程的 GPU 上) ---
                with torch.no_grad():
                    # 创建一个平均增量 (在 GPU 上)
                    avg_delta_gpu = {name: torch.zeros_like(param) 
                                     for name, param in self.meta_agent.q_net.named_parameters()}
                    
                    # 累加所有来自 CPU 的增量
                    for delta_cpu in all_deltas_cpu:
                        for name, delta_tensor_cpu in delta_cpu.items():
                            avg_delta_gpu[name] += delta_tensor_cpu.to(self.device)
                            
                    # 应用平均增量
                    num_successful_workers = len(all_deltas_cpu)
                    for name, param in self.meta_agent.q_net.named_parameters():
                        if name in avg_delta_gpu:
                            param.data += self.meta_lr * (avg_delta_gpu[name] / num_successful_workers)

                # --- 6. 衰减 Epsilon ---
                # 根据这个批次中 *实际发生* 的训练总步数来衰减
                meta_agent_total_steps += batch_steps_trained
                for _ in range(batch_steps_trained):
                    self.meta_agent.epsilon = max(self.meta_agent.epsilon_min, 
                                                self.meta_agent.epsilon * self.meta_agent.epsilon_decay)

                # --- 7. 日志和保存 ---
                batch_time = time.time() - batch_start_time
                tasks_processed_so_far = (meta_batch + 1) * self.parallel_batch_size

                if (meta_batch + 1) % self.print_freq_batch == 0:
                    elapsed_total = time.time() - start_time
                    avg_time_per_batch = elapsed_total / (meta_batch + 1 - start_batch)
                    
                    print(f"Meta Batch {meta_batch + 1}/{self.meta_batches} | "
                          f"Tasks {tasks_processed_so_far}/{self.total_meta_iterations} | "
                          f"Avg Batch Reward: {avg_batch_reward:.2f} | "
                          f"Avg Batch Loss: {avg_batch_loss:.4f} | "
                          f"Meta Epsilon: {self.meta_agent.epsilon:.4f} | "
                          f"Total Train Steps: {meta_agent_total_steps} | "
                          f"Time/Batch: {batch_time:.2f}s | "
                          f"Avg Time/Batch: {avg_time_per_batch:.2f}s")
                    # 重置总计时器，以便 Avg Time/Batch 更准确
                    start_time = time.time()
                    start_batch = meta_batch + 1


                # [修改] 按批次保存
                if (meta_batch + 1) % self.save_freq_batch == 0:
                    current_completed_iter_idx = tasks_processed_so_far - 1
                    self._save_checkpoint(self.model_save_path_interrupt, current_completed_iter_idx, meta_agent_total_steps)

        except KeyboardInterrupt:
            print(f"\n训练被中断。正在保存当前检查点...")
            # 保存上一个 *完成* 的批次
            current_completed_iter_idx = (meta_batch * self.parallel_batch_size) - 1
            if current_completed_iter_idx < 0: current_completed_iter_idx = 0
            
            self._save_checkpoint(self.model_save_path_interrupt, current_completed_iter_idx, meta_agent_total_steps)
            print("检查点已保存。安全退出。")
            
        finally:
            # [关键] 始终关闭进程池
            print("正在关闭工作进程池...")
            self.pool.close()
            self.pool.join()
            print("进程池已关闭。")

        # 训练结束后保存最终模型
        print("训练完成。正在保存最终模型...")
        final_iter_idx = self.total_meta_iterations - 1
        self._save_checkpoint(self.model_save_path_final, final_iter_idx, meta_agent_total_steps)

        # [修改] 绘制批次平均损失
        if all_batch_losses:
            plt.figure(figsize=(12, 6))
            plt.plot(all_batch_losses)
            plt.title('Average Batch Loss during Meta Training')
            plt.xlabel('Meta Batch')
            plt.ylabel('MSE Loss')
            plt.grid(True)
            if len(all_batch_losses) > 10:
                moving_avg = np.convolve(all_batch_losses, np.ones(10)/10, mode='valid')
                plt.plot(np.arange(9, len(all_batch_losses)), moving_avg, label='10-batch Moving Average')
                plt.legend()
            plt.savefig('reptile_batch_loss_curve.png')
            print("批次损失曲线图已保存为 reptile_batch_loss_curve.png")
        
        # [新增] 绘制批次平均奖励
        if all_batch_rewards:
            plt.figure(figsize=(12, 6))
            plt.plot(all_batch_rewards)
            plt.title('Average Batch Reward during Meta Training')
            plt.xlabel('Meta Batch')
            plt.ylabel('Average Reward')
            plt.grid(True)
            if len(all_batch_rewards) > 10:
                moving_avg = np.convolve(all_batch_rewards, np.ones(10)/10, mode='valid')
                plt.plot(np.arange(9, len(all_batch_rewards)), moving_avg, label='10-batch Moving Average')
                plt.legend()
            plt.savefig('reptile_batch_reward_curve.png')
            print("批次奖励曲线图已保存为 reptile_batch_reward_curve.png")


# --- 测试类 (基本保持不变) ---
# [修改] 需要确保它在初始化时也设置了 self.device
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
    # [关键] 必须设置 'spawn' 才能安全地将 CUDA 与 multiprocessing 一起使用
    # 这必须是 __name__ == '__main__' 块中的第一件事
    try:
        mp.set_start_method('spawn', force=True)
        print("Multiprocessing 启动方法已设置为 'spawn'。")
    except RuntimeError as e:
        print(f"注意：无法重置启动方法（可能已设置）：{e}")

    # [新增] 定义并行超参数
    PARALLEL_BATCH_SIZE = 14 # [可调] 并行运行 8 个任务。根据您的 CPU 核心数调整
    TOTAL_META_ITERATIONS = 500000 # [可调] 您的目标总任务数
    
    print("正在创建任务集 (这可能需要一些时间)...")
    generated_task = task_environment.create_task_sets(task_sets_config,num_tasks_per_set, pogema)
    print("任务集创建完毕。")

    model_path = 'reptile_drqn_meta_agent_interrupt.pth'if os.path.exists('reptile_drqn_meta_agent_interrupt.pth') else None

    reptile = Reptile(
        all_task_sets=generated_task,
        state_shape=(3, 11, 11),
        num_actions=5,
        num_agents=1,
        task_sets=task_sets_config,
        parallel_batch_size=PARALLEL_BATCH_SIZE, # [新增]
        total_meta_iterations=TOTAL_META_ITERATIONS, # [新增]
        model_path= model_path,
    )
    
    # 开始并行化训练
    reptile.meta_train()
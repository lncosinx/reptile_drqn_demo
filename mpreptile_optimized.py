#训练配置：RTX4090 24GB(99%使用率，23933MiB占用) 16vCPU(只分配了14个线程，1409%使用率) 120GB内存（占用11.3GB）
#pytorch: 2.1.2 python: 3.10 cuda:11.8
#由于cuda版本问题，cuda版本较新时，请注释299行，恢复300行，注释344行，恢复345行

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import random
import pogema
from collections import deque
import time
import os
import torch.multiprocessing as mp

# 任务配置保持不变
task_sets_config = {
    "T1":   {"num_targets": 2, "num_agents": 1, "density": 0.1, "width": 12, "height": 12, "obs_radius": 5},
    "T2":   {"num_targets": 4, "num_agents": 1, "density": 0.15, "width": 16, "height": 16, "obs_radius": 5},
    "T3":   {"num_targets": 6, "num_agents": 1, "density": 0.2, "width": 20, "height": 20, "obs_radius": 5},
    "T4":   {"num_targets": 8, "num_agents": 1, "density": 0.25, "width": 24, "height": 24, "obs_radius": 5},
    "T5":   {"num_targets": 4, "num_agents": 1, "density": 0.3, "width": 16, "height": 16, "obs_radius": 5},
}

# -------------------------------------------------------------------
# [优化] 工作函数 (Worker Function)
# -------------------------------------------------------------------
def reptile_worker(args):
    """
    在单独的进程中执行完整的内部循环（创建环境 + 收集数据 + 训练）。
    
    Args:
        args (tuple): 包含:
            task_config_dict (dict): *单个*任务的环境配置 (例如, {'width': 12, ..., 'seed': 12345})
            meta_q_net (CnnQnet): [共享内存] 元智能体q_net的引用 (这现在是 shared_q_net)
            meta_epsilon (float): 当前的元 epsilon
            worker_config (dict): 包含所有超参数的字典
    """
    try:
        task_config_dict, meta_q_net, meta_epsilon, worker_config = args
        
        # 1. [关键] 在工作进程中定义 device
        worker_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 2. [优化] 即时(JIT)创建环境
        # 从字典中解包配置
        env_config = pogema.GridConfig(
            num_agents=task_config_dict['num_agents'],
            width=task_config_dict['width'],
            height=task_config_dict['height'],
            density=task_config_dict['density'],
            seed=task_config_dict['seed'],
            max_episode_steps=task_config_dict['width'] * task_config_dict['height'],
            obs_radius=task_config_dict['obs_radius'],
            on_target='finish',
        )
        task_env = pogema.pogema_v0(grid_config=env_config)

        # 3. 重新创建 task_agent
        task_buffer = ReplayBuffer(
            worker_config['replay_buffer_capacity'],
            worker_config['seq_len'],
            worker_config['num_agents'],
            worker_config['state_shape'],
            worker_device
        )
        task_agent = DRQNAgent(
            worker_config['num_agents'],
            worker_config['state_shape'],
            worker_config['num_actions'],
            task_buffer,
            worker_config['inner_lr']
        )
        task_agent.batch_size = worker_config['batch_size']
        
        # 4. [优化] 从共享内存加载元权重并移动到 worker 的 GPU
        
        task_agent.q_net.load_state_dict(meta_q_net.state_dict())
        task_agent.target_q_net.load_state_dict(meta_q_net.state_dict())
        
        task_agent.q_net.to(worker_device)
        task_agent.target_q_net.to(worker_device)
        task_agent.q_net.lstm.flatten_parameters()
        task_agent.target_q_net.lstm.flatten_parameters()

        # 5. 设置当前的 epsilon
        task_agent.epsilon = meta_epsilon
        task_agent.epsilon_min = worker_config['epsilon_min']
        task_agent.epsilon_decay = worker_config['epsilon_decay']

        # --- 阶段 1: 收集经验 ---
        current_task_episodes = 0
        current_task_rewards = []
        
        while current_task_episodes < worker_config['episodes_per_task']:
            ep_states, ep_actions, ep_rewards, ep_next_states, ep_dones = [], [], [], [], []
            obs, info = task_env.reset()
            current_hidden_state = None
            terminated = [False] * worker_config['num_agents']
            truncated = [False] * worker_config['num_agents']
            episode_reward = 0

            while not (all(terminated) or all(truncated)):
                obs_np = np.array(obs)
                states_tensor = torch.tensor(obs_np, dtype=torch.float32, device=worker_device)
                
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

            # 确保 episode 足够长
            if len(ep_states) >= worker_config['seq_len']:
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
            # [关键超参数调整] 我们在这里大幅增加了内部训练步数
            for i in range(worker_config['inner_steps']):
                loss = task_agent.train() 
                if loss is not None:
                    inner_losses.append(loss)
                    steps_trained += 1
        
        avg_loss = np.mean(inner_losses) if inner_losses else 0
        avg_reward = np.mean(current_task_rewards) if current_task_rewards else 0

        # --- 阶段 3: 计算增量 (Deltas) 并移至 CPU ---
        with torch.no_grad():
            delta = {}
            # 从共享内存中获取 meta_weights
            meta_weights_gpu = {k: v.to(worker_device) for k, v in meta_q_net.state_dict().items()}

            for name, task_param in task_agent.q_net.named_parameters():
                param_delta = task_param.data - meta_weights_gpu[name]
                delta[name] = param_delta.cpu()
        
        # 返回 (增量, 平均损失, 训练步数, 平均奖励)
        return (delta, avg_loss, steps_trained, avg_reward)

    except Exception as e:
        print(f"[Worker Error] 工作进程失败 (Seed: {task_config_dict.get('seed')}): {e}")
        import traceback
        traceback.print_exc()
        return (None, 0, 0, 0)
    finally:
        # 确保环境被关闭
        if 'task_env' in locals() and hasattr(task_env, 'close'):
            task_env.close()


# [删除] task_environment 类
# 它不再被需要，我们将在 worker 中即时创建环境


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
            sampled_episodes = random.choices(self.buffer, k=batch_size)
        else:
            sampled_episodes = random.sample(self.buffer, batch_size)

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
        
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_scaler)
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
        
        with torch.cuda.amp.autocast(enabled=self.use_scaler):
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

# 定义 Reptile 元学习算法
class Reptile:
    def __init__(self, state_shape, num_actions, num_agents, task_sets_config, 
                 parallel_batch_size, total_meta_iterations, model_path=None):
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[Main] 主进程使用 device: {self.device}")
        
        # [优化] 只存储任务配置，不创建环境
        self.task_sets_config = task_sets_config
        self.task_set_names = list(task_sets_config.keys())
        
        self.state_shape = state_shape
        self.num_actions = num_actions
        self.num_agents = num_agents

        # --- 元学习超参数 ---
        self.parallel_batch_size = parallel_batch_size
        self.total_meta_iterations = total_meta_iterations
        
        # [超参数调整]
        self.meta_lr = 0.0001      # [建议] 降低元学习率 (原为 0.001)

        # --- 内循环超参数 ---
        self.inner_lr = 0.0001
        self.inner_steps = 512     # [建议] 大幅增加内部训练步数 (原为 32)
        self.episodes_per_task = 10
        
        # --- 回放池参数 ---
        # [建议] 容量应与收集的回合数匹配
        self.replay_buffer_capacity = self.episodes_per_task
        self.seq_len = 24
        self.batch_size = 256

        template_buffer = ReplayBuffer(1, 1, num_agents, state_shape, self.device)
        
        # meta_agent 的 q_net 和 target_q_net 始终保留在主 device (GPU) 上
        self.meta_agent = DRQNAgent(num_agents, state_shape, num_actions, template_buffer, lr=self.inner_lr)
        self.meta_agent.batch_size = self.batch_size
        # self.meta_agent 的所有组件 (q_net, target_q_net, optimizer) 都在 self.device (GPU) 上

        self.start_meta_iter = 0
        self.loaded_meta_agent_total_steps = 0

        if model_path and os.path.exists(model_path):
            print(f"正在从 '{model_path}' 加载检查点...")
            try:
                # [修改] 将检查点加载到主 device (GPU)
                checkpoint = torch.load(model_path, map_location=self.device)
                
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    self.meta_agent.q_net.load_state_dict(checkpoint['model_state_dict'])
                    self.meta_agent.target_q_net.load_state_dict(checkpoint['model_state_dict']) # [新增] 确保 target 也加载
                    self.meta_agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    self.meta_agent.epsilon = checkpoint['epsilon']
                    self.loaded_meta_agent_total_steps = checkpoint.get('meta_agent_total_steps', 0)
                    self.start_meta_iter = checkpoint.get('current_meta_iter', 0) + 1
                    print(f"成功加载检查点。将从 Iter {self.start_meta_iter} 和 Epsilon {self.meta_agent.epsilon:.4f} 处恢复。")
                else:
                    self.meta_agent.q_net.load_state_dict(checkpoint)
                    self.meta_agent.target_q_net.load_state_dict(checkpoint) # [新增] 确保 target 也加载
                    print(f"警告：成功加载旧格式模型权重。Epsilon 和优化器将重置。")

            except Exception as e:
                print(f"加载模型 '{model_path}' 失败: {e}。将从头开始训练。")
        else:
            print("未找到模型路径或 model_path 为 None。将从头开始训练。")
            
        # [新增] 创建一个独立的、始终在 CPU 上的共享模型
        self.shared_q_net = CnnQnet(state_shape, num_actions).to('cpu')
        self.shared_q_net.load_state_dict(self.meta_agent.q_net.state_dict()) # 从 GPU -> CPU 同步
        self.shared_q_net.share_memory()
        print("Meta-agent Q-Net 已创建并移至 CPU 共享内存。")
        
        

        self.print_freq = 100 # 按任务数打印
        self.save_freq = 1000 # 按任务数保存
        
        self.model_save_path_interrupt = 'reptile_drqn_meta_agent_interrupt.pth'
        self.model_save_path_final = 'reptile_drqn.pth'
        
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
        
        print(f"正在创建 {self.parallel_batch_size} 个工作进程的进程池...")
        self.pool = mp.Pool(self.parallel_batch_size)
        print("进程池创建完毕。")

    def _save_checkpoint(self, path, meta_iter_task_count, meta_agent_total_steps):
        try:
            # 保存时，我们保存主 GPU 模型的权重
            # torch.save 会自动处理从 GPU 移至 CPU
            checkpoint = {
                'model_state_dict': self.meta_agent.q_net.state_dict(),
                'optimizer_state_dict': self.meta_agent.optimizer.state_dict(),
                'epsilon': self.meta_agent.epsilon,
                'meta_agent_total_steps': meta_agent_total_steps,
                'current_meta_iter': meta_iter_task_count # 保存的是已完成的任务迭代次数
            }
            torch.save(checkpoint, path)
            print(f"\n--- 检查点已保存到 {path} (Tasks: {meta_iter_task_count + 1}) ---")
        except Exception as e:
            print(f"警告：保存检查点到 {path} 失败: {e}")

    def _get_random_task_config(self):
        """生成一个随机任务配置字典。"""
        task_name = random.choice(self.task_set_names)
        config = self.task_sets_config[task_name].copy()
        # [优化] JIT 创建：我们只需要配置 + 一个随机种子
        config['seed'] = random.randint(0, 1_000_000)
        return config

    def _task_generator(self):
        """一个无限生成器，用于为异步池提供任务。"""
        while True:
            task_config = self._get_random_task_config()
            current_meta_epsilon = self.meta_agent.epsilon
            # 始终传递 shared_q_net (它在 CPU 上)
            yield (task_config, self.shared_q_net, current_meta_epsilon, self.worker_config)

    # [优化] meta_train 被完全重构为异步模式
    def meta_train(self):
        print(f"开始 Reptile 元学习训练 (目标 {self.total_meta_iterations} 个任务, 异步批大小 {self.parallel_batch_size})...")
        start_time = time.time()
        meta_agent_total_steps = self.loaded_meta_agent_total_steps
        
        # 用于日志
        tasks_processed_count = self.start_meta_iter
        all_losses = []
        all_rewards = []
        
        # 收集一个批次的增量
        delta_batch = []
        
        try:
            # 创建异步任务迭代器
            task_iterator = self.pool.imap_unordered(reptile_worker, self._task_generator())
            
            # 持续从迭代器中获取结果，直到达到总任务数
            while tasks_processed_count < self.total_meta_iterations:
                
                # 1. [优化] 阻塞，直到 *任何一个* worker 返回结果
                result = next(task_iterator)
                tasks_processed_count += 1
                
                (delta, avg_loss, steps_trained, avg_reward) = result
                
                # 2. 收集结果
                if delta is not None:
                    delta_batch.append(delta)
                    all_losses.append(avg_loss)
                    all_rewards.append(avg_reward)
                    
                    # 3. 衰减 Epsilon
                    meta_agent_total_steps += steps_trained
                    for _ in range(steps_trained):
                        self.meta_agent.epsilon = max(self.meta_agent.epsilon_min, 
                                                    self.meta_agent.epsilon * self.meta_agent.epsilon_decay)
                
                # 4. [优化] 当收集到足够多的增量时，执行元更新
                if len(delta_batch) >= self.parallel_batch_size:
                    
                    # --- 5. 元更新 (在主进程的 device (GPU) 上) ---
                    
                    # meta_agent.q_net 始终在 GPU 上
                    avg_delta_gpu = {name: torch.zeros(param.shape, device=self.device) 
                                     for name, param in self.meta_agent.q_net.named_parameters()}
                    
                    for delta_cpu in delta_batch:
                        for name, delta_tensor_cpu in delta_cpu.items():
                            # [修改] 确保在正确的 device 上累加
                            if name in avg_delta_gpu:
                                avg_delta_gpu[name] += delta_tensor_cpu.to(self.device)
                    
                    
                    
                    # 应用平均增量
                    num_successful_workers = len(delta_batch)
                    with torch.no_grad():
                        for name, param in self.meta_agent.q_net.named_parameters():
                            if name in avg_delta_gpu:
                                param.data += self.meta_lr * (avg_delta_gpu[name] / num_successful_workers)

                    # 同步：将更新后的 GPU 权重复制到
                    # 1. GPU 上的 target network
                    self.meta_agent.target_q_net.load_state_dict(self.meta_agent.q_net.state_dict())
                    # 2. CPU 上的 shared network
                    self.shared_q_net.load_state_dict(self.meta_agent.q_net.state_dict())

    

                    # --- 6. 日志和清空 ---
                    if (tasks_processed_count // self.print_freq) > ((tasks_processed_count - len(delta_batch)) // self.print_freq):
                        elapsed = time.time() - start_time
                        tasks_per_sec = (tasks_processed_count - self.start_meta_iter) / elapsed if elapsed > 0 else 0
                        
                        print(f"Tasks {tasks_processed_count}/{self.total_meta_iterations} | "
                              f"Avg Reward (last {self.print_freq}): {np.mean(all_rewards[-self.print_freq:]):.2f} | "
                              f"Avg Loss (last {self.print_freq}): {np.mean(all_losses[-self.print_freq:]):.4f} | "
                              f"Meta Epsilon: {self.meta_agent.epsilon:.4f} | "
                              f"Total Train Steps: {meta_agent_total_steps} | "
                              f"Tasks/sec: {tasks_per_sec:.2f} | "
                              f"Timestamp: {time.strftime('%Y%m%d %X')}")

                    # --- 7. 保存 ---
                    if (tasks_processed_count // self.save_freq) > ((tasks_processed_count - len(delta_batch)) // self.save_freq):
                        self._save_checkpoint(self.model_save_path_interrupt, tasks_processed_count - 1, meta_agent_total_steps)

                    # 清空批次
                    delta_batch.clear()

        except KeyboardInterrupt:
            print(f"\n训练被中断。正在保存当前检查点...")
            self._save_checkpoint(self.model_save_path_interrupt, tasks_processed_count - 1, meta_agent_total_steps)
            print("检查点已保存。安全退出。")
            
        finally:
            print("正在关闭工作进程池...")
            self.pool.close()
            self.pool.join()
            print("进程池已关闭。")

        print("训练完成。正在保存最终模型...")
        self._save_checkpoint(self.model_save_path_final, self.total_meta_iterations - 1, meta_agent_total_steps)

        # 绘制曲线
        if all_losses:
            plt.figure(figsize=(12, 6))
            plt.plot(all_losses)
            plt.title('Task Average Loss during Meta Training (Per Task)')
            plt.xlabel('Tasks Processed')
            plt.ylabel('MSE Loss')
            plt.grid(True)
            if len(all_losses) > 100:
                moving_avg = np.convolve(all_losses, np.ones(100)/100, mode='valid')
                plt.plot(np.arange(99, len(all_losses)), moving_avg, label='100-task Moving Average', color='red')
                plt.legend()
            plt.savefig('reptile_task_loss_curve.png')
            print("任务损失曲线图已保存为 reptile_task_loss_curve.png")
        
        if all_rewards:
            plt.figure(figsize=(12, 6))
            plt.plot(all_rewards)
            plt.title('Task Average Reward during Meta Training (Per Task)')
            plt.xlabel('Tasks Processed')
            plt.ylabel('Average Reward')
            plt.grid(True)
            if len(all_rewards) > 100:
                moving_avg = np.convolve(all_rewards, np.ones(100)/100, mode='valid')
                plt.plot(np.arange(99, len(all_rewards)), moving_avg, label='100-task Moving Average', color='red')
                plt.legend()
            plt.savefig('reptile_task_reward_curve.png')
            print("任务奖励曲线图已保存为 reptile_task_reward_curve.png")


# --- 测试类 (基本保持不变) ---
# [修改] 需要确保它在初始化时也设置了 self.device
class TestAgent(DRQNAgent):
    def __init__(self, state_shape, num_actions, state_dict_file_path, finetune_lr, finetune_buffer_capacity, seq_len, num_agents_to_test):
        
        test_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        finetune_buffer = ReplayBuffer(finetune_buffer_capacity, seq_len, num_agents_to_test, state_shape, test_device)
        
        super().__init__(
            num_agents=num_agents_to_test,
            state_shape=state_shape,
            num_actions=num_actions,
            replay_buffer=finetune_buffer,
            lr=finetune_lr
        )
        
        try:
            # 加载到 self.device (在 super().__init__ 中设置)
            checkpoint = torch.load(state_dict_file_path, map_location=self.device)
            
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

            if len(ep_states) >= self.replay_buffer.seq_len:
                self.replay_buffer.push({
                    'states': ep_states, 'actions': ep_actions, 'rewards': ep_rewards,
                    'next_states': ep_next_states, 'dones': ep_dones
                })
                print(f"  微调回合 {ep+1}/{num_episodes} 完成 (长度: {ep_len}, 奖励: {ep_reward:.2f})。Buffer: {len(self.replay_buffer)}")
            else:
                 print(f"  微调回合 {ep+1}/{num_episodes} 完成 (长度: {ep_len}, 奖励: {ep_reward:.2f})。回合太短，已丢弃。")


            if len(self.replay_buffer) >= self.batch_size:
                for _ in range(num_steps_per_train): 
                    loss = self.train()
                    if loss is not None:
                         all_finetune_losses.append(loss)
                         total_steps_trained += 1

        avg_finetune_loss = np.mean(all_finetune_losses) if all_finetune_losses else 0
        print(f"微调完成。共训练 {total_steps_trained} 步。平均损失: {avg_finetune_loss:.4f}")

    def evaluate(self, task_env, render_animation=False, animation_filename="task_evaluation.svg"):
        print(f"开始评估 (零探索)...")
        
        if render_animation:
            try:
                env = pogema.AnimationMonitor(task_env)
            except Exception as e:
                print(f"无法启动 AnimationMonitor ({e})，将使用普通环境。")
                env = task_env
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

        if render_animation and isinstance(env, pogema.AnimationMonitor):
            try:
                env.save_animation(animation_filename)
                print(f"动画已保存至 {animation_filename}\n")
            except Exception as e:
                print(f"保存动画时出错: {e}")
        
        return total_rewards, step_count
    
    def task_env(self, task_config):
        config = pogema.GridConfig(
            num_agents=task_config["num_agents"],
            width=task_config["width"],
            height=task_config["height"],
            density=task_config["density"],
            seed=1024,
            max_episode_steps=task_config["width"] * task_config["height"],
            obs_radius=task_config["obs_radius"],
            on_target='finish',
        )
        env = pogema.pogema_v0(grid_config=config)
        return env

if __name__ == '__main__':
    try:
        mp.set_start_method('spawn', force=True)
        print("Multiprocessing 启动方法已设置为 'spawn'。")
    except RuntimeError as e:
        print(f"注意：无法重置启动方法（可能已设置）：{e}")

    # --- [调整] 超参数 ---
    # 核心数 (根据您的 CPU 调整)
    CPU_COUNT = os.cpu_count()
    #PARALLEL_BATCH_SIZE = int(CPU_COUNT * 0.9) if CPU_COUNT else 14 # 使用 80% 的核心
    PARALLEL_BATCH_SIZE = 14 # 使用14个核心
    if PARALLEL_BATCH_SIZE == 0: PARALLEL_BATCH_SIZE = 1
    
    print(f"检测到 {CPU_COUNT} 个 CPU 核心, 将使用 {PARALLEL_BATCH_SIZE} 个并行工作进程。")
    
    TOTAL_META_ITERATIONS = 500000 
    
    # [删除] 预创建任务集
    # print("正在创建任务集 (这可能需要一些时间)...")
    # generated_task = task_environment.create_task_sets(task_sets_config,num_tasks_per_set, pogema)
    # print("任务集创建完毕。")

    model_path = 'reptile_drqn_meta_agent_interrupt.pth' if os.path.exists('reptile_drqn_meta_agent_interrupt.pth') else None

    reptile = Reptile(
        state_shape=(3, 11, 11),
        num_actions=5,
        num_agents=1,
        task_sets_config=task_sets_config, # [修改] 传入配置，而不是环境
        parallel_batch_size=PARALLEL_BATCH_SIZE,
        total_meta_iterations=TOTAL_META_ITERATIONS,
        model_path= model_path,
    )
    
    reptile.meta_train()


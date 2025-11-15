#训练配置：RTX4090 24GB(99%使用率，23933MiB占用) 16vCPU(只分配了14个线程，1409%使用率) 120GB内存（占用11.3GB）
#pytorch: 2.1.2 python: 3.10 cuda:11.8

import torch
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import torch.multiprocessing as mp
import task_environment
from module_set import RewardSet, CRnnQnet, PrioritizedReplayBuffer, DRQNAgent


# -------------------------------------------------------------------
# [优化] 工作函数 (Worker Function)
# -------------------------------------------------------------------

def reptile_drqn_worker(args):
    """
    在单独的进程中执行完整的内部循环（创建环境 + 收集数据 + 训练）。
    
    Args:
        args (tuple): 包含:
            meta_q_net (CnnQnet): [共享内存] 元智能体q_net的引用 (这现在是 shared_q_net)
            meta_epsilon (float): 当前的元 epsilon
            worker_config (dict): 包含所有超参数的字典
    """
    try:
        meta_q_net, meta_epsilon, worker_config = args
        
        # 1. [关键] 在工作进程中定义 device
        worker_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 2. [优化] 即时(JIT)创建环境
        task_env, map_type, seed, num_targets = task_environment.create_task_env()

        num_agents = worker_config['num_agents']
        seq_len = worker_config['seq_len']
        state_shape = worker_config['state_shape']

        # 3. 重新创建 task_agent
        task_buffer = PrioritizedReplayBuffer(
            worker_config['replay_buffer_capacity'],
            seq_len,
            num_agents,
            state_shape,
            worker_device,
            alpha=worker_config['per_alpha'],
            beta_start=worker_config['per_beta_start'],
            beta_frames=worker_config['inner_steps']
        )
        task_agent = DRQNAgent(
            num_agents,
            state_shape,
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

        # 为填充（padding）准备“空”数据，以确保所有回合至少为 seq_len 长
        # (N, C, H, W)
        empty_obs_np = np.zeros((num_agents, *state_shape), dtype=np.float32) 
         # (N,)
        empty_action_np = np.zeros(num_agents, dtype=np.int64)
        # (N,)
        empty_reward_list = [0.0] * num_agents 
        # (N,)
        empty_done_list = [True] * num_agents
        
        while current_task_episodes < worker_config['episodes_per_task']: # 收集 episodes_per_task 回合
            ep_states, ep_actions, ep_rewards, ep_next_states, ep_dones = [], [], [], [], []
            obs, info = task_env.reset()
            current_hidden_state = None
            terminated = [False] * num_agents
            truncated = [False] * num_agents
            reward_calculator = RewardSet(num_agents, worker_device)
            
            while not (all(terminated) or all(truncated)): # 单个回合
                obs_np = np.array(obs)
                states_tensor = torch.tensor(obs_np, dtype=torch.float32, device=worker_device)
                
                actions, new_hidden_state = task_agent.select_actions(states_tensor, current_hidden_state)
                current_hidden_state = new_hidden_state
                
                next_obs, rewards, terminated, truncated, info = task_env.step(actions)
                next_obs_np = np.array(next_obs)

                rewards_tensor = reward_calculator.calculate_total_reward(rewards, states_tensor, actions) 

                ep_states.append(obs_np)
                ep_actions.append(actions)
                ep_rewards.append(rewards_tensor.tolist())
                ep_next_states.append(next_obs_np)
                ep_dones.append(terminated)
                
                obs = next_obs
                
            episode_length = len(ep_states)

            if episode_length < seq_len:
                # 计算需要填充多少步
                padding_needed = seq_len - episode_length
                
                # 填充所有列表
                ep_states.extend([empty_obs_np] * padding_needed)
                ep_actions.extend([empty_action_np] * padding_needed)
                ep_rewards.extend([empty_reward_list] * padding_needed)
                ep_next_states.extend([empty_obs_np] * padding_needed) # 下一个状态也是空的
                ep_dones.extend([empty_done_list] * padding_needed) # 标记为 "Done"
            
            # 现在，所有回合（无论长短）都至少是 seq_len 长
            # 如果回合长于 seq_len，回放池的采样会自动处理
            task_agent.replay_buffer.push({
                'states': ep_states, 'actions': ep_actions, 'rewards': ep_rewards,
                'next_states': ep_next_states, 'dones': ep_dones
            })
            
            current_task_episodes += 1
            current_task_rewards.append(reward_calculator.total_rewards()) 

        # --- 阶段 2: 训练 ---
        inner_losses = []
        steps_trained = 0
        
        if len(task_agent.replay_buffer) > 0:
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
        
        # 返回 (增量, 平均损失, 训练步数, 平均奖励，地图类型，种子)
        return (delta, avg_loss, steps_trained, avg_reward, map_type, seed)

    except Exception as e:
        print(f"[Worker Error] 工作进程失败 (Seed: {seed}): {e}")
        import traceback
        traceback.print_exc()
        return (None, 0, 0, 0)
    finally:
        # 确保环境被关闭
        if 'task_env' in locals() and hasattr(task_env, 'close'):
            task_env.close()


# 定义 Reptile 元学习算法
class Reptile:
    def __init__(self, state_shape, num_actions, num_agents,  
                 parallel_batch_size, total_meta_iterations, model_path=None):
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[Main] 主进程使用 device: {self.device}")
        
        self.state_shape = state_shape
        self.num_actions = num_actions
        self.num_agents = num_agents

        # --- 元学习超参数 ---
        self.parallel_batch_size = parallel_batch_size
        self.total_meta_iterations = total_meta_iterations
        
        # [超参数调整]
        self.meta_lr = 0.001      # [建议] 增加元学习率 (原为 0.0001)

        # --- 内循环超参数 ---
        self.inner_lr = 0.0001
        self.inner_steps = 512     # [建议] 大幅增加内部训练步数 (原为 32) 
        self.episodes_per_task = 300 # 增大 episodes_per_tas ，保证数据的多样性 
        
        # --- 回放池参数 ---
        # [建议] 容量应与收集的回合数匹配
        self.per_alpha = 0.6
        self.per_beta_start = 0.4
        self.replay_buffer_capacity = self.episodes_per_task
        self.seq_len = 24
        self.batch_size = 32  #减小batch size，解决批次间的高度相关性，提高采样效率（原为256）128

        template_buffer = PrioritizedReplayBuffer(1, 1, num_agents, state_shape, self.device)
        
        # meta_agent 的 q_net 和 target_q_net 始终保留在主 device (GPU) 上
        self.meta_agent = DRQNAgent(num_agents, state_shape, num_actions, template_buffer, lr=self.inner_lr)
        self.meta_agent.batch_size = self.batch_size
        # self.meta_agent 的所有组件 (q_net, target_q_net, optimizer) 都在 self.device (GPU) 上

        self.start_meta_iter = 0
        self.loaded_meta_agent_total_steps = 0

        if model_path and os.path.exists(model_path):
            print(f"正在从 '{model_path}' 加载检查点...")
            try:
                # 将检查点加载到主 device (GPU)
                checkpoint = torch.load(model_path, map_location=self.device)
                
                self.meta_agent.q_net.load_state_dict(checkpoint['model_state_dict'])
                self.meta_agent.target_q_net.load_state_dict(checkpoint['model_state_dict']) 
                self.meta_agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.meta_agent.epsilon = checkpoint['epsilon']
                self.loaded_meta_agent_total_steps = checkpoint.get('meta_agent_total_steps', 0)
                self.start_meta_iter = checkpoint.get('current_meta_iter', 0) + 1
                print(f"成功加载检查点。将从 Iter {self.start_meta_iter} 和 Epsilon {self.meta_agent.epsilon:.4f} 处恢复。")

            except Exception as e:
                print(f"加载模型 '{model_path}' 失败: {e}。将从头开始训练。")
        else:
            print("未找到模型路径或 model_path 为 None。将从头开始训练。")
            
        # [新增] 创建一个独立的、始终在 CPU 上的共享模型
        self.shared_q_net = CRnnQnet(state_shape, num_actions).to('cpu')
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
            'per_alpha': self.per_alpha,
            'per_beta_start': self.per_beta_start,
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

    def _task_generator(self):
        """一个无限生成器，用于为异步池提供任务。"""
        while True:
            current_meta_epsilon = self.meta_agent.epsilon
            # 始终传递 shared_q_net (它在 CPU 上)
            yield (self.shared_q_net, current_meta_epsilon, self.worker_config)

    # [优化] meta_train 被完全重构为异步模式
    def meta_train(self):
        print(f"开始 Reptile 元学习训练 (目标 {self.total_meta_iterations} 个任务, 异步批大小 {self.parallel_batch_size})...")
        start_time = time.time()
        meta_agent_total_steps = self.loaded_meta_agent_total_steps
        
        # 用于日志
        tasks_processed_count = self.start_meta_iter
        all_losses = []
        all_rewards = []
        all_seeds = []
        all_map_types = []
        
        # 收集一个批次的增量
        delta_batch = []
        
        try:
            # 创建异步任务迭代器
            task_iterator = self.pool.imap_unordered(reptile_drqn_worker, self._task_generator())
            
            # 持续从迭代器中获取结果，直到达到总任务数
            while tasks_processed_count < self.total_meta_iterations:
                
                # 1. [优化] 阻塞，直到 *任何一个* worker 返回结果
                result = next(task_iterator)
                tasks_processed_count += 1
                
                (delta, avg_loss, steps_trained, avg_reward, map_type, seed) = result
                
                # 2. 收集结果
                if delta is not None:
                    delta_batch.append(delta)
                    all_losses.append(avg_loss)
                    all_rewards.append(avg_reward)
                    all_map_types.append(map_type)
                    all_seeds.append(seed)
                    
                    # 3. 衰减 Epsilon
                    meta_agent_total_steps += steps_trained
                    for _ in range(steps_trained):
                        self.meta_agent.epsilon = max(self.meta_agent.epsilon_min, self.meta_agent.epsilon * self.meta_agent.epsilon_decay)
                
                # 4. [优化] 当收集到足够多的增量时，执行元更新
                if len(delta_batch) >= self.parallel_batch_size:
                    
                    # --- 5. 元更新 (在主进程的 device (GPU) 上) ---
                    
                    # meta_agent.q_net 始终在 GPU 上
                    avg_delta_gpu = {name: torch.zeros(param.shape, device=self.device) 
                                     for name, param in self.meta_agent.q_net.named_parameters()}
                    
                    for delta_cpu in delta_batch:
                        for name, delta_tensor_cpu in delta_cpu.items():
                            # 确保在正确的 device 上累加
                            if name in avg_delta_gpu:
                                avg_delta_gpu[name] += delta_tensor_cpu.to(self.device)
                    
                    # 应用平均增量
                    num_successful_workers = len(delta_batch)
                    with torch.no_grad():
                        for name, param in self.meta_agent.q_net.named_parameters():
                            if name in avg_delta_gpu:
                                param.data += self.meta_lr * (avg_delta_gpu[name] / num_successful_workers)

                    # 同步：将更新后的 GPU 权重复制到 CPU
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
  
if __name__ == '__main__':
    try:
        mp.set_start_method('spawn', force=True)
        print("Multiprocessing 启动方法已设置为 'spawn'。")
    except RuntimeError as e:
        print(f"注意：无法重置启动方法（可能已设置）：{e}")

    # --- [调整] 超参数 ---
    # 核心数 (根据 CPU 调整)
    CPU_COUNT = os.cpu_count()
    PARALLEL_BATCH_SIZE = int(CPU_COUNT * 0.9) if CPU_COUNT else 14 # 使用 80% 的核心
    # PARALLEL_BATCH_SIZE = 16 # 使用16个核心
    if PARALLEL_BATCH_SIZE == 0: PARALLEL_BATCH_SIZE = 1
    
    print(f"检测到 {CPU_COUNT} 个 CPU 核心, 将使用 {PARALLEL_BATCH_SIZE} 个并行工作进程。")
    
    TOTAL_META_ITERATIONS = 100000 # 总任务数 (epsilon_decay的设置方式：epsilon_decay**TOTAL_META_ITERATIONS = 0.1) 100000

    model_path = 'reptile_drqn_meta_agent_interrupt.pth' if os.path.exists('reptile_drqn_meta_agent_interrupt.pth') else None

    reptile = Reptile(
        state_shape=(3, 11, 11),
        num_actions=5,
        num_agents=1,
        parallel_batch_size=PARALLEL_BATCH_SIZE,
        total_meta_iterations=TOTAL_META_ITERATIONS,
        model_path= model_path,
    )
    
    reptile.meta_train()


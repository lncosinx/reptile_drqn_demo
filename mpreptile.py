import torch
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import torch.multiprocessing as mp
import task_environment
from module_set import RewardSet, ReplayBuffer, DRQNAgent

# -------------------------------------------------------------------
# 并行工作函数 (必须在全局范围)
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

        num_agents = config['num_agents']
        seq_len = config['seq_len']
        state_shape = config['state_shape']


        # 2. 重新创建 task_agent
        # (需要 ReplayBuffer, CnnQnet, DRQNAgent 类在全局范围内可用)
        task_buffer = ReplayBuffer(
            config['replay_buffer_capacity'],
            seq_len,
            num_agents,
            state_shape,
            worker_device
        )
        task_agent = DRQNAgent(
            num_agents,
            state_shape,
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
        
        # 为填充（padding）准备“空”数据
        # (N, C, H, W)
        empty_obs_np = np.zeros((num_agents, *state_shape), dtype=np.float32) 
         # (N,)
        empty_action_np = np.zeros(num_agents, dtype=np.int64)
        # (N,)
        empty_reward_list = [0.0] * num_agents 
        # (N,)
        empty_done_list = [True] * num_agents
        
        while current_task_episodes < config['episodes_per_task']:
            ep_states, ep_actions, ep_rewards, ep_next_states, ep_dones = [], [], [], [], []
            obs, info = task_env.reset()
            current_hidden_state = None
            terminated = [False] * num_agents
            truncated = [False] * num_agents
            reward_calculator = RewardSet(num_agents, worker_device)
            
            while not (all(terminated) or all(truncated)): # 修改终止条件以确保达到目标数量
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
            for i in range(config['inner_steps']):
                # 注意：task_agent.train() 会衰减它自己的 epsilon
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


# 定义 Reptile 元学习算法
class Reptile:
    def __init__(self, state_shape, num_actions, num_agents,  
                 parallel_batch_size, total_meta_iterations, model_path=None):
        
        # [新增] 主进程 device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[Main] 主进程使用 device: {self.device}")
        
        # self.all_task_sets = all_task_sets
        self.state_shape = state_shape # (C, H, W)
        self.num_actions = num_actions
        self.num_agents = num_agents
        # self.task_sets = task_sets

        # --- [修改] 元学习超参数 ---
        self.parallel_batch_size = parallel_batch_size # B: 每次并行运行多少个任务
        self.total_meta_iterations = total_meta_iterations # 目标总任务数
        self.meta_batches = self.total_meta_iterations // self.parallel_batch_size # 主循环次数
        self.meta_lr = 0.0001      # 外循环学习率，降低外部学习率 0.001->0.0001

        # --- [修改] 内循环超参数 (将打包发送给 worker) ---
        self.inner_lr = 0.0001     # 内循环学习率 (DRQNAgent 的 lr)
        self.inner_steps = 512   # k: 内循环中的 *梯度下降步数* ，32->512
        self.episodes_per_task = 300 # 每个任务收集多少个 episode 来填充 buffer
        
        # --- [修改] 回放池参数 (将打包发送给 worker) ---
        self.replay_buffer_capacity = self.episodes_per_task # 回放池容量设为每任务 episode 数
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
                
                # --- 2. 采样parallel_batch_size个任务 ---
                args_list = []
                # tasks_in_batch_names = [] # 用于日志
                for _ in range(self.parallel_batch_size):
                    task_env, map_type, seed, num_targets = task_environment.create_task_env()
                   
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
                          f"Avg Time/Batch: {avg_time_per_batch:.2f}s | "
                          f"Timestamp: {time.strftime('%Y%m%d %X')}")
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
    
    """ print("正在创建任务集 (这可能需要一些时间)...")
    generated_task = task_environment.create_task_sets(task_sets_config,num_tasks_per_set, pogema)
    print("任务集创建完毕。") """

    model_path = 'reptile_drqn_meta_agent_interrupt.pth'if os.path.exists('reptile_drqn_meta_agent_interrupt.pth') else None

    reptile = Reptile(
        state_shape=(3, 11, 11),
        num_actions=5,
        num_agents=1,
        parallel_batch_size=PARALLEL_BATCH_SIZE, # [新增]
        total_meta_iterations=TOTAL_META_ITERATIONS, # [新增]
        model_path= model_path,
    )
    
    # 开始并行化训练
    reptile.meta_train()
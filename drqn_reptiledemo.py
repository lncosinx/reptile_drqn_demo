import torch
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import task_environment
from module_set import RewardSet, PrioritizedReplayBuffer, DRQNAgent


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 定义 Reptile 元学习算法
class Reptile:
    def __init__(self, state_shape, num_actions, num_agents, model_path=None):
        self.state_shape = state_shape # (C, H, W)
        self.num_actions = num_actions
        self.num_agents = num_agents

        # --- 元学习超参数 ---
        self.meta_iterations = 20000
        self.meta_lr = 0.001       # 外循环学习率

        # --- 内循环超参数 ---
        self.inner_lr = 0.0001     # 内循环学习率 (DRQNAgent 的 lr)
        self.inner_steps = 512      # k: 内循环中的 *梯度下降步数*
        self.episodes_per_task = 300 # 每个任务收集多少个 episode 来填充 buffer
        
        # --- 回放池参数 ---
        self.replay_buffer_capacity = self.episodes_per_task  # 回放池容量
        self.seq_len = 24           # 序列长度
        self.batch_size = 128        # 批次大小

        # 1. 创建一个“模板”回放池。这实际上不会被 meta_agent 使用。
        template_buffer = PrioritizedReplayBuffer(self.replay_buffer_capacity, self.seq_len, num_agents, state_shape, device)
        
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
        # 修正：记录所有内部训练步骤的损失，而不仅仅是每个 meta iter 的平均值
        all_inner_step_losses = []
        # 修正：跟踪 meta_agent 的总训练步数，用于 epsilon 衰减
        meta_agent_total_steps = self.loaded_meta_agent_total_steps

        try:
            # [修改] 循环从 start_meta_iter 开始
            for meta_iter in range(self.start_meta_iter, self.meta_iterations):
                task_env = task_environment.create_task_env() # 每次 meta 迭代即时创建环境，不使用预创建的任务集

                # 3. 创建 task_agent, 复制 meta_agent 权重和 *当前的* epsilon
                task_agent = DRQNAgent()
                meta_weights_cpu = {k: v.cpu() for k, v in self.meta_agent.q_net.state_dict().items()}
                task_agent.q_net.load_state_dict(meta_weights_cpu)
                task_agent.target_q_net.load_state_dict(meta_weights_cpu)
                task_agent.q_net.to(device)
                task_agent.target_q_net.to(device)
                task_agent.q_net.lstm.flatten_parameters()
                task_agent.target_q_net.lstm.flatten_parameters()
                task_agent.replay_buffer = PrioritizedReplayBuffer(self.replay_buffer_capacity, self.seq_len, self.num_agents, self.state_shape, device)
                # 修正：确保 task_agent 使用 meta_agent 当前的 epsilon 开始
                task_agent.epsilon = self.meta_agent.epsilon

                # --- 内循环 ---
                current_task_episodes = 0
                current_task_rewards = []
                
                # 为填充（padding）准备“空”数据
                # (N, C, H, W)
                empty_obs_np = np.zeros((self.num_agents, *self.state_shape), dtype=np.float32) 
                  # (N,)
                empty_action_np = np.zeros(self.num_agents, dtype=np.int64)
                # (N,)
                empty_reward_list = [0.0] * self.num_agents 
                # (N,)
                empty_done_list = [True] * self.num_agents
        

                # 阶段 1: 收集经验 (保持不变)
                while current_task_episodes < self.episodes_per_task:
                    ep_states, ep_actions, ep_rewards, ep_next_states, ep_dones = [], [], [], [], []
                    obs, info = task_env.reset()
                    current_hidden_state = None
                    terminated = [False] * self.num_agents
                    truncated = [False] * self.num_agents
                    reward_calculator = RewardSet()
        
                    while not (all(terminated) or all(truncated)):
                        obs_np = np.array(obs)
                        states_tensor = torch.tensor(obs_np, dtype=torch.float32, device=device)
                        # 使用 task_agent 的 epsilon 进行探索
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

                    if episode_length < self.seq_len:
                        # 计算需要填充多少步
                        padding_needed = self.seq_len - episode_length
                
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

                total_episodes_run += current_task_episodes

                # --- 阶段 2: 训练 ---
                inner_losses_current_iter = [] # 只记录当前 meta iter 的损失用于平均
                if len(task_agent.replay_buffer) >= task_agent.batch_size:
                    for i in range(self.inner_steps):
                        loss = task_agent.train() # task_agent 训练并更新自己的网络权重,衰减自己的 epsilon
                        if loss is not None:
                            inner_losses_current_iter.append(loss)
                            all_inner_step_losses.append(loss) # 记录每一步的损失
                            # self.meta_agent.epsilon = max(self.meta_agent.epsilon_min, self.meta_agent.epsilon * self.meta_agent.epsilon_decay)
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
                        f"Avg Task Reward (Collect): {avg_task_reward:.2f} | "
                        # 修正：打印当前迭代的平均损失
                        f"Avg Inner Loss (Iter): {avg_inner_loss_iter:.4f} | "
                        # 修正：打印 meta_agent 的 epsilon
                        f"Meta Epsilon: {self.meta_agent.epsilon:.4f} | "
                        f"Total Train Steps: {meta_agent_total_steps} | " # 打印总训练步数
                        f"Time/Iter: {elapsed / self.print_freq:.2f}s | "
                        f"Timestamp: {time.strftime('%Y%m%d %X')}")
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

if __name__ == '__main__':

    model_path = 'reptile_drqn_meta_agent_interrupt.pth'if os.path.exists('reptile_drqn_meta_agent_interrupt.pth') else None

    reptile = Reptile(
    state_shape=(3, 11, 11),
    num_actions=5,
    num_agents=1,
    model_path= model_path,
    )
    reptile.meta_train()

# -*- coding: utf-8 -*-
"""
DQN 消融实验脚本 (Ablation Study for Reptile - DQN vs DRQN)

此脚本训练一个标准的 DQN 智能体 (无 LSTM, 序列长度为 1)。
它在一个连续的任务流上训练，以与 DRQN 消融实验和 Reptile-DRQN 进行比较。

为了保证可比性，它遵循相同的总训练步数：
Total Steps = TOTAL_TASKS * STEPS_PER_TASK
            = 100,000 * 512 = 51,200,000 步
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import time
import os

# 导入你提供的模块
import task_environment
# [关键] 导入修正后的 CnnQnet 和 DQNAgent
from module_set import RewardSet, PrioritizedReplayBuffer, CnnQnet, DQNAgent

# -------------------------------------------------------------------
# 辅助函数：保存检查点
# -------------------------------------------------------------------

def _save_checkpoint(path, agent, tasks_processed, total_steps):
    """保存 DQN 智能体的当前状态"""
    try:
        checkpoint = {
            'model_state_dict': agent.q_net.state_dict(),
            'optimizer_state_dict': agent.optimizer.state_dict(),
            'epsilon': agent.epsilon,
            'tasks_processed': tasks_processed,
            'total_steps_trained': total_steps
        }
        torch.save(checkpoint, path)
        print(f"\n--- 检查点已保存到 {path} (Tasks: {tasks_processed + 1}) ---")
    except Exception as e:
        print(f"警告：保存检查点到 {path} 失败: {e}")

# -------------------------------------------------------------------
# 主训练脚本
# -------------------------------------------------------------------

if __name__ == '__main__':
    
    # --- 1. 超参数设置 ---
    # 与 mpreptile_drqn.py 和 drqn_ablation.py 保持一致
    
    # 实验配置
    TOTAL_TASKS = 100000      # 你的 100,000 个任务
    STEPS_PER_TASK = 512      # 你的每个任务 512 步梯度下降
    EPISODES_PER_TASK = 300   # 每个任务收集的数据量
    
    # 环境和智能体配置
    STATE_SHAPE = (3, 11, 11)
    NUM_ACTIONS = 5
    NUM_AGENTS = 1
    LR = 0.0001               # 使用 Reptile 的 inner_lr
    BATCH_SIZE = 128
    
    # [关键区别] DQN 使用 seq_len = 1
    SEQ_LEN = 1
    
    # PER 和回放池配置
    PER_ALPHA = 0.6
    PER_BETA_START = 0.4
    REPLAY_BUFFER_CAPACITY = 10000 # 存储 10,000 个 *回合*
    
    # Epsilon 衰减配置
    TOTAL_TRAIN_STEPS = TOTAL_TASKS * STEPS_PER_TASK
    PER_BETA_FRAMES = TOTAL_TRAIN_STEPS
    EPSILON_START = 1.0
    EPSILON_MIN = 0.1
    EPSILON_DECAY = 0.999999955 # 保持一致

    # 日志和保存配置
    PRINT_FREQ = 100
    SAVE_FREQ = 1000
    MODEL_SAVE_PATH = 'dqn_ablation_agent.pth' 
    
    # --- 2. 初始化 ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- DQN (无记忆) 消融实验 ---")
    print(f"使用 Device: {device}")
    print(f"总任务数: {TOTAL_TASKS}, 每任务训练步数: {STEPS_PER_TASK}")
    print(f"总训练步数: {TOTAL_TRAIN_STEPS}")
    print(f"关键参数: SEQ_LEN = {SEQ_LEN}")

    # 创建全局回放池
    # [注意] 即使 SEQ_LEN=1, 这个回放池实现仍然有效
    global_buffer = PrioritizedReplayBuffer(
        REPLAY_BUFFER_CAPACITY,
        SEQ_LEN,
        NUM_AGENTS,
        STATE_SHAPE,
        device,
        alpha=PER_ALPHA,
        beta_start=PER_BETA_START,
        beta_frames=PER_BETA_FRAMES
    )

    # [关键区别]
    # 创建 *一个* 持续存在的 DQNAgent
    agent = DQNAgent(
        NUM_AGENTS,
        STATE_SHAPE,
        NUM_ACTIONS,
        global_buffer,
        LR
    )
    agent.batch_size = BATCH_SIZE
    agent.epsilon = EPSILON_START
    agent.epsilon_min = EPSILON_MIN
    agent.epsilon_decay = EPSILON_DECAY

    # --- 3. 加载检查点 (如果存在) ---
    start_task = 0
    total_steps_trained = 0
    
    if os.path.exists(MODEL_SAVE_PATH):
        print(f"正在从 '{MODEL_SAVE_PATH}' 加载检查点...")
        try:
            checkpoint = torch.load(MODEL_SAVE_PATH, map_location=device)
            agent.q_net.load_state_dict(checkpoint['model_state_dict'])
            agent.target_q_net.load_state_dict(checkpoint['model_state_dict'])
            agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            agent.epsilon = checkpoint['epsilon']
            start_task = checkpoint['tasks_processed'] + 1
            total_steps_trained = checkpoint['total_steps_trained']
            print(f"成功加载。将从 Task {start_task} 和 Epsilon {agent.epsilon:.4f} 处恢复。")
        except Exception as e:
            print(f"加载检查点失败: {e}。将从头开始训练。")
            start_task = 0
            total_steps_trained = 0

    # --- 4. 训练循环 ---
    print(f"开始训练, 从 Task {start_task} 到 {TOTAL_TASKS}...")
    start_time = time.time()
    all_losses = []
    all_rewards = []

    # 为填充（padding）准备“空”数据
    empty_obs_np = np.zeros((NUM_AGENTS, *STATE_SHAPE), dtype=np.float32) 
    empty_action_np = np.zeros(NUM_AGENTS, dtype=np.int64)
    empty_reward_list = [0.0] * NUM_AGENTS 
    empty_done_list = [True] * NUM_AGENTS

    try:
        for task_i in range(start_task, TOTAL_TASKS):
            # 1. 创建新环境
            task_env, map_type, seed, _ = task_environment.create_task_env()
            
            current_task_rewards = []

            # --- 阶段 1: 收集经验 (EPISODES_PER_TASK 次) ---
            for ep in range(EPISODES_PER_TASK):
                ep_states, ep_actions, ep_rewards, ep_next_states, ep_dones = [], [], [], [], []
                obs, info = task_env.reset()
                
                terminated = [False] * NUM_AGENTS
                truncated = [False] * NUM_AGENTS
                reward_calculator = RewardSet(NUM_AGENTS, device)

                while not (all(terminated) or all(truncated)): # 单个回合
                    obs_np = np.array(obs)
                    states_tensor = torch.tensor(obs_np, dtype=torch.float32, device=device)
                    
                    actions = agent.select_actions(states_tensor)
                    
                    next_obs, rewards, terminated, truncated, info = task_env.step(actions)
                    next_obs_np = np.array(next_obs)

                    rewards_tensor = reward_calculator.calculate_total_reward(rewards, states_tensor, actions) 

                    ep_states.append(obs_np)
                    ep_actions.append(actions)
                    ep_rewards.append(rewards_tensor.tolist())
                    ep_next_states.append(next_obs_np)
                    ep_dones.append(terminated)
                    
                    obs = next_obs
                
                # --- 填充逻辑 ---
                # [注意] 即使 SEQ_LEN=1, 这个逻辑也是必要的，
                # 因为如果回合长度为 0 (虽然不太可能), push 会失败。
                # 保持这个逻辑是安全的。
                episode_length = len(ep_states)
                if episode_length < SEQ_LEN:
                    padding_needed = SEQ_LEN - episode_length
                    ep_states.extend([empty_obs_np] * padding_needed)
                    ep_actions.extend([empty_action_np] * padding_needed)
                    ep_rewards.extend([empty_reward_list] * padding_needed)
                    ep_next_states.extend([empty_obs_np] * padding_needed)
                    ep_dones.extend([empty_done_list] * padding_needed)
                
                # 将数据推送到全局回放池
                agent.replay_buffer.push({
                    'states': ep_states, 'actions': ep_actions, 'rewards': ep_rewards,
                    'next_states': ep_next_states, 'dones': ep_dones
                })
                
                current_task_rewards.append(reward_calculator.total_rewards()) 
            
            task_env.close()
            avg_task_reward = np.mean(current_task_rewards) if current_task_rewards else 0
            all_rewards.append(avg_task_reward)

            # --- 阶段 2: 训练 (STEPS_PER_TASK 次) ---
            current_task_losses = []
            
            # 确保缓冲区中有足够的数据
            if len(agent.replay_buffer) > agent.batch_size:
                for _ in range(STEPS_PER_TASK):
                    loss = agent.train() 
                    if loss is not None:
                        current_task_losses.append(loss)
                    
                    # 在每一步训练后都衰减 Epsilon
                    agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)
                    total_steps_trained += 1

            avg_task_loss = np.mean(current_task_losses) if current_task_losses else 0
            all_losses.append(avg_task_loss)

            # --- 阶段 3: 日志 ---
            if (task_i + 1) % PRINT_FREQ == 0:
                elapsed = time.time() - start_time
                tasks_per_sec = (task_i + 1 - start_task) / elapsed if elapsed > 0 else 0
                
                print(f"Task {task_i + 1}/{TOTAL_TASKS} | "
                      f"Avg Reward (last {PRINT_FREQ}): {np.mean(all_rewards[-PRINT_FREQ:]):.2f} | "
                      f"Avg Loss (last {PRINT_FREQ}): {np.mean(all_losses[-PRINT_FREQ:]):.4f} | "
                      f"Epsilon: {agent.epsilon:.4f} | "
                      f"Total Steps: {total_steps_trained} | "
                      f"Tasks/sec: {tasks_per_sec:.2f} | "
                      f"Buffer: {len(agent.replay_buffer)}/{REPLAY_BUFFER_CAPACITY} | "
                      f"Timestamp: {time.strftime('%Y%m%d %X')}")

            # --- 阶段 4: 保存 ---
            if (task_i + 1) % SAVE_FREQ == 0:
                _save_checkpoint(MODEL_SAVE_PATH, agent, task_i, total_steps_trained)

    except KeyboardInterrupt:
        print(f"\n训练被中断。正在保存当前检查点...")
    
    finally:
        # --- 阶段 5: 最终保存和绘图 ---
        print("训练完成或被中断。正在保存最终模型...")
        final_model_path = 'dqn_ablation_agent_final.pth' # [更改]
        _save_checkpoint(final_model_path, agent, task_i, total_steps_trained)

        # 绘制曲线
        if all_losses:
            plt.figure(figsize=(12, 6))
            plt.plot(all_losses)
            plt.title('DQN Ablation: Task Average Loss (Per Task)') # [更改]
            plt.xlabel('Tasks Processed')
            plt.ylabel('MSE Loss')
            plt.grid(True)
            if len(all_losses) > 100:
                moving_avg = np.convolve(all_losses, np.ones(100)/100, mode='valid')
                plt.plot(np.arange(99, len(all_losses)), moving_avg, label='100-task Moving Average', color='red')
                plt.legend()
            plt.savefig('dqn_ablation_task_loss_curve.png') # [更改]
            print("任务损失曲线图已保存为 dqn_ablation_task_loss_curve.png")
        
        if all_rewards:
            plt.figure(figsize=(12, 6))
            plt.plot(all_rewards)
            plt.title('DQN Ablation: Task Average Reward (Per Task)') # [更改]
            plt.xlabel('Tasks Processed')
            plt.ylabel('Average Reward')
            plt.grid(True)
            if len(all_rewards) > 100:
                moving_avg = np.convolve(all_rewards, np.ones(100)/100, mode='valid')
                plt.plot(np.arange(99, len(all_rewards)), moving_avg, label='100-task Moving Average', color='red')
                plt.legend()
            plt.savefig('dqn_ablation_task_reward_curve.png') # [更改]
            print("任务奖励曲线图已保存为 dqn_ablation_task_reward_curve.png")
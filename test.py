# -*- coding: utf-8 -*-
"""
重写后的测试脚本 (test.py)

此脚本用于对四种模型进行全面的消融实验评估：
1.  Reptile-DRQN (元学习, 有记忆)
2.  Reptile-DQN  (元学习, 无记忆)
3.  DRQN Ablation (标准训练, 有记忆)
4.  DQN Ablation  (标准训练, 无记忆)

测试流程 (N 次):
在 N 个不同的、随机生成的测试环境上：
1.  [阶段 1: 零样本评估]
    - 加载所有 4 个训练好的模型。
    - 在*不*微调的情况下，在同一个新环境上评估所有 4 个模型。
    - 记录分数 (奖励/步数)。
2.  [阶段 2: 适应]
    - *只*对 Reptile-DRQN 和 Reptile-DQN 在该环境上进行微调。
    - DRQN 和 DQN (消融模型) *不*进行微调。
3.  [阶段 3: 适应后评估]
    - 再次在*同一个*环境上评估所有 4 个模型。
    - 记录分数 (奖励/步数)。
4.  [阶段 4: 报告]
    - 打印所有 N 次运行的平均结果。
"""
import task_environment
import torch
import numpy as np
import pogema
import time
import os
from copy import deepcopy

# 导入所有必需的模块
# 确保 module_set.py 包含 DRQNAgent 和 DQNAgent (在之前的步骤中已提供)
from module_set import RewardSet, PrioritizedReplayBuffer, DRQNAgent, DQNAgent

# -------------------------------------------------------------------
# 辅助函数：加载模型权重
# -------------------------------------------------------------------

def load_model_weights(agent_q_net, state_dict_file_path, device):
    """
    一个辅助函数，用于加载权重，能处理两种检查点格式：
    1. 字典: {'model_state_dict': ...}
    2. 直接的状态字典
    """
    if not os.path.exists(state_dict_file_path):
        print(f"!!! 警告: 找不到权重文件 {state_dict_file_path}。智能体将使用随机权重。")
        return False
        
    try:
        checkpoint = torch.load(state_dict_file_path, map_location=device)
        
        load_state_dict = None
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            load_state_dict = checkpoint['model_state_dict']
        else:
            load_state_dict = checkpoint

        agent_q_net.load_state_dict(load_state_dict)
        print(f"成功加载权重 from '{state_dict_file_path}'")
        return True
    except Exception as e:
        print(f"!!! 错误: 加载模型权重 '{state_dict_file_path}' 失败: {e}")
        return False

# -------------------------------------------------------------------
# 测试智能体类 (DRQN - 有记忆)
# -------------------------------------------------------------------

class TestDRQNAgent(DRQNAgent):
    """用于 DRQN 和 Reptile-DRQN 的测试智能体"""
    
    def __init__(self, state_shape, num_actions, state_dict_file_path, 
                 num_agents_to_test, finetune_config):
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # DRQN 的关键参数
        self.seq_len = 24 
        
        # 1. [关键修复] 首先调用 super().__init__()
        # 这会初始化 self.q_net, self.target_q_net, self.optimizer, self.device 等
        finetune_buffer = PrioritizedReplayBuffer(
            finetune_config['buffer_capacity'], 
            self.seq_len, 
            num_agents_to_test, 
            state_shape, 
            self.device
        )
        
        super().__init__(
            num_agents=num_agents_to_test,
            state_shape=state_shape,
            num_actions=num_actions,
            replay_buffer=finetune_buffer,
            lr=finetune_config['lr']
        )
        
        self.finetune_config = finetune_config
        
        # 2. 加载训练好的权重
        load_model_weights(self.q_net, state_dict_file_path, self.device)
        
        # 3. 同步 target_q_net
        self.target_q_net.load_state_dict(self.q_net.state_dict())

    def fine_tune_on_task(self, task_env):
        """
        在给定的单个任务上微调智能体 (仅用于 Reptile 模型)
        """
        num_episodes = self.finetune_config['num_episodes']
        num_steps_per_train = self.finetune_config['num_steps_per_train']
        
        print(f"  > [DRQN] 开始微调 {num_episodes} 个回合...")
        self.epsilon = 0.5 # 微调初始 epsilon
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.99
        
        # 为填充（padding）准备“空”数据
        empty_obs_np = np.zeros((self.num_agents, *self.state_shape), dtype=np.float32) 
        empty_action_np = np.zeros(self.num_agents, dtype=np.int64)
        empty_reward_list = [0.0] * self.num_agents 
        empty_done_list = [True] * self.num_agents

        for ep in range(num_episodes):
            ep_states, ep_actions, ep_rewards, ep_next_states, ep_dones = [], [], [], [], []
            obs, info = task_env.reset()
            current_hidden_state = None
            terminated = [False] * self.num_agents
            truncated = [False] * self.num_agents
            reward_calculator = RewardSet(self.num_agents, device=self.device)

            while not (all(terminated) or all(truncated)):
                obs_np = np.array(obs)
                states_tensor = torch.tensor(obs_np, dtype=torch.float32, device=self.device)
                
                actions, new_hidden_state = self.select_actions(states_tensor, current_hidden_state)
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

            # 填充逻辑
            if episode_length < self.seq_len:
                padding_needed = self.seq_len - episode_length
                ep_states.extend([empty_obs_np] * padding_needed)
                ep_actions.extend([empty_action_np] * padding_needed)
                ep_rewards.extend([empty_reward_list] * padding_needed)
                ep_next_states.extend([empty_obs_np] * padding_needed)
                ep_dones.extend([empty_done_list] * padding_needed)
            
            self.replay_buffer.push({
                'states': ep_states, 'actions': ep_actions, 'rewards': ep_rewards,
                'next_states': ep_next_states, 'dones': ep_dones
            })
            
            # 收集数据后进行训练
            if len(self.replay_buffer) > 0:
                for _ in range(num_steps_per_train): 
                    self.train() # 调用 DRQNAgent.train()
        
        print(f"  > [DRQN] 微调完成。")

    def evaluate(self, task_env, render_animation=False, animation_filename="task_evaluation.svg"):
        """评估智能体 (零探索)"""
        
        if render_animation:
            try:
                # 复制环境以进行渲染，避免污染原始环境
                env_to_render = pogema.AnimationMonitor(deepcopy(task_env))
            except Exception as e:
                print(f"  > 无法启动 AnimationMonitor ({e})，将使用普通环境。")
                env_to_render = deepcopy(task_env)
        else:
            env_to_render = deepcopy(task_env) # 始终使用副本进行评估

        states, info = env_to_render.reset()
        terminated = [False] * self.num_agents
        truncated = [False] * self.num_agents
        current_hidden_state = None
        step_count = 0

        original_epsilon = self.epsilon
        self.epsilon = 0.0 # [关键] 零探索
        self.q_net.eval() 
        reward_calculator = RewardSet(self.num_agents, device=self.device)

        with torch.no_grad():
            while not (all(terminated) or all(truncated)):
                states_tensor = torch.tensor(np.array(states), dtype=torch.float32, device=self.device)
                
                actions_np, new_hidden_state = self.select_actions(states_tensor, current_hidden_state)
                current_hidden_state = new_hidden_state
                
                next_states, rewards, terminated, truncated, info = env_to_render.step(actions_np)
                rewards_tensor = reward_calculator.calculate_total_reward(rewards, states_tensor, actions_np)
                states = next_states
                step_count += 1

        self.q_net.train() # 恢复训练模式
        self.epsilon = original_epsilon 
        env_to_render.close() # 关闭副本
        
        total_reward = reward_calculator.total_rewards()
        print(f"  > [DRQN] 评估完成! 总步数: {step_count}, 总奖励: {total_reward:.2f}")

        if render_animation and isinstance(env_to_render, pogema.AnimationMonitor):
            try:
                env_to_render.save_animation(animation_filename)
                print(f"  > 动画已保存至 {animation_filename}\n")
            except Exception as e:
                print(f"  > 保存动画时出错: {e}")
        
        return total_reward, step_count

# -------------------------------------------------------------------
# 测试智能体类 (DQN - 无记忆)
# -------------------------------------------------------------------

class TestDQNAgent(DQNAgent):
    """用于 DQN 和 Reptile-DQN 的测试智能体"""
    
    def __init__(self, state_shape, num_actions, state_dict_file_path, 
                 num_agents_to_test, finetune_config):
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # DQN 的关键参数
        self.seq_len = 1
        
        # 1. [关键修复] 首先调用 super().__init__()
        finetune_buffer = PrioritizedReplayBuffer(
            finetune_config['buffer_capacity'], 
            self.seq_len, 
            num_agents_to_test, 
            state_shape, 
            self.device
        )
        
        super().__init__(
            num_agents=num_agents_to_test,
            state_shape=state_shape,
            num_actions=num_actions,
            replay_buffer=finetune_buffer,
            lr=finetune_config['lr']
        )
        
        self.finetune_config = finetune_config
        
        # 2. 加载训练好的权重
        load_model_weights(self.q_net, state_dict_file_path, self.device)
        
        # 3. 同步 target_q_net
        self.target_q_net.load_state_dict(self.q_net.state_dict())

    def fine_tune_on_task(self, task_env):
        """
        在给定的单个任务上微调智能体 (仅用于 Reptile 模型)
        """
        num_episodes = self.finetune_config['num_episodes']
        num_steps_per_train = self.finetune_config['num_steps_per_train']
        
        print(f"  > [DQN] 开始微调 {num_episodes} 个回合...")
        self.epsilon = 0.5
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.99
        
        empty_obs_np = np.zeros((self.num_agents, *self.state_shape), dtype=np.float32) 
        empty_action_np = np.zeros(self.num_agents, dtype=np.int64)
        empty_reward_list = [0.0] * self.num_agents 
        empty_done_list = [True] * self.num_agents

        for ep in range(num_episodes):
            ep_states, ep_actions, ep_rewards, ep_next_states, ep_dones = [], [], [], [], []
            obs, info = task_env.reset()
            # [DQN 区别] 无隐藏状态
            terminated = [False] * self.num_agents
            truncated = [False] * self.num_agents
            reward_calculator = RewardSet(self.num_agents, device=self.device)

            while not (all(terminated) or all(truncated)):
                obs_np = np.array(obs)
                states_tensor = torch.tensor(obs_np, dtype=torch.float32, device=self.device)
                
                # [DQN 区别] select_actions 无隐藏状态
                actions = self.select_actions(states_tensor)
                
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

            # 填充 (即使 seq_len=1, 也需要以防 episode_length=0)
            if episode_length < self.seq_len:
                padding_needed = self.seq_len - episode_length
                ep_states.extend([empty_obs_np] * padding_needed)
                ep_actions.extend([empty_action_np] * padding_needed)
                ep_rewards.extend([empty_reward_list] * padding_needed)
                ep_next_states.extend([empty_obs_np] * padding_needed)
                ep_dones.extend([empty_done_list] * padding_needed)
            
            self.replay_buffer.push({
                'states': ep_states, 'actions': ep_actions, 'rewards': ep_rewards,
                'next_states': ep_next_states, 'dones': ep_dones
            })
            
            if len(self.replay_buffer) > 0:
                for _ in range(num_steps_per_train): 
                    self.train() # 调用 DQNAgent.train()
        
        print(f"  > [DQN] 微调完成。")

    def evaluate(self, task_env, render_animation=False, animation_filename="task_evaluation.svg"):
        """评估智能体 (零探索)"""
        
        if render_animation:
            try:
                env_to_render = pogema.AnimationMonitor(deepcopy(task_env))
            except Exception as e:
                print(f"  > 无法启动 AnimationMonitor ({e})，将使用普通环境。")
                env_to_render = deepcopy(task_env)
        else:
            env_to_render = deepcopy(task_env)

        states, info = env_to_render.reset()
        terminated = [False] * self.num_agents
        truncated = [False] * self.num_agents
        # [DQN 区别] 无隐藏状态
        step_count = 0

        original_epsilon = self.epsilon
        self.epsilon = 0.0 # [关键] 零探索
        self.q_net.eval() 
        reward_calculator = RewardSet(self.num_agents, device=self.device)

        with torch.no_grad():
            while not (all(terminated) or all(truncated)):
                states_tensor = torch.tensor(np.array(states), dtype=torch.float32, device=self.device)
                
                # [DQN 区别] select_actions 无隐藏状态
                actions_np = self.select_actions(states_tensor)
                
                next_states, rewards, terminated, truncated, info = env_to_render.step(actions_np)
                rewards_tensor = reward_calculator.calculate_total_reward(rewards, states_tensor, actions_np)
                states = next_states
                step_count += 1

        self.q_net.train() 
        self.epsilon = original_epsilon 
        env_to_render.close()
        
        total_reward = reward_calculator.total_rewards()
        print(f"  > [DQN] 评估完成! 总步数: {step_count}, 总奖励: {total_reward:.2f}")

        if render_animation and isinstance(env_to_render, pogema.AnimationMonitor):
            try:
                env_to_render.save_animation(animation_filename)
                print(f"  > 动画已保存至 {animation_filename}\n")
            except Exception as e:
                print(f"  > 保存动画时出错: {e}")
        
        return total_reward, step_count

# -------------------------------------------------------------------
# 主测试工具 (Main Test Harness)
# -------------------------------------------------------------------

if __name__ == "__main__":

    # --- 1. 配置 ---
    
    # 基础配置
    STATE_SHAPE = (3, 11, 11)
    NUM_ACTIONS = 5
    NUM_AGENTS_TO_TEST = 1
    N_TEST_ENVS = 5 # 在 5 个不同的随机环境上运行测试
    
    # 微调配置 (仅用于 Reptile 模型)
    FINETUNE_CONFIG = {
        'lr': 0.00001,
        'buffer_capacity': 500,
        'num_episodes': 20,       # 在新任务上收集 20 个回合的数据
        'num_steps_per_train': 32 # 每个回合后训练 32 步
    }
    
    # [重要] 确保这些路径指向你训练好的模型文件
    AGENT_PATHS = {
        "reptile_drqn": "reptile_drqn.pth",
        "reptile_dqn": "reptile_dqn.pth", # 确保 mpreptile_dqn.py 保存到这个文件！
        "drqn_ablation": "drqn_ablation_agent_final.pth",
        "dqn_ablation": "dqn_ablation_agent_final.pth",
    }
    
    AGENT_CLASSES = {
        "reptile_drqn": TestDRQNAgent,
        "reptile_dqn": TestDQNAgent,
        "drqn_ablation": TestDRQNAgent,
        "dqn_ablation": TestDQNAgent,
    }

    # 结果存储
    results = {}
    for name in AGENT_PATHS:
        results[name] = {
            "zero_shot_reward": [], 
            "adapted_reward": [], 
            "zero_shot_steps": [], 
            "adapted_steps": []
        }

    # --- 2. 运行测试循环 ---
    
    print("="*50)
    print(f"开始在 {N_TEST_ENVS} 个新环境上进行全面评估...")
    print("="*50)
    
    main_start_time = time.time()

    for i in range(N_TEST_ENVS):
        run_start_time = time.time()
        print(f"\n--- 开始测试第 {i+1}/{N_TEST_ENVS} 轮 ---")
        
        # 1. 创建 *一个* 共享的测试环境
        try:
            # 确保 task_environment.py 在路径中
            env, map_type, seed, _ = task_environment.create_task_env()
            print(f"测试环境已创建: 类型={map_type}, 种子={seed}")
        except Exception as e:
            print(f"!!! 严重错误: 无法创建环境: {e}")
            break
            
        # 2. 加载所有 4 个智能体
        agents_to_test = {}
        for name, path in AGENT_PATHS.items():
            AgentClass = AGENT_CLASSES[name]
            agents_to_test[name] = AgentClass(
                STATE_SHAPE, NUM_ACTIONS, path, NUM_AGENTS_TO_TEST, FINETUNE_CONFIG
            )

        # 3. [阶段 1: 零样本评估]
        print("\n--- [阶段 1: 零样本评估 (无微调)] ---")
        for name, agent in agents_to_test.items():
            print(f"正在评估 {name} (零样本)...")
            render = (i == 0) # 仅在第一轮渲染动画
            reward, steps = agent.evaluate(
                env, 
                render_animation=render, 
                animation_filename=f"{name}_run{i}_zero_shot.svg"
            )
            results[name]["zero_shot_reward"].append(reward)
            results[name]["zero_shot_steps"].append(steps)

        # 4. [阶段 2: 适应]
        print("\n--- [阶段 2: 适应 (仅微调 Reptile 模型)] ---")
        for name, agent in agents_to_test.items():
            if name.startswith("reptile"):
                print(f"正在微调 {name}...")
                agent.fine_tune_on_task(env)
            else:
                print(f"{name} (消融模型) 跳过微调。")

        # 5. [阶段 3: 适应后评估]
        print("\n--- [阶段 3: 适应后评估] ---")
        for name, agent in agents_to_test.items():
            print(f"正在评估 {name} (适应后)...")
            render = (i == 0) # 仅在第一轮渲染动画
            reward, steps = agent.evaluate(
                env, 
                render_animation=render, 
                animation_filename=f"{name}_run{i}_adapted.svg"
            )
            results[name]["adapted_reward"].append(reward)
            results[name]["adapted_steps"].append(steps)

        # 6. 清理
        env.close()
        del agents_to_test
        print(f"--- 第 {i+1} 轮测试完成 (耗时: {time.time() - run_start_time:.2f} 秒) ---")

    # --- 3. 报告最终平均结果 ---
    
    print("\n" + "="*50)
    print(f"所有 {N_TEST_ENVS} 轮测试完成 (总耗时: {time.time() - main_start_time:.2f} 秒)")
    print("最终平均结果:")
    print("="*50)

    for name in results:
        print(f"\n--- 结果: {name} ---")
        
        # 奖励
        zsr_avg = np.mean(results[name]['zero_shot_reward'])
        ar_avg = np.mean(results[name]['adapted_reward'])
        print(f"  [奖励] 零样本 (Avg): {zsr_avg:>8.2f}")
        print(f"  [奖励] 适应后 (Avg): {ar_avg:>8.2f}")
        print(f"  [奖励] 提升: {ar_avg - zsr_avg:>+8.2f}")
        
        # 步数
        zss_avg = np.mean(results[name]['zero_shot_steps'])
        ass_avg = np.mean(results[name]['adapted_steps'])
        print(f"  [步数] 零样本 (Avg): {zss_avg:>8.2f}")
        print(f"  [步数] 适应后 (Avg): {ass_avg:>8.2f}")
        print(f"  [步数] 提升: {ass_avg - zss_avg:>+8.2f} (越低越好)")
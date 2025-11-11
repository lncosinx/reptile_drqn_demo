import mpreptile_optimized
import task_environment
import torch
import numpy as np
import pogema
from module_set import  RewardSet, ReplayBuffer, DRQNAgent

class TestAgent(DRQNAgent):
    def __init__(self, state_shape, num_actions, state_dict_file_path, finetune_state_dict_file_path, finetune_lr, finetune_buffer_capacity, seq_len, num_agents_to_test):
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.seq_len = seq_len
        
        finetune_buffer = ReplayBuffer(finetune_buffer_capacity, self.seq_len, num_agents_to_test, state_shape, self.device)
        
        self.finetune_state_dict_file_path = finetune_state_dict_file_path

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

        # 为填充（padding）准备“空”数据
        # (N, C, H, W)
        empty_obs_np = np.zeros((self.num_agents, *self.state_shape), dtype=np.float32) 
        # (N,)
        empty_action_np = np.zeros(self.num_agents, dtype=np.int64)
        # (N,)
        empty_reward_list = [0.0] * self.num_agents 
        # (N,)
        empty_done_list = [True] * self.num_agents

        for ep in range(num_episodes):
            ep_states, ep_actions, ep_rewards, ep_next_states, ep_dones = [], [], [], [], []
            obs, info = task_env.reset()
            current_hidden_state = None
            terminated = [False] * self.num_agents
            truncated = [False] * self.num_agents
            reward_calculator = RewardSet()

            while not (all(terminated) or all(truncated)): # 修改终止条件以确保达到目标数量
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
            self.replay_buffer.push({
                'states': ep_states, 'actions': ep_actions, 'rewards': ep_rewards,
                'next_states': ep_next_states, 'dones': ep_dones
            })
            
            current_task_episodes += 1

            if len(self.replay_buffer) >= self.batch_size:
                for _ in range(num_steps_per_train): 
                    loss = self.train()
                    if loss is not None:
                         all_finetune_losses.append(loss)
                         total_steps_trained += 1

        avg_finetune_loss = np.mean(all_finetune_losses) if all_finetune_losses else 0
        print(f"微调完成。共训练 {total_steps_trained} 步。平均损失: {avg_finetune_loss:.4f}")

    def _save_fine_tune_checkpoint(self):
        try:
            path = self.finetune_state_dict_file_path
            checkpoint = {
                'model_state_dict': self.q_net.state_dict(),
            }
            torch.save(checkpoint, path)
            print(f"--- 微调模型已保存到 {path} ")
        except Exception as e:
            print(f"警告：微调模型未保存到 {path} 失败: {e}")

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
        current_hidden_state = None
        step_count = 0

        original_epsilon = self.epsilon
        self.epsilon = 0.0 
        self.q_net.eval() 
        num_get_obs_rewards_list = [0] * self.num_agents
        reward_calculator = RewardSet()

        with torch.no_grad():
            while not (all(terminated) or all(truncated)):
                states_tensor = torch.tensor(np.array(states), dtype=torch.float32, device=self.device)
                
                actions_np, new_hidden_state = self.select_actions(states_tensor, current_hidden_state)
                current_hidden_state = new_hidden_state
                next_states, rewards, terminated, truncated, info = env.step(actions_np)
                rewards_tensor = reward_calculator.calculate_total_reward(rewards, states_tensor, actions_np)
                states = next_states
                step_count += 1


        self.q_net.train() 
        self.epsilon = original_epsilon 
        
        print(f"评估完成! 总步数: {step_count}, 总奖励: {reward_calculator.total_rewards()}")

        if render_animation and isinstance(env, pogema.AnimationMonitor):
            try:
                env.save_animation(animation_filename)
                print(f"动画已保存至 {animation_filename}\n")
            except Exception as e:
                print(f"保存动画时出错: {e}")
        
        return reward_calculator.total_rewards(), step_count

if __name__ == "__main__":

    test_agent = mpreptile_optimized.TestAgent(
            state_shape=(3, 11, 11),
            num_actions=5,
            state_dict_file_path='reptile_drqn_meta_agent_interrupt.pth',
            finetune_state_dict_file_path='test_agent_finetuned.pth',
            finetune_lr=0.00001,
            finetune_buffer_capacity=500,
            seq_len=24,
            num_agents_to_test=1
        )
    # 重新设置batch size防止len(self.replay_buffer) >= self.batch_size永远为False
    # 因为这里的self.batch_size在DRQNAgent中被设置为256，导致这个条件不会被触发
    test_agent.batch_size = 4   
    env, _ ,_ ,_ = task_environment.create_task_env()
    test_agent.evaluate(env, render_animation=True, animation_filename="not_fine_tune.svg")
    test_agent.fine_tune_on_task(env, num_episodes=20, num_steps_per_train=32)                # 微调
    test_agent.evaluate(env, render_animation=True, animation_filename="fine_tuned.svg")
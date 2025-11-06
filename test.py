import mpreptile_optimized
import task_environment
import torch
import numpy as np
from mpreptile_optimized import DRQNAgent
from mpreptile_optimized import goal_in_obs_reward_multi_agent
from mpreptile_optimized import ReplayBuffer
import pogema


class TestAgent(DRQNAgent):
    def __init__(self, state_shape, num_actions, state_dict_file_path, finetune_state_dict_file_path, finetune_lr, finetune_buffer_capacity, seq_len, num_agents_to_test):
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        finetune_buffer = ReplayBuffer(finetune_buffer_capacity, seq_len, num_agents_to_test, state_shape, self.device)
        
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

        for ep in range(num_episodes):
            ep_states, ep_actions, ep_rewards, ep_next_states, ep_dones = [], [], [], [], []
            obs, info = task_env.reset()
            current_hidden_state = None
            terminated = [False] * self.num_agents
            truncated = [False] * self.num_agents
            ep_len = 0
            episode_reward_tensor = torch.zeros(self.num_agents, dtype=torch.float32, device=self.device)
            num_get_obs_rewards_list = [0] * self.num_agents

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
                ep_len += 1
                
                obs = next_obs
                goal_rewards_tensor, num_get_obs_rewards_list = goal_in_obs_reward_multi_agent(states_tensor, num_get_obs_rewards_list, self.device)
                rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=self.device)
                total_step_rewards = rewards_tensor + goal_rewards_tensor
                episode_reward_tensor += total_step_rewards

                for i in range(self.num_agents):   
                    if rewards[i] :
                        num_get_obs_rewards_list[i] = 0

            if len(ep_states) >= self.replay_buffer.seq_len:
                self.replay_buffer.push({
                    'states': ep_states, 'actions': ep_actions, 'rewards': ep_rewards,
                    'next_states': ep_next_states, 'dones': ep_dones
                })
                print(f"  微调回合 {ep+1}/{num_episodes} 完成 (长度: {ep_len}, 奖励: {episode_reward_tensor.cpu().numpy()})。Buffer: {len(self.replay_buffer)}")
            else:
                 print(f"  微调回合 {ep+1}/{num_episodes} 完成 (长度: {ep_len}, 奖励: {episode_reward_tensor.cpu().numpy()})。回合太短，已丢弃。")


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
        episode_reward_tensor = torch.zeros(self.num_agents, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            while not (all(terminated) or all(truncated)):
                states_tensor = torch.tensor(np.array(states), dtype=torch.float32, device=self.device)
                
                actions_np, new_hidden_state = self.select_actions(states_tensor, current_hidden_state)
                current_hidden_state = new_hidden_state
                next_states, rewards, terminated, truncated, info = env.step(actions_np)
                goal_rewards_tensor, num_get_obs_rewards_list = goal_in_obs_reward_multi_agent(states_tensor, num_get_obs_rewards_list, self.device)
                rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=self.device)
                total_step_rewards = rewards_tensor + goal_rewards_tensor
                episode_reward_tensor += total_step_rewards
                states = next_states
                step_count += 1

                for i in range(self.num_agents):   
                    if rewards[i] :
                        num_get_obs_rewards_list[i] = 0


        self.q_net.train() 
        self.epsilon = original_epsilon 
        
        print(f"评估完成! 总步数: {step_count}, 总奖励: {episode_reward_tensor.cpu().numpy()}")

        if render_animation and isinstance(env, pogema.AnimationMonitor):
            try:
                env.save_animation(animation_filename)
                print(f"动画已保存至 {animation_filename}\n")
            except Exception as e:
                print(f"保存动画时出错: {e}")
        
        return episode_reward_tensor.cpu().numpy(), step_count

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
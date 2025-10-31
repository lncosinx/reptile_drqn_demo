import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import time
import mpreptile

num_tasks_per_set = 20000

task_sets_config = {
    "T1":   {"num_targets": 2, "num_agents": 1, "density": 0.1, "width": 12, "height": 12, "obs_radius": 5},
    "T2":   {"num_targets": 4, "num_agents": 1, "density": 0.15, "width": 16, "height": 16, "obs_radius": 5},
    "T3":   {"num_targets": 6, "num_agents": 1, "density": 0.2, "width": 20, "height": 20, "obs_radius": 5},
    "T4":   {"num_targets": 8, "num_agents": 1, "density": 0.25, "width": 24, "height": 24, "obs_radius": 5},
    "T5":   {"num_targets": 4, "num_agents": 1, "density": 0.3, "width": 16, "height": 16, "obs_radius": 5},
}

test_agent = mpreptile.TestAgent(
        state_shape=(3, 11, 11),
        num_actions=5,
        state_dict_file_path='reptile_drqn_meta_agent_interrupt.pth',
        finetune_lr=0.00001,
        finetune_buffer_capacity=500,
        seq_len=8,
        num_agents_to_test=1
    )
env = test_agent.task_env(task_sets_config['T1'])
test_agent.evaluate(env, render_animation=True, animation_filename="not_fine_tune.svg")
test_agent.fine_tune_on_task(env, num_episodes=20, num_steps_per_train=32)                # 微调
test_agent.evaluate(env, render_animation=True, animation_filename="fine_tuned.svg")
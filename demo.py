
from pogema import pogema_v0, GridConfig, AStarAgent, AnimationMonitor

env = pogema_v0(GridConfig(
    num_agents=1,
    size =8,
    obs_radius=5,
    on_target='restart',
    observation_type='POMAPF'
))

env = AnimationMonitor(env)

num_target = 5

obs, info = env.reset()

agent = AStarAgent()

while True:
    obs, reward, terminated, truncated, info = env.step([agent.act(obs[0])])
    env.render()
    if (all(terminated) or all (truncated)) or (reward == num_target): 
        break
        
env.save_animation('demo.svg')
from pogema_toolbox.registry import ToolboxRegistry
import random

# 导入地图生成器
from pogema_toolbox.generators.maze_generator import MazeGenerator
from pogema_toolbox.generators.random_generator import generate_map as generate_random_map
from pogema_toolbox.generators.house_generator import HouseGenerator
from pogema_toolbox.generators.warehouse_generator import generate_warehouse, WarehouseConfig

# 导入环境创建工具
# 注册默认环境，以便 create_env_base 正常工作
from pogema_toolbox.create_env import create_env_base, Environment
ToolboxRegistry.register_env('Pogema-v0', create_env_base, Environment)


def create_task_env():
    """
    动态创建一个随机任务环境。
    每次调用此函数时，都会随机选择一个地图类型，
    并为该类型随机生成一套新参数。
    """
    
    # 1. 随机选择一个地图类型
    map_type = random.choice(['maze', 'random', 'house', 'warehouse'])
    
    map_str = None
    seed = None
    num_targets = None
    max_episode_steps = None

    # 2. 根据所选类型，“即时”生成随机参数
    if map_type == 'maze':
        # --- 迷宫参数 ---
        width = random.randint(8, 20)
        height = random.randint(8, 20)
        seed = random.randint(0, 1_000_000)
        num_targets = random.randint(1, 5)
        max_episode_steps = 400 # (迷宫通常更难，步数可以少点)
        
        map_str = MazeGenerator.generate_maze(
            width=width,
            height=height,
            obstacle_density=random.uniform(0.1, 0.3),
            wall_components=random.randint(2, 8),
            go_straight=random.uniform(0.7, 0.9),
            seed=seed
        )
        
    elif map_type == 'random':
        # --- 随机地图参数 ---
        width = random.randint(10, 30)
        height = random.randint(10, 30)
        seed = random.randint(0, 1_000_000)
        num_targets = random.randint(1, 5)
        max_episode_steps = 900 # (随机地图通常更大)
        
        # generate_random_map 需要一个设置字典
        random_settings = {
            "width": width,
            "height": height,
            "obstacle_density": random.uniform(0.05, 0.3),
            "seed": seed
        }
        map_str = generate_random_map(random_settings)

    elif map_type == 'house':
        # --- 房屋地图参数 ---
        width = random.randint(15, 30)
        height = random.randint(15, 30)
        seed = random.randint(0, 1_000_000)
        num_targets = random.randint(1, 5)
        max_episode_steps = 900

        map_str = HouseGenerator.generate(
            width=width,
            height=height,
            obstacle_ratio=random.randint(4, 8),
            remove_edge_ratio=random.randint(4, 10),
            seed=seed
            # 错误修复：移除了 max_episode_steps
        )

    elif map_type == 'warehouse':
        # --- 仓库地图参数 ---
        # 仓库地图是确定性的，但智能体放置需要种子
        seed = random.randint(0, 1_000_000) 
        num_targets = random.randint(1, 5)
        max_episode_steps = 1914 # (仓库地图可能非常大)

        wh_config = WarehouseConfig(
            wall_width=random.randint(3, 8),
            wall_height=random.randint(2, 3),
            walls_in_row=random.randint(2, 5),
            walls_rows=random.randint(2, 5), 
            horizontal_gap=random.randint(1, 2),
            vertical_gap=random.randint(2, 3)
        )
        map_str = generate_warehouse(wh_config)
    
    # 3. 创建环境配置
    env_config = Environment(
        map=map_str,               # 直接传入地图字符串
        num_agents=1,
        obs_radius=5,
        on_target='restart',
        use_maps=False,            # 关键：告诉 create_env_base 不要查找 map_name
        seed=seed,                 # 关键：用于 agent 和 target 的随机放置
        num_targets=num_targets,   # 传入 num_targets
        max_episode_steps=max_episode_steps # 在这里传入max_episode_steps
    )

    # 4. 创建环境实例
    task_env = create_env_base(config=env_config)
    
    # 返回环境和一些元数据，可用于 mpreptile_optimized.py 的 worker
    return task_env, map_type, seed, num_targets

if __name__ == '__main__':
    # --- 测试代码 ---
    print("--- 正在测试动态环境生成器 ---")
    
    env_list = []
    
    # 让我们创建 5 个随机环境并打印它们的类型
    for i in range(5):
        env, map_type, seed, num_targets = create_task_env()
        print(f"测试 {i+1}:")
        print(f"  - 类型: {map_type}")
        print(f"  - 种子: {seed}")
        print(f"  - 目标: {num_targets}")
        env_list.append(env)
    
    print("\n--- 测试完成 ---")
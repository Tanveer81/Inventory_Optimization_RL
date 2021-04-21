from gym.envs.registration import register

register(
    id="stockManager-v0",
    entry_point="gym_example.envs:StockManager",
)

register(
    id="stockManager-v1",
    entry_point="gym_example.envs:StockManagerSingleAction",
)

register(
    id="stockManager-v2",
    entry_point="gym_example.envs:StockManagerDate",
)

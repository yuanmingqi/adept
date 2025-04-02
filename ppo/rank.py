# import numpy as np
# import random
# import torch
# import gym, os

# from scipy.linalg import svd

# from procgen import ProcgenEnv
# from ppo_procgen import Agent

# def compute_effective_rank(singular_values: np.ndarray):
#     """ Computes the effective rank of the representation layer """

#     norm_sv = singular_values / np.sum(np.abs(singular_values))
#     entropy = 0.0
#     for p in norm_sv:
#         if p > 0.0:
#             entropy -= p * np.log(p)

#     return np.e ** entropy

# def compute_stable_rank(singular_values: np.ndarray):
#     """ Computes the stable rank of the representation layer """
#     sorted_singular_values = np.flip(np.sort(singular_values))
#     cumsum_sorted_singular_values = np.cumsum(sorted_singular_values) / np.sum(singular_values)
#     return np.sum(cumsum_sorted_singular_values < 0.99) + 1

# def compute_dead_units(output):
#     matrix = output.detach().cpu().numpy()
#     return np.where(matrix==0)[0].shape[0]

# # envs = [
# #     'bigfish', 'bossfight', 'caveflyer', 'chaser', 
# #         'climber', 'coinrun', 'dodgeball', 'fruitbot',
# #         'heist', 'jumper', 'leaper', 'maze', 
# #         'miner', 'ninja', 'plunder', 'starpilot'
# #         ]
# envs = ['bigfish']
# best_ucb = {'bigfish': 'ppo_ucb_runs_c=5.0_w=10', 'bossfight': 'ppo_ucb_runs_c=1.0_w=10', 'caveflyer': 'ppo_ucb_runs_c=5.0_w=10', 'chaser': 'ppo_ucb_runs_c=5.0_w=10', 
#             'climber': 'ppo_ucb_runs_c=5.0_w=10', 'coinrun': 'ppo_ucb_runs_c=5.0_w=100', 'dodgeball': 'ppo_ucb_runs_c=5.0_w=100', 'fruitbot': 'ppo_ucb_runs_c=5.0_w=100', 
#             'heist': 'ppo_ucb_runs_c=5.0_w=100', 'jumper': 'ppo_ucb_runs_c=5.0_w=100', 'leaper': 'ppo_ucb_runs_c=5.0_w=100', 'maze': 'ppo_ucb_runs_c=5.0_w=10', 
#             'miner': 'ppo_ucb_runs_c=5.0_w=10', 'ninja': 'ppo_ucb_runs_c=5.0_w=100', 'plunder': 'ppo_ucb_runs_c=5.0_w=10', 'starpilot': 'ppo_ucb_runs_c=5.0_w=10'}

# best_ts = {'bigfish': 'pg_ts_runs_w=50_a=0.1', 'bossfight': 'pg_ts_runs_w5', 'caveflyer': 'pg_ts_runs_w=50_a=0.1', 'chaser': 'pg_ts_runs_w5', 
#            'climber': 'pg_ts_runs_w100', 'coinrun': 'pg_ts_runs_w=50_a=0.1', 'dodgeball': 'pg_ts_runs_w100', 'fruitbot': 'pg_ts_runs_w=50_a=0.1', 
#            'heist': 'pg_ts_runs_w=50_a=0.1', 'jumper': 'pg_ts_runs_w100', 'leaper': 'pg_ts_runs_w100', 'maze': 'pg_ts_runs_w100', 
#            'miner': 'pg_ts_runs_w=50_a=0.1', 'ninja': 'pg_ts_runs_w100', 'plunder': 'pg_ts_runs_w=50_a=0.1', 'starpilot': 'pg_ts_runs_w10'}

# # relu_activation1 = None
# # relu_activation2 = None
# # def hook_fn1(module, input, output):
# #     # 这里output就是ReLU层的激活矩阵
# #     global relu_activation1
# #     relu_activation1 = output
# # def hook_fn2(module, input, output):
# #     # 这里output就是ReLU层的激活矩阵
# #     global relu_activation2
# #     relu_activation2 = output

# all_srank = []
# all_dunits = []

# for env_name in envs:
#     # model_dir = 'all_runs/ppo_ext_runs'
#     model_dir = f'all_runs/ppo_rr_runs'
    
#     env_srank = []
#     env_dunits = []

#     for seed in range(1, 4):
#         random.seed(seed)
#         np.random.seed(seed)
#         torch.manual_seed(seed)
#         torch.backends.cudnn.deterministic = True

#         device = torch.device('cuda:0')
#         envs = ProcgenEnv(num_envs=1, env_name=env_name, num_levels=0, start_level=0, distribution_mode="easy")
#         envs = gym.wrappers.TransformObservation(envs, lambda obs: obs["rgb"])
#         envs.single_action_space = envs.action_space
#         envs.single_observation_space = envs.observation_space["rgb"]
#         envs.is_vector_env = True
#         envs = gym.wrappers.RecordEpisodeStatistics(envs)
#         envs = gym.wrappers.NormalizeReward(envs, gamma=0.999)
#         envs = gym.wrappers.TransformReward(envs, lambda reward: np.clip(reward, -10, 10))

#         agent = Agent(envs).to(device)
        
#         for file in os.listdir(model_dir):
#             if env_name in file and f'__{seed}__' in file and '.npz' not in file:
#                 model_file = f'{model_dir}/{file}/model.pth'
#                 break

#         model = torch.load(model_file, map_location=device)
#         agent.load_state_dict(model)

#         # obs_samples = []
#         # next_obs = torch.Tensor(envs.reset()).to(device)

#         # with torch.no_grad():
#         #     while len(obs_samples) < 5000:
#         #         action = agent.act(next_obs)
#         #         next_obs, reward, next_done, info = envs.step(np.random.randint(0, 16, 1))
#         #         next_obs = torch.Tensor(next_obs).to(device)
#         #         obs_samples.append(next_obs)

#         #     obs_samples = torch.vstack(obs_samples)#.to(device)

#         #     np.save('bigfish.npy', obs_samples.cpu().numpy())
#         #     quit(0)

#         #     # 将钩子注册到ReLU层
#         #     # hook1 = agent.network[4].register_forward_hook(hook_fn1)
#         #     # hook2 = agent.network[6].register_forward_hook(hook_fn2)
#         #     output = agent.act(obs_samples)

#         obs_samples = torch.from_numpy(np.load('bigfish.npy')).to(device)
#         print(obs_samples.size())
#         output = agent.act(obs_samples)

#         singular_values = svd(output.detach().cpu().numpy(), compute_uv=False, lapack_driver="gesvd")
#         stable_rank_after = compute_stable_rank(singular_values)
#         dead_units_after = compute_dead_units(output)
#         # dead_units_after += compute_dead_units(relu_activation2)
#         dead_units_after = (dead_units_after / (256) / 1000)

#         env_srank.append(stable_rank_after)
#         env_dunits.append(dead_units_after)

#         # hook1.remove()
#         # hook2.remove()
    
#     all_srank.append(np.mean(env_srank))
#     all_dunits.append(np.mean(env_dunits))

# print('srank', np.array(all_srank).round(3).tolist())
# print('dunits', np.array(all_dunits).round(3).tolist())
# print(np.mean(all_srank), np.mean(all_dunits))

import numpy as np
import random
import torch
import gym, os

from scipy.linalg import svd

from procgen import ProcgenEnv
from ppo_procgen import Agent

def compute_effective_rank(singular_values: np.ndarray):
    """ Computes the effective rank of the representation layer """

    norm_sv = singular_values / np.sum(np.abs(singular_values))
    entropy = 0.0
    for p in norm_sv:
        if p > 0.0:
            entropy -= p * np.log(p)

    return np.e ** entropy

def compute_stable_rank(singular_values: np.ndarray):
    """ Computes the stable rank of the representation layer """
    sorted_singular_values = np.flip(np.sort(singular_values))
    cumsum_sorted_singular_values = np.cumsum(sorted_singular_values) / np.sum(singular_values)
    return np.sum(cumsum_sorted_singular_values < 0.99) + 1

def compute_dead_units(output):
    matrix = output.detach().cpu().numpy()
    return np.where(matrix==0)[0].shape[0]

envs = [
    'climber', 'dodgeball', 'jumper', 'leaper', 'maze'
        ]
# envs = ['bigfish', 'chaser', 'dodgeball', 'plunder']
best_ucb = {'bigfish': 'ppo_ucb_runs_c=5.0_w=10', 'bossfight': 'ppo_ucb_runs_c=1.0_w=10', 'caveflyer': 'ppo_ucb_runs_c=5.0_w=10', 'chaser': 'ppo_ucb_runs_c=5.0_w=10', 
            'climber': 'ppo_ucb_runs_c=5.0_w=10', 'coinrun': 'ppo_ucb_runs_c=5.0_w=100', 'dodgeball': 'ppo_ucb_runs_c=5.0_w=100', 'fruitbot': 'ppo_ucb_runs_c=5.0_w=100', 
            'heist': 'ppo_ucb_runs_c=5.0_w=100', 'jumper': 'ppo_ucb_runs_c=5.0_w=100', 'leaper': 'ppo_ucb_runs_c=5.0_w=100', 'maze': 'ppo_ucb_runs_c=5.0_w=10', 
            'miner': 'ppo_ucb_runs_c=5.0_w=10', 'ninja': 'ppo_ucb_runs_c=5.0_w=100', 'plunder': 'ppo_ucb_runs_c=5.0_w=10', 'starpilot': 'ppo_ucb_runs_c=5.0_w=10'}

best_ts = {'bigfish': 'pg_ts_runs_w=50_a=0.1', 'bossfight': 'pg_ts_runs_w5', 'caveflyer': 'pg_ts_runs_w=50_a=0.1', 'chaser': 'pg_ts_runs_w5', 
           'climber': 'pg_ts_runs_w100', 'coinrun': 'pg_ts_runs_w=50_a=0.1', 'dodgeball': 'pg_ts_runs_w100', 'fruitbot': 'pg_ts_runs_w=50_a=0.1', 
           'heist': 'pg_ts_runs_w=50_a=0.1', 'jumper': 'pg_ts_runs_w100', 'leaper': 'pg_ts_runs_w100', 'maze': 'pg_ts_runs_w100', 
           'miner': 'pg_ts_runs_w=50_a=0.1', 'ninja': 'pg_ts_runs_w100', 'plunder': 'pg_ts_runs_w=50_a=0.1', 'starpilot': 'pg_ts_runs_w10'}


all_srank = []
all_dunits = []

mapping = {'bigfish': best_ucb['bigfish'], 'chaser': 'ppo_rr_runs', 'dodgeball': best_ucb['dodgeball'], 'plunder': best_ts['plunder']}

sr_tb = ''
du_tb = ''

for env_name in envs:
    # model_dir = 'all_runs/ppo_ext_runs'
    # model_dir = f'all_runs/ppo_rr_runs'
    # model_dir = f'all_runs/{best_ucb[env_name]}'
    model_dir = f'./all_runs/{best_ts[env_name]}'
    # model_dir = f'all_runs/{mapping[env_name]}'

    env_srank = []
    env_dunits = []

    device = torch.device('cuda:0')
    obs_samples = torch.from_numpy(np.load(f'sampled_trajs/{env_name}.npy')).to(device)

    for seed in range(1, 4):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        
        envs = ProcgenEnv(num_envs=1, env_name=env_name, num_levels=0, start_level=0, distribution_mode="easy")
        envs = gym.wrappers.TransformObservation(envs, lambda obs: obs["rgb"])
        envs.single_action_space = envs.action_space
        envs.single_observation_space = envs.observation_space["rgb"]
        envs.is_vector_env = True
        envs = gym.wrappers.RecordEpisodeStatistics(envs)
        envs = gym.wrappers.NormalizeReward(envs, gamma=0.999)
        envs = gym.wrappers.TransformReward(envs, lambda reward: np.clip(reward, -10, 10))

        agent = Agent(envs).to(device)
        
        for file in os.listdir(model_dir):
            if env_name in file and f'__{seed}__' in file and '.npz' not in file:
                model_file = f'{model_dir}/{file}/model.pth'
                break

        model = torch.load(model_file, map_location=device)
        agent.load_state_dict(model)

        seed = seed - 1
        output = agent.act(obs_samples[1000*seed:1000*(seed+1)])

        singular_values = svd(output.detach().cpu().numpy(), compute_uv=False, lapack_driver="gesvd")
        stable_rank_after = compute_stable_rank(singular_values)
        dead_units_after = compute_dead_units(output)
        # dead_units_after += compute_dead_units(relu_activation2)
        dead_units_after = (dead_units_after / (256) / 1000)

        env_srank.append(stable_rank_after)
        env_dunits.append(dead_units_after)
    
    all_srank.append(np.mean(env_srank))
    all_dunits.append(np.mean(env_dunits))

    # print(env_name, 'srank', np.array(env_srank).round(3).tolist())
    # print(env_name, 'dunits', np.array(env_dunits).round(3).tolist())
    # print(model_dir, env_name, 'srank', np.mean(env_srank).round(3), 'dunits', np.mean(env_dunits).round(3))

    sr_item = f'{np.mean(env_srank).round(2)}$\pm${np.std(env_srank).round(2)} |'
    sr_tb += sr_item

    du_item = f'{np.mean(env_dunits).round(2)}$\pm${np.std(env_dunits).round(2)} |'
    du_tb += du_item

    print(env_name, sr_item, du_item)

print(sr_tb)
print(du_tb)
print(model_dir, 'srank', f'{np.mean(all_srank).round(2)}$\pm${np.std(all_srank).round(2)}', 'dunits', f'{np.mean(env_dunits).round(2)}$\pm${np.std(env_dunits).round(2)}')
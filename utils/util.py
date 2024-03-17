import csv
from datetime import datetime
import json
import os
import pickle
from pathlib import Path
import random
import string
import sys

import d4rl
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.distributions as td

# 这里定义了一个名为 DEFAULT_DEVICE 的变量，用于指定默认使用的设备（GPU 或 CPU）。
# 接下来，有一个名为 set_default_device() 的函数，它的作用是将 PyTorch 的默认张量类型设置为 CUDA 张量（即使用 GPU 计算）。
DEFAULT_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def set_default_device():
    """Set the default device.
    """
    torch.set_default_tensor_type(torch.cuda.FloatTensor)


# 这里有两个函数 to_torch_device() 和 torchify()，
# 它们用于将 NumPy 数组转换为 PyTorch 张量，并将它们放置在 DEFAULT_DEVICE 指定的设备上（GPU 或 CPU）。
def to_torch_device(x_np):
    return torch.FloatTensor(x_np)


def torchify(x):
    x = torch.from_numpy(x)
    if x.dtype is torch.float64:
        x = x.float()
    x = x.to(device=DEFAULT_DEVICE)
    return x


# 用于创建回放内存缓冲区，用于存储环境与智能体交互的经验
# 其中包括：
# init：类的构造函数，用于初始化回放内存的容量和随机种子
# push：将经验元组(state，action，reward，next_state,done)存储到回放内存缓冲区中
# sample：从回放内存缓冲区中随机采样一批经验数据，并返回该批数据
# len：返回当前回放内存缓冲区的大小，即存储的经验数量
# save_buffer：将回放内存缓冲区保存到指定路径
# load_buffer: 从指定路径架在回放内存缓冲区
class ReplayMemory:
    def __init__(self, capacity, seed):
        random.seed(seed)
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

    def save_buffer(self, env_name, suffix="", save_path=None):
        if not os.path.exists('checkpoints/'):
            os.makedirs('checkpoints/')

        if save_path is None:
            save_path = "checkpoints/sac_buffer_{}_{}".format(env_name, suffix)
        print('Saving buffer to {}'.format(save_path))

        with open(save_path, 'wb') as f:
            pickle.dump(self.buffer, f)

    def load_buffer(self, save_path):
        print('Loading buffer from {}'.format(save_path))

        with open(save_path, "rb") as f:
            self.buffer = pickle.load(f)
            self.position = len(self.buffer) % self.capacity


# 定义一个名为squeeze的自定义模块，实现对张量进行挤压操作的前向传播函数，可以在给定维度上删除大小为1的维度
class Squeeze(nn.Module):
    def __init__(self, dim=None):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return x.squeeze(dim=self.dim)


# 定义多层感知机MLP函数，采用一个整数列表dims，列表中每个元素表示对应层的维度大小
# 该函数构建了一个包含多个线性层和激活函数的序列，并返回一个构建好的MLP网络

# 这个MLP（多层感知机）网络中总共有 n_dims 层。维度列表 dims 包含了每一层的维度大小，其中 n_dims 为 dims 列表的长度。注意，MLP 要求至少有两个维度（输入和输出维度），因此 n_dims 必须大于等于 2。
#
# 输入层的维度大小为 dims[0]，即输入特征的维度大小。
# 隐藏层的维度大小分别为 dims[1], dims[2], ..., dims[n_dims-3]，其中包括了 n_dims-2 个隐藏层，每个隐藏层的维度分别为 dims[i+1]，其中 i 的取值范围为 0 到 n_dims-3。
# 输出层的维度大小为 dims[-1]，即输出动作或值函数的维度大小。
# 如果指定了 activation，则每个隐藏层都会在其线性变换之后应用激活函数。如果指定了 output_activation，则输出层也会在其线性变换之后应用激活函数。
#
# 如果 squeeze_output 为 True，且输出维度 dims[-1] 为 1，则会在输出层后应用 Squeeze(-1) 操作，
# 将输出张量的最后一个维度压缩掉，使得输出为形状 (batch_size,) 而不是 (batch_size, 1)。
#
# 例如，如果 dims = [input_dim, hidden_dim1, hidden_dim2, ..., output_dim]，
# 那么输入层的维度大小为 input_dim，隐藏层的维度大小分别为 hidden_dim1, hidden_dim2, ...，输出层的维度大小为 output_dim。
# 并且，如果指定了 activation=nn.ReLU，输出层没有激活函数（output_activation=None），并且 squeeze_output=False，那么这个MLP网络的构成如下：
#
# 输入层: 输入维度为 input_dim，无激活函数
# 隐藏层1: 线性层，输入维度 input_dim，输出维度 hidden_dim1，后接 ReLU 激活函数
# 隐藏层2: 线性层，输入维度 hidden_dim1，输出维度 hidden_dim2，后接 ReLU 激活函数
# ...
# 输出层: 线性层，输入维度 hidden_dim_(n_dims-3)，输出维度 output_dim
# 如果指定了 output_activation=nn.Tanh，那么输出层将在线性变换之后应用 Tanh 激活函数。
# 如果同时将 squeeze_output=True，且 output_dim == 1，则输出层后会进行 Squeeze 操作，将输出维度压缩为 (batch_size,)。

def mlp(dims, activation=nn.ReLU, output_activation=None, squeeze_output=False):
    n_dims = len(dims)
    assert n_dims >= 2, 'MLP requires at least two dims (input and output)'

    layers = []
    for i in range(n_dims - 2):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        layers.append(activation())
    layers.append(nn.Linear(dims[-2], dims[-1]))
    if output_activation is not None:
        layers.append(output_activation())
    if squeeze_output:
        assert dims[-1] == 1
        layers.append(Squeeze(-1))
    net = nn.Sequential(*layers)
    net.to(dtype=torch.float32)
    return net


# 上述MLP网络可以修改为attention网络
# class ComplexAttention(nn.Module):
#     def __init__(self, input_dim, hidden_dim, num_attention_layers=2, num_feedforward_layers=2, num_attention_heads=1, dropout=0.1):
#         super(ComplexAttention, self).__init__()
#
#         self.attention_layers = nn.ModuleList([
#             SelfAttention(input_dim, hidden_dim) for _ in range(num_attention_layers)
#         ])
#         self.feedforward_layers = nn.ModuleList([
#             FeedForward(hidden_dim, hidden_dim) for _ in range(num_feedforward_layers)
#         ])
#         self.dropout = nn.Dropout(dropout)
#
#         self.norm1 = nn.LayerNorm(input_dim)
#         self.norm2 = nn.LayerNorm(hidden_dim)
#
#         self.num_attention_layers = num_attention_layers
#         self.num_feedforward_layers = num_feedforward_layers
#
#     def forward(self, x):
#         # 多层自注意力层
#         for i in range(self.num_attention_layers):
#             residual = x
#             x = self.dropout(self.attention_layers[i](x))
#             x = self.norm1(x + residual)
#
#         # 多层前馈神经网络层
#         for i in range(self.num_feedforward_layers):
#             residual = x
#             x = self.dropout(self.feedforward_layers[i](x))
#             x = self.norm2(x + residual)
#
#         return x


# 用于计算目标网络和源网络之间的指数移动平均，它将源网络的参数按照制定的alpha权重加到目标网络的参数中，从而实现参数更新
def update_exponential_moving_average(target, source, alpha):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.mul_(1. - alpha).add_(source_param.data, alpha=alpha)


# 用于计算数据集中的回合（episode）的回报范围。它接受数据集和最大回合步数作为输入，并返回回合回报的最小值和最大值。
def return_range(dataset, max_episode_steps):
    returns, lengths = [], []
    ep_ret, ep_len = 0., 0
    for r, d in zip(dataset['rewards'], dataset['terminals']):
        ep_ret += float(r)
        ep_len += 1
        if d or ep_len == max_episode_steps:
            returns.append(ep_ret)
            lengths.append(ep_len)
            ep_ret, ep_len = 0., 0
    lengths.append(ep_len)
    assert sum(lengths) == len(dataset['rewards'])
    return min(returns), max(returns)


# 用于从数据集中随机采样一批数据。它接受数据集和批量大小作为输入，并返回随机采样得到的一批数据。
def sample_batch(dataset, batch_size):
    k = list(dataset.keys())[0]
    n, device = len(dataset[k]), dataset[k].device
    for v in dataset.values():
        assert len(v) == n, 'Dataset values must have same length'
    indices = torch.randint(low=0, high=n, size=(batch_size,), device=device)
    return {k: v[indices].cuda() for k, v in dataset.items()}


# 用于从回放内存缓冲区中获取一批数据，并将其转换为 PyTorch 张量。
def get_batch_from_buffer(memory, batch_size):
    state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample(batch_size=batch_size)

    state_batch = torch.FloatTensor(state_batch).to(DEFAULT_DEVICE)
    next_state_batch = torch.FloatTensor(next_state_batch).to(DEFAULT_DEVICE)
    action_batch = torch.FloatTensor(action_batch).to(DEFAULT_DEVICE)
    reward_batch = torch.FloatTensor(reward_batch).to(DEFAULT_DEVICE)
    mask_batch = torch.FloatTensor(mask_batch).to(DEFAULT_DEVICE)
    return state_batch, action_batch, next_state_batch, reward_batch, mask_batch


# 用于从数据集和回放内存缓冲区中获取一批数据，并将其转换为 PyTorch 张量。
# double_buffer 参数用于控制是否采用双缓冲方式。
def get_batch_from_dataset_and_buffer(dataset, buffer, batch_size, double_buffer):
    if double_buffer:
        half_batch_size = int(batch_size / 2)
        state_batch, action_batch, next_state_batch, reward_batch, terminals = get_batch_from_buffer(buffer,
                                                                                                     half_batch_size)

        res = sample_batch(dataset, batch_size - half_batch_size)

        state_batch0 = res['observations'].to(DEFAULT_DEVICE)
        action_batch0 = res['actions'].to(DEFAULT_DEVICE)
        reward_batch0 = res['rewards'].to(DEFAULT_DEVICE)
        next_state_batch0 = res['next_observations'].to(DEFAULT_DEVICE)
        terminals0 = res['terminals'].to(DEFAULT_DEVICE)

        state_batch = torch.cat([state_batch0, state_batch], dim=0)
        action_batch = torch.cat([action_batch0, action_batch], dim=0)
        next_state_batch = torch.cat([next_state_batch0, next_state_batch], dim=0)
        reward_batch = torch.cat([reward_batch0, reward_batch], dim=0)
        terminals = torch.cat([terminals0, terminals], dim=0)
    else:
        state_batch, action_batch, next_state_batch, reward_batch, terminals = get_batch_from_buffer(buffer, batch_size)

    return state_batch, action_batch, next_state_batch, reward_batch, terminals


# 用于设置随机种子。
# 它接受一个种子值 seed 和一个环境对象 env 作为输入，并在需要时设置随机种子。
def set_seed(seed, env=None):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    if env is not None:
        env.seed(seed)


# 用于生成一个带有日期和随机字符串的目录名称
def _gen_dir_name():
    now_str = datetime.now().strftime('%m-%d-%y_%H.%M.%S')
    rand_str = ''.join(random.choices(string.ascii_lowercase, k=4))
    return f'{now_str}_{rand_str}'


def get_mode(dist):
    """Get the (transformed) mode of the distribution.
    Borrowed from
    https://github.com/HorizonRobotics/alf/blob/0f8d0ec5d60ef6f30307c6a66ba388852e8c5372/alf/utils/dist_utils.py#L1134
    """
    if isinstance(dist, td.categorical.Categorical):
        mode = torch.argmax(dist.logits, -1)
    elif isinstance(dist, td.normal.Normal):
        mode = dist.mean
    elif isinstance(dist, td.Independent):
        mode = get_mode(dist.base_dist)
    elif isinstance(dist, td.TransformedDistribution):
        base_mode = get_mode(dist.base_dist)
        mode = base_mode
        for transform in dist.transforms:
            mode = transform(mode)
    return mode


def epsilon_greedy_sample(dist, eps=0.1):
    # 生成使用贪婪策略的样本，以最大化概率
    """Generate greedy sample that maximizes the probability.
    Borrowed from
    https://github.com/HorizonRobotics/alf/blob/0f8d0ec5d60ef6f30307c6a66ba388852e8c5372/alf/utils/dist_utils.py#L1106
    """

    def greedy_fn(dist):
        # 使用分布 dist 获取贪婪动作
        greedy_action = get_mode(dist)
        if eps == 0.0:
            return greedy_action
        # 采样动作
        sample_action = dist.sample()

        # 创建一个与sample_action相同形式的掩码，根据概率eps随机生成
        greedy_mask = torch.rand(sample_action.shape[0]) > eps
        # 对于掩码中为true的元素，使用贪婪动作替换采样动作
        sample_action[greedy_mask] = greedy_action[greedy_mask]
        return sample_action

    # 如果eps大于等于1.0，则直接返回采样动作
    if eps >= 1.0:
        return dist.sample()
    else:
        # 否则，使用贪婪策略生成样本
        return greedy_fn(dist)


# extract_sub_dict 函数接受一个前缀 prefix 和一个字典 dict 作为输入，
# 它从字典 dict 中提取以给定前缀 prefix 开头的键值对，并将键名中的前缀去除，然后返回提取后的子字典。
def extract_sub_dict(prefix, dict):
    def _remove_prefix(s, prefix):
        if s.startswith(prefix):
            return s[len(prefix):]
        else:
            return s

    sub_dict = {
        _remove_prefix(k, prefix + '.'): v
        for k, v in dict.items() if k.startswith(prefix)
    }

    return sub_dict


# 用于获取指定环境名称的环境对象和数据集。它调用 gym.make(env_name) 创建环境对象，然后调用 d4rl.qlearning_dataset(env) 获取环境的数据集。
# 接下来，根据环境名称的不同，定义了不同的 reward_transformer 函数用于对奖励值进行变换。
# 最后，将数据集中的数值类型转换为 PyTorch 张量，并返回环境对象、数据集和 reward_transformer 函数。
def get_env_and_dataset(env_name, max_episode_steps):
    env = gym.make(env_name)
    dataset = d4rl.qlearning_dataset(env)

    if any(s in env_name for s in ('halfcheetah', 'hopper', 'walker2d')):
        # min_ret, max_ret = return_range(dataset, max_episode_steps)
        # reward_transformer = lambda x: x * max_episode_steps / (max_ret - min_ret)
        reward_transformer = lambda x: x
    elif 'antmaze' in env_name:
        reward_transformer = lambda x: x - 1

    dataset['rewards'] = reward_transformer(dataset['rewards'])

    for k, v in dataset.items():
        dataset[k] = torchify(v)

    return env, dataset, reward_transformer


# 用于评估策略的性能。它在给定环境和算法的情况下，运行多个回合，并计算回报的平均值和标准差，并输出结果
def eval_policy(env, env_name, alg, max_episode_steps, n_eval_episodes):
    eval_returns = np.array([evaluate_policy(env, alg, max_episode_steps) \
                             for _ in range(n_eval_episodes)])
    normalized_returns = d4rl.get_normalized_score(env_name, eval_returns) * 100.0
    print({
        'return mean': round(eval_returns.mean(), 1),
        'return std': round(eval_returns.std(), 1),
        'normalized return mean': round(normalized_returns.mean(), 1),
        'normalized return std': round(normalized_returns.std(), 1),
    })


# 用于评估指定策略在给定环境中的性能。它运行一个回合，并返回累积的回报值。
def evaluate_policy(env, agent, max_episode_steps, deterministic=True):
    obs = env.reset()
    total_reward = 0.
    # for _ in range(max_episode_steps):
    done = False
    while not done:
        with torch.no_grad():
            action = agent.select_action(torchify(obs), evaluate=deterministic).detach().cpu().numpy()
        next_obs, reward, done, info = env.step(action)
        total_reward += reward
        obs = next_obs
    return total_reward

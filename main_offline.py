import os
from pathlib import Path
import torch
from tqdm import trange

from pex.algorithms.iql import IQL
from pex.networks.policy import GaussianPolicy
from pex.networks.value_functions import DoubleCriticNetwork, ValueNetwork
from pex.utils.util import (
    set_seed, DEFAULT_DEVICE, sample_batch,
    eval_policy, set_default_device, get_env_and_dataset)


def main(args):
    torch.set_num_threads(1)        # 设置pytorch使用的线程数为1

    # 检查日志目录是否存在，如果存在则提示用户指定一个不同的目录并返回，否则创建该目录
    if os.path.exists(args.log_dir):
        print(f"The directory {args.log_dir} exists. Please specify a different one.")
        return
    else:
        print(f"Creating directory {args.log_dir}")
        os.mkdir(args.log_dir)

    # 获取环境（env）和数据集（dataset），其中args.env_name是环境名称，args.max_episode_steps是每个回合的最大步数。
    env, dataset, _ = get_env_and_dataset(args.env_name, args.max_episode_steps)

    # 获取观察空间维度（obs_dim）和动作空间维度（act_dim）。
    obs_dim = dataset['observations'].shape[1]
    act_dim = dataset['actions'].shape[1]

    if args.seed is not None:
        set_seed(args.seed, env=env)        # 如果设置了随机种子（args.seed），则使用该种子初始化环境。

    if torch.cuda.is_available():
        set_default_device()


    # 获取动作空间（action_space）并创建一个高斯策略（GaussianPolicy）。
    action_space = env.action_space
    policy = GaussianPolicy(obs_dim, act_dim, hidden_dim=args.hidden_dim, n_hidden=args.hidden_num, action_space=action_space, scale_distribution=False, state_dependent_std=False)


    # 创建一个IQL实例，包括双重评论网络（DoubleCriticNetwork）、值网络（ValueNetwork）、策略网络（policy）以及其他相关参数。
    iql = IQL(
        critic=DoubleCriticNetwork(obs_dim, act_dim, hidden_dim=args.hidden_dim, n_hidden=args.hidden_num),
        vf=ValueNetwork(obs_dim, hidden_dim=args.hidden_dim, n_hidden=args.hidden_num),
        policy=policy,
        optimizer_ctor=lambda params: torch.optim.Adam(params, lr=args.learning_rate),
        max_steps=args.num_steps,
        tau=args.tau,
        beta=args.beta,
        target_update_rate=args.target_update_rate,
        discount=args.discount
    )
    # 进行args.num_steps次训练步骤，每次从数据集中采样一个批次并更新IQL。每隔args.eval_period步，评估一次策略。
    for step in trange(args.num_steps):
        iql.update(**sample_batch(dataset, args.batch_size))
        if (step + 1) % args.eval_period == 0:
            eval_policy(env, args.env_name, iql, args.max_episode_steps, args.eval_episode_num)


    # 训练完成后，将IQL的状态字典保存到日志目录下的offline_ckpt文件中
    torch.save(iql.state_dict(), args.log_dir + '/offline_ckpt')


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--env_name', required=True)
    parser.add_argument('--log_dir', required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--discount', type=float, default=0.99)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--hidden_num', type=int, default=2)
    parser.add_argument('--num_steps', type=int, default=1000001, metavar='N',
                        help='maximum number of training steps (default: 1000000)')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--target_update_rate', type=float, default=0.005)
    parser.add_argument('--tau', type=float, default=0.7)
    parser.add_argument('--beta', type=float, default=10.0,
                        help='IQL inverse temperature')
    parser.add_argument('--eval_period', type=int, default=10000)
    parser.add_argument('--eval_episode_num', type=int, default=10,
                        help='Number of evaluation episodes (default: 10)')
    parser.add_argument('--max_episode_steps', type=int, default=1000)
    main(parser.parse_args())
from pathlib import Path

import gym
import d4rl
import numpy as np
import itertools
import os
import torch
from tqdm import trange

from pex.algorithms.pex import PEX
from pex.algorithms.iql_online import IQL_online
from pex.networks.policy import GaussianPolicy
from pex.networks.value_functions import DoubleCriticNetwork, ValueNetwork
from pex.utils.util import (
    set_seed, ReplayMemory, torchify, eval_policy, torchify, DEFAULT_DEVICE,
    get_batch_from_dataset_and_buffer,
    eval_policy, set_default_device, get_env_and_dataset)


def main(args):
    torch.set_num_threads(1)

    # main函数接收一个名为args的参数，该参数用于传递命令行输入的参数，然后通过
    # torch.set_num_threads(1)设置pytorch使用的cpu线程数为1

    # 检查制定的args.log_dir是否存在，如果存在则打印一条消息并返回。如果不存在，则打印一条消息并创建该目录
    if os.path.exists(args.log_dir):
        print(f"The directory {args.log_dir} exists. Please specify a different one.")
        return
    else:
        print(f"Creating directory {args.log_dir}")
        os.mkdir(args.log_dir)

    # 使用get_env_and_dataset函数获取制定的gym环境（args.env_name）、对应的数据集dataset以及奖励reward_transformer
    # 此外，获取数据集大小dataset_size，观测维度obs_dim和动作维度act_dim
    env, dataset, reward_transformer = get_env_and_dataset(args.env_name, args.max_episode_steps)
    dataset_size = dataset['observations'].shape[0]
    obs_dim = dataset['observations'].shape[1]
    act_dim = dataset['actions'].shape[1]
    # 如果提供了随机种子 (args.seed)，则将该随机种子应用于环境，以保证实验的可重复性。
    if args.seed is not None:
        set_seed(args.seed, env=env)
    # 检查是否有可用的 CUDA 设备（GPU）。如果有，将默认设备设置为 GPU。
    if torch.cuda.is_available():
        set_default_device()

    # 获取环境的动作空间action_space,然后创建一个高斯策略GaussianPolicy。
    # 该策略由一个神经网络表示，接受观测维度obs_dim和动作维度act_dim作为输入，还有其他作为参数的隐藏层维度和数量hidden_dim和hidden_num
    # 该策略用于在给定观测下选择动作
    action_space = env.action_space
    policy = GaussianPolicy(obs_dim, act_dim, hidden_dim=args.hidden_dim, n_hidden=args.hidden_num,
                            action_space=action_space, scale_distribution=False, state_dependent_std=False)
    # 将 args.algorithm 转换为大写，以便不区分大小写地解析算法选项
    algorithm_option = args.algorithm.upper()
    # 如果选择的算法是 "SCRATCH"（从命令行参数中读取），则 double_buffer 设置为 False，然后创建一个 IQL_online 算法实例 (alg)。
    # 这个算法使用双重评论网络 (DoubleCriticNetwork) 和值函数网络 (ValueNetwork)，以及上面创建的策略 (policy)。
    # 使用 Adam 优化器训练网络，学习率为 args.learning_rate。这个算法将使用在线的方式进行训练，而不会使用任何来自离线数据集的信息
    if algorithm_option == "SCRATCH":
        double_buffer = False
        alg = IQL_online(
            critic=DoubleCriticNetwork(obs_dim, act_dim, hidden_dim=args.hidden_dim, n_hidden=args.hidden_num),
            vf=ValueNetwork(obs_dim, hidden_dim=args.hidden_dim, n_hidden=args.hidden_num),
            policy=policy,
            optimizer_ctor=lambda params: torch.optim.Adam(params, lr=args.learning_rate),
            tau=args.tau,
            beta=args.beta,
            target_update_rate=args.target_update_rate,
            discount=args.discount,
            ckpt_path=None
        )
    # 如果选择的算法是 "BUFFER"，则 double_buffer 设置为 True，并且创建一个与之前相同的 IQL_online 算法实例。这里使用离线数据集作为缓冲区，称为 "Buffer-IQL"。
    elif algorithm_option == "BUFFER":
        double_buffer = True
        alg = IQL_online(
            critic=DoubleCriticNetwork(obs_dim, act_dim, hidden_dim=args.hidden_dim, n_hidden=args.hidden_num),
            vf=ValueNetwork(obs_dim, hidden_dim=args.hidden_dim, n_hidden=args.hidden_num),
            policy=policy,
            optimizer_ctor=lambda params: torch.optim.Adam(params, lr=args.learning_rate),
            tau=args.tau,
            beta=args.beta,
            target_update_rate=args.target_update_rate,
            discount=args.discount,
            ckpt_path=None
        )
    # 如果选择的算法是 "DIRECT"，则 double_buffer 设置为 True，并且需要提供有效的检查点路径（args.ckpt_path）。算法实例还是 IQL_online，不过这里使用了一个预训练的离线模型。
    elif algorithm_option == "DIRECT":
        double_buffer = True
        assert args.ckpt_path, "need to provide a valid checkpoint path"
        alg = IQL_online(
            critic=DoubleCriticNetwork(obs_dim, act_dim, hidden_dim=args.hidden_dim, n_hidden=args.hidden_num),
            vf=ValueNetwork(obs_dim, hidden_dim=args.hidden_dim, n_hidden=args.hidden_num),
            policy=policy,
            optimizer_ctor=lambda params: torch.optim.Adam(params, lr=args.learning_rate),
            tau=args.tau,
            beta=args.beta,
            target_update_rate=args.target_update_rate,
            discount=args.discount,
            ckpt_path=args.ckpt_path
        )
    # 如果选择的算法是 "PEX"，则 double_buffer 设置为 True，并且需要提供有效的检查点路径（args.ckpt_path）。
    # 算法实例是 PEX 算法，该算法用于处理强化学习中的探索问题。它使用离线数据集进行训练，args.inv_temperature 是用于 PEX 动作选择的逆温度参数。
    elif algorithm_option == "PEX":
        double_buffer = True
        assert args.ckpt_path, "need to provide a valid checkpoint path"
        alg = PEX(
            critic=DoubleCriticNetwork(obs_dim, act_dim, hidden_dim=args.hidden_dim, n_hidden=args.hidden_num),
            vf=ValueNetwork(obs_dim, hidden_dim=args.hidden_dim, n_hidden=args.hidden_num),
            policy=policy,
            optimizer_ctor=lambda params: torch.optim.Adam(params, lr=args.learning_rate),
            tau=args.tau,
            beta=args.beta,
            target_update_rate=args.target_update_rate,
            discount=args.discount,
            ckpt_path=args.ckpt_path,
            inv_temperature=args.inv_temperature,
        )
    # 创建一个回放内存缓冲区 (memory)，其大小由 args.replay_size 指定，并且还可以使用随机种子 args.seed
    memory = ReplayMemory(args.replay_size, args.seed)
    # 初始化总的环境步数为 0。
    total_numsteps = 0
    # 进入主训练循环 for i_episode in itertools.count(1):。初始化每个回合的奖励 (episode_reward) 和步数 (episode_steps)，
    # 并设置 done 为 False。然后，通过 env.reset() 获取环境的初始状态 (state)。
    for i_episode in itertools.count(1):
        episode_reward = 0
        episode_steps = 0
        done = False
        state = env.reset()

        # 在每个回合内部，只要回合未结束，就进入一个内部循环。首先，使用算法 (alg) 在当前状态 (state) 下选择一个动作 (action)。
        # 然后，如果回放内存 (memory) 中存储的样本数量超过 args.initial_collection_steps，则进行网络更新。更新次数由 args.updates_per_step 决定。
        while not done:
            action = alg.select_action(torchify(state).to(DEFAULT_DEVICE)).detach().cpu().numpy()
            if len(memory) > args.initial_collection_steps:
                for i in range(args.updates_per_step):
                    alg.update(*get_batch_from_dataset_and_buffer(dataset, memory, args.batch_size, double_buffer))
            # 在内部循环中，根据选择的动作 (action) 与环境交互，获得下一个状态 (next_state)、奖励 (reward) 和完成标志 (done)。累计当前回合的步数、总环境步数和回合奖励。
            #
            # 然后，将原始奖励转换为回放缓冲区使用的奖励 (reward_for_replay)。terminal 是一个标志，用于指示当前回合是否结束（0 表示未结束，1 表示结束）。
            #
            # 最后，将状态、动作、转换后的奖励、下一个状态和完成标志存储在回放缓冲区中。
            next_state, reward, done, _ = env.step(action)
            episode_steps += 1
            total_numsteps += 1
            episode_reward += reward

            reward_for_replay = reward_transformer(reward)

            terminal = 0 if episode_steps == env._max_episode_steps else float(done)
            memory.push(state, action, reward_for_replay, next_state, terminal)
            state = next_state
            # 如果总环境步数达到了 args.eval_period 的倍数，并且 args.eval 参数为 True，则进行策略评估。调用 eval_policy 函数对当前策略进行评估
            if total_numsteps % args.eval_period == 0 and args.eval is True:
                print("Episode: {}, total env-steps: {}".format(i_episode, total_numsteps))
                eval_policy(env, args.env_name, alg, args.max_episode_steps, args.eval_episode_num)
        # 如果总环境步数超过了 args.total_env_steps，则跳出主循环，结束训练过程。
        if total_numsteps > args.total_env_steps:
            break

        # 在训练结束后，关闭环境
        env.close()
    # 保存训练完毕的算法模型的状态字典，命名为 {args.algorithm}_online_ckpt，并将其存储在指定的日志目录中 (args.log_dir)。
    torch.save(alg.state_dict(), args.log_dir + '/{}_online_ckpt'.format(args.algorithm))


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--algorithm', required=True)  # ['direct', 'buffer', 'pex']
    parser.add_argument('--env_name', required=True)
    parser.add_argument('--log_dir', required=True)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--discount', type=float, default=0.99)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--hidden_num', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--target_update_rate', type=float, default=0.005)
    parser.add_argument('--tau', type=float, default=0.7)
    parser.add_argument('--beta', type=float, default=10.0,
                        help='IQL inverse temperature')
    parser.add_argument('--ckpt_path', default=None,
                        help='path to the offline checkpoint')

    parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                        help='size of replay buffer (default: 10000000)')
    parser.add_argument('--total_env_steps', type=int, default=1000001, metavar='N',
                        help='total number of env steps (default: 1000000)')
    parser.add_argument('--initial_collection_steps', type=int, default=5000, metavar='N',
                        help='Initial environmental steps before training starts (default: 5000)')
    parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                        help='model updates per simulator step (default: 1)')
    parser.add_argument('--inv_temperature', type=float, default=10, metavar='G',
                        help='inverse temperature for PEX action selection (default: 10)')
    parser.add_argument('--eval', type=bool, default=True,
                        help='Evaluates a policy a policy every 10 episode (default: True)')
    parser.add_argument('--eval_period', type=int, default=10000)
    parser.add_argument('--eval_episode_num', type=int, default=10,
                        help='Number of evaluation episodes (default: 10)')
    parser.add_argument('--max_episode_steps', type=int, default=1000)

    main(parser.parse_args())

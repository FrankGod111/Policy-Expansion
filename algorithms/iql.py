import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

from pex.utils.util import DEFAULT_DEVICE, update_exponential_moving_average

EXP_ADV_MAX = 100.  # 全局变量，用于限制经验优势experience advantage 的最大值


# 计算期望值损失，输入参数diff，expectile为差值和期望值。

# 根据差值的正负，计算权重，然后返回加权平方差值得均值
def expectile_loss(diff, expectile):
    weight = torch.where(diff > 0, expectile, (1 - expectile))
    return (weight * (diff ** 2)).mean()


class IQL(nn.Module):
    def __init__(self, critic, vf, policy, optimizer_ctor, max_steps,
                 tau, beta, discount=0.99, target_update_rate=0.005, use_lr_scheduler=True):
        # critic网络，值函数网络、policy网络，优化器构造函数，最大步数，tau期望值。beta经验优势系数
        # discount 折扣因子，target_update_rate目标网络更新速率，use_lr_scheduler是否使用学习率调度器
        super().__init__()
        self.critic = critic.to(DEFAULT_DEVICE)
        self.target_critic = copy.deepcopy(critic).requires_grad_(False).to(DEFAULT_DEVICE)
        self.vf = vf.to(DEFAULT_DEVICE)
        self.policy = policy.to(DEFAULT_DEVICE)
        self.v_optimizer = optimizer_ctor(self.vf.parameters())
        self.q_optimizer = optimizer_ctor(self.critic.parameters())
        self.policy_optimizer = optimizer_ctor(self.policy.parameters())
        self.tau = tau
        self.beta = beta
        self.discount = discount
        self.target_update_rate = target_update_rate
        self.use_lr_scheduler = use_lr_scheduler
        if use_lr_scheduler:
            self.policy_lr_schedule = CosineAnnealingLR(self.policy_optimizer, max_steps)

    def update(self, observations, actions, next_observations, rewards, terminals):
        # update 方法：更新值函数、Q 函数和策略网络。
        # 输入参数为观察值（observations）、行动（actions）、下一个观察值（next_observations）、奖励（rewards）和终止标志（terminals）
        with torch.no_grad():
            target_q = self.target_critic.min(observations, actions)
            next_v = self.vf(next_observations)

        # Update value function
        v = self.vf(observations)
        adv = target_q.detach() - v
        v_loss = expectile_loss(adv, self.tau)
        self.v_optimizer.zero_grad(set_to_none=True)
        v_loss.backward()
        self.v_optimizer.step()

        # Update Q function
        targets = rewards + (1. - terminals.float()) * self.discount * next_v.detach()
        qs = self.critic(observations, actions)

        q_loss = sum(F.mse_loss(q, targets) for q in qs) / len(qs)
        self.q_optimizer.zero_grad(set_to_none=True)
        q_loss.backward()
        self.q_optimizer.step()

        # Update target Q network
        update_exponential_moving_average(self.target_critic, self.critic, self.target_update_rate)

        self.policy_update(observations, adv, actions)

    def policy_update(self, observations, adv, actions):
        # 更新策略网络。输入参数为观察值（observations）、经验优势（adv）和行动（actions）
        exp_adv = torch.exp(self.beta * adv.detach()).clamp(max=EXP_ADV_MAX)
        policy_out = self.policy(observations)
        bc_losses = -policy_out.log_prob(actions.detach())

        policy_loss = torch.mean(exp_adv * bc_losses)
        self.policy_optimizer.zero_grad(set_to_none=True)
        policy_loss.backward()
        self.policy_optimizer.step()
        if self.use_lr_scheduler:
            self.policy_lr_schedule.step()

    def select_action(self, state, evaluate=False):
        # 根据给定的状态（state）和评估标志（evaluate），选择一个行动。
        # 如果评估标志为 False，则返回采样的行动；否则，返回模式行动

        # 或者也可以用神经网络去学习这个采样概率，类似于DL class的那篇论文
        if evaluate is False:
            action_sample, _, _ = self.policy.sample(state)
            return action_sample
        else:
            _, _, action_mode = self.policy.sample(state)
            return action_mode

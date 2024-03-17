import copy
import torch

from pex.utils.util import (DEFAULT_DEVICE, epsilon_greedy_sample,
                            extract_sub_dict)
from pex.algorithms.iql import IQL, EXP_ADV_MAX


class PEX(IQL):
    def __init__(self, critic, vf, policy, optimizer_ctor,
                 tau, beta, discount, target_update_rate, ckpt_path, inv_temperature,
                 copy_to_target=False):  # copy_to_target是一个布尔值，表示是否将模型的参数复制到目标模型
        super().__init__(critic=critic, vf=vf, policy=policy,
                         optimizer_ctor=optimizer_ctor,  # 优化器
                         max_steps=None,
                         tau=tau, beta=beta,
                         discount=discount,
                         target_update_rate=target_update_rate,
                         use_lr_scheduler=False)

        self.policy_offline = copy.deepcopy(self.policy).to(
            DEFAULT_DEVICE)  # 创建一个离线策略，将当前策略self.policy copy给它，并将其移动到指定设备

        self._inv_temperature = inv_temperature  # 定义一个参数

        # load checkpoint if ckpt_path is not None
        if ckpt_path is not None:  #

            map_location = None
            if not torch.cuda.is_available():
                map_location = torch.device('cpu')
            checkpoint = torch.load(ckpt_path, map_location=map_location)

            # extract sub-dictionary
            policy_state_dict = extract_sub_dict("policy", checkpoint)
            critic_state_dict = extract_sub_dict("critic", checkpoint)

            self.policy_offline.load_state_dict(policy_state_dict)
            self.critic.load_state_dict(critic_state_dict)
            self.vf.load_state_dict(extract_sub_dict("vf", checkpoint))

            if copy_to_target:
                self.target_critic.load_state_dict(critic_state_dict)
            else:
                target_critic_state_dict = extract_sub_dict("target_critic", checkpoint)
                self.target_critic.load_state_dict(target_critic_state_dict)

    # 在下列代码中，evaluate 是一个布尔值参数，用于控制动作选择的方式。它可以影响动作选择过程中的探索与利用的权衡。
    #
    # 当 evaluate 为 True 时，意味着我们希望进行评估或测试阶段的动作选择。
    # 在评估阶段，我们更关注利用已经学到的知识，尽量选择最优的动作，而不是进行随机探索。
    # 因此，使用较小的 ε 值（例如 0.1）来进行动作选择，以保持一定的探索性。
    #
    # 当 evaluate 为 False 时，意味着我们处于训练或探索阶段。
    # 在训练阶段，我们需要进行一定程度的探索，以发现新的、未知的动作对于策略的潜在价值。
    # 因此，使用较大的 ε 值（例如 1.0）来进行动作选择，以增加随机性，促进探索过程。
    #
    # 通过控制 evaluate 参数，我们可以在训练和评估阶段之间灵活地调整动作选择策略，以满足特定的需求

    def select_action(self, observations, evaluate=False, return_all_actions=False):
        # select_action 方法接受一个参数 observations，表示观察到的状态。
        # observations 被调整为大小为 (1, ...) 的张量，即在第0维上添加了一个维度。
        # a1 是通过在离线策略上执行动作选择操作得到的结果，使用了确定性的方式。
        observations = observations.unsqueeze(0)
        a1 = self.policy_offline.act(observations, deterministic=True)

        # dist 是根据当前观察状态 observations，通过在线策略计算的一个分布。
        # 如果 evaluate 参数为 True，那么使用 ε-greedy 方法根据分布 dist 进行动作采样，使用较小的 ε 值（例如 0.1）。
        # 如果 evaluate 参数为 False，那么使用 ε-greedy 方法根据分布 dist 进行动作采样，使用较大的 ε 值（例如 1.0）。
        dist = self.policy(observations)
        if evaluate:
            a2 = epsilon_greedy_sample(dist, eps=0.1)
        else:
            a2 = epsilon_greedy_sample(dist, eps=1.0)

        # q1 是通过在评论家网络上使用状态 observations 和动作 a1 进行最小化操作得到的结果。
        # q2 是通过在评论家网络上使用状态 observations 和动作 a2 进行最小化操作得到的结果。
        q1 = self.critic.min(observations, a1)
        q2 = self.critic.min(observations, a2)

        # 将 q1 和 q2 沿着最后一个维度堆叠起来，得到 q。
        # logits 是 q 乘以一个逆温度系数的结果。
        # w_dist 是基于 logits 创建的一个 Categorical 分布对象。
        q = torch.stack([q1, q2], dim=-1)
        logits = q * self._inv_temperature
        w_dist = torch.distributions.Categorical(logits=logits)

        # 如果 evaluate 参数为 True，那么使用 ε-greedy 方法根据分布 w_dist 进行采样，使用较小的 ε 值（例如 0.1）。
        # 如果 evaluate 参数为 False，那么使用 ε-greedy 方法根据分布 w_dist 进行采样，使用较大的 ε 值（例如 1.0）。
        if evaluate:
            w = epsilon_greedy_sample(w_dist, eps=0.1)
        else:
            w = epsilon_greedy_sample(w_dist, eps=1.0)

        # 在 w 张量的最后一个维度上添加一个维度。
        # action 是通过计算 (1 - w) * a1 + w * a2 得到的结果。
        w = w.unsqueeze(-1)
        action = (1 - w) * a1 + w * a2

        # 如果 return_all_actions 参数为 False，那么返回经过压缩（去除维度为1的维度）的 action。
        # 如果 return_all_actions 参数为 True，那么返回经过压缩的 action、a1 和 a2。
        if not return_all_actions:
            return action.squeeze(0)
        else:
            return action.squeeze(0), a1.squeeze(0), a2.squeeze(0)

    # 策略更新：根据当前的观测值、优势值和动作来更新策略模型，使其能够更好地选择动作以最大化累积奖励
    def policy_update(self, observations, adv, actions):
        actions = self.select_action(observations)  # 调用了select_action函数来选择动作，根据输入的观测值（observations）来确定要执行的动作。
        with torch.no_grad():
            target_q = self.target_critic.min(observations, actions)  # 使用target_critic模型计算目标Q值。
            # 在这个上下文中，target_critic.min返回联合动作中的最小Q值，表示对当前观测值和动作的最佳估计值。
        v = self.vf(observations)  # 使用vf模型计算价值函数（value function）。这里的价值函数表示对给定观测值的状态价值估计
        adv = target_q.detach() - v  # 计算优势值（advantage），即目标Q值和状态价值之间的差异

        # 通过将优势值乘以一个参数self.beta并进行指数化，得到指数优势值（exponential advantage）。同时使用clamp函数将其限制在一个最大值（EXP_ADV_MAX）内
        exp_adv = torch.exp(self.beta * adv.detach()).clamp(max=EXP_ADV_MAX)

        # 使用policy模型计算策略输出，并计算策略输出与目标动作之间的负对数似然损失（negative log-likelihood loss）
        policy_out = self.policy(observations)
        bc_losses = -policy_out.log_prob(actions.detach())

        # 计算策略损失（policy loss），将指数优势值乘以负对数似然损失，并取平均值
        policy_loss = torch.mean(exp_adv * bc_losses)

        # 通过反向传播和优化器，更新策略模型的参数，以减小策略损失
        self.policy_optimizer.zero_grad(set_to_none=True)
        policy_loss.backward()
        self.policy_optimizer.step()
        # 如果使用学习率调度器（learning rate scheduler），则更新策略模型的学习率
        if self.use_lr_scheduler:
            self.policy_lr_schedule.step()

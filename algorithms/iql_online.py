import torch

from ..utils.util import DEFAULT_DEVICE, extract_sub_dict
from .iql import IQL


class IQL_online(IQL):
    def __init__(self, critic, vf, policy, optimizer_ctor,
                 tau, beta, discount, target_update_rate, ckpt_path, copy_to_target=True):
        #         构造函数的参数包括 critic 网络、值函数网络（vf）、策略网络（policy）、
        #         优化器构造函数（optimizer_ctor）、tau 期望值（tau）、经验优势系数（beta）、折扣因子（discount）、
        #         目标网络更新速率（target_update_rate）、
        #         检查点路径（ckpt_path）和是否将 critic 网络的权重复制到目标网络（copy_to_target）。

        # 调用父类参数
        super().__init__(critic=critic, vf=vf, policy=policy,
                         optimizer_ctor=optimizer_ctor,
                         max_steps=None,
                         tau=tau, beta=beta,
                         discount=discount,
                         target_update_rate=target_update_rate,
                         use_lr_scheduler=False)

        # load checkpoint if ckpt_path is not None
        if ckpt_path is not None:
            # 如果检查点路径（ckpt_path）不为空，则加载检查点。
            map_location = None
            if not torch.cuda.is_available():
                map_location = torch.device('cpu')
            checkpoint = torch.load(ckpt_path, map_location=map_location)

            # 如果 GPU 不可用，则将 map_location 设置为 CPU 设备。
            # 加载检查点文件。

            # extract sub-dictionary
            policy_state_dict = extract_sub_dict("policy", checkpoint)
            critic_state_dict = extract_sub_dict("critic", checkpoint)

            # 从检查点中提取策略网络和 critic 网络的状态dict

            self.policy.load_state_dict(policy_state_dict)
            self.critic.load_state_dict(critic_state_dict)

            # 加载策略网络和 critic 网络的状态字典。

            if copy_to_target:
                self.target_critic.load_state_dict(critic_state_dict)
            else:
                target_critic_state_dict = extract_sub_dict("target_critic", checkpoint)
                self.target_critic.load_state_dict(target_critic_state_dict)

            # 如果 copy_to_target 为 True，则将 critic 网络的权重复制到目标网络；
            # 否则，从检查点中提取目标 critic 网络的状态字典，并加载到目标 critic 网络。

            self.vf.load_state_dict(extract_sub_dict("vf", checkpoint))
            # 从检查点中提取值函数网络的状态字典，并加载到值函数网络。

#
# 1. IQL_online 类继承自 IQL 类，而 main.py 中定义了 IQL 类本身。
# 2. IQL_online 类的构造函数中，增加了从检查点文件加载模型权重的功能，而 IQL 类没有这个功能。
# 3. IQL_online 类在调用父类 IQL 的构造函数时，将 use_lr_scheduler 参数设置为 False，表示不使用学习率调度器。
# 而在 main.py 中的 IQL 类中，use_lr_scheduler 参数可以根据用户输入进行设置

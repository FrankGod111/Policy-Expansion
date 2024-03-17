import torch
import torch.nn as nn
from torch import optim
from torch.distributions import constraints
import torch.distributions as td
from pex.utils.util import mlp, get_mode
import numpy as np

LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0


# 高斯策略（Gaussian Policy）是一种在连续动作空间中使用的策略。
# 在这个策略中，我们使用高斯分布（正态分布）来表示动作的概率分布。
# 高斯策略的核心思想是，给定一个状态，我们可以从高斯分布中采样一个动作，这个动作在概率上最有可能使得我们获得最大的回报。
# - obs_dim：观察空间的维度。
# - act_dim：动作空间的维度。
# - hidden_dim：隐藏层的维度。
# - n_hidden：隐藏层的数量。
# - action_space：动作空间，用于限制动作的范围。
# - scale_distribution：是否对动作分布进行缩放。
# - state_dependent_std：是否使用状态依赖的标准差。
#
# 在训练过程中，高斯策略会根据当前状态生成一个动作。
# 这个动作是从高斯分布中采样得到的，这个分布的均值和标准差是由策略网络计算得到的。
# 通过训练策略网络，我们可以使得生成的动作更有可能获得更高的回报


class GaussianPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, n_hidden=2, action_space=None, scale_distribution=False,
                 state_dependent_std=False):
        # 初始化函数，接收输入维度、动作维度、隐藏层维度、隐藏层数量、动作空间、分布缩放scale-distribution、状态相关标准差标识
        super(GaussianPolicy, self).__init__()

        # 定义一个MLP网络，其中输入维度为num_inputs，隐藏层维度为hidden_dim，隐藏层数量为n_hidden，并使用relu激活函数
        self.net = mlp([num_inputs, *([hidden_dim] * n_hidden)],
                       output_activation=nn.ReLU)
        # 定义一个线性层，将隐藏层数输出映射到动作维度的空间
        self.mean_linear = nn.Linear(hidden_dim, num_actions)

        # 如果相关的标准差标志为真，则执行下列操作
        if state_dependent_std:
            self.log_std_linear = nn.Linear(hidden_dim, num_actions)
            # 定义另一个线性层，将隐藏层的输出映射到动作空间，用于生成动作的对数标准差
        else:
            # 定义一个可学习的参数--参数类型为nn.Parameter，用于生成动作的对数标准差
            self.log_std = nn.Parameter(torch.zeros(num_actions),
                                        requires_grad=True)
            self.log_std_linear = lambda _: self.log_std

        # 将权重初始化函数weights_init_应用到模型的所有模块上
        self.apply(weights_init_)
        # 赋值操作
        self._scale_distribution = scale_distribution
        self._state_dependent_std = state_dependent_std

        #
        if action_space is None:
            # 如果动作空间为空，则执行如下操作
            self._action_means = torch.tensor(0.)  # 将动作均值设置为0
            self._action_magnitudes = torch.tensor(1.)  # 将动作幅度设置为1
        else:
            self._action_means = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)  # 根据动作空间的上下限计算动作均值
            self._action_magnitudes = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)  # 根据动作空间的上下限计算动作幅度

        # 如果不进行分布scale_distribution
        if not scale_distribution:
            self._mean_transform = (
                lambda inputs: self._action_means + self._action_magnitudes
                               * inputs.tanh())  # 定义一个匿名函数，将输入通过双曲正切函数进行变换，得到均值
        else:
            self._mean_transform = lambda x: x  # 定义一个匿名函数，将输入作为输出
            self._transforms = [
                StableTanh(),
                td.AffineTransform(
                    loc=self._action_means, scale=self._action_magnitudes, cache_size=1)
            ]  # 定义一个变换列表，包含稳定双曲正切函数和仿射变换

    # 定义一个内部函数，用于生成正态分布对象
    def _normal_dist(self, means, stds):
        normal_dist = DiagMultivariateNormal(loc=means, scale=stds)
        if self._scale_distribution:
            squashed_dist = td.TransformedDistribution(
                base_distribution=normal_dist, transforms=self._transforms)
            return squashed_dist
        else:
            return normal_dist

    # 前向传播函数，接收输入x
    def forward(self, x):
        h = self.net(x)  # 通过多层感知机模型处理输入x，得到隐藏层的输出h
        mean = self._mean_transform(self.mean_linear(h))  # 将隐藏层输出通过均值变换得到均值
        log_std = self.log_std_linear(h)  # 将隐藏层输出通过对数标准差线性层得到对数标准差
        log_std = torch.clamp(log_std, min=LOG_STD_MIN, max=LOG_STD_MAX)  # 对对数标准差进行截断，确保在最小值和最大值之间
        return self._normal_dist(means=mean, stds=log_std.exp())  # 返回生成的正态分布对象

    def sample(self, obs):  # 从分布中采样一个动作，采样函数，接收观察值obs
        dist = self.forward(obs)  # 通过前向传播得到生成的正态分布对象
        action = dist.sample()  # 从分布中采样一个动作
        log_prob = dist.log_prob(action)  # 计算采样动作的对数概率
        mode = get_mode(dist)  # 获取分布的众数
        return action, log_prob, mode  # 返回采样的动作、对数概率和众数

    def act(self, obs, deterministic=False, enable_grad=False):
        # 动作函数，接受观察值obs以及确定性标志和梯度开启标志
        with torch.set_grad_enabled(enable_grad):
            dist = self.forward(obs)  # 根据梯度开启标志设置梯度计算的上下文环境
            return get_mode(dist) if deterministic else dist.sample()  # 如果确定性标志为真，则返回分布的众数，否则采样一个动作

    def to(self, device):
        # 将模型移动到指定设备上
        self._action_magnitudes = self._action_magnitudes.to(device)
        self._action_means = self._action_means.to(device)
        return super(GaussianPolicy, self).to(device)


# 定义了一个用于创建具有对角方差的多元正态分布的类，并提供了获取标准差的方法
class DiagMultivariateNormal(td.Independent):
    def __init__(self, loc, scale):
        """Create multivariate normal distribution with diagonal variance.

        Args:
            loc (Tensor): mean of the distribution
            scale (Tensor): standard deviation. Should have same shape as ``loc``.
        """
        # set validate_args to False here to enable the construction of Normal
        # distribution with zero scale.
        super().__init__(
            td.Normal(loc, scale, validate_args=False),
            reinterpreted_batch_ndims=1)

    @property
    def stddev(self):
        return self.base_dist.stddev


# 初始化神经网络模型中的权重和偏置
# 函数的输入参数为m，表示神经网络模型中的一个模块。在函数内部，首先通过isinstance(m, nn.Linear)判断m是否为线性层(nn.Linear)。如果是线性层，则执行以下操作：
#
# 使用torch.nn.init.xavier_uniform_函数对该线性层的权重进行初始化。xavier_uniform_是一种权重初始化方法，
# 旨在保持前向传播和反向传播时的信号传播具有相似的方差。gain=1表示权重的缩放因子为1。
# 使用torch.nn.init.constant_函数将该线性层的偏置初始化为常数0。
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


# 这段代码定义了一个名为StableTanh的类，它是一个可逆的转换（双射），用于计算 :math:Y = \tanh(X)，因此 :math:Y \in (-1, 1)。
#
# 代码中的StableTanh转换实际上是通过一系列变换的顺序来实现的，其中包括仿射变换（AffineTransform）和Sigmoid变换（SigmoidTransform）。但是，直接使用StableTanh转换更加数值稳定。
#
# 该类具有以下特征和方法：
#
# domain和codomain定义了转换的定义域和值域。
# bijective指示该转换是双射的。
# sign指示该转换的符号。
# 在构造函数中，cache_size参数用于控制缓存的大小，默认情况下使用缓存，因为反转过程在数值上是不稳定的。
#
# 该类还定义了以下方法：
#
# _call方法用于计算正向转换（从X到Y的转换）。
# _inverse方法用于计算逆向转换（从Y到X的转换）。
# log_abs_det_jacobian方法计算对数绝对行列式的对数值。
# with_cache方法返回一个具有指定缓存大小的新的StableTanh对象。
# 最后，__eq__方法用于比较两个StableTanh对象是否相等。
#
# 注意：以上代码已经经过了解释和修改。
class StableTanh(td.Transform):
    r"""Invertible transformation (bijector) that computes :math:`Y = tanh(X)`,
    therefore :math:`Y \in (-1, 1)`.

    This can be achieved by an affine transform of the Sigmoid transformation,
    i.e., it is equivalent to applying a list of transformations sequentially:

    .. code-block:: python

        transforms = [AffineTransform(loc=0, scale=2)
                      SigmoidTransform(),
                      AffineTransform(
                            loc=-1,
                            scale=2]

    However, using the ``StableTanh`` transformation directly is more numerically
    stable.
    """
    domain = constraints.real
    codomain = constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    def __init__(self, cache_size=1):
        # We use cache by default as it is numerically unstable for inversion
        super().__init__(cache_size=cache_size)

    def __eq__(self, other):
        return isinstance(other, StableTanh)

    def _call(self, x):
        return torch.tanh(x)

    def _inverse(self, y):
        # Based on https://github.com/tensorflow/agents/commit/dfb8c85a01d65832b05315928c010336df13f7b9#diff-a572e559b953f965c5c2cd1b9ded2c7b

        # 0.99999997 is the maximum value such that atanh(x) is valid for both
        # float32 and float64
        def _atanh(x):
            return 0.5 * torch.log((1 + x) / (1 - x))

        y = torch.where(
            torch.abs(y) <= 1.0, torch.clamp(y, -0.99999997, 0.99999997), y)
        return _atanh(y)

    def log_abs_det_jacobian(self, x, y):
        return 2.0 * (
                torch.log(torch.tensor(2.0, dtype=x.dtype, requires_grad=False)) -
                x - nn.functional.softplus(-2.0 * x))

    def with_cache(self, cache_size=1):
        if self._cache_size == cache_size:
            return self
        return StableTanh(cache_size)


# 定义神经网络模型
# 输入维度（Input Dimension）：输入层的维度由参数 state_dim 决定。这个神经网络被设计为接受一个大小为 state_dim 的状态向量作为输入。
#
# 隐藏层（Hidden Layer）：这个神经网络有一个隐藏层，使用线性全连接层 (nn.Linear) 来实现。
# 隐藏层的输出维度为 128，由 self.fc1 定义。隐藏层使用 ReLU 激活函数来引入非线性，将输入状态向量映射到一个更高维度的中间表示。
#
# 输出维度（Output Dimension）：输出层也是一个线性全连接层，由 self.fc2 定义。输出层的维度为 action_dim * 2，
# 因为它同时输出均值和标准差，用于参数化某种概率分布（通常是高斯分布），以产生连续动作空间中的动作。
#
# 因此，这个神经网络接受一个大小为 state_dim 的状态向量作为输入，经过一个具有128个神经元的隐藏层进行处理，最终输出大小为 action_dim * 2 的向量，
# 其中前一半表示均值，后一半表示标准差，以参数化一个概率分布，用于生成连续动作。

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, action_dim * 2)  # 输出均值和标准差

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = self.fc2(x)
        return x


# 探索算法
# action_dim = Q_values.shape[1] // 2：这行代码首先确定了动作空间的维度。
# Q_values 是一个包含 Q 值信息的张量，它被假设分为两个部分，前一半是均值部分，后一半是标准差部分。
# 所以，通过 Q_values.shape[1] 可以获取总的动作空间维度，然后使用 // 2 运算符获得动作的维度。
#
# mean_Q = Q_values[:, :action_dim] 和 std_Q = Q_values[:, action_dim:]：这两行代码将 Q_values 张量分为均值部分和标准差部分。
# mean_Q 包含了每个动作的均值估计，而 std_Q 包含了每个动作的标准差估计。
#
# exploration_bonus = c * torch.sqrt(torch.log(t + 1) / (std_Q + 1e-6))：这行代码计算了UCB探索奖励。
# UCB探索奖励的目的是在不确定性较大的动作上增加探索的可能性。
# 具体计算过程是首先计算每个动作的不确定性，即标准差 std_Q，然后将其与一个调整参数 c 相乘，同时除以 t + 1 的对数来控制不确定性对奖励的影响。这样，较高的不确定性将导致更高的探索奖励。
#
# UCB_scores = mean_Q + exploration_bonus：这行代码将均值估计和探索奖励相加，得到每个动作的UCB分数。UCB分数综合考虑了均值和不确定性，以帮助选择最优动作。
#
# selected_action = torch.argmax(UCB_scores, dim=1)：最后，根据UCB分数选择具有最高UCB分数的动作作为所选动作。
# torch.argmax 函数用于获取每行中最大值所在的列索引，即选择具有最高UCB分数的动作。



def UCB_exploration(Q_values, t, c=1.0):
    action_dim = Q_values.shape[1] // 2
    mean_Q = Q_values[:, :action_dim]  # 均值部分
    std_Q = Q_values[:, action_dim:]  # 标准差部分

    exploration_bonus = c * torch.sqrt(torch.log(t + 1) / (std_Q + 1e-6))  # 避免除零
    # exploration_bonus = λ * torch.max(mean_Q + std_Q, dim=1)[0]
    UCB_scores = mean_Q + exploration_bonus

    selected_action = torch.argmax(UCB_scores, dim=1)
    return selected_action


# 创建 Q-network 模型
state_dim = 3  # 状态空间维度
action_dim = 3  # 动作空间维度
q_network = QNetwork(state_dim, action_dim)

# 模拟环境状态
state = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)  # 示例状态

# 获取多个策略生成的动作
num_strategies = 5  # 策略数量
all_actions = []

for _ in range(num_strategies):
    Q_values = q_network(state)
    all_actions.append(Q_values)

# 将策略生成的动作堆叠成矩阵
all_actions = torch.cat(all_actions)

# 选择最优动作
timestep = 0  # 时间步
chosen_action = UCB_exploration(all_actions, timestep)

print("The selection action is:", chosen_action)



def interact_with_environment(selected_action):
    # 模拟与环境的交互，执行动作并获取奖励
    reward = np.random.randn()

    return reward


# 定义训练参数
learning_rate = 0.001
optimizer = optim.Adam(q_network.parameters(), lr=learning_rate)
num_episodes = 1000  # 假设有1000个训练周期

for episode in range(num_episodes):
    # 模拟环境状态，这里假设状态是随机的
    state = torch.tensor([np.random.rand(), np.random.rand(), np.random.rand()], dtype=torch.float32)

    # 获取多个策略生成的动作
    all_actions = []

    for _ in range(num_strategies):
        Q_values = q_network(state)
        all_actions.append(Q_values)

    # 将策略生成的动作堆叠成矩阵
    all_actions = torch.cat(all_actions)

    # 选择最优动作
    timestep = episode  # 假设时间步与训练周期相同
    chosen_action = UCB_exploration(all_actions, timestep)

    # 与环境交互，获取奖励
    reward = interact_with_environment(chosen_action)

    # 在这里可以根据奖励计算损失并进行反向传播和优化
    # 这一部分的代码根据具体的强化学习算法和奖励函数而异

    # 清空梯度，进行反向传播和优化
    optimizer.zero_grad()
    # 计算损失并反向传播


    # 输出训练信息
    print(f"Episode {episode}: Chosen Action: {chosen_action}, Reward: {reward}")

# 训练结束后，您可以使用训练好的模型来进行预测和决策

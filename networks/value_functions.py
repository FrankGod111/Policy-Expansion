import torch
import torch.nn as nn
from pex.utils.util import mlp

# 这段代码定义了一个名为DoubleCriticNetwork的类，它是nn.Module的子类。
#
# 在初始化方法（init）中，接受state_dim（状态维度）、action_dim（动作维度）、hidden_dim（隐藏层维度，默认为256）、n_hidden（隐藏层数量，默认为2）作为参数。
# 在初始化方法中，首先调用父类nn.Module的初始化方法（super().init()），
# 然后根据给定的参数构建一个dims列表，该列表包含了神经网络的各层维度信息。
# 然后，创建两个MLP（多层感知器）模型，分别为self.q1和self.q2，它们的输入维度为state_dim + action_dim，输出维度为1，
# 并且在创建MLP时设置了squeeze_output=True，表示将输出张量的维度进行压缩。
#
# 在前向传播方法（forward）中，接受state（状态）和action（动作）作为输入。首先，将state和action在最后一个维度上进行拼接，得到一个新的张量sa。
# 然后，分别将sa作为输入传递给self.q1和self.q2，得到两个输出张量，表示两个评估函数的输出。最后，将这两个输出张量作为元组返回。
#
# 另外还定义了一个名为min的方法，接受state和action作为输入。该方法调用forward方法获取self.q1和self.q2的输出，并使用torch.min函数对这两个输出张量进行比较，返回最小值。
#
# 总体而言，这段代码实现了一个具有两个评估函数的双重评论网络，并提供了前向传播和最小值计算的功能。
class DoubleCriticNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, n_hidden=2):
        super().__init__()
        dims = [state_dim + action_dim, *([hidden_dim] * n_hidden), 1]
        self.q1 = mlp(dims, squeeze_output=True)
        self.q2 = mlp(dims, squeeze_output=True)

    def forward(self, state, action):
        sa = torch.cat([state, action], -1)
        return self.q1(sa), self.q2(sa)

    def min(self, state, action):
        return torch.min(*self.forward(state, action))

# 这段代码定义了一个名为ValueNetwork的类，该类继承自nn.Module。该类用于构建值函数网络，用于估计给定状态的值。
#
# 在初始化方法__init__中，我们传入状态维度state_dim以及隐藏层维度hidden_dim和隐藏层数量n_hidden作为参数。
# 然后，我们定义了一个维度列表dims，其中包含了从输入状态到输出值的层的维度。这些层的维度由输入状态维度、隐藏层维度和隐藏层数量决定，并以1作为输出值的维度。
#
# 接下来，我们通过调用mlp函数来创建值函数网络，其中mlp函数是一个多层感知机（MLP）模型的构建函数。
# 我们传入了维度列表dims作为参数，并将squeeze_output设置为True以压缩输出值的维度。
#
# 在前向传播方法forward中，我们传入状态state作为输入，并调用值函数网络self.v来计算给定状态的值。然后将计算得到的值作为输出返回。
#
# 总而言之，该代码定义了一个值函数网络，并提供了前向传播方法用于计算给定状态的值。
class ValueNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim=256, n_hidden=2):
        super().__init__()
        dims = [state_dim, *([hidden_dim] * n_hidden), 1]
        self.v = mlp(dims, squeeze_output=True)

    def forward(self, state):
        return self.v(state)
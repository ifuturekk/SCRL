import torch
import torch.nn as nn
import torch.nn.functional as F


class GeneralizedCrossEntropyLoss(nn.Module):
    def __init__(self, q=0.7):
        """
        初始化 Generalized Cross Entropy 损失函数
        :param q: 调节参数 q，通常介于 0 和 1 之间
        """
        super(GeneralizedCrossEntropyLoss, self).__init__()
        self.q = q

    def forward(self, inputs, targets):
        """
        前向传播计算损失
        :param inputs: 模型的输出，形状为 (batch_size, num_classes)
        :param targets: 实际标签，形状为 (batch_size,)
        :return: 计算得到的损失值
        """
        # 将模型输出通过 softmax 转换为概率分布
        probabilities = F.softmax(inputs, dim=1)

        # 获取正确类别的预测概率
        targets_one_hot = F.one_hot(targets, num_classes=probabilities.size(1)).float()
        probabilities = torch.sum(probabilities * targets_one_hot, dim=1)

        # 计算 Generalized Cross Entropy 损失
        loss = (1.0 - probabilities ** self.q) / self.q

        # 返回平均损失
        return torch.mean(loss)


def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def factorization_loss(f_a, f_b):
    # empirical cross-correlation matrix
    f_a_norm = (f_a - f_a.mean(0)) / (f_a.std(0) + 1e-6)
    f_b_norm = (f_b - f_b.mean(0)) / (f_b.std(0) + 1e-6)
    c = torch.mm(f_a_norm.T, f_b_norm) / f_a_norm.size(0)

    on_diag = torch.diagonal(c).add_(-1).pow_(2).mean()
    off_diag = off_diagonal(c).pow_(2).mean()
    loss = on_diag + 0.005 * off_diag

    return loss


def squared_frobenius_norm_loss(f_a, f_b):
    f_a_norm = (f_a - f_a.mean(0)) / (f_a.std(0) + 1e-6)
    f_b_norm = (f_b - f_b.mean(0)) / (f_b.std(0) + 1e-6)
    c = torch.mm(f_a_norm.T, f_b_norm) / f_a_norm.size(0)
    loss = c.pow_(2).mean()
    return loss

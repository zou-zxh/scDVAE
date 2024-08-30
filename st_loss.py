"""
Original implementation of Contrastive-sc method
(https://github.com/ciortanmadalina/contrastive-sc)
By Madalina Ciortan (01/10/2020)
"""
import math
import pdb

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.modules.loss import _Loss

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    if torch.cuda.device_count() > 1:
        device = torch.device('cuda:0')
    else:
        device = torch.device('cuda')
else:
    device = torch.device('cpu')

EPS = 1e-8


def entropy(x, input_as_probabilities):
    """ 
    Helper function to compute the entropy over the batch 

    input: batch w/ shape [b, num_classes]
    output: entropy value [is ideally -log(num_classes)]
    """

    if input_as_probabilities:
        x_ = torch.clamp(x, min=EPS)
        b = x_ * torch.log(x_)
    else:
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)

    if len(b.size()) == 2:  # Sample-wise entropy
        return -b.sum(dim=1).mean()
    elif len(b.size()) == 1:  # Distribution-wise entropy
        return - b.sum()
    else:
        raise ValueError('Input tensor is %d-Dimensional' % (len(b.size())))


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR
    监督对比学习，它还支持SimCLR中的无监督对比损失"""

    def __init__(self, temperature=0.07, base_temperature=0.07, contrast_mode='all'):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
            如果说‘labels'和'mask'都为None,它退化为SimCLR无监督损失
            参数：特征、标签、mask（对比掩码）
            锚点anchors在对比学习中是指一组样本中的一部分，通常是其中的一部分样本，锚点的数量表示在给定一组样本中有多少个样本被选为锚点
            锚点样本将用于计算与其他样本的相似性
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        # 检查输入数据的维度
        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)
            # 获取输入特征'feature'中的对比数量，通常是样本的数量

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            # 如果既没有提供'labels'也没有提供'mask'，则创建一个单位矩阵（E矩阵），用于生成默认掩码
            # 这个掩码是一个对家矩阵，其中对角线的元素为1，其余元素为0，这意味着每个样本与自身的对比损失为正，而与其他样本的对比损失为0
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            # 如果提供了'labels'则首先将'labels'进行了视图变换，然后检查labels的数量是否与批次大小batch_size相匹配，若不匹配则报错
            # 计算一个掩码，其中掩码中的元素表示样本之间是否具有相同的标签，如果两个样本具有相同的标签则相应的掩码元素为1，否则为0
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device) # 如果提供了mask，则直接将输入的mask转换为浮点类型并移动到合适的设备

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        # torch.unbind(features,dim=1)-->将其拆分为两个([256,32],[256,32])，torch.cat按行拼接转变为[(512,32)]
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            print(anchor_feature.size)
            print(features.size)
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits，得到未经标准化的对数概率（logits)
        # torch.matmul是表示两个张量的矩阵乘积
        # torch.div执行张量元素的除法操作，torch.div(a,b)即a/b
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability,对未经标准化的logits进行数值稳定性处理，找到每个样本中的最大logits值
        # 将未经标准化的logits减去最大logits值，同时设置不需要梯度计算
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        # anchor_count表示锚点的数量，contrast_count表示对比的数量
        # mask.repeat表示在第一个维度上重复anchor_count次，在第二个维度上重复contrast_count次（256,256）-->(512,512)
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        # 创建新的掩码logits_mask，用于排除自对比（即锚点与自身对比的情况）
        # 将掩码logits_mask中的某些位置设置为0
        """
        torch.scatter(input, dim, index, src)
        input：目标张量，你要在其上执行操作的张量。
        dim：指定操作的维度，也就是在哪个维度上进行索引操作。
        index：一个张量，包含了用于索引的位置信息。
        src：一个张量，包含了要在 input 上执行的操作的源数据。
        
        torch.ones_like(mask)创建一个与'mask'具有相同形状的全为1的张量
        torch.arange(batch_size * anchor_count).view(-1, 1).to(device)：生成索引位置信息
        生成一个包含从 0 到 (batch_size * anchor_count - 1) 的一维张量
        并将其形状变换为二维张量，其中每行包含一个数字
        这些数字表示了所有可能的锚点和对比样本的组合。
        """
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1) # 计算了平均的正对比
        #         mean_log_prob_pos[torch.where(torch.isinf(mean_log_prob_pos))] = 0
        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos # 调整了一下损失函数
        loss = torch.where(torch.isnan(loss), torch.full_like(loss, 0), loss)  # 用于处理可能出现NaN值，将其替换为0
        loss = loss.view(anchor_count, batch_size).mean() # 将损失的形状调整为标量，并计算均值

        return loss


class WeightedSupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""

    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(WeightedSupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None, weights=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)
        w = torch.cat([weights, weights])
        w = w.repeat(w.shape[0], 1)
        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        mask = w * mask
        # compute mean of log-likelihood over positive
        denominator = mask.sum(1)
        mean_log_prob_pos = (mask * log_prob).sum(1) / denominator
        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

#!/usr/bin/env python
"""
#
#

# File Name: model.py
# Description:

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init,Parameter
from torch import optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR, ExponentialLR, ReduceLROnPlateau
from scDVAE.utils import mask_generator,pretext_generator
import time
import datetime
import math
import numpy as np
from torch.utils.data import TensorDataset,DataLoader
from tqdm import trange
from itertools import repeat
from sklearn.mixture import GaussianMixture

from .layer import Encoder, Decoder, build_mlp, DeterministicWarmup,buildNetwork1,buildNetwork
from .loss import elbo, elbo_DREAM
import st_loss
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def adjust_learning_rate(init_lr, optimizer, iteration):
    lr = max(init_lr * (0.9 ** (iteration//10)), 0.0002)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr	


import os

class Generator(nn.Module):

    """
    这个函数只用于生成潜在特征，将原本有20214个的特征压缩到10个基因特征，然后便于后续的分析
    """
    def __init__(self,dims,bn=True,dropout=0.15,binary=True,output_activation=nn.Sigmoid()):
        super(Generator, self).__init__()

        [x_dim,z_dim,encode_dim,decode_dim] = dims
        self.binary=binary
        if binary:
            decode_activation = nn.Sigmoid()
        else:
            decode_dim = None

        # dropout (bn)的设定
        self.model = build_mlp([z_dim, *decode_dim], bn=bn, dropout=dropout)
        #         self.hidden = build_mlp([z_dim]+h_dim, bn=bn, dropout=dropout)
        self.reconstruction = nn.Linear([z_dim, *decode_dim][-1], x_dim)
        #         self.reconstruction = nn.Linear(([z_dim]+h_dim)[-1], x_dim)

        self.output_activation = output_activation
        self.zilayer = ZILayer(annealing=True)

        # self.model = Decoder([z_dim,decode_dim,x_dim],bn=bn,dropout=dropout,output_activation=decode_activation)

        self.reset_parameters()

    def reset_parameters(self):
            """初始化网络模型的参数"""
            for m in self.modules():
                if isinstance(m,nn.Linear):
                    init.xavier_normal_(m.weight.data)
                    if m.bias is not None:
                        m.bias.data.zero_()


    def forward(self,z1,z2):
        z = torch.cat((z1,z2),dim=1)
        z = self.model(z)
        z = self.reconstruction(z)
        z = self.output_activation(z)
        gen_cell = self.zilayer(z)

        return gen_cell
    # 没有聚类特征的时候，即全部都是生成特征
    # def forward(self,z1):
    #
    #     z = self.model(z1)
    #     z = self.reconstruction(z)
    #     z = self.output_activation(z)
    #     gen_cell = self.zilayer(z)
    #
    #     return gen_cell


class GMM(nn.Module):

    def __init__(self,n_cluster=10,n_features=64):
        super(GMM, self).__init__()

        self.n_cluster = n_cluster
        self.n_features = n_features
        
        # fill_()表示全部填充为某一个数，实现初始均匀分布，全都是统一初始化0.1*n_cluster
        # mu_c、log_sigma2_c生成n_cluster*n_features的张量，初始化为0
        self.pi_ = nn.Parameter(torch.FloatTensor(self.n_cluster,).fill_(1)/self.n_cluster,requires_grad=True)
        self.mu_c = nn.Parameter(torch.FloatTensor(self.n_cluster,self.n_features).fill_(0),requires_grad=True)
        self.log_sigma2_c = nn.Parameter(torch.FloatTensor(self.n_cluster,
                                                           self.n_features).fill_(0),requires_grad=True)

    def predict(self,z):
        pi = self.pi_
        log_sigma2_c = self.log_sigma2_c
        mu_c = self.mu_c

        yita_c = torch.exp(torch.log(pi.unsqueeze(0))+self.gaussian_pdfs_log(z,mu_c,log_sigma2_c))
        yita = yita_c.detach().cpu().numpy()
        return np.argmax(yita,axis=1)


    def get_asign(self,z):

        det = 1e-10
        pi = self.pi_
        mu_c = self.mu_c
        log_sigma2_c = self.log_sigma2_c

        yita = torch.exp(torch.log(pi.unsqueeze(0))+self.gaussian_pdfs_log(z,mu_c,log_sigma2_c))+det

        yita_c = yita / (yita.sum(1).view(-1,1))  # 将概率进行归一化，确保权重之和为1

        pred = torch.argmax(yita_c,dim=1,keepdim=True) # 得到每一行细胞最大概率所属的类别索引
        oh = torch.zeros_like(yita_c)
        oh = oh.scatter_(1, pred, 1.) # 将类别预测结果转换为独热编码的形式，如[0，0，0，0，1，0，0，0，0，0]
        return yita_c,oh

    def gaussian_pdfs_log(self, x,mus,log_sigma2s):
        # 对于一个训练批次的所有样本分别计算属于十个类别的高斯概率密度函数值（便于以后聚类任务？）
        G = []
        for c in range(self.n_cluster):
            G.append(self.gaussian_pdf_log(x,mus[c:c+1,:],log_sigma2s[c:c+1,:]).view(-1,1))
            # mus[c:c+1,:]--->[1,7]每次单独取出一行
        return torch.cat(G,1)

    @staticmethod
    def gaussian_pdf_log(x, mu, log_sigma2):

        return -0.5*(torch.sum(torch.tensor(np.log(np.pi*2),dtype=torch.float).to(DEVICE)+
                               log_sigma2+(x-mu).pow(2)/torch.exp(log_sigma2),1))



class ZILayer(nn.Module):
    def __init__(self, init_t=1, min_t=0.5, anneal_rate=0.00001, annealing=False):
            super(ZILayer, self).__init__()
            self.init_t = init_t
            self.min_t = min_t
            self.anneal_rate = anneal_rate
            self.annealing = annealing
            self.iteration = nn.Parameter(torch.tensor(0, dtype=torch.int), requires_grad=False)
            self.temperature = nn.Parameter(torch.tensor(init_t, dtype=torch.float), requires_grad=False)

    def forward(self, probs):
        p = torch.exp(-(probs**2))
        q = 1-p
        logits = torch.log(torch.stack([p, q], dim=-1)+1e-20)
        g = self.sampling_gumbel(logits.shape).type_as(logits)
        samples = torch.softmax((logits+g)/self.temperature, dim=-1)[..., 1]
        output = probs * samples
        if self.training and self.annealing:
            self.adjust_temperature()
        # print(output.mean().item(), output.std().item(), probs.mean().item(), probs.std().item())
        return output

    def sampling_gumbel(self,shape,eps=1e-8):
        u = torch.rand(*shape)
        return -torch.log(-torch.log(u+eps)+eps)

    def adjust_temperature(self):
        self.iteration.data += 1
        if self.iteration % 100 == 0:
            t = torch.clamp(self.init_t*torch.exp(-self.anneal_rate * self.iteration),min=self.min_t)
            self.temperature.data = t




class Encoder(nn.Module):
    # 这个encoder是类似自动编码器中的decoder部分的
    def __init__(self,dims,bn=False,dropout=0,c_feature=7,r =9):
        super(Encoder, self).__init__()

        [x_dim,z_dim,encode_dim,decode_dim] = dims

        self.first_layer=build_mlp([x_dim,encode_dim[0]],bn=bn,dropout=dropout)
        self.model =build_mlp(encode_dim,bn=bn,dropout=dropout)

        self.mu = nn.Linear(encode_dim[2],z_dim)
        self.con = nn.Linear(encode_dim[2],z_dim)
        self.v = nn.Linear(encode_dim[2],1)
        self.c_feature = c_feature
        self.r = r

        self.reset_parameters()

    def reset_parameters(self):
            """初始化网络模型的参数"""
            for m in self.modules():
                if isinstance(m,nn.Linear):
                    init.xavier_normal_(m.weight.data)
                    if m.bias is not None:
                        m.bias.data.zero_()


    def _s_re(self,mu,log_sigma,v):

        """聚类的分布是类高斯分布？"""
        sigma = torch.exp(log_sigma*0.5) # 将一个经过对数操作（使用对数通常是为了确保输出值为正数或者方便优化的）的标准差log_sigma转化为真的sigma
        std_z = torch.from_numpy(np.random.normal(0,1,size=sigma.size())).float().to(DEVICE) # 生成与sigma相同形状的随机数，这些随机数符合标准正态分布
        e = torch.tensor(np.random.normal(0,1,size=sigma.size()),dtype=torch.float).to(DEVICE) # 和前一句其实大差不差的
        # 为什么要设置两个，是因为公式中是有两个符合表示的正态分布的，需要加以区分

        z1 = (v/2 - 1/3) * torch.pow(1+(e/torch.sqrt(9 * v / 2 - 3)),3) # 从Gamma分布中抽取
        z2 = torch.sqrt(v / (2 * z1))
        z = mu + sigma * z2 * std_z # 完整的Gamma分布表示，此时z变成了从gamma分布中抽取的数据（90，7）表示的是z_c聚类信息

        # 返回的log_sigma-torch.log(z1)是将经过对数化的标准差调整为一个新的参数
        # 考虑到在Gamma分布中抽取的随机性，引入一些随机性或不确定性，从而更好地捕捉数据的分布特性
        return z , mu ,log_sigma-torch.log(z1)


    def moex(self,x1, x2, dim=0):
        # 均值-标准差交换，增强数据的鲁棒性，便于模型的训练
        mean1 = x1.mean(dim=dim,keepdim=True)
        mean2 = x2.mean(dim=dim,keepdim=True)

        var1 = x1.var(dim=dim,keepdim=True)
        var2 = x2.var(dim=dim,keepdim=True)

        std1 = (var1 + 1e-5).sqrt()
        std2 = (var2 + 1e-5).sqrt()

        x1 = std2 * ((x1-mean1)/std1)+mean2
        x2 = std1* ((x2-mean2)/std2)+mean1

        return x1,x2



    def _forward(self,x):

        x = self.model(x) #（90,512)
        # 然后得到GMM的准确性，而后进行比较，所以我们需要准备两个数据，一个是分批次的训练数据，一个是不分批次的

        # mu（均值），log_sigma（标准差）高斯分布的两个参数
        mu = self.mu(x)  # （90，10）从20214个特征中选取10个特征值mu
        log_sigma = self.con(x) # （90，10）同理

        mu1 = mu[:, :self.c_feature]  # (90,7)这里是只取了前七位
        mu2 = mu[:, self.c_feature:]  #（90，3）这里是只取了后三位
        log_sigma1 = log_sigma[:, :self.c_feature]
        log_sigma2 = log_sigma[:, self.c_feature:]

        v = F.softplus(self.v(x))+self.r  # (32,1)，softplus(x) = log(1+exp(x))激活函数，经过v(x)代表一种线性变化，可能是表示自由度？

        # z1 = mu1 + (torch.exp(log_sigma1 * 0.5) * torch.randn_like(mu1))  # 验证不改变分布都是使用GMM先验的实验结果是怎么样的

        z1,mu1,log_sigma1 = self._s_re(mu1,log_sigma1,v) # z1此时变成了从gamma分布中抽取的数据，log_sigma1也加入随机性扰动
        """要切记在模型中处理的是对数标准差而不是实际的标准差
        因此在公式中需要计算标准差的时候，需要将其转化为真正的标准差
        torch.exp(log_sigma2 * 0.5)的作用就是如此
        """
        z2 = mu2 + (torch.exp(log_sigma2 * 0.5) * torch.randn_like(mu2)) # z2是对后面抽取的特征做的处理，是自编码器的生成能力表示

        return z1,z2,mu1,mu2,log_sigma1,log_sigma2


    def forward(self,x,a_x=None,argument=False):
        # 当调用encoder类的时候会自动执行forward函数，因为nn.Module函数的关系
        """当我们需要选定最适合GMM参数的时候我们不需要对数据进行增强，只需要对数据进行采用得到一个类高斯分布，然后对其进行检验
           因此默认a_x=None会再进过first_layer以后，直接通过_forward函数进行采样"""
        x = self.first_layer(x)
        argument_z = None
        if a_x is not None:
            a_x = self.first_layer(a_x)
            # 但是这个增强不是一开始就增强的，是当训练类50轮之后才会增强
            if argument:
                x,_ = self.moex(x,a_x)
            argument_z = self._forward(a_x)

        original_z = self._forward(x)
        return original_z,argument_z

class pretrain_model(nn.Module):
    def __init__(self,input_dim,z_dim,n_cluster,encoderLayer=[],decodeLayer=[],
                 activation="relu",sigma=1.,alpha=1.,gamma=1.,ml_weight =1.,cl_weight=1.):
        super(pretrain_model,self).__init__()
        self.z_dim = z_dim
        self.n_cluter = n_cluster
        self.activation = activation
        self.sigma = sigma
        self.alpha = alpha
        self.gamma = gamma
        self.ml_weight = ml_weight
        self.cl_weight = cl_weight
        self.encoder = buildNetwork1([input_dim]+encoderLayer,type="encode",activation=activation)
        self.decoder = buildNetwork([z_dim]+decodeLayer,type="decode",activation=activation)
        self.decoder_mask = buildNetwork([z_dim]+decodeLayer+[input_dim],type="decode",activation=activation) # [32,64,256,x_dim]
        self._enc_mu=nn.Linear(encoderLayer[-1],z_dim)
        self.mu = Parameter(torch.Tensor(n_cluster,z_dim)) # 可训练的模型参数mu，这个参数在训练过程中可以通过优化算法进行更新


    def soft_assign(self,z):
        """
        计算输入数据点到聚类中心的软分配,最终返回样本属于每个类别的软分配权重
        """
        q1 = 1.0/(1.0+torch.sum((z.unsqueeze(1)-self.mu)**2,dim=2)/self.alpha)
        q2 = q1 **((self.alpha+1.0)/2.0)
        q3 = torch.where(torch.isnan(q2),torch.full_like(q2,0),q2)
        q = (q3.t()/torch.sum(q3,dim=1)).t()
        return q



    def forward(self,x):
        """
        torch.rand_like(x)创建一个与输入‘x’相同形状的张量，是一个标准的正态分布
        gamma可以控制噪声的强度，标准差越大，噪声的影响就越明显
        引入噪声可以帮助模型更好地学习数据的分布和特征
        """
        h = self.encoder(x+torch.rand_like(x)*self.gamma)
        z = self._enc_mu(h)
        h = self.decoder(z)
        _m = self.decoder_mask(z)

        h0 = self.encoder(x)
        z0 = self._enc_mu(h0)
        q = self.soft_assign(z0)

        return z0,q,_m

    def pretrain_autoencoder(self,x,batch_size = 256,lr=0.001,epochs=150,ae_save=True,
                             ae_weights='AE_weights.pth.tar',datasets = None,alpha=2,beta=2,pm=0.7,mu=1,xishu=0.85,mixup=0.99):
        criterion_rep = st_loss.SupConLoss(temperature=0.07)
        use_cuda = torch.cuda.is_available()
        print("##########",use_cuda,"lr:",lr)

        if use_cuda:
            self.cuda()

        # def custom_collate(batch):
        #     x_batch = torch.stack([item for item in batch],dim =0)
        #     return x_batch

        trainloader = DataLoader(x, batch_size=batch_size, shuffle=True)


        # dataset = TensorDataset(torch.Tensor(x))
        # dataloder = DataLoader(dataset,batch_size=batch_size,shuffle=True,collate_fn=custom_collate)
        print("----------Pretraining Stage-----------")

        optimizer = optim.Adam(filter(lambda p:p.requires_grad,self.parameters()),lr=lr,amsgrad=True)
        now = datetime.datetime.now()
        # ae_weights = './results/pretrain/%s_model_epoch%d_%d_%d_%d%.3f.path.tar' %(epoch,now.day,now.hour,now.minute,now.second,pm)
        best_mean_loss = float('inf')
        best_mixup_z = None



        for epoch in range(epochs):
            losses = []
            generated_data_list = []
            # ae_weights = './results/pretrain/%s_model_epoch%d_%d_%d_%d%.3f.path.tar' % (epoch, now.day, now.hour, now.minute, now.second, pm)
            for batch_idx,x_batch in enumerate(trainloader):
                # 生成随机掩码m,m2
                m = mask_generator(p_m=0.7,x=x_batch)
                m2 = mask_generator(0.7,x_batch)
                # mask,mask2代表原始数据x_batch和生成的数据在哪些有差异，不相等则标记为1
                # mask_x,mask_x2代表生成的混合数据，其中保留了30%的原始数据
                mask,mask_x = pretext_generator(m,x_batch)
                mask2,mask_x2 = pretext_generator(m2,x_batch)

                x_tensor=Variable(mask_x).float().cuda() # 便于使用GPU来训练
                x_tensor2 = Variable(mask_x2).float().cuda()
                # mix_up=0.99
                x_tensor3 = mixup *x_tensor +(1-mixup)*x_tensor2
                mask_tensor = Variable(mask).cuda()
                # xishu=0.85
                x_tensor2 = xishu * x_tensor2 + (1-xishu) * x_tensor
                """
                z可以视作编码器提取的潜在特征
                _是代表一个批次中处理的细胞属于每个类别的软分配
                _m是代表在原始数据中加入了一些随机噪声，而后解码得到的重构数据
                """
                z, _, _m = self.forward(x_tensor)
                z2, _, _ = self.forward(x_tensor2)
                z3, _, _ = self.forward(x_tensor3)
                z_ = torch.nn.functional.normalize(z)
                z2_ = torch.nn.functional.normalize(z2)
                z3_ = torch.nn.functional.normalize(z3)
                mixup_z_ = mixup * z_ +(1-mixup) * z2_
                features = torch.cat([z_.unsqueeze(1),z2_.unsqueeze(1)],dim=1)

                cos_loss = criterion_rep.forward(features)

                # loss2 = nn.functional.binary_cross_entropy(_m.float(),mask_tensor.float()) # 或许可以将其换成和原来损失

                criterion = nn.BCEWithLogitsLoss()
                loss2 = criterion(_m.float(),mask_tensor.float())
                loss3 = criterion(mixup_z_.float(),z3_.float())



                if (epoch >= 10):
                    loss = beta *loss2 +cos_loss
                else:
                    loss = beta *loss2 +cos_loss+alpha * loss3

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                losses.append(loss.item())
                generated_data_list.append(_m.cpu().detach().numpy())

            mean_loss = sum(losses)/len(losses)

            if mean_loss < best_mean_loss:
                best_mean_loss = mean_loss
                best_generated_data_list = []
                best_generated_data_list = generated_data_list.copy()
                # torch.save({'ae_state_dict':self.state_dict(),'optimizer_state_dict':optimizer.state_dict()},ae_weights)
            # print('Pretrain epoch [{}],loss:{:.4f}'.format(epoch + 1,loss.item()))

        if ae_save:
            print(ae_weights)
            # torch.save({'ae_state_dict':self.state_dict(),
            # 'optimizer_state_dict':optimizer.state_dict()},ae_weights)
        return best_generated_data_list












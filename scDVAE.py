#!/usr/bin/env python
"""
#
#

# File Name: scDVAE.py
# Description: Dimensionality reduction and visualization of single-cell RNA-seq data with an improved deep variational autoencoder.
    Input: 
        single-cell RNA-seq data
    Output:
        1. latent feature
        2. cluster assignment

"""


import time
from itertools import chain
from time import time as get_time
import psutil


import torch

import numpy as np
import pandas as pd
import os
import argparse
import torch.nn as nn
from sklearn.mixture import GaussianMixture
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import normalized_mutual_info_score as NMI
from sklearn.metrics import adjusted_rand_score,homogeneity_score,completeness_score,silhouette_score
import torch.nn.functional as F
from scDVAE.model import adjust_learning_rate

from scDVAE import Generator,GMM,Encoder,pretrain_model
from scDVAE.dataset import SingleCellDataset
from scDVAE.utils import  estimate_k,cluster_acc,gmm_Loss
from sklearn.preprocessing import MaxAbsScaler
from torch.utils.data import DataLoader, TensorDataset
import scanpy as sc
import warnings

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Dimensionality reduction and visualization of single-cell RNA-seq data with an improved deep variational autoencoder')
    parser.add_argument('--dataset', '-d', type=str, help='input dataset path', default="data/yan.txt") # sim2_outprob_0.2
    parser.add_argument('--n_centroids', '-k', type=int, help='cluster number')
    parser.add_argument('--outdir', '-o', type=str, default='output/', help='Output path')
    parser.add_argument('--verbose', action='store_true', help='Print loss of training process')
    parser.add_argument('--pretrain', type=str, default=None, help='Load the trained model')
    parser.add_argument('--lr', type=float, default=0.00001, help='Learning rate')
    parser.add_argument('--batch_size', '-b', type=int, default=32, help='Batch size')
    parser.add_argument('--gpu', '-g', default=0, type=int, help='Select gpu device number when training')
    parser.add_argument('--seed', type=int, default=8, help='Random seed for repeat results')  #yan
    parser.add_argument('--encode_dim', type=int, nargs='*', default=[512,128,64], help='encoder structure')
    parser.add_argument('--decode_dim', type=int, nargs='*', default=[64,128,512], help='encoder structure')
    parser.add_argument('--latent', '-l',type=int, default=2, help='latent layer dim')
    parser.add_argument('--log_transform', action='store_true', help='Perform log2(x+1) transform')
    parser.add_argument('--weight_decay', type=float, default=10e-4)
    parser.add_argument('--reference', '-r',  type=str, help='Reference celltypes',default="data/baron_human2.txt")
    parser.add_argument('--transpose', '-t', action='store_true', help='Transpose the input matrix')
    parser.add_argument('--n_epochs','-n',type = int, default=150,help='train epochs')
    parser.add_argument('--gamma',type=float,default=11.,help='coefficient of clustering loss')
    parser.add_argument('--ae_weights',default=None)
    parser.add_argument('--pretrain_batch_size', '-pb', type=int, default=256, help='pretrain Batch size')
    parser.add_argument('--pretrain_epoch', '-pe', type=int, default=50, help='pretrain epochs')
    parser.add_argument('--ae_weight_file',  default='AE_weights_p0_1.pth.tar')
    args = parser.parse_args()


    def show_info():
        # 计算消耗内存
        pid = os.getpid()
        # 模块名比较容易理解：获得当前进程的pid
        p = psutil.Process(pid)
        # 根据pid找到进程，进而找到占用的内存值
        info = p.memory_full_info()
        memory = info.uss / 1024 / 1024
        return memory


    time_start = get_time()  ###开始计时
    start_memory2 = show_info()
    print("开始内存：%fMB" % (start_memory2))

    warnings.filterwarnings("ignore", category=UserWarning)
    DATASET = 'data/yan'  # sys.argv[1]
    filename = DATASET + '.txt'
    data = open(filename)
    n_epochs = args.n_epochs
    head = data.readline().rstrip().split()
    #print(head)
    label_file = open(DATASET + '_label.txt')
    label_dict = {}
    for line in label_file:
        temp = line.rstrip().split()
        if temp:
            label_dict[temp[0]] = temp[1]
    label_file.close()

    label = [] # 将标签提取出来成为一个列表
    for c in head: # 采用字典的方式
        if c in label_dict.keys():
            label.append(label_dict[c])
        else:
            print(c)

    label_set = [] # 剔除掉重复类别之后的label集合
    for c in label:
        if c not in label_set:
            label_set.append(c)
    name_map = {value: idx for idx, value in enumerate(label_set)} # 将label的字符串转化为数字表示，用字典存储
    id_map = {idx: value for idx, value in enumerate(label_set)}
    label = np.asarray([name_map[name] for name in label])
    print(label)
    args = parser.parse_args()
    # Set random seed
    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)



    if torch.cuda.is_available():  # cuda device
        device = 'cuda'
        torch.cuda.set_device(args.gpu)
    else:
        device = 'cpu'
    batch_size = args.batch_size

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")






    adata = sc.read_text(filename, delimiter='\t')
    # adata = sc.read_csv('data/Klein.csv',delimiter=",")
    adata = adata.T
    # sc.pp.filter_cells(adata,min_genes=200)
    # sc.pp.filter_genes(adata,min_cells=3)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)  # 因为patel中有负数值
    sc.pp.highly_variable_genes(adata, flavor='seurat', n_top_genes=3500)
    highly_variable_genes_idx = adata.var['highly_variable']
    adata = adata[:, highly_variable_genes_idx]
    adata_dataset = adata.X.toarray()



    normalizer = MaxAbsScaler()
    dataset = SingleCellDataset(args.dataset,
                                transpose=args.transpose, transforms=[normalizer.fit_transform])



    trainloader = DataLoader(adata_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    testloader = DataLoader(adata_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    # cell_num = dataset.shape[0]
    # input_dim = dataset.shape[1]
    cell_num = adata_dataset.shape[0]
    input_dim = adata_dataset.shape[1]

    # 估计混合高斯模型的初始簇类，最多是15个
    if args.n_centroids is None:
        k = min(estimate_k(adata_dataset), 15)
        print('Estimate k {}'.format(k))
    else:
        k = args.n_centroids


    lr = args.lr
    name = args.dataset.strip('/').split('/')[-1]

    outdir = args.outdir
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    models_dir = os.path.join(outdir,'models')
    log_path = os.path.join(outdir,'logs')




    print("\n**********************************************************************")
    print("Dimensionality reduction and visualization of single-cell RNA-seq data with an improved deep variational autoencoder")
    print("**********************************************************************\n")

    # 从这里开始修改
    train_batch_size = 256
    latent_dim = 10
    c_feature = 9
    single_size = (cell_num,input_dim)
    n_cluster = k

    dims =[input_dim,latent_dim,args.encode_dim,args.decode_dim]
    # 构建网络模型
    encoder = Encoder(
        dims,
        bn=False,
        dropout=0,
        c_feature=9,
        r=9
    )
    gen = Generator(dims)  # 解码器
    gmm = GMM(n_cluster=k,n_features=c_feature)

    # print(gen)
    # print(gmm)
    # print(encoder)


    gen.to(DEVICE)
    encoder.to(DEVICE)
    gmm.to(DEVICE)

    # 创建一个优化器对象，用于训练神经网络的模型


    # 通过训练来选择最好的GMM模型的参数
    best_model = None
    best_score = 0
    # datas = dataset.data.toarray()
    # datas = datas.astype(np.float32)
    # datas = torch.tensor(datas)   # 这个data是包含了所有的样本的数据  （90，20214）
    # datas = datas.to(DEVICE)

    datas = adata_dataset.astype(np.float32)
    datas = torch.tensor(datas)   # 这个data是包含了所有的样本的数据  （90，20214）
    datas = datas.to(DEVICE)

    print("searching best Gaussain mixture prior...")
    Z = []

    with torch.no_grad():
        z,_ = encoder(datas)  # 返回的是两种抽样方式的z,mu,sigma，共6组数据
        # Z.append(z[3])  # 这是用于聚类的7个特征的mu值
        Z.append(z[2]) # 这是用于聚类的7个特征的mu值


    # Y = torch.tensor(Y)
    Z = torch.cat(Z,0).detach().cpu().numpy()
    # Y = np.concatenate(Y,axis=0)



    for i in range(30):
        _gmm = GaussianMixture(n_components=k,covariance_type='diag')
        pre = _gmm.fit_predict(Z)
        acc = cluster_acc(pre,label)[0]*100  # 取返回数据的第一个表示准确率，然后重复三十次来得到一个相对较好的模型

        if best_score < acc:
            best_score = acc
            best_model = _gmm

    datas = torch.tensor(datas,device='cpu')
    torch.cuda.empty_cache()  # 清除GPU上的缓存
    del datas

    print('best accuracy is:{:.4f}'.format(best_score))
    # 得到初始化的GMM参数模块
    gmm.pi_.data = torch.from_numpy(best_model.weights_).to(DEVICE).float()  #(10,7)的结构，潜在特征10维度，每个维度7个特征值？
    gmm.mu_c.data = torch.from_numpy(best_model.means_).to(DEVICE).float()
    gmm.log_sigma2_c.data = torch.from_numpy(best_model.covariances_).to(DEVICE).float()

    print('begain training...')
    # epoch_bar = tqdm(range(0,n_epochs))
    epoch_bar = range(0,n_epochs)
    best_acc,best_nmi,best_ite = 0,0,0
    gen_weight = 0.15

    sd = 2.5
    train_dataset = adata_dataset
    model = pretrain_model(input_dim = input_dim,z_dim=32,n_cluster=n_cluster,encoderLayer=[256,64],decodeLayer=[64,256],sigma=sd,gamma=args.gamma).cuda() # 建立一个模型，首先对数据进行编码然后在对他进行预训练，得到类似经过数据增强后的单细胞数据,但是这个类别函数，我还需要再看看是怎么回事
    pretrain_dataset = []  # 将原始数据和预训练的数据提取出来，然后在将其转化为张量，再通过enumerate对两个数据集分批次训练
    if args.ae_weights is None:
        pretrain_dataset = model.pretrain_autoencoder(x=train_dataset,batch_size=args.pretrain_batch_size,epochs=args.pretrain_epoch,ae_weights=args.ae_weight_file)

    pretrain_dataset = np.concatenate(pretrain_dataset,axis=0)
    # train_dataset_s = dataset.data.toarray()
    train_dataset_s = adata_dataset
    s_dataset = TensorDataset(torch.Tensor(train_dataset_s),torch.Tensor(pretrain_dataset))
    s_dataloder = DataLoader(s_dataset,batch_size=args.batch_size,shuffle=True)


    train_lr = 0.0001
    weight_decay = 5e-4
    b1 = 0.9
    b2 = 0.99
    patience = 20000
    max_iter = 20000


    gen_enc_gmm_ops = torch.optim.Adam(chain(gen.parameters(),
                                             encoder.parameters(),
                                             gmm.parameters(),
                                             ), lr=train_lr, betas=(b1, b2), weight_decay=weight_decay)
    lr_s = StepLR(gen_enc_gmm_ops, step_size=10, gamma=0.95)
    for epoch in epoch_bar:

        total_loss = 0
        """enumerate(trainloader)会自动跳转到SingelCellDataset类，然后执行len函数以及__getitem__函数"""
        for index,(real_cell,pretrain_cell) in enumerate(s_dataloder):
            epoch_lr = adjust_learning_rate(train_lr,gen_enc_gmm_ops,epoch)
            real_cell,pretrain_cell = real_cell.to(DEVICE),\
                                  pretrain_cell.to(DEVICE)
            gen.train()
            gmm.train()
            encoder.train()
            gen_enc_gmm_ops.zero_grad()

            # pretrain_cell = None



            original_z,pretrain_z = encoder(real_cell,pretrain_cell,argument=(epoch>50))
            """
            注意original_z的结构是
            original_z[0]:z1 7, 
            original_z[1]:z2 3, 
            original_z[2]:mu1,
            original_z[3]:mu2, 
            original_z[4]:log_sigma1, 
            original_z[5]:log_sigma2，是一个列表，可以分别将上述的值取出
            """
            fake_cell = gen(original_z[0],gen_weight*original_z[1]) # 重建后的细胞
            # fake_cell = gen(original_z[1]) # 没有聚类特征的时候



            rec_loss =torch.mean(torch.sum(F.binary_cross_entropy(fake_cell,real_cell,reduction='none'),dim=1)) # 重构损失

            augment_loss = F.mse_loss(original_z[2],pretrain_z[2]) +F.mse_loss(original_z[3],pretrain_z[3])
            # augment_loss = F.mse_loss(original_z[3], pretrain_z[3]) # 没有聚类特征的时候
            # 原始细胞和预训练过后的细胞，聚类信息和生成信息的均值和方差的差值

            # 预训练数据和原始数据的gmm预测结果的差别
            c_loss = F.mse_loss(gmm.get_asign(original_z[0])[0],gmm.get_asign(pretrain_z[0])[0])


            kl_loss = torch.mean(-0.5 * torch.sum(1 + original_z[5]-original_z[3] ** 2 - original_z[5].exp(),dim=1),dim=0)
            # 变分自动编码器中的KL散度，以便学到的潜在变量分布更接近于先验分布，从而提高模型的生成和数据重构性

            kls_loss = gmm_Loss(original_z[0],original_z[2],original_z[4],gmm)
            # GMM的损失函数

            # sum_loss = 10 * (augment_loss) + rec_loss + kl_loss
            # sum_loss =  kls_loss + rec_loss + kl_loss # 验证没有agument_loss

            sum_loss = 10 *(augment_loss + c_loss) + kls_loss + rec_loss + kl_loss

            sum_loss.backward()
            gen_enc_gmm_ops.step()  # 更新网络模型的权重

            total_loss += sum_loss  # sum_loss是一个批次的损失函数，total_loss是一轮次的总的损失函数

        # if (epoch + 1) % 20 == 0:
        #     cheek_path =os.path.join(models_dir,"cheekpoint_{}".format(epoch))
        #     os.makedirs(cheek_path,exist_ok=True)
        #     torch.save(gen.state_dict(),os.path.join(cheek_path,'gen.pkl'))
        #     torch.save(encoder.state_dict(),os.path.join(cheek_path,'enc.pkl'))
        #     torch.save(gmm.state_dict(),os.path.join(cheek_path,'gmm.pkl'))

        # print('rec_loss:{:.4f},a_loss:{:.4f},c_loss:{:.4f},kls：{:.4f},kl_loss:{:.4f}'.format(rec_loss,augment_loss,c_loss,kls_loss,kl_loss))

        lr_s.step() # 调整学习率，有助于模型在训练的后期更加稳定地收敛
        # gen_enc_gmm_ops.step()

        # ================================test====================================
        # 模型切换到评估模式，模型不会更新权重，不包含任何随机性，以确保结果的可重复性和稳定性
        gen.eval()
        encoder.eval()
        gmm.eval()


        with torch.no_grad():
            Z = []
            Y = []
            Z1 = []
            Z2 = []
            for _data in testloader:
                _data = _data.to(DEVICE)
                # _data = torch.tensor(_data, dtype=torch.float32).clone().detach()
                _data = _data.clone().detach()
                _data = torch.tensor(_data,dtype=torch.float32)
                z,_ = encoder(_data.to(DEVICE))
                Z.append(z[0])
                Z1 = torch.cat((z[0], z[1]), dim=1)
                Z2.append(Z1)
                # Z.append(z[1])


            Z = torch.cat(Z,0)
            Z2 = torch.cat(Z2,0)
            # Z = Z.cpu().numpy()
            # pred, _ = clustering(Z, k=k)

            pred = gmm.predict(Z)
            acc = cluster_acc(pred,label)[0] * 100
            nmi = NMI(pred,label)
            rand = adjusted_rand_score(pred,label)
            # homo = homogeneity_score(pred,label)
            # completeness = completeness_score(pred,label)

            if best_acc < acc:
                best_acc,best_nmi,best_rand,best_ite = acc,nmi,rand,epoch

                cheek_path = os.path.join(models_dir, "cheekpoint_{}".format(epoch))
                os.makedirs(cheek_path,exist_ok=True)
                # 保存数据的
                # torch.save(gen.state_dict(), os.path.join(cheek_path, 'yan_gen.pkl'))
                # torch.save(encoder.state_dict(), os.path.join(cheek_path, 'yan_enc.pkl'))
                # torch.save(gmm.state_dict(), os.path.join(cheek_path, 'yan_gmm.pkl'))
                # filename = f'./output/feature_pollen_2.txt'
                # label_filename = f'./output/feature_pollen_2_label.txt'
                # np.savetxt(filename,Z2.cpu().numpy(),delimiter=",")
                # np.savetxt(label_filename, pred, delimiter=",")
                # print(best_acc)
                # print("Fnish save")

            logger = open(os.path.join(log_path,"log.txt"),'a')
            # len(trainloader)是表示数据加载器中的批次数量，total_loss是批次累积损失，求一个损失的平均值
            logger.write("[FSVAE]:epoch:{},total_loss:{:.4f},acc:{:.4f},nmi:{:.4f},rand:{:.4f}\n".format(epoch,total_loss/len(trainloader),acc,nmi,rand))
            logger.close()
            # print("[FSVAE]:epoch:{},total_loss:{:.4f},acc:{:.4f},nmi:{:.4f},rand:{:.4f},homo:{:.4f},completeness:{:.4f}".format(epoch,total_loss/len(trainloader),acc,nmi,rand,homo,completeness))

    print('complete training...best_acc is :{:.4f},best_nmi is:{:.4f},best_rand is:{:.4f},iteration is:{:.4f}'.format(best_acc,best_nmi,best_rand,best_ite))
    logger = open(os.path.join(log_path, "log.txt"), 'a')
    # len(trainloader)是表示数据加载器中的批次数量，total_loss是批次累积损失，求一个损失的平均值
    # logger.write('随机数种子：{:.4f}'.format(seed))
    logger.write('complete training...best_acc is :{:.4f},best_nmi is:{:.4f},best_rand is:{:.4f},iteration is:{:.4f}'.format(best_acc,best_nmi,best_rand,best_ite))
    logger.close()

    time = get_time() - time_start
    print("Running Time:" + str(time))
    zuihou = show_info()
    print("使用内存：%fMB" % (zuihou - start_memory2))
    print("结束内存：%fMB" % (zuihou))
    # torch.save(gen.state_dict(),os.path.join(models_dir,'gen.pkl'))
    # torch.save(encoder.state_dict(),os.path.join(models_dir,'enc.pkl'))
    # torch.save(gmm.state_dict(),os.path.join(models_dir,'gmm.pkl'))












import numpy as np
import math
import matplotlib.pyplot as plt
import torch
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from load_data import Data
from model import Mymodel

from torch.nn.init import xavier_normal_, xavier_uniform_, uniform_, zeros_


# 定义X data_num * fearures 原始
# 定义Y data_num * fearures 降维后


class myTSNE:
    def __init__(self, X, perp=30):
        '''类初始化

        Arguments:
            X {Tensor} -- 欲降维数据(n,nfeature)

        Keyword Arguments:
            perp {int} -- 困惑度 (default: {30})
        '''

        self.X = X
        self.N = X.shape[0]
        # 注意此处先在不要求grad时乘好常数再打开grad以保证其是叶子节点
        t.manual_seed(1)
        self.Y = (t.randn(self.N, 2) * 1e-4).requires_grad_()
        self.perp = perp

    def cal_distance(self, data):
        '''计算欧氏距离
        https://stackoverflow.com/questions/37009647/

        Arguments:
            data {Tensor} -- N*features

        Returns:
            Tensor -- N*N 距离矩阵，D[i,j]为distance(data[i],data[j])
        '''

        assert data.dim() == 2, '应为N*features'
        r = (data * data).sum(dim=1, keepdim=True)
        D = r - 2 * data @ data.t() + r.t()
        return D

    def Hbeta(self, D, beta=1.0):
        '''计算给定某一行(n,)与sigma的pj|i与信息熵H

        Arguments:
            D {np array} -- 距离矩阵的i行，不包含与自己的，大小（n-1,)

        Keyword Arguments:
            beta {float} -- 即1/(2sigma^2) (default: {1.0})

        Returns:
            (H,P) -- 信息熵 , 概率pj|i
        '''

        # Compute P-row and corresponding perplexity
        P = np.exp(-D.copy() * beta)
        sumP = sum(P)
        H = np.log(sumP) + beta * np.sum(D * P) / sumP
        P = P / sumP
        return H, P

    def p_j_i(self, distance_matrix, tol=1e-5, perplexity=30):
        '''由距离矩阵计算p(j|i)矩阵，应用二分查找寻找合适sigma

        Arguments:
            distance_matrix {np array} -- 距离矩阵(n,n)

        Keyword Arguments:
            tol {float} -- 二分查找允许误差 (default: {1e-5})
            perplexity {int} -- 困惑度 (default: {30})

        Returns:
            np array -- p(j|i)矩阵
        '''

        print("Computing pairwise distances...")
        (n, d) = self.X.shape
        D = distance_matrix
        P = np.zeros((n, n))
        beta = np.ones((n, 1))
        logU = np.log(perplexity)

        # 遍历每一个数据点
        for i in range(n):

            if i % 500 == 0:
                print("Computing P-values for point %d of %d..." % (i, n))

            # 准备Di，
            betamin = -np.inf
            betamax = np.inf
            Di = D[i, np.concatenate((np.r_[0:i], np.r_[i + 1:n]))]
            (H, thisP) = self.Hbeta(Di, beta[i])
            Hdiff = H - logU
            tries = 0
            # 开始二分搜索，直到满足误差要求或达到最大尝试次数
            while np.abs(Hdiff) > tol and tries < 50:

                if Hdiff > 0:
                    betamin = beta[i].copy()
                    if betamax == np.inf or betamax == -np.inf:
                        beta[i] = beta[i] * 2.
                    else:
                        beta[i] = (beta[i] + betamax) / 2.
                else:
                    betamax = beta[i].copy()
                    if betamin == np.inf or betamin == -np.inf:
                        beta[i] = beta[i] / 2.
                    else:
                        beta[i] = (beta[i] + betamin) / 2.

                (H, thisP) = self.Hbeta(Di, beta[i])
                Hdiff = H - logU
                tries += 1

            # 最后将算好的值写至P，注意pii处为0
            P[i, np.concatenate((np.r_[0:i], np.r_[i + 1:n]))] = thisP

        print("Mean value of sigma: %f" % np.mean(np.sqrt(1 / beta)))
        return P

    def cal_P(self, data):
        '''计算对称相似度矩阵

        Arguments:
            data {Tensor} - - N*N

        Keyword Arguments:
            sigma {Tensor} - - N个sigma(default: {None})

        Returns:
            Tensor - - N*N
        '''
        distance = self.cal_distance(data)  # 计算距离矩阵
        P = self.p_j_i(distance.detach().numpy(), perplexity=self.perp)  # 计算原分布概率矩阵
        P = t.from_numpy(P).float()  # p_j_i为numpy实现的，这里变回Tensor
        P = (P + P.t()) / P.sum()  # 对称化
        P = P * 4.  # 夸张
        P = t.max(P, t.tensor(1e-12))  # 保证计算稳定性
        return P

    def cal_Q(self, data):
        '''计算降维后相似度矩阵

        Arguments:
            data {Tensor} - - Y, N*2

        Returns:
            Tensor - - N*N
        '''

        Q = (1.0 + self.cal_distance(data)) ** -1
        # 对角线强制为零
        Q[t.eye(self.N, self.N, dtype=t.long) == 1] = 0
        Q = Q / Q.sum()
        Q = t.max(Q, t.tensor(1e-12))  # 保证计算稳定性
        return Q

    def train(self, epoch=1000, lr=10, weight_decay=0, momentum=0.9, show=False):
        '''训练

        Keyword Arguments:
            epoch {int} -- 迭代次数 (default: {1000})
            lr {int} -- 学习率，典型10-100 (default: {10})
            weight_decay {int} -- L2正则系数 (default: {0})
            momentum {float} -- 动量 (default: {0.9})
            show {bool} -- 是否显示训练信息 (default: {False})

        Returns:
            Tensor -- 降维结果(n,2)
        '''

        # 先算出原分布的相似矩阵
        P = self.cal_P(self.X)
        optimizer = optim.SGD(
            [self.Y],
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum
        )
        loss_his = []
        print('training started @lr={},epoch={},weight_decay={},momentum={}'.format(
            lr, epoch, weight_decay, momentum))
        for i in range(epoch):
            if i % 100 == 0:
                print('running epoch={}'.format(i))
            if epoch == 100:
                P = P / 4.0  # 100轮后取消夸张
            optimizer.zero_grad()
            Q = self.cal_Q(self.Y)
            loss = (P * t.log(P / Q)).sum()
            loss_his.append(loss.item())
            loss.backward()
            optimizer.step()
        print('train complete!')
        if show:
            print('final loss={}'.format(loss_his[-1]))
            plt.plot(np.log10(loss_his))
            loss_his = []
            plt.show()
        return self.Y.detach()

def main():
    # X = t.load('./x.pt').float()  # 0,1二值化的
    # X = t.from_numpy(PCA(n_components=30).fit_transform(X.numpy())).float()
    d = Data(data_dir='data/FB15k/')
    model = t.load('320CE2T.pth').cpu()

    type1 = 1671
    type2 = 2303
    type3 = 2688
    type4 = 2799
    type5 = 2043
    # type1 = 2713
    # type2 = 2489
    # type3 = 2430
    # type4 = 2521
    # type5 = 2505
    # type1 = 22619
    # type2 = 14924
    # type3 = 14853
    # type4 = 45109
    # type5 = 45100

    type_entity1 = list(d.type_to_entity_dict[type1])
    type_entity2 = list(d.type_to_entity_dict[type2])
    type_entity3 = list(d.type_to_entity_dict[type3])
    type_entity4 = list(d.type_to_entity_dict[type4])
    type_entity5 = list(d.type_to_entity_dict[type5])
    type_temp = []

    for v in type_entity3:
        if v not in type_entity4 and v not in type_entity2 and v not in type_entity1 and v not in type_entity5 :
            type_temp.append(v)
    type_entity3 = type_temp
    type_temp = []
    for v in type_entity2:
        if v not in type_entity4 and v not in type_entity3 and v not in type_entity1 and v not in type_entity5 :
            type_temp.append(v)
    type_entity2 = type_temp

    type_temp = []
    for v in type_entity1:
        if v not in type_entity4 and v not in type_entity3 and v not in type_entity2 and v not in type_entity5 :
            type_temp.append(v)
    type_entity1 = type_temp

    type_temp = []
    for v in type_entity4:
        if v not in type_entity1 and v not in type_entity3 and v not in type_entity2 and v not in type_entity5 :
            type_temp.append(v)
    type_entity4 = type_temp

    type_temp = []
    for v in type_entity5:
        if v not in type_entity4 and v not in type_entity2 and v not in type_entity1 and v not in type_entity3 :
            type_temp.append(v)
    type_entity5 = type_temp


    project_emb1 = model.get_project_vector(t.tensor(type_entity1,dtype=torch.long))
    project_emb2 = model.get_project_vector(t.tensor(type_entity2,dtype=torch.long))
    project_emb3 = model.get_project_vector(t.tensor(type_entity3,dtype=torch.long))
    project_emb4 = model.get_project_vector(t.tensor(type_entity4,dtype=torch.long))
    project_emb5 = model.get_project_vector(t.tensor(type_entity5,dtype=torch.long))

    #X = t.cat((over_emb,project_emb1,project_emb2), dim=0)

    X = t.cat((project_emb1, project_emb2, project_emb3, project_emb4,project_emb5), dim=0)

    X = t.from_numpy(PCA(200).fit_transform(X.detach().numpy())).float()

    C1 = t.zeros(project_emb1.shape[0]).float()
    C2 = 1 * t.ones(project_emb2.shape[0]).float()
    C3 = 2 * t.ones(project_emb3.shape[0]).float()
    C4 = 3 * t.ones(project_emb4.shape[0]).float()
    C5 = 4 * t.ones(project_emb5.shape[0]).float()

    C = t.cat((C1, C2, C3, C4,C5), dim=0)
    #C = t.cat((C1, C2,C3), dim=0)

    font2 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 30,
             }

    T = myTSNE(X)
    res = T.train(epoch=2000,lr=35, weight_decay=0.,momentum=0.5, show=True).numpy()
    plt.xticks([])  # 去掉横坐标值
    plt.yticks([])
    plt.scatter(res[:, 0], res[:, 1], c=C.numpy())
    plt.savefig("new_result1.jpeg", dpi=800, bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    main()

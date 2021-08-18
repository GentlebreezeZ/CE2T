import torch, math, itertools, os
from torch.nn import functional as F, Parameter
from torch.autograd import Variable
from itertools import permutations, product

from torch.nn.init import xavier_normal_, xavier_uniform_, uniform_, zeros_
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np

from itertools import chain
import config


def truncated_normal_(tensor, mean=0, std=1):
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)


# class Mymodel(torch.nn.Module):
#     def __init__(self, num_entity, num_entity_type, entity_embedding_size, entity_type_embedding_size, droupt_input,
#                  droupt_feature, droupt_output, filt_h, filt_w, num_filters=32, conv_stride = 1,h=20, w=20, pool_h=2, pool_w=2,
#                  pool_stride=2):
#         super(Mymodel, self).__init__()
#         self.num_filters = num_filters
#         self.filt_h = filt_h
#         self.filt_w = filt_w
#         self.conv_stride = conv_stride
#         self.h = h
#         self.w = w
#         self.pool_h = pool_h
#         self.pool_w = pool_w
#         self.pool_stride = pool_stride
#         self.emb_entity_size = entity_embedding_size
#         self.emb_type_size = entity_type_embedding_size
#
#         # self.batchNorm0 = torch.nn.BatchNorm2d(1, momentum=0.1)
#         self.emb_type = torch.nn.Embedding(num_entity_type, entity_type_embedding_size, padding_idx=0)
#         self.emb_entities = torch.nn.Embedding(num_entity, entity_embedding_size, padding_idx=0)
#
#         self.trans_matrix = torch.nn.Parameter(torch.Tensor(entity_embedding_size, entity_type_embedding_size))
#
#         self.conv1 = torch.nn.Conv2d(1, self.num_filters,
#                                      (self.filt_h, self.filt_w), 2, 0, bias=True)
#         truncated_normal_(self.conv1.weight, mean=0.0, std=0.1)
#         zeros_(self.conv1.bias.data)
#
#         self.pool = torch.nn.MaxPool2d(kernel_size=(self.pool_h, self.pool_w), stride=self.pool_stride)
#
#         self.batchNorm1 = torch.nn.BatchNorm2d(1)
#         self.batchNorm2 = torch.nn.BatchNorm1d(self.emb_type_size)
#         self.batchNorm3 = torch.nn.BatchNorm2d(num_filters)
#
#         self.dropout_input = torch.nn.Dropout(droupt_input)
#         self.dropout_feature = torch.nn.Dropout(droupt_feature)
#         self.dropout_output = torch.nn.Dropout(droupt_output)
#
#         self.register_parameter('b', Parameter(torch.zeros(num_entity_type)))
#
#         fc_length = (int(((((h - self.filt_h) / self.conv_stride) + 1) - self.pool_h) / self.pool_stride + 1) * int(
#             ((((w - self.filt_w) / self.conv_stride) + 1) - self.pool_w) / self.pool_stride + 1)) * self.num_filters
#         self.f_FCN_net = torch.nn.Linear(fc_length, self.emb_type_size)
#         xavier_normal_(self.f_FCN_net.weight.data)
#         zeros_(self.f_FCN_net.bias.data)
#
#         self.loss = torch.nn.BCELoss()
#
#     def init(self):
#         torch.nn.init.xavier_uniform_(self.emb_type.weight)
#         torch.nn.init.xavier_uniform_(self.emb_entities.weight)
#
#         stdv_transfer = 1. / math.sqrt(self.emb_type_size)
#         self.trans_matrix.data.uniform_(-stdv_transfer, stdv_transfer)
#
#     def forward(self, x_batch):
#         pass
#         # torch.mm(entity_embedding_vec, self.trans_matrix)
#         e = self.emb_entities(x_batch).view(-1, self.emb_entity_size)
#         e = torch.mm(e, self.trans_matrix)
#         e = e.view(-1, 1, self.h, self.w)
#
#         x = self.batchNorm1(e)
#         x = self.dropout_input(x)
#
#         x = self.conv1(x)
#         x = self.batchNorm3(x)
#         x = F.relu(x)
#
#         x = self.pool(x)
#
#         x = self.dropout_feature(x)
#         x = x.view(e.size(0), -1)
#
#         x = self.f_FCN_net(x)
#
#         x = self.dropout_output(x)
#
#         x = self.batchNorm2(x)
#
#         x = F.relu(x)
#
#         x = torch.mm(x, self.emb_type.weight.transpose(1, 0))
#         x += self.b.expand_as(x)
#         pred = F.sigmoid(x)
#         return pred



class Mymodel(torch.nn.Module):
    def __init__(self, num_entity, num_entity_type, entity_embedding_size, entity_type_embedding_size, droupt_input,
                 droupt_feature, droupt_output, filt_h, filt_w, num_filters=32, conv_stride = 1,h=20, w=20, pool_h=2, pool_w=2,
                 pool_stride=2):
        super(Mymodel, self).__init__()
        self.num_filters = num_filters
        self.filt_h = filt_h
        self.filt_w = filt_w
        self.conv_stride = conv_stride
        self.h = h
        self.w = w
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.pool_stride = pool_stride
        self.emb_entity_size = entity_embedding_size
        self.emb_type_size = entity_type_embedding_size

        # self.batchNorm0 = torch.nn.BatchNorm2d(1, momentum=0.1)
        self.emb_type = torch.nn.Embedding(num_entity_type, entity_type_embedding_size, padding_idx=0)
        self.emb_entities = torch.nn.Embedding(num_entity, entity_embedding_size, padding_idx=0)

        self.conv1 = torch.nn.Conv2d(1, self.num_filters,
                                     (self.filt_h, self.filt_w), 2, 0, bias=True)
        truncated_normal_(self.conv1.weight, mean=0.0, std=0.1)
        zeros_(self.conv1.bias.data)

        self.pool = torch.nn.MaxPool2d(kernel_size=(self.pool_h, self.pool_w), stride=self.pool_stride)

        self.batchNorm1 = torch.nn.BatchNorm2d(1)
        self.batchNorm2 = torch.nn.BatchNorm1d(self.emb_type_size)
        self.batchNorm3 = torch.nn.BatchNorm2d(num_filters)

        self.dropout_input = torch.nn.Dropout(droupt_input)
        self.dropout_feature = torch.nn.Dropout(droupt_feature)
        self.dropout_output = torch.nn.Dropout(droupt_output)

        self.register_parameter('b', Parameter(torch.zeros(num_entity_type)))

        fc_length = (int(((((h - self.filt_h) / self.conv_stride) + 1) - self.pool_h) / self.pool_stride + 1) * int(
            ((((w - self.filt_w) / self.conv_stride) + 1) - self.pool_w) / self.pool_stride + 1)) * self.num_filters
        self.f_FCN_net = torch.nn.Linear(fc_length, self.emb_type_size)
        xavier_normal_(self.f_FCN_net.weight.data)
        zeros_(self.f_FCN_net.bias.data)

        self.loss = torch.nn.BCELoss()

    def init(self):
        torch.nn.init.xavier_uniform_(self.emb_type.weight)
        torch.nn.init.xavier_uniform_(self.emb_entities.weight)

    def forward(self, x_batch):
        e = self.emb_entities(x_batch).view(-1, self.emb_entity_size)
        e = e.view(-1, 1, self.h, self.w)

        x = self.batchNorm1(e)
        x = self.dropout_input(x)

        x = self.conv1(x)
        x = self.batchNorm3(x)
        x = F.relu(x)

        x = self.pool(x)

        x = self.dropout_feature(x)
        x = x.view(e.size(0), -1)

        x = self.f_FCN_net(x)

        x = self.dropout_output(x)

        x = self.batchNorm2(x)

        x = F.relu(x)

        x = torch.mm(x, self.emb_type.weight.transpose(1, 0))
        x += self.b.expand_as(x)
        pred = F.sigmoid(x)
        return pred

    def get_entity_vec(self,x_batch):
        entity_id2x = torch.tensor(x_batch, dtype=torch.long).flatten()
        entity_embedding_vec = self.emb_entities.cpu()(entity_id2x).view(len(x_batch), -1)
        return entity_embedding_vec

    def get_type_vec(self,t):
        type_id2x = torch.tensor(t, dtype=torch.long).flatten()
        type_embedding_vec = self.emb_type.cpu()(type_id2x)
        return type_embedding_vec

    def get_project_vector(self, x_batch):
        e = self.emb_entities(x_batch).view(-1, self.emb_entity_size)
        e = e.view(-1, 1, self.h, self.w)

        x = self.batchNorm1(e)
        x = self.dropout_input(x)

        x = self.conv1(x)
        x = self.batchNorm3(x)
        x = F.relu(x)

        x = self.pool(x)

        x = self.dropout_feature(x)
        x = x.view(e.size(0), -1)

        x = self.f_FCN_net(x)


        return x


class ConnectE(torch.nn.Module):
    def __init__(self, num_entity, num_entity_type, entity_embedding_size, entity_type_embedding_size,margin=2.0):
        super(ConnectE, self).__init__()
        self.emb_entity_size = entity_embedding_size
        self.emb_type_size = entity_type_embedding_size
        self.margin = margin

        self.emb_type = torch.nn.Embedding(num_entity_type, entity_type_embedding_size, padding_idx=0)
        self.emb_entities = torch.nn.Embedding(num_entity, entity_embedding_size, padding_idx=0)

        self.trans_matrix = torch.nn.Parameter(torch.Tensor(entity_embedding_size, entity_type_embedding_size))
        self.loss = torch.nn.BCELoss()

    def init(self):
        torch.nn.init.xavier_uniform_(self.emb_type.weight)
        torch.nn.init.xavier_uniform_(self.emb_entities.weight)

        # entity_vec = np.load('data/FB15k/entity_vec_transe.npy')
        # self.emb_entities.weight = torch.nn.Parameter(torch.tensor(entity_vec))
        # self.emb_entities.requires_grad_(requires_grad=False)


        stdv_transfer = 1. / math.sqrt(self.emb_type_size)
        self.trans_matrix.data.uniform_(-stdv_transfer, stdv_transfer)


    def forward(self, x_batch):
        entity_emb_vec = self.emb_entities(x_batch).view(len(x_batch), -1)
        entity_emb_projrct = torch.mm(entity_emb_vec, self.trans_matrix)

        x = self.margin - torch.norm(entity_emb_projrct.unsqueeze(1) - self.emb_type.weight, p=1, dim=2)
        pred = torch.sigmoid(x)
        return pred

class ETE(torch.nn.Module):
    def __init__(self, num_entity, num_entity_type, entity_embedding_size, entity_type_embedding_size,margin=2.0):
        super(ETE, self).__init__()
        self.emb_entity_size = entity_embedding_size
        self.emb_type_size = entity_type_embedding_size
        self.margin = margin

        self.emb_type = torch.nn.Embedding(num_entity_type, entity_type_embedding_size, padding_idx=0)
        self.emb_entities = torch.nn.Embedding(num_entity, entity_embedding_size, padding_idx=0)

        self.loss = torch.nn.BCELoss()

    def init(self):
        torch.nn.init.xavier_uniform_(self.emb_type.weight)
        torch.nn.init.xavier_uniform_(self.emb_entities.weight)
        # entity_vec = np.load('data/FB15k/entity_vec_transe.npy')
        # self.emb_entities.weight = torch.nn.Parameter(torch.tensor(entity_vec))
        # self.emb_entities.requires_grad_(requires_grad=False)


    def forward(self, x_batch):
        entity_emb_vec = self.emb_entities(x_batch).view(len(x_batch), -1)

        x = self.margin - torch.norm(entity_emb_vec.unsqueeze(1) - self.emb_type.weight, p=1, dim=2)
        pred = torch.sigmoid(x)

        return pred




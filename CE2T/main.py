from copy import deepcopy
import torch
import numpy as np
from model import Mymodel, ConnectE,ETE
from datetime import datetime
import random
from collections import defaultdict
import config
from torch.optim.lr_scheduler import ExponentialLR
from logger_init import get_logger

logger = get_logger('train', True, file_log=True,filename='-'+str(config.args.dataset) + '-'+ str(config.args.model_name))
logger.info('START TIME : {}'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
#torch.cuda.set_device(1)

def chunks(L, n):
    """ Yield successive n-sized chunks from L."""
    for i in range(0, len(L), n):
        yield L[i:i + n]


class Experiment:
    def __init__(self, decay_rate=0.99, batch_size=128, learning_rate=0.001, entity_embedding_dim=200,
                 entity_type_embedding_dim=100,
                 epochs=50000, num_filters=200, droupt_input=0.2, droupt_feature=0.2, droupt_output=0.2,
                 label_smoothing=0.1, cuda=True, filt_h=1, filt_w=9, conv_stride=1, h=20, w=20, pool_h=2, pool_w=2,
                 pool_stride=2, model_name='CE2T', margin=2.0):
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.embedding_entity_dim = entity_embedding_dim
        self.embedding_entity_type_dim = entity_type_embedding_dim
        self.epochs = epochs
        self.num_filters = num_filters
        self.num_entity = len(config.d.entity_idxs)
        self.num_entity_type = len(config.d.entity_type_idxs)
        self.droupt_input = droupt_input
        self.droupt_feature = droupt_feature
        self.droupt_output = droupt_output
        self.label_smoothing = label_smoothing
        self.cuda = cuda
        self.filt_h = filt_h
        self.filt_w = filt_w
        self.conv_stride = conv_stride
        self.h = h
        self.w = w
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.pool_stride = pool_stride
        self.decay_rate = decay_rate
        self.model_name = model_name
        self.margin = margin
        logger.info('-------------model_name-------------: {} '.format(model_name))
        if self.model_name == 'CE2T':
            logger.info('embedding_entity_dim: {} '.format(self.embedding_entity_dim))
            logger.info('embedding_entity_type_dim: {} '.format(self.embedding_entity_type_dim))
            logger.info('batch_size: {} '.format(batch_size))
            logger.info('learning_rate: {} '.format(learning_rate))
            logger.info('num_filters: {} '.format(num_filters))
            logger.info('droupt_input: {} '.format(droupt_input))
            logger.info('droupt_feature: {} '.format(droupt_feature))
            logger.info('droupt_output: {} '.format(droupt_output))
            logger.info('label_smoothing: {} '.format(label_smoothing))
            logger.info('filt_h: {} '.format(filt_h))
            logger.info('filt_w: {} '.format(filt_w))
            logger.info('h: {} '.format(h))
            logger.info('w: {} '.format(w))
            logger.info('pool_h: {} '.format(pool_h))
            logger.info('pool_w: {} '.format(pool_w))
            logger.info('pool_stride: {} '.format(pool_stride))
        elif self.model_name == 'ConnectE':
            logger.info('embedding_entity_dim: {} '.format(self.embedding_entity_dim))
            logger.info('embedding_entity_type_dim: {} '.format(self.embedding_entity_type_dim))
            logger.info('batch_size: {} '.format(batch_size))
            logger.info('learning_rate: {} '.format(learning_rate))
            logger.info('label_smoothing: {} '.format(label_smoothing))
        elif self.model_name == 'ETE':
            logger.info('embedding_entity_dim: {} '.format(self.embedding_entity_dim))
            logger.info('embedding_entity_type_dim: {} '.format(self.embedding_entity_type_dim))
            logger.info('batch_size: {} '.format(batch_size))
            logger.info('learning_rate: {} '.format(learning_rate))
            logger.info('label_smoothing: {} '.format(label_smoothing))

    def get_er_vocab(self, data):
        er_vocab = defaultdict(list)
        for et in data:
            er_vocab[(et[0])].append(et[1])
        return er_vocab

    def get_batch(self, er_vocab, er_vocab_pairs, idx):
        batch = er_vocab_pairs[idx:min(idx + self.batch_size, len(er_vocab_pairs))]
        targets = np.zeros((len(batch), len(config.d.types)))
        for idx, pair in enumerate(batch):
            targets[idx, er_vocab[pair]] = 1.
        targets = torch.FloatTensor(targets)
        if self.cuda:
            targets = targets.cuda()
        return np.array(batch), targets

    def evaluate(self, model, data):
        hits = []
        ranks = []
        for i in range(10):
            hits.append([])

        test_data_idxs = data
        er_vocab = self.get_er_vocab(config.d.over_data)

        print("Number of data points: %d" % len(test_data_idxs))

        for i in range(0, len(test_data_idxs), self.batch_size):
            data_batch, _ = self.get_batch(er_vocab, test_data_idxs, i)
            e_idx = torch.tensor(data_batch[:, 0])
            t_idx = torch.tensor(data_batch[:, 1])
            if self.cuda:
                e_idx = e_idx.cuda()
                t_idx = t_idx.cuda()
            predictions = model.forward(e_idx)

            for j in range(data_batch.shape[0]):
                filt = er_vocab[(data_batch[j][0])]
                target_value = predictions[j, t_idx[j]].item()
                predictions[j, filt] = 0.0
                predictions[j, t_idx[j]] = target_value

            sort_values, sort_idxs = torch.sort(predictions, dim=1, descending=True)

            # sort_idxs = sort_idxs.cpu().numpy()
            for j in range(data_batch.shape[0]):
                rank = torch.where(sort_idxs[j] == t_idx[j])[0][0].cpu().item()
                ranks.append(rank + 1)

                for hits_level in range(10):
                    if rank <= hits_level:
                        hits[hits_level].append(1.0)
                    else:
                        hits[hits_level].append(0.0)

        logger.info('Hits @10: {0}'.format(np.mean(hits[9])))
        logger.info('Hits @3: {0}'.format(np.mean(hits[2])))
        logger.info('Hits @1: {0}'.format(np.mean(hits[0])))
        logger.info('Mean reciprocal rank: {0}'.format(np.mean(1. / np.array(ranks))))

        # print('Hits @10: {0}'.format(np.mean(hits[9])))
        # print('Hits @3: {0}'.format(np.mean(hits[2])))
        # print('Hits @1: {0}'.format(np.mean(hits[0])))
        # print('Mean rank: {0}'.format(np.mean(ranks)))
        # print('Mean reciprocal rank: {0}'.format(np.mean(1. / np.array(ranks))))

    def evaluate_1_1(self, model, data):
        hits = []
        ranks = []
        for i in range(10):
            hits.append([])

        test_data_idxs = data
        er_vocab = self.get_er_vocab(config.d.over_data)

        print("Number of data points: %d" % len(test_data_idxs))

        for i in range(0, len(test_data_idxs), self.batch_size):
            data_batch, _ = self.get_batch(er_vocab, test_data_idxs, i)
            e_idx = torch.tensor(data_batch[:, 0])
            t_idx = torch.tensor(data_batch[:, 1])
            if self.cuda:
                e_idx = e_idx.cuda()
                t_idx = t_idx.cuda()
            predictions = model.forward(e_idx)

            for j in range(data_batch.shape[0]):
                filt = er_vocab[(data_batch[j][0])]
                target_value = predictions[j, t_idx[j]].item()
                predictions[j, filt] = 0.0
                predictions[j, t_idx[j]] = target_value

            sort_values, sort_idxs = torch.sort(predictions, dim=1, descending=True)

            # sort_idxs = sort_idxs.cpu().numpy()
            for j in range(data_batch.shape[0]):
                rank = torch.where(sort_idxs[j] == t_idx[j])[0][0].cpu().item()
                ranks.append(rank + 1)

                for hits_level in range(10):
                    if rank <= hits_level:
                        hits[hits_level].append(1.0)
                    else:
                        hits[hits_level].append(0.0)

        logger.info('1-1 Hits @10: {0}'.format(np.mean(hits[9])))
        logger.info('1-1 Hits @3: {0}'.format(np.mean(hits[2])))
        logger.info('1-1 Hits @1: {0}'.format(np.mean(hits[0])))
        logger.info('1-1 Mean reciprocal rank: {0}'.format(np.mean(1. / np.array(ranks))))

    def evaluate_1_N(self, model, data):
        hits = []
        ranks = []
        for i in range(10):
            hits.append([])

        test_data_idxs = data
        er_vocab = self.get_er_vocab(config.d.over_data)

        print("Number of data points: %d" % len(test_data_idxs))

        for i in range(0, len(test_data_idxs), self.batch_size):
            data_batch, _ = self.get_batch(er_vocab, test_data_idxs, i)
            e_idx = torch.tensor(data_batch[:, 0])
            t_idx = torch.tensor(data_batch[:, 1])
            if self.cuda:
                e_idx = e_idx.cuda()
                t_idx = t_idx.cuda()
            predictions = model.forward(e_idx)

            for j in range(data_batch.shape[0]):
                filt = er_vocab[(data_batch[j][0])]
                target_value = predictions[j, t_idx[j]].item()
                predictions[j, filt] = 0.0
                predictions[j, t_idx[j]] = target_value

            sort_values, sort_idxs = torch.sort(predictions, dim=1, descending=True)

            # sort_idxs = sort_idxs.cpu().numpy()
            for j in range(data_batch.shape[0]):
                rank = torch.where(sort_idxs[j] == t_idx[j])[0][0].cpu().item()
                ranks.append(rank + 1)

                for hits_level in range(10):
                    if rank <= hits_level:
                        hits[hits_level].append(1.0)
                    else:
                        hits[hits_level].append(0.0)

        logger.info('1-N Hits @10: {0}'.format(np.mean(hits[9])))
        logger.info('1-N Hits @3: {0}'.format(np.mean(hits[2])))
        logger.info('1-N Hits @1: {0}'.format(np.mean(hits[0])))
        logger.info('1-N Mean reciprocal rank: {0}'.format(np.mean(1. / np.array(ranks))))

    def train_and_eval(self):
        er_vocab = self.get_er_vocab(config.d.train_idxs)
        er_vocab_pairs = list(er_vocab.keys())

        # num_entity, num_entity_type, entity_embedding_size, entity_type_embedding_size, droupt_input,
        # droupt_feature, droupt_output, filt_h, filt_w, num_filters = 32):

        if self.model_name == 'CE2T':
            model = Mymodel(num_entity=len(config.d.entity_idxs), num_entity_type=len(config.d.entity_type_idxs),
                            entity_embedding_size=self.embedding_entity_dim,
                            entity_type_embedding_size=self.embedding_entity_type_dim, droupt_input=self.droupt_input,
                            droupt_feature=self.droupt_feature, droupt_output=self.droupt_output, filt_h=self.filt_h,
                            filt_w=self.filt_w, num_filters=self.num_filters, h=self.h, w=self.w, pool_h=self.pool_h,
                            pool_w=self.pool_w, pool_stride=self.pool_stride, conv_stride=self.conv_stride)
        elif self.model_name == 'ConnectE':
            #    def __init__(self, num_entity, num_entity_type, entity_embedding_size, entity_type_embedding_size,margin=2.0):
            model = ConnectE(num_entity=len(config.d.entity_idxs), num_entity_type=len(config.d.entity_type_idxs),
                             entity_embedding_size=self.embedding_entity_dim,
                             entity_type_embedding_size=self.embedding_entity_type_dim, margin=self.margin)
        elif self.model_name == 'ETE':
            #(self, num_entity, num_entity_type, entity_embedding_size, entity_type_embedding_size, margin=2.0):
            model = ETE(num_entity=len(config.d.entity_idxs), num_entity_type=len(config.d.entity_type_idxs),
                             entity_embedding_size=self.embedding_entity_dim,
                             entity_type_embedding_size=self.embedding_entity_type_dim, margin=self.margin)
        model.init()
        if self.cuda:
            model.cuda()
        opt = torch.optim.Adam(model.parameters(), lr=self.learning_rate)

        if self.decay_rate:
            scheduler = ExponentialLR(opt, self.decay_rate)

        for it in range(1, self.epochs + 1):
            model.train()
            losses = []
            np.random.shuffle(er_vocab_pairs)
            for j in range(0, len(er_vocab_pairs), self.batch_size):
                data_batch, targets = self.get_batch(er_vocab, er_vocab_pairs, j)
                opt.zero_grad()
                e_idx = torch.tensor(data_batch)
                if self.cuda:
                    e_idx = e_idx.cuda()
                predictions = model.forward(e_idx)
                if self.label_smoothing:
                    targets = ((1.0 - self.label_smoothing) * targets) + (1.0 / targets.size(1))
                loss = model.loss(predictions, targets)
                loss.backward()
                opt.step()
            if self.decay_rate:
                scheduler.step()
            losses.append(loss.item())

            print(it)
            print(np.mean(losses))
            model.eval()
            with torch.no_grad():
                # print("Validation:")
                # self.evaluate(model, config.d.valid_idxs)
                if it >= 400 and it % 10 == 0:
                    logger.info('------------------------------------it:{}'.format(it))
                    self.evaluate(model, config.d.test_idxs)
                    #self.evaluate_1_1(model,config.d.test_data_1_1)
                    #self.evaluate_1_N(model,config.d.test_data_1_N)
                    #torch.save(model, 'ce2t.pth')


if __name__ == '__main__':
    experiment = Experiment(batch_size=config.args.batchsize, learning_rate=config.args.learningrate,
                            entity_embedding_dim=config.args.embsize_entity,
                            entity_type_embedding_dim=config.args.embsize_entity_type, epochs=config.args.epochs,
                            num_filters=config.args.num_filters, droupt_input=config.args.droupt_input,
                            droupt_feature=config.args.droupt_feature,
                            droupt_output=config.args.droupt_output, label_smoothing=config.args.label_smoothing,
                            cuda=config.args.CUDA, filt_h=config.args.filt_h, filt_w=config.args.filt_w,
                            h=config.args.h, w=config.args.w, pool_h=config.args.pool_h, pool_w=config.args.pool_w,
                            pool_stride=config.args.pool_stride, conv_stride=config.args.conv_stride,
                            model_name=config.args.model_name, margin=config.args.margin)
    seed = 1234
    np.random.seed(seed)
    torch.manual_seed(seed)
    experiment.train_and_eval()

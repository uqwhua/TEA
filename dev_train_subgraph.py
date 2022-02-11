import torch
import numpy as np
import torch.nn as nn
from pathlib import Path
from shutil import rmtree
import torch.nn.functional as F
from collections import OrderedDict

from torch.optim import Adagrad
from dev_cluster import ClusterModel
from dev_load_data import LoadDataDev
from dev_settings import DevSettings
from models import MultiLayerGCN, AttSeq
from dev_base import DevAttSeq
from util import print_time_info, set_random_seed, get_hits
from tqdm import tqdm

import sys

sys.path.append('..')
sys.path.append('/home/uqyliu42/workspace/explore')


def cosine_similarity_nbyn(a, b):
    '''
    a shape: [num_item_1, embedding_dim]
    b shape: [num_item_2, embedding_dim]
    return sim_matrix: [num_item_1, num_item_2]
    '''
    a = a / torch.clamp(a.norm(dim=-1, keepdim=True, p=2), min=1e-10)
    b = b / torch.clamp(b.norm(dim=-1, keepdim=True, p=2), min=1e-10)
    if b.shape[0] * b.shape[1] > 20000 * 128:
        return cosine_similarity_nbyn_batched(a, b)
    return torch.mm(a, b.t())


def cosine_similarity_nbyn_batched(a, b):
    '''
    a shape: [num_item_1, embedding_dim]
    b shape: [num_item_2, embedding_dim]
    return sim_matrix: [num_item_1, num_item_2]
    '''
    batch_size = 512
    data_num = b.shape[0]
    b = b.t()
    sim_matrix = []
    for i in range(0, data_num, batch_size):
        sim_matrix.append(torch.mm(a, b[:, i:i + batch_size]).cpu())
    sim_matrix = torch.cat(sim_matrix, dim=1)
    return sim_matrix


def torch_l2distance(a, b):
    # shape a = (num_ent1, embed_dim)
    # shape b = (num_ent2, embed_dim)
    assert len(a.size()) == len(b.size()) == 2
    assert a.shape[1] == b.shape[1]
    x1 = torch.sum(torch.pow(a, 2), dim=-1).view(-1, 1)  # shape = (num_ent1, 1)
    x2 = torch.sum(torch.pow(b, 2), dim=-1).view(-1, 1)  # shape = (num_ent2, 1)
    if b.shape[0] < 20000:
        x3 = -2 * torch.mm(a, b.t())  # shape = (num_ent1, num_ent2)
    else:
        x3 = -2 * torch_mm_batched(a, b.t())
    is_cuda = x3.is_cuda
    if not is_cuda:
        x1 = x1.cpu()
        x2 = x2.cpu()

    sim = x3 + x1 + x2.t()
    return sim.pow(0.5)


def torch_mm_batched(a, b):
    '''
    a shape: [dim1, dim2]
    b shape: [dim2, dim3]
    return sim_matrix: [dim1, dim3]
    '''
    batch_size = 512
    cols_num = b.shape[-1]
    output = []
    for i in range(0, cols_num, batch_size):
        output.append(torch.mm(a, b[:, i:i + batch_size]).cpu())
    output = torch.cat(output, dim=1)
    return output


def get_nearest_neighbor(sim, nega_sample_num=25):
    # Sim do not have to be a square matrix
    # Let us assume sim is a numpy array
    ranks = torch.argsort(sim, dim=1)
    ranks = ranks[:, 1:nega_sample_num + 1]
    return ranks


class AlignLoss(nn.Module):
    def __init__(self, margin, p=2, reduction='mean'):
        super(AlignLoss, self).__init__()
        self.p = p
        self.criterion = nn.TripletMarginLoss(margin, p=p, reduction=reduction)

    def forward(self, repre_sr, repre_tg):
        '''
        score shape: [batch_size, 2, embedding_dim]
        '''
        # distance = torch.abs(score).sum(dim=-1) * self.re_scale
        sr_true = repre_sr[:, 0, :]
        sr_nega = repre_sr[:, 1, :]
        tg_true = repre_tg[:, 0, :]
        tg_nega = repre_tg[:, 1, :]

        loss = self.criterion(torch.cat((sr_true, tg_true), dim=0), torch.cat((tg_true, sr_true), dim=0),
                              torch.cat((tg_nega, sr_nega), dim=0))
        return loss


class AuxPathLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(AuxPathLoss, self).__init__()

        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, aux_info):
        '''
        score shape: [batch_size, 2, embedding_dim]
        '''

        pos, neg1, neg2 = aux_info
        target_pos = torch.ones_like(pos)
        target_neg = torch.zeros_like(neg1)

        loss = self.criterion(pos, target_pos) \
               + self.criterion(neg1, target_neg) + self.criterion(neg2, target_neg)
        return loss


def sort_and_keep_indices(matrix, device):
    batch_size = 512
    data_len = matrix.shape[0]
    sim_matrix = []
    indice_list = []
    for i in range(0, data_len, batch_size):
        batch = matrix[i:i + batch_size]
        batch = torch.from_numpy(batch).to(device)
        sorted_batch, indices = torch.sort(batch, dim=-1)
        sorted_batch = sorted_batch[:, :500].cpu()
        indices = indices[:, :500].cpu()
        sim_matrix.append(sorted_batch)
        indice_list.append(indices)
    sim_matrix = torch.cat(sim_matrix, dim=0).numpy()
    indice_array = torch.cat(indice_list, dim=0).numpy()
    sim = np.concatenate([np.expand_dims(sim_matrix, 0), np.expand_dims(indice_array, 0)], axis=0)
    return sim


class GNNChannel(nn.Module):

    def __init__(self, ent_num_sr, ent_num_tg, dim, layer_num, drop_out, channels, settings=None):
        super(GNNChannel, self).__init__()
        assert len(channels) == 1

        if 'structure' in channels:
            self.gnn = StruGNN(ent_num_sr, ent_num_tg, dim, layer_num, drop_out, **channels['structure'])
        if 'attribute' in channels:
            self.gnn = AttSeq(layer_num, ent_num_sr, ent_num_tg, dim, drop_out, residual=True,
                              **channels['attribute'])
        if 'name' in channels:
            self.gnn = NameGCN(dim, layer_num, drop_out, **channels['name'])
        if 'time' in channels:
            self.gnn = DevAttSeq(layer_num, ent_num_sr, ent_num_tg, dim, drop_out, settings=settings, residual=True,
                                 **channels['time'])

    def forward(self, sr_ent_seeds, tg_ent_seeds, return_all_ent=False):
        sr_seed_hid, tg_seed_hid, sr_ent_hid, tg_ent_hid = self.gnn.forward(sr_ent_seeds,
                                                                            tg_ent_seeds)
        return sr_seed_hid, tg_seed_hid, sr_ent_hid, tg_ent_hid

    def predict(self, sr_ent_seeds, tg_ent_seeds, return_all_ent=False):
        with torch.no_grad():
            sr_seed_hid, tg_seed_hid, _, _ = self.forward(sr_ent_seeds, tg_ent_seeds)
            if isinstance(self.gnn, NameGCN):
                sim = torch_l2distance(sr_seed_hid, tg_seed_hid)
            else:
                sim = - cosine_similarity_nbyn(sr_seed_hid, tg_seed_hid)
        return sim

    def negative_sample(self, sr_ent_seeds, tg_ent_seeds, return_all_ent=True):
        with torch.no_grad():
            sr_seed_hid, tg_seed_hid, sr_ent_hid, tg_ent_hid = self.forward(sr_ent_seeds, tg_ent_seeds)
            if isinstance(self.gnn, NameGCN):
                sim_sr = torch_l2distance(sr_seed_hid, sr_ent_hid)
                sim_tg = torch_l2distance(tg_seed_hid, tg_ent_hid)
            else:
                sim_sr = - cosine_similarity_nbyn(sr_seed_hid, sr_ent_hid)
                sim_tg = - cosine_similarity_nbyn(tg_seed_hid, tg_ent_hid)
        return sim_sr, sim_tg


class NameGCN(nn.Module):
    def __init__(self, dim, layer_num, drop_out, sr_ent_embed, tg_ent_embed, edges_sr, edges_tg):
        super(NameGCN, self).__init__()
        self.embedding_sr = nn.Parameter(sr_ent_embed, requires_grad=False)
        self.embedding_tg = nn.Parameter(tg_ent_embed, requires_grad=False)
        self.edges_sr = nn.Parameter(edges_sr, requires_grad=False)
        self.edges_tg = nn.Parameter(edges_tg, requires_grad=False)
        in_dim = sr_ent_embed.shape[1]
        self.gcn = MultiLayerGCN(in_dim, dim, layer_num, drop_out, featureless=False, residual=True)

    def forward(self, sr_ent_seeds, tg_ent_seeds):
        sr_ent_hid = self.gcn(self.edges_sr, self.embedding_sr)
        tg_ent_hid = self.gcn(self.edges_tg, self.embedding_tg)
        sr_seed_hid = sr_ent_hid[sr_ent_seeds]
        tg_seed_hid = tg_ent_hid[tg_ent_seeds]
        return sr_seed_hid, tg_seed_hid, sr_ent_hid, tg_ent_hid


class StruGNN(nn.Module):
    def __init__(self, ent_num_sr, ent_num_tg, dim, layer_num, drop_out, edges_sr, edges_tg):
        super(StruGNN, self).__init__()
        # self.feats_sr = nn.Parameter(self.prepare_entity_feats(ent_num_sr, edges_sr), requires_grad=False)
        # self.feats_tg = nn.Parameter(self.prepare_entity_feats(ent_num_tg, edges_tg), requires_grad=False)
        embedding_weight = torch.zeros((ent_num_sr + ent_num_tg, dim), dtype=torch.float)
        nn.init.xavier_uniform_(embedding_weight)
        self.feats_sr = nn.Parameter(embedding_weight[:ent_num_sr], requires_grad=True)
        self.feats_tg = nn.Parameter(embedding_weight[ent_num_sr:], requires_grad=True)
        self.edges_sr = nn.Parameter(edges_sr, requires_grad=False)
        self.edges_tg = nn.Parameter(edges_tg, requires_grad=False)
        assert len(self.feats_sr) == ent_num_sr
        assert len(self.feats_tg) == ent_num_tg
        self.gcn = MultiLayerGCN(self.feats_sr.shape[-1], dim, layer_num, drop_out, featureless=True, residual=False)

    def forward(self, sr_ent_seeds, tg_ent_seeds):
        sr_ent_hid = self.gcn(self.edges_sr, self.feats_sr)
        tg_ent_hid = self.gcn(self.edges_tg, self.feats_tg)
        sr_ent_hid = F.normalize(sr_ent_hid, p=2, dim=-1)
        tg_ent_hid = F.normalize(tg_ent_hid, p=2, dim=-1)
        sr_seed_hid = sr_ent_hid[sr_ent_seeds]
        tg_seed_hid = tg_ent_hid[tg_ent_seeds]
        return sr_seed_hid, tg_seed_hid, sr_ent_hid, tg_ent_hid


class AttConf(object):

    def __init__(self):
        self.train_seeds_ratio = 0.3
        self.dim = 128
        self.drop_out = 0.0
        self.layer_num = 2
        self.epoch_num = 200
        self.nega_sample_freq = 5
        self.nega_sample_num = 25

        self.learning_rate = 0.001
        self.l2_regularization = 1e-2
        self.margin_gamma = 1.0

        self.log_comment = "comment"

        self.structure_channel = False
        self.name_channel = False
        self.attribute_value_channel = False
        self.literal_attribute_channel = False
        self.digit_attribute_channel = False
        self.time_channel = False

        self.load_new_seed_split = False

    def set_load_new_seed_split(self, load_new_seed_split):
        self.load_new_seed_split = load_new_seed_split

    def set_channel(self, channel_name):
        if channel_name == 'Literal':
            self.set_literal_attribute_channel(True)
        elif channel_name == 'Digital':
            self.set_digit_attribute_channel(True)
        elif channel_name == 'Attribute':
            self.set_attribute_value_channel(True)
        elif channel_name == 'Structure':
            self.set_structure_channel(True)
        elif channel_name == 'Name':
            self.set_name_channel(True)
        elif channel_name == 'Time':
            self.set_time_channel(True)
        # elif channel_name == 'Test':
        #     self.set_time_channel(True)
        elif 'Test' in channel_name:
            self.set_time_channel(True)
        else:
            raise Exception()

    def set_epoch_num(self, epoch_num):
        self.epoch_num = epoch_num

    def set_nega_sample_num(self, nega_sample_num):
        self.nega_sample_num = nega_sample_num

    def set_log_comment(self, log_comment):
        self.log_comment = log_comment

    def set_name_channel(self, use_name_channel):
        self.name_channel = use_name_channel

    def set_time_channel(self, use_time_channel):
        self.time_channel = use_time_channel

    def set_digit_attribute_channel(self, use_digit_attribute_channel):
        self.digit_attribute_channel = use_digit_attribute_channel

    def set_literal_attribute_channel(self, use_literal_attribute_channel):
        self.literal_attribute_channel = use_literal_attribute_channel

    def set_attribute_value_channel(self, use_attribute_value_channel):
        self.attribute_value_channel = use_attribute_value_channel

    def set_structure_channel(self, use_structure_channel):
        self.structure_channel = use_structure_channel

    def set_drop_out(self, drop_out):
        self.drop_out = drop_out

    def set_learning_rate(self, learning_rate):
        self.learning_rate = learning_rate

    def set_l2_regularization(self, l2_regularization):
        self.l2_regularization = l2_regularization

    def print_parameter(self, file=None):
        parameters = self.__dict__
        print_time_info('Parameter setttings:', dash_top=True, file=file)
        for key, value in parameters.items():
            if type(value) in {int, float, str, bool}:
                print('\t%s:' % key, value, file=file)
        print('---------------------------------------', file=file)

    def init_log(self, log_dir):
        log_dir = Path(log_dir)
        self.log_dir = log_dir
        if log_dir.exists():
            rmtree(str(log_dir), ignore_errors=True)
            # print_time_info("Warning! Forced remove directory %s." % (str(log_dir)))
        log_dir.mkdir()
        comment = log_dir.name
        with open(log_dir / 'parameters.txt', 'w') as f:
            print_time_info(comment, file=f)
            self.print_parameter(f)

    def init(self, directory, device, settings):
        set_random_seed()
        self.directory = Path(directory)
        self.loaded_data = LoadDataDev(self.train_seeds_ratio, self.directory, self.nega_sample_num,
                                       name_channel=self.name_channel, attribute_channel=self.attribute_value_channel,
                                       digit_literal_channel=self.digit_attribute_channel or self.literal_attribute_channel,
                                       dev_channel=self.time_channel,
                                       load_new_seed_split=self.load_new_seed_split, device=device)
        self.sr_ent_num = self.loaded_data.sr_ent_num
        self.tg_ent_num = self.loaded_data.tg_ent_num

        # print('ent_num:', self.sr_ent_num, self.tg_ent_num)
        # print('rel-facts:', len(self.loaded_data.triples_sr), len(self.loaded_data.triples_tg))
        # print('char-facts:', len(self.loaded_data.literal_triples_sr), len(self.loaded_data.literal_triples_tg))
        # print('dig-facts:', len(self.loaded_data.digital_triples_sr), len(self.loaded_data.digital_triples_tg))
        # print('time-facts:', len(self.loaded_data.time_triples_sr), len(self.loaded_data.time_triples_tg))

        self.att_num = self.loaded_data.att_num

        # Init graph adjacent matrix
        print_time_info('Begin preprocessing adjacent matrix')
        self.channels = {}

        double_edges_sr = [(x[1], x[0], x[2]) for x in self.loaded_data.triples_sr] + self.loaded_data.triples_sr
        double_edges_tg = [(x[1], x[0], x[2]) for x in self.loaded_data.triples_tg] + self.loaded_data.triples_tg
        edges_sr = torch.tensor(double_edges_sr)[:, :2]
        edges_tg = torch.tensor(double_edges_tg)[:, :2]
        # edges_sr = torch.tensor(self.loaded_data.triples_sr)[:, :2]
        # edges_tg = torch.tensor(self.loaded_data.triples_tg)[:, :2]
        edges_sr = torch.unique(edges_sr, dim=0)
        edges_tg = torch.unique(edges_tg, dim=0)

        if self.name_channel:
            self.channels['name'] = {'edges_sr': edges_sr, 'edges_tg': edges_tg,
                                     'sr_ent_embed': self.loaded_data.sr_embed,
                                     'tg_ent_embed': self.loaded_data.tg_embed, }
        if self.structure_channel:
            self.channels['structure'] = {'edges_sr': edges_sr, 'edges_tg': edges_tg}
        if self.attribute_value_channel:
            self.channels['attribute'] = {'edges_sr': edges_sr, 'edges_tg': edges_tg,
                                          'att_num': self.loaded_data.att_num,
                                          'attribute_triples_sr': self.loaded_data.attribute_triples_sr,
                                          'attribute_triples_tg': self.loaded_data.attribute_triples_tg,
                                          'value_embedding': self.loaded_data.value_embedding}
        if self.literal_attribute_channel:
            self.channels['attribute'] = {'edges_sr': edges_sr, 'edges_tg': edges_tg,
                                          'att_num': self.loaded_data.literal_att_num,
                                          'attribute_triples_sr': self.loaded_data.literal_triples_sr,
                                          'attribute_triples_tg': self.loaded_data.literal_triples_tg,
                                          'value_embedding': self.loaded_data.literal_value_embedding}
        if self.digit_attribute_channel:
            self.channels['attribute'] = {'edges_sr': edges_sr, 'edges_tg': edges_tg,
                                          'att_num': self.loaded_data.digit_att_num,
                                          'attribute_triples_sr': self.loaded_data.digital_triples_sr,
                                          'attribute_triples_tg': self.loaded_data.digital_triples_tg,
                                          'value_embedding': self.loaded_data.digit_value_embedding}
        if self.time_channel:
            clusters = None
            if settings.use_cluster:
                cluster_model = ClusterModel(settings, self.directory,
                                             self.loaded_data.time_triples_sr, self.loaded_data.time_triples_tg,
                                             self.loaded_data.time_att2id)
                clusters = cluster_model.get_clusters(settings.eps, settings.min_sample)

            self.channels['time'] = {'edges_sr': edges_sr, 'edges_tg': edges_tg,
                                     'att_num': self.loaded_data.time_att_num,
                                     'attribute_triples_sr': self.loaded_data.time_triples_sr,
                                     'attribute_triples_tg': self.loaded_data.time_triples_tg,
                                     'value_embedding': self.loaded_data.time_value_embedding,
                                     'clusters': clusters,
                                     'time_diff_sr': self.loaded_data.time_diff_sr,
                                     'time_diff_tg': self.loaded_data.time_diff_tg,
                                     'time_contexts_sr': self.loaded_data.time_contexts_sr,
                                     'time_contexts_tg': self.loaded_data.time_contexts_tg}

        print_time_info('Finished preprocesssing adjacent matrix')

    def train_on_batch(self, device, settings):
        set_random_seed()
        self.loaded_data.negative_sample()

        # Compose Graph NN
        gnn_channel = GNNChannel(self.sr_ent_num, self.tg_ent_num, self.dim, self.layer_num, self.drop_out,
                                 self.channels)
        self.gnn_channel = gnn_channel
        gnn_channel.to(device)
        gnn_channel.train()

        # Prepare optimizer
        optimizer = Adagrad(filter(lambda p: p.requires_grad, gnn_channel.parameters()), lr=self.learning_rate,
                            weight_decay=self.l2_regularization)
        criterion = AlignLoss(self.margin_gamma)

        best_hit_at_1 = 0
        best_epoch_num = 0

        settings = DevSettings()
        for epoch_num in range(1, self.epoch_num + 1):
            gnn_channel.train()

            batches_sr = torch.split(self.loaded_data.train_sr_ent_seeds, settings.batch_size, dim=0)
            batches_tg = torch.split(self.loaded_data.train_tg_ent_seeds, settings.batch_size, dim=0)
            assert len(batches_sr) == len(batches_tg)

            # hit_at_1 = self.evaluate(epoch_num, gnn_channel, print_info=False, device=device)
            # self.negative_sample()

            epoch_loss = 0.0
            for i in range(len(batches_sr)):
                optimizer.zero_grad()
                pos_batch_sr, neg_batch_sr = batches_sr[i].permute(1, 0)
                pos_batch_tg, neg_batch_tg = batches_tg[i].permute(1, 0)
                pos_sr_seed_hid, pos_tg_seed_hid, _, _ = gnn_channel.forward(pos_batch_sr,
                                                                             pos_batch_tg,
                                                                             return_all_ent=False)
                neg_sr_seed_hid, neg_tg_seed_hid, _, _ = gnn_channel.forward(neg_batch_sr,
                                                                             neg_batch_tg,
                                                                             return_all_ent=False)
                sr_seed_hid = torch.stack([pos_sr_seed_hid, neg_sr_seed_hid], dim=1)
                tg_seed_hid = torch.stack([pos_tg_seed_hid, neg_tg_seed_hid], dim=1)
                loss = criterion(sr_seed_hid, tg_seed_hid)
                epoch_loss += loss
                loss.backward()
                optimizer.step()

            if settings.print_loss:
                if epoch_num % settings.print_loss_freq == 0:
                    print('epoch', epoch_num, ' has loss', epoch_loss)
            if epoch_num % self.nega_sample_freq == 0:
                if str(self.directory).find('DWY100k') >= 0:
                    self.loaded_data.negative_sample()
                else:
                    self.negative_sample()
                hit_at_1 = self.evaluate(epoch_num, gnn_channel, print_info=False, device=device)
                print('model hit_at_1', hit_at_1, 'at epoch', epoch_num)
                if hit_at_1 > best_hit_at_1:
                    best_hit_at_1 = hit_at_1
                    best_epoch_num = epoch_num
        print('Model best Hit@1 on valid set is %.2f at %d epoch.' % (best_hit_at_1, best_epoch_num))
        return best_hit_at_1, best_epoch_num

    def train(self, device, settings):
        set_random_seed()
        self.loaded_data.negative_sample()
        # Compose Graph NN
        gnn_channel = GNNChannel(self.sr_ent_num, self.tg_ent_num, self.dim, self.layer_num, self.drop_out,
                                 self.channels, settings=settings)
        self.gnn_channel = gnn_channel
        gnn_channel.to(device)
        gnn_channel.train()

        # Prepare optimizer
        optimizer = Adagrad(filter(lambda p: p.requires_grad, gnn_channel.parameters()), lr=self.learning_rate,
                            weight_decay=self.l2_regularization)
        criterion = AlignLoss(self.margin_gamma)
        aux_loss_fn = AuxPathLoss()
        best_hit_at_1 = 0
        best_epoch_num = 0

        for epoch_num in range(1, self.epoch_num + 1):
            gnn_channel.train()
            optimizer.zero_grad()
            sr_seed_hid, tg_seed_hid, _, _ = gnn_channel.forward(
                self.loaded_data.train_sr_ent_seeds,
                self.loaded_data.train_tg_ent_seeds)

            loss = criterion(sr_seed_hid, tg_seed_hid)
            if settings.use_auxiliary_path:
                aux_info = None
                aux_loss = aux_loss_fn(aux_info)
                loss += aux_loss
            if settings.print_loss:
                if epoch_num % settings.print_loss_freq == 0:
                    print('epoch', epoch_num, ' has loss', loss)
            loss.backward()
            optimizer.step()
            if epoch_num % self.nega_sample_freq == 0:
                if str(self.directory).find('DWY100k') >= 0:
                    self.loaded_data.negative_sample()
                else:
                    self.negative_sample()
                hit_at_1 = self.evaluate(epoch_num, gnn_channel, print_info=False, device=device)
                if hit_at_1 > best_hit_at_1:
                    best_hit_at_1 = hit_at_1
                    best_epoch_num = epoch_num
        print('Model best Hit@1 on valid set is %.2f at %d epoch.' % (best_hit_at_1, best_epoch_num))
        return best_hit_at_1, best_epoch_num

    def evaluate(self, epoch_num, info_gnn, print_info=True, device='cpu'):
        info_gnn.eval()
        sim = info_gnn.predict(self.loaded_data.valid_sr_ent_seeds, self.loaded_data.valid_tg_ent_seeds)
        top_lr, top_rl, mr_lr, mr_rl, mrr_lr, mrr_rl = get_hits(sim, print_info=print_info, device=device)
        hit_at_1 = (top_lr[0] + top_rl[0]) / 2
        return hit_at_1

    def negative_sample(self, ):
        sim_sr, sim_tg = self.gnn_channel.negative_sample(self.loaded_data.train_sr_ent_seeds_ori,
                                                          self.loaded_data.train_tg_ent_seeds_ori)
        sr_nns = get_nearest_neighbor(sim_sr, self.nega_sample_num)
        tg_nns = get_nearest_neighbor(sim_tg, self.nega_sample_num)
        self.loaded_data.update_negative_sample(sr_nns, tg_nns)

    def save_sim_matrix(self, device):
        # Get the similarity matrix of the current model
        self.gnn_channel.eval()
        sim_train = self.gnn_channel.predict(self.loaded_data.train_sr_ent_seeds_ori,
                                             self.loaded_data.train_tg_ent_seeds_ori)
        sim_valid = self.gnn_channel.predict(self.loaded_data.valid_sr_ent_seeds,
                                             self.loaded_data.valid_tg_ent_seeds)
        sim_test = self.gnn_channel.predict(self.loaded_data.test_sr_ent_seeds, self.loaded_data.test_tg_ent_seeds)
        print_time_info('Best result on the valid set', dash_top=True)
        get_hits(sim_valid, print_info=True, device=device)
        print_time_info('Best result on the test set', dash_top=True)
        get_hits(sim_test, print_info=True, device=device)
        sim_train = sim_train.cpu().numpy()
        sim_valid = sim_valid.cpu().numpy()
        sim_test = sim_test.cpu().numpy()

        def save_sim(sim, comment):
            if sim.shape[0] > 20000:
                partial_sim = sort_and_keep_indices(sim, device)
                partial_sim_t = sort_and_keep_indices(sim.T, device)
                np.save(str(self.log_dir / ('%s_sim.npy' % comment)), partial_sim)
                np.save(str(self.log_dir / ('%s_sim_t.npy' % comment)), partial_sim_t)
            else:
                np.save(str(self.log_dir / ('%s_sim.npy' % comment)), sim)

        save_sim(sim_train, 'train')
        save_sim(sim_valid, 'valid')
        save_sim(sim_test, 'test')
        print_time_info("Model configs and predictions saved to directory: %s." % str(self.log_dir))

    def save_model(self):
        save_path = self.log_dir / 'model.pt'
        state_dict = self.gnn_channel.state_dict()
        state_dict = OrderedDict(filter(lambda x: x[1].layout != torch.sparse_coo, state_dict.items()))
        torch.save(state_dict, str(save_path))
        print_time_info("Model is saved to directory: %s." % str(self.log_dir))


def grid_search(log_comment, data_set, layer_num, device, load_new_seed_split=False, save_model=False,
                use_cluster=False, eps=0.1, samp=3, encoder_method='both', cluster_method='both',
                l2_regularization_range=[0], learning_rate_range=(2e-3, 4e-3)):
    # learning_rate_range=(1e-3, 4e-3, 7e-3, 5e-4)
    # l2_regularization_range=(0, 1e-4, 1e-3)

    settings = DevSettings()
    settings.use_cluster = use_cluster
    settings.eps = eps
    settings.samp = samp
    settings.encoder_method = encoder_method
    settings.cluster_method = cluster_method
    if settings.encoder_method == 'undirected':
        settings.attr_edge_direction = 'bi-direction'
        settings.edge_directions = 'bi-direction'
    print('channel:', log_comment)
    print('encoder_method:', encoder_method)
    print('cluster_method:', cluster_method)
    print('use_cluster:', use_cluster)
    print('cluster eps:', eps)
    print('cluster sampple num:', samp)
    # testing parameters
    if settings.grid_search:
        l2_regularization_range = [0, 1e-4, 1e-3]
        learning_rate_range = [1e-3, 4e-3, 7e-3]
        if settings.is_dev:
            h_method_range = ['None', 'add', 'cat']
            attr_edge_directions = ['ordered', 'bi-direction']
            edge_directions = ['j->e', 'e->j', 'bi-direction']
            h_norm_opts = [True, False]
            use_residual_ops = [False, True]
        else:
            # h_method_range = ['add']
            # attr_edge_directions = ['ordered']
            # edge_directions = ['j->e']
            # h_norm_opts = [False]
            # use_residual_ops = [False]
            h_method_range = [settings.h_method]
            attr_edge_directions = [settings.attr_edge_direction]
            edge_directions = [settings.edge_direction]
            h_norm_opts = [settings.h_norm]
            use_residual_ops = [settings.residual]
    else:
        l2_regularization_range = [settings.l2_regularization]
        learning_rate_range = [settings.lr]
        h_method_range = [settings.h_method]
        attr_edge_directions = [settings.attr_edge_direction]
        edge_directions = [settings.edge_direction]
        h_norm_opts = [settings.h_norm]
        use_residual_ops = [settings.residual]

    parameters = []
    for h_method in h_method_range:
        for h_norm in h_norm_opts:
            for attr_edge_direction in attr_edge_directions:
                for edge_direction in edge_directions:
                    for l2 in l2_regularization_range:
                        for learning_rate in learning_rate_range:
                            for residual in use_residual_ops:
                                parameter = h_method, h_norm, attr_edge_direction, edge_direction, learning_rate, l2, residual
                                parameters.append(parameter)

    att_conf = AttConf()
    att_conf.set_channel(log_comment)
    att_conf.set_epoch_num(settings.max_epoch)
    att_conf.set_nega_sample_num(25)
    att_conf.layer_num = layer_num
    att_conf.set_log_comment(log_comment)
    att_conf.set_load_new_seed_split(load_new_seed_split)
    att_conf.init('%s' % data_set, device, settings)

    data_set = data_set.split('/')[-1]
    best_hit_1 = 0
    best_epoch_num = 0
    best_parameter = (0, 0)
    if not os.path.exists('./cache_log'):
        os.mkdir('./cache_log')
    if not os.path.exists('./log'):
        os.mkdir('./log')

    if settings.grid_search:
        for parameter in parameters:
            hit_at_1, epoch_num = train_with_parameters(parameter, att_conf, settings, data_set, layer_num)
            if hit_at_1 > best_hit_1:
                best_hit_1 = hit_at_1
                best_epoch_num = epoch_num
                best_parameter = parameter
            print_time_info(
                "Current best hit@1 %.2f at %d epoch with %s" % (
                    best_hit_1, best_epoch_num, str(best_parameter)))
    else:
        best_parameter = parameters[0]
        best_hit_1 = 0.0
        best_epoch_num = settings.max_epoch

    print_time_info("The best hit@1 %.2f at %d epoch with %s" % (best_hit_1, best_epoch_num, str(best_parameter)))
    # parameter = h_method, h_norm, attr_edge_direction, edge_direction, learning_rate, l2,residual
    assert len(best_parameter) == 7
    settings.h_method = best_parameter[0]
    settings.h_norm = best_parameter[1]
    settings.attr_edge_direction = best_parameter[2]
    settings.edge_direction = best_parameter[3]
    att_conf.set_learning_rate(best_parameter[4])
    att_conf.set_l2_regularization(best_parameter[5])
    settings.residual = best_parameter[6]
    # att_conf.set_epoch_num(best_epoch_num)
    if load_new_seed_split:
        if layer_num == 2:
            att_conf.init_log('./log/grid_search_hard_%s_%s' % (att_conf.log_comment, data_set))
        else:
            att_conf.init_log('./log/grid_search_hard_%s_%s_%d' % (att_conf.log_comment, data_set, layer_num))
    else:
        if layer_num == 2:
            att_conf.init_log('./log/grid_search_%s_%s' % (att_conf.log_comment, data_set))
        else:
            att_conf.init_log('./log/grid_search_%s_%s_%d' % (att_conf.log_comment, data_set, layer_num))
    att_conf.train(device, settings)
    att_conf.save_sim_matrix(device)
    if save_model:
        att_conf.save_model()


def train_with_parameters(parameters, att_conf, settings, data_set, layer_num):
    h_method, h_norm, attr_edge_direction, edge_direction, learning_rate, l2, residual = parameters
    settings.h_method = h_method
    settings.h_norm = h_norm
    settings.attr_edge_direction = attr_edge_direction
    settings.edge_direction = edge_direction
    att_conf.set_l2_regularization(l2)
    att_conf.set_learning_rate(learning_rate)
    settings.residual = residual
    print('current testing on h_method:', h_method, 'h_norm:', h_norm, 'attr_edge_direction:', attr_edge_direction,
          'edge_direction:', edge_direction, 'residual:', residual, 'l2:', l2, 'learning_rate:', learning_rate)
    if layer_num == 2:
        att_conf.init_log(
            './cache_log/%s_%s_%s_%s' % (
                att_conf.log_comment, data_set, str(l2), str(learning_rate)))
    else:
        att_conf.init_log('./cache_log/%s_%s_%s_%s_%d' % (
            att_conf.log_comment, data_set, str(l2), str(learning_rate), layer_num))

    print('current regularization is', str(l2), 'learning_rate is', str(learning_rate))
    if settings.train_on_batch:
        hit_at_1, epoch_num = att_conf.train_on_batch(device, settings)
    else:
        hit_at_1, epoch_num = att_conf.train(device, settings)

    return hit_at_1, epoch_num


if __name__ == '__main__':
    '''
    python train_subgraph.py --dataset DBP15k/zh_en --channel Literal --gpu_id 5 --load_hard_split
    '''

    import argparse

    default_datadir = 'DBP15k'  # DBP15k,DWY100k
    default_dataset = 'ja_en'  # yg_dbp, wd_dbp
    # default_dataset = './bin/DWY100k/yg_dbp'  # yg_dbp
    parser = argparse.ArgumentParser()
    parser.add_argument('--channel', type=str, default='Time')  # all,Time,Literal,Digital,Structure,Name
    parser.add_argument('--gpu_id', type=int, default=-1)
    parser.add_argument('--dataset', type=str, default=default_dataset)
    parser.add_argument('--datadir', type=str, default=default_datadir)
    parser.add_argument('--load_hard_split', default=True)
    parser.add_argument('--hard', type=int, default=1)  # 0-false; 1-true
    parser.add_argument('--use_cluster', type=int, default=1)
    parser.add_argument('--layer_num', type=int, default=2)
    parser.add_argument('--eps', type=float, default=0.1)
    parser.add_argument('--sample', type=int, default=5)
    parser.add_argument('--encoder', type=str, default='both')
    parser.add_argument('--cluster', type=str, default='both')

    args = parser.parse_args()

    args.dataset = './bin/' + args.datadir + '/' + args.dataset
    if args.use_cluster == 1:
        use_cluster = True
    else:
        use_cluster = False
    if args.hard == 0:
        args.load_hard_split = False
    if args.eps == 1:
        args.eps = 0.1
    elif args.eps == 8:
        args.eps = 0.08
    elif args.eps == 5:
        args.eps = 0.05

    import os

    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    device = 'cuda:%d' % args.gpu_id if args.gpu_id >= 0 else 'cpu'
    # device = 'cuda:0'
    if args.channel == 'all':
        grid_search('Literal', args.dataset, args.layer_num, device, load_new_seed_split=args.load_hard_split,
                    save_model=False, )  # learning_rate_range=[0.004,], l2_regularization_range=[0.0001])
        grid_search('Digital', args.dataset, args.layer_num, device, load_new_seed_split=args.load_hard_split,
                    save_model=False, )  # learning_rate_range=[0.004,], l2_regularization_range=[0.0001])
        grid_search('Time', args.dataset, args.layer_num, device, load_new_seed_split=args.load_hard_split,
                    save_model=False, )  # learning_rate_range=[0.004,], l2_regularization_range=[0.0001])
        grid_search('Structure', args.dataset, args.layer_num, device, load_new_seed_split=args.load_hard_split,
                    save_model=False, )  # learning_rate_range=[0.004,], l2_regularization_range=[0.0001])
        grid_search('Name', args.dataset, args.layer_num, device, load_new_seed_split=args.load_hard_split,
                    save_model=False, )  # learning_rate_range=[0.004,], l2_regularization_range=[0.0001])
    else:
        grid_search(args.channel, args.dataset, args.layer_num, device, load_new_seed_split=args.load_hard_split,
                    save_model=False,
                    use_cluster=use_cluster, eps=args.eps,
                    encoder_method=args.encoder, cluster_method=args.cluster,
                    samp=args.sample)  # learning_rate_range=[0.007,], l2_regularization_range=[0.001])

import torch
import torch.nn as nn
import torch.nn.functional as F
from models import AttSeq, GraphConvolution, MultiLayerGCN
from dev_cluster import ClusterModel
from dev_models import DevTimeEncoder

from torch_geometric.nn import GCNConv, GATConv
import itertools

from torch.nn import Sequential as Seq


class StackGCNs(torch.nn.Module):
    def __init__(self, dim, dropout_rate=0.5):
        super(StackGCNs, self).__init__()
        self.conv1 = GCNConv(dim, dim)
        self.conv2 = GCNConv(dim, dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = self.dropout(h)

        h = self.conv2(h, edge_index)
        return h


class StackGATs(torch.nn.Module):
    def __init__(self, dim, dropout_rate=0.5):
        super(StackGATs, self).__init__()
        self.conv1 = GATConv(dim, dim)
        self.conv2 = GATConv(dim, dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = self.dropout(h)

        h = self.conv2(h, edge_index)
        return h


def _has_overlapping(context1, context2, min_pred=2):
    if len(context1) == 0:
        return False
    if len(context2) == 0:
        return False

    pred1 = [x[2] for x in context1]
    pred2 = [x[2] for x in context2]

    intersection = set(pred1) & set(pred2)
    if len(intersection) >= min_pred:
        return True
    return False


def _build_time_graph(num_ent, contexts):
    edges = []
    ents = range(len(contexts))
    assert len(ents) == num_ent

    pred_dic = {}
    for e_contexts in contexts:
        for context in e_contexts:
            e, v, p = context
            if p not in pred_dic:
                pred_set = set()
            else:
                pred_set = pred_dic[p]
            pred_set.add(e)
            pred_dic[p] = pred_set

    edge_set = set()
    preds = list(pred_dic.keys())
    for i in range(len(preds) - 1):
        pred_i_ents = pred_dic.get(preds[i])
        for j in range(i + 1, len(preds)):
            pred_j_ents = pred_dic.get(preds[j])
            for k in range(len(preds)):
                if k == i or k == j:
                    continue
                pred_k_ents = pred_dic.get(preds[k])
                inter_ents = pred_i_ents & pred_j_ents & pred_k_ents

                edges = set(itertools.permutations(inter_ents, 2))  # ordered a-b; b-a
                for edge in edges:
                    edge_set.add(edge)
            # for e_idx1 in range(len(inter_ents) - 1):
            #     for e_idx2 in range(e_idx1 + 1, len(inter_ents)):
            #         edge1 = (inter_ents[e_idx1], inter_ents[e_idx2])
            #         edge2 = (inter_ents[e_idx2], inter_ents[e_idx1])
            #         edge_set.add(edge1)
            #         edge_set.add(edge2)

    # for i in range(len(ents) - 1):
    #     e1 = ents[i]
    #     e1_context = contexts[e1]
    #     edges.append((e1, e1))
    #     for j in range(i + 1, len(ents)):
    #         e2 = ents[j]
    #         e2_context = contexts[e2]
    #         if _has_overlapping(e1_context, e2_context):
    #             edges.append((e1, e2))
    #             edges.append((e2, e1))
    edge_list = list(edge_set)
    return edge_list


class DevAttSeq(AttSeq):
    def __init__(self, layer_num, sr_ent_num, tg_ent_num, dim, drop_out, att_num, attribute_triples_sr,
                 attribute_triples_tg, value_embedding, edges_sr, edges_tg,
                 settings,
                 clusters=None,
                 time_diff_sr=None, time_diff_tg=None, time_contexts_sr=None, time_contexts_tg=None,
                 residual=True):
        super().__init__(layer_num, sr_ent_num, tg_ent_num, dim, drop_out, att_num, attribute_triples_sr,
                         attribute_triples_tg, value_embedding, edges_sr, edges_tg, residual)

        self.value_encoder = DevTimeEncoder(settings, attribute_triples_sr, attribute_triples_tg, dim,
                                            value_embedding.shape[1], self.edges_sr, self.edges_tg,
                                            clusters,
                                            sr_ent_num, tg_ent_num)
        self.sr_ent_num = sr_ent_num
        self.tg_ent_num = tg_ent_num

        self.settings = settings
        if self.settings.h_e_method == 'GCN':
            self.test_gcns = StackGCNs(dim)
        elif self.settings.h_e_method == 'GAT':
            self.test_gcns = StackGATs(dim)

        if self.settings.use_auxiliary_path:
            self.test_gcns2 = StackGCNs(dim)

        self.mlp = Seq(nn.Linear(2 * dim, dim),
                       nn.Tanh(),
                       nn.Linear(dim, 1))

        self.time_diff_sr = None
        self.time_diff_tg = None
        self.time_contexts_sr = time_contexts_sr
        self.time_contexts_tg = time_contexts_tg
        if self.settings.use_time_sequence:
            self.time_diff_sr = nn.Parameter(time_diff_sr, requires_grad=False)
            self.time_diff_tg = nn.Parameter(time_diff_tg, requires_grad=False)
            # self.context_dic_sr = _build_context_dic(time_diff_sr)
            # self.context_dic_tg = _build_context_dic(time_diff_tg)
        if self.settings.use_time_graph:
            time_edges_sr = _build_time_graph(sr_ent_num, time_contexts_sr)
            time_edges_tg = _build_time_graph(tg_ent_num, time_contexts_tg)
            self.time_edges_sr = nn.Parameter(torch.tensor(time_edges_sr), requires_grad=False)
            self.time_edges_tg = nn.Parameter(torch.tensor(time_edges_tg), requires_grad=False)

    def get_all_entity_feats(self, mode, attribute_triples, time_contexts, ent_feats):
        batch_contexts = torch.split(time_contexts, self.settings.batch_size, dim=0)
        all_feats = []
        for i in range(len(batch_contexts)):
            contexts = batch_contexts[i]
            seed_feats = self.value_encoder(mode, attribute_triples, self.att_feats, self.val_feats,
                                            ent_feats, contexts)
            all_feats.append(seed_feats)
        all_feats = torch.cat(all_feats, dim=0)
        return all_feats

    def forward(self, ent_seed_sr, ent_seed_tg, return_all_ent=False):
        if self.settings.train_on_batch:
            contexts_sr = self.time_diff_sr[ent_seed_sr]  # batch, seq, 3
            contexts_tg = self.time_diff_tg[ent_seed_tg]

            sr_seed_feats = self.value_encoder('sr', self.attribute_triples_sr, self.att_feats, self.val_feats,
                                               self.ent_feats_sr, contexts_sr)
            tg_seed_feats = self.value_encoder('tg', self.attribute_triples_tg, self.att_feats, self.val_feats,
                                               self.ent_feats_tg, contexts_tg)
            if not return_all_ent:
                ent_feats_sr = None
                ent_feats_tg = None
            else:
                ent_feats_sr = self.get_all_entity_feats('sr', self.attribute_triples_sr, self.time_diff_sr,
                                                         self.ent_feats_sr)
                ent_feats_tg = self.get_all_entity_feats('tg', self.attribute_triples_tg, self.time_diff_tg,
                                                         self.ent_feats_tg)

        else:
            ent_feats_sr, h_path_sr = self.value_encoder('sr', self.attribute_triples_sr, self.att_feats,
                                                         self.val_feats,
                                                         self.ent_feats_sr, self.time_diff_sr)
            ent_feats_tg, h_path_tg = self.value_encoder('tg', self.attribute_triples_tg, self.att_feats,
                                                         self.val_feats,
                                                         self.ent_feats_tg, self.time_diff_tg)

            if self.settings.h_e_method == 'None':
                ent_feats_sr = F.normalize(ent_feats_sr, p=2, dim=-1)
                ent_feats_tg = F.normalize(ent_feats_tg, p=2, dim=-1)
                sr_seed_feats = ent_feats_sr[ent_seed_sr]
                tg_seed_feats = ent_feats_tg[ent_seed_tg]

            else:
                ent_feats_sr = self.test_gcns(ent_feats_sr, self.edges_sr.permute(1, 0))
                ent_feats_tg = self.test_gcns(ent_feats_tg, self.edges_tg.permute(1, 0))
                ent_feats_sr = F.normalize(ent_feats_sr, p=2, dim=-1)
                ent_feats_tg = F.normalize(ent_feats_tg, p=2, dim=-1)
                sr_seed_feats = ent_feats_sr[ent_seed_sr]
                tg_seed_feats = ent_feats_tg[ent_seed_tg]

                sr_seed_path, tg_seed_path, aux_info = None, None, None
                if ent_seed_sr.shape[-1] == 2:
                    if self.settings.use_auxiliary_path:
                        sr_true = sr_seed_feats[:, 0, :]
                        sr_nega = sr_seed_feats[:, 1, :]
                        tg_true = tg_seed_feats[:, 0, :]
                        tg_nega = tg_seed_feats[:, 1, :]

                        pos = self.mlp(torch.cat([sr_true, tg_true], dim=1)).squeeze()
                        neg1 = self.mlp(torch.cat([sr_nega, tg_true], dim=1)).squeeze()
                        neg2 = self.mlp(torch.cat([sr_true, tg_nega], dim=1)).squeeze()

                        aux_info = (pos, neg1, neg2)
                    # h_path_sr = self.test_gcns2(h_path_sr, self.edges_sr.permute(1, 0))
                    # h_path_tg = self.test_gcns2(h_path_tg, self.edges_tg.permute(1, 0))
                    # h_path_sr = F.normalize(h_path_sr, p=2, dim=-1)
                    # h_path_tg = F.normalize(h_path_tg, p=2, dim=-1)
                    # sr_seed_path = h_path_sr[ent_seed_sr]
                    # tg_seed_path = h_path_tg[ent_seed_tg]

        return sr_seed_feats, tg_seed_feats, ent_feats_sr, ent_feats_tg

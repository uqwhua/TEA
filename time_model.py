import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv, GATConv
from util import print_time_info


class EntityEncoder(torch.nn.Module):
    def __init__(self, dim, dim_out):
        super(EntityEncoder, self).__init__()
        self.gcnconv1 = GCNConv(dim, dim)
        self.gcnconv2 = GCNConv(dim, dim_out)
        # self.dropout = nn.Dropout()

    def forward(self, x, edges):
        edge_index = edges.t().contiguous()
        h = self.gcnconv1(x, edge_index)
        h = h + x
        h = self.gcnconv2(h, edge_index)
        h = h + x
        return h


class ValueEncoder(torch.nn.Module):
    def __init__(self, key_dim, val_dim):
        super(ValueEncoder, self).__init__()
        dim = key_dim
        self.W = nn.Parameter(torch.zeros(val_dim + key_dim, key_dim))
        nn.init.xavier_uniform_(self.W)
        self.gatconv1 = GATConv(dim, dim, add_self_loops=True)
        self.gatconv1_re = GATConv(dim, dim, add_self_loops=True)

        self.gcnconv1 = GCNConv(dim, dim, add_self_loops=True)
        self.gcnconv2 = GCNConv(dim, dim, add_self_loops=True)

    def forward(self, attribute_triples, ent_edges, ent_edge_labels,
                val_edges, val_edge_labels,
                # edge_attrs_feats,
                att_feats, val_feats,
                ent_feats):
        h, val, att = attribute_triples.transpose(0, 1)
        num_ent = ent_feats.shape[0]
        dim = ent_feats.shape[1]
        num_val = len(val)
        device = ent_feats.device

        # cluster method
        ## nodes = {entities + values}
        ## edges = {(entity, value),(value,value in same cluster)}
        ## edge_labels = {attr, val-val-in-cluster-label}
        val_features = torch.cat((att_feats[att], val_feats[val]), dim=1) @ self.W
        nodes = torch.cat((ent_feats, val_features), dim=0)

        val_edges = val_edges.t().contiguous()
        nodes = self.gcnconv1(nodes, edge_index=val_edges)
        nodes = self.gcnconv2(nodes, edge_index=val_edges)

        ent_edges = ent_edges.t().contiguous()
        nodes = self.gatconv1(nodes, edge_index=ent_edges, edge_attr=ent_edge_labels) \
                + self.gatconv1_re(nodes, ent_edges[torch.tensor([0, 1])], ent_edge_labels)

        ent_vals = nodes[:num_ent] + ent_feats
        return ent_vals


class TimeModel(nn.Module):
    def __init__(self, layer_num, sr_ent_num, tg_ent_num, dim, drop_out, att_num, attribute_triples_sr,
                 attribute_triples_tg, value_embedding, edges_sr, edges_tg, time_graph,
                 residual=True
                 ):
        super(TimeModel, self).__init__()
        self.residual = residual
        ## KG Feature Loading
        self.edges_sr = nn.Parameter(edges_sr, requires_grad=False)
        self.edges_tg = nn.Parameter(edges_tg, requires_grad=False)
        self.attribute_triples_sr = nn.Parameter(attribute_triples_sr, requires_grad=False)  # shape = [E1, 3]
        self.attribute_triples_tg = nn.Parameter(attribute_triples_tg, requires_grad=False)  # shape = [E2, 3]
        self.val_feats = nn.Parameter(torch.from_numpy(value_embedding), requires_grad=False)

        time_graph_sr = time_graph['sr']
        time_graph_tg = time_graph['tg']
        self.time_graph_ent_value_edges_sr = nn.Parameter(time_graph_sr['ent_value_edges'], requires_grad=False)
        self.time_graph_value_value_edges_sr = nn.Parameter(time_graph_sr['value_value_edges'], requires_grad=False)
        self.time_graph_ent_value_edge_labels_sr = nn.Parameter(time_graph_sr['ent_value_edge_labels'],
                                                                requires_grad=False)
        self.time_graph_value_value_edge_labels_sr = nn.Parameter(time_graph_sr['value_value_edge_labels'],
                                                                  requires_grad=False)
        self.edge_attrs_sr = nn.Parameter(time_graph_sr['value_attr_value_labels'],
                                          requires_grad=False)

        self.time_graph_ent_value_edges_tg = nn.Parameter(time_graph_tg['ent_value_edges'], requires_grad=False)
        self.time_graph_value_value_edges_tg = nn.Parameter(time_graph_tg['value_value_edges'], requires_grad=False)
        self.time_graph_ent_value_edge_labels_tg = nn.Parameter(time_graph_tg['ent_value_edge_labels'],
                                                                requires_grad=False)
        self.time_graph_value_value_edge_labels_tg = nn.Parameter(time_graph_tg['value_value_edge_labels'],
                                                                  requires_grad=False)
        self.edge_attrs_tg = nn.Parameter(time_graph_tg['value_attr_value_labels'],
                                          requires_grad=False)

        att_num += 1  # + 1 for unrecognized attribute
        ## Initialize trainable embeddings
        embedding_weight = torch.zeros((att_num + sr_ent_num + tg_ent_num, dim), dtype=torch.float,
                                       requires_grad=False)
        nn.init.xavier_uniform_(embedding_weight)
        self.att_feats = nn.Parameter(embedding_weight[:att_num], requires_grad=True)
        self.ent_feats_sr = nn.Parameter(embedding_weight[att_num: att_num + sr_ent_num],
                                         requires_grad=True)
        self.ent_feats_tg = nn.Parameter(embedding_weight[att_num + sr_ent_num:], requires_grad=True)

        self.W_val_attr_edge = nn.Parameter(torch.zeros(2 * dim, dim))
        nn.init.xavier_uniform_(self.W_val_attr_edge)

        ## initialize networks
        self.value_encoder = ValueEncoder(dim, value_embedding.shape[1])
        self.ent_encoder = EntityEncoder(dim, dim)

    def forward(self, ent_seed_sr, ent_seed_tg):
        ent_feats_sr = self.value_encoder(self.attribute_triples_sr,
                                          self.time_graph_ent_value_edges_sr,
                                          self.time_graph_ent_value_edge_labels_sr,
                                          self.time_graph_value_value_edges_sr,
                                          self.time_graph_value_value_edge_labels_sr,
                                          # edge_attrs_feature_sr,
                                          self.att_feats,
                                          self.val_feats, self.ent_feats_sr)
        ent_feats_tg = self.value_encoder(self.attribute_triples_tg,
                                          self.time_graph_ent_value_edges_tg,
                                          self.time_graph_ent_value_edge_labels_tg,
                                          self.time_graph_value_value_edges_tg,
                                          self.time_graph_value_value_edge_labels_tg,
                                          # edge_attrs_feature_tg,
                                          self.att_feats,
                                          self.val_feats, self.ent_feats_tg)

        ent_feats_sr = self.ent_encoder(ent_feats_sr, self.edges_sr)
        ent_feats_tg = self.ent_encoder(ent_feats_tg, self.edges_tg)

        ent_feats_sr = F.normalize(ent_feats_sr, p=2, dim=-1)
        ent_feats_tg = F.normalize(ent_feats_tg, p=2, dim=-1)
        sr_seed_feats = ent_feats_sr[ent_seed_sr]
        tg_seed_feats = ent_feats_tg[ent_seed_tg]
        return sr_seed_feats, tg_seed_feats, ent_feats_sr, ent_feats_tg

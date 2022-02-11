import torch
import torch.nn as nn
import torch.nn.functional as F
from models import AttSeq, GraphConvolution, MultiLayerGCN
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.nn import GCNConv, EdgeConv, GATConv, SAGEConv, GatedGraphConv, ClusterGCNConv

from torch.nn import Sequential as Seq, Linear, ReLU
from torch_geometric.nn import MessagePassing
from units.rum_model import RUM, rotation_operator


def _build_in_cluster_dic(clusters):
    in_cluster_dic = {}
    for cluster in clusters:
        for i in range(len(cluster) - 1):
            c_i = cluster[i]
            key = tuple([c_i, c_i])
            in_cluster_dic[key] = 1
            for j in range(i + 1, len(cluster)):
                c_j = cluster[j]
                key = tuple([c_i, c_j])
                in_cluster_dic[key] = 1
    return in_cluster_dic


def build_ent_attr_edges(settings, num_ent, attribute_triples):
    ent_attr_edges = []  # edges between attrs (use entry idx)
    for i in range(len(attribute_triples)):
        ent, val, pred = attribute_triples[i]
        if settings.edge_direction == 'e->j':
            edge = (ent, i + num_ent)
            ent_attr_edges.append(edge)
        elif settings.edge_direction == 'j->e':
            edge = (i + num_ent, ent)
            ent_attr_edges.append(edge)
        elif settings.edge_direction == 'bi-direction':
            edge1 = (i + num_ent, ent)
            edge2 = (ent, i + num_ent)
            ent_attr_edges.append(edge1)
            ent_attr_edges.append(edge2)

        # ent_attr_edges.append(edge2)

    return ent_attr_edges


def build_attr_edge_lists(settings, ent_num, attribute_triples, list_size, clusters=None):
    attr_edge_lists = []  # edges between attrs (use entry idx)

    in_cluster_dic = {}
    if settings.use_cluster and clusters is not None:
        in_cluster_dic = _build_in_cluster_dic(clusters)

    for i in range(list_size + 1):
        list_i = []
        attr_edge_lists.append(list_i)

    for i in range(len(attribute_triples) - 1):
        triple_i = attribute_triples[i]
        ent, val, pred = attribute_triples[i]
        if settings.ent_at_every_layer:
            for n in range(list_size):
                edge = (ent, i + ent_num)
                attr_edge_lists[n].append(edge)
        else:
            edge = (ent, i + ent_num)
            attr_edge_lists[0].append(edge)

        num_neighbor = 0
        for j in range(i + 1, len(attribute_triples)):
            triple_j = attribute_triples[j]  # entity, value, predicate
            if triple_i[0] == triple_j[0]:
                if settings.use_cluster:
                    if (triple_i[-1], triple_j[-1]) not in in_cluster_dic:
                        continue

                if settings.attr_edge_direction == 'ordered':
                    if triple_i[1] > triple_j[1]:
                        edge = (i + ent_num, j + ent_num)
                    else:
                        edge = (j + ent_num, i + ent_num)
                    for n in range(list_size):
                        if num_neighbor <= n:
                            attr_edge_lists[n + 1].append(edge)
                else:
                    edge1 = (i + ent_num, j + ent_num)
                    edge2 = (j + ent_num, i + ent_num)
                    for n in range(list_size):
                        if num_neighbor <= n:
                            attr_edge_lists[n + 1].append(edge1)
                            attr_edge_lists[n + 1].append(edge2)

                num_neighbor += 1
                if num_neighbor > list_size:
                    break
            else:
                break

    return attr_edge_lists


def build_attr_edges(settings, attribute_triples, clusters=None):
    attr_edges = []  # edges between attrs (use entry idx)

    in_cluster_dic = {}
    if settings.use_cluster and clusters is not None:
        in_cluster_dic = _build_in_cluster_dic(clusters)

    for i in range(len(attribute_triples) - 1):
        triple_i = attribute_triples[i]
        for j in range(i + 1, len(attribute_triples)):
            triple_j = attribute_triples[j]  # entity, value, predicate
            if triple_i[0] == triple_j[0]:
                if settings.use_cluster:
                    if (triple_i[-1], triple_j[-1]) in in_cluster_dic:
                        attr_edges.append((i, j))
                else:
                    if settings.attr_edge_direction == 'ordered':
                        if triple_i[1] > triple_j[1]:
                            attr_edges.append((i, j))
                        else:
                            attr_edges.append((j, i))
                    else:
                        attr_edges.append((i, j))
                        attr_edges.append((j, i))

            else:
                break

    return attr_edges


class DevConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(DevConv, self).__init__(aggr='max')  # "Max" aggregation.
        self.lin = torch.nn.Linear(in_channels, out_channels)
        self.mlp = Seq(Linear(2 * in_channels, out_channels),
                       ReLU(),
                       Linear(out_channels, out_channels))

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # Step 1: Add self-loops to the adjacency matrix.
        # edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        #
        # # Step 2: Linearly transform node feature matrix.
        # x = self.lin(x)
        #
        # # Step 3: Compute normalization.
        # row, col = edge_index
        # deg = degree(col, x.size(0), dtype=x.dtype)
        # deg_inv_sqrt = deg.pow(-0.5)
        # norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return self.propagate(edge_index, x=x, norm=None)

    def message(self, x_i, x_j, norm):
        # x_i --- entity  [E, in_channels]
        # x_j --- attributes
        hidd = rotation_operator(x_i, x_j)

        tmp = torch.cat([x_i, hidd], dim=1)  # tmp has shape [E, 2 * in_channels]
        hidd = self.mlp(tmp)
        # hidd = norm.view(-1, 1) * hidd
        return hidd


class StackDevConv(torch.nn.Module):
    def __init__(self, dim, dim2, num_layer=2, dropout_rate=0.5):
        super(StackDevConv, self).__init__()
        self.conv1 = DevConv(dim, dim)
        self.conv2 = DevConv(dim, dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.num_layer = num_layer

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)

        if self.num_layer == 2:
            h = self.dropout(h)
            h = self.conv2(h, edge_index)
        return h


import math


def glorot(tensor):
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)


class StackGATs(torch.nn.Module):
    def __init__(self, dim, dim_out, num_layer=2, dropout_rate=0.5, self_loops=True):
        super(StackGATs, self).__init__()
        self.conv1 = GATConv(dim, dim, add_self_loops=self_loops)
        self.conv2 = GATConv(dim, dim_out, add_self_loops=self_loops)
        self.dropout = nn.Dropout(dropout_rate)
        self.num_layer = num_layer
        self.residual = False
        if not self_loops:
            self.residual = True

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        if self.residual:
            h = h + x
        if self.num_layer == 2:
            h = self.dropout(h)
            h = self.conv2(h, edge_index)
            if self.residual:
                h = h + x
        return h


class StackGCNs(torch.nn.Module):
    def __init__(self, dim, dim_out, num_layer=2, dropout_rate=0.5, self_loops=True):
        super(StackGCNs, self).__init__()
        self.conv1 = GCNConv(dim, dim, add_self_loops=self_loops)
        self.conv2 = GCNConv(dim, dim_out, add_self_loops=self_loops)
        self.conv3 = GCNConv(dim, dim_out, add_self_loops=self_loops)
        self.dropout = nn.Dropout(dropout_rate)
        self.num_layer = num_layer
        self.residual = False
        if not self_loops:
            self.residual = True

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        if self.residual:
            h = h + x
        if self.num_layer == 2:
            h = self.dropout(h)
            h = self.conv2(h, edge_index)
            if self.residual:
                h = h + x
        if self.num_layer == 3:
            h = self.dropout(h)
            h = self.conv3(h, edge_index)
            if self.residual:
                h = h + x
        return h


def get_context_encoder(name, num_layer, dim, self_loops=True):
    if name == 'GCN':
        return StackGCNs(dim, dim, num_layer, self_loops=self_loops)
    elif name == 'GAT':
        return StackGATs(dim, dim, num_layer, self_loops=self_loops)
    elif name == 'DEV':
        return StackDevConv(dim, dim, num_layer, self_loops)


class DevTimeEncoder(nn.Module):
    def __init__(self, settings, attribute_triples_sr, attribute_triples_tg, key_dim, val_dim=768,
                 edges_sr=None, edges_tg=None, clusters=None, sr_ent_num=None, tg_ent_num=None):
        super(DevTimeEncoder, self).__init__()
        self.a = nn.Linear(key_dim * 2, 1)
        nn.init.xavier_uniform_(self.a.weight)
        self.W = nn.Parameter(torch.zeros(key_dim + val_dim, key_dim))
        nn.init.xavier_uniform_(self.W)
        self.W_fact = nn.Parameter(torch.zeros(key_dim + key_dim + val_dim, key_dim))
        nn.init.xavier_uniform_(self.W_fact)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)  # For attention scores
        self.settings = settings
        self.dropout = nn.Dropout()
        self.list_size = settings.list_size

        # self.attribute_triples_sr = attribute_triples_sr
        # self.attribute_triples_tg = attribute_triples_tg
        self.sr_ent_num = sr_ent_num
        self.tg_ent_num = tg_ent_num
        self.ent_edges_sr = build_ent_attr_edges(settings, sr_ent_num, attribute_triples_sr.tolist())
        self.ent_edges_tg = build_ent_attr_edges(settings, tg_ent_num, attribute_triples_tg.tolist())
        self.clusters = clusters
        self.attr_edges_sr = build_attr_edges(settings, attribute_triples_sr.tolist(), clusters=self.clusters)
        self.attr_edges_tg = build_attr_edges(settings, attribute_triples_tg.tolist(), clusters=self.clusters)

        if self.settings.use_edge_list:
            self.attr_edges_list_sr = build_attr_edge_lists(settings, sr_ent_num,
                                                            attribute_triples_sr.tolist(), self.list_size,
                                                            clusters=self.clusters)
            self.attr_edges_list_tg = build_attr_edge_lists(settings, tg_ent_num,
                                                            attribute_triples_tg.tolist(), self.list_size,
                                                            clusters=self.clusters)

        # for aux hidd
        # self.rnn_cell = torch.nn.GRU(2 * key_dim, key_dim, 1)
        # self.h_cat_linear = nn.Linear(2 * key_dim, key_dim, bias=False)  # for h
        # nn.init.xavier_uniform_(self.h_cat_linear.weight)
        self.self_loops = True
        if self.settings.residual:
            self.self_loops = False

        if self.settings.h_method == 'cat':
            self.h_cat_linear = nn.Linear(2 * key_dim, key_dim, bias=False)  # for h
            nn.init.xavier_uniform_(self.h_cat_linear.weight)
        elif self.settings.h_method == 'weight':
            self.h_weight = nn.Linear(key_dim, key_dim, bias=False)
            nn.init.xavier_uniform_(self.h_weight.weight)

        if self.settings.context_method == 'GRU':
            # self.rnn_cell = RUM(2 * key_dim, key_dim)
            self.rnn_cell = torch.nn.GRU(key_dim, key_dim, 1)
            self.seq_vec = nn.Parameter(torch.zeros(key_dim, 1))  # for seq attention
            nn.init.xavier_uniform_(self.seq_vec)
        elif self.settings.context_method == 'AttSeq':
            self.att_seq_gcn_layers = MultiLayerGCN(key_dim, key_dim, 2)
        elif self.settings.context_method == 'GCN':
            self.stacked_layers = StackGCNs(key_dim, key_dim, 3, self_loops=self.self_loops)
        elif self.settings.context_method == 'GAT':
            self.stacked_layers = StackGATs(key_dim, key_dim, 1, self_loops=self.self_loops)
        elif self.settings.context_method == 'CNN':
            self.W_val = Linear(val_dim, key_dim)
            nn.init.xavier_uniform_(self.W_val.weight)
            self.conv3d1 = torch.nn.Conv3d(1, 1, kernel_size=[1, 1, 10], stride=2)
            self.conv3d2 = torch.nn.Conv3d(1, 1, kernel_size=[2, 2, 10], stride=2)
            self.maxpool3d = torch.nn.MaxPool3d(kernel_size=[2, 1, 1], stride=2)
            self.batch_norm3d = torch.nn.BatchNorm3d(1)
            self.convfc = Linear(int(0.5 * key_dim), key_dim)
            nn.init.xavier_uniform_(self.convfc.weight)
        elif self.settings.context_method == 'DenseGraph':
            self.encoder_layers = get_context_encoder(self.settings.attr_encoder, self.settings.attr_num_layer,
                                                      key_dim, self.self_loops)
        elif self.settings.context_method == 'AttrGraph':
            self.attr_conv_layers = get_context_encoder(self.settings.attr_encoder, self.settings.attr_num_layer,
                                                        key_dim, self.self_loops)
            self.ent_conv_layers = get_context_encoder(self.settings.ent_attr_encoder, self.settings.ent_num_layer,
                                                       key_dim, self.self_loops)

    def forward(self, mode, attribute_triples, att_feats, val_feats, ent_feats, diff_seq=None):
        trip_num = attribute_triples.shape[0]
        h, val, att = attribute_triples.transpose(0, 1)  # shape=[trip_num]

        to_feats = None
        h_n = 0.0
        args = {'mode': mode, 'ent_feats': ent_feats, 'att_feats': att_feats, 'val_feats': val_feats, 'h': h,
                'val': val,
                'att': att, 'trip_num': trip_num
                }

        if self.settings.context_method == 'GRU':
            h_n = self._GRU_based_representation(diff_seq, ent_feats, att_feats, val_feats, att_choice='share')
        elif self.settings.context_method == 'AttSeq':
            h_n = self._AttSeq_representation(**args)
        elif self.settings.context_method == 'GCN':  # geometric-version GCNs
            h_n = self._GCN_based_representation(**args)
        elif self.settings.context_method == 'GAT':  # geometric-version GCNs
            h_n = self._GCN_based_representation(**args)
        elif self.settings.context_method == 'CNN':
            h_n = self._CNN_representation(diff_seq, ent_feats, att_feats, val_feats)
        elif self.settings.context_method == 'DenseGraph':
            h_n = self._dense_graph_based_representation(mode, diff_seq, ent_feats, att_feats, val_feats, h, val,
                                                         att, trip_num)
        elif self.settings.context_method == 'AttrGraph':
            # if self.settings.use_cluster:
            #     h_n = self._cluster_AttrGraph_representation(mode, attribute_triples, diff_seq,
            #                                                  ent_feats, att_feats, val_feats)
            # else:
            h_n = self._AttrGraph_representation(mode, attribute_triples, diff_seq,
                                                 ent_feats, att_feats, val_feats)

        # h_path = self._GRU_based_representation(diff_seq, ent_feats, att_feats, val_feats, att_choice='entity')
        # h_n = F.normalize(h_n, p=2, dim=-1)
        # h_path = F.normalize(h_path, p=2, dim=-1)
        # h_n = self.h_cat_linear(torch.cat([h_n, h_path], dim=-1))
        if self.settings.h_norm:
            h_n = F.normalize(h_n, p=2, dim=-1)

        if self.settings.h_method == 'None':
            to_feats = h_n
        elif self.settings.h_method == 'add':
            to_feats = ent_feats + h_n
        elif self.settings.h_method == 'cat':
            to_feats = torch.cat([ent_feats, h_n], dim=-1)
            to_feats = self.h_cat_linear(to_feats)
        elif self.settings.h_method == 'weight':
            to_feats = self.h_weight(ent_feats + h_n)
        to_feats = F.elu(to_feats)

        h_path = None
        # if self.settings.use_auxiliary_path:
        #    h_path = self._GRU_based_representation(diff_seq, ent_feats, att_feats, val_feats, att_choice='entity')

        return to_feats, h_path

    def build_edges(self, h, ent_num, trip_num, attr_edges=None):
        device = next(self.parameters()).device
        ent_att_edges = torch.stack((h, torch.arange(ent_num, trip_num + ent_num, device=device)), dim=0)
        ent_att_edges = ent_att_edges.transpose(1, 0)
        if attr_edges is not None:
            attr_edges = torch.tensor(attr_edges, device=device) + ent_num
            edges = torch.cat([ent_att_edges, attr_edges], 0)  # *,2
        else:
            edges = ent_att_edges
        return edges

    def _GRU_based_representation(self, context_seq, ent_feats, att_feats, val_feats, att_choice='none'):
        assert att_choice in ['none', 'entity', 'share']

        e_seq, v_seq, p_seq = context_seq.permute(2, 1, 0)  # shape=[seq_length,batch] for GRU input
        # seq_vals = torch.cat((ent_feats[e_seq], att_feats[p_seq], val_feats[v_seq]),
        #                      dim=-1)  # shape [trip_num, dim1 + dim2]
        # seq_vals = seq_vals @ self.W_fact  # shape = [seq_length, batch, dim]

        seq_vals = torch.cat((att_feats[p_seq], val_feats[v_seq]),
                             dim=-1)
        seq_vals = seq_vals @ self.W
        # seq_vals = torch.cat([ent_feats[e_seq], seq_vals], dim=-1)

        # GRU method
        seq_output, h_n = self.rnn_cell(seq_vals)

        if att_choice == 'entity':
            ent_exp = ent_feats.unsqueeze(0).unsqueeze(-1)  # [1, batch, dim, 1]
            attention_score = torch.matmul(seq_output.unsqueeze(-2), ent_exp).squeeze(-1)
            h_n = torch.sum(attention_score * seq_output, dim=0)

        elif att_choice == 'share':
            attention_score = seq_output @ self.seq_vec  # shape = [seq_length, batch, 1]
            attention_score = F.softmax(attention_score, dim=0)
            h_n = torch.sum(attention_score * seq_output, dim=0)

        elif att_choice == 'none':
            h_n = h_n.squeeze()

        return h_n

    def _AttSeq_representation(self, mode, ent_feats, att_feats, val_feats, h, val, att, trip_num):
        ent_num = ent_feats.shape[0]
        if mode == 'sr':
            edges = self.build_edges(h, ent_num, trip_num, None)
        elif mode == 'tg':
            edges = self.build_edges(h, ent_num, trip_num, None)

        att_vals = torch.cat((att_feats[att], val_feats[val]), dim=1)  # shape [trip_num, dim1 + dim2]
        att_vals = att_vals @ self.W  # shape = [trip_num, dim]
        node_feats = torch.cat([ent_feats, att_vals], dim=0)

        gcn_out = self.att_seq_gcn_layers(edges, node_feats, is_adj=False)
        h_n = gcn_out[:ent_num]
        return h_n

    def _GCN_based_representation(self, mode, ent_feats, att_feats, val_feats, h, val, att, trip_num):
        ent_num = ent_feats.shape[0]
        if mode == 'sr':
            if self.settings.use_dense_graph:
                edges = self.build_edges(h, ent_num, trip_num, self.attr_edges_sr)
            else:
                edges = self.build_edges(h, ent_num, trip_num, None)
        elif mode == 'tg':
            if self.settings.use_dense_graph:
                edges = self.build_edges(h, ent_num, trip_num, self.attr_edges_tg)
            else:
                edges = self.build_edges(h, ent_num, trip_num, None)

        edges = edges.transpose(1, 0)
        assert len(edges) == 2

        att_vals = torch.cat((att_feats[att], val_feats[val]), dim=1)  # shape [trip_num, dim1 + dim2]
        att_vals = att_vals @ self.W  # shape = [trip_num, dim]
        node_feats = torch.cat([ent_feats, att_vals], dim=0)

        gcn_out = self.stacked_layers(node_feats, edges)
        # gcn_out = self.ent_conv_layers(node_feats, edges)
        h_n = gcn_out[:ent_num]
        return h_n

    def _CNN_representation(self, context_seq, ent_feats, att_feats, val_feats):
        e_seq, v_seq, p_seq = context_seq.permute(2, 1, 0)  # shape=[seq_length, batch] for GRU input
        e_seq_emb = ent_feats[e_seq]
        v_seq_emb = self.W_val(val_feats[v_seq])
        p_seq_emb = att_feats[p_seq]
        context_seq_emb = torch.stack([e_seq_emb, v_seq_emb, p_seq_emb], dim=0)  # 3, seq, batch, dim
        # conv3D takes x,y,time
        # conv2D takes x,y
        context_seq_emb = context_seq_emb.permute(2, 3, 0, 1)  # batch, dim, 3, seq
        context_seq_emb = context_seq_emb.unsqueeze(1)  # batch,1, dim, 3, seq

        context_h1 = self.conv3d1(context_seq_emb)  # batch, 1, 64, 2, 1
        context_h2 = self.conv3d2(context_seq_emb)  # batch, 1, 64, 1, 1

        hidd = torch.cat([context_h1, context_h2], dim=3)  # batch, 1, 32, 3, 1
        hidd = self.maxpool3d(hidd)  # batch, 1, 32, 2, 1
        hidd = self.batch_norm3d(hidd)
        batch, _, hid_dim, _, _ = hidd.shape

        hidd = hidd.reshape(batch, -1)
        h_n = self.convfc(hidd)
        return h_n

    def _rel_transition(self, e, v1, p1, v2, p2):
        att_val1 = torch.cat((v1, p1), dim=-1)  # shape [trip_num, dim1 + dim2]
        att_val1 = att_val1 @ self.W  # shape = [trip_num, dim]

        att_val2 = torch.cat((v2, p2), dim=-1)  # shape [trip_num, dim1 + dim2]
        att_val2 = att_val2 @ self.W  # shape = [trip_num, dim]

        h = self.mlp(torch.cat([e, att_val1 - att_val2], dim=-1))
        return h

    def _AttrGraph_representation(self, mode, attribute_triples, context_seq,
                                  ent_feats, att_feats, val_feats):
        ent_num = ent_feats.shape[0]
        trip_num = attribute_triples.shape[0]
        device = next(self.parameters()).device
        h, val, att = attribute_triples.transpose(0, 1)  # shape=[trip_num]

        att_vals = torch.cat((att_feats[att], val_feats[val]), dim=1)  # shape [trip_num, dim1 + dim2]
        node_feats = att_vals @ self.W  # shape = [trip_num, dim]
        node_feats = torch.cat([ent_feats, node_feats], dim=0)

        if self.settings.encoder_method != 'no_attr':
            for i in range(self.list_size):
                if mode == 'sr':
                    if self.settings.encoder_method == 'no_window':
                        attr_edges = self.attr_edges_sr
                    else:
                        attr_edges = self.attr_edges_list_sr[i]
                elif mode == 'tg':
                    if self.settings.encoder_method == 'no_window':
                        attr_edges = self.attr_edges_tg
                    else:
                        attr_edges = self.attr_edges_list_tg[i]

                attr_edges = torch.tensor(attr_edges, device=device)
                attr_edges = attr_edges.transpose(1, 0)
                node_feats = self.attr_conv_layers(node_feats, attr_edges)
                node_feats = self.dropout(node_feats)

        if self.settings.encoder_method != 'no_ent':
            if mode == 'sr':
                ent_edges = self.ent_edges_sr
            elif mode == 'tg':
                ent_edges = self.ent_edges_tg
            ent_edges = torch.tensor(ent_edges, device=device)
            ent_edges = ent_edges.transpose(1, 0)
            node_feats = self.ent_conv_layers(node_feats, ent_edges)

        h_n = node_feats[:ent_num]
        return h_n

    def _dense_graph_based_representation(self, mode, diff_seq,
                                          ent_feats, att_feats, val_feats, h, val, att, trip_num):
        ent_num = ent_feats.shape[0]
        if mode == 'sr':
            edges = self.build_edges(h, ent_num, trip_num, self.attr_edges_sr)
        elif mode == 'tg':
            edges = self.build_edges(h, ent_num, trip_num, self.attr_edges_tg)
        edges = edges.transpose(1, 0)
        assert len(edges) == 2

        att_vals = torch.cat((att_feats[att], val_feats[val]), dim=1)  # shape [trip_num, dim1 + dim2]
        att_vals = att_vals @ self.W  # shape = [trip_num, dim]
        node_feats = torch.cat([ent_feats, att_vals], dim=0)

        gcn_out = self.encoder_layers(node_feats, edges)
        h_n = gcn_out[:ent_num]
        return h_n

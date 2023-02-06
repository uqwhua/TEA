import itertools
import numpy as np
from pathlib import Path

import sklearn.preprocessing as preprocessing
from embeddings import BERT, ValueEmbedding, get_cache_file_path
from sklearn.cluster import DBSCAN
from util import print_time_info


def _create_2d_array(row_num, col_num):
    arr = []
    for i in range(row_num):
        col = []
        for j in range(col_num):
            col.append(0)
        arr.append(col)
    return arr


def _build_entity_dict(attr_triples):
    e_dic = {}
    for e, val, att in attr_triples:
        if e not in e_dic:
            e_dic[e] = [att]
        else:
            e_dic[e].append(att)

    return e_dic


def _build_attr2di_dict(attrs):
    id_dic = {}
    idx = 0
    for attr in attrs:
        id_dic[attr] = idx
        idx += 1
    return id_dic


def _make_co_occurrence_matrix(n, attrs, triples):
    # attrs = set([triple[2] for triple in triples])
    arr = _create_2d_array(n, n)
    e_dic = _build_entity_dict(triples)
    for ent, attrs in e_dic.items():
        # remove duplicate in attrs
        attrs = set(attrs)
        combines = itertools.combinations(attrs, r=2)
        for combine in combines:
            attr1, attr2 = combine
            arr[attr1][attr2] += 1
            arr[attr2][attr1] += 1
    return arr


def _make_attr_count_dict(triples, max_attr):
    count_dic = {}
    for ent, val, attr in triples:
        if attr not in count_dic:
            count_dic[attr] = 1
        else:
            count_dic[attr] += 1

    for i in range(max_attr):
        if i not in count_dic:
            count_dic[i] = 0
    return count_dic


def _template_discovery(attrs, coocur_matrix, attr_count_dict, t0_threshold=0.7, t1_threshold=0.1):
    t0_template = []
    t1_template = []
    unknown = []

    for attr in attrs:
        count_pred = attr_count_dict[attr]
        est__parent_count = max(count_pred, max(coocur_matrix[attr]))
        if est__parent_count > t0_threshold:
            t0_template.append(attr)
        elif est__parent_count < t1_threshold:
            t1_template.append(attr)
        else:
            unknown.append(attr)

    return t0_template, t1_template


def compute_correlation_score(attrs, triples, min_freq=1):
    n = len(attrs) + 1  # for BERT padding
    score_arr = _create_2d_array(n, n)

    count_dic = _make_attr_count_dict(triples, max_attr=n)
    coocur_matrix = _make_co_occurrence_matrix(n, attrs, triples)
    t0_template, t1_template = _template_discovery(attrs, coocur_matrix, count_dic)

    for i in range(n):
        for j in range(n):
            if i == j:
                s_ij = 1.0
                score_arr[i][j] = s_ij
                continue
            count_i_j_occ = coocur_matrix[i][j]
            count_i = count_dic[i]
            count_j = count_dic[j]
            if count_i > min_freq and count_j > min_freq:
                s_ij = count_i_j_occ / min(count_i, count_j)
            else:
                s_ij = 0
            if i in t0_template and j in t1_template:
                s_ij = 0
            elif i in t1_template and j in t0_template:
                s_ij = 0
            score_arr[i][j] = s_ij
    score_arr = np.array(score_arr)
    return score_arr


def get_clusters_with_name(dataset, labels):
    cluster_ids = set(labels)
    if -1 in cluster_ids:
        cluster_ids.remove(-1)
    num_cluster = len(cluster_ids)
    print('num_cluster', num_cluster)

    unclustered = []
    cluster_lists = []
    for i in range(num_cluster):
        cluster_lists.append([])
    for i in range(len(labels)):
        label = labels[i]
        data_item = dataset[i]
        # data_item = data_item.encode('utf-8', 'ignore').decode('utf-8')
        if label != -1:
            cluster_lists[label].append(data_item)
        else:
            unclustered.append(data_item)

    return cluster_lists, unclustered


def get_clusters(dataset, labels):
    cluster_ids = set(labels)
    if -1 in cluster_ids:
        cluster_ids.remove(-1)
    num_cluster = len(cluster_ids)
    print('num_cluster', num_cluster)

    unclustered = []
    cluster_lists = []
    for i in range(num_cluster):
        cluster_lists.append([])

    for i in range(len(labels)):
        label = labels[i]
        if label != -1:
            cluster_lists[label].append(i)
        else:
            unclustered.append(i)

    return cluster_lists, unclustered


def get_embeddings(times, directory, device, name):
    # id2atts = read_mapping(directory / 'id2atts.txt')
    attr_seqs = []
    for time in times:
        attr_seqs.append([time])

    value_embed_encoder = ValueEmbedding(device=device)
    temp_file_dir = directory / 'Cluster'
    value_embed_cache_path, id2value_cache_path = get_cache_file_path(temp_file_dir, name)
    value_embedding, id2value = value_embed_encoder.load_value(attr_seqs, value_embed_cache_path, id2value_cache_path,
                                                               load_from_cache=False)
    return value_embedding, id2value


def generate_random_clusters(attrs):
    clusters = []
    import random
    cluster_size = random.randint(1, len(attrs))

    def rand_cluster(attrs, used_attrs, min_size, max_size):
        size = random.randint(min_size, max_size)
        sample_attrs = set(set(attrs) - used_attrs)
        cluster = random.sample(sample_attrs, size)
        return cluster

    available_attrs = set(attrs)
    used_attrs = set()
    for i in range(cluster_size - 1):
        min_size = 1
        max_size = len(available_attrs) - (cluster_size - i)
        cluster_i = rand_cluster(available_attrs, used_attrs, min_size, max_size)
        used_attrs = used_attrs.union(set(cluster_i))
        available_attrs = available_attrs.difference(set(cluster_i))
        clusters.append(cluster_i)
    rest_attr = list(available_attrs)
    clusters.append(rest_attr)
    return clusters


class ClusterModel:
    def __init__(self, device, directory, sr_time_triples, tg_time_triples, time_att2id ):
        self.device = device
        self.directory = directory
        # self.sr_time_triples = sr_time_triples
        # self.tg_time_triples = tg_time_triples
        self.time_triples = sr_time_triples.tolist() + tg_time_triples.tolist()
        self.time_att2id = time_att2id

    def get_clusters(self, eps=0.1, min_sample=5):
        attrs = set(self.time_att2id.keys())
        attr_ids = set()
        for att, id in self.time_att2id.items():
            attr_ids.add(id)
        correlation_score = compute_correlation_score(attr_ids, self.time_triples,
                                                      min_freq=10)
        print_time_info("get time attribute name embeddings.")
        val_embs, id2val = get_embeddings(self.time_att2id.keys(), self.directory, self.device, name='Cluster')

        att_embs = np.concatenate((correlation_score, val_embs), axis=-1)
        att_embs = preprocessing.normalize(att_embs, norm='l2', axis=0)

        print_time_info("fitting the DBSCAN algorithm with time attribute embeddings.")
        clustering = DBSCAN(eps=eps, min_samples=min_sample, metric='cosine').fit(
            att_embs)  # cosine, euclidean, cityblock
        # print(clustering.labels_)

        # clusters_named, unclustered_named = get_clusters_with_name(id2val, clustering.labels_)
        # print(clusters_named)
        clusters, unclustered = get_clusters(id2val, clustering.labels_)
        clusters.append(unclustered)
        print('cluster size:', len(clusters))

        return clusters


if __name__ == '__main__':
    cluster = ClusterModel()

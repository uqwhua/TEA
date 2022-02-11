import itertools
import numpy as np
from pathlib import Path

from dev_load_data import ValueEmbedding, get_cache_file_path
import sklearn.preprocessing as preprocessing
from sklearn.cluster import DBSCAN


def _get_level(attr, t0_template, t1_template):
    if attr in t0_template:
        return 0
    elif attr in t1_template:
        return 1
    return -1


def _in_same_level(attr1, attr2, t0_template, t1_template):
    l1 = _get_level(attr1, t0_template, t1_template)
    l2 = _get_level(attr2, t0_template, t1_template)
    if l1 == l2:
        return True
    if l1 == -1:
        return True
    if l2 == -1:
        return True
    return False


def _get_distribution(target_att, t0_template, t1_template, count_dic, min_freq=10, min_ratio=0.0):
    distribution = []
    keys = list(count_dic.keys())
    keys.sort()
    for i in range(len(keys)):
        distribution.append(0.0)

    for key in keys:
        if key == target_att:
            distribution[key] = 1.0
            continue
        pair = [key, target_att]
        pair.sort()
        key_count = _count_appear_key(pair, count_dic)
        if key_count < min_freq:
            distribution[key] = 0.0
            continue
        val_count = _count_appear_value(pair, count_dic)
        ratio = key_count / val_count
        if ratio < min_ratio:
            distribution[key] = 0.0
        else:
            if _in_same_level(target_att, key, t0_template, t1_template):
                distribution[key] = ratio
            else:
                distribution[key] = 0.0

    return distribution


def distribution_discovery(time_triples, padding=True):
    count_dic = _build_count_dict(time_triples)
    attrs = list(count_dic.keys())
    if padding:
        attrs.append(len(attrs))
    attrs.sort()
    distribution_list = []
    t0_template, t1_template, _ = _template_discovery(count_dic)
    for attr in attrs:
        distr = _get_distribution(attr, t0_template, t1_template, count_dic)
        if padding:
            distr.append(0.0)
        assert len(distr) == len(attrs)
        distribution_list.append(distr)

    distribution_list = np.array(distribution_list)
    return distribution_list


def _template_discovery(count_dic, min_freq=10, t0_threshold=0.8, t1_threshold=0.1):
    t0_template = []
    t1_template = []
    unknow = []

    attrs = list(count_dic.keys())
    attrs.sort()

    for attr in attrs:
        attr_count = count_dic[attr].get('val', 0)
        max_pred = -1
        max_count = 0

        visited = set()
        combines = count_dic.get(attr, [])
        for comb in combines:
            if comb == 'val':
                continue
            visited.add(comb)
            count = count_dic[attr][comb].get('val', 0)
            if max_pred == -1:
                max_count = count
                max_pred = comb
            else:
                if max_count < count:
                    max_pred = comb
                    max_count = count
        left = set(attrs) - visited
        left = list(left)
        left.sort()
        for cand in left:
            if cand >= attr:
                break
            cand_pair = count_dic.get(cand, [])
            if attr in cand_pair:
                count = cand_pair.get('val', 0)
                if count > max_count:
                    max_pred = comb
                    max_count = count

        if max_pred != -1:
            est = max(max_count, attr_count)
            if est < min_freq:
                continue

            if (attr_count / est) > t0_threshold:
                t0_template.append(attr)

            est = max(max_count, attr_count)
            if (attr_count / est) < t1_threshold:
                t1_template.append(attr)

            # unknown.append(attr)

    return t0_template, t1_template, unknow


def _count_appear_key(attr_seq, count_dic):
    sub_dic = count_dic
    attr_seq = [x for x in attr_seq]
    attr_seq.sort()
    while len(attr_seq) > 0:
        att = attr_seq.pop(0)
        sub_dic = sub_dic.get(att, {})
    occur_count = sub_dic.get('val', 0)
    return occur_count


def _count_appear_value(attr_seq, count_dic):
    min_count = 0
    for att in attr_seq:
        att_dic = count_dic.get(att, {})
        count = att_dic.get('val', 0)
        if min_count == 0:
            min_count = count
        else:
            if count < min_count:
                min_count = count
    return min_count


def _build_count_dict(attr_triples):
    e_dic = {}
    for e, val, att in attr_triples:
        if e not in e_dic:
            e_dic[e] = [att]
        else:
            e_dic[e].append(att)

    count_dic = {}
    for e, att_list in e_dic.items():
        atts = set(att_list)
        atts = list(atts)
        atts.sort()
        for i in range(1, len(atts)):
            combines = set(itertools.combinations(atts, r=i))
            for combine in combines:
                _update_count_dic(combine, count_dic)

    max_key = max(count_dic.keys())
    for i in range(max_key):
        if i not in count_dic:
            count_dic[i] = {}
            count_dic[i]['val'] = 0
    return count_dic


def _update_count_dic(combine, count_dic):
    att_seq = [x for x in combine]
    att_seq.sort()
    sub_dic = count_dic
    while len(att_seq) > 0:
        att_j = att_seq.pop(0)
        if att_j not in sub_dic:
            sub_dic[att_j] = {}
        sub_dic = sub_dic[att_j]
    if 'val' not in sub_dic:
        sub_dic['val'] = 1
    else:
        sub_dic['val'] += 1


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


def get_embeddings(times, directory, name='Test'):
    # id2atts = read_mapping(directory / 'id2atts.txt')
    attr_seqs = []
    for time in times:
        attr_seqs.append([time])

    value_embed_encoder = ValueEmbedding(device='cpu')
    temp_file_dir = directory / 'cluster'
    value_embed_cache_path, id2value_cache_path = get_cache_file_path(temp_file_dir, name)
    value_embedding, id2value = value_embed_encoder.load_value(attr_seqs, value_embed_cache_path, id2value_cache_path,
                                                               load_from_file=False, ordered_value=False)
    return value_embedding, id2value


class ClusterModel:
    def __init__(self, setting, directory, sr_time_triples, tg_time_triples, time_att2id):
        self.setting = setting
        self.directory = directory
        # self.sr_time_triples = sr_time_triples
        # self.tg_time_triples = tg_time_triples
        self.time_triples = sr_time_triples.tolist() + tg_time_triples.tolist()
        self.time_att2id = time_att2id

    def get_clusters(self, eps=0.3, min_sample=3):
        distribution_list = distribution_discovery(self.time_triples, padding=True)
        val_embs, id2val = get_embeddings(self.time_att2id.keys(), self.directory)

        att_embs = val_embs
        print('cluster_method:', self.setting.cluster_method)
        if self.setting.cluster_method == 'no':
            clusters = [[]]
            for pred in range(len(distribution_list)):
                clusters[0].append(pred)
            print('cluster size:', len(clusters))
            return clusters
        if self.setting.cluster_method == 'both':
            att_embs = np.concatenate((distribution_list, val_embs), axis=-1)
            att_embs = preprocessing.normalize(att_embs, norm='l2', axis=1, copy=True, return_norm=False)
        if self.setting.cluster_method == 'occ':
            att_embs = preprocessing.normalize(distribution_list, norm='l2', axis=1, copy=True, return_norm=False)
        if self.setting.cluster_method == 'semantic':
            att_embs = preprocessing.normalize(val_embs, norm='l2', axis=1, copy=True, return_norm=False)
        if self.setting.cluster_method == 'random':
            att_embs = np.concatenate((distribution_list, val_embs), axis=-1)
            att_embs = np.random.rand(att_embs.shape[0], att_embs.shape[1])
            att_embs = preprocessing.normalize(att_embs, norm='l2', axis=1, copy=True, return_norm=False)

        clustering = DBSCAN(eps=eps, min_samples=min_sample, metric='cosine').fit(
            att_embs)  # cosine, euclidean, cityblock
        # print(clustering.labels_)

        # clusters_named, unclustered_named = get_clusters_with_name(id2val, clustering.labels_)
        clusters, unclustered = get_clusters(id2val, clustering.labels_)
        clusters.append(unclustered)
        print('cluster size:', len(clusters))

        return clusters

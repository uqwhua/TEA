import re
import json
import torch
import random
import numpy as np
from datetime import datetime
from util import print_time_info
from transformers import BertTokenizer, BertModel

from time_graph import build_value_edges_in_cluster, build_entity_and_value_edges
from time_graph import build_dynamic_value_edges_in_cluster, build_dynamic_time_graphs
from cluster import ClusterModel
from embeddings import BERT, ValueEmbedding, get_cache_file_path


def read_mapping(path):
    def _parser(lines):
        for idx, line in enumerate(lines):
            i, name = line.strip().split('\t')
            lines[idx] = (int(i), name)
        return dict(lines)

    return read_file(path, _parser)


def read_triples(path):
    '''
    triple pattern: (head_id, tail_id, relation_id)
    '''
    return read_file(path, lambda lines: [tuple([int(item) for item in line.strip().split('\t')]) for line in lines])


def read_seeds(path):
    return read_file(path, lambda lines: [tuple([int(item) for item in line.strip().split('\t')]) for line in lines])


def read_file(path, parse_func):
    num = -1
    with open(path, 'r', encoding='utf8') as f:
        line = f.readline().strip()
        if line.isdigit():
            num = int(line)
        else:
            f.seek(0)
        lines = f.readlines()

    lines = parse_func(lines)

    if len(lines) != num and num >= 0:
        print_time_info('File: %s has corruptted, data_num: %d/%d.' %
                        (path, num, len(lines)))
        raise ValueError()
    return lines


def _load_language(directory, language):
    triples = read_triples(directory / ('triples_' + language + '.txt'))
    id2entity = read_mapping(directory / ('id2entity_' + language + '.txt'))
    id2relation = read_mapping(directory / ('id2relation_' + language + '.txt'))
    return triples, id2entity, id2relation


def _load_seeds(directory, train_seeds_ratio, load_new_seed_split):
    train_data_path = directory / 'train_entity_seeds.txt'
    valid_data_path = directory / 'valid_entity_seeds.txt'
    test_data_path = directory / 'test_entity_seeds.txt'
    entity_seeds = read_seeds(directory / 'entity_seeds.txt')
    if load_new_seed_split:
        train_data_path = directory / 'hard_split' / 'train_entity_seeds.txt'
        valid_data_path = directory / 'hard_split' / 'valid_entity_seeds.txt'
        test_data_path = directory / 'hard_split' / 'test_entity_seeds.txt'
        print_time_info("Loading adversarially-splitted train/valid/test set from %s." % str(directory / 'hard_split'))
        train_entity_seeds = read_seeds(train_data_path)
        valid_entity_seeds = read_seeds(valid_data_path)
        test_entity_seeds = read_seeds(test_data_path)
    elif train_data_path.exists() and valid_data_path.exists() and test_data_path.exists():
        print_time_info("Loading pre-splitted train/valid/test set from %s." % str(directory))
        train_entity_seeds = read_seeds(train_data_path)
        valid_entity_seeds = read_seeds(valid_data_path)
        test_entity_seeds = read_seeds(test_data_path)
    else:
        test_sr_ids_path = directory / ('test_sr_ids_%d.txt' % int(train_seeds_ratio * 100))
        if not test_sr_ids_path.exists():
            print_time_info("Randomly split train/valid set from %s." % str(directory))
            tmp_entity_seeds = [seed for seed in entity_seeds]
            random.shuffle(tmp_entity_seeds)
            train_entity_seeds = tmp_entity_seeds[:int(len(entity_seeds) * train_seeds_ratio)]
            valid_entity_seeds = tmp_entity_seeds[int(len(entity_seeds) * train_seeds_ratio):]
            test_entity_seeds = valid_entity_seeds
            test_sr_ent_set = set(x[0] for x in test_entity_seeds)
            with open(test_sr_ids_path, 'w', encoding='utf8') as f:
                for idx in test_sr_ent_set:
                    f.write(str(idx) + '\n')
        else:
            print_time_info('Loading previously random splitted data set.')
            with open(test_sr_ids_path, 'r', encoding='utf8') as f:
                test_sr_ent_set = [int(line.strip()) for line in f.readlines()]
                test_sr_ent_set = set(test_sr_ent_set)
            train_entity_seeds = [seed for seed in entity_seeds if seed[0] not in test_sr_ent_set]
            valid_entity_seeds = [seed for seed in entity_seeds if seed[0] in test_sr_ent_set]
            test_entity_seeds = valid_entity_seeds
    return train_entity_seeds, valid_entity_seeds, test_entity_seeds, entity_seeds


def _load_trans(directory, language):
    with open(directory / ('id2trans_%s.txt' % language), 'r') as f:
        lines = [line.strip().split('\t') for line in f.readlines()]
    id2trans = {int(idx): sr_trans for idx, sr_trans, sr_ent in lines}
    return id2trans


def _load_dbpedia_properties(data_path, entity2id, language, filter_alias=False):
    # filter_name: mask all the attribute that is potentially an alias of the entity
    potential_alias_pattern = ['name', 'alias', '名字', '别名']

    if language in {'en', 'zh', 'ja', 'fr'}:
        Prefix.set_language(language)
        remove_prefix = Prefix.remove_prefix
    else:
        remove_prefix = lambda x: x

    with open(data_path, 'r', encoding='utf8') as f:
        lines = [line.strip().split('\t') for line in f.readlines()]
    att_triples = []
    for line in lines:
        try:
            subject, property, value, _ = line
        except ValueError:
            subject, property, value = line
        subject = remove_prefix(subject)

        # filter the alias
        if filter_alias:
            for alias in potential_alias_pattern:
                if property.lower().find(alias) >= 0:
                    value = ''
        try:
            value = value.encode('utf8').decode('unicode_escape')
        except UnicodeDecodeError:
            pass
        ent_id = entity2id[subject]
        att = property
        att_triples.append((ent_id, value, att))
    # For all the triples: (head, tail, relation)
    return att_triples


def _get_train_value_and_attribute(train_ent_ids, att_triples):
    train_value_and_attribute = []
    train_ent_ids = set(train_ent_ids)
    for ent_id, value, att in att_triples:
        if ent_id in train_ent_ids:
            train_value_and_attribute.append((value, att))
    return train_value_and_attribute


class TimeExpression(object):
    def __init__(self):
        self.regex = {'year': re.compile(r'^\d{4}$'), 'date': re.compile(r'^(\d+)-(\d+)-(\d+)$'),
                      'month_day': re.compile(r'^--(\d{2})-(\d{2})$')}
        self.regex_func = {'year': lambda x: (int(x.group(0)), None, None),
                           'date': lambda x: (int(x.group(1)), int(x.group(2)), int(x.group(3))),
                           'month_day': lambda x: (None, int(x.group(1)), int(x.group(2)))}

    def is_time_exp(self, text):
        for regex_name in self.regex:
            is_numeral, result = self.__regex_pattern(text, regex_name)
            if is_numeral:
                return is_numeral, result
        is_numeral, result = self.__month_year_pattern(text)
        if is_numeral:
            return is_numeral, result
        return False, None

    def __regex_pattern(self, text, regex_name):
        regex = self.regex[regex_name]
        result = regex.match(text)
        if result:
            return True, self.regex_func[regex_name](result)
        return False, None

    def __month_year_pattern(self, text):
        try:
            data = datetime.strptime(text, '%B %Y')
            return True, (data.year, data.month, None)
        except ValueError:
            return False, None
        except:
            raise Exception()

    # todo, we can have a more accuarete way to find time expressions


def _split_time_attribute(value_and_attribute_pairs, threshold, att_set):
    time_exp = TimeExpression()
    att_is_time = {}
    for value, att_id in value_and_attribute_pairs:
        # 0 digit, 1 literal
        is_time_exp, reuslt = time_exp.is_time_exp(value)
        if att_id in att_is_time:
            if is_time_exp:
                att_is_time[att_id][0] += 1
            else:
                att_is_time[att_id][1] += 1
        else:
            if is_time_exp:
                att_is_time[att_id] = [1, 0]
            else:
                att_is_time[att_id] = [0, 1]

    time_atts = {att for att, count in att_is_time.items() if count[0] / sum(count) > threshold}
    time_atts = list(time_atts)
    time_atts.sort()
    time_att2id = {digit_att: idx for idx, digit_att in enumerate(time_atts)}
    return time_att2id


def _split_digit_attribute_and_literal_attribute(value_and_attribute_pairs, digit_threshold, att_set):
    numeral = Numeral()
    att_is_number = {}
    for value, att_id in value_and_attribute_pairs:
        # 0 digit, 1 literal
        is_numeral, number = numeral.is_numeral(value)
        if att_id in att_is_number:
            if is_numeral:
                att_is_number[att_id][0] += 1
            else:
                att_is_number[att_id][1] += 1
        else:
            if is_numeral:
                att_is_number[att_id] = [1, 0]
            else:
                att_is_number[att_id] = [0, 1]

    digit_atts = {att for att, count in att_is_number.items() if count[0] / sum(count) > digit_threshold}
    literal_atts = {att for att in att_set if att not in digit_atts}

    digit_atts = list(digit_atts)
    digit_atts.sort()
    literal_atts = list(literal_atts)
    literal_atts.sort()
    digit_att2id = {digit_att: idx for idx, digit_att in enumerate(digit_atts)}
    literal_att2id = {literal_att: idx for idx, literal_att in enumerate(literal_atts)}
    return digit_att2id, literal_att2id


def _split_time_triple(att_triples, time_att2id):
    time_triples = []
    for ent_id, value, att in att_triples:
        if att in time_att2id:
            time_triples.append((ent_id, value, time_att2id[att]))
    return time_triples


def _split_digit_and_literal_triple(att_triples, digit_att2id, literal_att2id):
    digit_triples = []
    literal_triples = []
    digit_num = 0
    literal_num = 0
    for ent_id, value, att in att_triples:
        if att in digit_att2id:
            digit_triples.append((ent_id, value, digit_att2id[att]))
            digit_num += 1
        else:
            literal_triples.append((ent_id, value, literal_att2id[att]))
            literal_num += 1
    return digit_triples, literal_triples


class LoadData(object):
    def __init__(self, train_seeds_ratio, directory, nega_sample_num, name_channel,
                 attribute_channel, digit_literal_channel, time_channel,
                 load_new_seed_split=False,
                 device='cpu'):
        self.device = device
        self.directory = directory
        self.nega_sample_num = nega_sample_num
        self.train_seeds_ratio = train_seeds_ratio
        self.language_sr, self.language_tg = directory.name.split('_')
        self.load_seed_alignment(load_new_seed_split)
        self.load_structure_feature()
        if name_channel:
            self.load_name_feature()
        if attribute_channel or digit_literal_channel or time_channel:
            self.load_attribute_feature(attribute_channel, digit_literal_channel, time_channel)
        self.negative_sample()
        self.to_torch()

    def to_torch(self):
        self.valid_sr_ent_seeds = torch.from_numpy(self.valid_sr_ent_seeds).to(self.device)
        self.valid_tg_ent_seeds = torch.from_numpy(self.valid_tg_ent_seeds).to(self.device)

    def update_negative_sample(self, sr_nega_sample, tg_nega_sample):
        # nega sample shape = (data_len, negative_sample_num)
        assert sr_nega_sample.shape == (len(self.train_sr_ent_seeds_ori), self.nega_sample_num)
        assert tg_nega_sample.shape == (len(self.train_tg_ent_seeds_ori), self.nega_sample_num)

        if not (hasattr(self, "sr_posi_sample") and hasattr(self, "tg_posi_sample")):
            sr_posi_sample = np.tile(self.train_sr_ent_seeds_ori.reshape((-1, 1)), (1, self.nega_sample_num))
            tg_posi_sample = np.tile(self.train_tg_ent_seeds_ori.reshape((-1, 1)), (1, self.nega_sample_num))
            self.sr_posi_sample = torch.from_numpy(sr_posi_sample.reshape((-1, 1))).to(self.device)
            self.tg_posi_sample = torch.from_numpy(tg_posi_sample.reshape((-1, 1))).to(self.device)

        sr_nega_sample = sr_nega_sample.reshape((-1, 1))
        tg_nega_sample = tg_nega_sample.reshape((-1, 1))

        # sr_nega_sample = sr_nega_sample.to(self.device)
        # tg_nega_sample = tg_nega_sample.to(self.device)
        self.train_sr_ent_seeds = torch.cat((self.sr_posi_sample, sr_nega_sample), dim=1)
        self.train_tg_ent_seeds = torch.cat((self.tg_posi_sample, tg_nega_sample), dim=1)

    def negative_sample(self):
        # Randomly negative sample
        sr_nega_sample = negative_sample(self.train_sr_ent_seeds_ori, self.sr_ent_num, self.nega_sample_num)
        tg_nega_sample = negative_sample(self.train_tg_ent_seeds_ori, self.tg_ent_num, self.nega_sample_num)
        sr_nega_sample = torch.from_numpy(sr_nega_sample).to(self.device)
        tg_nega_sample = torch.from_numpy(tg_nega_sample).to(self.device)

        self.update_negative_sample(sr_nega_sample, tg_nega_sample)

    def load_structure_feature(self):
        # Load triples and entity mapping
        id2atts = read_mapping(self.directory / 'id2atts.txt')
        self.att2id = {att: idx for idx, att in id2atts.items()}
        self.att_num = len(self.att2id)
        self.triples_sr, self.id2entity_sr, self.id2relation_sr = _load_language(self.directory, self.language_sr)
        self.triples_tg, self.id2entity_tg, self.id2relation_tg = _load_language(self.directory, self.language_tg)
        self.sr_ent_num = len(self.id2entity_sr)
        self.tg_ent_num = len(self.id2entity_tg)

    def load_name_feature(self):
        # Load translations
        if self.language_sr in {'zh', 'ja', 'fr'}:
            id2trans_sr = _load_trans(self.directory, self.language_sr)
            id2trans_sr = sorted(id2trans_sr.items(), key=lambda x: x[0])
            sr_text = [x[1] for x in id2trans_sr]
        else:
            id2entity_sr = sorted(self.id2entity_sr.items(), key=lambda x: x[0])
            sr_text = [x[1] for x in id2entity_sr]
        id2entity_tg = sorted(self.id2entity_tg.items(), key=lambda x: x[0])
        tg_text = [x[1] for x in id2entity_tg]
        bert = BERT()
        bert.to(self.device)
        self.sr_embed = bert.pooled_encode_batched(sr_text, layer=1)
        self.tg_embed = bert.pooled_encode_batched(tg_text, layer=1)
        del bert

    def load_seed_alignment(self, load_new_seed_split):
        # Load alignment seeds
        train_entity_seeds, valid_entity_seeds, test_entity_seeds, entity_seeds = _load_seeds(self.directory,
                                                                                              self.train_seeds_ratio,
                                                                                              load_new_seed_split)
        self.entity_seeds = entity_seeds  # The entity seeds in the original order

        # train_ent_seeds shape = [length, 2]
        train_sr_ent_seeds_ori, train_tg_ent_seeds_ori = zip(*train_entity_seeds)

        self.train_sr_ent_seeds_ori = np.asarray(train_sr_ent_seeds_ori)
        self.train_tg_ent_seeds_ori = np.asarray(train_tg_ent_seeds_ori)

        # valid_ent_seeds shape = [length]
        valid_sr_ent_seeds, valid_tg_ent_seeds = zip(*valid_entity_seeds)
        self.valid_sr_ent_seeds = np.asarray(valid_sr_ent_seeds)
        self.valid_tg_ent_seeds = np.asarray(valid_tg_ent_seeds)
        test_sr_ent_seeds, test_tg_ent_seeds = zip(*test_entity_seeds)
        self.test_sr_ent_seeds = np.asarray(test_sr_ent_seeds)
        self.test_tg_ent_seeds = np.asarray(test_tg_ent_seeds)

    def load_attribute_feature(self, load_attribute, load_digit_literal, time_channel):
        directory = self.directory
        language_sr = self.language_sr
        language_tg = self.language_tg

        entity2id_sr = {ent: idx for idx, ent in self.id2entity_sr.items()}
        entity2id_tg = {ent: idx for idx, ent in self.id2entity_tg.items()}
        att_triples_sr = _load_dbpedia_properties(directory / ("atts_properties_%s.txt" % language_sr),
                                                  entity2id_sr, language_sr)
        att_triples_tg = _load_dbpedia_properties(directory / ("atts_properties_%s.txt" % language_tg),
                                                  entity2id_tg, language_tg)
        temp_file_dir = directory / 'running_temp'
        value_embed_encoder = ValueEmbedding(self.device)

        if load_attribute:
            self.att_triples_sr = [(ent_id, value, self.att2id[att]) for ent_id, value, att in att_triples_sr]
            self.att_triples_tg = [(ent_id, value, self.att2id[att]) for ent_id, value, att in att_triples_tg]

            ent_id_seq_sr, att_id_seq_sr, value_seq_sr = transform_triple2seq(self.att_triples_sr, language_sr)
            ent_id_seq_tg, att_id_seq_tg, value_seq_tg = transform_triple2seq(self.att_triples_tg, language_tg)
            value_embed_cache_path, id2value_cache_path = get_cache_file_path(temp_file_dir, 'Attribute')
            self.value_embedding, self.id2value = value_embed_encoder.load_value(value_seq_sr + value_seq_tg,
                                                                                 value_embed_cache_path,
                                                                                 id2value_cache_path, )
            value2id = {value: idx for idx, value in enumerate(self.id2value)}
            value_id_seq_sr = [[value2id.get(value, value2id['[PAD]']) for value in value_seq] for value_seq in
                               value_seq_sr]
            value_id_seq_tg = [[value2id.get(value, value2id['[PAD]']) for value in value_seq] for value_seq in
                               value_seq_tg]

            attribute_triples_sr = []
            for ent_id, att_seq, val_seq in zip(ent_id_seq_sr, att_id_seq_sr, value_id_seq_sr):
                for att, val in zip(att_seq, val_seq):
                    attribute_triples_sr.append((ent_id, val, att))
            self.attribute_triples_sr = torch.tensor(attribute_triples_sr)

            attribute_triples_tg = []
            for ent_id, att_seq, val_seq in zip(ent_id_seq_tg, att_id_seq_tg, value_id_seq_tg):
                for att, val in zip(att_seq, val_seq):
                    attribute_triples_tg.append((ent_id, val, att))
            self.attribute_triples_tg = torch.tensor(attribute_triples_tg)

        if time_channel:
            train_value_and_attribute_sr = _get_train_value_and_attribute(self.train_sr_ent_seeds_ori, att_triples_sr)
            train_value_and_attribute_tg = _get_train_value_and_attribute(self.train_tg_ent_seeds_ori, att_triples_tg)

            time_threshold = 0.5
            time_att2id = _split_time_attribute(
                train_value_and_attribute_sr + train_value_and_attribute_tg, time_threshold,
                set(self.att2id.keys()))

            # manually insert a virtual edge label
            time_att2id['Value_Edge'] = len(time_att2id)
            self.VALUE_LABEL_ID = time_att2id['Value_Edge']
            self.time_att2id = time_att2id
            self.time_att_num = len(time_att2id)

            time_triples_sr = list(set(_split_time_triple(att_triples_sr, time_att2id)))
            time_triples_tg = list(set(_split_time_triple(att_triples_tg, time_att2id)))

            print('------------------')
            print_time_info('time attribute number: %d' % (len(time_att2id)))
            print_time_info('time triple number: %d, %d' % (len(time_triples_sr), len(time_triples_tg)))
            print('------------------')

            time_ent_id_seq_sr, time_att_id_seq_sr, time_value_seq_sr = transform_triple2seq(time_triples_sr,
                                                                                             language_sr, False)
            time_ent_id_seq_tg, time_att_id_seq_tg, time_value_seq_tg = transform_triple2seq(time_triples_tg,
                                                                                             language_tg, False)

            time_value_embed_cache_path, time_id2value_cache_path = get_cache_file_path(temp_file_dir, 'Time')
            self.time_value_embedding, self.time_id2value = value_embed_encoder.load_value(
                time_value_seq_sr + time_value_seq_tg, time_value_embed_cache_path,
                time_id2value_cache_path)
            time_value2id = {value: idx for idx, value in enumerate(self.time_id2value)}

            time_value_id_seq_sr = [[time_value2id.get(value, time_value2id['[PAD]']) for value in value_seq] for
                                    value_seq in time_value_seq_sr]
            time_value_id_seq_tg = [[time_value2id.get(value, time_value2id['[PAD]']) for value in value_seq] for
                                    value_seq in time_value_seq_tg]

            time_triples_sr = []
            for ent_id, att_seq, val_seq in zip(time_ent_id_seq_sr, time_att_id_seq_sr, time_value_id_seq_sr):
                for att, val in zip(att_seq, val_seq):
                    time_triples_sr.append((ent_id, val, att))
            time_triples_tg = []
            for ent_id, att_seq, val_seq in zip(time_ent_id_seq_tg, time_att_id_seq_tg, time_value_id_seq_tg):
                for att, val in zip(att_seq, val_seq):
                    time_triples_tg.append((ent_id, val, att))

            # time_triples_sr = list(set(time_triples_sr))
            # time_triples_tg = list(set(time_triples_tg))
            time_triples_sr.sort(key=lambda x: self.time_id2value[x[2]], reverse=False)
            self.time_triples_sr = torch.tensor(time_triples_sr)

            time_triples_tg.sort(key=lambda x: self.time_id2value[x[2]], reverse=False)
            self.time_triples_tg = torch.tensor(time_triples_tg)

            self.clusterModel = ClusterModel(self.device,
                                             self.directory,
                                             self.time_triples_sr,
                                             self.time_triples_tg,
                                             self.time_att2id)
            self.clusters = self.clusterModel.get_clusters()

            ent_value_edge_sr = build_entity_and_value_edges(self.sr_ent_num, time_triples_sr)
            ent_value_edge_tg = build_entity_and_value_edges(self.tg_ent_num, time_triples_tg)
            value_edges_sr, edge_attrs_sr = build_value_edges_in_cluster(self.sr_ent_num,
                                                                         self.time_id2value,
                                                                         time_triples_sr, self.clusters)
            value_edges_tg, edge_attrs_tg = build_value_edges_in_cluster(self.tg_ent_num,
                                                                         self.time_id2value,
                                                                         time_triples_tg, self.clusters)
            self.time_graph_ent_value_edges_sr = torch.tensor(ent_value_edge_sr)
            self.time_graph_ent_value_edges_tg = torch.tensor(ent_value_edge_tg)
            self.time_graph_value_value_edges_sr = torch.tensor(value_edges_sr)
            self.time_graph_value_value_edges_tg = torch.tensor(value_edges_tg)

            dy_value_edges_sr, dy_value_edge_attrs_sr = build_dynamic_time_graphs(self.sr_ent_num,
                                                                                  self.time_id2value,
                                                                                  time_triples_sr,
                                                                                  self.clusters)
            dy_value_edges_tg, dy_value_edge_attrs_tg = build_dynamic_time_graphs(self.tg_ent_num,
                                                                                  self.time_id2value,
                                                                                  time_triples_tg,
                                                                                  self.clusters)
            self.dy_value_edges_sr = [torch.tensor(x) for x in dy_value_edges_sr]
            self.dy_value_edge_attrs_sr = [torch.tensor(x) for x in dy_value_edge_attrs_sr]
            self.dy_value_edges_tg = [torch.tensor(x) for x in dy_value_edges_tg]
            self.dy_value_edge_attrs_tg = [torch.tensor(x) for x in dy_value_edge_attrs_tg]

            def concate_all_seq(time_triples, idx):
                seqs = []
                for triple in time_triples:
                    seqs.append(triple[idx])
                return seqs

            ent_val_ids_sr = concate_all_seq(time_triples_sr, 2)
            ent_val_ids_tg = concate_all_seq(time_triples_tg, 2)
            val_edge_ids_sr = [self.VALUE_LABEL_ID] * len(value_edges_sr)
            val_edge_ids_tg = [self.VALUE_LABEL_ID] * len(value_edges_tg)
            self.time_graph_ent_value_edge_labels_sr = torch.tensor(ent_val_ids_sr)
            self.time_graph_ent_value_edge_labels_tg = torch.tensor(ent_val_ids_tg)
            self.time_graph_value_attr_value_labels_sr = torch.tensor(edge_attrs_sr)
            self.time_graph_value_attr_value_labels_tg = torch.tensor(edge_attrs_tg)
            self.time_graph_value_value_edge_labels_sr = torch.tensor(val_edge_ids_sr)
            self.time_graph_value_value_edge_labels_tg = torch.tensor(val_edge_ids_tg)
            self.time_graph_sr = {
                'ent_value_edges': self.time_graph_ent_value_edges_sr,
                'value_value_edges': self.time_graph_value_value_edges_sr,
                'ent_value_edge_labels': self.time_graph_ent_value_edge_labels_sr,
                'value_attr_value_labels': self.time_graph_value_attr_value_labels_sr,
                'value_value_edge_labels': self.time_graph_value_value_edge_labels_sr,
                'dy_value_edges': self.dy_value_edges_sr,
                'dy_value_edge_attrs': self.dy_value_edge_attrs_sr
            }
            self.time_graph_tg = {
                'ent_value_edges': self.time_graph_ent_value_edges_tg,
                'value_value_edges': self.time_graph_value_value_edges_tg,
                'ent_value_edge_labels': self.time_graph_ent_value_edge_labels_tg,
                'value_attr_value_labels': self.time_graph_value_attr_value_labels_tg,
                'value_value_edge_labels': self.time_graph_value_value_edge_labels_tg,
                'dy_value_edges': self.dy_value_edges_tg,
                'dy_value_edge_attrs': self.dy_value_edge_attrs_tg
            }
            self.time_graph = {
                'sr': self.time_graph_sr,
                'tg': self.time_graph_tg
            }
            print_time_info('finish loading time info.')

        if load_digit_literal:
            train_value_and_attribute_sr = _get_train_value_and_attribute(self.train_sr_ent_seeds_ori, att_triples_sr)
            train_value_and_attribute_tg = _get_train_value_and_attribute(self.train_tg_ent_seeds_ori, att_triples_tg)

            digit_threshold = 0.5
            digit_att2id, literal_att2id = _split_digit_attribute_and_literal_attribute(
                train_value_and_attribute_sr + train_value_and_attribute_tg,
                digit_threshold, set(self.att2id.keys()))

            self.digit_att2id = digit_att2id
            self.literal_att2id = literal_att2id
            self.digit_att_num = len(digit_att2id)
            self.literal_att_num = len(literal_att2id)

            digit_triples_sr, literal_triples_sr = _split_digit_and_literal_triple(att_triples_sr, digit_att2id,
                                                                                   literal_att2id)
            digit_triples_tg, literal_triples_tg = _split_digit_and_literal_triple(att_triples_tg, digit_att2id,
                                                                                   literal_att2id)

            print_time_info('digit attribute number: %d' % (len(self.digit_att2id)))
            print_time_info('digit triple number: %d, %d' % (len(digit_triples_sr), len(digit_triples_tg)))
            print_time_info('literal attribute number: %d' % (len(self.literal_att2id)))
            print_time_info('literal triple number: %d, %d' % (len(digit_triples_tg), len(literal_triples_tg)))
            digit_ent_id_seq_sr, digit_att_id_seq_sr, digit_value_seq_sr = transform_triple2seq(digit_triples_sr,
                                                                                                language_sr, False)
            digit_ent_id_seq_tg, digit_att_id_seq_tg, digit_value_seq_tg = transform_triple2seq(digit_triples_tg,
                                                                                                language_tg, False)

            literal_ent_id_seq_sr, literal_att_id_seq_sr, literal_value_seq_sr = transform_triple2seq(
                literal_triples_sr, language_sr, False)
            literal_ent_id_seq_tg, literal_att_id_seq_tg, literal_value_seq_tg = transform_triple2seq(
                literal_triples_tg, language_tg, False)

            literal_value_embed_cache_path, literal_id2value_cache_path = get_cache_file_path(temp_file_dir,
                                                                                              'Literal')
            digit_value_embed_cache_path, digit_id2value_cache_path = get_cache_file_path(temp_file_dir, 'Digit')

            self.literal_value_embedding, self.literal_id2value = value_embed_encoder.load_value(
                literal_value_seq_sr + literal_value_seq_tg, literal_value_embed_cache_path,
                literal_id2value_cache_path)

            self.digit_value_embedding, self.digit_id2value = value_embed_encoder.load_value(
                digit_value_seq_sr + digit_value_seq_tg, digit_value_embed_cache_path, digit_id2value_cache_path, )

            literal_value2id = {value: idx for idx, value in enumerate(self.literal_id2value)}
            digit_value2id = {value: idx for idx, value in enumerate(self.digit_id2value)}

            digit_value_id_seq_sr = [[digit_value2id.get(value, digit_value2id['[PAD]']) for value in value_seq] for
                                     value_seq in digit_value_seq_sr]
            digit_value_id_seq_tg = [[digit_value2id.get(value, digit_value2id['[PAD]']) for value in value_seq] for
                                     value_seq in digit_value_seq_tg]

            literal_value_id_seq_sr = [
                [literal_value2id.get(value, literal_value2id['[PAD]']) for value in value_seq]
                for value_seq in literal_value_seq_sr]
            literal_value_id_seq_tg = [
                [literal_value2id.get(value, literal_value2id['[PAD]']) for value in value_seq]
                for value_seq in literal_value_seq_tg]

            literal_triples_sr = []
            for ent_id, att_seq, val_seq in zip(literal_ent_id_seq_sr, literal_att_id_seq_sr,
                                                literal_value_id_seq_sr):
                for att, val in zip(att_seq, val_seq):
                    literal_triples_sr.append((ent_id, val, att))
            self.literal_triples_sr = torch.tensor(literal_triples_sr)

            literal_triples_tg = []
            for ent_id, att_seq, val_seq in zip(literal_ent_id_seq_tg, literal_att_id_seq_tg,
                                                literal_value_id_seq_tg):
                for att, val in zip(att_seq, val_seq):
                    literal_triples_tg.append((ent_id, val, att))
            self.literal_triples_tg = torch.tensor(literal_triples_tg)

            digital_triples_sr = []
            for ent_id, att_seq, val_seq in zip(digit_ent_id_seq_sr, digit_att_id_seq_sr, digit_value_id_seq_sr):
                for att, val in zip(att_seq, val_seq):
                    digital_triples_sr.append((ent_id, val, att))
            self.digital_triples_sr = torch.tensor(digital_triples_sr)

            digital_triples_tg = []
            for ent_id, att_seq, val_seq in zip(digit_ent_id_seq_tg, digit_att_id_seq_tg, digit_value_id_seq_tg):
                for att, val in zip(att_seq, val_seq):
                    digital_triples_tg.append((ent_id, val, att))
            self.digital_triples_tg = torch.tensor(digital_triples_tg)

        del value_embed_encoder


def negative_sample(pos_ids, data_range, nega_sample_num):
    # Output shape = (data_len, negative_sample_num)
    nega_ids_arrays = np.random.randint(low=0, high=data_range - 1, size=(len(pos_ids), nega_sample_num))
    for idx, pos_id in enumerate(pos_ids):
        for j in range(nega_sample_num):
            if nega_ids_arrays[idx][j] >= pos_id:
                nega_ids_arrays[idx][j] += 1
    assert nega_ids_arrays.shape == (len(pos_ids), nega_sample_num), print(nega_ids_arrays.shape)
    return nega_ids_arrays


class Prefix(object):
    pattern_en_value_type = ["http://www.w3.org/1999/02/22-rdf-syntax-ns#", "http://www.w3.org/2001/XMLSchema#",
                             "http://dbpedia.org/datatype/"]

    @classmethod
    def set_language(cls, language):
        if language == 'en':
            cls.regex_ent = re.compile(r'http:\/\/dbpedia\.org\/resource\/(.*)')
            cls.pattern_prop = 'http://dbpedia.org/property/'
        elif language in {'zh', 'fr', 'ja'}:
            cls.regex_ent = re.compile(r'http:\/\/%s\.dbpedia\.org\/resource\/(.*)' % language)
            cls.pattern_prop = 'http://%s.dbpedia.org/property/' % language
        else:
            raise Exception()

    @classmethod
    def remove_prefix(cls, input):
        if isinstance(input, str):
            input = cls.regex_ent.match(input).group(1)
            return input.replace('_', ' ')
        return [cls.remove_prefix(item) for item in input]

    @classmethod
    def remove_prop_prefix(cls, input):
        if isinstance(input, str):
            if input.find(cls.pattern_prop) >= 0:
                return input.split(cls.pattern_prop)[1]
            raise Exception()
        return [cls.remove_prop_prefix(item) for item in input]

    @classmethod
    def remove_value_type(cls, input):
        if isinstance(input, str):
            for pattern in cls.pattern_en_value_type:
                if input.find(pattern) >= 0:
                    return input.split(pattern)[1]
            raise Exception()
        return [cls.remove_value_type(item) for item in input]


def transform_triple2seq(att_triples, language, concate_values=False):
    # ent_id_seq = [ent1_id, ent2_id, ent3_id...]
    # prop_num = [ent1_num_prop, ent2_num_prop...]
    # att_id_seq = [[ent1_prop1_id, ent1_prop2_id, ...]...]
    # value_seq = [[ent1_value1, ent1_value2, ...]...]
    # Fixme: select the first 20 attributes
    # Fixme: Original average property number 26 --> only one property average property number 15.9 --> top 20 property 10.09
    if language in {'zh', 'en', 'ja', 'fr'}:
        top_k_att = 20
    else:
        top_k_att = 3
    ent_id_seq = []
    prop2value_seq = []
    for ent_id, value, att_id in att_triples:
        if len(ent_id_seq) == 0:
            ent_id_seq.append(ent_id)
            prop2value_seq.append(dict())
        if ent_id != ent_id_seq[-1]:
            ent_id_seq.append(ent_id)
            prop2value_seq.append(dict())
        if not concate_values:
            prop2value_seq[-1][att_id] = value
        else:
            if att_id in prop2value_seq[-1]:
                prop2value_seq[-1][att_id] += '. ' + value
            else:
                prop2value_seq[-1][att_id] = value
    att_id_seq = []
    value_seq = []
    for prop2value in prop2value_seq:
        att_ids, values = zip(*list(prop2value.items()))
        assert len(values) == len(att_ids)
        att_id_seq.append(att_ids[:top_k_att])
        value_seq.append(values[:top_k_att])
    return ent_id_seq, att_id_seq, value_seq


def construct_ent_id2info(ent_num, ent_id_seq, att_id_seq, value_id_seq, att_pad_id, value_pad_id, language):
    if language in {'zh', 'en', 'fr', 'ja'}:
        top_k_att = 20
    else:
        top_k_att = 3

    assert len(ent_id_seq) == len(att_id_seq) == len(value_id_seq)
    entid2atts = [[] for _ in range(ent_num)]
    entid2values = [[] for _ in range(ent_num)]
    for ent_id, att_ids, value_ids in zip(ent_id_seq, att_id_seq, value_id_seq):
        entid2atts[ent_id] += att_ids
        entid2values[ent_id] += value_ids

    entid2atts = [item[:top_k_att] for item in entid2atts]
    entid2values = [item[:top_k_att] for item in entid2values]

    max_len1 = max(len(item) for item in entid2atts)
    max_len2 = max(len(item) for item in entid2values)
    assert max_len1 == max_len2 == top_k_att
    ent2att_num = np.zeros(ent_num, dtype=np.int)
    ent2atts = np.ones((ent_num, max_len1), dtype=np.int) * att_pad_id
    ent2values = np.ones((ent_num, max_len1), dtype=np.int) * value_pad_id

    att_num = 0
    for idx, (atts, values) in enumerate(zip(entid2atts, entid2values)):
        assert len(atts) == len(values)
        ent2att_num[idx] = len(atts)
        ent2atts[idx, :len(atts)] = atts
        ent2values[idx, :len(atts)] = values
        att_num += len(atts)
    return ent2att_num, ent2atts, ent2values


class Numeral(object):
    def __init__(self):
        self.regex = {'year': re.compile(r'^\d{3,4}$'), 'date': re.compile(r'^(\d+)-(\d+)-(\d+)$'),
                      'month_day': re.compile(r'^--(\d{2})-(\d{2})$')}
        self.regex_func = {'year': lambda x: (int(x.group(0)), None, None),
                           'date': lambda x: (int(x.group(1)), int(x.group(2)), int(x.group(3))),
                           'month_day': lambda x: (None, int(x.group(1)), int(x.group(2)))}

    def is_numeral(self, text):
        # for regex_name in self.regex:
        #     is_numeral, result = self.__regex_pattern(text, regex_name)
        #     if is_numeral:
        #         return is_numeral, result
        # is_numeral, result = self.__month_year_pattern(text)
        # if is_numeral:
        #     return is_numeral, result
        is_numeral, result = self.__float_pattern(text)
        if is_numeral:
            return is_numeral, result
        return False, None

    def __regex_pattern(self, text, regex_name):
        regex = self.regex[regex_name]
        result = regex.match(text)
        if result:
            return True, self.regex_func[regex_name](result)
        return False, None

    def __month_year_pattern(self, text):
        try:
            data = datetime.strptime(text, '%B %Y')
            return True, (data.year, data.month, None)
        except ValueError:
            return False, None
        except:
            raise Exception()

    def __float_pattern(self, text):
        special_patterns = ['inf', 'nan']
        for pattern in special_patterns:
            if text.lower().find(pattern) >= 0:
                return False, None
        try:
            data = float(text)
            return True, data
        except ValueError:
            return False, None
        except:
            raise Exception()

# from transformers import AutoTokenizer, AutoModel

from load_data import *
from load_data import _load_dbpedia_properties, _get_train_value_and_attribute, _load_language, _load_trans
from load_data import _split_digit_attribute_and_literal_attribute, _split_digit_and_literal_triple
from dev_settings import DevSettings


class ValueEmbedding(object):
    def __init__(self, device):
        self.bert = BERT()
        self.bert.to(device)

    def encode_value(self, value_seqs, ordered_value=False):
        value2id = {}
        if ordered_value:
            value_set = set()
            for value_seq in value_seqs:
                for value in value_seq:
                    value_set.add(value)
            value_list = list(value_set)
            value_list.sort()
            for value in value_list:
                value2id[value] = len(value2id)
        else:
            for value_seq in value_seqs:
                for value in value_seq:
                    if value not in value2id:
                        value2id[value] = len(value2id)
        # Add the [PAD] token for value embeddings
        value2id[self.bert.tokenizer.pad_token] = len(value2id)

        ## id2value is a sequence of English text
        id2value = sorted(value2id.items(), key=lambda x: x[1])
        id2value = [item[0] for item in id2value]  # it is a list
        best_layer = 1
        value_embedding = self.bert.pooled_encode_batched(id2value, layer=best_layer, batch_size=128,
                                                          save_gpu_memory=True)
        value_embedding = value_embedding.numpy()
        return value_embedding, id2value

    def load_value(self, value_seqs, value_embedding_cache_path, id2value_cache_path, load_from_file=True,
                   ordered_value=False):
        if load_from_file and value_embedding_cache_path.exists() and id2value_cache_path.exists():
            value_embedding = np.load(value_embedding_cache_path)
            with open(id2value_cache_path, 'r', encoding='utf8', errors='ignore') as f:
                id2value = json.load(f)
            print_time_info("Loaded value embedding from %s." % value_embedding_cache_path)
            print_time_info("Loaded values from %s." % id2value_cache_path)
        else:
            value_embedding, id2value = self.encode_value(value_seqs, ordered_value)
            np.save(value_embedding_cache_path, value_embedding)
            with open(id2value_cache_path, 'w', encoding='utf8', errors='ignore') as f:
                json.dump(id2value, f, ensure_ascii=False)
        assert len(value_embedding) == len(id2value)
        return value_embedding, id2value


class NumeralOrDate(object):
    def __init__(self):
        self.regex = {'year': re.compile(r'^\d{4}$'), 'date': re.compile(r'^(\d+)-(\d+)-(\d+)$'),
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

    def is_date_exp(self, text):
        for regex_name in self.regex:
            is_numeral, result = self.__regex_pattern(text, regex_name)
            if is_numeral:
                return is_numeral, result
        is_numeral, result = self.__month_year_pattern(text)
        if is_numeral:
            return is_numeral, result
        return False, None

    def __ckeck_year(self, year_str):
        int_year = int(year_str)
        if int_year > 2040:
            return False
        return True

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


def _split_time_digit_and_literal_attribute(value_and_attribute_pairs, digit_threshold, att_set):
    numeral_or_date = NumeralOrDate()
    att_is_number = {}
    for value, att_id in value_and_attribute_pairs:
        # 0 digit, 1 literal
        is_numeral, number = numeral_or_date.is_numeral(value)
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

    att_is_date = {}
    for value, att_id in value_and_attribute_pairs:
        # 0 date, 1 others
        is_date, date_exp = numeral_or_date.is_date_exp(value)
        if att_id in att_is_date:
            if is_date:
                att_is_date[att_id][0] += 1
                ingore_this = False
                for ignore_att in ['Start', 'End', 'Year', 'Date', 'start', 'end', 'date', 'year']:
                    if ignore_att in att_id:
                        ingore_this = True

                # if not ingore_this:
                #     print('att_id')

            else:
                att_is_date[att_id][1] += 1
        else:
            if is_date:
                att_is_date[att_id] = [1, 0]
            else:
                att_is_date[att_id] = [0, 1]
    # date format and frequency
    date_atts = {att for att, count in att_is_date.items() if count[0] / sum(count) > 0.90 and count[0] > 20}
    digit_atts = digit_atts - date_atts

    date_atts = list(date_atts)
    date_atts.sort()
    digit_atts = list(digit_atts)
    digit_atts.sort()
    literal_atts = list(literal_atts)
    literal_atts.sort()
    date_att2id = {date_atts: idx for idx, date_atts in enumerate(date_atts)}
    digit_att2id = {digit_att: idx for idx, digit_att in enumerate(digit_atts)}
    literal_att2id = {literal_att: idx for idx, literal_att in enumerate(literal_atts)}
    return date_att2id, digit_att2id, literal_att2id


def _split_time_digit_and_literal_triple(att_triples, time_att2id, digit_att2id, literal_att2id):
    time_triples = []
    digit_triples = []
    literal_triples = []
    digit_num = 0
    literal_num = 0
    time_num = 0
    for ent_id, value, att in att_triples:
        if att in digit_att2id:
            digit_triples.append((ent_id, value, digit_att2id[att]))
            digit_num += 1
        elif att in time_att2id:
            time_triples.append((ent_id, value, time_att2id[att]))
            time_num += 1
        else:
            literal_triples.append((ent_id, value, literal_att2id[att]))
            literal_num += 1
    return time_triples, digit_triples, literal_triples


class LoadDataDev(LoadData):
    def __init__(self, train_seeds_ratio, directory, nega_sample_num, name_channel,
                 attribute_channel, digit_literal_channel, dev_channel=False, load_new_seed_split=False, device='cpu'):
        super().__init__(train_seeds_ratio, directory, nega_sample_num, name_channel,
                         attribute_channel, digit_literal_channel, load_new_seed_split, device)

        if dev_channel:
            self.load_dev_features(load_attribute=False, load_digit_literal=False, load_digit_literal_time=True)

    # override with padding
    def load_name_feature(self):
        # Load translations
        if self.language_sr in {'zh', 'ja', 'fr'}:
            id2trans_sr = _load_trans(self.directory, self.language_sr)
            id2trans_sr[len(id2trans_sr)] = '[PAD]'
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

    # override with padding
    def load_structure_feature(self):
        # Load triples and entity mapping
        id2atts = read_mapping(self.directory / 'id2atts.txt')
        self.att2id = {att: idx for idx, att in id2atts.items()}
        self.att_num = len(self.att2id)
        self.triples_sr, self.id2entity_sr, self.id2relation_sr = _load_language(self.directory, self.language_sr)
        self.triples_tg, self.id2entity_tg, self.id2relation_tg = _load_language(self.directory, self.language_tg)

        self.triples_sr.append((len(self.id2entity_sr), len(self.id2relation_sr), len(self.id2entity_sr)))
        self.triples_tg.append((len(self.id2entity_tg), len(self.id2relation_tg), len(self.id2entity_tg)))
        self.id2entity_sr[len(self.id2entity_sr)] = '[PAD]'
        self.id2entity_tg[len(self.id2entity_tg)] = '[PAD]'
        self.id2relation_sr[len(self.id2relation_sr)] = '[PAD]'
        self.id2relation_tg[len(self.id2relation_tg)] = '[PAD]'
        self.att2id['[PAD]'] = len(self.att2id)

        print('att2id of [PAD]', self.att2id['[PAD]'])
        print('id2entity_sr of [PAD]', len(self.id2entity_sr))
        print('id2entity_tg of [PAD]', len(self.id2entity_tg))
        print('id2relation_sr of [PAD]', len(self.id2relation_sr))
        print('id2relation_tg of [PAD]', len(self.id2relation_tg))

        self.sr_ent_num = len(self.id2entity_sr)
        self.tg_ent_num = len(self.id2entity_tg)

    # override with padding and remove time attributes
    def load_attribute_feature(self, attribute_channel, digit_literal_channel, digit_literal_time_channel=False):
        self.load_dev_features(attribute_channel, digit_literal_channel, digit_literal_time_channel)

    # override with padding and remove time attributes
    def load_dev_features(self, load_attribute, load_digit_literal, load_digit_literal_time):
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

        elif load_digit_literal_time:
            train_value_and_attribute_sr = _get_train_value_and_attribute(self.train_sr_ent_seeds_ori, att_triples_sr)
            train_value_and_attribute_tg = _get_train_value_and_attribute(self.train_tg_ent_seeds_ori, att_triples_tg)

            digit_threshold = 0.5
            time_att2id, digit_att2id, literal_att2id = _split_time_digit_and_literal_attribute(
                train_value_and_attribute_sr + train_value_and_attribute_tg, digit_threshold, set(self.att2id.keys()))
            if '[PAD]' not in time_att2id:
                time_att2id['[PAD]'] = len(time_att2id)
            if '[PAD]' not in entity2id_sr:
                entity2id_sr['[PAD]'] = len(entity2id_sr)
            if '[PAD]' not in entity2id_tg:
                entity2id_tg['[PAD]'] = len(entity2id_tg)

            print('time_att2id of PAD', time_att2id['[PAD]'])
            print('entity2id_sr of PAD', entity2id_sr['[PAD]'])
            print('entity2id_tg of PAD', entity2id_tg['[PAD]'])

            self.time_att2id = time_att2id
            self.digit_att2id = digit_att2id
            self.literal_att2id = literal_att2id
            self.time_att_num = len(time_att2id)
            self.digit_att_num = len(digit_att2id)
            self.literal_att_num = len(literal_att2id)
            time_triples_sr, digit_triples_sr, literal_triples_sr = _split_time_digit_and_literal_triple(att_triples_sr,
                                                                                                         time_att2id,
                                                                                                         digit_att2id,
                                                                                                         literal_att2id)
            time_triples_tg, digit_triples_tg, literal_triples_tg = _split_time_digit_and_literal_triple(att_triples_tg,
                                                                                                         time_att2id,
                                                                                                         digit_att2id,
                                                                                                         literal_att2id)

            time_ent_id_seq_sr, time_att_id_seq_sr, time_value_seq_sr = transform_triple2seq(time_triples_sr,
                                                                                             language_sr, False)
            time_ent_id_seq_tg, time_att_id_seq_tg, time_value_seq_tg = transform_triple2seq(time_triples_tg,
                                                                                             language_tg, False)

            digit_ent_id_seq_sr, digit_att_id_seq_sr, digit_value_seq_sr = transform_triple2seq(digit_triples_sr,
                                                                                                language_sr, False)
            digit_ent_id_seq_tg, digit_att_id_seq_tg, digit_value_seq_tg = transform_triple2seq(digit_triples_tg,
                                                                                                language_tg, False)

            literal_ent_id_seq_sr, literal_att_id_seq_sr, literal_value_seq_sr = transform_triple2seq(
                literal_triples_sr, language_sr, False)
            literal_ent_id_seq_tg, literal_att_id_seq_tg, literal_value_seq_tg = transform_triple2seq(
                literal_triples_tg, language_tg, False)

            literal_value_embed_cache_path, literal_id2value_cache_path = get_cache_file_path(temp_file_dir, 'Literal')
            digit_value_embed_cache_path, digit_id2value_cache_path = get_cache_file_path(temp_file_dir, 'Digit')
            time_value_embed_cache_path, time_id2value_cache_path = get_cache_file_path(temp_file_dir, 'Time')

            self.literal_value_embedding, self.literal_id2value = value_embed_encoder.load_value(
                literal_value_seq_sr + literal_value_seq_tg, literal_value_embed_cache_path,
                literal_id2value_cache_path)

            self.digit_value_embedding, self.digit_id2value = value_embed_encoder.load_value(
                digit_value_seq_sr + digit_value_seq_tg, digit_value_embed_cache_path, digit_id2value_cache_path, )

            self.time_value_embedding, self.time_id2value = value_embed_encoder.load_value(
                time_value_seq_sr + time_value_seq_tg, time_value_embed_cache_path, time_id2value_cache_path,
                ordered_value=True)

            literal_value2id = {value: idx for idx, value in enumerate(self.literal_id2value)}
            digit_value2id = {value: idx for idx, value in enumerate(self.digit_id2value)}
            time_value2id = {value: idx for idx, value in enumerate(self.time_id2value)}

            print('digit_value2id of PAD', digit_value2id['[PAD]'])
            print('literal_value2id of PAD', literal_value2id['[PAD]'])
            print('time_value2id of PAD', time_value2id['[PAD]'])
            self.time_val_pad_id = time_value2id['[PAD]']

            time_value_id_seq_sr = [[time_value2id.get(value, time_value2id['[PAD]']) for value in value_seq] for
                                    value_seq in time_value_seq_sr]
            time_value_id_seq_tg = [[time_value2id.get(value, time_value2id['[PAD]']) for value in value_seq] for
                                    value_seq in time_value_seq_tg]

            digit_value_id_seq_sr = [[digit_value2id.get(value, digit_value2id['[PAD]']) for value in value_seq] for
                                     value_seq in digit_value_seq_sr]
            digit_value_id_seq_tg = [[digit_value2id.get(value, digit_value2id['[PAD]']) for value in value_seq] for
                                     value_seq in digit_value_seq_tg]

            literal_value_id_seq_sr = [[literal_value2id.get(value, literal_value2id['[PAD]']) for value in value_seq]
                                       for value_seq in literal_value_seq_sr]
            literal_value_id_seq_tg = [[literal_value2id.get(value, literal_value2id['[PAD]']) for value in value_seq]
                                       for value_seq in literal_value_seq_tg]

            literal_triples_sr = []
            for ent_id, att_seq, val_seq in zip(literal_ent_id_seq_sr, literal_att_id_seq_sr, literal_value_id_seq_sr):
                for att, val in zip(att_seq, val_seq):
                    literal_triples_sr.append((ent_id, val, att))
            self.literal_triples_sr = torch.tensor(literal_triples_sr)

            literal_triples_tg = []
            for ent_id, att_seq, val_seq in zip(literal_ent_id_seq_tg, literal_att_id_seq_tg, literal_value_id_seq_tg):
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

            time_triples_sr = []
            for ent_id, att_seq, val_seq in zip(time_ent_id_seq_sr, time_att_id_seq_sr, time_value_id_seq_sr):
                for att, val in zip(att_seq, val_seq):
                    time_triples_sr.append((ent_id, val, att))
            time_triples_sr.sort(key=lambda x: (x[0], x[1]))
            self.time_triples_sr = torch.tensor(time_triples_sr)

            time_triples_tg = []
            for ent_id, att_seq, val_seq in zip(time_ent_id_seq_tg, time_att_id_seq_tg, time_value_id_seq_tg):
                for att, val in zip(att_seq, val_seq):
                    time_triples_tg.append((ent_id, val, att))
            time_triples_tg.sort(key=lambda x: (x[0], x[1]))
            self.time_triples_tg = torch.tensor(time_triples_tg)
            self.time_diff_sr = None
            self.time_contexts_sr = []
            self.time_diff_tg = None
            self.time_contexts_tg = []

            settings = DevSettings()
            if settings.use_time_sequence:
                pad_item = [entity2id_sr['[PAD]'], time_value2id['[PAD]'], time_att2id['[PAD]']]
                time_contexts_sr = build_contexts_list(self.time_triples_sr, len(entity2id_sr))
                time_diff_sr = pad_context_seqs(time_contexts_sr, pad_item, settings.context_size)
                self.time_contexts_sr = time_contexts_sr
                self.time_diff_sr = torch.tensor(time_diff_sr)

                time_contexts_tg = build_contexts_list(self.time_triples_tg, len(entity2id_tg))
                time_diff_tg = pad_context_seqs(time_contexts_tg, pad_item, settings.context_size)
                self.time_contexts_tg = time_contexts_tg
                self.time_diff_tg = torch.tensor(time_diff_tg)



        elif load_digit_literal:
            if load_digit_literal:
                train_value_and_attribute_sr = _get_train_value_and_attribute(self.train_sr_ent_seeds_ori,
                                                                              att_triples_sr)
                train_value_and_attribute_tg = _get_train_value_and_attribute(self.train_tg_ent_seeds_ori,
                                                                              att_triples_tg)

                digit_threshold = 0.5
                digit_att2id, literal_att2id = _split_digit_attribute_and_literal_attribute(
                    train_value_and_attribute_sr + train_value_and_attribute_tg, digit_threshold,
                    set(self.att2id.keys()))
                self.digit_att2id = digit_att2id
                self.literal_att2id = literal_att2id
                self.digit_att_num = len(digit_att2id)
                self.literal_att_num = len(literal_att2id)
                digit_triples_sr, literal_triples_sr = _split_digit_and_literal_triple(att_triples_sr, digit_att2id,
                                                                                       literal_att2id)
                digit_triples_tg, literal_triples_tg = _split_digit_and_literal_triple(att_triples_tg, digit_att2id,
                                                                                       literal_att2id)

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


def build_contexts_list(time_triples, num_ent):
    time_triples = time_triples.tolist()
    e_dic = {}
    for triple in time_triples:
        h, val, pred = triple
        if h not in e_dic:
            e_dic[h] = [(h, val, pred)]
        else:
            e_dic[h].append((h, val, pred))

    for i in range(num_ent):
        if i not in e_dic:
            e_dic[i] = []

    keys = list(e_dic.keys())
    keys.sort()
    context_seqs = []
    for e in keys:
        e_pairs = e_dic.get(e, [])
        e_pairs.sort(key=lambda x: x[1])
        context_seqs.append(e_pairs)
    return context_seqs


def pad_context_seqs(context_seqs, pad_item, max_item=10):
    # pad_item = [ent_pad_id, val_pad_id, pred_pad_id]
    padded_time_seqs = []
    for e_pairs in context_seqs:
        copy_pairs = [x for x in e_pairs]
        while len(copy_pairs) < max_item:
            copy_pairs.append(pad_item)
        if len(copy_pairs) > max_item:
            out = random.sample(range(len(copy_pairs)), k=max_item)
            out.sort()
            copy_pairs = [copy_pairs[i] for i in out]
        padded_time_seqs.append(copy_pairs)

    return padded_time_seqs


def build_time_difference_seqs(time_triples, ent_pad_id, val_pad_id, pred_pad_id, num_ent, has_ent=False, max_item=10):
    if has_ent:
        pad_item = [ent_pad_id, pred_pad_id, pred_pad_id, val_pad_id, val_pad_id]
    else:
        pad_item = [pred_pad_id, pred_pad_id, val_pad_id, val_pad_id]
    time_triples = time_triples.tolist()
    e_dic = {}
    for triple in time_triples:
        h, val, pred = triple
        if h not in e_dic:
            e_dic[h] = [(val, pred)]
        else:
            e_dic[h].append((val, pred))

    for i in range(num_ent):
        if i not in e_dic:
            e_dic[i] = [pad_item]

    time_diff_seqs = []
    for e, pairs in e_dic.items():
        e_seq = []
        pairs.sort(key=lambda x: x[0])
        for i in range(len(pairs) - 1):
            val_i, pred_i = pairs[i]
            for j in range(i + 1, len(pairs)):
                val_j, pred_j = pairs[j]
                if has_ent:
                    e_seq.append([e, pred_i, pred_j, val_i, val_j])
                else:
                    e_seq.append([pred_i, pred_j, val_i, val_j])
        time_diff_seqs.append(e_seq)

    current_max = 0
    for time_seq in time_diff_seqs:
        if len(time_seq) > current_max:
            current_max = len(time_seq)

    if current_max < max_item:
        max_item = current_max

    padded_time_seqs = []
    for time_seq in time_diff_seqs:
        while len(time_seq) < max_item:
            time_seq.append(pad_item)
        if len(time_seq) > max_item:
            out = random.sample(range(len(time_seq)), k=max_item)
            out.sort()
            time_seq = [time_seq[i] for i in out]
        padded_time_seqs.append(time_seq)

    return padded_time_seqs

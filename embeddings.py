import json
import re
import json
import torch
import random
import numpy as np
from tqdm import tqdm

from util import print_time_info
from transformers import BertTokenizer, BertModel


def minus_mask(inputs, input_lens):
    # Inputs shape = (batch_size, sent_len, embed_dim)
    # input_len shape = [batch_sie]
    # max_len scalar
    assert inputs.shape[0] == input_lens.shape[0]
    assert len(input_lens.shape) == 1
    assert len(inputs.shape) == 3
    device = inputs.device

    max_len = torch.max(input_lens)
    batch_size = inputs.shape[0]
    mask = torch.arange(max_len).expand(batch_size, max_len).to(device)
    mask = mask >= input_lens.view(-1, 1)
    mask = mask.float()
    mask = mask.reshape(-1, max_len, 1) * (-1e30)
    # Employ mask
    inputs = inputs + mask
    return inputs


class BERT(object):
    # For entity alignment, the best layer is 1
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)
        self.model = BertModel.from_pretrained('bert-base-cased', output_hidden_states=True)
        self.model.eval()
        self.pad_token_id = self.tokenizer.encode(self.tokenizer.pad_token)[0]
        self.cls_token_id = self.tokenizer.encode(self.tokenizer.cls_token)[0]
        self.sep_token_id = self.tokenizer.encode(self.tokenizer.sep_token)[0]
        self.device = 'cpu'

    def to(self, device):
        self.device = device
        self.model.to(device)

    def pooled_encode_batched(self, sentences, batch_size=512, layer=1, save_gpu_memory=False):
        # Split the sentences into batches and further encode
        sent_batch = [sentences[i:i + batch_size] for i in range(0, len(sentences), batch_size)]
        outputs = []
        for batch in tqdm(sent_batch):
            out = self.pooled_bert_encode(batch, layer)
            if save_gpu_memory:
                out = out.cpu()
            outputs.append(out)
        outputs = torch.cat(outputs, dim=0)
        return outputs

    def pooled_bert_encode(self, sentences, layer=1):
        required_layer_hidden_state, sent_lens = self.bert_encode(sentences, layer)
        required_layer_hidden_state = minus_mask(required_layer_hidden_state, sent_lens.to(self.device))
        # Max pooling
        required_layer_hidden_state, indices = torch.max(required_layer_hidden_state, dim=1, keepdim=False)
        return required_layer_hidden_state

    def bert_encode(self, sentences, layer=1):
        # layer: output the max pooling over the designated layer hidden state

        # Limit batch size to avoid exceed gpu memory limitation
        sent_num = len(sentences)
        assert sent_num <= 512

        ## The 382 is to avoid exceed bert's maximum seq_len and to save memory
        sentences = [[self.cls_token_id] + self.tokenizer.encode(sent)[:382] + [self.sep_token_id] for sent in
                     sentences]
        sent_lens = [len(sent) for sent in sentences]
        max_len = max(sent_lens)
        sent_lens = torch.tensor(sent_lens)
        sentences = torch.tensor([sent + (max_len - len(sent)) * [self.pad_token_id] for sent in sentences]).to(
            self.device)
        with torch.no_grad():
            # old version
            # last_hidden_state, pooled_output, all_hidden_state = self.model(sentences)
            outputs = self.model(sentences)
            last_hidden_states = outputs.last_hidden_state
            pooled_output = outputs.pooler_output
            all_hidden_state = outputs.hidden_states
        assert len(all_hidden_state) == 13
        required_layer_hidden_state = all_hidden_state[layer]
        return required_layer_hidden_state, sent_lens


class ValueEmbedding(object):
    def __init__(self, device):
        self.bert = BERT()
        self.bert.to(device)

    def encode_value(self, value_seqs):
        value2id = {}
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

    def load_value(self, value_seqs, value_embedding_cache_path, id2value_cache_path, load_from_cache=False):
        if load_from_cache and value_embedding_cache_path.exists() and id2value_cache_path.exists():
            value_embedding = np.load(value_embedding_cache_path)
            with open(id2value_cache_path, 'r', encoding='utf8', errors='ignore') as f:
                id2value = json.load(f)
            print_time_info("Loaded value embedding from %s." % value_embedding_cache_path)
            print_time_info("Loaded values from %s." % id2value_cache_path)
        else:
            value_embedding, id2value = self.encode_value(value_seqs)
            np.save(value_embedding_cache_path, value_embedding)
            with open(id2value_cache_path, 'w', encoding='utf8', errors='ignore') as f:
                json.dump(id2value, f, ensure_ascii=False)
        assert len(value_embedding) == len(id2value)
        return value_embedding, id2value


def get_cache_file_path(temp_file_dir, attribute_channel_name):
    if not temp_file_dir.exists():
        temp_file_dir.mkdir()
    assert attribute_channel_name in {'Literal', 'Digit', 'Attribute', 'Time', 'Cluster'}
    embedding_file_name = 'value_embedding'
    id2values_file_name = 'id2value'
    embedding_file_name = '%s_%s' % (embedding_file_name, attribute_channel_name)
    id2values_file_name = '%s_%s' % (id2values_file_name, attribute_channel_name)
    embedding_file_path = temp_file_dir / ('%s.npy' % embedding_file_name)
    id2values_file_path = temp_file_dir / ('%s.json' % id2values_file_name)
    return embedding_file_path, id2values_file_path

"""
:author: Qizhi Li
"""
import torch
import pyprind
import numpy as np
import pandas as pd
import torch.nn as nn
from transformers import BertConfig, BertTokenizer, BertForMaskedLM
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class Config:
    def __init__(self):
        self.prefix = 'Totally, it was [MASK].'
        self.verbalizer = {
            'good': 1,
            'fascinating': 1,
            'perfect': 1,
            'bad': 0,
            'horrible': 0,
            'terrible': 0,
        }

        self.max_seq_length = 512
        self.batch_size = 64


config = Config()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_path = '../static_data/bert-base-uncased'
model_class, tokenizer_class, pretrained_weight = (BertForMaskedLM,
                                                   BertTokenizer,
                                                   model_path)
bert_config = BertConfig.from_pretrained(pretrained_weight)
tokenizer = tokenizer_class.from_pretrained(pretrained_weight)
bert = model_class.from_pretrained(pretrained_weight, config=bert_config).to(device)
softmax = nn.Softmax(dim=1)


def obtain_verbalizer_ids(verbalizer, tokenizer):
    """
    将 verbalizer 中的词语转成 Embedding layer 的 id
    :param verbalizer: dict
    :param tokenizer: Object
    :return verbalizer_ids: list
            verbalizer 中所有词语在 Embedding layer 中的 id
    :return index2ids: dict
            verbalizer_ids 中 index 与 token id 之间的映射
    """
    verbalizer_ids = tokenizer.convert_tokens_to_ids(list(verbalizer.keys()))
    index2ids = {i: verbalizer_ids[i] for i in range(len(verbalizer_ids))}
    return verbalizer_ids, index2ids


verbalizer_ids, index2ids = obtain_verbalizer_ids(config.verbalizer, tokenizer)


def concatenate_prefix(texts, config):
    """
    将 prefix 拼接在每个文本之前
    :param texts: list
    :param config: Object
    :return prefix_texts: list
            带有 prefix 的文本
    """
    prefix_texts = []

    for text in texts:
        prefix_texts.append('{}{}'.format(config.prefix, text))

    return prefix_texts


def load_data(config):
    """
    加载数据集
    :return texts: list
    :return labels: list
    """
    # ['texts', 'labels']
    df = pd.read_csv('../static_data/IMDB.csv')
    original_texts = df['texts'].tolist()
    labels = df['labels'].tolist()

    # texts = truncation(original_texts, config, tokenizer)
    texts = concatenate_prefix(original_texts, config)

    return texts, labels


texts, labels = load_data(config)


def pack_batch(texts, labels, batch_size):
    """
    将数据打包为 batch
    :param texts: list
    :param labels: list
    :param batch_size: int
    :return batch_X: list
            [[text11, text12, ...], [text21, text22, ...], ...]
    :return batch_y: list
            [[label11, label12, ...], [label21, label22, ...], ...]
    :return batch_count: int
            一共有多少个 batch
    """
    assert len(texts) == len(labels)

    # 如果正好数据长度整除 batch_size, 则 batch_count = num_texts / batch_size
    # 否则 batch_count = (num_texts / batch_size) + 1
    if len(texts) % batch_size != 0:
        flag = False
        batch_count = int(len(texts) / batch_size) + 1
    else:
        flag = True
        batch_count = int(len(texts) / batch_size)

    batch_X, batch_y = [], []

    # 如果正好数据长度整除 batch_size, 则正好把所有数据打包到 batch 中
    # 否则剩下的数据也要打包到最后一个 batch 中
    if flag:
        for i in range(batch_count):
            batch_X.append(texts[i * batch_size: (i + 1) * batch_size])
            batch_y.append(labels[i * batch_size: (i + 1) * batch_size])
    else:
        for i in range(batch_count):
            if i == batch_count - 1:
                batch_X.append(texts[i * batch_size:])
                batch_y.append(labels[i * batch_size:])
            else:
                batch_X.append(texts[i * batch_size: (i + 1) * batch_size])
                batch_y.append(labels[i * batch_size: (i + 1) * batch_size])

    return batch_X, batch_y, batch_count


batch_X, batch_y, batch_count = pack_batch(texts, labels, config.batch_size)

with torch.no_grad():
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    pper = pyprind.ProgPercent(batch_count)
    for i in range(batch_count):
        inputs = batch_X[i]
        labels = batch_y[i]

        tokens = tokenizer.batch_encode_plus(inputs, add_special_tokens=True,
                                             max_length=config.max_seq_length,
                                             padding='max_length', truncation=True)
        ids = torch.tensor(tokens['input_ids']).to(device)
        attention_mask = torch.tensor(tokens['attention_mask']).to(device)

        # shape: (batch_size, max_seq_length, vocab_size)
        logits = bert(ids, attention_mask=attention_mask).logits

        # mask_token_index[0]: 第 i 条数据
        # mask_token_index[1]: 第 i 条数据的 [MASK] 在序列中的位置
        mask_token_index = (ids == tokenizer.mask_token_id).nonzero(as_tuple=True)

        # 找到 [MASK] 的 logits
        # shape: (batch_size, vocab_size)
        masked_logits = logits[mask_token_index[0], mask_token_index[1], :]

        # 将 [MASK] 位置中 verbalizer 里的词语的 logits 给提取出来
        # shape: (batch_size, verbalizer_size)
        verbalizer_logits = masked_logits[:, verbalizer_ids]

        # 将这些 verbalizer 中的 logits 给构造一个伪分布
        pseudo_distribution = softmax(verbalizer_logits)

        # 找到伪分布中概率最大的 index
        pred_indices = pseudo_distribution.argmax(axis=-1).tolist()
        # 将 index 转换为词语的 id
        pred_ids = [index2ids[index] for index in pred_indices]
        # 将 id 转换为 token
        pred_tokens = tokenizer.convert_ids_to_tokens(pred_ids)
        # 找到 token 对应的 label
        pred_labels = [config.verbalizer[token] for token in pred_tokens]

        predict_all = np.append(predict_all, pred_labels)
        labels_all = np.append(labels_all, labels)

        pper.update()

    acc = accuracy_score(labels_all, predict_all)
    p = precision_score(labels_all, predict_all)
    r = recall_score(labels_all, predict_all)
    f1 = f1_score(labels_all, predict_all)

    print('accuracy: %f | precision: %f | recall: %f | f1: %f' % (acc, p, r, f1))




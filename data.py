import csv
import torch

from file_handling import read_json
from pathlib import Path
from util import shuffle_index


class DPRDataset(torch.utils.data.Dataset):

    def __init__(self, path):
        extension = Path(path).suffix
        if extension == '.json':  # train/dev data
            self.samples = read_json(path)
        elif extension == '.csv':  # test data
            self.samples = []
            with open(path) as f:
                reader = csv.reader(f, delimiter='\t')
                for row in reader:
                    self.samples.append({'question': row[0],
                                         'answers': eval(row[1])})
        else:
            raise ValueError('Invalid DPR file extension: ' + extension)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index], index


class WikiPassageDataset(torch.utils.data.Dataset):

    def __init__(self, path):
        self.samples = []
        with open(path) as f:
            reader = csv.reader(f, delimiter='\t')
            for row in reader:
                if row[0] == 'id':
                    continue
                self.samples.append({'pid': int(row[0]), 'text': row[1],
                                     'title': row[2]})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


def text2tensor(tokenizer, inputs, titles=None, max_length=256,
                pad_to_max=False):
    padding = 'max_length' if pad_to_max else True

    if titles is not None:
        tensor = tokenizer(titles, inputs, padding=padding, truncation=True,
                           max_length=max_length, return_tensors='pt')
    else:
        tensor = tokenizer(inputs, padding=padding, truncation=True,
                           max_length=max_length, return_tensors='pt')

    if pad_to_max:  # For consistency with original DPR code
        tensor['input_ids'][:, -1] = tokenizer.sep_token_id
        tensor['attention_mask'][:, -1] = 1
        tensor['token_type_ids'].fill_(0)

    return tensor


def tensorize_train(samples, tokenizer, max_length, num_hard_negatives=1,
                    num_other_negatives=0, pad_to_max=False, shuffle=False):
    queries = []
    labels = []
    titles = []
    texts = []
    indices_sample = []  # Tracking dataset indices for debugging
    indices_other_lst = []  # Tracking other negative indices for debugging
    indices_hard_lst = []  # Tracking hard negative indices for debugging

    for sample, index in samples:
        others, indices_other = shuffle_index(sample['negative_ctxs'], shuffle)
        hards, indices_hard = shuffle_index(sample['hard_negative_ctxs'],
                                            shuffle)

        queries.append(sample['question'])
        labels.append(len(titles))

        titles.append(sample['positive_ctxs'][0]['title'])
        texts.append(sample['positive_ctxs'][0]['text'])

        other_negs = others[:num_other_negatives]
        titles.extend([other_neg['title'] for other_neg in other_negs])
        texts.extend([other_neg['text'] for other_neg in other_negs])

        # Following DPR, but note if hard negs fall back to other negs, AND
        # other negs are used, they're repeated.
        hard_negs = hards if hards != [] else others
        hard_negs = hard_negs[:num_hard_negatives]
        titles.extend([hard_neg['title'] for hard_neg in hard_negs])
        texts.extend([hard_neg['text'] for hard_neg in hard_negs])
        indices_sample.append(index)
        indices_other_lst.append(indices_other[:num_other_negatives])
        indices_hard_lst.append(indices_hard[:num_hard_negatives])

    queries = text2tensor(tokenizer, queries, max_length=max_length,
                          pad_to_max=pad_to_max)
    passages = text2tensor(tokenizer, texts, titles=titles,
                           max_length=max_length, pad_to_max=pad_to_max)

    Q = queries['input_ids']  # (B, L)
    Q_mask = queries['attention_mask']
    Q_type = queries['token_type_ids']
    P = passages['input_ids']  # (MB, L)
    P_mask = passages['attention_mask']
    P_type = passages['token_type_ids']
    labels = torch.LongTensor(labels)  # (B,) elements in [0, MB)

    return Q, Q_mask, Q_type, P, P_mask, P_type, labels, indices_sample, \
        indices_other_lst, indices_hard_lst


def tensorize_questions(samples, tokenizer, max_length, pad_to_max=False):
    queries = []

    for sample, _ in samples:
        queries.append(sample['question'])

    queries = text2tensor(tokenizer, queries, max_length=max_length,
                          pad_to_max=pad_to_max)
    Q = queries['input_ids']  # (B, L)
    Q_mask = queries['attention_mask']
    Q_type = queries['token_type_ids']

    return Q, Q_mask, Q_type


def tensorize_passages(samples, tokenizer, max_length, pad_to_max=False):
    titles = []
    texts = []
    pids = []

    for sample in samples:
        titles.append(sample['title'])
        texts.append(sample['text'])
        pids.append(sample['pid'])

    passages = text2tensor(tokenizer, texts, titles=titles,
                           max_length=max_length, pad_to_max=pad_to_max)

    P = passages['input_ids']  # (B, L)
    P_mask = passages['attention_mask']
    P_type = passages['token_type_ids']
    I = torch.LongTensor(pids)  # Need to save pids for distributed

    return P, P_mask, P_type, I

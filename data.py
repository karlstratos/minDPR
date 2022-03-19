import csv
import torch

from file_handling import read_json
from pathlib import Path


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

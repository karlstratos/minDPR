import faiss
import os
import random
import sys
import unicodedata

from datetime import datetime


def get_flat_index(dim, metric):
    if metric == 'IP':
        return faiss.IndexFlatIP(dim)
    elif metric == 'L2':
        return faiss.IndexFlatL2(dim)
    else:
        raise ValueError('Invalid metric for Flat Index: ' + metric)


def get_faiss_metric(metric):
    if metric == 'IP':
        return faiss.METRIC_INNER_PRODUCT
    elif metric == 'L2':
        return faiss.METRIC_L2
    else:
        raise ValueError('Invalid faiss metric: ' + metric)


def check_distributed():
    if 'RANK' in os.environ:
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
    else:
        rank = -1
        local_rank = -1
        world_size = -1
    return rank, local_rank, world_size


def strtime(datetime_checkpoint):
    diff = datetime.now() - datetime_checkpoint
    return str(diff).rsplit('.')[0]  # Ignore below seconds


def uninorm(text): # https://en.wikipedia.org/wiki/Unicode_equivalence
    return unicodedata.normalize('NFD', text)


def shuffle_index(lst, shuffle=True):
    if lst == []:
        return [], []
    lst_indexed = list(enumerate(lst))
    if shuffle:
        random.shuffle(lst_indexed)
    shuffled_indices, shuffled_lst = zip(*lst_indexed)
    return shuffled_lst, shuffled_indices


def contains(lst, sublst):  # True iff lst contains sublst
    for i in range(0, len(lst) - len(sublst) + 1):
        if sublst == lst[i: i + len(sublst)]:
            return True
    return False


class Logger(object):

    def __init__(self, log_path=None, on=True):
        self.log_path = log_path
        self.on = on

        if self.on and self.log_path is not None:
            while os.path.isfile(self.log_path):
                self.log_path += '+'

    def log(self, string, newline=True, force=False):
        if self.on or force:
            sys.stdout.write(string)
            if newline: sys.stdout.write('\n')
            sys.stdout.flush()

            if self.on and self.log_path is not None:
                with open(self.log_path, 'a') as logf:
                    logf.write(string)
                    if newline: logf.write('\n')
                    logf.flush()

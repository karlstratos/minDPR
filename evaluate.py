import argparse
import os

from collections import Counter
from file_handling import read_json
from simple_tokenizer import word_tokenize
from util import contains, uninorm


def has_answer(candidate, answers, mode='string', lower=True):
    if mode == 'string':  # True if candidate['text'] contains an answer.
        text_words = word_tokenize(uninorm(candidate['text']), lower=lower)
        for answer in answers:
            answer_words = word_tokenize(uninorm(answer), lower=lower)
            if contains(text_words, answer_words):
                return True
        return False

    elif mode == 'title':  # True if candidate['title'] matches an answer.
        title = candidate['title'].strip()
        if lower:
            title = title.lower()
        for answer in answers:
            answer = answer.strip()
            if lower:
                answer = answer.lower()
            if title == answer:
                return True
        return False

    else:
        raise ValueError('Invalid has_answer mode: ' + mode)


def topk_retrieval_accuracy(results, k_values, unnormalized=False):
    # Each result has candidates. Each candidate has 'has_answer'.
    k_num_correct = Counter()
    for k in k_values:
        k_num_correct[k] = 0
    for result in results:
        rank_min = float('inf')
        candidates = result['candidates'] if 'candidates' in result else \
                     result['ctxs']  # DPR released
        for rank, candidate in enumerate(candidates):
            if candidate['has_answer']:
                rank_min = rank
                break
        for k in k_values:
            if rank_min < k:
                k_num_correct[k] += 1

    num_queries = len(results)
    def get_acc(num_correct):
        return num_correct if unnormalized else num_correct / num_queries
    k_accuracy = {k: get_acc(num_correct) for k, num_correct in
                  k_num_correct.items()}
    k_accuracy['num_queries'] = num_queries
    return k_accuracy


def print_performance(file_k_num_correct, k_values, micro=False):
    # file_k_num_correct[file(path)] =  {k: num_correct_k (ks), num_queries: N}

    def compute_average_accuracy(k):
        if micro:  # Micro accuracy: makes sense when files same domain/nature
            num_correct_total = sum(file_k_num_correct[f][k] for f in
                                    file_k_num_correct)
            num_queries_total = sum(file_k_num_correct[f]['num_queries'] for f
                                    in file_k_num_correct)
            acc_micro = num_correct_total / num_queries_total
            return acc_micro
        else:  # Macro accuracy: makes more sense when files different domains
            accs = [file_k_num_correct[f][k] / \
                    file_k_num_correct[f]['num_queries'] for f in
                    file_k_num_correct]
            acc_macro = sum(accs) / len(accs)
            return acc_macro

    AVG = 'Avg' + ('(micro)' if micro else '(macro)')
    file_name = {f: os.path.basename(f) for f in file_k_num_correct}
    file_name[AVG] = AVG
    files_sorted = sorted(file_k_num_correct.keys(), key=lambda f: file_name[f])
    if len(files_sorted) > 1:
        files_sorted.append(AVG)

    table_lines = []
    for f in files_sorted:
        table_line = []
        for k in k_values:
            if f == AVG:
                acc = compute_average_accuracy(k)
            else:
                acc = file_k_num_correct[f][k] / \
                      file_k_num_correct[f]['num_queries']
            table_line.append(f'{acc:.1%}')
        table_lines.append(table_line + [file_name[f]] +
                           [str(file_k_num_correct[f]['num_queries'])
                            if f != AVG else '-'])

    header = [f'k={k}' for k in k_values] + ['filename', 'num_queries']
    print('\t'.join(header))
    for line in table_lines:
        print('\t'.join(line))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('results', type=str)
    parser.add_argument('--k_values', type=str, default='1,5,20,100')
    args = parser.parse_args()

    results = read_json(args.results)
    k_values = sorted([int(k) for k in args.k_values.split(',')])
    k_accuracy = topk_retrieval_accuracy(results, k_values, unnormalized=True)
    print_performance({args.results: k_accuracy}, k_values)

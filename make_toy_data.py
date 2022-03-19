# python make_toy_data.py data/nq-train.json --N 11
import argparse
import json
import os

from copy import deepcopy


def main(args):
    print(args)
    with open(args.data) as f:
        examples = json.load(f)
    toy_data = []
    for example in examples[:args.N]:
        toy = deepcopy(example)
        toy['positive_ctxs'] = toy['positive_ctxs'][:args.N]
        toy['negative_ctxs'] = toy['negative_ctxs'][:args.N]
        toy['hard_negative_ctxs'] = toy['hard_negative_ctxs'][:args.N]
        toy_data.append(toy)

    split1, split2 = os.path.splitext(args.data)
    outpath = split1 + str(args.N) + split2
    with open(outpath, 'w') as f:
        f.write(json.dumps(toy_data, indent=4))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data', type=str, help='DPR retriever training data')
    parser.add_argument('--N', type=int, default=11, help='max num of things')
    args = parser.parse_args()

    main(args)

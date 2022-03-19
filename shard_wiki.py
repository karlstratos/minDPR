# python shard_wiki.py data/psgs_w100_first11.tsv --num_shards 2
# python shard_wiki.py data/psgs_w100.tsv --num_shards 10
import argparse
import os

from math import ceil
from pathlib import Path


def main(args):
    print(args)
    with open(args.wiki) as f:
        lines = f.readlines()

    num_items = len(lines) - 1  # Excluding the header
    num_items_per_shard = ceil(num_items / args.num_shards)
    shards = [lines[i: i + num_items_per_shard]
              for i in range(1, num_items + 1, num_items_per_shard)]
    print(f'{num_items} items, splitting into {args.num_shards} shards')

    wiki_dir = os.path.dirname(os.path.abspath(args.wiki))
    for shard_num, shard in enumerate(shards):
        shard_name = Path(args.wiki).stem + f'_shard{shard_num}.tsv'
        shard_path = os.path.join(wiki_dir, shard_name)
        print(f'Writing {shard_path} ({len(shard)} items)')
        with open(shard_path, 'w') as f:
            f.write(lines[0])
            f.write(''.join(shard))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('wiki', type=str)
    parser.add_argument('--num_shards', type=int, default=10)
    args = parser.parse_args()

    main(args)

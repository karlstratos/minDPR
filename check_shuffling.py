# Code to check the behavior of shuffling
#
# python check_shuffling.py  --gpus 0 --no_shuffle
# python check_shuffling.py  --gpus 0
# python check_shuffling.py  --gpus 0 --drop_last_loader
# python check_shuffling.py  --gpus 0 --seed 12345
#
# torchrun --standalone --nnodes=1 --nproc_per_node=2 check_shuffling.py --gpus 0,1 --no_shuffle
# torchrun --standalone --nnodes=1 --nproc_per_node=2 check_shuffling.py --gpus 0,1
# torchrun --standalone --nnodes=1 --nproc_per_node=2 check_shuffling.py --gpus 0,1 --seed 12345
# torchrun --standalone --nnodes=1 --nproc_per_node=2 check_shuffling.py --gpus 0,1 --turn_off_set_epoch
# torchrun --standalone --nnodes=1 --nproc_per_node=2 check_shuffling.py --gpus 0,1 --drop_last_loader
# torchrun --standalone --nnodes=1 --nproc_per_node=2 check_shuffling.py --gpus 0,1 --drop_last_sampler  # Each process assigned 5 samples instead of 6
import argparse
import os


def main(args):
    import random
    import torch
    import transformers

    from copy import deepcopy
    from data import DPRDataset, tensorize_train
    from datetime import datetime
    from torch.distributed import init_process_group, barrier
    from torch.utils.data import DataLoader
    from torch.utils.data.distributed import DistributedSampler
    from transformers import AutoTokenizer, set_seed
    from util import Logger, check_distributed, strtime

    set_seed(args.seed)
    rank, local_rank, world_size = check_distributed()
    is_main_process = local_rank in [-1, 0]
    is_distributed = world_size != -1

    logger = Logger(on=is_main_process)
    logger.log(str(args))
    logger.log(f'rank {rank} local_rank {local_rank} world_size {world_size}',
               force=True)

    if is_distributed:
        torch.cuda.set_device(local_rank)
        device = torch.device('cuda', local_rank)
        init_process_group('nccl')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logger.log(f'Using device: {str(device)}', force=True)

    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    dataset = DPRDataset(args.data)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank,
                                 shuffle=not args.no_shuffle, seed=args.seed,
                                 drop_last=args.drop_last_sampler) \
                                 if is_distributed else None
    collate_fn = lambda samples: tensorize_train(
        samples, tokenizer, args.max_length, args.num_hard_negatives,
        args.num_other_negatives, args.pad_to_max,
        shuffle=not args.no_shuffle)

    loader = DataLoader(dataset, args.batch_size,
                        shuffle=(sampler is None and not args.no_shuffle),
                        sampler=sampler, num_workers=args.num_workers,
                        collate_fn=collate_fn, drop_last=args.drop_last_loader)

    for epoch in range(args.epochs):
        logger.log('-' * 80)
        if is_distributed:
            barrier()  # Just for readability of the output
            if not args.turn_off_set_epoch:
                loader.sampler.set_epoch(epoch)

        for batch_num, batch in enumerate(loader):
            indices_sample, indices_other_lst, indices_hard_lst = batch[-3:]

            logger.log(f'rank {rank} batch {batch_num}: samples '
                       f'{str(indices_sample)}, hard {str(indices_hard_lst)}, '
                       f'other {str(indices_other_lst)}', force=True)

        if is_distributed:
            barrier()  # Just for readability of the output


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/nq-train11.json')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--max_length', type=int, default=256)
    parser.add_argument('--num_hard_negatives', type=int, default=2)
    parser.add_argument('--num_other_negatives', type=int, default=2)
    parser.add_argument('--pad_to_max', action='store_true')
    parser.add_argument('--no_shuffle', action='store_true')
    parser.add_argument('--drop_last_sampler', action='store_true')
    parser.add_argument('--drop_last_loader', action='store_true')
    parser.add_argument('--turn_off_set_epoch', action='store_true')
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpus', default='', type=str)
    args = parser.parse_args()

    # Set environment variables before importing libraries that use them!
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    main(args)

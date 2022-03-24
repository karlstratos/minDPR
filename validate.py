# Code to check validation
#
# python validate.py --batch_size_val 11 --num_hard_negatives 3 --num_other_negatives 4 --subbatch_size 7 --gpus 0
# torchrun --standalone --nnodes=1 --nproc_per_node=2 validate.py --batch_size_val 6 --num_hard_negatives 3 --num_other_negatives 4 --subbatch_size 7 --gpus 0,1
# torchrun --standalone --nnodes=1 --nproc_per_node=8 validate.py --data_val data/nq-dev.json --batch_size_val 512 --num_hard_negatives 30 --num_other_negatives 30 --subbatch_size 1024 --gpus 0,1,2,3,4,5,6,7 --num_workers 2 --pad_to_max  # 15210/49584 per process, 1-2m

import argparse
import os


def main(args):
    import torch
    import transformers

    from data import DPRDataset, tensorize_train
    from datetime import datetime
    from model import make_bert_model, validate_by_rank
    from torch.distributed import init_process_group
    from torch.utils.data import DataLoader
    from torch.utils.data.distributed import DistributedSampler
    from transformers import AutoTokenizer
    from util import Logger, check_distributed, strtime

    transformers.logging.set_verbosity_error()

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

    # Validation loader: deterministic iteration
    dataset_val = DPRDataset(args.data_val)
    sampler_val = DistributedSampler(dataset_val, num_replicas=world_size,
                                     rank=rank, shuffle=False) \
                                     if is_distributed else None
    collate_fn_val = lambda samples: tensorize_train(
        samples, tokenizer, args.max_length, args.num_hard_negatives_val,
        args.num_other_negatives_val, args.pad_to_max)
    loader_val = DataLoader(dataset_val, args.batch_size_val, shuffle=False,
                            sampler=sampler_val, num_workers=args.num_workers,
                            collate_fn=collate_fn_val)

    model = make_bert_model(tokenizer).to(device)

    if is_distributed:
        logger.log('DDP wrapping')
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
            find_unused_parameters=False)
    else:
        logger.log('Single-process single-device, no model wrapping')

    start_time = datetime.now()
    avgrank, num_cands_avg = validate_by_rank(model, loader_val, rank,
                                              world_size, device,
                                              args.subbatch_size)
    logger.log(f'\nDone | On average, rank {avgrank:4.3f} out of '
               f'{num_cands_avg:4.3f} cands per process | '
               f'{strtime(start_time)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_val', type=str, default='data/nq-train11.json')
    parser.add_argument('--max_length', type=int, default=256)
    parser.add_argument('--batch_size_val', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--num_hard_negatives_val', type=int, default=30)
    parser.add_argument('--num_other_negatives_val', type=int, default=30)
    parser.add_argument('--subbatch_size', type=int, default=128)
    parser.add_argument('--pad_to_max', action='store_true')
    parser.add_argument('--gpus', default='', type=str)
    args = parser.parse_args()

    # Set environment variables before importing libraries that use them!
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    main(args)

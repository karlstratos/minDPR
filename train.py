# python train.py /tmp/model data/nq-train11.json data/nq-train11.json --batch_size 3 --batch_size_val 4 --lr 1e-4 --num_warmup_steps 1 --epochs 5 --start_epoch_val 0  --gpus 0
# torchrun --standalone --nnodes=1 --nproc_per_node=2  train.py /tmp/model data/nq-train11.json data/nq-train11.json --batch_size 3 --batch_size_val 4 --lr 1e-4 --num_warmup_steps 1 --epochs 5 --start_epoch_val 0  --gpus 0,1
import argparse
import os


def main(args):
    import random
    import torch
    import transformers

    from copy import deepcopy
    from data import DPRDataset, tensorize_train
    from datetime import datetime
    from file_handling import mkdir_optional
    from model import make_bert_model, get_bert_optimizer, get_loss, \
        validate_by_rank
    from torch.distributed import init_process_group
    from torch.utils.data import DataLoader
    from torch.utils.data.distributed import DistributedSampler
    from transformers import AutoTokenizer, set_seed, \
        get_linear_schedule_with_warmup
    from util import Logger, check_distributed, strtime

    transformers.logging.set_verbosity_error()

    set_seed(args.seed)
    rank, local_rank, world_size = check_distributed()
    is_main_process = local_rank in [-1, 0]
    is_distributed = world_size != -1

    mkdir_optional(os.path.dirname(args.model))
    logger = Logger(log_path=args.model + '.log', on=is_main_process)
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

    # Train loader: data shuffling + dynamic within-example shuffling
    dataset_train = DPRDataset(args.data_train)
    sampler_train = DistributedSampler(dataset_train, num_replicas=world_size,
                                       rank=rank, shuffle=True,
                                       seed=args.seed) \
                                       if is_distributed else None
    collate_fn_train = lambda samples: tensorize_train(
        samples, tokenizer, args.max_length, args.num_hard_negatives_train,
        args.num_hard_negatives_train, args.pad_to_max, shuffle=True)
    loader_train = DataLoader(dataset_train, args.batch_size,
                              shuffle=(sampler_train is None),
                              sampler=sampler_train,
                              num_workers=args.num_workers,
                              collate_fn=collate_fn_train)
    num_training_steps = len(loader_train) * args.epochs

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

    model = make_bert_model(tokenizer, dropout=args.dropout).to(device)
    optimizer = get_bert_optimizer(model, learning_rate=args.lr,
                                   adam_eps=args.adam_eps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.num_warmup_steps,
        num_training_steps=num_training_steps)

    if is_distributed:
        logger.log('DDP wrapping')
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank)
    else:
        logger.log('Single-process single-device, no model wrapping')

    # Training
    loss_val_best = float('inf')
    sd_best = None
    start_time = datetime.now()
    for epoch in range(args.epochs):
        model.train()
        loss_sum = 0.
        num_correct_sum = 0
        if is_distributed:
            loader_train.sampler.set_epoch(epoch)

        for batch_num, batch in enumerate(loader_train):
            loss, num_correct = get_loss(model, batch, rank, world_size, device)
            loss_sum += loss.item()
            num_correct_sum += num_correct.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        loss_avg = loss_sum / len(loader_train)
        acc = num_correct_sum / len(dataset_train) * 100.

        loss_val = -1
        is_best_string = ''
        if epoch >= args.start_epoch_val:
            model.eval()
            avgrank, num_cands = validate_by_rank(model, loader_val, rank,
                                                  world_size, device,
                                                  args.subbatch_size)
            loss_val = avgrank
            if loss_val < loss_val_best:
                sd = model.module.state_dict() if is_distributed else \
                     model.state_dict()
                sd_best = deepcopy(sd)
                is_best_string = ' <-------------'
                loss_val_best = loss_val

        logger.log(f'Epoch {epoch:3d}: loss {loss_avg:10.2f}, acc '
                   f'{acc:10.2f}, val {loss_val:10.3f} {is_best_string}')

    if is_main_process and sd_best is not None:
        logger.log(f'\nDone training | total time {strtime(start_time)} | '
                   f'saving best model to {args.model}')
        torch.save({'sd': sd_best, 'args': args}, args.model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str)
    parser.add_argument('data_train', type=str)
    parser.add_argument('data_val', type=str)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--batch_size_val', type=int, default=512)
    parser.add_argument('--max_length', type=int, default=256)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--adam_eps', type=float, default=1e-8)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--num_warmup_steps', type=int, default=0)
    parser.add_argument('--clip', type=float, default=2.)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--num_hard_negatives_train', type=int, default=1)
    parser.add_argument('--num_other_negatives_train', type=int, default=0)
    parser.add_argument('--num_hard_negatives_val', type=int, default=30)
    parser.add_argument('--num_other_negatives_val', type=int, default=30)
    parser.add_argument('--subbatch_size', type=int, default=1024)
    parser.add_argument('--start_epoch_val', type=int, default=30)
    parser.add_argument('--pad_to_max', action='store_true')
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpus', default='', type=str)
    args = parser.parse_args()

    # Set environment variables before importing libraries that use them!
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    main(args)

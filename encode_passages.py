import argparse
import os


def main(args):
    import glob
    import pickle
    import torch
    import torch.distributed as dist
    import transformers

    from data import WikiPassageDataset, tensorize_passages
    from datetime import datetime
    from file_handling import mkdir_optional
    from model import load_model
    from pathlib import Path
    from torch.distributed import init_process_group
    from torch.utils.data import DataLoader
    from torch.utils.data.distributed import DistributedSampler
    from transformers import AutoTokenizer
    from tqdm import tqdm
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
    model, args_saved, is_dpr = load_model(args.model, tokenizer, device)

    if is_distributed:
        logger.log('DDP wrapping')
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
            find_unused_parameters=True)
    else:
        logger.log('Single-process single-device, no model wrapping')

    passage_files = sorted(glob.glob(args.passages))
    logger.log(f'passage files identified: {str(passage_files)}')
    start_time = datetime.now()
    model.eval()

    mkdir_optional(args.outdir)
    collate_fn = lambda samples: tensorize_passages(samples, tokenizer,
                                                    args_saved.max_length,
                                                    pad_to_max=is_dpr)
    for passage_file in passage_files:
        dataset = WikiPassageDataset(passage_file)
        logger.log(f'{passage_file} ({len(dataset)} passages)')

        sampler = None if world_size == -1 else \
                  DistributedSampler(dataset, num_replicas=world_size,
                                     rank=rank, shuffle=False)
        loader = DataLoader(dataset, args.batch_size, sampler=sampler,
                            num_workers=args.num_workers, collate_fn=collate_fn)

        pid_seen = {}
        bucket = []
        with torch.no_grad():
            for batch_num, batch in enumerate(tqdm(loader)):
                P, P_mask, P_type, I  = [tensor.to(device) for tensor in batch]
                _, Y = model(P=P, P_mask=P_mask, P_type=P_type)
                if world_size != -1:
                    Y_list = [torch.zeros_like(Y) for _ in range(world_size)]
                    I_list = [torch.zeros_like(I) for _ in range(world_size)]
                    dist.all_gather(tensor_list=Y_list, tensor=Y.contiguous())
                    dist.all_gather(tensor_list=I_list, tensor=I.contiguous())
                    Y = torch.cat(Y_list, 0)
                    I = torch.cat(I_list)

                # Prevent sample redundancy in DDP (last process padded with
                # earliest indices).
                pid_embedding_pairs = list(zip(I.tolist(), Y.cpu().numpy()))
                for pid, embedding in pid_embedding_pairs:
                    if pid not in pid_seen:
                        pid_seen[pid] = True
                        bucket.append([pid, embedding])

            # Sort by pid since DDP may have processed items in diff order.
            bucket.sort(key=lambda x: x[0])

            assert len(bucket) == len(dataset)

            # Sanity check
            pids = [int(sample['pid']) for sample in dataset.samples]
            for i, (pid_bucket, _) in enumerate(bucket):
                assert pid_bucket == pids[i]

            if is_main_process and bucket:
                name = Path(passage_file).stem + f'_encoded.pickle'
                path = os.path.join(args.outdir, name)
                with open(path, 'wb') as f:
                    logger.log(f'Dumping {path} (size {len(bucket)})')
                    pickle.dump(bucket, f)

    logger.log(f'\nDone | total time {strtime(start_time)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str)
    parser.add_argument('passages', type=str, help='regex for psg shards')
    parser.add_argument('outdir', type=str)
    parser.add_argument('--batch_size', type=int, default=3)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--gpus', default='', type=str)

    args = parser.parse_args()

    # Set environment variables before importing libraries that use them!
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    main(args)

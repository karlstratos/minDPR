import argparse
import os


def main(args):
    import faiss
    import glob
    import numpy as np
    import pickle
    import time
    import torch
    import transformers

    from data import DPRDataset, tensorize_questions, WikiPassageDataset
    from datetime import datetime
    from evaluate import has_answer, topk_retrieval_accuracy, print_performance
    from faiss.contrib.exhaustive_search import knn
    from file_handling import write_json
    from model import load_model
    from pathlib import Path
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer
    from tqdm import tqdm
    from util import strtime, get_flat_index, get_faiss_metric

    transformers.logging.set_verbosity_error()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(args)
    print(f'Using device: {str(device)}')
    start_time = datetime.now()

    # Question embeddings
    print(f'Model {args.model}, batch size {args.batch_size}')
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    model, args_saved = load_model(args.model, tokenizer, device)

    print(f'Computing question embeddings from: {args.queries}')
    dataset_queries = DPRDataset(args.queries)
    collate_fn = lambda samples: tensorize_questions(samples, tokenizer,
                                                     args_saved.max_length,
                                                     args_saved.pad_to_max)
    loader = DataLoader(dataset_queries, batch_size=args.batch_size,
                        shuffle=False, num_workers=args.num_workers,
                        collate_fn=collate_fn)

    question_matrix = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader):
            Q, Q_mask, Q_type = [tensor.to(device) for tensor in batch]
            X, _ = model(Q=Q, Q_mask=Q_mask, Q_type=Q_type)
            question_matrix.append(X.cpu().numpy())
    question_matrix = np.concatenate(question_matrix, axis=0)
    num_questions, dim = question_matrix.shape
    print(f'Question matrix: {num_questions} x {dim}')
    del model
    del loader

    # Get an index. Note: faiss hardcodes 8 bits/vec for many PQ indexes
    faiss_metric = get_faiss_metric(args.metric)
    start_time_index = datetime.now()
    if args.index_method == 'Flat':
        index = None if args.skip_index else get_flat_index(dim, args.metric)
    elif args.index_method == 'IVF':
        quantizer = get_flat_index(dim, args.metric_coarse)
        index = faiss.IndexIVFFlat(quantizer, dim, args.num_clusters,
                                   faiss_metric)
        index.nprobe = args.num_probe
    elif args.index_method == 'PQ':  # OPQ bad unless composite index
        factory_string = f'PQ{args.num_subquantizers}x{args.num_bits}'
        if args.use_opq:
            factory_string = f'OPQ{args.num_subquantizers},' + factory_string
        index = faiss.index_factory(dim, factory_string, faiss_metric)
    elif args.index_method == 'IVFPQ':  # OPQ bad unless composite index
        factory_string = f'IVF{args.num_clusters},' + \
                         f'PQ{args.num_subquantizers}x{args.num_bits}'
        if args.use_opq:
            factory_string = f'OPQ{args.num_subquantizers},' + factory_string
        index = faiss.index_factory(dim, factory_string, faiss_metric)
        index.nprobe = args.num_probe
    elif args.index_method == 'HNSW':
        index = faiss.IndexHNSWFlat(dim, args.num_neighbors, faiss_metric)
        index.hnsw.efConstruction = args.num_neighbors_over
        index.hnsw.efSearch = args.num_neighbors_over_search
        index.hnsw.search_bounded_queue = args.bounded_queue
    else:
        raise ValueError('Invalid index method: ' + args.index_method)

    if str(device) == 'cuda' and args.use_faiss_gpu and not args.skip_index:
        # Faiss w/ GPUs seems much faster, but unfortunately it's not an option
        # for all but smallest indices since each GPU stores the entire index.
        ngpus = faiss.get_num_gpus()
        print(f'Using {ngpus} GPUs for faiss index')
        index = faiss.index_cpu_to_all_gpus(index)

    # Passage embeddings
    passage_emb_files = sorted(glob.glob(args.passage_embs))
    print(f'Getting passage embeddings from: {str(passage_emb_files)}')
    i2pid = []
    passage_matrix = []
    giant_passage_matrix = []
    for passage_emb_file in passage_emb_files:
        with open(passage_emb_file, 'rb') as f:
            print('\nLoading', passage_emb_file)
            pairs = pickle.load(f)
            i2pid.extend([pid for pid, _ in pairs])
            passage_matrix = np.concatenate(
                np.expand_dims([emb for _, emb in pairs], axis=0))

            if args.index_method != 'Flat' or args.skip_index:
                giant_passage_matrix.append(passage_matrix)
            else:
                index.add(passage_matrix)  # In flat index, we add incrementally

            del passage_matrix  # Try to reduce unnecessary memory usage

    if args.index_method != 'Flat' or args.skip_index:
        print('\tConcatenating all passage matrices')
        # np.concatenate is memory intensive: ~120G for 21m Wiki embs
        giant_passage_matrix = np.concatenate(giant_passage_matrix, axis=0)
        num_passages, dim = giant_passage_matrix.shape
        print(f'\tGiant passage matrix: {num_passages} x {dim}')

    # Unless we incrementally built a flat, we must now index the giant matrix
    if args.index_method != 'Flat':

        # Furthermore, except HNSW, we need training for quantization
        if args.index_method != 'HNSW':
            print('Training the index')
            start_time_train = datetime.now()
            assert not index.is_trained
            index.train(giant_passage_matrix)
            assert index.is_trained
            print(f'\tDone, training time {strtime(start_time_train)}')

        print(f'\tAdding giant passage matrix to index')
        start_time_add = datetime.now()
        index.add(giant_passage_matrix)  # Bloats memory usage: up to 155G
        print(f'\tDone, adding time {strtime(start_time_add)}')

    if not args.skip_index:
        print(index)  # Print for sanity check
        print(f'Total {index.ntotal} items in passage index, load+index time '
              f'{strtime(start_time_index)}\n')
        print('Trying to release memory for giant passage matrix')
        del giant_passage_matrix  # Now we should have only the index in memory

    # Search
    print(f'Searching')
    start_time_search = datetime.now()
    t0 = time.time()
    if args.index_method == 'Flat' and args.skip_index:
        # Not much better than just building a flat index incrementally: both
        # use 130-140G for building the index, and 87G for the index alone
        print(f'Directly computing KNN without an index')
        scores, ind_matrix = knn(question_matrix, giant_passage_matrix,
                                 args.num_cands, faiss_metric)
    else:  # Example ms/query: ~40 flat, ~0.4 hnsw
        scores, ind_matrix = index.search(question_matrix, args.num_cands)
    t1 = time.time()
    mpq = (t1 - t0) * 1000.0 / num_questions
    print(f'Done, search time {strtime(start_time_search)}')
    print(f'{mpq:7.3f} ms per query\n')

    topks = [[i2pid[ind] for ind in inds] for inds in ind_matrix]
    remapped = [(topks[row], scores[row]) for row in range(len(ind_matrix))]

    print('Trying to release memory for index and question matrix')
    if not args.skip_index:
        del index
    del question_matrix

    # Link pid to text
    print(f'Loading all passages from {args.passages_all}')
    dataset_passages = WikiPassageDataset(args.passages_all)
    results = []
    print(f'{len(dataset_passages)} loaded, preparing {args.num_cands} '
          f'candidates for {len(remapped)} queries')
    for question_num, (topk_pids, topk_scores) in enumerate(remapped):
        question = dataset_queries.samples[question_num]['question']
        answers = dataset_queries.samples[question_num]['answers']
        candidates = [{'id': pid,
                       'text': dataset_passages.samples[pid - 1]['text'],
                       'score': str(score)}  # JSON must be string
                      for pid, score in zip(topk_pids, topk_scores)]
        results.append({'question': question, 'answers': answers,
                        'candidates': candidates })

    print(f'Populating has_answer field in each candidate')
    for result in results:
        for candidate in result['candidates']:
            candidate['has_answer'] = has_answer(candidate, result['answers'],
                                                 mode='string', lower=True)

    write_json(args.outpath, results, verbose=True, pretty=True)

    k_values = sorted([int(k) for k in args.k_values.split(',')])
    k_accuracy = topk_retrieval_accuracy(results, k_values, unnormalized=True)
    print_performance({args.queries: k_accuracy}, k_values)
    print(f'Total time {strtime(start_time)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str)
    parser.add_argument('queries', type=str)
    parser.add_argument('passage_embs', type=str, help='regex for psg embs')
    parser.add_argument('outpath', type=str)
    parser.add_argument('passages_all', type=str)
    parser.add_argument('--index_method', type=str, default='Flat',
                        choices=['Flat', 'IVF', 'PQ', 'IVFPQ', 'HNSW'],
                        help='[%(default)s]')
    parser.add_argument('--metric', type=str, default='IP',
                        choices=['L2', 'IP'], help='[%(default)s]')
    parser.add_argument('--metric_coarse', type=str, default='IP',
                        choices=['L2', 'IP'], help='[%(default)s]')
    parser.add_argument('--k_values', type=str, default='1,5,20,100',
                        help='[%(default)s]')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='[%(default)d]')
    parser.add_argument('--num_cands', type=int, default=100,
                        help='[%(default)d]')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='[%(default)d]')
    parser.add_argument('--num_clusters', type=int, default=1000,
                        help='[%(default)d]')
    parser.add_argument('--num_bits', type=int, default=8,
                        help='[%(default)d]')
    parser.add_argument('--num_subquantizers', type=int, default=16,
                        help='[%(default)d]')
    parser.add_argument('--num_probe', type=int, default=20,
                        help='[%(default)d]')
    parser.add_argument('--num_neighbors', type=int, default=32,
                        help='[%(default)d]')
    parser.add_argument('--num_neighbors_over', type=int, default=500,
                        help='[%(default)d]')
    parser.add_argument('--num_neighbors_over_search', type=int, default=256,
                        help='[%(default)d]')
    parser.add_argument('--use_faiss_gpu', action='store_true')
    parser.add_argument('--skip_index', action='store_true')
    parser.add_argument('--bounded_queue', action='store_true')
    parser.add_argument('--use_opq', action='store_true')
    parser.add_argument('--gpu', default='', type=str)

    args = parser.parse_args()

    # Set environment variables before importing libraries that use them!
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    main(args)

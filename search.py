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
    from file_handling import write_json
    from model import load_model
    from pathlib import Path
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer
    from tqdm import tqdm
    from util import strtime

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

    # Passage embeddings
    start_time_index = datetime.now()
    if args.index_method == 'Flat':
        index = faiss.IndexFlatIP(dim)
    elif args.index_method == 'IVFFlat':
        # https://github.com/facebookresearch/faiss/blob/main/tutorial/python/2-IVFFlat.py
        quantizer = faiss.IndexFlatIP(dim)
        nlist = args.num_clusters
        index = faiss.IndexIVFFlat(quantizer, dim, nlist,
                                   faiss.METRIC_INNER_PRODUCT)

    else:
        raise ValueError('Invalid index method: ' + args.index_method)

    if str(device) == 'cuda' and args.use_faiss_gpu:
        # Faiss w/ GPUs seems much faster, but unfortunately it's not an option
        # for all but smallest indices since each GPU stores the entire index.
        ngpus = faiss.get_num_gpus()
        print(f'Using {ngpus} GPUs for faiss index')
        index = faiss.index_cpu_to_all_gpus(index)

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

            if args.index_method == 'Flat':
                # Incrementally ad one matrix block: much more memory efficient
                print(f'Adding {len(pairs)} vectors to the index')
                index.add(passage_matrix)
            elif args.index_method == 'IVFFlat':
                # Need the entire matrix for clustering
                giant_passage_matrix.append(passage_matrix)

    if args.index_method == 'IVFFlat':
        print('Preparing IVFFlat index')
        print('\tConcatenating all passage matrices')
        giant_passage_matrix = np.concatenate(giant_passage_matrix, axis=0)
        num_passages, dim = giant_passage_matrix.shape
        print(f'\tGiant passage matrix: {num_passages} x {dim}')
        print(f'\tClustering into {args.num_clusters}')
        start_time_cluster = datetime.now()
        assert not index.is_trained
        index.train(giant_passage_matrix)
        assert index.is_trained
        print(f'\tDone, clustering time {strtime(start_time_cluster)}')

        print(f'\tAdding giant passage matrix to index')
        start_time_add = datetime.now()
        index.add(giant_passage_matrix)
        print(f'\tDone, adding time {strtime(start_time_add)}')


    print(f'Total {index.ntotal} items in passage index, index time '
          f'{strtime(start_time_index)}\n')

    # Search
    print(f'Searching')
    start_time_search = datetime.now()
    t0 = time.time()
    scores, ind_matrix = index.search(question_matrix, args.num_cands)
    t1 = time.time()
    mpq = (t1 - t0) * 1000.0 / num_questions
    print(f'Done, search time {strtime(start_time_search)}')
    print(f'{mpq:7.3f} ms per query\n')

    topks = [[i2pid[ind] for ind in inds] for inds in ind_matrix]
    remapped = [(topks[row], scores[row]) for row in range(len(ind_matrix))]

    print('Trying to release some memory')
    del giant_passage_matrix
    del passage_matrix
    del index
    del model
    del loader
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
                        choices=['Flat', 'IVFFlat', 'IVFPQ'])
    parser.add_argument('--k_values', type=str, default='1,5,20,100')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_cands', type=int, default=100)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--num_clusters', type=int, default=100)
    parser.add_argument('--use_faiss_gpu', action='store_true')
    parser.add_argument('--gpu', default='', type=str)

    args = parser.parse_args()

    # Set environment variables before importing libraries that use them!
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    main(args)

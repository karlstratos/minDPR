# python check_data.py data/nq-dev.json
import argparse
import random
import statistics as stat

from file_handling import read_json
from simple_tokenizer import word_tokenize


def main(args):
    random.seed(args.seed)
    examples = read_json(args.data, verbose=True)
    num_same_negs = 0
    num_same_hards = 0

    num_answers = []
    num_positives = []
    num_negatives = []
    num_hards = []

    len_question = []
    len_answer = []
    len_title = []  # only from positive
    len_passage = []  # only from positive

    for example in examples:
        question = example['question']
        answers = example['answers']
        title_positives = [x['title'] for x in example['positive_ctxs']]
        title_negatives = [x['title'] for x in example['negative_ctxs']]
        title_hards = [x['title'] for x in example['hard_negative_ctxs']]
        positives = [x['title'] + ' ||| ' + x['text'][:args.text_size] for x in
                     example['positive_ctxs']]
        negatives = [x['title'] + ' ||| ' + x['text'][:args.text_size] for x in
                     example['negative_ctxs']]
        hards = [x['title'] + ' ||| ' + x['text'][:args.text_size] for x in
                 example['hard_negative_ctxs']]

        num_answers.append(len(answers))
        num_positives.append(len(positives))
        num_negatives.append(len(negatives))
        num_hards.append(len(hards))

        len_question.append(len(word_tokenize(question)))
        len_answer.extend([len(word_tokenize(answer)) for answer in answers])
        len_title.extend([len(word_tokenize(x['title'])) for x in
                          example['positive_ctxs']])
        len_passage.extend([len(word_tokenize(x['text'])) for x in
                            example['positive_ctxs']])

        # Positive passage may be assoc. w/ negative passages from same article,
        # so they have the same title
        if set(title_positives).intersection(title_negatives):
            num_same_negs += 1
        if set(title_positives).intersection(title_hards):
            num_same_hards += 1

        # But they should not be the same content! Only some other passage.
        bad_negatives = set(positives).intersection(negatives)
        bad_hards = set(positives).intersection(hards)
        if bad_negatives:  # Not used in in-batch negative training
            print(f'Warning: bad negatives {str(bad_negatives)}')
        if bad_hards:
            print(f'Warning: bad hards {str(bad_hards)}')

    print('-' * 80)
    print(f'{num_same_negs} / {len(examples)} has rand neg from same entity')
    print(f'{num_same_hards} / {len(examples)} has hard neg from same entity')
    print('-' * 80)
    print('# answers (min, max, avg): {:d} {:d} {:.1f}'.format(
        min(num_answers), max(num_answers), stat.mean(num_answers)))
    print('# positives (min, max, avg): {:d} {:d} {:.1f}'.format(
        min(num_positives), max(num_positives), stat.mean(num_positives)))
    print('# negatives (min, max, avg): {:d} {:d} {:.1f}'.format(
        min(num_negatives), max(num_negatives), stat.mean(num_negatives)))
    print('# hards (min, max, avg): {:d} {:d} {:.1f}'.format(
        min(num_hards), max(num_hards), stat.mean(num_hards)))
    print('-' * 80)
    print('# tokens in question: {:.1f}'.format(stat.mean(len_question)))
    print('# tokens in answer: {:.1f}'.format(stat.mean(len_answer)))
    print('# tokens in title (pos): {:.1f}'.format(stat.mean(len_title)))
    print('# tokens in passage (pos): {:.1f}'.format(stat.mean(len_passage)))


    for _ in range(args.sample_size):
        print('-' * 80)
        rand = examples[random.randrange(len(examples))]
        print('Q:', rand['question'])
        print('A:', rand['answers'])
        p0 = rand['positive_ctxs'][0] if rand['positive_ctxs'] else None
        n0 = rand['negative_ctxs'][0] if rand['negative_ctxs'] else None
        h0 = rand['hard_negative_ctxs'][0] if rand['hard_negative_ctxs'] else \
             None
        if p0:
            print('p0:', p0['title'] + ' ||| ' + p0['text'][:args.text_size])
        if n0:
            print('n0:', n0['title'] + ' ||| ' + n0['text'][:args.text_size])
        if h0:
            print('h0:', h0['title'] + ' ||| ' + h0['text'][:args.text_size])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data', type=str, help='DPR retriever training data')
    parser.add_argument('--text_size', type=int, default=200,
                        help='# chars to show in text examples')
    parser.add_argument('--sample_size', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    main(args)

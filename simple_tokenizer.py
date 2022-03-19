# Adapted from https://github.com/facebookresearch/DrQA
import regex


class SimpleTokenizer:
    ALPHA_NUM = r'[\p{L}\p{N}\p{M}]+'  # Letter, number, combining mark
    NON_WS = r'[^\p{Z}\p{C}]'

    def __init__(self):
        self._regexp = regex.compile(
            '(%s)|(%s)' % (self.ALPHA_NUM, self.NON_WS),
            flags=regex.IGNORECASE | regex.UNICODE | regex.MULTILINE
        )

    def tokenize(self, text):
        tuples = []
        matches = [m for m in self._regexp.finditer(text)]
        for i in range(len(matches)):
            token = matches[i].group()  # Text
            s, t = matches[i].span()
            if i + 1 < len(matches):
                t = matches[i + 1].span()[0]
            tuples.append((token, text[s: t], (s, t)))
        return Tokens(tuples)


SIMPLE_TOKENIZER = SimpleTokenizer()


def word_tokenize(string, lower=False):
    return SIMPLE_TOKENIZER.tokenize(string).words(lower=lower)


class Tokens:
    """A class to represent a list of tokenized text."""
    TEXT = 0
    TEXT_WS = 1
    SPAN = 2

    def __init__(self, tuples):
        self.tuples = tuples

    def __len__(self):
        return len(self.tuples)

    def words(self, lower=False):
        return [t[self.TEXT].lower() for t in self.tuples] if lower else \
            [t[self.TEXT] for t in self.tuples]

    def untokenize(self):
        return ''.join([t[self.TEXT_WS] for t in self.tuples]).strip()

    def offsets(self):
        """Returns a list of [start, end) character offsets of each token."""
        return [t[self.SPAN] for t in self.tuples]

    def slice(self, i, j):
        """Return a view of the list of tokens from [i, j)."""
        new_tokens = copy.copy(self)  # Shallow copy: new obj but shared refs
        new_tokens.tuples = self.tuples[i: j]
        return new_tokens


    def ngrams(self, n=1, lower=False, filter_fn=None, as_strings=True):
        """Returns a list of all ngrams from length 1 to n.
        Args:
            n: upper limit of ngram length
            lower: lowercases text
            filter_fn: user function that takes in an ngram list and returns
              True or False to keep or not keep the ngram
            as_string: return the ngram as a string vs list
        """

        def _skip(gram):
            if not filter_fn:
                return False
            return filter_fn(gram)

        words = self.words(lower)
        ngrams = [(s, e + 1) for s in range(len(words))
                  for e in range(s, min(s + n, len(words)))
                  if not _skip(words[s:e + 1])]

        if as_strings:
            ngrams = ['{}'.format(' '.join(words[s:e])) for (s, e) in ngrams]

        return ngrams

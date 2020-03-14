import sys
from tokenise_file import tokenise


def ngrams(line):
    out = []
    line = tokenise(line).split()
    ngram = 1
    while ngram < 5:
        all_grams = []
        i = 0
        while i + ngram <= len(line):
            all_grams.append(" ".join(line[i:i + ngram]))
            i += 1
        out.append(all_grams)
        ngram += 1
    return out


if __name__ == "__main__":
    for ngram in enumerate(ngrams(sys.stdin.readline())):
        print("{}-gram: {}".format(ngram[0] + 1, ngram[1]))

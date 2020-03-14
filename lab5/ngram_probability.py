from lab3 import ngrams
from frequency import get_ngram_occurrences


def ngram_probability(sentence, corpus_location, ngram=1):
    corpus_ngrams = []
    for n in range(ngram):
        corpus_ngrams.append(get_ngram_occurrences(corpus_location, n + 1))
    sentence_ngrams = ngrams.ngrams(sentence)[:ngram]
    probabilities = [1] * ngram

    for n in range(ngram):
        i = 0
        ngram_tokens = sentence_ngrams[n]
        if n > 0:
            mgram_tokens = sentence_ngrams[n - 1]
        else:
            mgram_tokens = None

        while i < len(ngram_tokens):
            ngram_token = ngram_tokens[i]
            ngram_occurrences = 1

            if ngram_token in corpus_ngrams[n]:
                ngram_occurrences += corpus_ngrams[n][ngram_token]

            mgram_occurrences = None
            if mgram_tokens:
                mgram_token = mgram_tokens[i + 1]
                mgram_occurrences = len(corpus_ngrams[n - 1])
                if mgram_token in corpus_ngrams[n - 1]:
                    mgram_occurrences += corpus_ngrams[n - 1][mgram_token]

            if mgram_occurrences:
                probabilities[n] *= ngram_occurrences / mgram_occurrences
            else:
                total_tokens = sum(corpus_ngrams[n].values())
                probabilities[n] *= ngram_occurrences / total_tokens
            i += 1

    return probabilities


if __name__ == "__main__":
    print(ngram_probability(input("Sentence: "), "../corpus/lab5-corpus.txt", 4))

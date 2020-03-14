from lab3 import ngrams
from frequency import get_ngram_occurrences


def bigram_probability(sentence, corpus_location="../corpus/train-data.en"):
    unigram_occurrences = get_ngram_occurrences(corpus_location)
    bigram_occurrences = get_ngram_occurrences(corpus_location, 2)
    sentence_unigrams, sentence_bigrams = ngrams.ngrams(sentence)[:2]
    probability = 1

    i = 0
    while i < len(sentence_bigrams):
        bigram = sentence_bigrams[i]
        unigram = sentence_unigrams[i + 1]
        numerator = 1
        denominator = len(unigram_occurrences)
        if bigram in bigram_occurrences:
            numerator += bigram_occurrences[bigram]
        if unigram in unigram_occurrences:
            denominator += unigram_occurrences[unigram]
        probability *= numerator / denominator
        i += 1

    return probability


if __name__ == "__main__":
    print(bigram_probability(input("Sentence: "), "../corpus/lab5-corpus.txt"))

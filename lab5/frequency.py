from lab3 import ngrams


def calculate_frequencies(corpus, ngram):
    token_frequencies = get_ngram_occurrences(corpus, ngram)

    total_tokens = sum(token_frequencies.values())
    for token in token_frequencies:
        token_frequencies[token] = token_frequencies[token] / total_tokens

    return token_frequencies


def calculate_frequencies_file(corpus_location="../corpus/train-data.en",
                               ngram=1):
    with open(corpus_location, "r") as corpus:
        calculate_frequencies(corpus, ngram)


def get_ngram_occurrences(corpus, ngram=1):
    all_occurrences = {}
    for line in corpus:
        tokens = ngrams.ngrams(line)[ngram - 1]
        get_token_occurrences(tokens, all_occurrences)

    return all_occurrences


def get_token_occurrences(sentence_tokens, occurrences=None):
    if occurrences is None:
        occurrences = {}

    for token in sentence_tokens:
        if token in occurrences:
            occurrences[token] += 1
        else:
            occurrences[token] = 1

    return occurrences


if __name__ == "__main__":
    print(calculate_frequencies_file())

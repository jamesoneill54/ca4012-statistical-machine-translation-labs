from lab2 import tokenise_file
from frequency import calculate_frequencies


def unigram_probability(sentence, corpus_location="../corpus/train-data.en"):
    tokens = tokenise_file.tokenise(sentence).split()
    frequencies = calculate_frequencies(corpus_location)
    probability = 1

    for token in tokens:
        try:
            probability *= frequencies[token]
        except KeyError:
            print("[ERROR]: token \"{}\" not in corpus.".format(token))

    return probability


if __name__ == "__main__":
    user_sentence = input("Sentence: ")
    print(unigram_probability(user_sentence))

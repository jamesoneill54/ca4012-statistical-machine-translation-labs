import statistics
import math
from ngrams import ngrams


def get_bleu_score(translation, references):
    brevity_penalty = calculate_brevity_penalty(translation, references)
    ngram_overlap = get_ngram_overlap(translation, references)
    print("Brevity: {}\t|\tN-Gram Overlap: {}".format(brevity_penalty, ngram_overlap))
    return brevity_penalty * ngram_overlap


def calculate_brevity_penalty(translation, references):
    mean_ref_len = statistics.mean([len(ngrams(ref)[0]) for ref in references])
    mean_ref_len += 0.5
    return min(1, (len(ngrams(translation)[0])/math.floor(mean_ref_len)))


def match_ngrams(trans_ngrams, multi_ref_ngrams):
    matched = []
    for ngram in trans_ngrams:
        i = 0
        found_ngram = False
        while not found_ngram and i < len(multi_ref_ngrams):
            if ngram in multi_ref_ngrams[i]:
                matched.append(ngram)
                found_ngram = True
            i += 1
    return matched


def get_ngram_overlap(translation, references):
    trans_ngrams = ngrams(translation)
    all_references = [[] for ignored in range(len(trans_ngrams))]
    for reference in references:
        ref_ngrams = ngrams(reference)
        i = 0
        while i < len(ref_ngrams):
            all_references[i].append(ref_ngrams[i])
            i += 1
    product = 1
    i = 0
    while i < len(trans_ngrams):
        matched = match_ngrams(trans_ngrams[i], all_references[i])
        product = product * (len(matched) / len(trans_ngrams[i]))
        i += 1
    return product ** 0.25


if __name__ == "__main__":
    #  tran = input("translation: ")
    #  ref = input("reference: ")
    #  print(get_bleu_score(tran, ref))
    print(get_bleu_score("The gunman was shot dead by police.",
                         ["The gunman was shot to death by the police.",
                          "The gunman was shot to death by the police.",
                          "Police killed the gunman.",
                          "The gunman was shot dead by the police."]))


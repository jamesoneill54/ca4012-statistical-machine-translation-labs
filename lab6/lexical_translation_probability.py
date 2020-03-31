from lab3.tokenise_file import tokenise


def initialise_pairs(sources, targets):
    unique_tgt_tokens = get_unique_tokens(sources)
    unique_src_tokens = get_unique_tokens(targets)
    translation_prob = {}
    for tgt_token in unique_tgt_tokens:
        for src_token in unique_src_tokens:
            translation_prob[(src_token, tgt_token)] = 1 / len(unique_tgt_tokens)
    return translation_prob


def get_unique_tokens(sentences):
    unique_tokens = set()
    for sentence in sentences:
        tokens = tokenise(sentence).split()
        unique_tokens.update(tokens)
    return unique_tokens


def compute_normalisation(pair, translation_probabilities):
    tgt = pair[0].split()
    src = pair[1].split()
    token_totals = {}
    for tgt_token in tgt:
        token_totals[tgt_token] = 0
        for src_token in src:
            token_totals[tgt_token] += translation_probabilities[(src_token,
                                                                 tgt_token)]
    return token_totals


def collect_counts(pair, sentence_totals, translation_probabilities):
    tgt = pair[0].split()
    src = pair[1].split()
    counts = {}
    totals = {}
    for tgt_token in tgt:
        for src_token in src:
            count = translation_probabilities[(src_token, tgt_token)] / \
                     sentence_totals[tgt_token]
            if (src_token, tgt_token) in counts:
                counts[(src_token, tgt_token)] += count
            else:
                counts[(src_token, tgt_token)] = count
            if src_token in totals:
                totals[src_token] += count
            else:
                totals[src_token] = count
    return counts, totals


def lexical_translation_prob(sources, targets, iterations):
    pairs = [(tokenise(sources[i]), tokenise(targets[i]))
             for i in range(len(sources))]
    translation_probabilities = initialise_pairs(sources, targets)
    for _ in range(iterations):
        counts = {}
        totals = {}
        for pair in pairs:
            sentence_totals = \
                compute_normalisation(pair, translation_probabilities)
            pair_counts, pair_totals = collect_counts(pair, sentence_totals,
                                                      translation_probabilities)
            for p in pair_counts:
                if p in counts:
                    counts[p] += pair_counts[p]
                else:
                    counts[p] = pair_counts[p]
            for p in pair_totals:
                if p in totals:
                    totals[p] += pair_totals[p]
                else:
                    totals[p] = pair_totals[p]
        for src_token, tgt_token in translation_probabilities.keys():
            translation_probabilities[(src_token, tgt_token)] = \
                counts[(src_token, tgt_token)] / totals[src_token]
        print(translation_probabilities)
    return 0


if __name__ == "__main__":
    lexical_translation_prob(["the house", "house"], ["la maison", "maison"], 2)

# ID: 16410652
# Name: James O'Neill

def calculate_alignment_probability(f, e, a, T, epsilon=1.0):
    ''' Calculates the probability of a sentence given the alignment, the translation
        probabilities and a normalization constant

        :param f: the foreign sentence (string)
        :param e: the target sentence (string)
        :param a: the alignment (dictionary)
        :param T: the translation probabilities (dictionary)
        :param epsilon: normalization constant (float, default = 1.0)
        :returns: the probability of the sentence (float)
    '''

    # Firstly calculate the fractional counts.
    fractional_counts = [T[(src, a[src])] for src in a]

    # Then multiply all fractional counts together to get the initial alignment probability
    alignment_prob = 1
    for prob in fractional_counts:
        alignment_prob *= prob

    # Then apply normalisation using the normalisation constant
    alignment_prob *= (epsilon / ((len(f.split()) + 1) ** len(e.split())))
    return alignment_prob

# Do not modify the following lines
def main():
    a = {'wo': 'am', 'zai': 'i', 'xuexi': 'studying'}
    T={('wo', 'am'): 0.5, ('ziji', 'i'): 0.2, ('wo', 'me'): 0.3, ('ziji', 'me'): 0.1, ('zai', 'i'): 0.3, ('shi', 'am'): 0.5, ('zai', 'is'): 0.2, ('shi', 'is'): 0.5, ('xuexi', 'study'): 0.6, ('yanjiu', 'study'): 0.3, ('xuexi', 'studying'): 0.5, ('yanjiu', 'studying'): 0.5}
    f = 'i am studying'
    e = 'wo zai xuexi'

    p_a = calculate_alignment_probability(f, e, a, T)

    print(p_a)


if __name__ == "__main__":
    main()
# ID: 16410652
# Name: James O'Neill

def calculate_translation_probability(c):
    ''' Calculates the translation probabilities given the fractional counts

        :param c: fractional counts (dictionary)
        :returns: translation probabilities (dictionary)
    '''

    # Must divide the fractional counts by the sum of the fractional counts
    # where the target token occurs.

    probabilities = {}
    for source_token, target_token in c:
        target_occurrences_counts = 0
        for count in c:
            # check if target token occurs in this fractional count.
            if count[1] == target_token:
                # add this fractional count to all fractional counts where the
                # target occurs.
                target_occurrences_counts += c[count]
        # now all fractional counts with occurrences of the target token have
        # been summed, divide the current count by the total.
        probabilities[(source_token, target_token)] = c[(source_token, target_token)] / target_occurrences_counts

    return probabilities

# Do not modify the following lines:
def main():
    c = {('la', 'the'): 0.5, ('maison', 'the'): 0.5, ('la', 'house'): 0.5, ('maison', 'house'): 1.5}

    t = calculate_translation_probability(c)

    print(t)


if __name__ == "__main__":
    main()

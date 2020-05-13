# 2019 Paper

## Section A

### Question 1 

> **(a)** Any statistical approach to MT requires the availability of aligned bilingual corpora which are (i) large, (ii) good-quality, and (iii) representative. Explain why all three requirements are important.

**(i)** A corpus needs to be large because it needs to contain enough data to adequately gather statistics on the words used in the language. Word counts and word contexts in sentences need to be gathered to adequately translate an unseen sentence, and this translation cannot occur if many of the words in the unseen sentence are not present in the corpus. 

**(ii)** A corpus needs to contain good-quality bilingual translations, as it will be using this to train its language model and predict good translations for unseen sentences. If the corpus is not of good-quality, then fluency and adequacy of the resulting translation of the unseen sentence will suffer.

**(iii)** A corpus needs to be representative of the language context because the context of the sentence to be translated can lead to many differing translations. For example, if the corpus contains bilingual data from legal documents, then it should only be used to translate sentences in a legal context. So if a corpus is not representative of the language context it is supposed to translate, it will lead to inadequate translations. 

> **(b)** Provide the fundamental equations of (i) the noisy channel model of SMT, and (ii) the log-linear model of SMT. With reference to these equations, name the different components in both models, and describe their basic function.

**(i)** The noisy channel model of SMT can be described by the following function:
```
e = argmax p(e) p(f|e)
```

This function comprises three components which are: 
1. The Decoder (The `argmax` part of the formula). This part of the formula involves searching for the best translation among all possible translations.
2. The Language Model (The `p(e)` part of the formula). This part of the formula calculates how probable the translation (given by the decoder) is to being a fluent translation; that is, how correct the translation is from a fluency standpoint. The language model is built from a corpus of the target language, and is not bilingual. 
3. The Translation Model (The `p(f|e)` part of the formula). This part of the formula calculates how probable the translation (given by the decoder) is to being an acceptable sentence; that is, how likely that `f` is a translation of `e`. The translation model is built from a corpus of bilingual translations from the source language to the target language. 

**(ii)** The log-linear model of SMT is an extension of the phrase-based model that allows for extensions on the three models included in the phrased based model, which are:
1. The Translation Model.
2. The Reordering/Distance-Based Model.
3. The Language Model.

The log linear model allows for:
- Multiple translation models.
- Multiple language models.
- Multiple linguistic features.
- Multiple lexical probabilities and phrase probabilities.

The log-linear model also allows for each of the components in the model to be weighted. This allows us to specify which models should have a greater influence on the resulting translation. 

## Section B

### Question 3

> **(a)** Explain the Markov assumption. Why do we need to take it into account when building n-gram language models?

The Markov Assumption is that only previous history matters, and it is limited to a specific distance in the history, ie. the further in the past a word is, the less likely it is to affect the translation of the current word. 

> **(b)** How is the Maximum Likelihood estimate of a trigram language model computed? Compute P(ate|Bukka) from the following unigram and bigram counts.

The maximum likelihood estimate is computed by getting the count of the amount of times a trigram appears in the corpus, and dividing that number by the amount of times the bigram appears in the corpus. Given by the function: `count(w1, w2, w3) / count(w1, w2)`

```
P(ate|Bukka) = count(Bukka ate) / count(Bukka)
             = 16/35
```

> **(c)** Assume (i) a unigram and (ii) a bigram language model trained on standard English. Based on these two different language models, how would you expect the probability of the phrase “the sandwich Hakka ate” to compare to the probability of the sentence “Hakka ate the sandwich”?

(i) With a unigram language model, the probability of the phrases would be the same for both sentences. This is because unigram language models do not take into account word order. Unigram language models only focus on one word at a time, and determine a phrase's probability using its frequency in the corpus. No matter the order of the words in the phrase, if they both contain the same words, then the overall phrase probabilities will be the same for both phrases. 

(ii) On the other hand, a bigram language model does take into account the word order. Instead of getting the frequencies of each individual word in the corpus and multiplying them together to get the phrase probability, it gets the frequencies of word pairs in the corpus instead. This means that word order is preserved. In the sample sentences, the second phrase would have a higher probability than the first phrase. This is because word pair "sandwich Hakka" would show up far less often than the other word pair that the other phrase contains, which is "ate the". This results in a higher probability for the second phrase. 

> **(d)** Why do n-gram language models need to be smoothed? Name three methods of smoothing, and explain one method in detail.

N-gram models need to be smoothed in order to handle unseen words or n-grams appropriately. The unseen n-grams must be assigned a non-zero probability, otherwise a sentence which contains an unseen n-gram but has other seen n-grams will get assigned a probability of zero, just because of the one unseen n-gram. 

Three methods of smoothing are:
1. Count adjustment
2. Interpolation
3. Back-off

**Back-off:** Back-off involves trusting the highest order language model which contains the n-gram. This means that, given trigram, bigram and unigram language models, firstly start with the trigram count and use the probability from that model if possible. Otherwise, move on to the bigram count, and then the unigram count if the bigram count does not work. 

### Question 4

> **(a)** In decoding, what is pruning, and why it is important? How many of the following hypotheses will be pruned if we prune all hypotheses that are at least 0.4 times worse than the best hypothesis? Show your calculations.

Pruning involves considering a certain fraction of the hypotheses in the search space. The aim is to remove bad hypotheses early. There are two different methods of pruning; 
- Histogram pruning, where only the top `n` hypotheses are considered. 
- Threshold pruning, where only hypotheses that are within a certain distance from the best hypothesis are considered. 

For the hypotheses 0.02, 0.03, 0.06 and 0.08, anything below 0.048 would be pruned from the search space. Workings:
```
0.08 * 0.4 = 0.032

Any hypothesis with a value below 0.048 will be pruned. 
```

> **(b)** Assume the following partial phrase table:
>```
> se                she             0.4
> bhalobase         likes           0.3
> bhalobase         likes to        0.5
> se bhalobase      he likes        0.3
> se bhalobase      he likes to     0.6
> khete             eat             0.6
> khete             eating          0.7
> khete bhalobase   likes eating    0.3
> khete bhalobase   likes to eat    0.7    
>```          
> Assume that only monotone word order is permitted and that the language model is ignored. 
>
> Draw the search graph constructed during decoding for the sentence **“se bhalobase khete”**.

> **(c)** Using the constructed search graph for Q 4(b), calculate all possible hypotheses, and indicate which hypothesis provides the optimal translation for the sentence above.

The hypothesis that provides the optimal translation is "he likes to" -> "eating", which has a score of 0.42. 

### Question 5

> **(a)** Explain Word Error Rate (WER). How is the WER score computed?

Word Error Rate is the number of insertions, deletions and substitutions required to transform an MT system's output translation into the reference translation. The WER is computed by the following function: 
```
    (insertions + deletions + substitutions) / reference_length
```

> **(b)** Given the following two candidate translations (from MT system 1 and from MT system 2), compute their WER scores with respect to the reference provided.        
> - Candidate MT system 1: near the shore the ship sank
> - Candidate MT system 2: the big ship sank close to the shore
> - Reference: the large ship sank near the shore

- System 1 WER: `7/7 = 1`
- System 2 WER: `3/7 = 0.42857...`
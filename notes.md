# CA4012 - Statistical & Neural Machine Translation Notes

## Contents
1. [Introduction](#introduction)
2. [SMT & NMT Evaluation](#smt--nmt-evaluation)
3. [Decoding](#decoding)

## Introduction

- SMT is mostly all about the data.
- SMT uses conditional probability to determine likely words that follow a certain other word. 
- SMT learns from two sources: 
    1. Source documents and their human translations.
    2. Target language collections. 
- The type of document matters; if the machine has been given news articles, it will be good at translating news articles and bad at other document types. 
- **Word Alignment:** Involves deducing a word's translation using pattern matching techniques. 
- Word-based translation falls victim to incorrect translations when dealing with context-sensitive translations/words. 
- Parallel corpora is important, but not available in a wide range of language pairs. 
- Any statistical approach to MT requires availability of parallel corpora which are:
    1. large,
    2. good quality,
    3. representative (of the language). 
- Context of the source data can skew the language, and can cause it to be non-representative. 
- **Translation Model:** Models how likely it is that `f` is a translation of `e` - **Adequacy** 
    - Developed from a parallel data. 
- **Language Model:** Models how likely it is that `e` is an acceptable sentence - **Fluency**
    - Developed from a monolingual data.
- The translation model and language model make up two of the three parts of an SMT, the third being the **decoder.**
    ```
    (1) Decoder -> (2) Language Model -> (3) Translation Model
    ```
  
## SMT & NMT Evaluation

- Quality of a translation is ambiguous, and varies for different situations and changes for different users. 
- Goals for MT Evaluation:
    - Meaningful
    - Consistent
    - Correct
    - Low Cost
    - Tunable
- Other issues include:
    - Speed
    - Size
    - Integration
    - Customisation
- Evaluation uses **Adequacy** and **Fluency**.
    - **Adequacy:** How well the meaning has been translated.
    - **Fluency:** The flow and naturalness of the translated sentence.
- **Automatic evaluation** requires reference sentences from human translators. 
- **Word Error Rate (WER)** is a basic automatic evaluation method. Counts insertions, deletions, and substitutions and divides them by the reference length:
    ```
    (<Insertions> + <substitutions> + <deletions>) / <reference length>
    ```
    - WER does not account for shifts, so if MT output has the correct word but in a different position, then it counts as a deletion and an insertion. 
- **BLEU** does n-gram matching, comparing a number of words together against the same number of words in the MT translation. 
    ```
    'the cat sat on the mat'
    2-gram/bigram  = ['the cat', 'cat sat', 'sat on', 'on the', 'the mat']
    3-gram/trigram = ['the cat sat', 'cat sat on', 'sat on the', on the mat']
    ```
    - BLEU gives preference to more verbose translations, applying a brevity penalty if the MT is shorter than the reference. 
- **METEOR** gives a score for synonyms (partial credit).
- **Criticism of Automatic Translation:**
    - Ignores relevant words.
    - Operates on local level (doesn't consider overall grammar).
    - Is like a 'black box'.
- **Advantages of Automatic Translation:**
    - Cheap.
    - Consistent score for same translation. 
- **Task-Based Evaluation (Post-Editing):**
    - Assesses usefulness of MT system in production.
    - Identify common errors.
    - Creates new training/test data. 

## Decoding

### Distance-Based Reordering

- Calculates how far a *phrase* in the MT is from its position in the reference sentence. 
- The reordering function penalises reordering over long distances. 
- Reordering function: 
    ```
    i = the phrase number
    f = the phrase
  
    start_i - end_i-1 - 1 = distance
       ^         ^
       |       The position of the last word in the previous phrase
     The position of the first word in the current phrase
    ```

### Log-Linear Model

- Phrase-based model is comprised of three sub-models:
    1. The Translation Model,
    2. The Reordering/Distance-Based Model, and
    3. The Language Model.
- Weights are added to the individual models to allow scaling of the three components.
- The log-linear model accounts for the weightings of the components, and is defined as:
    ```
    p(x) = exp((Σn,i=1)weighting_i * component_i(x))
  
    Where:
    - n = 3
    - component_1(x) = log component 1
    - component_2(x) = log component 2
    - component_3(x) = log component 3
    ```
- Log-linear model makes it easier to add more information sources: 
    - Multiple translation models (n-many models)
    - Multiple language models (n-many models)
    - Multiple linguistic features
    - Multiple lexical probabilities and phrase probabilities


### Decoding

> The process of searching for the best translation among all possible translations. 

- Two types of errors in the decoding process:
    1. The most probable translation is bad - SOLUTION: fix the model
    2. The search does not find the most probable translation - SOLUTION: fix the search
- Decoding is evaluated by *search error*, not quality of translations.
- **Decoding process:**
    1. Select phrase to be translated
    2. Find phrase translation
    3. Add phrase to end of partial translation
    4. Mark phrase as translated
- The decoding process moves from left to right, and can handle one-to-many and many-to-one translation. 
- The decoding process also handles reordering.
- The decoding process keeps track of the score of the translation.
- **Translation Options:**
    - There are many different ways to segment words into phrases.
    - Many different ways to translate each phrase. 
    - Decoding considers the top matching ngrams for varying phrase lengths. 
- **Hypothesis Expansion:**
    - Starting at the null hypothesis.
    - Chooses a number of possible translations to serve as the next hypothesis.
    - Continue to build on the number of hypotheses until the end is reached.
    - Choose the set of translations that have the highest score. 

### Making Decoding Manageable

- The decoding problem is NP-complete, which means that exhaustively examining all possible translations, scoring them and picking the best one is too computationally heavy. 
- **Management Strategies:** 
    1. Hypothesis Recombination (risk-free).
    2. Pruning the search space (risky).

**Hypothesis Recombination:**
- A translation hypothesis is a partial translation.
- The same partial translation can be achieved in more than one way. 
- Hypothesis recombination takes advantage of this by storing only the most likely path associated with a particular hypothesis. 
- When the same partial translation is achieved by two separate methods, the less likely hypothesis is removed (Worse hypothesis is dropped).

**Pruning The Search Space:**
- Pruning removes bad hypotheses early. 
- It puts comparable hypotheses into a stack, and limits the number of hypotheses in each stack. 
- The hypotheses are stored in stacks based on the number of words translated. 
- Unlikely hypotheses are pruned from the stack. 
- Pruning Strategies:
    1. Histogram Pruning - Keep a maximum of `m` hypotheses in a stack. (eg. keep only the top 4 hypotheses.)
    2. Threshold or Beam Pruning - Keep only those hypotheses that are within a threshold `a` of the best hypothesis (`a * best_score(a < 1)`). Any hypothesis that is `a` times worse than the best is pruned. (eg. keep only hypotheses that have a probability of 0.5 or higher.)
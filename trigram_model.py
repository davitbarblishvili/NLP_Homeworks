import sys
from collections import defaultdict
import math
import random
import os
import os.path
from typing import final
"""
COMS W4705 - Natural Language Processing
Homework 1 - Programming Component: Trigram Language Models
Yassine Benajiba
"""

TRIGRAM = 3
BIGRAM = 2
UNIGRAM = 1


def corpus_reader(corpusfile, lexicon=None):
    with open(corpusfile, 'r') as corpus:
        for line in corpus:
            if line.strip():
                sequence = line.lower().strip().split()
                if lexicon:
                    yield [word if word in lexicon else "UNK" for word in sequence]
                else:
                    yield sequence


def get_lexicon(corpus):
    word_counts = defaultdict(int)
    for sentence in corpus:
        for word in sentence:
            word_counts[word] += 1
    return set(word for word in word_counts if word_counts[word] > 1)


def get_ngrams(sequence, n):
    """
    COMPLETE THIS FUNCTION (PART 1)
    Given a sequence, this function should return a list of n-grams, where each n-gram is a Python tuple.
    This should work for arbitrary values of 1 <= n < len(sequence).
    """
    ngrams_arr = []
    start_array = ['START'] * (n-1) if n > 2 else ['START']
    final_array = start_array + sequence + ['STOP']

   # library_ngrams = ngrams(start_array, n) // using nltk ngram
    for i in range(len(final_array)-n+1):
        ngrams_arr.append(tuple(final_array[i:i+n]))
    return ngrams_arr


class TrigramModel(object):

    def __init__(self, corpusfile):

        # Iterate through the corpus once to build a lexicon
        generator = corpus_reader(corpusfile)
        self.lexicon = get_lexicon(generator)
        self.lexicon.add("UNK")
        self.lexicon.add("START")
        self.lexicon.add("STOP")
        self.computed_unigram_denominator = False

        # Now iterate through the corpus again and count ngrams
        generator = corpus_reader(corpusfile, self.lexicon)
        self.count_ngrams(generator)

    def count_ngrams(self, corpus):
        """
        COMPLETE THIS METHOD (PART 2)
        Given a corpus iterator, populate dictionaries of unigram, bigram,
        and trigram counts. 
        """

        self.unigramcounts = defaultdict(int)
        self.bigramcounts = defaultdict(int)
        self.trigramcounts = defaultdict(int)

        for sentence in corpus:
            unigrams = get_ngrams(sentence, UNIGRAM)
            bigrams = get_ngrams(sentence, BIGRAM)
            trigrams = get_ngrams(sentence, TRIGRAM)

            for each_unigram in unigrams:
                self.unigramcounts[each_unigram] += 1

            for each_bigram in bigrams:
                self.bigramcounts[each_bigram] += 1

            for each_trigram in trigrams:
                self.trigramcounts[each_trigram] += 1

                if each_trigram[:2] == ('START', 'START'):
                    self.bigramcounts[('START', 'START')] += 1

        # print(self.trigramcounts[('START', 'START', 'the')])  # returns 5478
        return

    def raw_trigram_probability(self, trigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) trigram probability
        """
        #count(u,v,w)/ count(u,v)
        u_v_count = self.bigramcounts[trigram[:2]]
        if u_v_count != 0:
            return self.trigramcounts[trigram]/u_v_count
        else:
            return 0.0

    def raw_bigram_probability(self, bigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) bigram probability
        """
        # count(u,w) / count(u)
        u_count = self.unigramcounts[bigram[:1]]
        if u_count != 0:
            return self.bigramcounts[bigram]/u_count
        else:
            return 0.0

    def raw_unigram_probability(self, unigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) unigram probability.
        """
        if not self.computed_unigram_denominator:
            self.unigram_sum = sum(self.unigramcounts.values())
            self.computed_unigram_denominator = True
            # ignoring STOP and START count but I do not know if we need this line or not
            # self.unigram_sum -= - self.unigramcounts[('START',)] + self.unigramcounts[('STOP',)]

        return self.unigramcounts[unigram]/self.unigram_sum

        # hint: recomputing the denominator every time the method is called
        # can be slow! You might want to compute the total number of words once,
        # store in the TrigramModel instance, and then re-use it.

    def generate_sentence(self, t=20):
        """
        COMPLETE THIS METHOD (OPTIONAL)
        Generate a random sentence from the trigram model. t specifies the
        max length, but the sentence may be shorter if STOP is reached.
        """
        return result

    def smoothed_trigram_probability(self, trigram):
        """
        COMPLETE THIS METHOD (PART 4)
        Returns the smoothed trigram probability (using linear interpolation). 
        """
        lambda1 = 1/3.0
        lambda2 = 1/3.0
        lambda3 = 1/3.0

        smoothed_trigram_prob = 0.0
        smoothed_trigram_prob += lambda1 * \
            self.raw_trigram_probability(trigram)
        smoothed_trigram_prob += lambda2 * self.raw_bigram_probability(
            trigram[1:])
        smoothed_trigram_prob += lambda3 * \
            self.raw_unigram_probability(trigram[2:])
        return smoothed_trigram_prob

    def sentence_logprob(self, sentence):
        """
        COMPLETE THIS METHOD (PART 5)
        Returns the log probability of an entire sequence.
        """
        log_probability = 0.0
        ngrams = get_ngrams(sentence, TRIGRAM)
        for trigram in ngrams:
            smoothed_tri_prob = self.smoothed_trigram_probability(trigram)
            log_probability += math.log2(smoothed_tri_prob)

        return log_probability

    def perplexity(self, corpus):
        """
        COMPLETE THIS METHOD (PART 6) 
        Returns the log probability of an entire sequence.
        """
        sum_log_prob = 0
        total_words = 0
        for each_sentence in corpus:
            sentence_log_prob = self.sentence_logprob(each_sentence)
            sum_log_prob += sentence_log_prob
            total_words += len(each_sentence)

        l = sum_log_prob/total_words
        return 2**(-l)


def essay_scoring_experiment(training_file1, training_file2, testdir1, testdir2):

    model1 = TrigramModel(training_file1)
    model2 = TrigramModel(training_file2)

    total = 0
    correct = 0

    for f in os.listdir(testdir1):
        pp = model1.perplexity(corpus_reader(
            os.path.join(testdir1, f), model1.lexicon))
        pp_model2 = model2.perplexity(corpus_reader(
            os.path.join(testdir1, f), model2.lexicon))
        total += 1
        correct += (pp < pp_model2)

    for f in os.listdir(testdir2):
        pp = model2.perplexity(corpus_reader(
            os.path.join(testdir2, f), model2.lexicon))
        pp_model1 = model1.perplexity(corpus_reader(
            os.path.join(testdir2, f), model1.lexicon))
        total += 1
        correct += (pp < pp_model1)

    return correct/total


if __name__ == "__main__":

    #model = TrigramModel(sys.argv[1])
   # print(model.raw_trigram_probability(('walter', 'lippmann', 'and')))
   # print(model.raw_bigram_probability(('START', 'the')))
   # print(model.raw_unigram_probability(('the',)))

    # put test code here...
    # or run the script from the command line with
    # $ python -i trigram_model.py [corpus_file]
    # >>>
    #
    # you can then call methods on the model instance in the interactive
    # Python prompt.

    # Testing perplexity:
   # dev_corpus = corpus_reader(sys.argv[2], model.lexicon)
  #  pp = model.perplexity(dev_corpus)
   # print(f"pp is {pp}")

    # Essay scoring experiment:
    acc = essay_scoring_experiment(
        "hw1_data/ets_toefl_data/train_high.txt",
        "hw1_data/ets_toefl_data/train_low.txt",
        "hw1_data/ets_toefl_data/test_high",
        "hw1_data/ets_toefl_data/test_low")
    print(acc)

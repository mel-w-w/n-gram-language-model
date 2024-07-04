import sys
from collections import defaultdict
import math
import random
import os
import os.path
from collections import Counter



def corpus_reader(corpusfile, lexicon=None):
    with open(corpusfile, 'r') as corpus:  # open the corpus
        for line in corpus:  # for loop to go through every line in the corpus
            if line.strip():  # if statement to see if line has string
                sequence = line.lower().strip().split()  # cleans up the line
                if lexicon:
                    yield [word if word in lexicon else "UNK" for word in sequence]
                else:
                    yield sequence  # return a list of strings per sentence in corpus file


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

    if sequence[0] != 'START':
        sequence.insert(0, "START")

    if(n > 2):
        for i in range(n-2):
            sequence.insert(0, "START")

    if sequence[-1] != 'STOP':
        sequence.append("STOP")

    my_tuple = zip(*[sequence[element:] for element in range(n)])
    return list(my_tuple)


class TrigramModel(object):

    def __init__(self, corpusfile):

        # Iterate through the corpus once to build a lexicon
        generator = corpus_reader(corpusfile)
        self.lexicon = get_lexicon(generator)
        self.lexicon.add("UNK")
        self.lexicon.add("START")
        self.lexicon.add("STOP")

        # Now iterate through the corpus again and count ngrams
        generator = corpus_reader(corpusfile, self.lexicon)
        self.count_ngrams(generator)

    def count_ngrams(self, corpus):
        """
        COMPLETE THIS METHOD (PART 2)
        Given a corpus iterator, populate dictionaries of unigram, bigram,
        and trigram counts.
        """

        # Your code here

        # Make some defaultdicts
        self.unigramcounts = defaultdict(int)
        self.bigramcounts = defaultdict(int)
        self.trigramcounts = defaultdict(int)
        self.totalunigrams = 0
        self.totaltypes = 0

        # Make some arrays to store n-grams
        unigrams = []
        bigrams = []
        trigrams = []

        # Make a for loop that populates the arrays
        # and updates our defaultdicts
        for sentence in corpus:
            unigrams = get_ngrams(sentence, 1)
            for element in unigrams:
                self.unigramcounts[element] += 1
            # Track number of unigrams, or tokens
            self.totalunigrams += len(unigrams)
            # Track number of types
            self.totaltypes += len(self.unigramcounts)
            bigrams = get_ngrams(sentence, 2)
            for element in bigrams:
                self.bigramcounts[element] += 1
            trigrams = get_ngrams(sentence, 3)
            for element in trigrams:
                self.trigramcounts[element] += 1

        return

    def raw_trigram_probability(self, trigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) trigram probability
        """

        # if statements to handle edge cases
        if trigram[:1] == trigram[1:2] == ('START',):
            return self.trigramcounts[trigram] / self.unigramcounts[('STOP',)]
         

        if self.bigramcounts[trigram[:-1]] == 0:
            return 1 / self.totaltypes

        # trigram count divided by bigram count
        
        return self.trigramcounts[trigram] / self.bigramcounts[trigram[:-1]]

    def raw_bigram_probability(self, bigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) bigram probability
        """

        # Edge case
        if self.unigramcounts[bigram[:-1]] == 0:
            return 1 / self.totaltypes

        if tuple(bigram[0]) == ('START',):
            return self.bigramcounts[bigram] / self.unigramcounts[('STOP',)]

        return self.bigramcounts[bigram] / self.unigramcounts[bigram[:-1]]

    def raw_unigram_probability(self, unigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) unigram probability.
        """

        return self.unigramcounts[unigram] / self.totalunigrams

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

        result = (lambda1 * self.raw_trigram_probability(trigram)) + (lambda2*self.raw_bigram_probability(trigram[1:])) + (lambda3*self.raw_unigram_probability(trigram[2:]))

        return result

    def sentence_logprob(self, sentence):
        """
        COMPLETE THIS METHOD (PART 5)
        Returns the log probability of an entire sequence.
        """

        # a list for sentence log probabilities
        list_trigrams = []

        for element in get_ngrams(sentence, 3):

            smooth_prob = self.smoothed_trigram_probability(element)

            smooth_prob_log = math.log2(smooth_prob)
            list_trigrams.append(smooth_prob_log)

        result = sum(list_trigrams)

        return float(result)

    def perplexity(self, corpus):
        """
        COMPLETE THIS METHOD (PART 6)
        Returns the log probability of an entire sequence.
        """

        # M = total number of words
        # Get sentence_logprob for each sentence in the corpus
        # Sum all sentence_logprob
        # divide sum by M

        M = 0
        list_sentences = []

        for sentence in corpus:
            list_sentences.append(self.sentence_logprob(sentence))
            M = M + len(sentence) + 1

        sum_sentence_prob = sum(list_sentences)

        result = sum_sentence_prob / M

        return float(2 ** -(result))


def essay_scoring_experiment(training_file1, training_file2, testdir1, testdir2):

    model1 = TrigramModel(training_file1)
    model2 = TrigramModel(training_file2)

    total = 0
    correct = 0

    # good tester
    for f in os.listdir(testdir1):
        pp = model1.perplexity(corpus_reader(
            os.path.join(testdir1, f), model1.lexicon))

        second_p = model2.perplexity(corpus_reader(
            os.path.join(testdir1, f), model2.lexicon))

        if pp < second_p:
            correct += 1
        total += 1

    # bad tester
    for f in os.listdir(testdir2):
        pp = model2.perplexity(corpus_reader(
            os.path.join(testdir2, f), model2.lexicon))

        second_p = model1.perplexity(corpus_reader(
            os.path.join(testdir2, f), model1.lexicon))

        if pp < second_p:
            correct += 1
        total += 1

    accuracy = correct / total

    return accuracy


if __name__ == "__main__":

    # model = TrigramModel(sys.argv[1])

    def main():

        model = TrigramModel(
            "hw1_data/brown_train.txt")

        # Test Part 1

        print(get_ngrams(["natural", "language", "processing"], 1))
        print(get_ngrams(["natural", "language", "processing"], 2))
        print(get_ngrams(["natural", "language", "processing"], 3))

        # Test Part 2, 3, and 4

        unigram = ('the',)
        print("Number of unigrams:", model.unigramcounts[unigram])
        print("Raw unigram probability of 'the': ",
              model.raw_unigram_probability(unigram))

        bigram = ('START', 'the')
        print("Number of bigrams: ", model.bigramcounts[bigram])
        print("Raw bigram probability of 'STARTs, 'the': ",
              model.raw_bigram_probability(bigram))

        trigram = ('START', 'START', 'the')
        print("Number of trigrams: ", model.trigramcounts[trigram])
        print("Raw trigram probability of 'START', 'START', 'the': ",
              model.raw_trigram_probability(trigram))
        print("Smoothed trigram probability of 'START', 'START', 'the': ",
              model.smoothed_trigram_probability(trigram))

        # Test Part 5
        list = ['his', 'petition', 'charged', 'mental', 'cruelty']
        print(f"Sentence log probability of {list}: ", model.sentence_logprob(
            list))

        # Test Part 6
        dev_corpus = corpus_reader(
            "hw1_data/brown_test.txt", model.lexicon)
        print("Perplexity of brown_test.txt: ", model.perplexity(dev_corpus))

        dev_corpus2 = corpus_reader(
            "hw1_data/brown_train.txt", model.lexicon)
        print("Perplexity of brown_train.txt: ", model.perplexity(dev_corpus2))

        # Test Part 7
        acc = essay_scoring_experiment("hw1_data/ets_toefl_data/train_high.txt", "hw1_data/ets_toefl_data/train_low.txt",
                                       "hw1_data/ets_toefl_data/test_high", "hw1_data/ets_toefl_data/test_low")
        print("Accuracy of prediction: ", acc)

    main()

# n-gram-language-model

This project was completed for the course Natural Language Processing at Columbia University.

Goals:
- build a trigram language model in Python.
- apply the model to a text classification task. In this case, the project used a data set of essays written by non-native speakers of English for the ETS TOEFL test. These essays are scored as "high" or "low". The model will automatically score these essays, and the perplexity of the model will be calculated based on each essay. 

Functions and their descriptions:
- `get_ngrams()`: takes a list of strings and an integer n as input, and returns padded n-grams over the list of strings. The result is a list of Python tuples.
- `corpus_reader()`: takes the name of a text file as a parameter and returns a Python generator object.
- `get_lexicon(corpus)`: takes a corpus iterator as a parameter and returns a set of all words that appear in the corpus more than once.
- `__init__`: takes the filename of a corpus file and iterates twice through the corpus - once to collect the lexicon, and once to count n-grams.
- `count_ngrams()`: counst the occurrence frequencies for ngrams in the corpus
- `raw_trigram_probability(trigram)`, `raw_bigram_probability(bigram)`, and `raw_unigram_probability(unigram)`: returns an unsmoothed probability computed from the trigram, bigram, and unigram counts.
- `smoothed_trigram_probability(self, trigram)`: uses linear interpolation between the raw trigram, unigram, and bigram probabilities.
- `sentence_logprob(sentence)`: returns log probability of an entire sequence. Converts each probability into logspace using `math.log2`
- `perplexity(corpus)`: computes the perplexity of the model of an entire corpus.
- `essay_scoring_experiment()`: takes 2 training text files, and 2 testing directories. Returns the accuracy of the prediction. 

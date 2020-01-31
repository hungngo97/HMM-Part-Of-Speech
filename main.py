from utils.io_utils import (read_sentences_from_file)
from constants.constants import (
    DEV_DATA_FILE, TRAIN_DATA_FILE, TEST_DATA_FILE)
from models.BigramHMM import (HMM)

# Read input
train_sentences = read_sentences_from_file(TRAIN_DATA_FILE)
dev_sentences = read_sentences_from_file(DEV_DATA_FILE)
test_sentences = read_sentences_from_file(TEST_DATA_FILE)
print(train_sentences[0])

bigram_hmm = HMM()
bigram_hmm.find_word_tag_pairs(train_sentences)
bigram_hmm.tag_unigrams = bigram_hmm.find_ngrams(train_sentences, 1)
bigram_hmm.tag_bigrams = bigram_hmm.find_ngrams(train_sentences, 2)
tag_unigrams = bigram_hmm.tag_unigrams
tag_bigrams = bigram_hmm.tag_bigrams
tags_frequency = bigram_hmm.tags_frequency

sentence_tags = bigram_hmm.predict_tag_one_sentence(train_sentences[10])
emission_transition = bigram_hmm.emission_transition


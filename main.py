from utils.io_utils import (read_sentences_from_file)
from constants.constants import (
    DEV_DATA_FILE, TRAIN_DATA_FILE, TEST_DATA_FILE)
from models.BigramHMM import (HMM)
from models.TrigramHMM import (TrigramHMM)
from evaluator.EvaluatorPOS import (EvaluatorPOS)

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

# sentence_tags = bigram_hmm.predict_tag_one_sentence(dev_sentences[10])
sentence_tags = bigram_hmm.predict_tags(dev_sentences)

""" Evaluate model"""
evaluator = EvaluatorPOS()
#train_sentence_tags = bigram_hmm.predict_tags(train_sentences)
#train_confusion_matrix, train_correct, train_total, error_bigram_train = evaluator.test_label_pos_prediction(train_sentences, train_sentence_tags)

dev_sentence_tags = bigram_hmm.predict_tags(dev_sentences)
dev_confusion_matrix, dev_correct, dev_total, error_bigram_dev = evaluator.test_label_pos_prediction(dev_sentences, dev_sentence_tags)

test_sentence_tags = bigram_hmm.predict_tags(test_sentences)
test_confusion_matrix, test_correct, test_total, error_bigram_test = evaluator.test_label_pos_prediction(test_sentences, test_sentence_tags)

""" Trigram HMM """
trigram_hmm = TrigramHMM()
trigram_hmm.find_word_tag_pairs(train_sentences)
trigram_hmm.tag_unigrams = trigram_hmm.find_ngrams(train_sentences, 1)
trigram_hmm.tag_bigrams = trigram_hmm.find_ngrams(train_sentences, 2)
trigram_hmm.tag_trigrams = trigram_hmm.find_ngrams(train_sentences, 3)

tag_trigrams = trigram_hmm.tag_trigrams
sentence_tags = trigram_hmm.predict_tag_one_sentence(dev_sentences[10])

dev_sentence_tags_trigram = trigram_hmm.predict_tags(dev_sentences)
dev_confusion_matrix_trigram, dev_correct_trigram, dev_total_trigram, error_trigram_dev = evaluator.test_label_pos_prediction(dev_sentences, dev_sentence_tags_trigram)

test_sentence_tags_trigram = trigram_hmm.predict_tags(test_sentences)
test_confusion_matrix_trigram, test_correct_trigram, test_total_trigram, error_trigram_test = evaluator.test_label_pos_prediction(test_sentences, test_sentence_tags_trigram)


""" Try Bigram with better selective UNK """
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

train_sentences_selective_unk = read_sentences_from_file(TRAIN_DATA_FILE, selective_unk=True)
dev_sentences_selective_unk = read_sentences_from_file(DEV_DATA_FILE, selective_unk=True)
test_sentence_selective_unks = read_sentences_from_file(TEST_DATA_FILE, selective_unk=True)

def print_pretty_matrix(d):
    
    # find all the columns and all the rows, sort them    
    columns = sorted(set(key for dictionary in d.values() for key in dictionary))
    rows = sorted(d)
    
    # figure out how wide each column is
    col_width = max(max(len(thing) for thing in columns),
                        max(len(thing) for thing in rows)) + 3
    
    # preliminary format string : one column with specific width, right justified
    fmt = '{{:>{}}}'.format(col_width)
    
    # format string for all columns plus a 'label' for the row
    fmt = fmt * (len(columns) + 1)
    
    # print the header
    print(fmt.format('', *columns))
    
    # print the rows
    for row in rows:
        dictionary = d[row]
        s = fmt.format(row, *(dictionary.get(col, 'inf') for col in columns))
        print(s)


bigram_hmm = HMM()
bigram_hmm.find_word_tag_pairs(train_sentences)
bigram_hmm.tag_unigrams = bigram_hmm.find_ngrams(train_sentences, 1)
bigram_hmm.tag_bigrams = bigram_hmm.find_ngrams(train_sentences, 2)
tag_unigrams = bigram_hmm.tag_unigrams
tag_bigrams = bigram_hmm.tag_bigrams
tags_frequency = bigram_hmm.tags_frequency

sentence_tags = bigram_hmm.predict_tag_one_sentence(train_sentences[10])
emission_transition = bigram_hmm.emission_transition

""" Evaluate model"""
evaluator = EvaluatorPOS()
train_sentence_tags = bigram_hmm.predict_tags(train_sentences)
train_confusion_matrix, train_correct, train_total, error_bigram_train = evaluator.test_label_pos_prediction(train_sentences, train_sentence_tags)

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


""" ================= Try Bigram with better selective UNK ============== """
bigram_hmm = HMM()
bigram_hmm.find_word_tag_pairs(train_sentences_selective_unk)
bigram_hmm.tag_unigrams = bigram_hmm.find_ngrams(train_sentences_selective_unk, 1)
bigram_hmm.tag_bigrams = bigram_hmm.find_ngrams(train_sentences_selective_unk, 2)
tag_unigrams = bigram_hmm.tag_unigrams
tag_bigrams = bigram_hmm.tag_bigrams
tags_frequency = bigram_hmm.tags_frequency

sentence_tags = bigram_hmm.predict_tag_one_sentence(train_sentences_selective_unk[10])
emission_transition = bigram_hmm.emission_transition

""" Evaluate model"""
evaluator = EvaluatorPOS()
train_sentence_tags_selective_unk = bigram_hmm.predict_tags(train_sentences_selective_unk)
train_confusion_matrix_selective_unk, train_correct_selective_unk, train_total_selective_unk, error_bigram_train_selective_unk = evaluator.test_label_pos_prediction(train_sentences_selective_unk, train_sentence_tags_selective_unk)

dev_sentence_tags_selective_unk = bigram_hmm.predict_tags(dev_sentences_selective_unk)
dev_confusion_matrix_selective_unk, dev_correct_selective_unk, dev_total_selective_unk, error_bigram_dev_selective_unk = evaluator.test_label_pos_prediction(dev_sentences_selective_unk, dev_sentence_tags_selective_unk)

test_sentence_tags_selective_unk = bigram_hmm.predict_tags(test_sentence_selective_unks)
test_confusion_matrix_selective_unk, test_correct_selective_unk, test_total_selective_unk, error_bigram_test_selective_unk = evaluator.test_label_pos_prediction(test_sentence_selective_unks, test_sentence_tags_selective_unk)

""" Trigram Model with better Selective UNK """
trigram_hmm = TrigramHMM()
trigram_hmm.find_word_tag_pairs(train_sentences_selective_unk)
trigram_hmm.tag_unigrams = trigram_hmm.find_ngrams(train_sentences_selective_unk, 1)
trigram_hmm.tag_bigrams = trigram_hmm.find_ngrams(train_sentences_selective_unk, 2)
trigram_hmm.tag_trigrams = trigram_hmm.find_ngrams(train_sentences_selective_unk, 3)

tag_trigrams = trigram_hmm.tag_trigrams
print(dev_sentences[10])
sentence_tags = trigram_hmm.predict_tag_one_sentence(dev_sentences[10])

dev_sentence_tags_trigram_selective_unk = trigram_hmm.predict_tags(dev_sentences_selective_unk)
dev_confusion_matrix_trigram_selective_unk, dev_correct_trigram_selective_unk, dev_total_trigram_selective_unk, error_trigram_dev_selective_unk = evaluator.test_label_pos_prediction(dev_sentences_selective_unk, dev_sentence_tags_trigram_selective_unk)

test_sentence_tags_trigram_selective_unks = trigram_hmm.predict_tags(test_sentence_selective_unks)
test_confusion_matrix_trigram_selective_unks, test_correct_trigram_selective_unks, test_total_trigram_selective_unks, error_trigram_test_selective_unks = evaluator.test_label_pos_prediction(test_sentence_selective_unks, test_sentence_tags_trigram_selective_unks)


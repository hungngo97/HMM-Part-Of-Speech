# -*- coding: utf-8 -*-
from utils.io_utils import (read_sentences_from_file)
from constants.constants import (
    DEV_DATA_FILE, TRAIN_DATA_FILE, TEST_DATA_FILE)
from models.BigramHMM import (HMM)
from models.ForwardBackward import (ForwardBackward)
from evaluator.EvaluatorPOS import (EvaluatorPOS)

train_sentences_selective_unk = read_sentences_from_file(TRAIN_DATA_FILE, selective_unk=True)
dev_sentences_selective_unk = read_sentences_from_file(DEV_DATA_FILE, selective_unk=True)
test_sentence_selective_unks = read_sentences_from_file(TEST_DATA_FILE, selective_unk=True)


# Try with dev first
# Split dev in half
data = dev_sentences_selective_unk
index = int(len(data) / 2)
supervised_data = data[:index]
unsupervised_data = data[index:]

# Train HMM viterbi on the supervised data part
train_sentences = supervised_data
test_sentences = unsupervised_data

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
test_sentence_tags = bigram_hmm.predict_tags(test_sentences)
test_confusion_matrix, test_correct, test_total, error_bigram_test = evaluator.test_label_pos_prediction(test_sentences, test_sentence_tags)


# Forward backward to get expected counts
print('Unigram before backward forward', tag_unigrams)
print('Bigram before backward forward', tag_bigrams)
for sentence in test_sentences:
    print('Processing...')
    training = ForwardBackward(sentence, bigram_hmm)
    expected_count_single, expected_count_bigram = training.compute_expected_counts()
    
    # Get single count first
    for tag in expected_count_single.keys():
        count = expected_count_single[tag]
        tag_unigrams[tag] += count
        print('Adding ' + str(count))
    
    # Get bi gram count
    for tag in expected_count_bigram.keys():
        next_tag_map = expected_count_bigram[tag]
        for next_tag in next_tag_map.keys():
            key = tag + "_" + next_tag
            if key not in tag_bigrams.keys():
                tag_bigrams[key] = 0
            tag_bigrams[key] += next_tag_map[next_tag]
            
print('Unigram after backward forward', tag_unigrams)
print('Bigram after backward forward', tag_bigrams)
# -*- coding: utf-8 -*-
import numpy as np 
from constants.model_constants import (
    UNK, STOP, START, TWEET_DIRECT_UNK, EMOJI_UNK, HASHTAG_UNK, URL_UNK, TIME_UNK, NUM_UNK)

class ForwardBackward:
    def __init__(self, sentence, hmm):
        self.words = sentence[0]
        self.tags = sentence[1]
        self.tag_unigrams = hmm.tag_unigrams
        self.wordIDs = hmm.tag_unigrams #TODO: Not sure, we don't have word vocab
        self.N = len(hmm.tag_unigrams)
        self.forward = {}
        self.backward = {}
        self.hmm = hmm
    
    def update_lexical_dict(self, lex_dict, expected_counts):
        lex_dict += expected_counts
        return lex_dict
    
    def compute_expected_counts(self):
        """
            compute the counts of tags for every position int he sentence
        """
        self.forward = self.compute_forward_prob()
        self.backward = self.compute_backward_prob()
        expected_count_single = self.expected_count_single_tag()
        expected_count_bigram = self.expected_count_bigram_tag()
        return (expected_count_single, expected_count_bigram)

    def compute_forward_prob(self):
        forward = {}
        # Map from { index -> { tag_i -> sum }}
        # Base case
        forward[0] = {}
        for tag in self.tag_unigrams:
            if tag == START:
                forward[0][tag] = 1
            else:
                forward[0][tag] = 0
                
        for i in range(1, len(self.words)):
            if i not in forward.keys():
                forward[i] = {}
            
            for current_tag in self.tag_unigrams:
                alpha = 0
                for previous_tag in self.tag_unigrams:
                    alpha += self.hmm.e(self.words[i], current_tag) * self.hmm.q(current_tag, previous_tag) * forward[i - 1][previous_tag]
                forward[i][current_tag] = alpha
        return forward
    
    def compute_backward_prob(self):
        backward = {}
        # Map from { index -> { tag_i -> sum }}
        # Base case
        backward[len(self.words) - 1] = {}
        for tag in self.tag_unigrams:
            if tag == STOP:
                backward[len(self.words) - 1][tag] = 1
            else:
                backward[len(self.words) - 1][tag] = 0
                
        for i in range(len(self.words) - 2, -1, -1):
            if i not in backward.keys():
                backward[i] = {}
            
            for current_tag in self.tag_unigrams:
                beta = 0
                for next_tag in self.tag_unigrams:
                    beta += self.hmm.e(self.words[i + 1], next_tag) * self.hmm.q(next_tag, current_tag) * backward[i + 1][next_tag]
                backward[i][current_tag] = beta
        return backward
                    
    def p_sentence_and_tag(self, i, tag):
        return self.forward[i][tag] * self.backward[i][tag]
    
    def p_sentence_and_tag_and_next_tag(self, i, current_tag, next_tag):
        return self.forward[i][current_tag] * self.hmm.q(next_tag, current_tag) * self.hmm.e(self.words[i + 1], next_tag) * self.backward[i + 1][next_tag]
        
    def expected_count_single_tag(self):
        expected_count_tag_position = {}
        # Map frm { position -> {tag -> p(tag, position) }}
        for i in range(len(self.words)):
            if i not in expected_count_tag_position.keys():
                expected_count_tag_position[i] = {}
            for tag in self.tag_unigrams:
                expected_count_tag_position[i][tag] = self.p_sentence_and_tag(i, tag)
        
        # Find the probability mass total for each position
        sum_denom_position_map = {}
        # Map from { position -> total prob for each position}
        for i in range(len(self.words)):
            sum = 0
            for tag in self.tag_unigrams:
                sum += expected_count_tag_position[i][tag]
            sum_denom_position_map[i] = sum
        
        # Recompute the expected count by dividing by the prob mass of each position
        expected_count = {}
        # Map from { tag -> count}
        for tag in self.tag_unigrams:
            expected_count_current_tag = 0
            for i in range(len(self.words)):
                expected_count_current_tag += expected_count_tag_position[i][tag] / sum_denom_position_map[i]
            expected_count[tag] = expected_count_current_tag
        return expected_count
    
    def expected_count_bigram_tag(self):
        expected_count_tag_position = {}
        # Map frm { position -> {current -> next_tag }}
        for i in range(len(self.words) - 1):
            if i not in expected_count_tag_position.keys():
                expected_count_tag_position[i] = {}
            for tag in self.tag_unigrams:
                if tag not in expected_count_tag_position[i].keys():
                    expected_count_tag_position[i][tag] = {}
                for next_tag in self.tag_unigrams:
                    expected_count_tag_position[i][tag][next_tag] = self.p_sentence_and_tag_and_next_tag(i, tag, next_tag)
        
        # Find the probability mass total for each position
        sum_denom_position_map = {}
        # Map from { position -> total prob for each position}
        for i in range(len(self.words) - 1):
            sum = 0
            for tag in self.tag_unigrams:
                for next_tag in self.tag_unigrams:
                    sum += expected_count_tag_position[i][tag][next_tag]
            sum_denom_position_map[i] = sum
        
        # Recompute the expected count by dividing by the prob mass of each position
        expected_count = {}
        # Map from { tag -> {next_tag -> count}}
        for tag in self.tag_unigrams:
            if tag not in expected_count.keys():
                expected_count[tag] = {}
            for next_tag in self.tag_unigrams:
                expected_count_current_tag_next_tag = 0
                for i in range(len(self.words) - 1):
                    expected_count_current_tag_next_tag += expected_count_tag_position[i][tag][next_tag] / sum_denom_position_map[i]
                expected_count[tag][next_tag] = expected_count_current_tag_next_tag
        return expected_count
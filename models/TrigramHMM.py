from constants.model_constants import (
    UNK, START, STOP, START_TAG, STOP_TAG, PRESTART_TAG)
import sys


class TrigramHMM:
    def __init__(self):
        self.tag_unigrams = {}
        self.tag_bigrams = {}
        self.tag_trigrams = {}
        self.tags_frequency = {}
        self.emission_transition = {}
        self.SMOOTHING = 1000
        self.CURRENT_SENTENCE = -1
        

    """
      @param: sentences consists of sentence in form of [[word_i,..., word_n], [tag_i,..., tag_n]]
    """

    def find_word_tag_pairs(self, sentences):
        for sentence in sentences:
            words = sentence[0]
            tags = sentence[1]
            for i in range(len(words)):
            # for tag in tags:
                word_tag = words[i] + "_" + tags[i]
                if not word_tag in self.tags_frequency:
                    self.tags_frequency[word_tag] = 1
                else:
                    self.tags_frequency[word_tag] += 1

    """
    Emission function 
    E(x|y) <--> E(word|tag)
    """

    def e(self, x, y):
        word_tag = x + "_" + y
        e_value = 0
        if y in self.tag_unigrams.keys() and word_tag in self.tags_frequency.keys():
            e_value = (self.tags_frequency[word_tag] + 1) / \
                float(self.tag_unigrams[y] + self.SMOOTHING)
        else:
            # If that word has never been seen with that tag before
            e_value = 1 / float(self.tag_unigrams[y] + self.SMOOTHING)
        return e_value

    """
      Bigram transition function
      q(y2 | y1 ) <--> q(current_tag | previous_tag) == count(current_tag, previous_tag) / count(previous_tag)
    """

    def q(self, current_tag, previous_tag, previous_previous_tag):
        trigram = previous_previous_tag + "_" + previous_tag + "_" + current_tag
        bigram = previous_tag + "_" + current_tag
        unigram = previous_tag
        if trigram in self.tag_trigrams.keys() and bigram in self.tag_bigrams.keys():
            q_value = float(self.tag_trigrams[trigram]) / float(self.tag_bigrams[bigram])
        else:
            q_value = 0  # TODO: Maybe we can add some smoothing here for transition function
        return q_value

    """
      @returns a dictionary of ngrams for a specified n
      dictionary in shape of:
        Keys: The Ngram signature separated by "_" (i.e: Subject_Verb) indicates verb follow subject 
          in a bigram example
        Values: The frequency of that specific ngram

    """

    def find_ngrams(self, sentences, n):
        ngrams = {}
        for sentence in sentences:
            words, tags = sentence
            for i in range(n):
                # TODO: May need to fix this to use START_TAG
                tags = ['*'] + tags
            for i in range(len(tags) + 1 - n):
                gram = ""
                for j in range(n):
                    gram = gram + tags[i + j]
                    if j != n - 1:
                        gram = gram + "_"
                if gram not in ngrams:
                    ngrams[gram] = 1
                else:
                    ngrams[gram] += 1
        return ngrams

    """
      Util function to store backpointers to keep track of best candidate
      @Returns: 
        Add the word_index with an inner dictionary represents (proposed tag --> (value, previous_tag))
    """

    def add_word_tag_value_backtrace(self, word, word_index, max_prod, candidate_previous_previous_tag, candidate_previous_tag,
                                     current_tag):
        if word_index not in self.emission_transition.keys():
            self.emission_transition[word_index] = {}
        if current_tag not in self.emission_transition[word_index].keys():
            self.emission_transition[word_index][current_tag] = {}
        self.emission_transition[word_index][current_tag][candidate_previous_tag] = [
            max_prod, candidate_previous_previous_tag]

    """
      Returns the corresponding value of the proposed tag in previous index
    """

    def previous_max_tag_value_backtrace(self, word_index, tag, previous_tag):
        value = 0  # Represents the maximum previous tag
        index_value = word_index - 1  # Previous word index
        if index_value == -1 or index_value == 0: #TODO: Not sure if this is corret
            # Start word of the sentence
            value = 1  # Since the first tag is always START
        elif index_value in self.emission_transition.keys() and tag in self.emission_transition[index_value].keys() and \
            previous_tag in self.emission_transition[index_value][tag].keys():
            value = self.emission_transition[index_value][tag][previous_tag][0]
        return value

    def predict_tags(self, sentences):
        sentence_tags = []
        i = 0
        for sentence in sentences:
            self.CURRENT_SENTENCE = i
            self.emission_transition = {}
            words, tags = sentence
            for i in range(len(words)):
                if i == 0:
                    # Start of sentence
                    self.set_begin_sentence(words[i])
                else:
                    self.find_current_tag_viterbi(words[i], i)  # Viterbi
            predicted_tags = self.find_tag_sentence(words)
            sentence_tags.append(predicted_tags)
            i += 1
        return sentence_tags
    
    def predict_tag_one_sentence(self, sentence):
        sentence_tags = []
        self.emission_transition = {}
        words, tags = sentence
        i = 0
        for i in range(len(words)):
            if i == 0:
                # Start of sentence
                self.set_begin_sentence(words[i])
            else:
                self.find_current_tag_viterbi(words[i], i)  # Viterbi
        predicted_tags = self.find_tag_sentence(words)
        sentence_tags.append(predicted_tags)
        i += 1
        return sentence_tags

    def set_begin_sentence(self, word):
        candidate_previous_previous_tag = "*"
        candidate_previous_tag = "*"
        max_value = -1
        for tag in self.tag_unigrams.keys():
            if tag != '*':
                value = self.e(word, tag) * self.q(tag, candidate_previous_tag, candidate_previous_previous_tag)
                max_value = max(value, max_value)
                if value > 0:
                    self.add_word_tag_value_backtrace(word, 0, value, "*", "*", tag)

    def find_current_tag_viterbi(self, word, word_index):
        tags = self.tag_unigrams.keys()
        # print('Word', word)
        for current_tag in tags:
                # Try all possible tags for current word
            max_value = 0
            previous_tag = ""
            previous_previous_tag = ""
            emission_value = self.e(word, current_tag)
            #print('---- Current tag: ', current_tag)
            #print('+++++++ Emission: ', emission_value)
            if emission_value != 0:
                    # Shortcut circuit to stop calculating if the current tag is impossible
                for candidate_previous_tag in tags:
                    # Try all candidate previous tags
                    if candidate_previous_tag in self.emission_transition[word_index - 1].keys():
                        for candidate_previous_previous_tag in tags:
                    
                            # Dynamic Programming!
                            """
                            print('**** Curr tag', current_tag)
                            print('***** Prev tag', candidate_previous_tag)
                            print('+++++++++ Q(current, prev)', self.q(current_tag, candidate_previous_tag))
                            print('+++++++++ value(prev, prev_tag))', self.previous_max_tag_value_backtrace(word_index - 1, candidate_previous_tag))
                            """
                            value = self.q(current_tag, candidate_previous_tag, candidate_previous_previous_tag) * \
                                self.previous_max_tag_value_backtrace(
                                    word_index, candidate_previous_tag, candidate_previous_previous_tag)
                            if value > max_value:
                                max_value = value
                                previous_tag = candidate_previous_tag
                                previous_previous_tag = candidate_previous_previous_tag
                        # q(yi | y_i-1) * e(y|x) * pi(x_i - 1, y_i - 1)
                        max_value *= emission_value
                        if max_value > 0:
                            assert current_tag != ""
                            self.add_word_tag_value_backtrace(
                                word, word_index, max_value, previous_previous_tag ,previous_tag, current_tag)

    def find_tag_sentence(self, words):
        # Doing backtracing through the DP table to find the maximum path for tags labeling
        try:
            number_of_words = len(words)
            i = number_of_words - 1
            word_tags = {}  # Result map (word -> tag)
            tag_predicted = ""
            previous_tag_predicted = ""
            previous_previous_tag_predicted = ""
            max_value = -1
            for tag in self.emission_transition[i].keys():
                # Find the maximum tag for the last word
                for previous_tag in self.emission_transition[i][tag].keys():
                    tag_value = self.emission_transition[i][tag][previous_tag][0]
                    previous_previous_tag = self.emission_transition[i][tag][previous_tag][1]
                    if tag_value > max_value:
                        max_value = tag_value
                        assert tag in self.tag_unigrams.keys()
                        tag_predicted = tag
                        previous_tag_predicted = previous_tag
                        previous_previous_tag_predicted = previous_previous_tag
                        
            
            word_tags[i] = tag_predicted
            # i -= 1
            word_tags[i - 1] = previous_tag_predicted
            # previous_tag_predicted = tag_predicted
    
            current_tag = tag_predicted
            previous_tag = previous_tag_predicted
            while i > 1:
                previous_previous_tag = self.emission_transition[i][current_tag][previous_tag][1]
                word_tags[i - 2] = previous_previous_tag
                i -= 1
                current_tag = previous_tag
                previous_tag = previous_previous_tag
    
            predicted_tags = []
            # print('Number of words', len(words))
            #print(words)
            # print(word_tags)
            for j in range(len(words)):
                # print("word: ", words[j], " , Tag: ", word_tags[j])
                predicted_tags.append(word_tags[j])
            j = 0
            return predicted_tags
        except Exception as e:
            print(('Failure',e,  words, self.emission_transition, current_tag, previous_tag, i))
            sys.exit()
            return ('Failure',e,  words, self.emission_transition, current_tag, previous_tag, i)

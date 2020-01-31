class EvaluatorPOS:
    def __init__(self):
        # Confusion matrix with row is true label, column is prediction
        # { TrueLabel -> {prediction -> Count}}
        self.name = 'EvaluatorPOS'
    
    def test_label_pos_prediction(self, sentences, tags_predictions):
        assert len(sentences) == len(tags_predictions)
        confusion_matrix = {}
        correct_prediction = 0
        total_prediction = 0
        error_sentences = []
        for i in range(len(sentences)):
            sentence = sentences[i]
            words, labels = sentence
            tags_prediction = tags_predictions[i]
            if len(labels) != len(tags_prediction):
                # Error prediction
                error_sentences.append((words, labels, tags_prediction))
                continue
            assert len(labels) == len(tags_prediction)
            for j in range(len(labels)):
                label = labels[j]
                if label == 'Failure':
                    error_sentences.append((words, labels, tags_prediction))
                    break

                tag_prediction = tags_prediction[j]
                
                if isinstance(tag_prediction, list):
                    error_sentences.append((words, labels, tags_prediction))
                    break
                confusion_matrix[label] = confusion_matrix.get(label, {})
                confusion_matrix[label][tag_prediction] = confusion_matrix[label].get(tag_prediction, 0) + 1
                
                if label == tag_prediction:
                    correct_prediction += 1
                total_prediction += 1
        return (confusion_matrix, correct_prediction, total_prediction, error_sentences)
                    
            
            
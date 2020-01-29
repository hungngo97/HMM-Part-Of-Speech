from utils.io_utils import (read_sentences_from_file)
from constants.constants import (
    DEV_DATA_FILE, TRAIN_DATA_FILE, TEST_DATA_FILE)

# Read input
train_sentences = read_sentences_from_file(TRAIN_DATA_FILE)
print(train_sentences[0])

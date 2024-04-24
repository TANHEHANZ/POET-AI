import numpy as np
from sklearn.model_selection import train_test_split
from components.preprocess import preprocess_poem
from components.data_loader import load_poems_from_csv
from components.model import build_and_train_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

preprocessed_poems_data = load_poems_from_csv('poems.csv')

all_lines = [line for poem in preprocessed_poems_data for line in poem[2]]

# Initialize and fit Tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_lines)
sequences = tokenizer.texts_to_sequences(all_lines)
max_sequence_length = max(len(seq) for seq in sequences)
padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length, padding='post')

# Convert word sequences to numpy array
X = np.array(padded_sequences)
y = np.zeros_like(X) 
y[:, :-1] = X[:, 1:]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert outputs to one-hot encoding
vocab_size = len(tokenizer.word_index) + 1
sequence_length = max_sequence_length - 1 
y_train_one_hot = to_categorical(y_train, num_classes=vocab_size)
y_test_one_hot = to_categorical(y_test, num_classes=vocab_size)

# Build and train model
model = build_and_train_model(X_train, y_train_one_hot, X_test, y_test_one_hot, vocab_size, sequence_length)

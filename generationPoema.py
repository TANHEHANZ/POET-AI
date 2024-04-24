import csv
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

# Descargar recursos de NLTK
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Función para preprocesar un poema
def preprocess_poem(poem):
    poem = poem.lower()    
    words = word_tokenize(poem)    
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words and len(word) > 1]    
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)

# Preprocesamiento de los poemas
preprocessed_poems_data = []
with open('poems.csv', newline='', encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)
    for row in reader:
        title = row[0]
        author = row[1]
        lines = row[2].strip("[]").split(", ")
        preprocessed_lines = [preprocess_poem(line) for line in lines]
        preprocessed_poems_data.append((title, author, preprocessed_lines))

# Obtener todas las líneas de los poemas preprocesadas
all_lines = [line for poem in preprocessed_poems_data for line in poem[2]]

# Inicializar y ajustar el Tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_lines)
sequences = tokenizer.texts_to_sequences(all_lines)
max_sequence_length = max(len(seq) for seq in sequences)
padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length, padding='post')

# Convertir las secuencias de palabras en un arreglo numpy
X = np.array(padded_sequences)
y = np.zeros_like(X) 
y[:, :-1] = X[:, 1:]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Convertir las salidas a one-hot encoding
vocab_size = len(tokenizer.word_index) + 1
sequence_length = max_sequence_length - 1 
y_train_one_hot = to_categorical(y_train, num_classes=vocab_size)
y_test_one_hot = to_categorical(y_test, num_classes=vocab_size)

# Definir el modelo LSTM
embedding_dim = 100
model = Sequential([
    Embedding(vocab_size, embedding_dim, input_length=sequence_length),
    LSTM(128),
    Dense(vocab_size, activation='softmax')
])

# Compilar el modelo
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Entrenar el modelo
history = model.fit(X_train, y_train_one_hot, epochs=20, validation_data=(X_test, y_test_one_hot))

# Evaluar el modelo (opcional)
loss, accuracy = model.evaluate(X_test, y_test_one_hot)
print("Loss:", loss)
print("Accuracy:", accuracy)

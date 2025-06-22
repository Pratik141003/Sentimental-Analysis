
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

MODEL_PATH = "sentiment_model.h5"

# GPU check
device_name = tf.test.gpu_device_name()
print(f"Using device: {'GPU - ' + device_name if device_name else 'CPU'}")

# Load dataset
df = pd.read_csv("IMDBDataset.csv")
df['label'] = df['sentiment'].map({'positive': 1, 'negative': 0})

# Preprocess text
texts = df['review'].astype(str).values
labels = df['label'].values

tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
padded_sequences = pad_sequences(sequences, maxlen=200, padding='post', truncating='post')

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, labels, test_size=0.2, random_state=42)

# Build and train model (if not already saved)
if not os.path.exists(MODEL_PATH):
    print("Training model...")
    model = Sequential([
        Embedding(10000, 128, input_length=200),
        Bidirectional(LSTM(64, return_sequences=True)),
        Dropout(0.5),
        Bidirectional(LSTM(32)),
        Dense(32, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=10, batch_size=64, validation_split=0.1, verbose=1)
    model.save(MODEL_PATH)
    print("Model trained and saved.")
else:
    print("Loading saved model...")
    model = load_model(MODEL_PATH)

# Predict sentiment from user input
print("\nEnter a review (type 'exit' to quit):")
while True:
    text = input("Your review: ")
    if text.lower() == 'exit':
        break
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=200, padding='post', truncating='post')
    prediction = model.predict(padded, verbose=0)[0][0]
    sentiment = "Positive" if prediction >= 0.5 else "Negative"
    print(f"Predicted Sentiment: {sentiment}\n")

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load IMDb dataset
def load_imdb_dataset(imdb_file_path):
    imdb_data = pd.read_csv(imdb_file_path)
    imdb_data.columns = ["text", "label"]
    imdb_data["label"] = imdb_data["label"].str.lower()
    return imdb_data

# Load IMDb data from IMDBDataset.csv
data = load_imdb_dataset("IMDBDataset.csv")
print(f"✅ IMDb Dataset loaded successfully with {len(data)} samples")

# Encode labels: positive -> 1, negative -> 0
data["label"] = data["label"].apply(lambda x: 1 if x == "positive" else 0)

# Tokenization and padding
max_words = 10000
max_len = 200
tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
tokenizer.fit_on_texts(data["text"])

X = tokenizer.texts_to_sequences(data["text"])
X = pad_sequences(X, maxlen=max_len, padding="post", truncating="post")
y = np.array(data["label"])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the model
model = Sequential()
model.add(Embedding(input_dim=max_words, output_dim=100, input_length=max_len))
model.add(LSTM(128, return_sequences=False))
model.add(Dropout(0.5))
model.add(Dense(1, activation="sigmoid"))

# Compile model
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Train the model
history = model.fit(
    X_train,
    y_train,
    epochs=5,
    batch_size=64,
    validation_data=(X_test, y_test)
)

# Save the model
model.save("model.h5")
print("✅ Model trained and saved as 'model.h5'")

# Load the trained model
model = load_model("model.h5")

# Function to predict a review
def predict_review(review):
    review_seq = tokenizer.texts_to_sequences([review])
    review_padded = pad_sequences(review_seq, maxlen=max_len, padding="post", truncating="post")
    prediction = model.predict(review_padded)[0][0]
    
    if prediction > 0.5:
        return f"😊 Positive Review ({prediction * 100:.2f}% confidence)"
    else:
        return f"😞 Negative Review ({(1 - prediction) * 100:.2f}% confidence)"

# REAL-TIME USER INPUT SYSTEM
print("\n🎤 Real-Time Review Sentiment Analyzer 🎤")
print("Type a review below and press Enter to see the sentiment prediction!")
print("Type 'exit' to quit.\n")

while True:
    user_review = input("📝 Enter your review: ")
    
    if user_review.lower() == "exit":
        print("🚪 Exiting... Have a great day! 😊")
        break
    
    print("🤖 AI Prediction:", predict_review(user_review))
    print("-" * 60)

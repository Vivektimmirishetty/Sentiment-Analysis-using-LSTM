Python 3.10.4 (tags/v3.10.4:9d38120, Mar 23 2022, 23:13:41) [MSC v.1929 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, SpatialDropout1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load IMDb dataset (Keras provides pre-split train/test)
from tensorflow.keras.datasets import imdb

# Parameters
vocab_size = 5000  # Limit to top 5000 words
max_len = 200      # Max length of sequences (after padding)
embedding_dim = 128

# Load the dataset (already preprocessed as sequences of integers)
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=vocab_size)

# Padding the sequences to the same length
x_train = pad_sequences(x_train, maxlen=max_len)
x_test = pad_sequences(x_test, maxlen=max_len)

# Building the LSTM model
model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_len))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Summary of the model
model.summary()

# Training the model
batch_size = 64
epochs = 5

history = model.fit(x_train, y_train, validation_data=(x_test, y_test),
                    epochs=epochs, batch_size=batch_size, verbose=2)

# Evaluate the model
score = model.evaluate(x_test, y_test, verbose=0)
print(f'Test Accuracy: {score[1]:.2f}')

# Making predictions
y_pred = (model.predict(x_test) > 0.5).astype("int32")
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
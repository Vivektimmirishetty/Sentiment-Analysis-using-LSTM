Python 3.10.4 (tags/v3.10.4:9d38120, Mar 23 2022, 23:13:41) [MSC v.1929 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load custom dataset
df = pd.read_csv('train.csv')

# Split into input and labels
reviews = df['review'].values
labels = df['label'].values

# Tokenize the reviews
tokenizer = Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(reviews)

# Convert reviews to sequences of integers
sequences = tokenizer.texts_to_sequences(reviews)

# Padding the sequences
x_data = pad_sequences(sequences, maxlen=max_len)

# Split into training and test sets
x_train, x_test, y_train, y_test = train_test_split(x_data, labels, test_size=0.2, random_state=42)
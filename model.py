import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

def create_model(Tx, n_vocab, lr = 0.01):
    model = Sequential([
        LSTM(128, input_shape=(Tx, n_vocab), return_sequences=True),
        Dropout(0.3),
        LSTM(128, return_sequences=True),
        Dropout(0.3),
        Dense(n_vocab, activation='softmax')
    ])

    opt = Adam(learning_rate=lr)

    model.compile(loss='categorical_crossentropy', optimizer='adam')

    return model
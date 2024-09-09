import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, LSTM, Dropout, Dense, Reshape
from tensorflow.keras.models import Model
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

    model.compile(loss='categorical_crossentropy', optimizer=opt)

    return model

def create_model_functional_api(Tx, n_vocab, lr = 0.01):
    inputs = Input(shape=(Tx, n_vocab))
    x = LSTM(128, return_sequences=True)(inputs)
    x = Dropout(0.3)(x)
    x = LSTM(128, return_sequences=True)(x)
    x = Dropout(0.3)(x)
    x = Dense(n_vocab, activation='softmax')(x)

    model = Model(inputs, x)
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=lr))

    return model

def create_model_functionql_api_nested(Tx, lstm1_units, lstm2_units, n_vocab):
    reshaper = Reshape((1, n_vocab))

    lstm1_units = 128
    lstm2_units = 128

    # input of the model
    x0 = Input(shape=(1, n_vocab))
    a0_1 = Input(shape=(lstm1_units,), name='a0_1')
    c0_1 = Input(shape=(lstm1_units,), name='c0_1')
    a0_2 = Input(shape=(lstm2_units,), name='a0_2')
    c0_2 = Input(shape=(lstm2_units,), name='c0_2')

    # initial state
    a1 = a0_1
    c1 = c0_1
    a2 = a0_2
    c2 = c0_2
    x = x0

    # define layers
    X = Input(shape=(Tx, n_vocab))
    lstm1 = LSTM(lstm1_units, return_sequences=True, return_state=True)
    dropout1 = Dropout(0.3)
    lstm2 = LSTM(lstm2_units, return_sequences=True, return_state=True)
    dropout2 = Dropout(0.3)
    dense = Dense(n_vocab, activation='softmax')

    outputs = []

    for t in range(Tx):
        # Select the "t"th time step vector from X.
        x = X[:, t, :]
        # Use reshaper to reshape x to be (1, n_values) (≈1 line)
        x = reshaper(x)
        # LSTM 1
        lstm_out_1, a1, c1 = lstm1(inputs=x, initial_state=[a1, c1])
        # Dropout 1
        lstm_out_1 = dropout1(lstm_out_1)
        # LSTM 2
        lstm_out_2, a2, c2 = lstm2(inputs=lstm_out_1, initial_state=[a2, c2])
        # Dropout 2
        a2_dropout = dropout2(a2)
        # Apply densor to the hidden state output of LSTM_Cell
        out = dense(inputs=a2_dropout)
        # Step 2.E: append the output to "outputs"
        outputs.append(out)

    model = Model(inputs=[X, a0_1, c0_1, a0_2, c0_2], outputs=outputs)

    return model
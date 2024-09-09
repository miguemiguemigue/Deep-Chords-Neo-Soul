import numpy as np
from music21 import converter, stream, midi, chord
import os
import tensorflow as tf
import json
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from dataset_loader import load_dataset
from model_builder import create_model, create_model_functional_api
from sampling_utils import sample_with_model_duration, save_to_midi, sampling_with_lstm_and_dense_layers

# load input and output sequences
print("Loading dataset...")
Tx = 30
X, Y, note_to_int, int_to_note = load_dataset(Tx)
n_vocab = len(note_to_int)
print(f"Dataset loaded. Vocabulary size: {n_vocab}, Input shape: {X.shape}, Output shape: {Y.shape}")

# build model
print("Building the LSTM model...")
lstm1_units = 128
lstm2_units = 128
model = create_model_functional_api(Tx, n_vocab)
model.summary()

# train and save it
opt = Adam(lr=0.01)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

epochs = 100
print(f"Training the model for {epochs} epochs...")
history = model.fit(X, Y, epochs=epochs)

# Plot results
print("Plotting the training loss and accuracy...")
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training loss')
plt.title('Training loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training accuracy')
plt.title('Training accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()
print("Saving the trained model to 'data/model/trained_model.h5'...")
model.save('data/model/trained_model.h5')
print("Model saved successfully.")
# save dictionaries for later inference
with open('data/dict/note_to_int.json', 'w') as f:
    json.dump(note_to_int, f)

with open('data/dict/int_to_note.json', 'w') as f:
    json.dump(int_to_note, f)

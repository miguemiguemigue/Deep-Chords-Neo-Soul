from music21 import converter, stream, midi, chord
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from dataset_loader import load_dataset
from model import create_model
from sampling import sample_with_model_duration, save_to_midi

# load input and output sequences
Tx = 30
X, Y, note_to_int, int_to_note = load_dataset(Tx)
n_vocab = len(note_to_int)

# LSTM model
model = create_model(Tx, n_vocab, lr = 0.01)

# Train
model.fit(X, Y, epochs= 100)

# Sampling
seed_sequence = ["A3.C4.E4.G4.B4_3.5", "A3_10/3", "E3_3.5", "A4_0.5"]

# Generar una secuencia con duración utilizando el modelo entrenado
generated_sequence = sample_with_model_duration(
    model, seed_sequence, note_to_int, int_to_note, n_vocab, Tx, length=100
)

# Exportar la secuencia generada a un archivo MIDI
output_midi_file = "generated_sequence.mid"
save_to_midi(generated_sequence, output_midi_file)
import json

import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.models import Model

from sampling_utils import sample_with_model_duration, save_to_midi

# load pretrained model
print("Loading the trained model...")
pretrained_model = load_model('data/model/trained_model.h5')

Tx = pretrained_model.input.shape[1]

# load dictionaries for inference
with open('data/dict/note_to_int.json', 'r') as f:
    note_to_int = json.load(f)

with open('data/dict/int_to_note.json', 'r') as f:
    int_to_note = json.load(f)

n_vocab = len(note_to_int)

# Sampling
print("Generating new chord progression...")
seed_sequence = []
generated_sequence = sample_with_model_duration(
    pretrained_model, seed_sequence, note_to_int, int_to_note, n_vocab, Tx, length=100
)

# export midi
output_midi_file = "data/output/generated_sequence.mid"
print(f"Saving generated sequence to {output_midi_file}...")
save_to_midi(generated_sequence, output_midi_file)

print("MIDI file saved successfully!")
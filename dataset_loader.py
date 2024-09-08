import numpy as np
import music21.instrument
from music21 import converter, stream, midi, chord
import os

# Extract notes from midi to corpus
# Create dictionary of notes
# Create one-hot dim for each note
# Create dataset Tx and Ty from corpus with a given sequence length


folder_path = "data/neosoul_chords/"

def load_midi_files():
    midi_parsed_files = []
    for i in os.listdir(folder_path):
        if i.endswith(".mid"):
            midi_location = folder_path + i
            try:
                midi = converter.parse(midi_location)
                midi_parsed_files.append(midi)
            except Exception as e:
                print(f"Error processing {folder_path}: {e}")

    return midi_parsed_files

def extract_notes_from_midi(midi_files):
    notes = []
    for parsed_midi in midi_files:
        parts = music21.instrument.partitionByInstrument(parsed_midi)

        if parts: # polyphonic
            notes_to_parse = parts.parts[0].recurse()
        else: # monophonic
            notes_to_parse = parsed_midi.flat.notes

        for element in notes_to_parse:
            if isinstance(element, music21.note.Note):  # it's a note
                note_str = f"{str(element.pitch)}_{element.quarterLength}"
                notes.append(note_str)
            elif isinstance(element, music21.note.Rest):  # it's a rest
                rest_str = f"rest_{element.quarterLength}"
                notes.append(rest_str)
            elif isinstance(element, music21.chord.Chord):  # it's a chord
                chord_notes = '.'.join(str(pitch) for pitch in element.pitches)
                chord_str = f"{chord_notes}_{element.quarterLength}"
                notes.append(chord_str)

    return notes

def create_sequences(notes, sequence_length, n_vocab):

    # create dictionaries
    note_to_int = {note: number for number, note in enumerate(sorted(set(notes)))}
    int_to_note = {number: note for note, number in note_to_int.items()}

    X = []
    Y = []

    for i in range(0, len(notes) - sequence_length):
        # input sequence
        seq_in = notes[i:i + sequence_length]
        # output sequence
        seq_out = notes[i + 1:i + sequence_length + 1]

        # note to integer
        X.append([note_to_int[note] for note in seq_in])
        Y.append([note_to_int[note] for note in seq_out])

    # To numpy array
    X = np.array(X)
    Y = np.array(Y)

    # To one-hot encoding
    X = np.array([np.eye(n_vocab)[x] for x in X])
    Y = np.array([np.eye(n_vocab)[y] for y in Y])

    return X, Y, note_to_int, int_to_note

def load_dataset(sequence_length = 30):

    parsed_midi_files = load_midi_files()

    corpus = extract_notes_from_midi(parsed_midi_files)

    n_vocab = len(set(corpus))  # unique notes in corpus

    X, Y, note_to_int, int_to_note = create_sequences(corpus, sequence_length, n_vocab)

    return X, Y, note_to_int, int_to_note

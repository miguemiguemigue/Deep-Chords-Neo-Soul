import random
import numpy as np
import tensorflow as tf
from music21 import stream, note, chord, midi, duration


def sampling_with_lstm_and_dense_layers(LSTM_cell1, LSTM_cell2, dense, note_to_int, int_to_note, n_vocab, length=100):
    '''
    WIP function to generate notes using LSTM and dense layers. This function is not used yet.
    Advantage: wouldn't require an input sequence of Tx elements
    :param LSTM_cell1:
    :param LSTM_cell2:
    :param dense:
    :param note_to_int:
    :param int_to_note:
    :param n_vocab:
    :param length:
    :return:
    '''
    # Initialize the hidden state of the LSTMs
    n_a = LSTM_cell1.units
    a0_1 = np.zeros((1, n_a))  # hidden state lstm 1
    c0_1 = np.zeros((1, n_a))  # initial memory cell lstm 1
    a0_2 = np.zeros((1, n_a))  # hidden state lstm 2
    c0_2 = np.zeros((1, n_a))  # initial memory cell lstm 2

    # Get a random note as the initial input
    x0 = np.zeros((1, 1, n_vocab))  # initial one-hot vector
    first_note_idx = np.random.choice(list(note_to_int.values()))
    x0[0, 0, first_note_idx] = 1 # set as 1 the right note
    x0 = tf.convert_to_tensor(x0, dtype=tf.float32)

    generated_notes = []

    a1 = a0_1
    c1 = c0_1
    a2 = a0_2
    c2 = c0_2
    x = x0

    for _ in range(length):
        # pass through LSTM and dense layers
        a1, _, c1 = LSTM_cell1(x, initial_state=[a1, c1])
        a2, _, c2 = LSTM_cell2(a1, initial_state=[a2, c2])
        out = dense(a2)

        # dense output to note, using random choice related to the calculated probabilities
        probabilities = out.numpy().flatten()
        predicted_note_idx = np.random.choice(range(n_vocab), p=probabilities)
        predicted_note = int_to_note[predicted_note_idx]

        generated_notes.append(predicted_note)

        # update next iteration input
        x = np.zeros((1, 1, n_vocab))  # Reiniciamos el input
        x[0, 0, predicted_note_idx] = 1  # One-hot de la nota predicha

    return generated_notes

def sample_with_model_duration(model, seed_sequence, note_to_int, int_to_note, n_vocab, Tx, length=100):

    generated_notes = seed_sequence[:]
    sequence_length = len(seed_sequence)

    # convert to int representation
    current_sequence = [note_to_int[note] for note in seed_sequence]

    # if seed sequence is not long enough, add random notes
    if len(current_sequence) < Tx:
        random_sequence = [random.choice(list(note_to_int.values())) for _ in range(Tx - len(current_sequence))]
        current_sequence = random_sequence + current_sequence  # Llenar con notas aleatorias

    # generate notes
    for _ in range(length):
        # one-hot encoding
        input_sequence = np.eye(n_vocab)[current_sequence]

        # (1, Tx, n_vocab)
        input_sequence = np.expand_dims(input_sequence, axis=0)

        # predict next note
        predictions = model.predict(input_sequence, verbose=0)
        predicted_note_idx = np.argmax(predictions[0][-1])  # Tomamos la última predicción

        # int to note
        predicted_note = int_to_note[str(predicted_note_idx)]
        generated_notes.append(predicted_note)

        # update input sequence to add predicted note, keep same size by shifting
        current_sequence.append(predicted_note_idx)
        current_sequence = current_sequence[1:]

    return generated_notes


def convert_duration(duration_str):
    """
    Handle duration conversion issues with music21
    """
    try:

        return duration.Duration(float(duration_str))
    except ValueError:
        try:
            numerator, denominator = map(int, duration_str.split('/'))
            return duration.Duration(numerator / denominator)
        except Exception as e:
            print(f"Error converting duration '{duration_str}': {e}")
            return duration.Duration(1.0)  # Valor por defecto


def save_to_midi(generated_notes, output_midi_file):
    """
    Generated notes to MIDI file.

    Parameters:
    - generated_notes: Generated notes (format: "C4_1.0", "rest_0.5", "C4.E4.G4_2.0", etc.).
    - output_midi_file: midi file path
    """
    midi_stream = stream.Stream()

    for pattern in generated_notes:
        # split note (pitch) and duration
        note_duration = pattern.split('_')
        if len(note_duration) != 2:
            continue  # ignore invalid formats

        element = note_duration[0] # note, chord or rest
        duration_str = note_duration[1]  # string element duration

        # convert duration string to music21.Duration
        dur = convert_duration(duration_str)

        if element.startswith('rest'):
            # create rest element
            midi_element = note.Rest()
        elif '.' in element:
            # create chord element
            pitch_names = element.split('.')
            midi_element = chord.Chord(pitch_names)
        else:
            # create note element
            midi_element = note.Note(element)

        # set element duration
        midi_element.duration = dur

        # add to stream
        midi_stream.append(midi_element)

    # save stream to midi
    midi_stream.write('midi', fp=output_midi_file)
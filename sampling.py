import random
import numpy as np
from music21 import stream, note, chord, midi, duration


def sample_with_model_duration(model, seed_sequence, note_to_int, int_to_note, n_vocab, Tx, length=100):
    """
    Genera una nueva secuencia de notas con duración usando un modelo LSTM entrenado.

    Parameters:
    - model: El modelo LSTM entrenado.
    - seed_sequence: Secuencia inicial de notas (lista de notas).
    - note_to_int: Diccionario de notas a enteros.
    - int_to_note: Diccionario de enteros a notas.
    - n_vocab: Número total de notas únicas (tamaño del vocabulario).
    - length: Longitud de la secuencia a generar (en número de notas).

    Returns:
    - generated_notes: Lista de notas generadas con duración.
    """
    generated_notes = seed_sequence[:]  # Copiamos la semilla inicial
    sequence_length = len(seed_sequence)

    # Convertir la secuencia semilla a su representación en enteros
    current_sequence = [note_to_int[note] for note in seed_sequence]

    # Si la secuencia es más corta que Tx, rellenar con ceros al inicio
    if len(current_sequence) < Tx:
        random_sequence = [random.choice(list(note_to_int.values())) for _ in range(Tx - len(current_sequence))]
        current_sequence = random_sequence + current_sequence  # Llenar con notas aleatorias

    for _ in range(length):
        # Preparamos la secuencia actual en el formato correcto (one-hot encoding)
        input_sequence = np.eye(n_vocab)[current_sequence]  # One-hot encoding

        # Redimensionamos para que sea (1, Tx, n_vocab)
        input_sequence = np.expand_dims(input_sequence, axis=0)

        # Predecir la próxima nota con duración
        predictions = model.predict(input_sequence, verbose=0)
        predicted_note_idx = np.argmax(predictions[0][-1])  # Tomamos la última predicción

        # Convertimos el índice predicho a una nota con duración
        predicted_note = int_to_note[predicted_note_idx]

        # Añadimos la nota predicha a la secuencia generada
        generated_notes.append(predicted_note)

        # Actualizamos la secuencia actual eliminando el primer elemento y añadiendo el predicho
        current_sequence.append(predicted_note_idx)
        current_sequence = current_sequence[1:]  # Mantener la longitud de la secuencia constante

    return generated_notes


def convert_duration(duration_str):
    """
    Convierte una cadena de duración en formato decimal o fraccionario a una duración music21.

    Parameters:
    - duration_str: Duración en formato decimal o fraccionario como string.

    Returns:
    - Una duración music21 adecuada.
    """
    try:
        # Intentar convertir a flotante directamente
        return duration.Duration(float(duration_str))
    except ValueError:
        # Si falla, intentar convertir la fracción
        try:
            numerator, denominator = map(int, duration_str.split('/'))
            return duration.Duration(numerator / denominator)
        except Exception as e:
            print(f"Error al convertir la duración '{duration_str}': {e}")
            return duration.Duration(1.0)  # Valor por defecto


def save_to_midi(generated_notes, output_midi_file):
    """
    Convierte una lista de notas/acordes generados con duración en un archivo MIDI.

    Parameters:
    - generated_notes: Lista de notas generadas (con formato "C4_1.0", "rest_0.5", "C4.E4.G4_2.0", etc.).
    - output_midi_file: Ruta del archivo MIDI de salida.
    """
    midi_stream = stream.Stream()

    for pattern in generated_notes:
        # Separar las notas (o acorde) de la duración
        note_duration = pattern.split('_')
        if len(note_duration) != 2:
            continue  # Si el formato no es correcto, lo ignoramos

        element = note_duration[0]  # Puede ser una nota, acorde o 'rest'
        duration_str = note_duration[1]  # Duración como cadena

        # Convertir la duración a un objeto music21.Duration
        dur = convert_duration(duration_str)

        if element.startswith('rest'):
            # Si es un silencio
            midi_element = note.Rest()
        elif '.' in element:
            # Si es un acorde
            pitch_names = element.split('.')
            midi_element = chord.Chord(pitch_names)
        else:
            # Si es una nota
            midi_element = note.Note(element)

        # Establecer la duración del elemento
        midi_element.duration = dur

        # Añadir el elemento al flujo
        midi_stream.append(midi_element)

    # Guardar el flujo como un archivo MIDI
    midi_stream.write('midi', fp=output_midi_file)
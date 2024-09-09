# Neo-Soul Chord Progression Generator with LSTM

This project implements an LSTM-based neural network that learns neo-soul chord progressions from MIDI files and generates new chord progressions. The model is trained on MIDI files and is capable of producing MIDI outputs with predicted chord sequences.

## Project Structure

- **`train_model.py`**: Script for training the LSTM model on neo-soul MIDI files.
- **`inference_sampling_model.py`**: Script for generating new chord progressions based on a trained model. The script samples an initial random sequence and makes predictions, which are exported as a MIDI file.

## Requirements

Before running the scripts, ensure you have the necessary dependencies installed. You can install them using:

```bash
pip install -r requirements.txt
```

Common dependencies include:

- TensorFlow / Keras
- Music21 (for MIDI processing)
- Numpy
- JSON (for handling mappings between notes and integers)

## Training the Model

To train the model on your dataset of MIDI files, simply run the `train_model.py` script:

```bash
python train_model.py
```

This will preprocess the MIDI files, convert them into sequences that the LSTM can learn, and then start the training process. The model will be saved in the `data/model/` directory after training.

## Generating Chord Progressions

Once the model is trained, you can generate new chord progressions by running the `inference_sampling_model.py` script. This script uses a random initial sequence to sample chords and generates new sequences based on the trained model.

```bash
python inference_sampling_model.py
```

This will output a MIDI file containing the predicted chord progression, which will be saved in the `data/output/` directory.

## Usage Example

### Training the Model

The model can be trained using the following command:

```bash
python train_model.py
```

This script will:

1. Load the dataset of MIDI files.
2. Convert the MIDI notes into a sequence of integers using `note_to_int.json` and `int_to_note.json`.
3. Train an LSTM model on the chord sequences.
4. Save the trained model for inference.

### Generating Chord Progressions

After the model has been trained, you can generate new chord progressions by running:

```bash
python inference_sampling_model.py
```

This script will:

1. Load the trained LSTM model.
2. Generate a random initial sequence.
3. Predict the next set of chords based on the initial sequence.
4. Export the predicted progression as a MIDI file.

## Data and Preprocessing

The project assumes that your dataset of MIDI files is stored in a directory named `data/midi/`. These files are preprocessed into numerical sequences that the LSTM can use for training. Two mapping dictionaries, `note_to_int.json` and `int_to_note.json`, are used to convert between notes and integers during both training and inference.

## Model Architecture

The model consists of two LSTM layers with dropout regularization, followed by a dense layer with a softmax activation function. The LSTM layers capture the temporal dependencies in the chord sequences, while the dense layer predicts the next chord in the sequence.

```python
# Example architecture
model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(Tx, n_vocab)),
    Dropout(0.3),
    LSTM(128, return_sequences=True),
    Dropout(0.3),
    Dense(n_vocab, activation='softmax')
])
```

from pathlib import Path
from typing import List
import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator
import mido
import logging
from collections import Counter
import pretty_midi

logger = logging.getLogger()


def load_midi_to_df(filename: Path) -> pd.DataFrame:
    mid = mido.MidiFile(filename)
    messages = [msg for track in mid.tracks for msg in track]
    df = pd.DataFrame([m.dict() for m in messages])
    return df

def load_midi_to_df_norm(filename: Path, fixed_tempo: int = 500000, max_duration: float = 120.0) -> pd.DataFrame:
    """
    Normalizes the songs to the same tempo and duration.
    This function takes a file as an input and normalizes it, 
    to output a dataframe with specified tempo and duration.
    """
    mid = mido.MidiFile(filename)
    messages = []
    for track in mid.tracks:
        for msg in track:
            if not msg.is_meta:
                msg_dict = msg.dict()
                msg_dict['tempo'] = fixed_tempo  # force uniform tempo
                messages.append(msg_dict)
    df = pd.DataFrame(messages)
    ticks_per_beat = mid.ticks_per_beat
    df['seconds'] = df['time'] * fixed_tempo / (ticks_per_beat * 1e6)
    df['time_from_start'] = df['seconds'].cumsum()
    df = df[df['time_from_start'] <= max_duration]
    return df


class MidiPathToDataFrameNorm(TransformerMixin, BaseEstimator):
    """
    Transformer that loads and normalizes MIDI files into pandas DataFrames.

    This class reads MIDI files from a given directory and converts each into a DataFrame,
    normalizing all files to a fixed tempo and truncating them to a maximum duration.

    Notes
    -----
    - Only non-meta messages are retained.
    - Timing is converted from ticks to seconds using the fixed tempo.
    - A cumulative time column (`time_from_start`) is added for chronological alignment.
    - If a file fails to load or parse, it is skipped with a warning.
    """
    def __init__(self, data_dir: Path, fixed_tempo: int = 500000, max_duration: float = 120.0):
        self.data_dir = data_dir
        self.fixed_tempo = fixed_tempo
        self.max_duration = max_duration

    def fit(self, X, y=None):
        return self

    def transform(self, X: List[str]) -> List[pd.DataFrame]:
        dfs = []
        for i, filename in enumerate(X):
            try:
                if i % max(1, len(X) // 20) == 0:
                    logger.info(f'Loaded {i} of {len(X)}')
                df = load_midi_to_df_norm(
                    self.data_dir / filename,
                    fixed_tempo=self.fixed_tempo,
                    max_duration=self.max_duration
                )
                dfs.append(df)
            except Exception as e:
                logger.warning(f"Failed to process {filename}: {e}")
        return dfs


class MidiPathToDataFrame(TransformerMixin, BaseEstimator):
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir

    def fit(self, X, y=None):
        return self

    def transform(self, X: List[str]) -> List[pd.DataFrame]:
        dfs = []
        for i, filename in enumerate(X):
            if i % (len(X) // 20) == 0:
                logger.info(f'Loaded {i} of {len(X)}')
            dfs.append(load_midi_to_df(self.data_dir / filename))
        return dfs


class PreprocessMidiDataFrame(TransformerMixin, BaseEstimator):
    """
    Slims the dataframe to just notes played.
    Will pass along a dataframe with `time_from_start`, `note`, `velocity`, and `duration` columns.
    """
    def __init__(self):
        self.warn_cols = ['note_off', 'polytouch']

    def fit(self, X, y=None):
        return self

    def transform(self, X: List[pd.DataFrame]) -> List[pd.DataFrame]:
        processed = []
        for df in X:
            processed.append(self._transform_single(df))
        return processed

    def _transform_single(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in self.warn_cols:
            if col in df:
                logger.warning(f'Column `{col}` encountered')
        # TODO: standardize time based on time signature metadata
        df['time_from_start'] = df.time.cumsum()
        df = df[df.type == 'note_on'].copy()
        df.loc[:, 'next_note_event_time'] = df.groupby('note')['time_from_start'].shift(-1)
        df['duration'] = df.next_note_event_time - df.time_from_start
        df = df[df.velocity > 0]
        df = df[['time_from_start', 'note', 'velocity', 'duration']].reset_index(drop=True)
        return df


class BagOfNotes(TransformerMixin, BaseEstimator):
    def __init__(self, normalize: bool = False):
        self.normalize = normalize

    def fit(self, X: List[pd.DataFrame], y=None):
        return self

    def transform(self, X: List[pd.DataFrame]) -> np.ndarray:
        results = []
        for df in X:
            notes = np.zeros(128)
            counts_per_note = df.note.value_counts()
            notes[counts_per_note.index.to_numpy().astype(np.int8)] = counts_per_note.values
            if self.normalize:
                notes = notes / np.sum(notes)
            results.append(notes)
        return np.array(results)


class NfIsf(TransformerMixin, BaseEstimator):
    """
    Note frequency - Inverse song frequency
    Like Tf-Idf but for music.
    """
    def __init__(self):
        self.bag = BagOfNotes(normalize=True)

    def fit(self, X: List[pd.DataFrame], y=None):
        # counts will be Nx128
        counts = self.bag.fit_transform(X)
        # sum number of songs that contain each note
        # +1 to avoid /0 error
        songs_per_note = np.sum(counts > 0, axis=1, keepdims=True) +1
        self.inv_song_freq_ = np.log((counts.shape[0] +1) / songs_per_note)
        return self

    def transform(self, X: List[pd.DataFrame]) -> np.ndarray:
        counts = self.bag.transform(X)
        return counts * self.inv_song_freq_


class BagOfChords(TransformerMixin, BaseEstimator):
    """
    Vectorizes MIDI dataframes by chord frequency. 

    Transformer that converts MIDI note dataframes into a fixed-size vector representing the frequency
    of chords (simultaneously played notes) within each song.
    """
    def __init__(self, time_threshold: float = 0.05, vocab_size: int = 500):
        self.time_threshold = time_threshold
        self.vocab_size = vocab_size

    def fit(self, X: list[pd.DataFrame], y=None):
        chord_counter = Counter()
        for df in X:
            chords = self._extract_chords(df)
            chord_counter.update(chords)
        most_common = chord_counter.most_common(self.vocab_size) # Keep only the top N most common chords
        self.vocab_ = {chord: idx for idx, (chord, _) in enumerate(most_common)}
        return self

    def transform(self, X: list[pd.DataFrame]) -> np.ndarray:
        results = []
        for df in X:
            chord_vector = np.zeros(len(self.vocab_))
            chords = self._extract_chords(df)
            for chord in chords:
                if chord in self.vocab_:
                    chord_vector[self.vocab_[chord]] += 1
            results.append(chord_vector)
        return np.array(results)

    def _extract_chords(self, df: pd.DataFrame) -> List[tuple]:
        df = df.sort_values('time_from_start')
        df = df.dropna(subset=['note', 'time_from_start'])
        chords = []
        current_chord = []
        last_time = None
        for _, row in df.iterrows():
            t = row["time_from_start"]
            note = int(row["note"])
            if last_time is None or t - last_time <= self.time_threshold:
                current_chord.append(note)
            else:
                if len(current_chord) > 1:
                    chords.append(tuple(sorted(current_chord)))
                current_chord = [note]
            last_time = t
        if len(current_chord) > 1:
            chords.append(tuple(sorted(current_chord)))
        return chords

class MidiPathToPrettyMidi(TransformerMixin, BaseEstimator):
    """
    Transformer that loads MIDI file paths and converts them to PrettyMIDI objects.

    Used to get MIDI features easily. 
    """
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir

    def fit(self, X, y=None):
        return self

    def transform(self, X: list[str]) -> list:
        loaded = []
        for f in X:
            path = self.data_dir / f
            try:
                pm = pretty_midi.PrettyMIDI(str(path))
                loaded.append(pm)
            except FileNotFoundError:
                print(f"File not found: {path}")
            except Exception as e:
                print(f"Error loading {path}: {e}")
        return loaded



class InstrumentAwareBoN(TransformerMixin, BaseEstimator):
    """
    Vectorizes MIDI dataframes by note frequency per instrument, and concatenates them.

    Transformer that converts a PrettyMIDI object into a Bag-of-Notes representation, split by instrument.
    """
    def __init__(self, instruments=None):
        self.instruments = instruments

    def fit(self, X, y=None):
        # Determine vocabulary of instruments if not provided
        if not self.instruments:
            all_instr = set()
            for pm in X:
                for inst in pm.instruments:
                    name = pretty_midi.program_to_instrument_name(inst.program) if not inst.is_drum else "drums"
                    all_instr.add(name)
            self.instruments = sorted(all_instr)
        return self

    def transform(self, X: list[pretty_midi.PrettyMIDI]) -> np.ndarray:
        data = []
        for pm in X:
            song_vector = []
            for name in self.instruments:
                notes = np.zeros(128)
                for inst in pm.instruments:
                    inst_name = pretty_midi.program_to_instrument_name(inst.program).lower() if not inst.is_drum else "drums"
                    if inst_name == name:
                        for note in inst.notes:
                            notes[note.pitch] += 1
                song_vector.extend(notes)
            data.append(song_vector)
        return np.array(data)

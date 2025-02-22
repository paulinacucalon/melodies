from pathlib import Path
from typing import List
import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator
import mido
import logging

logger = logging.getLogger()


def load_midi_to_df(filename: Path) -> pd.DataFrame:
    mid = mido.MidiFile(filename)
    messages = [msg for track in mid.tracks for msg in track]
    df = pd.DataFrame([m.dict() for m in messages])
    return df


class MidiPathToDataFrame(TransformerMixin, BaseEstimator):
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir

    def fit(self, X, y=None):
        return self

    def transform(self, X: List[str]) -> List[pd.DataFrame]:
        dfs = []
        for i, filename in enumerate(X):
            if i % (len(X) // 20) == 0:
                logger.debug(f'Loaded {i} of {len(X)}')
            dfs.append(load_midi_to_df(self.data_dir / filename))
        return dfs


class PreprocessMidiDataFrame(TransformerMixin, BaseEstimator):
    def __init__(self):
        self.warn_cols = ['note_off', 'polytouch']

    def fit(self, X, y=None):
        return self

    def transform(self, X: List[pd.DataFrame]) -> List[pd.DataFrame]:
        processed = []
        for df in X:
            processed.append(self._transform_single(df))

    def _transform_single(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in self.warn_cols:
            if col in df:
                logger.warning(f'Column `{col}` encountered')
        # TODO: standardize time based on time signature metadata
        df['time_from_start'] = df.time.cumsum()
        df = df[df.type == 'note_on']
        # TODO: add a duration column
        df = df[df.velocity > 0]
        df = df[['time_from_start', 'note', 'velocity']]
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

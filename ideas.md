# Big Picture

Overall we want to create informative low-dimensional representations of music.
Ways we could do so involve exploring ways to represent music, and then applying various techniques of dimensionality reduction.

## Data Representation

#### MIDI

MIDI files allow for replaying songs on a synthesizer.
The files are made up of messages, which tell you what note to play and how long to hold it.

Some potential ways we could vectorize MIDI files:
1. Bag of notes
    - 88-dimensional vector of how many times each note is hit (assuming standard 88-key piano)
    - Could consolidate octaves to 12 or 24 dimensions (e.g. high octave and low octave)
2. Bag of chords
    - N-dimensional vector where we track how many combinations of notes are hit at once and make each new combination an entry in the vector
3. TF/IDF

Questions
- Encoding speed? I think it's based on time, not like whether it's a quarter note. So might be hard to split notes or chords into sub-buckets of note length.

#### Raw Audio

Using raw audio is probably much more difficult, but would make it much easier to translate to new music.
Might be able to just parse frequencies raw and assume they translate to consistent notes.

Maybe need fourier transforms or something?

## Dimensionality Reduction

- PCA
- ISOMAP
- Embedding

## Predictive Applications

- Examine clusters for the key of the song
- Predict composer
- Predict nationality of composer
- Predict "type" (e.g. Sonata or Etude)
- Moonshot: predict top charts
- Find classical piece most like pop song

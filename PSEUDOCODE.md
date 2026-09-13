# Proposed Speaker Recognition Pipeline

```text
INPUT:
    Set of unlabelled audio recordings
    Approximate number of speakers ≈ 200



FOR each audio file:

    Read file metadata

    Record:
        - file format
        - duration
        - sample rate
        - number of channels

    IF file cannot be read:
        Flag file as corrupted


Randomly select a subset of recordings

FOR each selected recording:

    Listen manually

    Inspect:
        - background noise
        - microphone quality
        - language / accent
        - presence of one or multiple speakers

    Plot:
        - waveform
        - spectrogram



2. AUDIO PREPROCESSING


FOR each valid audio recording:

    Convert audio to mono

    Resample audio to 16 kHz

    Detect speech regions using
    Voice Activity Detection (VAD)

    Remove non-speech / silent regions

    Split longer speech regions into
    approximately 2–3 second windows

    FOR each speech window:

        Store:
            - source filename
            - segment start time
            - segment end time
            - duration



3. SPEAKER EMBEDDING EXTRACTION


Load pretrained ECAPA-TDNN speaker model

FOR each speech window:

    embedding =
        ECAPA_TDNN(speech_window)

    embedding =
        L2_NORMALISE(embedding)

    Store:
        embedding
        +
        segment metadata



4. SPEAKER CLUSTERING


Collect all speaker embeddings

Calculate cosine similarity / distance
between embeddings

Apply Agglomerative Hierarchical Clustering:

    metric = cosine distance
    linkage = average
    initial cluster target ≈ 200


Repeat clustering using nearby values:

    e.g. 180 → 220 clusters

Compare results to assess
cluster stability


Assign anonymous speaker labels:

    Cluster 0 → Speaker001
    Cluster 1 → Speaker002
    Cluster 2 → Speaker003
    ...



5. OUTLIER HANDLING


Apply HDBSCAN as an alternative
or complementary clustering method

IF a sample does not fit strongly
into any dense cluster:

    Label sample as:
        UNKNOWN / UNASSIGNED

This prevents noisy or corrupted
recordings from being forced into
a valid speaker cluster



6. EVALUATION 


Calculate:

    Silhouette Score
        → higher is better

    Davies-Bouldin Index
        → lower is better


Inspect:

    - cluster sizes
    - unusually large clusters
    - unusually small clusters
    - stability across cluster settings


Project embeddings into 2D using UMAP

Visually inspect:

    - cluster separation
    - overlapping clusters
    - outliers


Randomly select several clusters

FOR each selected cluster:

    Listen to multiple recordings

    Check whether recordings
    appear to contain the same speaker



7. FUTURE SPEAKER RECOGNITION


FOR each reliable speaker cluster:

    Calculate representative
    cluster embedding


FOR each new recording:

    preprocess audio

    extract ECAPA-TDNN embedding

    compare embedding against
    known speaker cluster embeddings

    IF similarity is above
    an accepted threshold:

        return best matching SpeakerID

    ELSE:

        return UNKNOWN


OUTPUT:
    source_file
    segment_start
    segment_end
    predicted_speaker
```
OUTPUT:
audio segment → Speaker001 / Speaker002 / ... / UNKNOWN

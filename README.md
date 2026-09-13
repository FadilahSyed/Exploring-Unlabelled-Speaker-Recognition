# Exploring-Unlabelled-Speaker-Recognition

## Objective

To investigate potential approaches for detecting and recognizing individual speakers from a dataset of over 200 unlabelled microphone recordings. When a voice is present, the goal is to identify which of the 200 speakers it belongs to.

---
## How to Navigate This Repository

The repository separates the written proposal from the supporting conceptual code.

```text
unlabelled-speaker-recognition/
│
├── README.md
├── PSEUDOCODE.md
└── src/
    ├── explore_audio.py
    └── preprocess.py
```

### Files

- **`README.md`**  
  Contains the main submission, including the proposed approach, implementation strategy, evaluation methods, challenges, and assumptions.

- **`PSEUDOCODE.md`**  
  Provides a high-level overview of the complete proposed pipeline, from audio exploration and preprocessing to speaker embedding extraction, clustering, and evaluation.

- **`src/explore_audio.py`**  
  Demonstrates how the dataset could initially be inspected, including file duration, sampling rate, number of channels, and basic error checking.

- **`src/preprocess.py`**  
  Demonstrates the proposed preprocessing steps, including converting audio to mono, resampling to 16 kHz, removing silent regions, and splitting recordings into shorter speech segments.

### Recommended Reading Order

1. Start with **`README.md`** for the overall problem analysis and proposed solution.
2. Refer to **`PSEUDOCODE.md`** for a compact view of the full processing pipeline.
3. Review the files in **`src/`** to see how the initial data exploration and preprocessing stages could be implemented.

- The assessment dataset was not provided, so the code is intended to demonstrate the proposed implementation approach rather than provide tested experimental results. Parameters such as speech-segmentation thresholds, window size, and clustering settings would need to be validated and tuned once the actual recordings are available.
---


## 1. Data Exploration

To explore the dataset of audio recordings, 3 things would be inspected.

### 1) Metadata and File properties

This includes:

- Number of files
- File formats
- Individual and total duration of recordings
- Sampling rates
- Channel counts (mono vs. stereo)
- Amplitude limits
- Corrupted files

This is to check the integrity of the dataset and to consider any necessary preprocessing, such as resampling or downmixing to mono. Unreadable data will also be flagged to clean the dataset. These properties can be checked using Python libraries (like librosa) to scan the entire directory and generate a summary of its statistics.

### 2) Auditory Spot-Checking (Listening manually)

Randomly select 20-30 recordings and manually listen to them to check:

- Background noise/music levels
- Microphone quality
- Languages or accents
- If one recording has one or more speakers

This is done to check any environment changes or noise that may need to be considered. It also determines whether one embedding can represent an entire file (whether one recording has one or multiple speakers).

### 3) Visualise Signals

For a random 5-10 recordings, plot time-domain waveforms and frequency-domain spectrograms. With this visualisation, certain properties can be seen, such as:

- Silence-to-speech ratios
- Amplitude shifts
- Noisy vs. clean speech regions
- Any overlapping acoustic events

With these, it can inform the parameters for Voice Activity Detection (VAD), silence removal and help determine the segmentation strategy used before feature extraction.

From these properties, several challenges can already be expected. Since there are no speaker labels, standard supervised training cannot be used as the classification of a speaker cannot be directly calculated. Therefore, unsupervised methods should be used. However, this may be tricky because the same speaker can sound very different depending on their mood, volume, microphone or environment, which may trick the model into splitting one person into multiple groups, or multiple speakers into one. Background noise and mic quality may also cause the model to group audio clips by room or hardware rather than the speaker. Lastly, short clips do not provide enough audio data for a reliable profile, and clips with multiple speakers may mess up models that expect only one voice per file.

---

## 2. Proposed Solution and Justification

As mentioned before, since the recordings are completely unlabelled, this problem should be treated as speaker representation and clustering rather than classification. The proposed pipeline would transform raw audio files into numerical representations (called embeddings) that capture unique vocal characteristics, which can then be grouped into speaker clusters using unsupervised learning algorithms.

### 1. Feature extraction

To process the audio files, Voice Activity Detection (VAD) is used to remove silence from the files, before being split into 2-3 second sliding windows.

Embeddings from these windows are extracted using the pretrained model ECAPA-TDNN from SpeechBrain. According to its model card, it can be used to extract speaker embeddings, and performs speaker verification using cosine distance between embeddings. Using this model, each speech segment can be converted into a fixed-dimensional embedding. Embeddings from the same speaker should ideally be closer together than embeddings from different speakers.

The main advantage of using a pretrained model is that the model already contains speaker-discriminative knowledge learned from a large labelled speech dataset (in this case, VoxCeleb dataset). This avoids the need to train a speaker model from scratch using the unlabelled recordings.

Embeddings should be L2-normalised and compared using cosine similarity, where higher similarity between two embeddings indicate that they are more likely to belong to the same speaker. Cosine distance is used as it evaluates the angle between vectors rather than their absolute magnitude. This is particularly important as it would make the representation invariant to volume. If a speaker speaks loudly in one clip and soft in another, it should still identify them as the same person based on the pitch structure and voice identity.

### 2. Clustering

The primary clustering approach can be Agglomerative Hierarchical Clustering (AHC). This is a bottom-up approach where every audio file starts its own individual cluster. Clusters that are similar are then progressively merged until a stopping criteria is reached.

AHC is suitable as:
  - The approximate number of speakers is known, so `n_clusters=200` can be set as a base target, while testing neighbouring values (e.g., 180-220) to account for any missing or dominant speakers.
  - AHC accepts cosine distance matrices, which matches the embedding representation cleanly.
  - It provides clusters and a visual diagram based on hierarchical relationships, which can be inspected.

### An alternative or complimentary method

Standard AHC does force every single audio recording into one of the 200 clusters, so if a recording contains heavy static noise, overlapping voices or microphone corruption, AHC will improperly force it into a clean speaker’s cluster.
- An alternative is the HDBSCAN, which can be used as an alternative or to complement AHC. Its purpose is to detect any outliers.
- HDBSCAN groups embeddings based on spatial density rather than spherical clusters, so it can be used as a way to automatically bin noise. Any audio sample that does not strongly fit into a dense speaker cluster can be labeled as Unassigned (or -1). Therefore, corrupted clips do not pollute valid speaker profiles.

---

## Implementation strategy

The proposed strategy follows this workflow: Preprocessing → Feature Extraction → Unsupervised Clustering → Unsupervised Evaluation

### Preprocessing

1. Standardize the Audio Format: Convert all input audio files to mono-channel, 16 kHz WAV format (the standard sampling rate expected by pre-trained speech models).
2. Voice Activity Detection (VAD): Extract speech regions from the audio clip to reduce the influence of silence and background audio during speaker feature extraction.
3. Noise reduction: The noisereduce library in python reduces noise using spectral gating.
4. Windowing: Splitting long recordings into 2-3 second chunks.

### Feature extraction

Each valid speech segment is passed through the pretrained ECAPA-TDNN model. For each segment, information like filename, `segment_start`, `segment_end`, duration, and embedding should be stored. Each final embedding should be L2 normalised.

### Unsupervised Clustering

Using the normalised embeddings, a full pairwise Cosine Distance matrix is generated. AHC can be applied using average linkage on this matrix. Initially, the baseline `n_clusters=200`, but a parameter sweep from `n_clusters=180` to `n_clusters=220` should be tested to observe if there is any influence on the stability of the clusters. Clusters are assigned to anonymous identities like Speaker001, Speaker002 and so on.

---

## 3. Evaluation

Due to the lack of ground-truth labels, evaluation should combine quantitative cluster metrics, manual spot-checking and 2D visualisations.

### Quantitative cluster metrics

- Silhouette Score (Target: High, close to +1): Measures how similar an audio file is to its assigned cluster compared to neighboring clusters. High average silhouette score confirms tight, well-separated speaker groups.
- Davies-Bouldin Index (Target: Low): Evaluates the average similarity ratio of each cluster with its most similar cluster. Lower values indicate better cluster separation.

### Visual Inspections

- Project the clusters into a 2D space (e.g., using UMAP) and check for 200 distinct clusters. Unusually large or small clusters can also be identified.

### Manually listening

- Check a random cluster and listen manually to verify that all audios sound like the same person.

Once reliable clusters are obtained, a representative embedding can be calculated for each cluster. Future recordings can then be compared against these speaker profiles. Low confidence matches should return an “unknown” tag instead of forcing the wrong speaker class.

---

## Challenges and Considerations

| Challenge | Mitigation |
|---|---|
| No speaker labels | Combine cluster metrics, stability testing, and manual validation |
| Same speaker split across clusters | Compare cluster centroids and investigate recording-condition effects |
| Different speakers merged together | Inspect within-cluster variability and use stricter similarity thresholds |
| Noise or microphone variation | Standardise preprocessing and check for device/environment bias |
| Very short speech | Apply minimum-duration or confidence filtering |
| Multiple speakers in one file | Use segment-level embeddings or speaker diarization |
| Uncertain assignments | Return UNKNOWN or flag for review |
| Larger datasets | Use batching or approximate nearest-neighbour search |

Two important failure modes are:

**Over-clustering:** one speaker is divided into several clusters.

**Under-clustering:** multiple speakers are combined into one cluster.

---

## Key assumptions

The approach assumes that:

- The dataset contains approximately 200 unique speakers.
- At least some speakers appear in multiple usable speech segments.
- Recordings contain sufficient speech for speaker embedding extraction.
- Speaker identity should primarily be determined from voice rather than recording metadata.

---

## References

- https://www.geeksforgeeks.org/machine-learning/agglomerative-clustering/
- https://arxiv.org/abs/2005.07143
- https://medium.com/tech-ai-made-easy/beginners-friendly-introduction-to-speaker-recognition-models-speakernet-ecapa-tdnn-8285350bcc26

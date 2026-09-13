INPUT: unlabelled audio recordings

1. Explore the recordings
   - Count files
   - Check format, duration, sample rate and channels
   - Flag corrupted files
   - Manually listen to a small random sample
   - Plot selected waveforms and spectrograms

2. Preprocess each recording
   - Convert to mono
   - Resample to 16 kHz
   - Remove silence / detect speech regions
   - Split speech into 2–3 second windows

3. Extract speaker embeddings
   FOR each speech window:
       embedding = ECAPA_TDNN(window)
       embedding = L2_NORMALISE(embedding)
       save embedding with filename and timestamps

4. Cluster embeddings
   - Use Agglomerative Hierarchical Clustering
   - Use cosine distance and average linkage
   - Start with approximately 200 clusters
   - Test neighbouring cluster counts

5. Evaluate without labels
   - Silhouette Score
   - Davies-Bouldin Index
   - UMAP visualisation
   - Cluster-size inspection
   - Manual listening

6. Optional outlier handling
   - Use HDBSCAN to flag uncertain / noisy samples

OUTPUT:
audio segment → Speaker001 / Speaker002 / ... / UNKNOWN

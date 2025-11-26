import librosa
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage
from collections import Counter

def ahc_language_diarization(audio_path, n_clusters=2, window_len=2.0, hop_size=0.5, n_mfcc=13, pca_dims=None):
    
    
    print(f"--- Starting Diarization for {audio_path} ---")

    y, sr = librosa.load(audio_path, sr=None)
    
    window_samples = int(window_len * sr)
    hop_samples = int(hop_size * sr)
    
    feature_vectors = []
    time_stamps = []
    
    for i in range(0, len(y) - window_samples + 1, hop_samples):
        window = y[i:i + window_samples]
        
        mfccs = librosa.feature.mfcc(y=window, sr=sr, n_mfcc=n_mfcc)
        
        mfcc_mean = np.mean(mfccs, axis=1)
        feature_vectors.append(mfcc_mean)
        
        time_stamps.append(i / sr)

    feature_vectors = np.array(feature_vectors)
    print(f"Extracted {feature_vectors.shape[0]} feature vectors of dimension {feature_vectors.shape[1]}.")

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(feature_vectors)
    
    if pca_dims is not None and pca_dims < scaled_features.shape[1]:
        pca = PCA(n_components=pca_dims)
        features_to_cluster = pca.fit_transform(scaled_features)
        print(f"Features reduced to {pca_dims} dimensions using PCA.")
    else:
        features_to_cluster = scaled_features
    
    
    Z = linkage(features_to_cluster, method='average', metric='cosine')
    
    ahc = AgglomerativeClustering(
        n_clusters=n_clusters, 
        metric='cosine', 
        linkage='average'
    )
    cluster_labels = ahc.fit_predict(features_to_cluster)
    
    print(f"Clustering complete. Found {len(np.unique(cluster_labels))} clusters.")
    print(f"Cluster distribution: {Counter(cluster_labels)}")
    
    diarization_segments = []
    if len(time_stamps) > 0:
        start_time = time_stamps[0]
        current_label = cluster_labels[0]
        
        for i in range(1, len(time_stamps)):
            if cluster_labels[i] != current_label:
                end_time = time_stamps[i]
                diarization_segments.append({
                    'start': start_time,
                    'end': end_time,
                    'label': current_label
                })
                start_time = time_stamps[i]
                current_label = cluster_labels[i]

        final_end_time = librosa.get_duration(y=y, sr=sr)
        diarization_segments.append({
            'start': start_time,
            'end': final_end_time,
            'label': current_label
        })
        
    
    plt.figure(figsize=(15, 7))
    dendrogram(
        Z,
        truncate_mode='lastp',
        p=50,
        leaf_rotation=90.,
        leaf_font_size=8.,
        show_contracted=True,
    )
    plt.title('AHC Dendrogram')
    plt.xlabel('Sample index or (Cluster Size)')
    plt.ylabel('Distance (Cosine Affinity)')
    plt.tight_layout()
    plt.savefig('ahc_dendrogram.png')
    plt.show() 

    labels = [f"Cluster {i}" for i in cluster_labels]
    times = np.array(time_stamps)
    
    plt.figure(figsize=(15, 4))
    plt.scatter(times, cluster_labels, c=cluster_labels, cmap='viridis', s=20)
    plt.yticks(np.unique(cluster_labels), [f'Cluster {l}' for l in np.unique(cluster_labels)])
    plt.xlabel("Time (seconds)")
    plt.title(f"AHC Diarization Output")
    plt.grid(axis='x', alpha=0.5)
    plt.ylim(-0.5, n_clusters - 0.5)
    plt.tight_layout()
    plt.show()
    return diarization_segments



audio_file = r"C:\Users\GANAPATHY\langdiar\clips\mixed_audio.mp3"



segments = ahc_language_diarization(
        audio_file, 
        n_clusters=2, 
        window_len=2.0, 
        hop_size=0.5, 
        n_mfcc=13,
        pca_dims=8 
    )

print("\n--- Final Diarization Segments ---")
for seg in segments:
        start_t = f"{seg['start']:.2f}"
        end_t = f"{seg['end']:.2f}"
        print(f"[ {start_t}s - {end_t}s ] : Cluster {seg['label']}")
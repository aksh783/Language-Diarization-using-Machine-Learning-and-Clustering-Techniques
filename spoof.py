import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np



def extract_cqcc_features(audio_path, sr=16000, n_coeffs=20):
    try:
        y, sr = librosa.load(audio_path, sr=sr)
       
        features = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_coeffs)
       
        features = librosa.power_to_db(features, ref=np.max)
        return features, sr
    except Exception as e:
        print(f"Error loading {audio_path}: {e}")
        return None, None

def generate_cqcc_spoof_plot(path_natural, path_synthetic):
    
    
    cqcc_natural, sr_n = extract_cqcc_features(path_natural)
    cqcc_synthetic, sr_s = extract_cqcc_features(path_synthetic)

    if cqcc_natural is None or cqcc_synthetic is None:
        return

    print("Successfully extracted features for comparison.")

    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    
    img1 = librosa.display.specshow(cqcc_natural, x_axis='time', ax=axes[0], sr=sr_n, 
                                    cmap='magma')
    axes[0].set(title='Natural Speech: MFCC Features')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Cepstral Coefficient Index')

    img2 = librosa.display.specshow(cqcc_synthetic, x_axis='time', ax=axes[1], sr=sr_s, 
                                    cmap='magma')
    axes[1].set(title='Synthetic Speech: MFCC Features (Spoof)')
    axes[1].set_xlabel('Time (s)')
    
    fig.colorbar(img1, ax=axes, format="%+2.f dB")
    
    fig.tight_layout()
    
    plt.savefig('cqcc_spoof.png') 
    print("Saved required plot: cqcc_spoof.png")


audio_natural = r"C:\Users\GANAPATHY\langdiar\clips\clips_english\common_voice_en_42696402.mp3"
audio_synthetic = r"C:\Users\GANAPATHY\Downloads\ElevenLabs_2025-11-26T03_43_50_Rachel_pre_sp100_s50_sb75_se0_b_m2.mp3"


generate_cqcc_spoof_plot(audio_natural, audio_synthetic)
import librosa
import numpy as np
import pickle
import joblib

with open("lang_model.pkl", "rb") as f:
    model = pickle.load(f)

scaler = joblib.load("scaler.pkl")
le = joblib.load("label_encoder.pkl")

test_audio = r"C:\Users\GANAPATHY\langdiar\clips\clips_hindi\common_voice_hi_43100793.mp3"

y, sr = librosa.load(test_audio, sr=None)
mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
mfcc_mean = np.mean(mfcc, axis=1).reshape(1, -1)

mfcc_scaled = scaler.transform(mfcc_mean)

pred = model.predict(mfcc_scaled)
pred_lang = le.inverse_transform(pred)[0]

pred_lang_clean = pred_lang.replace("clips_", "").capitalize()

print(f"Predicted language: {pred_lang_clean}")

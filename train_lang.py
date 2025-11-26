import os
import numpy as np
import librosa

data = []
labels = []

base_dir = r"C:\Users\GANAPATHY\langdiar\clips"  
for lang in os.listdir(base_dir):
    lang_path = os.path.join(base_dir, lang)
    if not os.path.isdir(lang_path):
        continue

    for fname in os.listdir(lang_path):
        if not fname.endswith(".mp3") and not fname.endswith(".wav"):
            continue

        fpath = os.path.join(lang_path, fname)
        y, sr = librosa.load(fpath, sr=None)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc, axis=1)       
        data.append(mfcc_mean)
        labels.append(lang)

X = np.array(data)
y = np.array(labels)

print("Feature matrix shape:", X.shape)
print("Labels:", np.unique(y))

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

le = LabelEncoder()
y_encoded = le.fit_transform(y)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42
)

clf = SVC(kernel='linear', C=1, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

import joblib
joblib.dump(clf, "lang_classifier.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(le, "label_encoder.pkl")

import pickle

import pickle

with open("lang_model.pkl", "wb") as f:
    pickle.dump(clf, f)

print(" Model saved as lang_model.pkl")


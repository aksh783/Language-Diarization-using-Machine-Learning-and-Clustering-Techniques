import librosa
import numpy as np
import joblib
import pickle
import os
import csv

AUDIO = r"C:\Users\GANAPATHY\langdiar\clips\mixed_audio.mp3"  
FRAME_DUR = 1.5    
HOP_DUR = 0.75     
MIN_SEG_SEC = 2.0  
GAP_MERGE_SEC = 0.5  

def load_obj(path_joblib, path_pickle):
    if os.path.exists(path_joblib):
        return joblib.load(path_joblib)
    if os.path.exists(path_pickle):
        with open(path_pickle, "rb") as f:
            return pickle.load(f)
    raise FileNotFoundError(f"Neither {path_joblib} nor {path_pickle} found.")

model = None
try:
    model = load_obj("lang_classifier.pkl", "lang_model.pkl")
except FileNotFoundError:
    if os.path.exists("lang_model.pkl"):
        with open("lang_model.pkl", "rb") as f:
            model = pickle.load(f)
    else:
        raise

scaler = None
le = None
if os.path.exists("scaler.pkl"):
    try:
        scaler = joblib.load("scaler.pkl")
    except Exception:
        with open("scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
elif os.path.exists("scaler_pickle.pkl"):
    with open("scaler_pickle.pkl", "rb") as f:
        scaler = pickle.load(f)
else:
    raise FileNotFoundError("scaler.pkl not found.")

if os.path.exists("label_encoder.pkl"):
    try:
        le = joblib.load("label_encoder.pkl")
    except Exception:
        with open("label_encoder.pkl", "rb") as f:
            le = pickle.load(f)
elif os.path.exists("label_encoder_pickle.pkl"):
    with open("label_encoder_pickle.pkl", "rb") as f:
        le = pickle.load(f)
else:
    raise FileNotFoundError("label_encoder.pkl not found.")

def predict_proba_safe(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        scores = np.asarray(scores)
        if scores.ndim == 1:
            scores = np.vstack([-scores, scores]).T
        exp = np.exp(scores - np.max(scores, axis=1, keepdims=True))
        probs = exp / np.sum(exp, axis=1, keepdims=True)
        return probs
    preds = model.predict(X)
    probs = np.zeros((len(preds), len(le.classes_)))
    for i, p in enumerate(preds):
        idx = list(le.classes_).index(p)
        probs[i, idx] = 1.0
    return probs

y, sr = librosa.load(AUDIO, sr=None)
duration = librosa.get_duration(y=y, sr=sr)
print(f"Loaded {AUDIO} | Duration: {duration:.2f} s | SR: {sr}")

frame_len = int(FRAME_DUR * sr)
hop_len = int(HOP_DUR * sr)
starts = list(range(0, max(1, len(y) - frame_len + 1), hop_len))
if len(starts) == 0:
    starts = [0]

rms = []
for s in starts:
    seg = y[s:s+frame_len]
    if len(seg) == 0:
        rms.append(0.0)
    else:
        rms.append(np.sqrt(np.mean(seg**2)))
rms = np.array(rms)
sil_thresh = 0.05 * rms.max() if rms.size else 0.0

time_centers = []
probs_list = []
for i, s in enumerate(starts):
    seg = y[s:s+frame_len]
    center = s / sr + FRAME_DUR/2.0
    time_centers.append(center)

    if rms[i] < sil_thresh:
        probs_list.append(np.ones(len(le.classes_)) / len(le.classes_))
        continue

    mfcc = librosa.feature.mfcc(y=seg, sr=sr, n_mfcc=13)
    feat = np.mean(mfcc, axis=1).reshape(1, -1)
    feat_scaled = scaler.transform(feat)
    probs = predict_proba_safe(model, feat_scaled)[0]
    probs_list.append(probs)

probs_arr = np.vstack(probs_list)  

k = 3  
pad = k // 2
smoothed = np.zeros_like(probs_arr)
for c in range(probs_arr.shape[1]):
    col = probs_arr[:, c]
    sm_col = np.copy(col)
    for i in range(len(col)):
        st = max(0, i-pad)
        en = min(len(col), i+pad+1)
        sm_col[i] = np.median(col[st:en])
    smoothed[:, c] = sm_col

pred_idx = np.argmax(smoothed, axis=1)
pred_labels = [le.classes_[i] for i in pred_idx]
pred_labels_clean = [lab.replace("clips_", "").capitalize() for lab in pred_labels]

segments = []
for i, lab in enumerate(pred_labels_clean):
    start = starts[i] / sr
    end = min((starts[i] + frame_len) / sr, duration)
    if segments and segments[-1][2] == lab:
        prev_s, prev_e, prev_l = segments[-1]
        segments[-1] = (prev_s, end, prev_l)
    else:
        segments.append((start, end, lab))

def merge_short_segments(segs, min_dur):
    out = []
    i = 0
    while i < len(segs):
        s, e, l = segs[i]
        dur = e - s
        if dur >= min_dur:
            out.append((s, e, l))
            i += 1
            continue
        prev_idx = len(out) - 1
        next_idx = i + 1 if i + 1 < len(segs) else None
        if prev_idx >= 0 and next_idx is not None:
            ps, pe, pl = out[prev_idx]
            ns, ne, nl = segs[next_idx]
            if (pe - ps) >= (ne - ns):
                out[prev_idx] = (ps, e, pl)
            else:
                segs[next_idx] = (s, ne, nl)
        elif prev_idx >= 0:
            ps, pe, pl = out[prev_idx]
            out[prev_idx] = (ps, e, pl)
        elif next_idx is not None:
            ns, ne, nl = segs[next_idx]
            segs[next_idx] = (s, ne, nl)
        else:
            out.append((s, e, l))
        i += 1
    merged = []
    for s,e,l in out:
        if merged and merged[-1][2] == l:
            ps, pe, pl = merged[-1]
            merged[-1] = (ps, e, pl)
        else:
            merged.append((s,e,l))
    return merged

segments = merge_short_segments(segments, MIN_SEG_SEC)

final = []
for s,e,l in segments:
    if final and final[-1][2] == l and (s - final[-1][1]) <= GAP_MERGE_SEC:
        ps, pe, pl = final[-1]
        final[-1] = (ps, e, pl)
    else:
        final.append((s,e,l))

tidy = []
for s,e,l in final:
    if tidy and tidy[-1][2] == l:
        ps, pe, pl = tidy[-1]
        tidy[-1] = (ps, e, pl)
    else:
        tidy.append((s,e,l))

print("\n Language Segments (strict):\n")
for s,e,l in tidy:
    print(f"{s:6.2f}s – {e:6.2f}s → {l}")

out_csv = "predicted_language_segments_strict.csv"
with open(out_csv, "w", newline="", encoding="utf-8") as cf:
    writer = csv.writer(cf)
    writer.writerow(["start_s", "end_s", "language"])
    for s,e,l in tidy:
        writer.writerow([f"{s:.3f}", f"{e:.3f}", l])
print(f"\nSaved segments to {out_csv}")

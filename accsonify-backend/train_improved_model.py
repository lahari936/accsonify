"""
Improved SVM training script with better feature extraction and analysis
"""

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import joblib

import whisper
import torch
import librosa

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.svm import SVC

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

# ============================================================================
# CONFIGURATION - Update these paths
# ============================================================================

BASE_PATH = "/content/drive/MyDrive/speech-accent-archive"
AUDIO_DIR = BASE_PATH + "/recordings/recordings"
CSV_PATH = BASE_PATH + "/speakers_all.csv"

# ============================================================================
# Load and prepare data
# ============================================================================

df = pd.read_csv(CSV_PATH)

def map_region(country):
    if not isinstance(country, str):
        return None

    c = country.lower().strip()

    # South Asia
    if c in ["india", "pakistan", "bangladesh", "sri lanka", "nepal"]:
        return "South_Asia"

    # East Asia
    if c in ["china", "japan", "south korea", "north korea", "taiwan"]:
        return "East_Asia"

    # Middle East
    if c in ["iran", "iraq", "israel", "palestine", "jordan",
             "saudi arabia", "uae", "oman", "yemen", "kuwait", "qatar"]:
        return "Middle_East"

    # Africa
    if c in ["nigeria", "ghana", "kenya", "ethiopia", "south africa",
             "uganda", "tanzania"]:
        return "Africa"

    return None

df["region"] = df["country"].apply(map_region)
df = df[df["region"].notna()].reset_index(drop=True)

print("\nRegion Distribution:")
print(df["region"].value_counts())
print(f"Total samples: {len(df)}")

# Get audio paths
import glob

def get_audio_path(row):
    pattern = os.path.join(AUDIO_DIR, f"{row['filename']}*.mp3")
    matches = glob.glob(pattern)
    return matches[0] if len(matches) > 0 else None

df["audio_path"] = df.apply(get_audio_path, axis=1)
df = df[df["audio_path"].notna()].reset_index(drop=True)

print(f"Final samples with audio: {len(df)}")

# ============================================================================
# Load Whisper and extract features
# ============================================================================

whisper_model = whisper.load_model("base").to(device)
whisper_model.eval()

def extract_whisper_embedding(audio_path):
    """Extract Whisper encoder embedding"""
    try:
        audio = whisper.load_audio(audio_path)
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio).to(device)

        with torch.no_grad():
            enc = whisper_model.encoder(mel.unsqueeze(0))

        return enc.mean(dim=1).cpu().numpy().squeeze()
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None

def extract_acoustic_features(audio_path):
    """Extract handcrafted acoustic features"""
    try:
        y, sr = librosa.load(audio_path, sr=16000, mono=True)
        
        # Pitch features
        pitch = librosa.yin(y, fmin=50, fmax=400)
        pitch_mean = np.nanmean(pitch)
        pitch_std = np.nanstd(pitch)
        
        # Energy features
        energy = librosa.feature.rms(y=y)[0]
        energy_mean = np.mean(energy)
        energy_std = np.std(energy)
        
        # MFCC features
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)
        
        # Spectral features
        spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        spec_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        
        features = np.concatenate([
            [pitch_mean, pitch_std, energy_mean, energy_std],
            mfcc_mean, mfcc_std,
            [np.mean(spec_centroid), np.mean(spec_rolloff)]
        ])
        
        return features
    except Exception as e:
        print(f"Error extracting acoustic features from {audio_path}: {e}")
        return None

# Extract features
print("\nExtracting features...")
X_whisper, X_acoustic, y = [], [], []

for _, row in tqdm(df.iterrows(), total=len(df)):
    audio_path = row["audio_path"]
    
    # Whisper embedding
    emb = extract_whisper_embedding(audio_path)
    if emb is None:
        continue
    
    # Acoustic features
    acous = extract_acoustic_features(audio_path)
    if acous is None:
        continue
    
    X_whisper.append(emb)
    X_acoustic.append(acous)
    y.append(row["region"])

X_whisper = np.array(X_whisper)
X_acoustic = np.array(X_acoustic)
y = np.array(y)

print(f"Final dataset: {X_whisper.shape[0]} samples")
print(f"Whisper embedding shape: {X_whisper.shape}")
print(f"Acoustic features shape: {X_acoustic.shape}")

# ============================================================================
# Combine features and normalize
# ============================================================================

# Normalize acoustic features
scaler = StandardScaler()
X_acoustic = scaler.fit_transform(X_acoustic)

# Combine features (Whisper embeddings + acoustic features)
X = np.concatenate([X_whisper, X_acoustic], axis=1)

print(f"Combined feature shape: {X.shape}")

# ============================================================================
# Train/Val Split
# ============================================================================

le = LabelEncoder()
y_enc = le.fit_transform(y)

X_train, X_val, y_train, y_val = train_test_split(
    X, y_enc, test_size=0.2, stratify=y_enc, random_state=42
)

print(f"\nTraining set: {X_train.shape[0]}")
print(f"Validation set: {X_val.shape[0]}")

# ============================================================================
# Train SVM with tuned hyperparameters
# ============================================================================

print("\nTraining SVM...")
clf = SVC(
    kernel="rbf",
    C=100,  # Increased regularization
    gamma="scale",
    probability=True,
    class_weight="balanced"  # Handle class imbalance
)

clf.fit(X_train, y_train)

# ============================================================================
# Evaluation
# ============================================================================

y_pred = clf.predict(X_val)

print("\n" + "="*50)
print("VALIDATION RESULTS")
print("="*50)

acc = accuracy_score(y_val, y_pred)
f1 = f1_score(y_val, y_pred, average="macro")

print(f"Accuracy: {acc:.4f}")
print(f"Macro F1-Score: {f1:.4f}")

print("\nClass Mapping:")
for i, label in enumerate(le.classes_):
    print(f"  {i} → {label}")

print("\nConfusion Matrix:")
cm = confusion_matrix(y_val, y_pred)
print(cm)

print("\nClassification Report:")
print(classification_report(y_val, y_pred, target_names=le.classes_))

# Cross-validation
cv_scores = cross_val_score(clf, X, y_enc, cv=5)
print(f"\n5-Fold CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# ============================================================================
# Save models
# ============================================================================

joblib.dump(clf, "svm_model.pkl")
joblib.dump(le, "label_encoder.pkl")
joblib.dump(scaler, "acoustic_scaler.pkl")

print("\nModels saved!")
print("  - svm_model.pkl")
print("  - label_encoder.pkl")
print("  - acoustic_scaler.pkl")

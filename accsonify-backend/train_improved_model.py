"""
SVM training script using Whisper embeddings only
Based on working Colab implementation
"""

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import joblib

import whisper
import torch

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report
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
    
    # North America (Native English speakers - American accent)
    if c in ["usa", "united states", "canada"]:
        return "North_America"
    
    # Europe (Native English speakers - British accent)  
    if c in ["uk", "united kingdom", "england", "scotland", "wales", "ireland"]:
        return "Europe"

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
# Load Whisper and extract embeddings
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

# Extract features
print("\nExtracting Whisper embeddings...")
X, y = [], []

for _, row in tqdm(df.iterrows(), total=len(df)):
    audio_path = row["audio_path"]
    
    emb = extract_whisper_embedding(audio_path)
    if emb is None:
        continue
    
    X.append(emb)
    y.append(row["region"])

X = np.array(X)
y = np.array(y)

print(f"Final dataset: {X.shape[0]} samples")
print(f"Feature shape: {X.shape}")

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
    C=10,
    gamma="scale",
    probability=True,
    class_weight="balanced"
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

print("\nClassification Report:")
print(classification_report(y_val, y_pred, target_names=le.classes_))

# ============================================================================
# Save models
# ============================================================================

joblib.dump(clf, "svm_model.pkl")
joblib.dump(le, "label_encoder.pkl")

print("\nModels saved!")
print("  - svm_model.pkl")
print("  - label_encoder.pkl")

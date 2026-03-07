"""Train the exact Whisper-embedding + SVM pipeline used in the provided code."""

import argparse
import glob
import os

import joblib
import numpy as np
import pandas as pd
import torch
import whisper
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from tqdm import tqdm


def map_region(country):
    if not isinstance(country, str):
        return None

    c = country.lower().strip()

    if c in ["india", "pakistan", "bangladesh", "sri lanka", "nepal"]:
        return "South_Asia"

    if c in ["china", "japan", "south korea", "north korea", "taiwan"]:
        return "East_Asia"

    if c in [
        "iran",
        "iraq",
        "israel",
        "palestine",
        "jordan",
        "saudi arabia",
        "uae",
        "oman",
        "yemen",
        "kuwait",
        "qatar",
    ]:
        return "Middle_East"

    if c in ["nigeria", "ghana", "kenya", "ethiopia", "south africa", "uganda", "tanzania"]:
        return "Africa"

    return None


def parse_args():
    parser = argparse.ArgumentParser(description="Train SVM using Whisper encoder embeddings")
    parser.add_argument(
        "--base-path",
        default=os.getenv("ACCSONIFY_DATASET_BASE", ""),
        help=(
            "Dataset root folder. Expected layout: <base>/recordings/recordings and <base>/speakers_all.csv"
        ),
    )
    parser.add_argument(
        "--audio-dir",
        default=os.getenv("ACCSONIFY_AUDIO_DIR", ""),
        help="Optional explicit audio folder override.",
    )
    parser.add_argument(
        "--csv-path",
        default=os.getenv("ACCSONIFY_CSV_PATH", ""),
        help="Optional explicit CSV path override.",
    )
    return parser.parse_args()


def resolve_paths(args):
    base = args.base_path.strip()
    audio_dir = args.audio_dir.strip() if args.audio_dir else ""
    csv_path = args.csv_path.strip() if args.csv_path else ""

    if not audio_dir:
        if not base:
            raise ValueError(
                "Dataset path missing. Use --base-path or set ACCSONIFY_DATASET_BASE."
            )
        audio_dir = os.path.join(base, "recordings", "recordings")

    if not csv_path:
        if not base:
            raise ValueError("CSV path missing. Use --csv-path or set ACCSONIFY_CSV_PATH.")
        csv_path = os.path.join(base, "speakers_all.csv")

    if not os.path.isdir(audio_dir):
        raise FileNotFoundError(f"Audio dir not found: {audio_dir}")
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    return audio_dir, csv_path


def main():
    args = parse_args()
    audio_dir, csv_path = resolve_paths(args)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)
    print("AUDIO_DIR:", audio_dir)
    print("CSV_PATH:", csv_path)

    df = pd.read_csv(csv_path)
    df["region"] = df["country"].apply(map_region)
    df = df[df["region"].notna()].reset_index(drop=True)

    print(df["region"].value_counts())
    print("Total samples:", len(df))

    def get_audio_path(row):
        pattern = os.path.join(audio_dir, f"{row['filename']}*.mp3")
        matches = glob.glob(pattern)
        if len(matches) == 0:
            return None
        return matches[0]

    df["audio_path"] = df.apply(get_audio_path, axis=1)
    df = df[df["audio_path"].notna()].reset_index(drop=True)
    print("Final samples:", len(df))

    whisper_model = whisper.load_model("base").to(device)
    whisper_model.eval()

    def extract_whisper_embedding(audio_path):
        audio = whisper.load_audio(audio_path)
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio).to(device)
        with torch.no_grad():
            enc = whisper_model.encoder(mel.unsqueeze(0))
        return enc.mean(dim=1).cpu().numpy().squeeze()

    X = []
    y = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        audio_path = row["audio_path"]
        X.append(extract_whisper_embedding(audio_path))
        y.append(row["region"])

    X = np.array(X)
    y = np.array(y)
    print(X.shape, y.shape)

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y_enc, test_size=0.2, stratify=y_enc, random_state=42
    )

    clf = SVC(kernel="rbf", C=10, gamma="scale", probability=True)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred, average="macro")

    print("Validation Accuracy:", acc)
    print("Macro F1:", f1)

    for i, label in enumerate(le.classes_):
        print(i, "->", label)

    out_dir = os.path.dirname(__file__)
    svm_out = os.path.join(out_dir, "svm_model.pkl")
    le_out = os.path.join(out_dir, "label_encoder.pkl")
    joblib.dump(clf, svm_out)
    joblib.dump(le, le_out)
    print("Saved:", svm_out)
    print("Saved:", le_out)


if __name__ == "__main__":
    main()

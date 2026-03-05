import os
import torch
import librosa
import numpy as np
import edge_tts
import asyncio
import joblib

# Optional whisper import, handle if not installed yet during dev
try:
    import whisper
except ImportError:
    whisper = None

# ACCENT MAP
region_to_accent = {
    "South_Asia": "indian",
    "Middle_East": "british",
    "Africa": "african",
    "East_Asia": "american"
}

accent_voice_map = {
    "indian": {
        "male": "en-IN-PrabhatNeural",
        "female": "en-IN-NeerjaNeural"
    },
    "british": {
        "male": "en-GB-RyanNeural",
        "female": "en-GB-SoniaNeural"
    },
    "american": {
        "male": "en-US-GuyNeural",
        "female": "en-US-JennyNeural"
    },
    "african": {
        "male": "en-NG-AbeoNeural",
        "female": "en-NG-ChimaNeNeural"
    }
}

class AccentModelManager:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        self.clf = None
        self.le = None
        self.scaler = None
        self.whisper_model = None
        self.mock_mode_svm = False
        self.mock_mode_whisper = False

    def load_models(self):
        # Load SVM
        if os.path.exists("svm_model.pkl") and os.path.exists("label_encoder.pkl"):
            try:
                self.clf = joblib.load("svm_model.pkl")
                self.le = joblib.load("label_encoder.pkl")
                
                # Try to load acoustic scaler if it exists
                if os.path.exists("acoustic_scaler.pkl"):
                    self.scaler = joblib.load("acoustic_scaler.pkl")
                
                print("Loaded real SVM models.")
            except Exception as e:
                print(f"Error loading SVM models: {e}. Falling back to mock.")
                self.mock_mode_svm = True
        else:
            print("SVM models not found. Running in MOCK mode for classification.")
            self.mock_mode_svm = True

        # Load Whisper
        if whisper:
            try:
                print("Loading Whisper base model...")
                self.whisper_model = whisper.load_model("base").to(self.device)
                self.whisper_model.eval()
                print("Loaded Whisper model.")
            except Exception as e:
                print(f"Error loading Whisper model: {e}. Falling back to mock.")
                self.mock_mode_whisper = True
        else:
            print("Whisper library not available. Running in MOCK mode for transcription.")
            self.mock_mode_whisper = True

    def extract_whisper_embedding(self, audio_path):
        if self.mock_mode_whisper or self.whisper_model is None:
            return np.random.rand(512)
            
        audio = whisper.load_audio(audio_path)
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio).to(self.device)

        with torch.no_grad():
            enc = self.whisper_model.encoder(mel.unsqueeze(0))

        return enc.mean(dim=1).cpu().numpy().squeeze()
    
    def extract_acoustic_features(self, audio_path):
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
            print(f"Error extracting acoustic features: {e}")
            return np.zeros(32)  # Return zeros if extraction fails

    def predict_accent(self, audio_path):
        if self.mock_mode_svm or self.clf is None:
            # Mock prediction
            regions = list(region_to_accent.keys())
            detected = np.random.choice(regions)
            confidence = float(np.random.uniform(0.6, 0.99))
            return detected, confidence

        # Extract combined features
        whisper_emb = self.extract_whisper_embedding(audio_path)
        acoustic_feat = self.extract_acoustic_features(audio_path)
        
        # Normalize acoustic features if scaler is available
        if self.scaler is not None:
            acoustic_feat = self.scaler.transform([acoustic_feat])[0]
        
        # Combine features
        combined_features = np.concatenate([whisper_emb, acoustic_feat]).reshape(1, -1)
        
        pred = self.clf.predict(combined_features)
        
        # Try to get probabilities if supported
        try:
            probs = self.clf.predict_proba(combined_features)[0]
            confidence = float(np.max(probs))
        except:
            confidence = 0.85 # Default fallback
            
        label = self.le.inverse_transform(pred)[0]
        return label, confidence

    def transcribe(self, audio_path):
        if self.mock_mode_whisper or self.whisper_model is None:
            return "This is a mock transcription because Whisper is not loaded."
            
        result = self.whisper_model.transcribe(audio_path)
        return result["text"]


def extract_speaker_style(audio_path):
    try:
        y, sr = librosa.load(audio_path, sr=16000, mono=True)
        pitch_values = librosa.yin(y, fmin=50, fmax=300)
        avg_pitch = np.nanmean(pitch_values)
        if np.isnan(avg_pitch):
            avg_pitch = 150 # fallback

        frame_energy = librosa.feature.rms(y=y)[0]
        speaking_rate = np.mean(frame_energy) * 1000

        return avg_pitch, speaking_rate
    except Exception as e:
        print(f"Error extracting style: {e}")
        return 150.0, 1.5

def detect_gender_from_audio(audio_path):
    try:
        y, sr = librosa.load(audio_path, sr=16000, mono=True)
        pitch_values = librosa.yin(y, fmin=50, fmax=300)
        pitch = np.nanmean(pitch_values)
        if np.isnan(pitch):
            return "male" # fallback
        return "female" if pitch > 165 else "male"
    except Exception as e:
        print(f"Error detecting gender: {e}")
        return "male"

async def convert_accent_tts(text, sample_audio, target_accent, output_file):
    pitch, speaking_rate = extract_speaker_style(sample_audio)
    gender = detect_gender_from_audio(sample_audio)

    # Fallback to american if target_accent is unknown
    if target_accent not in accent_voice_map:
        target_accent = "american"
        
    voice = accent_voice_map[target_accent][gender]

    rate_value = int((speaking_rate - 1.5) * 5)
    rate_value = max(min(rate_value, 10), -10)
    rate = f"{rate_value:+d}%"

    pitch_value = int((pitch - 150) / 150 * 5)
    pitch_value = max(min(pitch_value, 5), -5)
    pitch_shift = f"{pitch_value:+d}Hz"

    communicate = edge_tts.Communicate(
        text=text,
        voice=voice,
        rate=rate,
        pitch=pitch_shift
    )

    await communicate.save(output_file)
    return {
        "voice_used": voice,
        "gender_detected": gender,
        "output_file": output_file
    }

# Initialize a global manager
model_manager = AccentModelManager()

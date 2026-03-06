import os
import shutil
import numpy as np

# Optional heavy/runtime-sensitive imports for serverless compatibility.
try:
    import torch
except Exception:
    torch = None

try:
    import librosa
except Exception:
    librosa = None

try:
    import edge_tts
except Exception:
    edge_tts = None

try:
    import joblib
except Exception:
    joblib = None

# Optional whisper import, handle if not installed yet during dev
try:
    import whisper
except ImportError:
    whisper = None

# ACCENT MAP
# Maps detected geographic regions to accent categories for TTS
# Note: Model detects speaker's origin region, we map to closest available TTS accent
region_to_accent = {
    "South_Asia": "indian",      # Indian subcontinent speakers
    "East_Asia": "british",      # East Asian English speakers (closest to British RP)
    "Middle_East": "british",    # Middle Eastern English speakers
    "Africa": "african",         # African English speakers
    "North_America": "american", # Native American English speakers
    "Europe": "british"          # Native British English speakers
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
        self.device = "cpu"
        if torch is not None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        self.clf = None
        self.le = None
        self.whisper_model = None
        self.mock_mode_svm = False
        self.mock_mode_whisper = False
        self.allow_mock_mode = os.getenv("ALLOW_MOCK_MODE", "false").lower() == "true"

    def load_models(self):
        """Load models with error handling"""
        base_dir = os.path.dirname(__file__)
        svm_model_path = os.path.join(base_dir, "svm_model.pkl")
        label_encoder_path = os.path.join(base_dir, "label_encoder.pkl")

        # Load SVM
        if joblib and os.path.exists(svm_model_path) and os.path.exists(label_encoder_path):
            try:
                self.clf = joblib.load(svm_model_path)
                self.le = joblib.load(label_encoder_path)
                print("✓ Loaded real SVM models.")
            except Exception as e:
                print(f"✗ Error loading SVM models: {e}. Running in MOCK mode.")
                self.mock_mode_svm = True
        else:
            print("✗ SVM models or joblib not available. Running in MOCK mode for classification.")
            self.mock_mode_svm = True

        # Load Whisper
        if whisper and torch is not None:
            try:
                print("Loading Whisper base model...")
                self.whisper_model = whisper.load_model("base", device=self.device)
                self.whisper_model.eval()
                print("✓ Loaded Whisper model.")
            except Exception as e:
                print(f"✗ Error loading Whisper model: {e}. Running in MOCK mode.")
                self.mock_mode_whisper = True
        else:
            print("✗ Whisper library not available. Running in MOCK mode for transcription.")
            self.mock_mode_whisper = True

        if (self.mock_mode_svm or self.mock_mode_whisper) and not self.allow_mock_mode:
            missing = []
            if self.mock_mode_svm:
                missing.append("SVM classifier artifacts (svm_model.pkl/label_encoder.pkl) or joblib")
            if self.mock_mode_whisper:
                missing.append("Whisper runtime dependencies/model")
            raise RuntimeError(
                "Real inference is required but model stack is incomplete: " + ", ".join(missing)
            )

    def extract_whisper_embedding(self, audio_path):
        """Extract Whisper encoder embedding - matches training script exactly"""
        if self.mock_mode_whisper or self.whisper_model is None:
            return np.random.rand(512)
        
        try:
            audio = whisper.load_audio(audio_path)
            audio = whisper.pad_or_trim(audio)
            mel = whisper.log_mel_spectrogram(audio).to(self.device)

            with torch.no_grad():
                enc = self.whisper_model.encoder(mel.unsqueeze(0))

            return enc.mean(dim=1).cpu().numpy().squeeze()
        except Exception as e:
            print(f"Error extracting Whisper embedding: {e}")
            return np.random.rand(512)

    def predict_accent(self, audio_path):
        """Predict accent using ONLY Whisper embeddings - matches training script"""
        if self.mock_mode_svm or self.clf is None:
            if not self.allow_mock_mode:
                raise RuntimeError("Accent model unavailable: real SVM classifier not loaded")
            regions = list(region_to_accent.keys())
            detected = np.random.choice(regions)
            confidence = float(np.random.uniform(0.6, 0.99))
            return detected, confidence

        # Extract Whisper embedding ONLY (no acoustic features)
        embedding = self.extract_whisper_embedding(audio_path)
        
        # Predict using the embedding
        pred = self.clf.predict([embedding])
        
        # Get probabilities for confidence
        try:
            probs = self.clf.predict_proba([embedding])[0]
            confidence = float(np.max(probs))
        except:
            confidence = 0.85
            
        label = self.le.inverse_transform(pred)[0]
        return label, confidence

    def transcribe(self, audio_path):
        if self.mock_mode_whisper or self.whisper_model is None:
            if not self.allow_mock_mode:
                raise RuntimeError("Transcription model unavailable: Whisper is not loaded")
            return "This is a mock transcription because Whisper is not loaded."
            
        result = self.whisper_model.transcribe(audio_path)
        return result["text"]


def extract_speaker_style(audio_path):
    if librosa is None:
        return 150.0, 1.5

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
    if librosa is None:
        return "male"

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
    if edge_tts is None:
        raise RuntimeError("TTS dependency unavailable: edge-tts is not installed")

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

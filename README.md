# SpanishSlangSTT

**Fine-tuned Speech-to-Text for Regional Spanish Slang and Informal Speech**

A custom STT pipeline optimized for recognizing regional Spanish slang and informal speech, featuring Whisper fine-tuning, a production-ready API, and comprehensive evaluation tools.

## Why This Project?

Standard STT models often struggle with informal speech, slang, and regional variations. This project addresses:

- **Regional slang**: Mexican ("chido", "padre", "neta"), Argentinian ("che", "boludo", "pibe"), Spanish ("mola", "guay", "

")
- **Informal expressions**: Everyday colloquialisms and casual speech patterns
- **Code-switching**: Natural mixing of formal and informal registers

This project fine-tunes Whisper on Spanish slang data to improve recognition accuracy for informal speech.

## Features

- **Whisper Fine-tuning Pipeline**: Train on your own Spanish audio data
- **FastAPI Transcription Service**: Production-ready REST API with word-level timestamps
- **Evaluation Suite**: WER/CER metrics, model comparison benchmarks
- **Synthetic Data Generation**: Bootstrap training data using ElevenLabs TTS
- **Demo UI**: Browser-based recording and transcription interface

## Project Structure

```
spanish-slang-stt/
├── src/
│   ├── data/           # Data loading, preprocessing, synthetic generation
│   ├── training/       # Whisper fine-tuning pipeline
│   ├── evaluation/     # WER/CER metrics, benchmarking
│   └── api/            # FastAPI transcription service
├── data/
│   ├── raw/            # Original audio files
│   ├── processed/      # Preprocessed audio (16kHz mono)
│   ├── transcripts/    # Text transcriptions
│   └── splits/         # Train/val/test manifests
├── demo/               # Browser-based demo UI
├── models/             # Saved model checkpoints
├── configs/            # Training configurations
└── notebooks/          # Exploration and analysis
```

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/spanish-slang-stt.git
cd spanish-slang-stt

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the API (with base Whisper)

```bash
# Start the transcription API
uvicorn src.api.app:app --reload --port 8000

# Test it
curl -X POST http://localhost:8000/transcribe/upload \
  -F "file=@your_audio.wav" \
  -F "language=es"
```

### 3. Open the Demo UI

```bash
# Serve the demo
python -m http.server 8080 --directory demo

# Open http://localhost:8080 in your browser
```

## Training Your Own Model

### Step 1: Prepare Your Data

Place your Spanish audio data in `data/raw/` with corresponding transcriptions.

| Dataset | Language | Size | Hours | License |
|---------|----------|------|-------|---------|
| [Spanish Conversational](https://magichub.com/datasets/spanish-conversational-speech-corpus/) | Spanish | 514 MB | 5.5h | CC BY-NC-ND 4.0 |
| Your custom data | Spanish | - | - | - |

```bash
# List available datasets
python scripts/download_datasets.py --list

# Download Spanish conversational corpus (requires MagicHub account)
python scripts/download_datasets.py --dataset spanish
```

> **Note**: The Spanish Conversational Corpus requires a free MagicHub account. The script will provide instructions.

### Step 2: Preprocess & Create Manifests

```bash
# Process downloaded data and create train/val/test splits
python scripts/prepare_datasets.py \
    --input-dir data/raw \
    --output-dir data/processed \
    --manifests-dir data/splits

# Process with custom split ratios
python scripts/prepare_datasets.py \
    --train-ratio 0.85 \
    --val-ratio 0.10 \
    --workers 8
```

This will:
- Resample all audio to 16kHz mono
- Normalize audio levels
- Trim silence
- Create JSON manifests for training

**Manifest format** (`data/splits/train.json`):
```json
[
  {
    "audio_path": "data/processed/spanish-conversational/sample_001.wav",
    "transcript": "Oye, ¿qué onda?",
    "duration": 2.5,
    "language": "es",
    "speaker_id": "spk_001",
    "dataset": "spanish-conversational"
  }
]
```

### Step 3: Fine-tune Whisper

```python
from src.training import WhisperTrainer, TrainingConfig
from src.data import load_manifest

# Load config
config = TrainingConfig.from_yaml("configs/default.yaml")

# Load data
train_samples = load_manifest("data/splits/train.json")
val_samples = load_manifest("data/splits/val.json")

# Train
trainer = WhisperTrainer(config)
model_path = trainer.train(train_samples, val_samples)
```

### Step 4: Evaluate

```python
from src.evaluation import run_comparison

# Compare your model against vanilla Whisper
report = run_comparison(
    audio_paths=[...],
    references=[...],
    models={
        "whisper-small": vanilla_transcribe_fn,
        "spanish-slang-whisper": finetuned_transcribe_fn,
    }
)
print(report)
```

## Generating Synthetic Training Data

If you don't have audio recordings, you can bootstrap with ElevenLabs TTS:

```python
from src.data import SyntheticDataGenerator

# Initialize with your ElevenLabs API key
generator = SyntheticDataGenerator(
    api_key="your-elevenlabs-api-key",
    output_dir="data/synthetic"
)

# Generate audio from text
texts = [
    "Hola, ¿qué onda?",
    "¡Está chido, güey!",
    "Che, ¿cómo andás?",
]

results = generator.synthesize_batch(texts, alternate_voices=True)
manifest_path = generator.create_manifest(results)
```

## API Reference

### `POST /transcribe`

Transcribe audio from base64 or URL.

```json
{
  "audio_base64": "...",
  "language": "es",
  "word_timestamps": true
}
```

### `POST /transcribe/upload`

Transcribe uploaded audio file.

```bash
curl -X POST http://localhost:8000/transcribe/upload \
  -F "file=@audio.wav" \
  -F "language=es" \
  -F "word_timestamps=true"
```

### `POST /evaluate`

Transcribe and compare against expected text.

```json
{
  "audio_base64": "...",
  "expected_text": "¿Qué onda, güey?",
  "language": "es"
}
```

### `GET /health`

Check API health and model status.

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **WER** | Word Error Rate - percentage of words incorrectly transcribed |
| **CER** | Character Error Rate - percentage of characters incorrect |
| **MER** | Match Error Rate |
| **WIL** | Word Information Lost |

## Configuration

Edit `configs/default.yaml` to customize:

```yaml
model:
  base_model: "openai/whisper-small"
  language: "es"

training:
  num_train_epochs: 3
  learning_rate: 1.0e-5
  per_device_train_batch_size: 8

slang_markers:
  lexical:
    - "tio"
    - "mola"
    - "guay"
    - "chido"
  regional:
    mexico:
      - "chido"
      - "padre"
      - "neta"
    argentina:
      - "che"
      - "boludo"
      - "pibe"
```

## Deployment

### Docker

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Hugging Face Spaces

Deploy as a Gradio app on Hugging Face Spaces for a public demo.

## Roadmap

- [x] Set up training pipeline
- [ ] Fine-tune Whisper-small on Spanish slang data
- [ ] Benchmark against vanilla Whisper
- [ ] Deploy to Hugging Face Spaces
- [ ] Add streaming transcription support
- [ ] Integrate with ElevenLabs for full speech-to-speech loop

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- [OpenAI Whisper](https://github.com/openai/whisper) for the base model
- [ElevenLabs](https://elevenlabs.io) for TTS API
- [faster-whisper](https://github.com/guillaumekln/faster-whisper) for optimized inference
